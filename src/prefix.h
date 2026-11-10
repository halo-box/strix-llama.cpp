#pragma once

#include "llama-batch.h"
#include "llama-kv-cells.h"
#include "qsa-prefix-state.h"
#include <algorithm>
#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <vector>

// PLAN-v3 §4.2 step 3: the cells of one stream belong to at most one sequence. The non-compact scan ranks the cells in the
// mask's (pos, y, x) order only then (llama_memory_hybrid_idx::set_input_qsa_scan), and the compact predicate accepts 2-D
// rows only then: both call this one function, so the two cannot diverge (test-qsa-visibility asserts it per ubatch)
static bool qsa_stream_one_seq(const llama_kv_cells & cells) {
    int n_seq_present = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (cells.seq_pos_min(s) >= 0 && ++n_seq_present > 1) {
            return false;
        }
    }
    return true;
}

// the scan's ranking decision: positions repeat inside a group (an M-RoPE image) and the stream holds one sequence
static bool qsa_scan_ranks(const llama_kv_cells & cells, bool dup, bool pos_2d) {
    return dup && pos_2d && qsa_stream_one_seq(cells);
}

// why the compact predicate rejected a ubatch (0 = compact); the graph-build check and the re-split log the name
enum qsa_compact_reason {
    QSA_COMPACT_OK          = 0,
    QSA_NC_INPUT            = 1, // the ubatch lacks positions or sequence ids, or a stream split does not divide it
    QSA_NC_POS_RANGE        = 2, // a position outside [0, 2^24): the producer's f32 cell ids are exact only below it
    QSA_NC_DUP_KEY          = 3, // two cells of one sequence with the same (pos, y, x) key: the mask order is not total
    QSA_NC_2D_MULTI_SEQ     = 4, // a sequence of the stream holds 2-D cells while another sequence has cells in the stream
    QSA_NC_SHARED_BLOCK     = 5, // a position block holds one sequence under two sequence sets (seq_cp inside the block)
};

static const char * qsa_compact_reason_name(int r) {
    switch (r) {
        case QSA_COMPACT_OK:      return "compact";
        case QSA_NC_INPUT:        return "ubatch-input";
        case QSA_NC_POS_RANGE:    return "position>=2^24";
        case QSA_NC_DUP_KEY:      return "duplicate-(pos,y,x)-key";
        case QSA_NC_2D_MULTI_SEQ: return "2-D-cells-with-another-sequence-in-the-stream";
        case QSA_NC_SHARED_BLOCK: return "block-shared-by-two-sequence-sets";
    }
    return "unknown";
}

// the mask's key of a cell: (pos, y, x) under M-RoPE (llama-kv-cache.cpp set_input_kq_mask_impl, is_2d_gt), pos alone otherwise
static bool qsa_key_2d(const llama_kv_cells & cells, uint32_t j) {
    return cells.ext_get(j).is_2d_gt(cells.pos_get(j), cells.pos_get(j));
}

// duplicate (pos, y, x) keys among the cells of seq: the positions come sorted from seq_pos_get, so only the cells of a
// position that occurs more than once are compared on (y, x). A sequence with unique positions (every text sequence,
// seq_cp copies included) is answered from the cells' duplicate count without a walk: the predicate runs on every graph
// build and can_reuse of a multi-sequence context, and the walk visited every cell of every sequence of the ubatch there.
// walked (test hook, test-qsa-dup-key): the set entries visited
static bool qsa_seq_has_dup_key(const llama_kv_cells & cells, llama_seq_id seq, bool pos_2d, size_t * walked = nullptr) {
    if (cells.seq_pos_n_dup(seq) == 0) {
        return false;   // no position repeats: no key can
    }
    if (!pos_2d) {
        return true;    // the key is the position alone
    }
    const auto & sp = cells.seq_pos_get(seq);
    std::vector<std::pair<llama_pos, llama_pos>> yx;
    for (auto it = sp.begin(); it != sp.end();) {
        auto end = it;
        size_t n = 0;
        while (end != sp.end() && end->first == it->first) { ++end; ++n; }
        if (walked) { *walked += n; }
        if (n > 1) {
            yx.clear();
            for (auto c = it; c != end; ++c) {
                const auto & e = cells.ext_get(c->second);
                yx.emplace_back(e.y, e.x);
            }
            std::sort(yx.begin(), yx.end());
            if (std::adjacent_find(yx.begin(), yx.end()) != yx.end()) { return true; }
        }
        it = end;
    }
    return false;
}

// compact (maskless) visibility of the tokens [i0, i0 + n) of a ubatch: one stream, its cells, its own sequence ids
// (PLAN-v3 §4.2, D7). Each query reads the block row of its first seq id, as the attention mask does.
//  - embedding (image) ubatches are accepted like token ubatches (step 1);
//  - only the sequence ids of this stream's tokens count (step 2);
//  - a sequence holding 2-D cells is compact iff the stream holds cells of that one sequence (step 3): the mask makes cell j
//    visible to query q iff key(j) <=lex key(q) on (pos, y, x), so the visible set is a prefix of the rank order the scan
//    builds (qsa_scan_ranks), whatever the physical cell order, and the limits are rank-based;
//  - degenerate input (a position >= 2^24, a duplicate key) is rejected (step 4): the non-compact route reads the mask;
//  - text ubatches in a multi-sequence unified stream stay compact when no sequence of the ubatch holds 2-D cells (step 5),
//    unless a block holds one sequence under two sequence sets (seq_cp inside the block): it has no complete group for it,
//    so the maskless path would drop its cells.
static int qsa_compact_visibility_stream(const llama_kv_cells & cells, uint32_t count, uint32_t ratio, const llama_ubatch & u,
        uint32_t i0, uint32_t n) {
    if (ratio==0 || count>cells.size() || !u.pos || !u.n_tokens || !u.n_pos || !u.seq_id || !u.n_seq_id || n==0 ||
            i0+n>u.n_tokens) {
        return QSA_NC_INPUT;
    }
    const bool pos_2d = u.is_pos_2d();
    llama_kv_cells::seq_set_t rows;
    bool has_2d = false;
    for (uint32_t i=i0;i<i0+n;++i) {
        if (u.n_seq_id[i]<1 || !u.seq_id[i] || u.seq_id[i][0]<0 || u.seq_id[i][0]>=LLAMA_MAX_SEQ) { return QSA_NC_INPUT; }
        if (u.pos[i]<0 || u.pos[i]>=16777216) { return QSA_NC_POS_RANGE; }
        // text under M-RoPE: the three rope axes carry the position, the fourth (unused by the visibility) is 0
        for (uint32_t axis=1;axis<u.n_pos;++axis) {
            const llama_pos p = u.pos[i+axis*u.n_tokens];
            if (p!=u.pos[i] && !(axis==3 && p==0)) { has_2d = true; }
        }
        rows.set(u.seq_id[i][0]);
    }
    // only blocks with a shared cell can split
    std::unordered_map<llama_pos, std::vector<llama_kv_cells::seq_set_t>> shared;
    for (uint32_t j=0;j<count;++j) {
        if (cells.is_empty(j)) { continue; }
        if ((cells.seq_get_all(j) & rows).none()) { continue; }
        if (cells.pos_get(j)>=16777216) { return QSA_NC_POS_RANGE; }
        if (pos_2d && !has_2d && qsa_key_2d(cells, j)) { has_2d = true; }
        if (cells.seq_get_all(j).count()>1) { shared[cells.pos_get(j)/ratio]; }
    }
    for (llama_seq_id s=0;s<LLAMA_MAX_SEQ;++s) {
        if (rows.test(s) && qsa_seq_has_dup_key(cells, s, pos_2d)) { return QSA_NC_DUP_KEY; }
    }
    if (has_2d && !qsa_stream_one_seq(cells)) { return QSA_NC_2D_MULTI_SEQ; }
    for (uint32_t j=0;!shared.empty() && j<count;++j) {
        if (cells.is_empty(j) || (cells.seq_get_all(j) & rows).none()) { continue; }
        const auto it = shared.find(cells.pos_get(j)/ratio);
        if (it == shared.end()) { continue; }
        const auto & set = cells.seq_get_all(j);
        for (const auto & other : it->second) {
            if (other != set && (other & set & rows).any()) { return QSA_NC_SHARED_BLOCK; }
        }
        if (std::find(it->second.begin(), it->second.end(), set) == it->second.end()) { it->second.push_back(set); }
    }
    return QSA_COMPACT_OK;
}

// the whole ubatch on one stream (the single-stream callers and the unit tests)
static bool qsa_scalar_visibility_cells(const llama_kv_cells & cells, uint32_t count, uint32_t ratio, const llama_ubatch & u) {
    return qsa_compact_visibility_stream(cells, count, ratio, u, 0, u.n_tokens) == QSA_COMPACT_OK;
}

static bool qsa_single_sequence_prefix(const llama_kv_cells & cells, uint32_t count, llama_seq_id seq) {
    if (count>cells.size()) { return false; }
    std::vector<llama_pos> positions;
    positions.reserve(count);
    for (uint32_t i=0;i<count;++i) {
        if (cells.is_empty(i)) { continue; }
        if (cells.seq_get_all(i).count()!=1 || !cells.seq_has(i,seq) || cells.pos_get(i)<0) { return false; }
        positions.push_back(cells.pos_get(i));
    }
    std::sort(positions.begin(),positions.end());
    return std::adjacent_find(positions.begin(),positions.end())==positions.end();
}

// rebuild the QSA prefix of seq from the cells (out: sized to the cache): its cells in the mask's order (position, then y,
// then x), keyed as the cache stores them; apply checks the order (no two cells with one key) and fills every tracked field
// a unified cache also holds other sequences (other slots of a server): skip their cells, seq does not attend to them;
// a cell shared with another sequence is refused
static bool qsa_rebuild_prefix(const llama_kv_cells & raw, llama_seq_id seq, qsa_prefix_state & out) {
    out.reset();
    const llama_pos pmax = raw.seq_pos_max(seq);
    if (pmax < 0) { out.sequence = seq; return true; }
    // counting sort by position (linear: a switch between sequences rebuilds), then (y, x) among an image's cells
    std::vector<uint32_t> first(size_t(pmax) + 2, 0);
    for (uint32_t cell=raw.used_min(); cell<raw.used_max_p1(); ++cell) {
        if (raw.is_empty(cell) || !raw.seq_has(cell, seq)) { continue; }
        const llama_pos pos = raw.pos_get(cell);
        if (pos < 0 || pos > pmax || raw.seq_get_all(cell).count() != 1) { return false; }
        ++first[pos + 1];
    }
    for (size_t i = 1; i < first.size(); ++i) { first[i] += first[i-1]; }
    const size_t n = first.back();
    std::vector<uint32_t> slots(n); std::vector<int32_t> p(n), y(n), x(n);
    for (uint32_t cell=raw.used_min(); cell<raw.used_max_p1(); ++cell) {
        if (raw.is_empty(cell) || !raw.seq_has(cell, seq)) { continue; }
        const llama_pos pos = raw.pos_get(cell);
        const auto & ext = raw.ext_get(cell);
        const uint32_t k = first[pos]++;
        slots[k] = cell; p[k] = pos; y[k] = ext.y; x[k] = ext.x;
    }
    for (size_t i = 0; i < n;) {
        size_t j = i + 1;
        while (j < n && p[j] == p[i]) { ++j; }
        if (j - i > 1) {
            std::vector<std::tuple<int32_t, int32_t, uint32_t>> group;
            for (size_t k = i; k < j; ++k) { group.emplace_back(y[k], x[k], slots[k]); }
            std::sort(group.begin(), group.end());
            for (size_t k = i; k < j; ++k) { std::tie(y[k], x[k], slots[k]) = group[k - i]; }
        }
        i = j;
    }
    if (n > 0 && !out.apply(seq, 0, slots, p, y, x)) { return false; }
    out.sequence = seq;
    return true;
}

static std::vector<int64_t> qsa_prefix_limits(const llama_pos * pos, int64_t tokens, int64_t strip,
        int64_t ratio, int64_t blocks, int64_t budget) {
    if (!pos || tokens<=0 || strip<=0 || ratio<=0 || budget<=0 || blocks<budget) { return {}; }
    std::vector<int64_t> limits;
    for (int64_t first=0;first<tokens;first+=strip) {
        int64_t maximum=-1;
        for (int64_t i=first;i<std::min(tokens,first+strip);++i) {
            if (pos[i]<0 || pos[i]>=16777216) { return {}; }
            maximum=std::max(maximum,int64_t(pos[i]));
        }
        limits.push_back(std::min(blocks,std::max(budget,(maximum+1)/ratio)));
    }
    return limits;
}
