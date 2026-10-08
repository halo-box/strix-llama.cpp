#pragma once

#include "llama-batch.h"
#include "llama-kv-cells.h"
#include "qsa-prefix-state.h"
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

// compact (maskless) visibility: each query reads the block row of its first seq id, as the attention mask does.
// Blocks are keyed on (sequence set, position block): a block holding one sequence under two sets (seq_cp inside the
// block) has no complete group for it, so the maskless path would drop its cells; the masked path keeps them
static bool qsa_scalar_visibility_cells(const llama_kv_cells & cells, uint32_t count, uint32_t ratio, const llama_ubatch & u) {
    if (ratio==0 || count>cells.size() || !u.pos || !u.n_tokens || !u.n_pos || !u.seq_id || !u.n_seq_id) { return false; }
    llama_kv_cells::seq_set_t rows;
    for (uint32_t i=0;i<u.n_tokens;++i) {
        if (u.n_seq_id[i]<1 || !u.seq_id[i] || u.seq_id[i][0]<0 || u.seq_id[i][0]>=LLAMA_MAX_SEQ) { return false; }
        if (u.pos[i]<0 || u.pos[i]>=16777216) { return false; }
        for (uint32_t axis=1;axis<u.n_pos;++axis) {
            if (u.pos[i+axis*u.n_tokens]!=u.pos[i]) { return false; }
        }
        rows.set(u.seq_id[i][0]);
    }
    // a 2-D (image) cell past its linear position can break the scalar test: the mrope mask compares 2-D extents only
    // between a cell and a query at the same linear position. With one sequence in the cache the selection ranks the
    // cells in the mask's own order (position, then y, then x), so such a cell is harmless unless a query of this
    // ubatch sits at its position. Several sequences are not ranked: any 2-D cell keeps the masked path there.
    // only blocks with a shared cell can split
    std::unordered_map<llama_pos, std::vector<llama_kv_cells::seq_set_t>> shared;
    std::vector<llama_pos> pos_2d;
    bool one_seq = rows.count()==1;
    for (uint32_t j=0;j<count;++j) {
        if (cells.is_empty(j)) { continue; }
        one_seq = one_seq && cells.seq_get_all(j)==rows;
        if ((cells.seq_get_all(j) & rows).none()) { continue; }
        if (u.is_pos_2d() && cells.ext_get(j).is_2d_gt(cells.pos_get(j),cells.pos_get(j))) { pos_2d.push_back(cells.pos_get(j)); }
        if (cells.seq_get_all(j).count()>1) { shared[cells.pos_get(j)/ratio]; }
    }
    if (!pos_2d.empty()) {
        // the same test the selection uses to rank: no other sequence anywhere in the cache
        for (llama_seq_id s=0;one_seq && s<LLAMA_MAX_SEQ;++s) { one_seq = rows.test(s) || cells.seq_pos_min(s)<0; }
        if (!one_seq) { return false; }
        std::sort(pos_2d.begin(), pos_2d.end());
        for (uint32_t i=0;i<u.n_tokens;++i) {
            if (std::binary_search(pos_2d.begin(), pos_2d.end(), u.pos[i])) { return false; }
        }
    }
    for (uint32_t j=0;!shared.empty() && j<count;++j) {
        if (cells.is_empty(j) || (cells.seq_get_all(j) & rows).none()) { continue; }
        const auto it = shared.find(cells.pos_get(j)/ratio);
        if (it == shared.end()) { continue; }
        const auto & set = cells.seq_get_all(j);
        for (const auto & other : it->second) {
            if (other != set && (other & set & rows).any()) { return false; }
        }
        if (std::find(it->second.begin(), it->second.end(), set) == it->second.end()) { it->second.push_back(set); }
    }
    return true;
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

// rebuild the QSA prefix of seq from the cells (out: fresh, sized to the cache): each position in [0, n) must be in one cell that no other sequence shares
// a unified cache also holds other sequences (other slots of a server): skip their cells, seq does not attend to them
static bool qsa_rebuild_prefix(const llama_kv_cells & raw, llama_seq_id seq, qsa_prefix_state & out) {
    out.cells.assign(raw.seq_pos_max(seq) + 1, -1);
    for (uint32_t cell=raw.used_min(); cell<raw.used_max_p1(); ++cell) {
        if (raw.is_empty(cell) || !raw.seq_has(cell, seq)) { continue; }
        const int32_t pos = raw.pos_get(cell);
        const auto & ext = raw.ext_get(cell);
        if (pos < 0 || size_t(pos) >= out.cells.size() || out.cells[pos] >= 0 ||
            raw.seq_get_all(cell).count() != 1 ||
            !((ext.x == 0 && ext.y == 0) || (ext.x == pos && ext.y == pos))) { return false; }
        out.cells[pos] = cell; out.positions[cell] = pos;
    }
    if (std::find(out.cells.begin(), out.cells.end(), -1) != out.cells.end()) { return false; }
    out.sequence = seq;
    for (size_t b=0; b<out.cells.size()/4; ++b) { out.block_positions.push_back(b*4); }
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
