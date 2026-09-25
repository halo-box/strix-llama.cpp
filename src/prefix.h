#pragma once

#include "llama-batch.h"
#include "llama-kv-cells.h"
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
    // a 2-D (image) cell past its linear position breaks the scalar test; only blocks with a shared cell can split
    std::unordered_map<llama_pos, std::vector<llama_kv_cells::seq_set_t>> shared;
    for (uint32_t j=0;j<count;++j) {
        if (cells.is_empty(j) || (cells.seq_get_all(j) & rows).none()) { continue; }
        if (u.is_pos_2d() && cells.ext_get(j).is_2d_gt(cells.pos_get(j),cells.pos_get(j))) { return false; }
        if (cells.seq_get_all(j).count()>1) { shared[cells.pos_get(j)/ratio]; }
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
