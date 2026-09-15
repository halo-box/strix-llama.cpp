#pragma once

#include "llama-kv-cells.h"
#include <algorithm>
#include <cstdint>
#include <vector>

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
