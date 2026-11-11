#pragma once

#include <cstdint>

// The QSA graph-build check. A non-compact ubatch on the block-selection path takes the TE rows, which the reserve covers only for at most U tokens in a multi-sequence context: past that it is a decode error, returned before allocation.
// Layers where block selection is ineligible take the TE rows for every ubatch, the reserve included, so they never fail.
//
// The per-layer terms of qwen4exp_use_block_selection, shared with the NC0 re-split so it never cuts an ineligible ubatch.
static inline bool qwen4exp_block_selection_eligible(bool blk_bias, bool causal_attn, int64_t ratio, int64_t top_k,
                                                     float max_alibi_bias, bool attn_soft_cap, int64_t head_k, int64_t head_v) {
    return blk_bias && causal_attn && ratio > 1 && top_k % ratio == 0 && max_alibi_bias == 0.0f && !attn_soft_cap &&
        head_k == 256 && head_v == 256;
}

static inline bool qwen4exp_qsa_noncompact_error(bool blocks, bool swa, bool force_nc, bool scalar, int64_t n_kv, int64_t width,
                                                 uint32_t n_tokens, uint32_t U, bool multi_seq) {
    if (!blocks || swa || force_nc || scalar || n_kv <= width) {
        return false;
    }
    return n_tokens > U || !multi_seq;
}
