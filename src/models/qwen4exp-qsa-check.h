#pragma once

#include <cstdint>

// The always-on graph-build check of a QSA layer (PLAN-v3 §4.3, no env gate), as a pure function of the layer's decision
// (tests/test-qsa-noncompact-check.cpp). A ubatch on the complete-block selection path whose visibility is not scalar (a
// non-compact ubatch) takes the TE rows; the reserve covers that only for at most U tokens in a context serving several
// sequences (the non-compact variant, §7.1.4) and not at all in a single-sequence context (§7.1.5), so such a ubatch is
// a decode error, which process_ubatch returns before the graph is allocated.
//
// The check applies only where block selection is eligible (`blocks`: blk_bias, causal attention, ratio > 1,
// top_k % ratio == 0, head 256 ..., qwen4exp_use_block_selection) and the layer has no SWA: where it is not, the layer
// takes the TE rows for every ubatch, the reserve's own pp / tg graphs are built on that route (the decision does not
// depend on the ubatch), and the ubatch is not "non-compact" in the sense of the reserve. Testing only `scalar` there
// failed every ubatch past the shortcut (a non-causal context, a ratio-1 layer) with the reason 'compact'.
// The terms of the block-selection eligibility (qwen4exp_use_block_selection) that hold per layer and context, before
// n_kv: shared with the memory's NC0 re-split (llama_memory_hybrid_idx::qsa_nc0), which must not cut a ubatch the layer
// routes to the TE rows anyway (no reserve to protect there, test-qsa-resplit "ineligible").
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
