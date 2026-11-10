// The always-on QSA graph-build check (qwen4exp_qsa_noncompact_error, src/models/qwen4exp-qsa-check.h) fails only ubatches
// that are non-compact on the complete-block selection path. Where block selection is not eligible (a non-causal context:
// LLAMA_ATTENTION_TYPE_NON_CAUSAL or llama_set_causal_attn(ctx, false); a ratio-1 layer; top_k % ratio != 0; ...) every
// ubatch takes the TE rows, which the reserve's own pp / tg graphs build, so decode must go on: the check tested `scalar`
// alone (which also requires `blocks`) and failed every ubatch past the shortcut there, in a single-sequence context with
// the self-contradictory reason 'compact'. Pure function: no model.
#include "../src/models/qwen4exp-qsa-check.h"

#include <cstdio>
#include <initializer_list>

static int n_fail = 0;

#define CHECK(cond, what) do { if (!(cond)) { printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, what); n_fail++; } } while (0)

int main() {
    const int64_t width = 2048 + 4 - 1;   // indexer_top_k + ratio - 1
    const uint32_t U = 1024;
    for (uint32_t n_tokens : { 1u, 512u, 1024u, 4096u }) {
        for (int64_t n_kv : { (int64_t) 2048, (int64_t) 4096, (int64_t) 131072 }) {
            for (bool multi : { false, true }) {
                const bool past = n_kv > width;
                // block selection not eligible (non-causal, ratio 1, ...): scalar is false by construction, never an error
                CHECK(!qwen4exp_qsa_noncompact_error(false, false, false, false, n_kv, width, n_tokens, U, multi),
                      "blocks ineligible: a decode error");
                // SWA layer: the TE rows on every ubatch, never an error
                CHECK(!qwen4exp_qsa_noncompact_error(true, true, false, false, n_kv, width, n_tokens, U, multi),
                      "SWA layer: a decode error");
                // the reserve's non-compact variant, and compact ubatches: never an error
                CHECK(!qwen4exp_qsa_noncompact_error(true, false, true, false, n_kv, width, n_tokens, U, multi), "forced variant: error");
                CHECK(!qwen4exp_qsa_noncompact_error(true, false, false, true, n_kv, width, n_tokens, U, multi), "compact ubatch: error");
                // eligible and non-compact past the shortcut: an error in a single-sequence context, and past U tokens
                const bool want = past && (n_tokens > U || !multi);
                CHECK(qwen4exp_qsa_noncompact_error(true, false, false, false, n_kv, width, n_tokens, U, multi) == want,
                      "eligible non-compact ubatch: wrong decision");
            }
        }
    }
    // the per-context eligibility terms shared by qwen4exp_use_block_selection and the NC0 re-split
    CHECK(qwen4exp_block_selection_eligible(true, true, 4, 2048, 0.0f, false, 256, 256), "Qwen3.8 QSA layer: not eligible");
    CHECK(!qwen4exp_block_selection_eligible(true, false, 4, 2048, 0.0f, false, 256, 256), "non-causal: eligible");
    CHECK(!qwen4exp_block_selection_eligible(false, true, 4, 2048, 0.0f, false, 256, 256), "no blk_bias (alibi): eligible");
    CHECK(!qwen4exp_block_selection_eligible(true, true, 1, 2048, 0.0f, false, 256, 256), "ratio 1: eligible");
    CHECK(!qwen4exp_block_selection_eligible(true, true, 0, 2048, 0.0f, false, 256, 256), "ratio 0: eligible");
    CHECK(!qwen4exp_block_selection_eligible(true, true, 3, 2048, 0.0f, false, 256, 256), "top_k % ratio != 0: eligible");
    CHECK(!qwen4exp_block_selection_eligible(true, true, 4, 2048, 8.0f, false, 256, 256), "alibi bias: eligible");
    CHECK(!qwen4exp_block_selection_eligible(true, true, 4, 2048, 0.0f, true, 256, 256), "soft cap: eligible");
    CHECK(!qwen4exp_block_selection_eligible(true, true, 4, 2048, 0.0f, false, 128, 256), "head 128: eligible");
    printf("test-qsa-noncompact-check: %s\n", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
