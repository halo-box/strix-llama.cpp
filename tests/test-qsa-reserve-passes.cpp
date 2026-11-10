// The QSA reserve graphs of llama_context::sched_reserve (llama_qsa_reserve_passes, PLAN-v3 §7.1) cover every graph shape a
// runtime ubatch can build. Here: the one-stream dense-shortcut graph of a split multi-sequence context. Pass B reserves the
// shortcut at n_kv_b over n_seqs streams, so each stream holds n_tok_b / n_seqs rows and the route of that graph is decided
// at that per-stream n_q (prefer_dense in qwen4exp.cpp; on HIP a non-f16 pair is dense FA at n_q <= 127 and the AV rows +
// SPARSE_ATTN route at n_q >= 128). A slot's own prompt ubatch at n_kv <= n_kv_b is a one-stream graph of up to
// min(ub, n_kv_b) rows, which only a one-stream pass at n_kv_b builds: -np 8 at ub 512 reserved 8 x 64 rows (dense) while
// a 512-token one-sequence ubatch ran the AV route. Pure function of the context shape: no model, no device.
#include "../src/llama-context.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); n_fail++; } } while (0)

static std::string names(const std::vector<llama_qsa_reserve_pass_desc> & p) {
    std::string s;
    for (const auto & d : p) {
        s += std::string(s.empty() ? "" : " ") + d.name;
    }
    return s;
}

// the qwen4exp shape: indexer_top_k 2048, ratio 4 -> width 2051, shortcut n_kv_b 2048
static llama_qsa_reserve_shape shape(uint32_t n_tokens, uint32_t n_seqs, bool unified, uint32_t n_kv_full) {
    llama_qsa_reserve_shape s = {};
    s.n_tokens       = n_tokens;
    s.n_seqs         = n_seqs;
    s.n_outputs_pp   = n_tokens;
    s.kv_unified     = unified;
    s.mtp            = false;
    s.width          = 2048 + 4 - 1;
    s.n_kv_full      = n_kv_full;
    s.noncompact_cap = std::min<uint32_t>(n_tokens, 2 * 512);
    return s;
}

int main() {
    for (uint32_t n_tokens : { 256u, 512u, 2048u, 4096u, 8192u }) {
        for (uint32_t n_seqs : { 1u, 2u, 4u, 8u }) {
            for (bool unified : { false, true }) {
                for (uint32_t n_kv_full : { 2048u, 4096u, 32768u }) {
                    const llama_qsa_reserve_shape s = shape(n_tokens, n_seqs, unified, n_kv_full);
                    const auto p = llama_qsa_reserve_passes(s);
                    const uint32_t n_kv_b = s.width / 256 * 256;
                    const bool shortcut = n_kv_b < n_kv_full;
                    const bool split_multi = !unified && n_seqs > 1;
                    printf("ub %u np %u %s n_kv %u: %s\n", n_tokens, n_seqs, unified ? "unified" : "split", n_kv_full, names(p).c_str());

                    // the one-stream graph of the shortcut: one sequence, n_stream 1, at n_kv_b, with the largest row count a
                    // one-sequence ubatch can have there
                    const uint32_t n_tok_1 = std::min(n_tokens, n_kv_b);
                    bool one_stream_b = false;
                    for (const auto & d : p) {
                        // the streams of the pass's graph: one in a unified cache; else the view's (every stream: n_seqs)
                        const uint32_t streams = unified ? 1 : (d.n_kv != 0 && d.n_stream != 0 ? d.n_stream : n_seqs);
                        const bool one_stream = streams == 1;
                        const uint32_t view_kv = d.n_kv == 0 ? n_kv_full : d.n_kv;
                        if (one_stream && view_kv == n_kv_b && d.n_tokens >= n_tok_1 && d.qsa == 0) {
                            one_stream_b = true;
                        }
                        CHECK(d.n_tokens >= 1 && d.n_tokens <= n_tokens, "%s: n_tokens %u", d.name, d.n_tokens);
                        CHECK(d.n_outputs <= d.n_tokens || strcmp(d.name, "noncompact") == 0, "%s: n_outputs %u > n_tokens %u", d.name, d.n_outputs, d.n_tokens);
                    }
                    if (shortcut) {
                        CHECK(one_stream_b, "ub %u np %u %s n_kv %u: no reserve graph builds the one-stream shortcut at n_kv %u with %u rows",
                              n_tokens, n_seqs, unified ? "unified" : "split", n_kv_full, n_kv_b, n_tok_1);
                    } else {
                        for (const auto & d : p) {
                            CHECK(d.n_kv == 0 || d.n_kv == n_kv_full, "%s: a reduced view at n_kv %u without a shortcut", d.name, d.n_kv);
                        }
                    }
                    // the passes before this change keep their shapes (the refactor of sched_reserve into this function)
                    bool has_b = false, has_1s = false;
                    for (const auto & d : p) {
                        if (strcmp(d.name, "passB") == 0) {
                            has_b = true;
                            CHECK(d.n_kv == n_kv_b && d.n_stream == 0 && d.n_seqs == n_seqs &&
                                  d.n_tokens == std::min(n_tokens, n_kv_b * (unified ? 1 : n_seqs)), "passB shape");
                        }
                        if (strncmp(d.name, "1s-", 3) == 0) {
                            has_1s = true;
                            CHECK(d.n_stream == 1 && d.n_seqs == 1, "%s: not one-stream", d.name);
                        }
                    }
                    CHECK(has_b == shortcut, "passB present %d, shortcut %d", (int) has_b, (int) shortcut);
                    CHECK(has_1s == split_multi, "1s passes present %d, split multi-sequence %d", (int) has_1s, (int) split_multi);
                }
            }
        }
    }
    printf("test-qsa-reserve-passes: %s\n", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
