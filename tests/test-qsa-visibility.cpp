// PLAN-v3 §4.2 proof obligation (W9): the compact visibility predicate and its limits, on the production memory.
//
// Each layout is built through the real init_batch / apply path of a qwen4exp hybrid-idx memory (tests/qsa-mem-fixture.h).
// On every checked ubatch, right after its apply (the graph-time state):
//  - the predicate's verdict is compared with the expected one (positives, and the negative cases of P2-1);
//  - when it accepts, the scan fills the compact limits, block cells and tails (set_input_qsa_blocks, the graph's inputs),
//    and for every row the cells they make visible ({blocks with start < tail start} x 4 cells, plus the tail cells) must
//    equal the cells the KQ mask keeps (llama_kv_cache::set_input_kq_mask, the attention's own rule), with no duplicate;
//  - scan/predicate agreement: on a stream whose rows' sequences hold 2-D cells, the scan ranked iff the predicate accepted
//    (both call qsa_stream_one_seq), unless the rejection is a degenerate key.
// The layouts are synthetic reconstructions of the recorded shapes named in PLAN-v3 §4.2 (C4 p1/p2, B65t of C19, request C
// of L64, the R1-RES split-stream image ubatch) plus the synthetic ones (several images, back to back across a ubatch
// boundary, image at position 0, text after images, depth > 32k, holes after seq_rm) and the negative cases.
#include "qsa-mem-fixture.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <cinttypes>
#include <cstring>
#include <string>

static int g_fail = 0;
static int64_t g_rows = 0, g_ubatches = 0, g_agree = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail++; } } while (0)

enum expect_t { EXPECT_COMPACT, EXPECT_NONCOMPACT, EXPECT_ANY };

// does a stream's ubatch rows [i0, i0+n) carry a 2-D row, or do the cells of their sequences hold a 2-D cell
static bool stream_has_2d(const llama_kv_cells & cells, const llama_ubatch & u, uint32_t i0, uint32_t n) {
    llama_kv_cells::seq_set_t rows;
    for (uint32_t i = i0; i < i0 + n; ++i) {
        rows.set(u.seq_id[i][0]);
        if (u.pos[i + u.n_tokens] != u.pos[i] || u.pos[i + 2*u.n_tokens] != u.pos[i]) { return true; }
    }
    for (uint32_t j = 0; j < cells.size(); ++j) {
        if (!cells.is_empty(j) && (cells.seq_get_all(j) & rows).any() && cells.ext_get(j).is_2d_gt(cells.pos_get(j), cells.pos_get(j))) {
            return true;
        }
    }
    return false;
}

static void check_ubatch(const char * name, const llama_memory_hybrid_idx * mem, const llama_memory_hybrid_idx_context * mctx,
        const llama_ubatch & u, expect_t expect) {
    const uint32_t r = 4;
    const int64_t n_kv = mctx->get_idx()->get_n_kv();
    if (n_kv <= 2048 + r - 1) {
        return;   // the dense shortcut: no selection, nothing to check
    }
    g_ubatches++;
    const int64_t ns = mctx->get_n_stream();
    const int64_t n_tps = u.n_tokens / ns;
    const int64_t n_blocks = ((int64_t) mctx->qsa_n_kv_window() + r - 1) / r;
    const int64_t n_seq = mem->get_n_seq_max();
    const int reason = mctx->qsa_compact_reason(u, r);
    const bool compact = reason == QSA_COMPACT_OK;
    CHECK(expect == EXPECT_ANY || compact == (expect == EXPECT_COMPACT), "%s: ubatch of %u tokens at pos %d (%s): predicate %s (%s), expected %s",
            name, u.n_tokens, u.pos[0], u.token ? "text" : "embd", compact ? "compact" : "non-compact", qsa_compact_reason_name(reason),
            expect == EXPECT_COMPACT ? "compact" : "non-compact");
    // the graph decides with this same function on the indexer cells (qsa_scalar_visibility)
    CHECK(mctx->qsa_scalar_visibility(u, r) == compact, "%s: qsa_scalar_visibility disagrees with the predicate", name);

    ggml_init_params ip = { 8*ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor * cell_blk  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_kv, ns);
    ggml_tensor * blk_cells = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, r*n_blocks, ns);
    ggml_tensor * blk_pos   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4*n_blocks*ns);
    ggml_tensor * bias = compact ? (ns == 1 ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_blocks*n_seq + 2*n_tps)
                                            : ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_blocks + n_tps, ns))
                                 : ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_blocks, n_tps, ns);
    ggml_tensor * tails = compact ? ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I32, r - 1, n_tps, ns) : nullptr;
    ggml_tensor * mask  = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, n_kv, n_tps, 1, ns);
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
    GGML_ASSERT(buf);

    if (compact) {
        mctx->set_input_qsa_blocks(cell_blk, blk_cells, blk_pos, bias, tails, &u, r);
    } else {
        mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, &u, r, true, true);
    }
    mctx->get_attn()->set_input_kq_mask(mask, &u, true);

    // agreement: per stream, a 2-D layout is ranked iff it is compact (degenerate keys aside)
    const auto & ranked = mem->qsa_last_scan_ranked();
    CHECK((int64_t) ranked.size() == ns, "%s: the scan recorded %zu streams, the ubatch has %" PRId64, name, ranked.size(), ns);
    for (int64_t s = 0; s < ns && (int64_t) ranked.size() == ns; ++s) {
        const auto & cells = mem->get_mem_idx()->get_cells(u.seq_id[s*n_tps][0]);
        const int rs = qsa_compact_visibility_stream(cells, (uint32_t) n_kv, r, u, (uint32_t) (s*n_tps), (uint32_t) n_tps);
        if (stream_has_2d(cells, u, (uint32_t) (s*n_tps), (uint32_t) n_tps) && (rs == QSA_COMPACT_OK || rs == QSA_NC_2D_MULTI_SEQ)) {
            g_agree++;
            CHECK((ranked[s] != 0) == (rs == QSA_COMPACT_OK), "%s: stream %" PRId64 ": scan ranked=%d, predicate %s", name, s, ranked[s],
                    qsa_compact_reason_name(rs));
        }
    }
    if (!compact) {
        return;
    }

    const int32_t * lim = (const int32_t *) bias->data;
    const int32_t * bc  = (const int32_t *) blk_cells->data;
    const int32_t * tl  = (const int32_t *) tails->data;
    const float   * mk  = (const float *) mask->data;
    std::vector<int32_t> got, want;
    for (int64_t s = 0; s < ns; ++s) {
        for (int64_t ii = 0; ii < n_tps; ++ii) {
            got.clear(); want.clear();
            int32_t tail_start;
            const int32_t * starts;
            if (ns == 1) {
                const int64_t base = n_blocks*n_seq;
                tail_start = lim[base + ii];
                const int32_t row = lim[base + n_tps + ii];
                CHECK(row == u.seq_id[ii][0], "%s: row %" PRId64 " reads the start row of seq %d", name, ii, row);
                starts = lim + (int64_t) row*n_blocks;
            } else {
                starts = lim + s*(n_blocks + n_tps);
                tail_start = starts[n_blocks + ii];
            }
            for (int64_t b = 0; b < n_blocks; ++b) {
                if (starts[b] < tail_start) {
                    for (uint32_t k = 0; k < r; ++k) { got.push_back(bc[s*r*n_blocks + b*r + k]); }
                }
            }
            for (uint32_t k = 0; k < r - 1; ++k) {
                const int32_t c = tl[(s*n_tps + ii)*(r - 1) + k];
                if (c >= 0) { got.push_back(c); }
            }
            for (int64_t j = 0; j < n_kv; ++j) {
                if (mk[(s*n_tps + ii)*n_kv + j] == 0.0f) { want.push_back((int32_t) j); }
            }
            std::sort(got.begin(), got.end());
            const bool dup = std::adjacent_find(got.begin(), got.end()) != got.end();
            CHECK(!dup, "%s: row %" PRId64 " lists a cell twice", name, s*n_tps + ii);
            got.erase(std::unique(got.begin(), got.end()), got.end());
            if (got != want) {
                CHECK(false, "%s: stream %" PRId64 " row %" PRId64 " (pos %d): compact limits give %zu cells, the mask %zu", name, s, ii,
                        u.pos[s*n_tps + ii], got.size(), want.size());
                return;
            }
            g_rows++;
        }
    }
}

// feed tokens, check every ubatch with the expectation
static void feed_check(const char * name, qsa_fixture & f, const std::vector<qsa_fixture::tok> & toks, uint32_t n_ubatch, expect_t expect) {
    const bool ok = f.feed(toks, n_ubatch, [&](const llama_memory_hybrid_idx_context * mctx, const llama_ubatch & u, uint32_t) {
        check_ubatch(name, f.mem.get(), mctx, u, expect);
    });
    CHECK(ok, "%s: the batch of %zu tokens was refused", name, toks.size());
}

static void feed_plain(const char * name, qsa_fixture & f, const std::vector<qsa_fixture::tok> & toks, uint32_t n_ubatch) {
    CHECK(f.feed(toks, n_ubatch), "%s: the batch of %zu tokens was refused", name, toks.size());
}

int main() {
    // C4 p1 / p2 shape (one sequence, one 32 x 32 image after a text prompt, text after it), np1
    {
        qsa_fixture f(16384, 1, true);
        feed_plain("c4", f, f.text(0, 2600), 512);
        feed_check("c4-image", f, f.image(0, 32, 32), 512, EXPECT_COMPACT);
        feed_check("c4-text-after", f, f.text(0, 1500), 512, EXPECT_COMPACT);
        feed_check("c4-decode", f, f.text(0, 1), 512, EXPECT_COMPACT);
    }
    // B65t shape (C19): a deep image past 32k (depth > 32k), image ubatches of 1024 at ub1024, then text
    {
        qsa_fixture f(40960, 1, true);
        feed_plain("b65t", f, f.text(0, 33000), 4096);
        feed_check("b65t-image", f, f.image(0, 24, 40), 1024, EXPECT_COMPACT);
        feed_check("b65t-text", f, f.text(0, 700), 1024, EXPECT_COMPACT);
    }
    // request C shape (L64): five images with text between them, one sequence; two images back to back across a 512 boundary
    {
        qsa_fixture f(32768, 1, true);
        feed_plain("reqc", f, f.text(0, 2100), 512);
        for (int k = 0; k < 5; ++k) {
            feed_check("reqc-image", f, f.image(0, 16 + 2*k, 20), 512, EXPECT_COMPACT);
            feed_check("reqc-text", f, f.text(0, 300 + 77*k), 512, EXPECT_COMPACT);
        }
        // one embedding batch of two images (256 + 320 tokens): the 512-token ubatch boundary falls inside the second
        auto two = f.image(0, 16, 16);
        const auto second = f.image(0, 16, 20);
        two.insert(two.end(), second.begin(), second.end());
        feed_check("reqc-back-to-back", f, two, 512, EXPECT_COMPACT);
        feed_check("reqc-tail", f, f.text(0, 513), 512, EXPECT_COMPACT);
    }
    // image at position 0, text after it
    {
        qsa_fixture f(8192, 1, true);
        feed_check("pos0-image", f, f.image(0, 48, 48, 0), 512, EXPECT_COMPACT);
        feed_check("pos0-text", f, f.text(0, 1200), 512, EXPECT_COMPACT);
    }
    // holes after seq_rm: a rollback in the middle of the text and of an image's extent, then more text and an image
    {
        qsa_fixture f(16384, 1, true);
        feed_plain("holes", f, f.text(0, 3000), 1024);
        feed_plain("holes-image", f, f.image(0, 20, 20), 1024);
        feed_plain("holes-text", f, f.text(0, 800), 1024);
        f.mem->seq_rm(0, 3500, -1);   // the text after the image goes, the image stays
        f.next_pos[0] = 3500;
        feed_check("holes-after-rm", f, f.text(0, 700), 512, EXPECT_COMPACT);
        feed_check("holes-image2", f, f.image(0, 10, 30), 512, EXPECT_COMPACT);
        feed_check("holes-text2", f, f.text(0, 100), 512, EXPECT_COMPACT);
    }
    // R1-RES split-stream image ubatch: -np 2 split, slot 0 image + text, slot 1 text, a ubatch with both streams
    {
        qsa_fixture f(16384, 2, false);
        feed_plain("split", f, f.text(0, 2500), 1024);
        feed_plain("split", f, f.text(1, 3000), 1024);
        feed_plain("split-image", f, f.image(0, 32, 32), 1024);
        auto both = f.text(0, 600);
        auto b1 = f.text(1, 600);
        both.insert(both.end(), b1.begin(), b1.end());
        feed_check("split-both-streams", f, both, 1024, EXPECT_COMPACT);
        auto img = f.image(0, 16, 16);
        feed_check("split-image-stream", f, img, 1024, EXPECT_COMPACT);
    }
    // negative (P2-1): unified stream, seq A holds an image, seq B holds text cells: A rejected (text and image), B accepted;
    // then B's cells removed by seq_rm: A accepted again
    {
        qsa_fixture f(32768, 2, true);
        feed_plain("neg", f, f.text(0, 2200), 1024);
        feed_plain("neg", f, f.image(0, 32, 32), 1024);
        feed_plain("neg", f, f.text(1, 2500), 1024);
        feed_check("neg-A-text", f, f.text(0, 600), 512, EXPECT_NONCOMPACT);
        feed_check("neg-A-image", f, f.image(0, 16, 16), 512, EXPECT_NONCOMPACT);
        feed_check("neg-B-text", f, f.text(1, 700), 512, EXPECT_COMPACT);
        f.mem->seq_rm(1, -1, -1);
        feed_check("neg-A-text-after-rm", f, f.text(0, 600), 512, EXPECT_COMPACT);
        feed_check("neg-A-image-after-rm", f, f.image(0, 12, 12), 512, EXPECT_COMPACT);
    }
    // the same two sequences in split streams: each stream holds one sequence, both accepted
    {
        qsa_fixture f(16384, 2, false);
        feed_plain("neg-split", f, f.text(0, 2200), 1024);
        feed_plain("neg-split", f, f.image(0, 32, 32), 1024);
        feed_plain("neg-split", f, f.text(1, 2500), 1024);
        feed_check("neg-split-A-text", f, f.text(0, 600), 512, EXPECT_COMPACT);
        feed_check("neg-split-A-image", f, f.image(0, 16, 16), 512, EXPECT_COMPACT);
        feed_check("neg-split-B-text", f, f.text(1, 700), 512, EXPECT_COMPACT);
    }
    // text only, two sequences in one unified stream (no 2-D cells): compact as today (step 5); a seq_cp prefix that ends
    // inside a block (R1-CB, L = 8193) splits that block over two sequence sets: non-compact for both
    {
        qsa_fixture f(32768, 3, true);
        feed_plain("text-ms", f, f.text(0, 3000), 1024);
        feed_plain("text-ms", f, f.text(1, 2600), 1024);
        feed_check("text-ms-0", f, f.text(0, 500), 512, EXPECT_COMPACT);
        feed_check("text-ms-1", f, f.text(1, 500), 512, EXPECT_COMPACT);
        f.mem->seq_rm(0, -1, -1); f.mem->seq_rm(1, -1, -1); f.next_pos[0] = f.next_pos[1] = 0;
        feed_plain("seqcp", f, f.text(0, 8193), 4096);
        f.mem->seq_cp(0, 2, -1, -1);
        f.next_pos[2] = f.next_pos[0];
        feed_check("seqcp-src", f, f.text(0, 40), 512, EXPECT_NONCOMPACT);
        feed_check("seqcp-dst", f, f.text(2, 40), 512, EXPECT_NONCOMPACT);
    }
    // degenerate (step 4): an embedding batch may start at the last stored position (llama-batch.cpp: X <= Y for embd), so an
    // image at t0 = X puts its first cell on the key (X, X, X) of the text cell at X: a duplicate key, non-compact
    {
        qsa_fixture f(8192, 1, true);
        feed_plain("dup", f, f.text(0, 3000), 1024);
        feed_check("dup-image", f, f.image(0, 8, 8, 2999), 512, EXPECT_NONCOMPACT);
        CHECK(f.mem->get_mem_idx() != nullptr, "dup: no indexer");
    }

    printf("test-qsa-visibility: %" PRId64 " ubatches checked, %" PRId64 " rows compared against the KQ mask, %" PRId64 " scan/predicate agreement checks, %d failures\n",
            g_ubatches, g_rows, g_agree, g_fail);
    printf("%s\n", g_fail == 0 ? "PASS" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}
