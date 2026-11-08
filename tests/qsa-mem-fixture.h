// A qwen4exp hybrid-idx memory without weights or devices, fed through its real init_batch / apply path (PLAN-v3 W9 tests):
// the attention cache, the indexer cache and their cells are the production objects; only the model has no layers
// (test-qsa-prefix.cpp test_scan_foreign_image builds the memory the same way).
#pragma once

#include "../src/llama-batch.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-memory-hybrid-idx.h"
#include "../src/llama-model.h"
#include "../src/prefix.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <vector>

struct qsa_fixture {
    std::unique_ptr<llama_model> model;
    std::unique_ptr<llama_memory_hybrid_idx> mem;
    uint32_t n_seq_max;
    bool unified;
    std::vector<llama_pos> next_pos;   // the next free M-RoPE position of every sequence (past an image's extent)

    static constexpr uint32_t N_EMBD = 8;

    qsa_fixture(uint32_t kv_size, uint32_t n_seq_max, bool unified, uint32_t n_rs_seq = 0, uint32_t strips = 1) :
            n_seq_max(n_seq_max), unified(unified), next_pos(LLAMA_MAX_SEQ, 0) {
        model.reset(llama_model_create(LLM_ARCH_QWEN4EXP, llama_model_default_params()));
        model->hparams = {};
        model->hparams.rope_type = LLAMA_ROPE_TYPE_IMROPE;   // n_pos_per_embd = 4, as every qwen4exp model
        mem.reset(new llama_memory_hybrid_idx(*model, GGML_TYPE_F16, GGML_TYPE_F16, false, kv_size, 1, 0, LLAMA_SWA_TYPE_NONE,
                GGML_TYPE_F32, GGML_TYPE_F32, std::max<uint32_t>(n_seq_max, 1), n_seq_max, n_rs_seq, false, unified,
                nullptr, nullptr, [](int32_t) { return false; }));
        // the model has no layer: the QSA layers' ratio and top-k of Qwen3.8 Flash Next (ratio 4, top-k 2048)
        mem->qsa_resplit_configure({ 4 }, 2048, strips);
    }

    struct tok { llama_seq_id seq; llama_pos p, y, x; bool embd; };

    // the token list of a text run and of an image (row-major h x w at t0: pos [t0, t0 + row, t0 + col, 0])
    std::vector<tok> text(llama_seq_id s, uint32_t n) {
        std::vector<tok> r;
        for (uint32_t i = 0; i < n; ++i) { const llama_pos p = next_pos[s]++; r.push_back({ s, p, p, p, false }); }
        return r;
    }
    std::vector<tok> image(llama_seq_id s, uint32_t h, uint32_t w, llama_pos t0 = -1) {
        if (t0 < 0) { t0 = next_pos[s]; }
        std::vector<tok> r;
        for (uint32_t a = 0; a < h; ++a) { for (uint32_t b = 0; b < w; ++b) { r.push_back({ s, t0, t0 + (llama_pos) a, t0 + (llama_pos) b, true }); } }
        next_pos[s] = std::max<llama_pos>(next_pos[s], t0 + (llama_pos) std::max(h, w));
        return r;
    }

    // one llama_batch_ext with these tokens; false when the allocator refuses it
    bool make_batch(const std::vector<tok> & toks, llama_batch_allocr & balloc, llama_batch_ext & batch) {
        static std::vector<float> zeros(N_EMBD, 0.0f);
        batch.clear();
        for (const auto & t : toks) {
            const int32_t idx = batch.add_token(t.seq);
            if (idx < 0) { return false; }
            if (t.embd) {
                GGML_ASSERT(batch.set_token_embd(idx, { zeros.data(), 1, N_EMBD }));
                const llama_pos pos[4] = { t.p, t.y, t.x, 0 };
                GGML_ASSERT(batch.set_token_pos(idx, pos));
            } else {
                GGML_ASSERT(batch.set_token_id(idx, 1 + (t.p % 97)));
                GGML_ASSERT(batch.set_token_pos(idx, &t.p));
            }
        }
        return balloc.init(batch, model->vocab, false);
    }

    std::unique_ptr<llama_batch_ext> new_batch(size_t n_max) {
        return std::make_unique<llama_batch_ext>(n_max, N_EMBD, N_EMBD, (llama_seq_id) n_seq_max, mem.get(), 1000, 4);
    }

    // init_batch + apply of every ubatch; cb(mctx, ubatch, k) right after ubatch k is applied (its graph-time state)
    template <typename F>
    bool feed(const std::vector<tok> & toks, uint32_t n_ubatch, F && cb) {
        auto batch = new_batch(toks.size());
        llama_batch_allocr balloc(4);
        if (!make_batch(toks, balloc, *batch)) { return false; }
        auto mctx_ptr = mem->init_batch(balloc, n_ubatch, false);
        if (!mctx_ptr || mctx_ptr->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) { return false; }
        auto * mctx = static_cast<llama_memory_hybrid_idx_context *>(mctx_ptr.get());
        uint32_t k = 0;
        do {
            const llama_ubatch & ub = mctx->get_ubatch();
            GGML_ASSERT(mctx->apply());
            cb(mctx, ub, k++);
        } while (mctx->next());
        return true;
    }
    bool feed(const std::vector<tok> & toks, uint32_t n_ubatch) {
        return feed(toks, n_ubatch, [](const llama_memory_hybrid_idx_context *, const llama_ubatch &, uint32_t) {});
    }
};
