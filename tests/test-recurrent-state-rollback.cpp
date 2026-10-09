// TODO: merge with test-save-load-state.cpp
// TODO: merge with test-state-restore-fragmented.cpp

#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "log.h"
#include "llama-cpp.h"
#include "llama.h"

#include "../src/llama-io.h"
#include "../src/llama-memory.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

enum class test_status {
    PASS,
    FAIL,
    SKIP,
};

static const char * test_status_str(test_status status) {
    switch (status) {
        case test_status::PASS: return "\033[1;32mPASS\033[m";
        case test_status::FAIL: return "\033[1;31mFAIL\033[m";
        case test_status::SKIP: return "\033[1;33mSKIP\033[m";
    }
    return "";
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens) {
    common_batch batch(ctx);
    for (uint32_t pos = 0; pos < tokens.size(); ++pos) {
        batch.add(tokens[pos], pos, 0, pos + 1 == tokens.size());
    }
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq = 0) {
    common_batch batch(ctx);
    batch.add(tok, pos, seq, true);
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

struct cache_buffer_collector : llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

static llama_context_ptr init_ctx(llama_model * model, llama_context_params cparams, uint8_t fill) {
    llama_context_ptr ctx{llama_init_from_model(model, cparams)};
    if (!ctx || fill == 0) {
        return ctx;
    }

    // Use a full ubatch so buffer discovery preserves prefill allocation sizes.
    const uint32_t n_tokens = llama_n_ubatch(ctx.get());
    if (!decode_tokens(ctx.get(), std::vector<llama_token>(n_tokens, 0))) {
        return nullptr;
    }
    llama_synchronize(ctx.get());
    cache_buffer_collector collector;
    llama_get_memory(ctx.get())->state_write(collector);
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    if (collector.buffers.empty()) {
        LOG_ERR("%s: no cache buffers found\n", __func__);
        return nullptr;
    }
    for (auto * buffer : collector.buffers) {
        ggml_backend_buffer_clear(buffer, fill);
    }
    return ctx;
}

static llama_context_ptr make_ctx(const common_params & params, llama_model * model, uint8_t fill) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return init_ctx(model, cparams, fill);
}

static float logit_diff(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) ? std::fabs(a - b) : std::numeric_limits<float>::infinity();
}

static double nmse(const float * a, const float * b, int n) {
    double mse_ab = 0.0;
    double mse_a0 = 0.0;
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return std::numeric_limits<double>::infinity();
        }
        const double diff = (double) a[i] - b[i];
        mse_ab += diff*diff;
        mse_a0 += (double) a[i]*a[i];
    }
    return mse_a0 == 0.0 ? (mse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : mse_ab/mse_a0;
}

// Roll back multiple sequences, then replay them in a single batch whose
// per-seq token count exceeds n_ubatch: each seq's replay spans several
// ubatches while its rollback restore is still pending. Compared against a
// reference context that never advanced past the rollback point and decodes
// the identical replay batch.
static test_status test_multi_seq_split_replay(const common_params & params, llama_model * model, uint8_t fill) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 20;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return init_ctx(model, cparams, fill);
    };

    llama_context_ptr ctx_roll = make_ctx_multi();
    llama_context_ptr ctx_ref  = make_ctx_multi();
    if (!ctx_roll || !ctx_ref) {
        LOG_ERR("%s: failed to init multi-seq contexts\n", __func__);
        return test_status::FAIL;
    }

    if (llama_n_rs_seq(ctx_roll.get()) < n_rollback) {
        LOG_INF("%s: skipping because n_rs_seq is too small\n", __func__);
        return test_status::SKIP;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // decode tokens [p_begin, p_end) of seq s, a batch belongs to one context so it is built per call
    const auto decode_range = [&](llama_context * ctx, uint32_t s, llama_pos p_begin, llama_pos p_end) {
        common_batch batch(ctx);
        for (llama_pos pos = p_begin; pos < p_end; ++pos) {
            batch.add(tok(s, pos), pos, (llama_seq_id) s, false);
        }
        return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
    };

    // Keep the anchor at p0 - 1 when rolling back the tail.
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        ok = ok && decode_range(ctx_roll.get(), s, 0, (llama_pos) p0 - 1);
        ok = ok && decode_range(ctx_ref.get(),  s, 0, (llama_pos) p0 - 1);

        ok = ok && decode_range(ctx_roll.get(), s, (llama_pos) p0 - 1, (llama_pos) n_prompt);
        ok = ok && decode_range(ctx_ref.get(),  s, (llama_pos) p0 - 1, (llama_pos) p0);

        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0, -1);

        // a second partial removal while one is pending must be refused
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0 - 1, -1);
    }
    if (!ok) {
        LOG_ERR("%s: multi-seq prefill/rollback failed\n", __func__);
        return test_status::FAIL;
    }

    // all seqs replay in a single batch
    const auto decode_replay = [&](llama_context * ctx) {
        common_batch batch(ctx);
        for (uint32_t s = 0; s < n_seqs; ++s) {
            for (uint32_t i = 0; i < n_replay; ++i) {
                const llama_pos pos = p0 + (llama_pos) i;
                batch.add(tok(s, pos), pos, (llama_seq_id) s, true);
            }
        }
        return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
    };
    ok = decode_replay(ctx_roll.get());
    ok = ok && decode_replay(ctx_ref.get());
    if (!ok) {
        LOG_ERR("%s: multi-seq replay decode failed\n", __func__);
        return test_status::FAIL;
    }

    // both contexts decode identical batches, so the logits should match;
    // random dummy models can still drift up to ~1.7e-5, so the bound is 1e-4
    constexpr float nmse_eps = 1e-4f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    double   nmse_ab   = 0.0;
    double   nmse_a0   = 0.0;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll.get(), i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            LOG_ERR("%s: missing multi-seq logits at index %u\n", __func__, i);
            return test_status::FAIL;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            const float diff = logit_diff(r, f);
            if (diff > 0.0f && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_ab += d*d;
                nmse_a0 += (double) r*r;
            } else {
                nmse_ab = std::numeric_limits<double>::infinity();
                nmse_a0 = 1.0;
            }
        }
    }
    const double nmse_val = nmse_a0 == 0.0 ? (nmse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_ab/nmse_a0;

    if (nmse_val > nmse_eps) {
        LOG_ERR("%s: multi-seq split replay logits mismatch (max diff %g, nmse %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, nmse_val, seq_first, pos_first);
        return test_status::FAIL;
    }

    LOG_INF("%s: multi-seq split replay matched (max diff %g, nmse %g)\n", __func__, (double) diff_max, nmse_val);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        common_batch batch_tail(ctx_ref.get());
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            batch_tail.add(tok(0, pos + 7), pos, 0, false);
        }
        ok = llama_process(ctx_ref.get(), LLAMA_PROCESS_TYPE_DECODE, batch_tail.get()) == 0;
    }

    float diff_tail = 0.0f;
    double nmse_tail_ab = 0.0;
    double nmse_tail_a0 = 0.0;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        ok = decode_one(ctx_roll.get(), tok(1, pos), pos, 1);
        ok = ok && decode_one(ctx_ref.get(), tok(1, pos), pos, 1);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll.get(), 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            diff_tail = std::max(diff_tail, logit_diff(r, f));
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_tail_ab += d*d;
                nmse_tail_a0 += (double) r*r;
            } else {
                nmse_tail_ab = std::numeric_limits<double>::infinity();
                nmse_tail_a0 = 1.0;
            }
        }
    }
    const double nmse_tail = nmse_tail_a0 == 0.0 ? (nmse_tail_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_tail_ab/nmse_tail_a0;

    if (!ok || nmse_tail > nmse_eps) {
        LOG_ERR("%s: seq-1-only decode leaked seq 0 state (ok=%d, max diff %g, nmse %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail, nmse_tail);
        return test_status::FAIL;
    }

    LOG_INF("%s: seq-1-only decode independent of seq 0 (max diff %g, nmse %g)\n", __func__, (double) diff_tail, nmse_tail);
    return test_status::PASS;
}

static void set_tensor_data_scaled(ggml_tensor * tensor, void * userdata) {
    const size_t seed = *(const size_t *) userdata ^ std::hash<std::string>{}(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0f);

    // norm weights and matrices at scale 1: at the 0.01 scale of the generated models the recurrent
    // branch is too small to change the logits, so a wrong recurrent state would go unnoticed
    const bool  is_norm = strstr(tensor->name, "norm") != nullptr;
    const float scale   = ggml_n_dims(tensor) > 1 ? 1.0f : 1.0e-2f;

    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    std::vector<float> tmp(ggml_nelements(tensor));
    for (auto & x : tmp) {
        x = is_norm ? 1.0f : scale*dis(gen);
    }
    ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
}

// same hparams as the model file, new random weights
static llama_model * load_model_scaled(common_params & params) {
    gguf_init_params gparams = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * meta_file = gguf_init_from_file(params.model.path.c_str(), gparams);
    if (meta_file == nullptr) {
        return nullptr;
    }
    gguf_context * meta = gguf_init_empty();
    gguf_set_kv(meta, meta_file);
    gguf_free(meta_file);

    size_t seed = 1234;
    llama_model * model = llama_model_init_from_user(meta, set_tensor_data_scaled, &seed, common_model_params_to_llama(params));
    gguf_free(meta);
    return model;
}

static void set_graph_reuse_disable(bool disable) {
#ifdef _WIN32
    _putenv_s("LLAMA_GRAPH_REUSE_DISABLE", disable ? "1" : "");
#else
    if (disable) {
        setenv("LLAMA_GRAPH_REUSE_DISABLE", "1", 1);
    } else {
        unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    }
#endif
}

// Two sequences share ubatches while seq_cp swaps them, forks one and rolls it back, with graph reuse on.
// With n_rs_seq = 0 the recurrent state can be read in place when no state is copied, so a seq_cp between
// two same-shape ubatches must stop the graph from being reused. Checked against a context without graph
// reuse (same ubatches, so bit-exact), a context with n_rs_seq = 1, which never reads the state in place,
// and a context that decodes one sequence per ubatch.
static bool test_seq_cp_graph_reuse(const common_params & params, llama_model * model) {
    const char * func    = __func__;
    const int    n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    const auto make = [&](uint32_t n_rs_seq, bool reuse) {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = 3;
        cparams.n_rs_seq   = n_rs_seq;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 64;
        cparams.n_ubatch   = 64;
        cparams.kv_unified = true;
        set_graph_reuse_disable(!reuse);
        llama_context * ctx = llama_init_from_model(model, cparams);
        set_graph_reuse_disable(false);
        return ctx;
    };

    struct arm {
        const char *    name;
        llama_context * ctx;
        bool            split;    // one llama_decode per sequence
        float           eps;      // max |logit diff| against the first arm, relative to its max |logit|
        float           diff_max;
    };
    std::vector<arm> arms = {
        { "reuse",       make(0, true),  false, 0.0f,  0.0f },
        { "no-reuse",    make(0, false), false, 1e-6f, 0.0f },
        { "no-in-place", make(1, false), false, 1e-5f, 0.0f },
        { "seq-split",   make(0, false), true,  1e-4f, 0.0f },
    };

    const auto cleanup = [&]() {
        for (auto & a : arms) {
            llama_free(a.ctx);
        }
    };

    for (const auto & a : arms) {
        if (a.ctx == nullptr) {
            fprintf(stderr, "%s : failed to init %s context\n", __func__, a.name);
            cleanup();
            return false;
        }
    }

    llama_pos pos[3] = { 0, 0, 0 };
    uint32_t  n_step = 0;
    bool      ok     = true;

    const auto decode = [&](const std::vector<llama_seq_id> & seqs, uint32_t n_tokens) {
        if (!ok) {
            return;
        }
        n_step++;
        const auto token = [&](llama_seq_id s, uint32_t i) {
            return (llama_token) ((13*n_step + 7*(uint32_t) s + 3*i + 1) % (uint32_t) n_vocab);
        };

        const size_t n_rows = seqs.size()*n_tokens;
        std::vector<std::vector<float>> logits(arms.size());
        for (size_t k = 0; ok && k < arms.size(); ++k) {
            common_batch batch(arms[k].ctx);
            logits[k].reserve(n_rows*n_vocab);
            const size_t n_batches = arms[k].split ? seqs.size() : 1;
            for (size_t b = 0; ok && b < n_batches; ++b) {
                batch.clear();
                for (size_t j = 0; j < seqs.size(); ++j) {
                    if (arms[k].split && j != b) {
                        continue;
                    }
                    for (uint32_t i = 0; i < n_tokens; ++i) {
                        batch.add(token(seqs[j], i), pos[seqs[j]] + (llama_pos) i, seqs[j], true);
                    }
                }
                ok = llama_process(arms[k].ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
                for (int32_t i = 0; ok && i < batch.size(); ++i) {
                    const float * l = llama_get_logits_ith(arms[k].ctx, i);
                    logits[k].insert(logits[k].end(), l, l + n_vocab);
                }
            }
            if (!ok) {
                fprintf(stderr, "%s : %s decode failed at step %u\n", func, arms[k].name, n_step);
            }
        }

        for (size_t r = 0; ok && r < n_rows; ++r) {
            const float * ref = logits[0].data() + r*n_vocab;
            float ref_max = 0.0f;
            for (int t = 0; t < n_vocab; ++t) {
                ref_max = std::max(ref_max, std::fabs(ref[t]));
            }
            for (size_t k = 1; k < arms.size(); ++k) {
                const float * cur = logits[k].data() + r*n_vocab;
                float diff = 0.0f;
                for (int t = 0; t < n_vocab; ++t) {
                    diff = std::max(diff, logit_diff(ref[t], cur[t]));
                }
                diff /= std::max(ref_max, std::numeric_limits<float>::min());
                arms[k].diff_max = std::max(arms[k].diff_max, diff);
                if (!(diff <= arms[k].eps)) {
                    fprintf(stderr, "%s : step %u, seq %d row %zu: %s differs from %s by %g (rel)\n",
                            func, n_step, seqs[r/n_tokens], r % n_tokens, arms[k].name, arms[0].name, (double) diff);
                    ok = false;
                }
            }
        }
        for (llama_seq_id s : seqs) {
            pos[s] += (llama_pos) n_tokens;
        }
    };

    const auto seq_cp = [&](llama_seq_id src, llama_seq_id dst) {
        for (auto & a : arms) {
            llama_memory_t mem = llama_get_memory(a.ctx);
            ok = ok && llama_memory_seq_rm(mem, dst, -1, -1);
            llama_memory_seq_cp(mem, src, dst, -1, -1);
        }
        pos[dst] = pos[src];
    };

    const auto seq_rm = [&](llama_seq_id s) {
        for (auto & a : arms) {
            ok = ok && llama_memory_seq_rm(llama_get_memory(a.ctx), s, -1, -1);
        }
        pos[s] = 0;
    };

    const auto swap_01 = [&]() {
        seq_cp(0, 2);
        seq_cp(1, 0);
        seq_cp(2, 1);
        seq_rm(2);
    };

    // seq 1 is decoded first, so the first shared ubatch reorders the cells
    decode({ 1 }, 5);
    decode({ 0 }, 5);
    for (uint32_t n_tokens : { 2u, 3u, 1u }) {
        decode({ 0, 1 }, n_tokens);
        decode({ 0, 1 }, n_tokens);
        decode({ 0, 1 }, n_tokens);

        // same ubatch shape, head and rs_z as the step before: only the state copy map changes
        swap_01();
        decode({ 0, 1 }, n_tokens);
        decode({ 0, 1 }, n_tokens);

        decode({ 0 }, n_tokens);
        decode({ 1 }, n_tokens);
        decode({ 0 }, n_tokens);
        decode({ 1 }, n_tokens);

        // fork seq 0, advance it, then roll it back to the fork
        seq_cp(0, 2);
        decode({ 0, 1 }, n_tokens);
        decode({ 0, 1 }, n_tokens);
        seq_cp(2, 0);
        decode({ 0, 1 }, n_tokens);
        seq_rm(2);
        decode({ 0, 1 }, n_tokens);
        decode({ 0, 1 }, n_tokens);
    }

    const int32_t n_reused = llama_perf_context(arms[0].ctx).n_reused;
    if (ok && n_reused == 0) {
        fprintf(stderr, "%s : graph reuse was not exercised\n", __func__);
        ok = false;
    }

    if (ok) {
        fprintf(stderr, "%s : %u steps matched, %d graphs reused (max rel diff: %s %g, %s %g, %s %g)\n", __func__, n_step, n_reused,
                arms[1].name, (double) arms[1].diff_max, arms[2].name, (double) arms[2].diff_max, arms[3].name, (double) arms[3].diff_max);
    }

    cleanup();
    return ok;
}

static test_status test_seq_cp_graph_reuse_for_model(const std::string & model_path, const common_params & base_params) {
    common_params params = base_params;
    params.model.path = model_path;

    gguf_init_params gparams = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * meta = gguf_init_from_file(model_path.c_str(), gparams);
    if (meta == nullptr) {
        LOG_ERR("%s: failed to read model metadata\n", __func__);
        return test_status::FAIL;
    }
    const int64_t key = gguf_find_key(meta, "general.architecture");
    const std::string arch = key >= 0 ? gguf_get_val_str(meta, key) : "";
    gguf_free(meta);

    // the in-place recurrent state is only enabled for qwen35 and qwen35moe
    if (arch != "qwen35" && arch != "qwen35moe") {
        LOG_INF("%s: skipping test_seq_cp_graph_reuse for %s\n", __func__, arch.c_str());
        return test_status::SKIP;
    }

    llama_model_ptr model_scaled{load_model_scaled(params)};
    if (!model_scaled) {
        LOG_ERR("%s: failed to create scaled model\n", __func__);
        return test_status::FAIL;
    }
    return test_seq_cp_graph_reuse(params, model_scaled.get()) ? test_status::PASS : test_status::FAIL;
}

// Save a rolled-back single-seq state, restore it into fresh and dirty
// contexts, and verify exact logit matches on replay.
static test_status test_rollback(const common_params & params, llama_model * model, uint8_t fill) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_ptr ctx_src = make_ctx(params, model, fill);
    llama_context_ptr ctx_dst = make_ctx(params, model, fill);
    if (!ctx_src || !ctx_dst) {
        LOG_ERR("%s: failed to init contexts\n", __func__);
        return test_status::FAIL;
    }

    if (llama_n_rs_seq(ctx_src.get()) == 0) {
        LOG_INF("%s: skipping because n_rs_seq is disabled\n", __func__);
        return test_status::SKIP;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src.get(), "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src.get());
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        LOG_INF("%s: skipping because n_rs_seq is too small\n", __func__);
        return test_status::SKIP;
    }
    if (tokens.empty()) {
        LOG_ERR("%s: not enough prompt tokens\n", __func__);
        return test_status::FAIL;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src.get(), tokens)) {
        LOG_ERR("%s: failed to decode prompt\n", __func__);
        return test_status::FAIL;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1)) {
        LOG_ERR("%s: rollback failed\n", __func__);
        return test_status::FAIL;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src.get(), 0, 0);
    ckpt.load_tgt(ctx_dst.get(), 0, 0);

    constexpr float nmse_eps = 0.0;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src.get(), tokens[pos], pos) ||
                !decode_one(ctx_dst.get(), tokens[pos], pos)) {
                LOG_ERR("%s: %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src.get(), 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst.get(), 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                LOG_ERR("%s: missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            const double nmse_val = nmse(logits_src, logits_dst, n_vocab);
            int token_first = -1;
            for (int token = 0; token < n_vocab; ++token) {
                if (logit_diff(logits_src[token], logits_dst[token]) > 0.0f && token_first < 0) {
                    token_first = token;
                }
            }
            if (nmse_val > nmse_eps) {
                LOG_ERR("%s: %s logits mismatch at position %d, first token %d, nmse %g\n",
                        __func__, mode, pos, token_first, nmse_val);
                return false;
            }
        }
        return true;
    };
    if (!replay_and_compare("full")) {
        return test_status::FAIL;
    }

    // TODO: this test is invalid because RS rollback is only correct once after a ubatch with more than n_rs_seq tokens
    //       this is not the case here. add asserts and guardrails to prevent such attempts
    //if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1) ||
    //    !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
    //    fprintf(stderr, "%s : partial rollback failed\n", __func__);
    //    return 1;
    //}

    //constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    //common_prompt_checkpoint ckpt_partial;
    //ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    //ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    //if (!replay_and_compare("partial")) {
    //    return 1;
    //}

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context_ptr ctx_dirty = make_ctx(params, model, fill);
    if (!ctx_dirty) {
        LOG_ERR("%s: failed to init dirty ctx\n", __func__);
        return test_status::FAIL;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty.get(), noise)) {
        LOG_ERR("%s: dirty prompt decode failed\n", __func__);
        return test_status::FAIL;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty.get()), 0, rollback_pos, -1)) {
        LOG_ERR("%s: dirty rollback failed\n", __func__);
        return test_status::FAIL;
    }

    ckpt.load_tgt(ctx_dirty.get(), 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty.get(), tokens[pos], pos)) {
            LOG_ERR("%s: dirty replay failed at position %d\n", __func__, pos);
            return test_status::FAIL;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty.get(), 0);
        if (logits_dirty == nullptr) {
            LOG_ERR("%s: missing dirty logits at position %d\n", __func__, pos);
            return test_status::FAIL;
        }

        const double nmse_dirty = nmse(logits_src_replay[i].data(), logits_dirty, n_vocab);
        int token_first = -1;
        for (int token = 0; token < n_vocab; ++token) {
            if (logit_diff(logits_src_replay[i][token], logits_dirty[token]) > 0.0f && token_first < 0) {
                token_first = token;
            }
        }
        if (nmse_dirty > nmse_eps) {
            LOG_ERR("%s: dirty-ctx logits mismatch at position %d, first token %d, nmse %g\n",
                    __func__, pos, token_first, nmse_dirty);
            return test_status::FAIL;
        }
    }

    LOG_INF("%s: recurrent rollback checkpoint restored successfully\n", __func__);
    return test_status::PASS;
}

// Decode a prompt into seq 0, share its cells with a second sequence, then keep
// decoding both.
static test_status test_shared_seq_reserve(const common_params & params, llama_model * model, uint8_t fill) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // these archs reserve the final pp graph with n_seqs = 1, so every multi-seq
    // graph has a different layout and re-reserves by design
    // see [TAG_RESERVE_DIAG_DECAY] in llama-context.cpp
    char arch_str[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch_str, sizeof(arch_str));
    if (strcmp(arch_str, "kimi-linear") == 0 || strcmp(arch_str, "minimax-01") == 0) {
        LOG_INF("%s: skipping %s, its reserve uses n_seqs = 1\n", __func__, arch_str);
        return test_status::SKIP;
    }

    constexpr uint32_t n_seqs     = 2;
    constexpr uint32_t n_prompt   = 128;
    constexpr uint32_t n_continue = 32;

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = n_seqs;
    cparams.n_ctx      = 512;
    cparams.n_batch    = 256;
    cparams.n_ubatch   = 64;
    cparams.kv_unified = true; // only a unified cache shares cells on seq_cp

    llama_context_ptr ctx = init_ctx(model, cparams, fill);
    if (!ctx) {
        LOG_ERR("%s: failed to init context\n", __func__);
        return test_status::FAIL;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    {
        common_batch batch(ctx.get());
        for (llama_pos pos = 0; pos < (llama_pos) n_prompt; ++pos) {
            batch.add(tok(0, pos), pos, 0, false);
        }
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: prompt decode failed\n", __func__);
            return test_status::FAIL;
        }
    }

    // this is what llama-batched-bench does for -pps
    llama_memory_seq_cp(llama_get_memory(ctx.get()), 0, 1, -1, -1);

    for (uint32_t i = 0; i < n_continue; ++i) {
        const llama_pos pos = (llama_pos) (n_prompt + i);

        common_batch batch(ctx.get());
        for (uint32_t s = 0; s < n_seqs; ++s) {
            batch.add(tok(s, pos), pos, (llama_seq_id) s, true);
        }
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: shared-seq decode failed at step %u\n", __func__, i);
            return test_status::FAIL;
        }

        for (uint32_t s = 0; s < n_seqs; ++s) {
            const float * logits = llama_get_logits_ith(ctx.get(), (int) s);
            if (logits == nullptr) {
                LOG_ERR("%s: missing shared-seq logits at index %u\n", __func__, s);
                return test_status::FAIL;
            }
            for (int t = 0; t < n_vocab; ++t) {
                if (!std::isfinite(logits[t])) {
                    LOG_ERR("%s: non-finite shared-seq logit at step %u, seq %u, index %d\n", __func__, i, s, t);
                    return test_status::FAIL;
                }
            }
        }
    }

    LOG_INF("%s: shared-seq decode succeeded (%u tokens after seq_cp)\n", __func__, n_continue*n_seqs);
    return test_status::PASS;
}

static test_status merge_status(test_status a, test_status b) {
    if (a == test_status::FAIL || b == test_status::FAIL) {
        return test_status::FAIL;
    }
    if (a == test_status::PASS || b == test_status::PASS) {
        return test_status::PASS;
    }
    return test_status::SKIP;
}

struct test_results {
    test_status rollback = test_status::SKIP;
    test_status replay   = test_status::SKIP;
    test_status shared   = test_status::SKIP;
};

// Run every test for an initialized model over both cache fills.
static test_results run_tests(const common_params & params, llama_model * model) {
    test_results res;
    for (uint8_t fill : { 0, 0x3e }) {
        LOG_INF("%s: testing with cache fill 0x%02x\n", __func__, fill);
        const test_status rb = test_rollback(params, model, fill);
        const test_status rp = test_multi_seq_split_replay(params, model, fill);
        const test_status ss = test_shared_seq_reserve(params, model, fill);
        res.rollback = merge_status(res.rollback, rb);
        res.replay   = merge_status(res.replay,   rp);
        res.shared   = merge_status(res.shared,   ss);
        if (rb == test_status::FAIL || rp == test_status::FAIL || ss == test_status::FAIL) {
            break;
        }
    }
    return res;
}

// Run the tests for a single model file.
// Returns the per-test statuses.
static test_results run_tests_for_model(const std::string & model_path, const struct common_params & base_params) {
    struct common_params params = base_params;
    params.model.path = model_path;

    auto llama_init = common_init_from_params(params, true);
    auto * model = llama_init->model();

    if (model == nullptr) {
        LOG_ERR("%s: failed to init model '%s'\n", __func__, model_path.c_str());
        // a model that cannot be loaded is a failure, not a skip
        return { test_status::FAIL, test_status::FAIL, test_status::FAIL };
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        LOG_INF("%s: skipping for non-recurrent model\n", __func__);
        return {};
    }

    return run_tests(params, model);
}

static void print_usage(int /* argc */, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n  %s -m your_model.gguf\n", argv[0]);
    LOG("\n  %s --models tests/test-models\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    // extract our own --models DIR option before handing the rest to the common arg parser
    std::string models_dir;
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --models requires a directory argument\n", __func__);
                return 1;
            }
            models_dir = argv[i + 1];
            i++;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    filtered_argv.push_back(nullptr);
    const int fargc = (int)filtered_argv.size() - 1;

    // in --models mode there is no single model; set a placeholder so the common parser's
    // "--model is required" check passes (each model is set individually inside the loop)
    if (!models_dir.empty()) {
        params.model.path = models_dir;
    }

    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    llama_backend_init();

    if (!models_dir.empty()) {
        // run every test over each dummy model in the directory
        if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
            LOG_ERR("%s: models directory '%s' does not exist\n", __func__, models_dir.c_str());
            return 1;
        }

        std::vector<std::string> models;
        for (const auto & entry : std::filesystem::directory_iterator(models_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                models.push_back(entry.path().string());
            }
        }
        std::sort(models.begin(), models.end());

        if (models.empty()) {
            LOG_ERR("%s: no .gguf models found in '%s'\n", __func__, models_dir.c_str());
            return 1;
        }

        size_t name_width = 5; // "Model"
        for (const auto & model_path : models) {
            name_width = std::max(name_width, std::filesystem::path(model_path).filename().string().size());
        }

        // silence everything but the table itself (LOG has verbosity LOG_LEVEL_OUTPUT = 0)
        common_log_set_verbosity_thold(0);

        LOG("%-*s  %-8s  %-12s  %-10s  %s\n", (int) name_width, "Model", "rollback", "split replay", "shared seq", "seq cp graph reuse");
        common_log_flush(common_log_main());

        size_t n_pass[4] = { 0, 0, 0, 0 };
        size_t n_skip[4] = { 0, 0, 0, 0 };
        size_t n_fail[4] = { 0, 0, 0, 0 };
        for (const auto & model_path : models) {
            const auto name = std::filesystem::path(model_path).filename().string();

            LOG("%-*s", (int) name_width, name.c_str());

            const test_results res = run_tests_for_model(model_path, params);
            const test_status graph = (res.rollback == test_status::FAIL || res.replay == test_status::FAIL)
                    ? test_status::SKIP : test_seq_cp_graph_reuse_for_model(model_path, params);

            // all status strings have the same raw length, so the columns line up;
            // pad the first status to the width of the "rollback" header + separator
            LOG("  %s      %s          %s       %s", test_status_str(res.rollback), test_status_str(res.replay),
                test_status_str(res.shared), test_status_str(graph));
            LOG("\n");
            common_log_flush(common_log_main());

            const test_status all[4] = { res.rollback, res.replay, res.shared, graph };
            for (int t = 0; t < 4; ++t) {
                switch (all[t]) {
                    case test_status::PASS: n_pass[t]++; break;
                    case test_status::FAIL: n_fail[t]++; break;
                    case test_status::SKIP: n_skip[t]++; break;
                }
            }
        }

        common_log_set_verbosity_thold(LOG_DEFAULT_LLAMA);
        common_log_flush(common_log_main());

        LOG_INF("%s: rollback:     %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, n_pass[0], n_skip[0], n_fail[0], models.size());
        LOG_INF("%s: split replay: %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, n_pass[1], n_skip[1], n_fail[1], models.size());
        LOG_INF("%s: shared seq:   %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, n_pass[2], n_skip[2], n_fail[2], models.size());

        LOG_INF("%s: seq cp graph reuse: %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, n_pass[3], n_skip[3], n_fail[3], models.size());

        return (n_fail[0] + n_fail[1] + n_fail[2] + n_fail[3]) == 0 ? 0 : 1;
    }

    // single-model mode
    const test_results res = run_tests_for_model(params.model.path, params);

    if (res.rollback == test_status::FAIL || res.replay == test_status::FAIL || res.shared == test_status::FAIL) {
        return 1;
    }
    return test_seq_cp_graph_reuse_for_model(params.model.path, params) == test_status::FAIL ? 1 : 0;
}
