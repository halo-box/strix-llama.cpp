#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include "../src/llama-io.h"
#include "../src/llama-memory.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
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

static llama_context * init_ctx(llama_model * model, llama_context_params cparams, uint8_t fill) {
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr || fill == 0) {
        return ctx;
    }

    // Use a full ubatch so buffer discovery preserves prefill allocation sizes.
    const uint32_t n_tokens = llama_n_ubatch(ctx);
    if (!decode_tokens(ctx, std::vector<llama_token>(n_tokens, 0), n_tokens)) {
        llama_free(ctx);
        return nullptr;
    }
    llama_synchronize(ctx);
    cache_buffer_collector collector;
    llama_get_memory(ctx)->state_write(collector);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (collector.buffers.empty()) {
        fprintf(stderr, "%s : no cache buffers found\n", __func__);
        llama_free(ctx);
        return nullptr;
    }
    for (auto * buffer : collector.buffers) {
        ggml_backend_buffer_clear(buffer, fill);
    }
    return ctx;
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint8_t fill) {
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

// Roll back multiple sequences, then replay them in a single batch whose
// per-seq token count exceeds n_ubatch: each seq's replay spans several
// ubatches while its rollback restore is still pending. Compared against a
// reference context that never advanced past the rollback point and decodes
// the identical replay batch.
static bool test_multi_seq_split_replay(const common_params & params, llama_model * model, const int n_vocab, uint8_t fill) {
    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
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

    llama_context * ctx_roll = make_ctx_multi();
    llama_context * ctx_ref  = make_ctx_multi();
    if (ctx_roll == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "%s : failed to init multi-seq contexts\n", __func__);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx_roll);
        llama_free(ctx_ref);
    };

    if (llama_n_rs_seq(ctx_roll) < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        cleanup();
        return true;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // both contexts decode the identical [0, p0) prefill; only ctx_roll decodes
    // the tail, which is then rolled back so its restore is pending at replay
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (llama_pos pos = 0; pos < (llama_pos) p0; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        ok = ok && llama_decode(ctx_ref,  batch) == 0;

        common_batch_clear(batch);
        for (llama_pos pos = p0; pos < (llama_pos) n_prompt; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        llama_batch_free(batch);

        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0, -1);

        // a second partial removal while one is pending must be refused
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0 - 1, -1);
    }
    if (!ok) {
        fprintf(stderr, "%s : multi-seq prefill/rollback failed\n", __func__);
        cleanup();
        return false;
    }

    llama_batch batch = llama_batch_init(n_seqs*n_replay, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t i = 0; i < n_replay; ++i) {
            const llama_pos pos = p0 + (llama_pos) i;
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, true);
        }
    }
    ok = llama_decode(ctx_roll, batch) == 0;
    ok = ok && llama_decode(ctx_ref, batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        fprintf(stderr, "%s : multi-seq replay decode failed\n", __func__);
        cleanup();
        return false;
    }

    // identical ubatch shapes from bit-exact states: a correct implementation
    // matches bitwise, so eps only allows backend scheduling noise
    constexpr float eps = 1e-7f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll, i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            fprintf(stderr, "%s : missing multi-seq logits at index %u\n", __func__, i);
            cleanup();
            return false;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float diff = logit_diff(l_roll[t], l_ref[t]);
            if (diff > eps && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
        }
    }

    if (diff_max > eps) {
        fprintf(stderr, "%s : multi-seq split replay logits mismatch (max diff %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, seq_first, pos_first);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : multi-seq split replay matched (max diff %g)\n", __func__, (double) diff_max);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        llama_batch batch_tail = llama_batch_init(n_tail, 0, 1);
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            common_batch_add(batch_tail, tok(0, pos + 7), pos, { 0 }, false);
        }
        ok = llama_decode(ctx_ref, batch_tail) == 0;
        llama_batch_free(batch_tail);
    }

    float diff_tail = 0.0f;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        llama_batch batch_one = llama_batch_init(1, 0, 1);
        common_batch_add(batch_one, tok(1, pos), pos, { 1 }, true);
        ok = llama_decode(ctx_roll, batch_one) == 0;
        ok = ok && llama_decode(ctx_ref, batch_one) == 0;
        llama_batch_free(batch_one);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll, 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            diff_tail = std::max(diff_tail, logit_diff(l_roll[t], l_ref[t]));
        }
    }

    if (!ok || diff_tail > eps) {
        fprintf(stderr, "%s : seq-1-only decode leaked seq 0 state (ok=%d, max diff %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : seq-1-only decode independent of seq 0 (max diff %g)\n", __func__, (double) diff_tail);
    cleanup();
    return true;
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
        llama_batch batch = llama_batch_init(n_rows, 0, 1);
        for (size_t k = 0; ok && k < arms.size(); ++k) {
            logits[k].reserve(n_rows*n_vocab);
            const size_t n_batches = arms[k].split ? seqs.size() : 1;
            for (size_t b = 0; ok && b < n_batches; ++b) {
                common_batch_clear(batch);
                for (size_t j = 0; j < seqs.size(); ++j) {
                    if (arms[k].split && j != b) {
                        continue;
                    }
                    for (uint32_t i = 0; i < n_tokens; ++i) {
                        common_batch_add(batch, token(seqs[j], i), pos[seqs[j]] + (llama_pos) i, { seqs[j] }, true);
                    }
                }
                ok = llama_decode(arms[k].ctx, batch) == 0;
                for (int32_t i = 0; ok && i < batch.n_tokens; ++i) {
                    const float * l = llama_get_logits_ith(arms[k].ctx, i);
                    logits[k].insert(logits[k].end(), l, l + n_vocab);
                }
            }
            if (!ok) {
                fprintf(stderr, "%s : %s decode failed at step %u\n", func, arms[k].name, n_step);
            }
        }
        llama_batch_free(batch);

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

static int test_rollback(const common_params & params, llama_model * model, uint8_t fill) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_context * ctx_src = make_ctx(params, model, fill);
    llama_context * ctx_dst = make_ctx(params, model, fill);
    if (ctx_src == nullptr || ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    if (llama_n_rs_seq(ctx_src) == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src, "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src);
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }
    if (tokens.empty()) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src, tokens, n_tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        return 1;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_dst, 0, 0);

    constexpr float eps = 1e-5f;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src, tokens[pos], pos) ||
                !decode_one(ctx_dst, tokens[pos], pos)) {
                fprintf(stderr, "%s : %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src, 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst, 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                fprintf(stderr, "%s : missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            for (int token = 0; token < n_vocab; ++token) {
                if (logit_diff(logits_src[token], logits_dst[token]) > eps) {
                    fprintf(stderr, "%s : %s logits mismatch at position %d, token %d (%g != %g)\n",
                            __func__, mode, pos, token, (double) logits_src[token], (double) logits_dst[token]);
                    return false;
                }
            }
        }
        return true;
    };
    if (!replay_and_compare("full")) {
        return 1;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1) ||
        !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : partial rollback failed\n", __func__);
        return 1;
    }

    constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    common_prompt_checkpoint ckpt_partial;
    ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    if (!replay_and_compare("partial")) {
        return 1;
    }

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context * ctx_dirty = make_ctx(params, model, fill);
    if (ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init dirty ctx\n", __func__);
        return 1;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty, noise, n_tokens)) {
        fprintf(stderr, "%s : dirty prompt decode failed\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : dirty rollback failed\n", __func__);
        return 1;
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty, tokens[pos], pos)) {
            fprintf(stderr, "%s : dirty replay failed at position %d\n", __func__, pos);
            return 1;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty, 0);
        if (logits_dirty == nullptr) {
            fprintf(stderr, "%s : missing dirty logits at position %d\n", __func__, pos);
            return 1;
        }

        for (int token = 0; token < n_vocab; ++token) {
            if (logit_diff(logits_src_replay[i][token], logits_dirty[token]) > eps) {
                fprintf(stderr, "%s : dirty-ctx logits mismatch at position %d, token %d (%g != %g)\n",
                        __func__, pos, token, (double) logits_src_replay[i][token], (double) logits_dirty[token]);
                return 1;
            }
        }
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);
    llama_free(ctx_src);
    llama_free(ctx_dst);
    llama_free(ctx_dirty);

    if (!test_multi_seq_split_replay(params, model, n_vocab, fill)) {
        return 1;
    }

    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    for (uint8_t fill : { 0, 0x3e }) {
        fprintf(stderr, "%s : testing with cache fill 0x%02x\n", __func__, fill);
        if (test_rollback(params, model, fill) != 0) {
            return 1;
        }
    }

    // the in-place recurrent state is only enabled for qwen35 and qwen35moe
    char arch[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    if (strcmp(arch, "qwen35") != 0 && strcmp(arch, "qwen35moe") != 0) {
        fprintf(stderr, "%s : skipping test_seq_cp_graph_reuse for %s\n", __func__, arch);
        return 0;
    }

    llama_model * model_scaled = load_model_scaled(params);
    if (model_scaled == nullptr) {
        fprintf(stderr, "%s : failed to create scaled model\n", __func__);
        return 1;
    }
    const bool ok = test_seq_cp_graph_reuse(params, model_scaled);
    llama_model_free(model_scaled);
    if (!ok) {
        return 1;
    }

    return 0;
}
