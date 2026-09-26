// Regression test for llama_context_params::mtp_draft_vocab.
//
// The draft vocabulary subset belongs to the MTP context that asked for it. Contexts over the same model with a
// different setting (0 = full vocabulary, or another N) must each draft over their own vocabulary, whatever order
// they were created in and while they are alive at the same time.
//
// Needs a qwen35 or qwen35moe model with an MTP head whose MTP block uses the model LM head, e.g.:
//
//   test-mtp-draft-vocab -m Qwen3.8-27B-UD-IQ4_XS.gguf -ngl 99
//
// A context with N > 0 must produce -inf exactly at the token ids outside {id < N} + control + user-defined tokens,
// and its finite logits must match those of a full-vocabulary context fed the same inputs. A context with N = 0 must
// produce finite logits everywhere.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../src/llama-ext.h"

#include <algorithm>
#include <atomic>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct mtp_ctx {
    int32_t        n_vocab_draft; // requested mtp_draft_vocab
    llama_context_ptr ctx;
};

static llama_context_ptr make_ctx(llama_model * model, int32_t n_vocab_draft) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 256;
    cparams.n_batch         = 16;
    cparams.n_ubatch        = 16;
    cparams.n_seq_max       = 1;
    cparams.ctx_type        = LLAMA_CONTEXT_TYPE_MTP;
    cparams.mtp_draft_vocab = n_vocab_draft;
    cparams.n_rs_seq        = 0;
    llama_context_ptr ctx(llama_init_from_model(model, cparams));
    if (ctx) {
        llama_set_embeddings_nextn(ctx.get(), true, /*masked*/ true);
    }
    return ctx;
}

// true for the token ids a context with mtp_draft_vocab = n may draft
static std::vector<uint8_t> draft_mask(const llama_vocab * vocab, int32_t n) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<uint8_t> mask((size_t) n_vocab, 1);
    if (n <= 0 || n >= n_vocab) {
        return mask;
    }
    for (int32_t t = 0; t < n_vocab; ++t) {
        const int attr = (int) llama_vocab_get_attr(vocab, t);
        mask[(size_t) t] = t < n || (attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED));
    }
    return mask;
}

// decode one token with a pseudo-random hidden state at position pos; return the draft logits
static bool decode_step(llama_model * model, llama_context * ctx, llama_pos pos, std::vector<float> & out) {
    const int32_t n_embd  = llama_model_n_embd_out(model);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_batch batch = llama_batch_init(1, n_embd, 1);
    batch.token = (llama_token *) malloc(sizeof(llama_token));

    std::mt19937 rng(1000 + pos);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int32_t i = 0; i < n_embd; ++i) {
        batch.embd[i] = dist(rng);
    }
    batch.token[0]     = (llama_token) ((pos * 7919 + 13) % 30000);
    batch.pos[0]       = pos;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]    = 1;
    batch.n_tokens     = 1;

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        fprintf(stderr, "llama_decode failed: %d\n", rc);
        return false;
    }
    const float * logits = llama_get_logits_ith(ctx, -1);
    out.assign(logits, logits + n_vocab);
    return true;
}

// check the draft logits of a context against its own mask, and against full-vocabulary logits for the same inputs
static bool check_logits(const std::string & label, const std::vector<float> & logits, const std::vector<uint8_t> & mask,
        const std::vector<float> & ref) {
    size_t n_cmp     = 0;    // finite logits compared with the reference
    size_t n_bad_inf = 0;    // -inf (or non-finite) where the context may draft
    size_t n_bad_fin = 0;    // finite where the context must not draft
    size_t n_bad_val = 0;    // finite, but different from the full-vocabulary reference
    double max_diff  = 0.0;
    for (size_t t = 0; t < logits.size(); ++t) {
        const float v = logits[t];
        if (mask[t]) {
            if (!std::isfinite(v)) {
                n_bad_inf++;
            } else if (!ref.empty()) {
                n_cmp++;
                const double d = std::fabs((double) v - (double) ref[t]);
                max_diff = std::max(max_diff, d);
                if (d > 1e-2 * std::max(1.0, std::fabs((double) ref[t]))) {
                    n_bad_val++;
                }
            }
        } else if (!(std::isinf(v) && v < 0)) {
            n_bad_fin++;
        }
    }
    size_t n_keep = 0;
    for (uint8_t m : mask) {
        n_keep += m;
    }
    const size_t i_max = std::max_element(logits.begin(), logits.end()) - logits.begin();
    const bool ok = n_bad_inf == 0 && n_bad_fin == 0 && n_bad_val == 0;
    fprintf(stderr, "  %-44s %s draftable %zu/%zu, non-finite inside %zu, finite outside %zu, argmax %zu (%.3f), "
            "vs full vocab: %zu compared, %zu mismatches, max |diff| %.3g\n",
            label.c_str(), ok ? "OK  " : "FAIL", n_keep, mask.size(), n_bad_inf, n_bad_fin, i_max, logits[i_max],
            n_cmp, n_bad_val, max_diff);
    return ok;
}

// decode the same inputs, interleaved, in every context plus a full-vocabulary reference context, and check each
static bool run_interleaved(llama_model * model, const std::string & name, std::vector<mtp_ctx> & ctxs, int n_steps) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    bool ok = true;
    for (int step = 0; step < n_steps; ++step) {
        std::vector<std::vector<float>> logits(ctxs.size());
        for (size_t i = 0; i < ctxs.size(); ++i) {
            if (!decode_step(model, ctxs[i].ctx.get(), step, logits[i])) {
                return false;
            }
        }
        const std::vector<float> * ref = nullptr;
        for (size_t i = 0; i < ctxs.size(); ++i) {
            if (ctxs[i].n_vocab_draft == 0) {
                ref = &logits[i];
            }
        }
        for (size_t i = 0; i < ctxs.size(); ++i) {
            const std::string label = name + " step " + std::to_string(step) + " ctx " + std::string(1, char('A' + i)) +
                                      " (N=" + std::to_string(ctxs[i].n_vocab_draft) + ")";
            ok &= check_logits(label, logits[i], draft_mask(vocab, ctxs[i].n_vocab_draft),
                               ref && ctxs[i].n_vocab_draft != 0 ? *ref : std::vector<float>());
        }
    }
    return ok;
}

// create contexts one after the other, in the given order, then decode them interleaved
static bool test_sequential(llama_model * model, const std::vector<int32_t> & ns, bool add_ref) {
    std::string name;
    std::vector<mtp_ctx> ctxs;
    for (int32_t n : ns) {
        name += (name.empty() ? "" : ",") + std::to_string(n);
        ctxs.push_back({ n, make_ctx(model, n) });
        if (!ctxs.back().ctx) {
            fprintf(stderr, "failed to create context N=%d\n", n);
            return false;
        }
    }
    name = "create " + name;
    if (add_ref) {
        ctxs.push_back({ 0, make_ctx(model, 0) });
    }
    fprintf(stderr, "%s:\n", name.c_str());
    return run_interleaved(model, name, ctxs, 3);
}

// create two contexts from two threads at the same time, then decode them interleaved
static bool test_concurrent(llama_model * model, int32_t n1, int32_t n2) {
    const std::string name = "concurrent " + std::to_string(n1) + "," + std::to_string(n2);
    fprintf(stderr, "%s:\n", name.c_str());
    std::vector<mtp_ctx> ctxs(2);
    ctxs[0].n_vocab_draft = n1;
    ctxs[1].n_vocab_draft = n2;
    std::atomic<bool> go(false);
    std::thread t0([&] { while (!go) { std::this_thread::yield(); } ctxs[0].ctx = make_ctx(model, n1); });
    std::thread t1([&] { while (!go) { std::this_thread::yield(); } ctxs[1].ctx = make_ctx(model, n2); });
    go = true;
    t0.join();
    t1.join();
    if (!ctxs[0].ctx || !ctxs[1].ctx) {
        fprintf(stderr, "failed to create contexts\n");
        return false;
    }
    ctxs.push_back({ 0, make_ctx(model, 0) });
    return run_interleaved(model, name, ctxs, 2);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    ggml_backend_load_all();

    llama_model_params mparams = common_model_params_to_llama(params);
    mparams.load_mtp = true;
    llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), mparams));
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    if (llama_model_n_layer_nextn(model.get()) == 0) {
        fprintf(stderr, "model has no MTP layers, skipping\n");
        return 0;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    const int32_t n1 = std::min(65536, n_vocab / 2);
    const int32_t n2 = std::min(32768, n_vocab / 4);

    bool ok = true;
    ok &= test_sequential(model.get(), { n1, 0 },  false); // A = N then B = 0
    ok &= test_sequential(model.get(), { 0, n1 },  false); // A = 0 then B = N
    ok &= test_sequential(model.get(), { n1, n2 }, true);  // A = N1 then B = N2
    ok &= test_sequential(model.get(), { n2, n2 }, true);  // two contexts with the same N
    ok &= test_concurrent(model.get(), n1, n2);            // simultaneous creation
    // a subset freed with its last context and requested again
    ok &= test_sequential(model.get(), { n2 }, true);

    fprintf(stderr, "%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
