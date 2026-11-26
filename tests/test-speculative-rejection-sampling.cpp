#include "sampling.h"
#include "llama.h"
#include "llama-cpp.h"
#include "common.h"
#include "arg.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <random>
#include <unordered_map>

// Tests that common_sampler_sample_and_accept_n_rejection preserves the target
// distribution regardless of what the draft distribution Q looks like.

static int test_speculative_rejection_sampling(common_params & params, int n_trials) {
    params.sampling.backend_sampling = false;
    auto llama_init = common_init_from_params(params);

    auto * model_tgt = llama_init->model();
    auto * ctx_tgt   = llama_init->context();
    if (!model_tgt || !ctx_tgt) {
        LOG_ERR("failed to initialize target model and context\n");
        return 1;
    }

    const llama_tokens prompt = { 1, 2, 3, 1, 2 };

    // Process the prompt with the target model to get its true distribution for
    // the position right after the prompt.
    common_batch batch = common_batch_get_one(ctx_tgt, prompt);
    if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        LOG_ERR("failed to decode the test prompt\n");
        return 1;
    }

    common_params_sampling sampling;
    sampling.seed     = 18;
    sampling.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE };
    sampling.temp     = 1.0f;

    // Initialize the sampler for the target model.
    common_sampler_ptr smpl(common_sampler_init(model_tgt, sampling));
    if (!smpl) {
        LOG_ERR("failed to initialize target sampler\n");
        return 1;
    }

    // Sample once so the sampler chain runs and populates cur_p. We don't need
    // the returned token id itself so the return value is ignored.
    common_sampler_sample(smpl.get(), ctx_tgt, -1);

    // Get the target candidates distribution in sorted order.
    const llama_token_data_array * dist_tgt = common_sampler_get_candidates(smpl.get(), true);

    // Clone smpl before the trial loop starts. We need the dist_tgt pointer to stay valid.
    common_sampler_ptr smpl_verify(common_sampler_clone(smpl.get()));
    if (!smpl_verify) {
        LOG_ERR("failed to clone target sampler\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::mt19937 rng(18);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    std::unordered_map<llama_token, int> token_counts;

    std::vector<llama_token_data> q(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        q[i].id = i;
        q[i].logit = 0.0f;
    }

    for (int trial = 0; trial < n_trials; ++trial) {
        float q_sum = 0.0f;

        for (auto & td : q) {
            td.p = uni(rng);
            q_sum += td.p;
        }

        const float inv_sum = 1.0f / q_sum;
        for (auto & td : q) {
            td.p *= inv_sum;
        }

        float u = uni(rng);

        // Fallback to the last bucket to absorb residual probability from FP precision loss
        llama_token draft_tok = q.back().id;
        for (const auto & e : q) {
            u -= e.p;
            if (u <= 0.0f) {
                draft_tok = e.id;
                break;
            }
        }

        // Both indices reuse the prompt's last output logits. The second entry
        // is the bonus draw slot; it is always stale and its result is ignored.
        const llama_tokens draft = { draft_tok };
        const std::vector<std::vector<llama_token_data>> draft_q = { q };
        const std::vector<int> idxs = { -1, -1 };

        const std::vector<llama_token> result =
            common_sampler_sample_and_accept_n_rejection(smpl_verify.get(), ctx_tgt, idxs, draft, draft_q, false);

        token_counts[result[0]]++;
    }

    LOG("\n");
    LOG("Empirical vs target distribution after %d trials:\n", n_trials);

    bool   ok      = true;
    size_t covered = 0;

    auto get_count_for = [&](llama_token id) -> int {
        const auto it = token_counts.find(id);
        return it != token_counts.end() ? it->second : 0;
    };

    // Iterate over the target distribution
    for (size_t i = 0; i < dist_tgt->size; ++i) {
        const llama_token id  = dist_tgt->data[i].id;
        const float       p_t = dist_tgt->data[i].p;

        const int   count = get_count_for(id);
        const float p_e   = (float) count / n_trials;
        const float diff  = std::fabs(p_t - p_e);

        // We only deal with ids in dist_tgt, and if each trial result was one
        // of dist_tgt's candidates then covered ends up equal to the number of
        // trials. If not this inconsistency tells us how many trials produced
        // a token outside of the target's distribution.
        covered += count;

        const float standard_error = std::sqrt(p_t * (1.0f - p_t) / n_trials);
        const float tolerance      = 6.0f * standard_error;

        LOG("token %3d : target = %f, empirical = %f, diff = %f, tol = %f\n", id, p_t, p_e, diff, tolerance);

        if (diff > tolerance) {
            LOG_ERR("token %d: empirical probability %f too far from target %f (diff %f > tol %f)\n",
                    id, p_e, p_t, diff, tolerance);
            ok = false;
        }
    }

    if (covered != (size_t) n_trials) {
        LOG_ERR("%zu of %d trials produced a token outside dist_tgt's candidates\n",
                (size_t) n_trials - covered, n_trials);
        ok = false;
    }

    if (!ok) {
        return 1;
    }

    return 0;
}

// Tests the backend verification (dist_spec): the first output token must follow the target distribution, and each
// trial must give the token of a host implementation of the same step that uses the same random stream.
static int test_speculative_rejection_backend(common_params & params, int n_trials, int top_k) {
    params.sampling.backend_sampling = false;
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx_ref = llama_init->context();
    if (!model || !ctx_ref) {
        LOG_ERR("failed to initialize target model and context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const llama_tokens prompt = { 1, 2, 3, 1, 2 };
    const int n_prompt = (int) prompt.size();

    // reference p after the prompt, from the logits on the host
    {
        common_batch batch = common_batch_get_one(ctx_ref, prompt);
        if (llama_process(ctx_ref, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("failed to decode the test prompt\n");
            return 1;
        }
    }
    std::vector<double> p_ref(n_vocab);
    {
        const float * logits = llama_get_logits_ith(ctx_ref, -1);
        double max_l = logits[0];
        for (int i = 1; i < n_vocab; ++i) {
            max_l = std::max(max_l, (double) logits[i]);
        }
        double cutoff = -INFINITY;
        if (top_k > 0 && top_k < n_vocab) {
            std::vector<float> sorted(logits, logits + n_vocab);
            std::nth_element(sorted.begin(), sorted.begin() + top_k - 1, sorted.end(), std::greater<float>());
            cutoff = sorted[top_k - 1];
        }
        double sum = 0.0;
        for (int i = 0; i < n_vocab; ++i) {
            p_ref[i] = logits[i] >= cutoff ? std::exp(logits[i] - max_l) : 0.0;
            sum += p_ref[i];
        }
        for (auto & p : p_ref) {
            p /= sum;
        }
    }

    // a second context with the backend chain: the draft row and the bonus row
    llama_context_params cparams = common_context_params_to_llama(params);
    cparams.n_outputs_max_per_seq = 2;
    llama_context_ptr ctx_be(llama_init_from_model(model, cparams));
    if (!ctx_be) {
        LOG_ERR("failed to create the backend sampling context\n");
        return 1;
    }

    common_params_sampling sampling;
    sampling.seed             = 18;
    sampling.samplers         = { COMMON_SAMPLER_TYPE_TEMPERATURE };
    if (top_k > 0) {
        sampling.top_k = top_k;
        sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE };
    }
    sampling.temp             = 1.0f;
    sampling.backend_sampling = true;
    sampling.spec_verify      = true;

    common_sampler_ptr smpl(common_sampler_init(model, sampling));
    if (!smpl || !llama_set_sampler(ctx_be.get(), 0, common_sampler_get(smpl.get()))) {
        LOG_ERR("failed to attach the backend sampler\n");
        return 1;
    }

    {
        common_batch batch = common_batch_get_one(ctx_be.get(), llama_tokens(prompt.begin(), prompt.end() - 1));
        if (llama_process(ctx_be.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("failed to decode the prompt prefix\n");
            return 1;
        }
    }

    // the same stream as the sampler: two draws for each of the 2 rows of each decode
    // (the prefix decode above drew for its rows too)
    std::mt19937 rng_be(18);
    std::uniform_real_distribution<double> uni_be(0.0, 1.0);
    for (int k = 0; k < 2*2; ++k) {
        (void) uni_be(rng_be);
    }

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    // the draft candidates: the 8 most likely tokens and random others
    std::vector<int> order(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) { return p_ref[a] > p_ref[b]; });

    std::vector<int> counts(n_vocab, 0);
    int n_same     = 0;
    int n_accepted = 0;

    common_batch batch(ctx_be.get());
    for (int trial = 0; trial < n_trials; ++trial) {
        const int n_q = 1 + (int) (uni(rng) * (COMMON_SPEC_VERIFY_N_CAND - 1));
        std::vector<int> ids(order.begin(), order.begin() + std::min(8, n_q));
        while ((int) ids.size() < n_q) {
            const int id = (int) (uni(rng) * n_vocab) % n_vocab;
            if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
                ids.push_back(id);
            }
        }
        std::vector<llama_token_data> q(n_q);
        float q_sum = 0.0f;
        for (int k = 0; k < n_q; ++k) {
            q[k] = { ids[k], 0.0f, uni(rng) + 1e-3f };
            q_sum += q[k].p;
        }
        for (auto & e : q) {
            e.p /= q_sum;
        }
        float w = uni(rng);
        llama_token d = q.back().id;
        for (const auto & e : q) {
            w -= e.p;
            if (w <= 0.0f) {
                d = e.id;
                break;
            }
        }

        if (!common_sampler_spec_set_draft(smpl.get(), { d }, { q })) {
            LOG_ERR("failed to set the draft\n");
            return 1;
        }

        llama_memory_seq_rm(llama_get_memory(ctx_be.get()), 0, n_prompt - 1, -1);
        batch.clear();
        batch.add(prompt.back(), n_prompt - 1, 0, true);
        batch.add(d,             n_prompt,     0, true);
        if (llama_process(ctx_be.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("failed to decode the verify batch\n");
            return 1;
        }

        // the backend chain returns only the token of each row
        if (llama_get_sampled_token_ith(ctx_be.get(), 0) == LLAMA_TOKEN_NULL || llama_get_sampled_probs_ith(ctx_be.get(), 0) != nullptr) {
            LOG_ERR("the backend did not verify the draft\n");
            return 1;
        }

        const std::vector<llama_token> result = common_sampler_sample_and_accept_n(smpl.get(), ctx_be.get(), { 0, 1 }, llama_tokens{ d });
        counts[result[0]]++;
        n_accepted += result[0] == d;

        // host implementation of the same step
        const float u = (float) uni_be(rng_be);
        const float v = (float) uni_be(rng_be);
        (void) uni_be(rng_be);
        (void) uni_be(rng_be);

        double q_d = 0.0;
        for (const auto & e : q) {
            q_d += e.id == d ? e.p : 0.0;
        }
        llama_token expect;
        if (u * q_d < p_ref[d]) {
            expect = d;
        } else {
            std::vector<double> r(p_ref);
            for (const auto & e : q) {
                r[e.id] = std::max(0.0, p_ref[e.id] - e.p);
            }
            double sum = 0.0;
            for (double x : r) {
                sum += x;
            }
            expect = n_vocab - 1;
            double acc = 0.0;
            for (int i = 0; i < n_vocab; ++i) {
                acc += r[i];
                if (acc > v * sum) {
                    expect = i;
                    break;
                }
            }
        }
        n_same += result[0] == expect;
    }

    LOG("\nBackend verification after %d trials: %d accepted, %d of %d equal to the host implementation\n",
            n_trials, n_accepted, n_same, n_trials);

    bool ok = true;
    for (int i = 0; i < n_vocab; ++i) {
        const double p_e = (double) counts[i] / n_trials;
        const double tol = 6.0 * std::sqrt(p_ref[i] * (1.0 - p_ref[i]) / n_trials) + 1e-9;
        if (std::fabs(p_e - p_ref[i]) > tol) {
            LOG_ERR("token %d: empirical probability %f too far from target %f (tol %f)\n", i, p_e, p_ref[i], tol);
            ok = false;
        }
    }
    // the host sums in another order, so a draw near a boundary can pick a neighbour
    if (n_same < n_trials - n_trials / 200) {
        LOG_ERR("only %d of %d trials equal the host implementation\n", n_same, n_trials);
        ok = false;
    }

    return ok ? 0 : 1;
}

struct parsed_args {
    int n_trials;
    std::vector<char *> argv;
};

static std::optional<parsed_args> parse_n_trials_arg(int argc, char ** argv, int default_n_trials) {
    parsed_args result;
    result.n_trials = default_n_trials;
    result.argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--n-trials") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --n-trials requires a number argument\n", __func__);
                return std::nullopt;
            }
            int value = std::atoi(argv[i + 1]);
            if (value <= 0) {
                LOG_ERR("%s: --n-trials requires a positive number argument\n", __func__);
                return std::nullopt;
            }
            result.n_trials = value;
            i++;
        } else {
            result.argv.push_back(argv[i]);
        }
    }
    result.argv.push_back(nullptr);

    return result;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_ctx = 256;

    common_init();

    auto parsed = parse_n_trials_arg(argc, argv, 5000);
    if (!parsed) {
        return 1;
    }
    const int fargc = (int) parsed->argv.size() - 1;

    if (!common_params_parse(fargc, parsed->argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    LOG("Using target model '%s'\n", params.model.path.c_str());

    llama_backend_init();

    int result = test_speculative_rejection_sampling(params, parsed->n_trials);
    if (result == 0) {
        result = test_speculative_rejection_backend(params, parsed->n_trials, 0);
    }
    if (result == 0) {
        result = test_speculative_rejection_backend(params, parsed->n_trials, 5);
    }
    if (result == 0) {
        LOG("\nAll tests passed.\n");
    } else {
        LOG("\nSome tests failed.\n");
    }

    llama_backend_free();
    return result;
}
