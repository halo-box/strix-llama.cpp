#pragma once

#include "llama.h"
#include "common.h"

#include <algorithm>
#include <cstdlib>

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

// return true if the target and draft models have compatible vocabs
bool common_speculative_are_compatible(const llama_model * model_tgt, const llama_model * model_dft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   pos0;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // candidate distribution per drafted token; set it to make draft-simple and draft-mtp sample
    std::vector<std::vector<llama_token_data>> * result_q = nullptr;

    // the target's temp and seed, read only when the drafter samples probabilistically
    float    temp = 1.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const common_batch & batch);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

// 7a (Gufo #603, hi5/ITEM7-DRAFT-POLICY.md 3a): keep the k-th draft token (k >= 1) of a chain whose kept tokens so far have the
// top-1 probability product `acc`, when the candidate's top-1 probability is `p` and the threshold is `pcum` (0 = off).
// The first draft is always kept; otherwise the token is kept iff acc * p >= pcum. A width-only rule: it never changes a token.
static inline bool common_speculative_mtp_pcum_keep(float pcum, int k, float acc, float p) {
    return pcum <= 0.0f || k <= 1 || acc * p >= pcum;
}

// --spec-draft-cost-aware (MTP): choose each round's draft width w in 0..cap to maximise expected emitted tokens per unit of
// round time, E(w) / T(w).
//   E(w) = 1 + sum_{k=1..w} prod_{d<=k} a_d, where a_d is the conditional acceptance at depth d from decayed counts. Only the
//          first rejected depth is a failure; the unverified tail and a fully accepted chain are censored.
//   T(w) = wall time of a round of width w (draft steps, verify of w + 1 rows, draft catch-up, sampling), learned online. A
//          width with fewer than 2 samples is priced from the nearest measured width with a per-draft slope prior.
// The first round of a request drafts cap (the fixed-width behaviour). w = 0 is a plain target step with no draft step; after
// choosing it the policy stays on plain steps for ar_hold rounds, then drafts one token to look again. A width change must
// beat the current width's score by `margin`, since each change rebuilds the verify graph. Width only: greedy verify emits
// the same tokens whatever is drafted. Acceptance resets per request (reset()); the round-time table persists.
struct common_speculative_mtp_cost_policy {
    static constexpr int   W_MAX   = 16;
    static constexpr float decay   = 0.90f; // per observation of a depth (~10 observations of memory)
    static constexpr float prior_s = 4.0f;  // prior a_d = 0.8, worth 5 observations
    static constexpr float prior_f = 1.0f;
    static constexpr float relax   = 0.95f; // per round, an undrafted depth drifts back to the prior
    static constexpr float slope   = 0.30f; // prior extra round time per draft token, as a fraction of a plain step
    static constexpr float t_alpha = 0.25f; // EMA weight of a new round-time sample
    static constexpr float t_out   = 3.0f;  // a sample above t_out x the estimate is an outlier (idle gap, rebuild)
    static constexpr float margin  = 1.03f;
    static constexpr int   ar_hold = 16;

    int   w_max = 3;
    float succ[W_MAX];
    float fail[W_MAX];
    int   n_obs   = 0;
    int   ar_left = 0;
    bool  probe   = false;
    int   w_cur   = -1;

    float t_ms[W_MAX + 1];
    int   t_n [W_MAX + 1];

    explicit common_speculative_mtp_cost_policy(int w = 3) { init(w); }

    void init(int w) {
        w_max = w < 1 ? 1 : (w > W_MAX ? W_MAX : w);
        for (int i = 0; i <= W_MAX; ++i) { t_ms[i] = 0.0f; t_n[i] = 0; }
        reset();
    }

    void reset() {
        for (int d = 0; d < W_MAX; ++d) { succ[d] = prior_s; fail[d] = prior_f; }
        n_obs = 0; ar_left = 0; probe = false; w_cur = -1;
    }

    float acc(int d) const { return succ[d] / (succ[d] + fail[d]); }

    float expected(int w) const {
        float e = 1.0f, s = 1.0f;
        for (int d = 0; d < w && d < W_MAX; ++d) { s *= acc(d); e += s; }
        return e;
    }

    // round time of width w: ms when any width is measured, else relative to a plain step
    float cost(int w) const {
        if (t_n[w] >= 2) {
            return t_ms[w];
        }
        int m = -1;
        for (int k = 0; k <= w_max; ++k) {
            if (t_n[k] >= 2 && (m < 0 || std::abs(k - w) < std::abs(m - w))) {
                m = k;
            }
        }
        return m < 0 ? 1.0f + slope * w : t_ms[m] * (1.0f + slope * w) / (1.0f + slope * m);
    }

    // score of every width 0..cap (costs forced non-decreasing in w), returns the best
    int best(int cap, float * score) const {
        float c_prev = 0.0f;
        int   b      = 0;
        for (int w = 0; w <= cap; ++w) {
            const float c = std::max(cost(w), c_prev);
            c_prev   = c;
            score[w] = expected(w) / c;
            if (score[w] > score[b]) {
                b = w;
            }
        }
        return b;
    }

    int choose(int cap) {
        cap = std::min(cap, w_max);
        if (cap <= 0) {
            return 0;
        }
        if (ar_left > 0) {
            if (--ar_left == 0) {
                probe = true;
            }
            return 0;
        }
        if (n_obs == 0) {
            return w_cur = cap;
        }
        float score[W_MAX + 1];
        int b = best(cap, score);
        if (w_cur >= 0 && w_cur <= cap && b != w_cur && score[b] < score[w_cur] * margin) {
            b = w_cur;
        }
        if (probe) {
            probe = false;
            b = std::max(b, 1);
        }
        if (b == 0) {
            ar_left = ar_hold - 1;
            probe   = ar_left == 0;
        }
        return w_cur = b;
    }

    void observe(int drafted, int accepted) {
        if (drafted <= 0) {
            return;
        }
        drafted  = std::min(drafted, W_MAX);
        accepted = std::min(accepted, drafted);
        const int n = accepted + (accepted < drafted ? 1 : 0);
        for (int d = 0; d < n; ++d) {
            succ[d] = decay * succ[d] + (d < accepted ? 1.0f : 0.0f);
            fail[d] = decay * fail[d] + (d == accepted ? 1.0f : 0.0f);
        }
        // depths past the drafted width get no evidence while the width stays below them: let them drift back to the
        // prior, so one bad stretch cannot keep a depth (and the width that needs it) shut for the rest of the request
        for (int d = drafted; d < w_max; ++d) {
            succ[d] = relax * succ[d] + (1.0f - relax) * prior_s;
            fail[d] = relax * fail[d] + (1.0f - relax) * prior_f;
        }
        n_obs++;
    }

    // a round of width w took ms; `changed` = its width differs from the round before (graph rebuild, not counted)
    void observe_time(int w, float ms, bool changed) {
        if (w < 0 || w > W_MAX || changed || !(ms > 0.0f)) {
            return;
        }
        if (t_n[w] == 0) {
            t_ms[w] = ms;
        } else if (ms <= t_out * t_ms[w]) {
            t_ms[w] = (1.0f - t_alpha) * t_ms[w] + t_alpha * ms;
        } else {
            return;
        }
        t_n[w]++;
    }
};

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
