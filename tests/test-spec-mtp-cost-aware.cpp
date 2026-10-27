// --spec-draft-cost-aware: the MTP draft-width policy (common_speculative_mtp_cost_policy). Pure rule, CPU only.
// A simulated drafter accepts each depth with a fixed probability; a round of width w takes t0 * (1 + 0.3 w) ms. The policy
// must start at the cap, settle on the width that maximises tokens per ms, and drop to plain steps when drafting cannot pay.
#include "speculative.h"

#include <cstdio>
#include <random>

static int fails = 0;
static void check(bool ok, const char * what) {
    if (!ok) { fprintf(stderr, "FAIL %s\n", what); fails++; }
}

struct sim_result {
    int    first;     // width of the first round
    int    hist[8];   // rounds per width after the first 10
    double tok_per_ms;
};

// p: per-depth acceptance, cap: n_max, t0: plain step ms, rounds: rounds to run
static sim_result run(float p, int cap, float t0, int rounds, unsigned seed) {
    common_speculative_mtp_cost_policy pol(cap);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    sim_result r = {};
    double tok = 0.0, ms = 0.0;
    int w_prev = -1, w_prev2 = -1;
    for (int i = 0; i < rounds; ++i) {
        const int w = pol.choose(cap);
        if (i == 0) {
            r.first = w;
        }
        int acc = 0;
        while (acc < w && u(rng) < p) {
            acc++;
        }
        const float t = t0 * (1.0f + 0.3f * w);
        pol.observe(w, acc);
        w_prev2 = w_prev; w_prev = w;
        pol.observe_time(w, t, w_prev != w_prev2);
        tok += 1 + acc;
        ms  += t;
        if (i >= 10) {
            r.hist[w]++;
        }
    }
    r.tok_per_ms = tok / ms;
    return r;
}

static int mode(const sim_result & r) {
    int m = 0;
    for (int w = 1; w < 8; ++w) if (r.hist[w] > r.hist[m]) m = w;
    return m;
}

// expected tokens per ms of a fixed width w at acceptance p
static double fixed_rate(float p, int w, float t0) {
    double e = 1.0, s = 1.0;
    for (int d = 0; d < w; ++d) { s *= p; e += s; }
    return e / (t0 * (1.0 + 0.3 * w));
}

int main() {
    // the first round drafts the cap (the fixed-width behaviour) before anything is observed
    {
        common_speculative_mtp_cost_policy pol(3);
        check(pol.choose(3) == 3, "first round = cap");
        check(pol.choose(2) <= 2, "never above the cap");
    }
    // a cap of 0 (no room left) is a plain step
    {
        common_speculative_mtp_cost_policy pol(3);
        check(pol.choose(0) == 0, "cap 0 -> 0");
    }
    // censoring: a fully accepted chain records no failure; the first reject is the only failure
    {
        common_speculative_mtp_cost_policy pol(4);
        const float a0 = pol.acc(0), a2 = pol.acc(2);
        pol.observe(4, 4);
        check(pol.acc(0) > a0 && pol.acc(3) > a0, "full accept raises every depth");
        pol.observe(4, 1);
        check(pol.acc(1) < pol.acc(0), "first reject at depth 1 lowers depth 1 only");
        check(pol.acc(2) > a2, "depths past the reject keep their earlier evidence");
    }
    // round-time learning: a width change is not a sample; outliers are dropped
    {
        common_speculative_mtp_cost_policy pol(3);
        pol.observe_time(1, 50.0f, true);
        check(pol.t_n[1] == 0, "changed-width round not timed");
        pol.observe_time(1, 50.0f, false);
        pol.observe_time(1, 50.0f, false);
        pol.observe_time(1, 500.0f, false);
        check(pol.t_n[1] == 2 && pol.t_ms[1] == 50.0f, "outlier dropped");
        check(pol.cost(3) > pol.cost(1) && pol.cost(0) < pol.cost(1), "unmeasured widths priced from the measured one");
    }

    const float t0 = 38.0f;
    // min_rel: the first round drafts the cap and a plain-step stretch probes one draft every ar_hold rounds, so at p 0.15
    // the policy pays some rounds a plain step would not
    struct { float p; int want; double min_rel; const char * what; } cases[] = {
        { 0.84f, 3, 0.97, "p 0.84 (predictable text): width 3" },
        { 0.55f, 1, 0.95, "p 0.55 (prose at depth): width 1" },
        { 0.15f, 0, 0.90, "p 0.15: plain steps" },
    };
    for (const auto & c : cases) {
        double r_best = 0.0;
        for (int w = 0; w <= 3; ++w) r_best = std::max(r_best, fixed_rate(c.p, w, t0));
        double r_avg = 0.0;
        const int n_seed = 200;
        int ok_mode = 0;
        for (int s = 0; s < n_seed; ++s) {
            const sim_result r = run(c.p, 3, t0, 64, 1234 + s);
            check(r.first == 3, "first round at the cap");
            ok_mode += mode(r) == c.want;
            r_avg += r.tok_per_ms / n_seed;
        }
        printf("%-40s modal width ok %2d/%d, rate %.4f tok/ms = %.3f of the best fixed width\n",
               c.what, ok_mode, n_seed, r_avg, r_avg / r_best);
        check(ok_mode >= n_seed * 3 / 5, c.what);
        check(r_avg >= c.min_rel * r_best, "rate close to the best fixed width over 64 rounds");
    }

    printf("%s: %d failure(s)\n", fails ? "FAILED" : "OK", fails);
    return fails ? 1 : 0;
}
