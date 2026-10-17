// 7a: the probability-product draft stop (common_speculative_mtp_pcum_keep, --spec-draft-mtp-pcum). Pure rule, CPU only.
// keep(pcum, k, acc, p): off at pcum 0; the first draft (k = 1) always kept; otherwise kept iff acc * p >= pcum.
#include "speculative.h"

#include <cstdio>

static int fails = 0;
static void check(bool got, bool want, const char * what) {
    if (got != want) { fprintf(stderr, "FAIL %s: got %d want %d\n", what, got, want); fails++; }
}

int main() {
    // off
    check(common_speculative_mtp_pcum_keep(0.0f, 3, 0.01f, 0.01f), true,  "off keeps everything");
    // the first draft is always kept, however unlikely
    check(common_speculative_mtp_pcum_keep(0.6f, 1, 1.0f, 0.05f), true,  "first draft kept at p 0.05");
    // the product rule
    check(common_speculative_mtp_pcum_keep(0.6f, 2, 0.9f, 0.7f),  true,  "0.9 * 0.7 = 0.63 >= 0.6 kept");
    check(common_speculative_mtp_pcum_keep(0.6f, 2, 0.9f, 0.6f),  false, "0.9 * 0.6 = 0.54 < 0.6 cut");
    check(common_speculative_mtp_pcum_keep(0.6f, 3, 0.63f, 0.95f), false, "0.63 * 0.95 = 0.5985 < 0.6 cut");
    check(common_speculative_mtp_pcum_keep(0.3f, 3, 0.63f, 0.95f), true,  "the same chain at 0.3 kept");
    // the boundary is inclusive
    check(common_speculative_mtp_pcum_keep(0.5f, 2, 0.5f, 1.0f),  true,  "0.5 * 1.0 = 0.5 >= 0.5 kept");
    // a chain of confident drafts runs to its length
    float acc = 1.0f; int kept = 0;
    for (int k = 1; k <= 8; ++k) { if (!common_speculative_mtp_pcum_keep(0.6f, k, acc, 0.98f)) break; acc *= 0.98f; kept++; }
    check(kept == 8, true, "0.98^8 = 0.85 >= 0.6: all 8 kept");
    acc = 1.0f; kept = 0;
    for (int k = 1; k <= 8; ++k) { if (!common_speculative_mtp_pcum_keep(0.6f, k, acc, 0.8f)) break; acc *= 0.8f; kept++; }
    check(kept == 2, true, "0.8: kept 1 (always), 2 (0.64), cut at 3 (0.512)");
    printf("%s: %d failure(s)\n", fails ? "FAILED" : "OK", fails);
    return fails ? 1 : 0;
}
