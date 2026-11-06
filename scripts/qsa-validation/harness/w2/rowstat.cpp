// W2 diagnostic: per layer of a C3 rows file, how many rows are directly usable (valid cells strictly ascending, every
// invalid entry after them) and where the -1 entries sit
#include <cstdio>
#include <cstdint>
#include <vector>
struct rec { char name[64]; int32_t n_q, ns, nk, has_mask, pos0, kt; };
int main(int, char ** argv) {
    FILE * f = fopen(argv[1], "rb"); rec h; int L = 0;
    while (fread(&h, sizeof h, 1, f) == 1) {
        std::vector<int32_t> r((size_t) h.n_q * h.ns); if (fread(r.data(), 4, r.size(), f) != r.size()) return 1;
        long raw = 0, tail_only = 0, first_hole_sum = 0;
        for (int q = 0; q < h.n_q; ++q) {
            const int32_t * x = &r[(size_t) q * h.ns]; bool ok = true; int prev = -1; bool seen_inv = false; int fh = -1;
            for (int j = 0; j < h.ns; ++j) { bool v = x[j] >= 0 && x[j] < h.nk; if (!v) { if (fh < 0) fh = j; seen_inv = true; continue; }
                if (seen_inv || x[j] <= prev) ok = false; prev = x[j]; }
            raw += ok; first_hole_sum += fh;
        }
        printf("layer %d raw_usable=%ld/%d mean_first_invalid_index=%.1f\n", L++, raw, h.n_q, (double) first_hole_sum / h.n_q);
    }
}
