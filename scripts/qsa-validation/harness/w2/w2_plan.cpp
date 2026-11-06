// W2 (PLAN-v3 §13 W2): the C3 plan-stage harness, run on the GGML_OP_SPARSE_ATTN_PLAN op of the rewrite.
//
// Reads a C3 rows file (c3_capture: per QSA layer a record {char name[64]; int32 n_q, ns, nk, has_mask, pos0, kt;}
// followed by n_q * ns int32 cells), optionally shuffles every row (C3's control: std::mt19937(seed + layer)), and for
// each layer and each G:
//   - timing: one graph of R independent plan nodes over the same sel on the GPU backend, computed reps times after
//     3 warm-up computes; per rep, wall time of ggml_backend_graph_compute (which synchronizes) / R = the cost of one
//     plan op including its launches and the gaps between them. Reported per layer: median and mean over the reps.
//   - bytes: the whole dst of the first and the last GPU node vs the CPU backend's dst and vs the C3 oracle
//     (tests/sparse-attn-ref.h, std::map per group), memcmp over ggml_nbytes(dst).
// usage: w2_plan rows_file reps R shuffle_seed(0 = none) groups(e.g. 4,1) [backend name, default ROCm0]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "sparse-attn-ref.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct rec { char name[64]; int32_t n_q, ns, nk, has_mask, pos0, kt; };

static std::vector<uint8_t> run_plan(ggml_backend_t be, const std::vector<int32_t> & rows, int n_q, int ns, int nk, int G, int R,
                                     int reps, std::vector<double> * ms_per_op, std::vector<uint8_t> * last) {
    ggml_init_params ip = { ggml_tensor_overhead() * (R + 4) + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * sel = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, ns, n_q, 1, 1);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    std::vector<ggml_tensor *> plans;
    for (int r = 0; r < R; ++r) {
        ggml_tensor * p = ggml_sparse_attn_plan(ctx, sel, nk, G);
        ggml_set_output(p);
        ggml_build_forward_expand(gf, p);
        plans.push_back(p);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "alloc failed\n"); exit(1); }
    ggml_backend_tensor_set(sel, rows.data(), 0, rows.size() * 4);
    for (int w = 0; w < (ms_per_op ? 3 : 1); ++w) {
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
    }
    if (ms_per_op) {
        for (int i = 0; i < reps; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
            const auto t1 = std::chrono::steady_clock::now();
            ms_per_op->push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / R);
        }
    }
    std::vector<uint8_t> first(ggml_nbytes(plans[0]));
    ggml_backend_tensor_get(plans[0], first.data(), 0, first.size());
    if (last) {
        last->resize(ggml_nbytes(plans[R - 1]));
        ggml_backend_tensor_get(plans[R - 1], last->data(), 0, last->size());
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return first;
}

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: w2_plan rows_file reps R shuffle_seed groups [backend]\n"); return 1; }
    const int reps = atoi(argv[2]), R = atoi(argv[3]), seed = atoi(argv[4]);
    std::vector<int> groups;
    for (const char * p = argv[5]; *p; ) { groups.push_back(atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
    const char * bname = argc > 6 ? argv[6] : "ROCm0";
    ggml_backend_load_all();
    ggml_backend_t gpu = ggml_backend_init_by_name(bname, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!gpu || !cpu) { fprintf(stderr, "backend init failed\n"); return 1; }
    FILE * f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 1; }
    printf("W2PLAN file=%s reps=%d R=%d shuffle=%d backend=%s graphs=%s plan_global=%s\n", argv[1], reps, R, seed, bname,
           getenv("GGML_CUDA_DISABLE_GRAPHS") ? "off" : "on", getenv("GGML_CUDA_SA_PLAN_GLOBAL") ? getenv("GGML_CUDA_SA_PLAN_GLOBAL") : "unset");
    rec h;
    int layer = 0;
    int n_bad_cmp = 0;
    while (fread(&h, sizeof(h), 1, f) == 1) {
        std::vector<int32_t> rows((size_t) h.n_q * h.ns);
        if (fread(rows.data(), 4, rows.size(), f) != rows.size()) { fprintf(stderr, "short read\n"); return 1; }
        if (seed) {
            std::mt19937 rng(seed + layer);
            for (int q = 0; q < h.n_q; ++q) std::shuffle(rows.begin() + (size_t) q * h.ns, rows.begin() + (size_t) (q + 1) * h.ns, rng);
        }
        for (int G : groups) {
            std::vector<double> ms;
            std::vector<uint8_t> last;
            const std::vector<uint8_t> g0  = run_plan(gpu, rows, h.n_q, h.ns, h.nk, G, R, reps, &ms, &last);
            const std::vector<uint8_t> c0  = run_plan(cpu, rows, h.n_q, h.ns, h.nk, G, 1, 0, nullptr, nullptr);
            const std::vector<uint8_t> ref = sa_ref::plan(rows, h.ns, h.n_q, 1, h.nk, G);
            const bool eq_cpu  = g0 == c0 && last == c0;
            const bool eq_ref  = g0 == ref;
            n_bad_cmp += !eq_cpu || !eq_ref;
            std::vector<double> s = ms;
            std::sort(s.begin(), s.end());
            double mean = 0; for (double x : ms) mean += x; mean /= ms.size();
            const ggml_sparse_attn_plan_hdr * hd = (const ggml_sparse_attn_plan_hdr *) g0.data();
            const uint32_t * counts = (const uint32_t *) (g0.data() + hd->off_counts);
            double ent = 0; for (uint32_t g = 0; g < hd->n_groups; ++g) ent += counts[g];
            printf("LAYER %d name=%s pos0=%d nq=%d ns=%d nk=%d G=%d nbytes=%zu cap=%u n_bad=%u mean_entries=%.1f "
                   "ms_med=%.4f ms_mean=%.4f ms_min=%.4f ms_max=%.4f bytes_eq_cpu=%d bytes_eq_c3oracle=%d\n",
                   layer, h.name, h.pos0, h.n_q, h.ns, h.nk, G, g0.size(), hd->cap, hd->n_bad, ent / hd->n_groups,
                   s[s.size() / 2], mean, s.front(), s.back(), eq_cpu, eq_ref);
            fflush(stdout);
        }
        ++layer;
    }
    printf("W2PLAN_DONE layers=%d mismatches=%d\n", layer, n_bad_cmp);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return n_bad_cmp != 0;
}
