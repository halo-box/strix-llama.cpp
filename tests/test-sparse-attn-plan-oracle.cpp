// GGML_OP_SPARSE_ATTN_PLAN on the CPU against the C3 oracle (sparse-attn-ref.h: C3's std::map cpu_plan, generalised to
// G in {1, 4} and n_stream, in the ggml.h dst layout), memcmp over the whole dst.
//
//   test-sparse-attn-plan-oracle                          synthetic rows (ctest), plus the over-allocation check
//   test-sparse-attn-plan-oracle [-t N] [--shuffle S] FILE...
//                                                         captured rows: records {char name[64]; int32 n_q, ns, nk,
//                                                         has_mask, pos0, kt;} + n_q * ns int32 (the format of C3's
//                                                         c3_capture and C5's c5_capture), one record per QSA layer
//
// For captured rows it prints, per layer, the statistics C3's c3_plan prints (valid cells per query, ascending rows,
// mean padded G4 union "union_asc"), so a recapture can be matched against C3's logs, and the memcmp verdict for G = 1
// and G = 4 on the rows as captured and (--shuffle S) with every row shuffled by std::mt19937(S + layer) as c3_plan does.
//
// The over-allocation check: a buffer type whose get_alloc_size over-allocates every SPARSE_ATTN_PLAN / SPARSE_ATTN
// node allocates a plan + attend graph through ggml-alloc (ggml_backend_buft_get_alloc_size asserts that an op which
// over-allocates is listed in ggml_op_alloc_size_may_expand, ggml-backend.cpp), and its result must equal the one of
// the plain CPU buffer type.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "sparse-attn-ref.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

static int g_threads = 8;

// the CPU op's dst bytes for these rows (rows[((s * n_q) + i) * n_sel + j])
static std::vector<uint8_t> cpu_plan(ggml_backend_t be, const std::vector<int32_t> & rows, int64_t n_sel, int64_t n_q,
                                     int64_t n_stream, int64_t n_kv, int32_t G) {
    ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * sel  = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, n_sel, n_q, 1, n_stream);
    ggml_tensor * plan = ggml_sparse_attn_plan(ctx, sel, n_kv, G);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, plan);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    GGML_ASSERT(buf);
    ggml_backend_tensor_set(sel, rows.data(), 0, ggml_nbytes(sel));
    // poison the dst: every byte of it must be written by the op
    ggml_backend_tensor_memset(plan, 0xA5, 0, ggml_nbytes(plan));
    GGML_ASSERT(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS);
    std::vector<uint8_t> out(ggml_nbytes(plan));
    ggml_backend_tensor_get(plan, out.data(), 0, out.size());
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

static bool check(ggml_backend_t be, const std::vector<int32_t> & rows, int64_t n_sel, int64_t n_q, int64_t n_stream,
                  int64_t n_kv, int32_t G, const char * label, double * mean_count = nullptr) {
    const std::vector<uint8_t> ref = sa_ref::plan(rows, n_sel, n_q, n_stream, n_kv, G);
    const std::vector<uint8_t> got = cpu_plan(be, rows, n_sel, n_q, n_stream, n_kv, G);
    bool ok = got.size() == ref.size() && memcmp(got.data(), ref.data(), ref.size()) == 0;
    size_t first = 0;
    if (!ok && got.size() == ref.size()) {
        while (got[first] == ref[first]) ++first;
    }
    const ggml_sparse_attn_plan_layout L = ggml_sparse_attn_plan_get_layout(n_sel, n_q, n_stream, n_kv, G);
    const uint32_t * counts = (const uint32_t *) (ref.data() + L.off_counts);
    double mc = 0.0;
    for (int64_t e = 0; e < n_stream * L.n_groups; ++e) mc += counts[e];
    mc /= (double) (n_stream * L.n_groups);
    if (mean_count) *mean_count = mc;
    const ggml_sparse_attn_plan_hdr * h = (const ggml_sparse_attn_plan_hdr *) ref.data();
    printf("PLAN_ORACLE %s G=%d n_q=%" PRId64 " n_sel=%" PRId64 " n_kv=%" PRId64 " n_stream=%" PRId64 " bytes=%zu n_bad=%u mean_count=%.1f %s",
           label, G, n_q, n_sel, n_kv, n_stream, ref.size(), h->n_bad, mc, ok ? "IDENTICAL" : "DIFFERENT");
    if (!ok) printf(" first_diff_byte=%zu", first);
    printf("\n");
    fflush(stdout);
    return ok;
}

// synthetic rows: permuted, duplicates, -1, out-of-range values, all -1, cells across quads
static std::vector<int32_t> synth_rows(int kind, int64_t n_sel, int64_t n_q, int64_t n_stream, int64_t n_kv, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<int32_t> rows((size_t) (n_sel * n_q * n_stream), -1);
    const int32_t oob[4] = { (int32_t) n_kv, -2, INT32_MIN, INT32_MAX };
    for (size_t r = 0; r < (size_t) (n_q * n_stream); ++r) {
        int32_t * row = rows.data() + r * n_sel;
        for (int64_t j = 0; j < n_sel; ++j) {
            const uint32_t u = rng() % 16;
            switch (kind) {
                case 0: row[j] = (int32_t) (rng() % n_kv); break;                                             // random, dups
                case 1: row[j] = u < 12 ? (int32_t) (rng() % n_kv) : u < 14 ? -1 : oob[rng() % 4]; break;       // + -1, oob
                case 2: row[j] = -1; break;                                                                     // all -1
                case 3: row[j] = (int32_t) ((j * 4 + (r % 4)) % n_kv); break;                                   // one cell per quad
                case 4: row[j] = (int32_t) ((r * 131 + j) % n_kv); break;                                       // ascending runs
            }
        }
        if (kind == 4 && r % 2) std::reverse(row, row + n_sel);                                               // descending
    }
    return rows;
}

// ---- over-allocation check ----

static ggml_backend_buffer_type g_over_buft;

static size_t over_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * t) {
    GGML_UNUSED(buft);
    size_t n = ggml_nbytes(t);
    if (t->op == GGML_OP_SPARSE_ATTN_PLAN) {
        const ggml_tensor * sel = t->src[0];
        const int32_t n_kv = t->op_params[0];
        // as a GPU backend: the max over G of the plan bytes, plus the rows scratch
        n = std::max(ggml_sparse_attn_plan_nbytes(sel->ne[0], sel->ne[1], sel->ne[3], n_kv, 1),
                     ggml_sparse_attn_plan_nbytes(sel->ne[0], sel->ne[1], sel->ne[3], n_kv, 4)) +
            ggml_sparse_attn_plan_scratch_nbytes(sel->ne[0], sel->ne[1], sel->ne[3]);
    } else if (t->op == GGML_OP_SPARSE_ATTN) {
        n += (size_t) 1 << 20;   // a split-K / V-pack workspace
    }
    return n;
}

static bool check_over_alloc(ggml_backend_t be) {
    ggml_backend_buffer_type_t cpu = ggml_backend_cpu_buffer_type();
    g_over_buft = *cpu;
    g_over_buft.iface.get_alloc_size = over_get_alloc_size;

    const int64_t D = 256, n_q = 9, n_kv = 2051, n_sel = 257, H = 2, gqa = 12;
    std::vector<float> out[2];
    size_t alloc[2] = {0, 0};
    for (int pass = 0; pass < 2; ++pass) {
        ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * q   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, n_q, H * gqa, 1);
        ggml_tensor * k   = ggml_new_tensor_4d(ctx, GGML_TYPE_Q8_0, D, 2052, H, 1);
        ggml_tensor * v   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, 2052, H, 1);
        ggml_tensor * sel = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, n_sel, n_q, 1, 1);
        ggml_set_input(q); ggml_set_input(k); ggml_set_input(v); ggml_set_input(sel);
        ggml_tensor * plan = ggml_sparse_attn_plan(ctx, sel, n_kv, 4);
        ggml_tensor * o    = ggml_sparse_attn(ctx, q, k, v, plan, 0.0625f, 64, 64);
        ggml_set_output(o);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);
        ggml_gallocr_t ga = ggml_gallocr_new(pass == 0 ? cpu : &g_over_buft);
        GGML_ASSERT(ggml_gallocr_alloc_graph(ga, gf));
        alloc[pass] = ggml_gallocr_get_buffer_size(ga, 0);
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> qf(ggml_nelements(q));
        for (auto & x : qf) x = dist(rng);
        ggml_backend_tensor_set(q, qf.data(), 0, ggml_nbytes(q));
        std::vector<float> kf(ggml_nelements(k));
        for (auto & x : kf) x = dist(rng);
        std::vector<uint8_t> kq(ggml_nbytes(k));
        ggml_quantize_chunk(GGML_TYPE_Q8_0, kf.data(), kq.data(), 0, ggml_nrows(k), D, nullptr);
        ggml_backend_tensor_set(k, kq.data(), 0, kq.size());
        std::vector<ggml_fp16_t> vh(ggml_nelements(v));
        for (auto & x : vh) x = ggml_fp32_to_fp16(dist(rng));
        ggml_backend_tensor_set(v, vh.data(), 0, ggml_nbytes(v));
        std::vector<int32_t> rows = synth_rows(1, n_sel, n_q, 1, n_kv, 3);
        ggml_backend_tensor_set(sel, rows.data(), 0, ggml_nbytes(sel));
        GGML_ASSERT(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS);
        out[pass].resize(ggml_nelements(o));
        ggml_backend_tensor_get(o, out[pass].data(), 0, ggml_nbytes(o));
        ggml_gallocr_free(ga);
        ggml_free(ctx);
    }
    const bool ok = alloc[1] > alloc[0] && memcmp(out[0].data(), out[1].data(), out[0].size() * sizeof(float)) == 0;
    printf("OVER_ALLOC plain_buffer=%zu over_allocating_buffer=%zu outputs %s -> %s\n", alloc[0], alloc[1],
           memcmp(out[0].data(), out[1].data(), out[0].size() * sizeof(float)) == 0 ? "IDENTICAL" : "DIFFERENT", ok ? "OK" : "FAIL");
    return ok;
}

struct rec { char name[64]; int32_t n_q, ns, nk, has_mask, pos0, kt; };

int main(int argc, char ** argv) {
    std::vector<std::string> files;
    int shuffle = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) {
            g_threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--shuffle") && i + 1 < argc) {
            shuffle = atoi(argv[++i]);
        } else {
            files.push_back(argv[i]);
        }
    }
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, g_threads);
    int n_fail = 0, n_ok = 0;

    if (files.empty()) {
        // synthetic suite: n_kv edge values, G 1 / 4, 1 and 2 streams, n_q not a multiple of 4
        for (int kind = 0; kind < 5; ++kind) {
            for (int64_t n_kv : {4, 5, 2051, 4096, 270336, 1 << 24}) {
                for (int64_t n_q : {1, 3, 129}) {
                    for (int64_t n_sel : {1, 257, 2051}) {
                        for (int64_t ns : {1, 2}) {
                            for (int G : {1, 4}) {
                                const auto rows = synth_rows(kind, n_sel, n_q, ns, n_kv, (uint32_t) (kind * 1000 + n_q + n_sel));
                                char lab[64]; snprintf(lab, sizeof(lab), "synthetic kind=%d", kind);
                                (check(be, rows, n_sel, n_q, ns, n_kv, G, lab) ? n_ok : n_fail)++;
                            }
                        }
                    }
                }
            }
        }
        (check_over_alloc(be) ? n_ok : n_fail)++;
    }

    for (const std::string & fn : files) {
        FILE * f = fopen(fn.c_str(), "rb");
        if (!f) { perror(fn.c_str()); return 1; }
        rec h;
        int layer = 0;
        while (fread(&h, sizeof(h), 1, f) == 1) {
            std::vector<int32_t> rows((size_t) h.n_q * h.ns);
            if (fread(rows.data(), 4, rows.size(), f) != rows.size()) { fprintf(stderr, "short read\n"); return 1; }
            // the statistics c3_plan prints for a layer (c3_plan.cu:318-341)
            long long nvalid = 0; int nsorted = 0;
            for (int q = 0; q < h.n_q; ++q) {
                int prev = -1; bool asc = true;
                for (int j = 0; j < h.ns; ++j) { const int k = rows[(size_t) q * h.ns + j]; if (k >= 0 && k < h.nk) { ++nvalid; if (k <= prev) asc = false; prev = k; } }
                nsorted += asc;
            }
            for (int sh = 0; sh <= (shuffle ? 1 : 0); ++sh) {
                std::vector<int32_t> r = rows;
                if (sh) {
                    std::mt19937 rng(shuffle + layer);
                    for (int q = 0; q < h.n_q; ++q) std::shuffle(r.begin() + (size_t) q * h.ns, r.begin() + (size_t) (q + 1) * h.ns, rng);
                }
                char lab[512];
                double mc4 = 0.0;
                for (int G : {4, 1}) {
                    snprintf(lab, sizeof(lab), "file=%s layer=%d name=%s pos0=%d shuffle=%d", fn.c_str(), layer, h.name, h.pos0, sh ? shuffle : 0);
                    (check(be, r, h.ns, h.n_q, 1, h.nk, G, lab, G == 4 ? &mc4 : nullptr) ? n_ok : n_fail)++;
                }
                printf("LAYER %d name=%s pos0=%d nq=%d ns=%d nk=%d valid_per_q=%.1f sorted_rows=%d/%d union_asc=%.1f shuffle=%d\n",
                       layer, h.name, h.pos0, h.n_q, h.ns, h.nk, (double) nvalid / h.n_q, nsorted, h.n_q, mc4, sh ? shuffle : 0);
            }
            ++layer;
        }
        fclose(f);
    }

    printf("SUMMARY identical=%d different=%d\n", n_ok, n_fail);
    ggml_backend_free(be);
    return n_fail == 0 ? 0 : 1;
}
