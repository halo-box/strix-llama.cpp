// ggml.h item 3 of the ggml_sparse_attn contract, for a cell whose score is -inf and whose V row is not finite. Such a cell
// has p = 0, adds nothing to l, and enters PV as 0 * v like every attended cell: 0 * inf and 0 * NaN are NaN, so that
// dimension of the row is NaN, while every other dimension is the softmax over the row's other cells. Only a masked cell
// (item 4) contributes nothing whatever its bytes. The CPU op (the reference of every backend), O_CON
// (tests/sparse-attn-ref.h) and, by code reading, the GPU tiers (HIP G4 WMMA + fixup, G1 WMMA, portable; Vulkan cm, scalar,
// portable) all compute 0 * v there; the WMMA tiers cannot skip a cell per row inside a shared PV product. The text before
// this test said "the row is the softmax over the other cells" for every such cell, which no backend did for a non-finite V.
//
// One CPU graph (plan + attend, f16 K / V, D 256, 2 KV heads x 4, 3 queries, 40 selected cells of 64). In KV head 0 cell 7
// has K element 0 = -inf (f16) and every query's element 0 > 0, so it scores -inf for every head of that KV head; its V row
// holds +inf at dim 5, NaN at dim 9 and -inf at dim 200. Query 2 selects cell 7 alone (with another -inf cell, 11), so its
// rows have l = 0. Checks, KV head 0:
//   queries 0, 1: dims 5, 9, 200 NaN; every other dim bit-equal to O_CON over the selection without cell 7;
//   query 2: every dim 0 (item 5, l = 0);
// KV head 1 (cell 7 finite there): bit-equal to O_CON. The CPU op equals O_CON everywhere, NaN-aware.
// Every GPU device of the build then runs the same graph: its NaN dims must be the CPU op's, and its other dims within
// 1e-3 of the largest |CPU| value (the GPU tiers are not bitwise; their bitwise checks are test-backend-ops SPARSE_ATTN).
// usage: test-sparse-attn-neginf-v [--old-text]: --old-text checks the statement this test replaced (every dim the softmax
// over the other cells) and must FAIL; without it the current statement is checked. Exit status 0 = the checks held.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "sparse-attn-ref.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { if (g_fail++ < 20) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } } while (0)

static sa_ref::view to_view(const ggml_tensor * t) {
    sa_ref::view v;
    v.type = t->type;
    for (int i = 0; i < 4; ++i) { v.ne[i] = t->ne[i]; v.nb[i] = t->nb[i]; }
    v.bytes.assign((const uint8_t *) t->data, (const uint8_t *) t->data + ggml_nbytes(t));
    return v;
}

static bool same_bits(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) {
        return std::isnan(a) && std::isnan(b);
    }
    uint32_t x, y;
    memcpy(&x, &a, 4);
    memcpy(&y, &b, 4);
    return x == y;
}

int main(int argc, char ** argv) {
    const bool old_text = argc > 1 && std::string(argv[1]) == "--old-text";
    const int64_t D = 256, n_head_kv = 2, r = 4, n_head = n_head_kv * r, n_q = 3, n_kv = 64, n_sel = 40;
    const float scale = 0.0625f;
    const int32_t bad = 7, bad2 = 11;
    const int64_t dims_bad[] = { 5, 9, 200 };

    ggml_init_params ip = { 64u << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, n_q, n_head, 1);
    ggml_tensor * k   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, n_kv, n_head_kv, 1);
    ggml_tensor * v   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, n_kv, n_head_kv, 1);
    ggml_tensor * sel = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, n_sel, n_q, 1, 1);

    // q > 0 in dim 0 for every query and head (so K element 0 = -inf gives a -inf score), small values elsewhere
    uint32_t seed = 1;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return ((seed >> 8) & 0xffff) / 65536.0f - 0.5f; };
    float * qd = (float *) q->data;
    for (int64_t i = 0; i < ggml_nelements(q); ++i) { qd[i] = rnd(); }
    for (int64_t h = 0; h < n_head; ++h) {
        for (int64_t i = 0; i < n_q; ++i) { qd[h * n_q * D + i * D] = 0.25f + 0.5f * std::fabs(rnd()); }
    }
    ggml_fp16_t * kd = (ggml_fp16_t *) k->data, * vd = (ggml_fp16_t *) v->data;
    for (int64_t i = 0; i < ggml_nelements(k); ++i) { kd[i] = ggml_fp32_to_fp16(rnd()); vd[i] = ggml_fp32_to_fp16(rnd()); }
    // KV head 0: cells 7 and 11 score -inf; cell 7's V row is not finite in three dims
    for (int32_t c : { bad, bad2 }) { kd[c * D + 0] = ggml_fp32_to_fp16(-INFINITY); }
    vd[bad * D + dims_bad[0]] = ggml_fp32_to_fp16(INFINITY);
    vd[bad * D + dims_bad[1]] = ggml_fp32_to_fp16(NAN);
    vd[bad * D + dims_bad[2]] = ggml_fp32_to_fp16(-INFINITY);

    // queries 0, 1: 40 cells including 7 (cell 7 at different places of the ascending order); query 2: cells 7 and 11 only
    std::vector<int32_t> rows((size_t) (n_q * n_sel), -1);
    for (int64_t j = 0; j < n_sel; ++j) { rows[(size_t) j] = (int32_t) j; }               // 0..39, 7 early
    for (int64_t j = 0; j < n_sel; ++j) { rows[(size_t) (n_sel + j)] = (int32_t) (63 - j); } // 63..24: 7 not in it, add it
    rows[(size_t) (n_sel + n_sel - 1)] = bad;
    rows[(size_t) (2 * n_sel + 0)] = bad;
    rows[(size_t) (2 * n_sel + 1)] = bad2;
    memcpy(sel->data, rows.data(), rows.size() * sizeof(int32_t));

    ggml_tensor * plan = ggml_sparse_attn_plan(ctx, sel, n_kv, 1);
    ggml_tensor * out  = ggml_sparse_attn(ctx, q, k, v, plan, scale, 0, 0);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (ggml_graph_compute_with_ctx(ctx, gf, 4) != GGML_STATUS_SUCCESS) {
        printf("compute failed\n");
        return 1;
    }
    const float * got = (const float *) out->data;   // [D, n_head, n_q, 1]

    const sa_ref::view qv = to_view(q), kv = to_view(k), vv = to_view(v);
    const std::vector<float> ocon = sa_ref::attend(qv, kv, vv, rows, n_sel, n_kv, scale, 0, 0, 4);
    // O_CON over the selection without cell 7 (and without 11 for query 2)
    std::vector<int32_t> rows_wo = rows;
    for (auto & c : rows_wo) { if (c == bad || c == bad2) { c = -1; } }
    const std::vector<float> ocon_wo = sa_ref::attend(qv, kv, vv, rows_wo, n_sel, n_kv, scale, 0, 0, 4);

    auto at = [&](const float * o, int64_t i, int64_t h, int64_t d) { return o[(i * n_head + h) * D + d]; };
    int64_t n_nan = 0, n_cmp = 0;
    for (int64_t i = 0; i < n_q; ++i) {
        for (int64_t h = 0; h < n_head; ++h) {
            for (int64_t d = 0; d < D; ++d) {
                const float g = at(got, i, h, d);
                CHECK(same_bits(g, at(ocon.data(), i, h, d)), "q %d h %d d %d: CPU op %g != O_CON %g", (int) i, (int) h, (int) d, g, at(ocon.data(), i, h, d));
                const bool kv0 = h < r;
                const bool dim_bad = d == dims_bad[0] || d == dims_bad[1] || d == dims_bad[2];
                if (kv0 && i == 2) {
                    CHECK(g == 0.0f, "q 2 h %d d %d: %g, want 0 (every cell -inf: l = 0, item 5)", (int) h, (int) d, g);
                } else if (kv0 && dim_bad && !old_text) {
                    CHECK(std::isnan(g), "q %d h %d d %d: %g, want NaN (0 * a non-finite V element of the -inf cell)", (int) i, (int) h, (int) d, g);
                    n_nan += std::isnan(g);
                } else {
                    // the softmax over the other cells (KV head 1: the same as O_CON, cell 7 scores finite there)
                    const float want = kv0 ? at(ocon_wo.data(), i, h, d) : at(ocon.data(), i, h, d);
                    CHECK(same_bits(g, want), "q %d h %d d %d: %g, want the softmax over the other cells %g", (int) i, (int) h, (int) d, g, want);
                    n_cmp++;
                }
            }
        }
    }
    // the GPU devices of the build on the same inputs
    ggml_backend_load_all();
    float amax = 0.0f;
    for (int64_t e = 0; e < ggml_nelements(out); ++e) { if (std::isfinite(got[e])) { amax = std::max(amax, std::fabs(got[e])); } }
    for (size_t di = 0; di < ggml_backend_dev_count(); ++di) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(di);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        ggml_init_params gp = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * g = ggml_init(gp);
        ggml_tensor * gq = ggml_dup_tensor(g, q), * gk = ggml_dup_tensor(g, k), * gv = ggml_dup_tensor(g, v), * gs = ggml_dup_tensor(g, sel);
        ggml_tensor * gpl = ggml_sparse_attn_plan(g, gs, n_kv, 1);
        ggml_tensor * go  = ggml_sparse_attn(g, gq, gk, gv, gpl, scale, 0, 0);
        if (!ggml_backend_dev_supports_op(dev, gpl) || !ggml_backend_dev_supports_op(dev, go)) {
            printf("  %s: SPARSE_ATTN not supported, skipped\n", ggml_backend_dev_name(dev));
            ggml_free(g);
            continue;
        }
        ggml_cgraph * ggf = ggml_new_graph(g);
        ggml_build_forward_expand(ggf, go);
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(g, be);   // every tensor, with the backends' alloc sizes
        GGML_ASSERT(buf != nullptr);
        ggml_backend_tensor_set(gq, q->data, 0, ggml_nbytes(q));
        ggml_backend_tensor_set(gk, k->data, 0, ggml_nbytes(k));
        ggml_backend_tensor_set(gv, v->data, 0, ggml_nbytes(v));
        ggml_backend_tensor_set(gs, sel->data, 0, ggml_nbytes(sel));
        GGML_ASSERT(ggml_backend_graph_compute(be, ggf) == GGML_STATUS_SUCCESS);
        std::vector<float> res(ggml_nelements(go));
        ggml_backend_tensor_get(go, res.data(), 0, res.size() * sizeof(float));
        int64_t n_nan_g = 0, n_off = 0;
        float worst = 0.0f;
        for (size_t e = 0; e < res.size(); ++e) {
            const bool nan_c = std::isnan(got[e]), nan_g = std::isnan(res[e]);
            n_nan_g += nan_g;
            if (nan_c != nan_g) {
                n_off++;
            } else if (!nan_c) {
                worst = std::max(worst, std::fabs(res[e] - got[e]));
            }
        }
        printf("  %s: %lld NaN dims (CPU %lld), %lld NaN positions differ, max |GPU - CPU| %.3g (bound %.3g)\n", ggml_backend_dev_name(dev),
               (long long) n_nan_g, (long long) (n_nan + 0), (long long) n_off, worst, 1e-3f * amax);
        if (!old_text) {
            CHECK(n_off == 0, "%s: the NaN dims differ from the CPU op's in %lld places", ggml_backend_dev_name(dev), (long long) n_off);
            CHECK(worst <= 1e-3f * amax, "%s: a finite dim differs from the CPU op by %g", ggml_backend_dev_name(dev), worst);
        }
        ggml_backend_buffer_free(buf);
        ggml_backend_free(be);
        ggml_free(g);
    }

    printf("test-sparse-attn-neginf-v (%s statement): %lld NaN dims as stated, %lld dims compared with the softmax over the other cells\n",
           old_text ? "old" : "current", (long long) n_nan, (long long) n_cmp);
    printf("%s (%d failed checks)\n", g_fail == 0 ? "OK" : "FAILED", g_fail);
    ggml_free(ctx);
    return g_fail == 0 ? 0 : 1;
}
