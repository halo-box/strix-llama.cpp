// Checks of GGML_OP_SPARSE_ATTN_PLAN / GGML_OP_SPARSE_ATTN and of the QSA graph pieces, one section per ctest:
//   neginf-v      a -inf cell's non-finite V element, the CPU op against the host reference and every GPU device
//   shape-guard   supports_op rejects nodes that break ggml_sparse_attn's shape invariants (an RPC server's nodes)
//   rpc-view      the SPARSE_ATTN alloc size of the node an RPC server rebuilds equals the linked node's
//   gate-scratch  the output placement of the HIP gated SPARSE_ATTN fusion (host only)
//   prefer-dense  every dense pick of the QSA shortcut policy is a FLASH_ATTN_EXT the device takes
//   allvis-rows   the all-visible rows of the QSA shortcut against the KQ mask, on every backend
// usage: test-sparse-attn-checks [section [args...]]   (no section: every section, without arguments)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "sparse-attn-ref.h"
#include "../ggml/src/ggml-cuda/sparse-attn-gate-scratch.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <set>
#include <string>
#include <vector>

static int g_fail = 0;

// printf-style message; the first 20 failures of a section are printed
#define CHECK(cond, ...) do { if (!(cond)) { if (g_fail++ < 20) { printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } } while (0)

// the devices of the build a section checks (an ACCEL device such as BLAS has no sparse-attention op)
static std::vector<ggml_backend_dev_t> sa_devices() {
    std::vector<ggml_backend_dev_t> devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            devs.push_back(dev);
        }
    }
    return devs;
}

// ---- neginf-v --------------------------------------------------------------------------------------------------------
// A selected cell whose score is -inf has p = 0 but still enters PV as 0 * v (the WMMA tiers cannot skip a cell per row):
// a non-finite V element makes that dimension NaN, every other dimension is the softmax over the other cells. Only a
// masked cell contributes nothing whatever its bytes. Cell 7 of KV head 0 scores -inf with V = +inf, NaN, -inf in dims
// 5, 9, 200; query 2 selects only -inf cells (l = 0: every dim 0). The CPU op must equal O_CON bitwise; every
// GPU device must match the CPU's NaN dims, and its other dims within 1e-3 of the largest |CPU| value.
// usage: test-sparse-attn-checks neginf-v [--old-text]: --old-text checks "every dim is the softmax over the other cells"
// instead, and must FAIL. Exit status 0 = the checks held.

namespace neginf_v {

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

static int run(int argc, char ** argv) {
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

} // namespace neginf_v

// ---- shape-guard -----------------------------------------------------------------------------------------------------
// An RPC server rebuilds nodes from the wire without checking that a node's shapes agree with its srcs, so supports_op must
// reject a node ggml_sparse_attn(_plan) would never build, or the kernels read past K / V. Every device that takes the
// well-formed node must reject each malformed one. Shapes only (no_alloc, no compute).

namespace shape_guard {

// op_params access (ggml-impl.h is not public)
static int32_t op_i32(const ggml_tensor * t, int i) {
    int32_t v;
    memcpy(&v, (const char *) t->op_params + i * sizeof(int32_t), sizeof(v));
    return v;
}
static void set_op_i32(ggml_tensor * t, int i, int32_t v) {
    memcpy((char *) t->op_params + i * sizeof(int32_t), &v, sizeof(v));
}

// a tensor with the given type and shape, contiguous strides, and the op / op_params of `like` (if any): what
// deserialize_tensor rebuilds (type, ne, nb, op, op_params; no data)
static ggml_tensor * mk(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, const ggml_tensor * like = nullptr) {
    ggml_tensor * t = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
    if (like) {
        t->op = like->op;
        memcpy(t->op_params, like->op_params, sizeof(t->op_params));
    }
    return t;
}

struct shape {
    int64_t n_q = 9, n_kv = 8192, n_sel = 2051, n_head_kv = 2, gqa = 12, n_stream = 1;
    int32_t G = 1;
    ggml_type type = GGML_TYPE_F16;
};

struct nodes {
    ggml_tensor * q, * k, * v, * sel, * plan, * attn;
};

// the well-formed node (built by ggml_sparse_attn), then copied field by field into a server-style node whose shapes
// `mutate` may change
static nodes build(ggml_context * ctx, const shape & s, const std::function<void(nodes &)> & mutate) {
    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, s.n_q, s.n_head_kv * s.gqa, s.n_stream);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx, s.type, 256, 4 * ((s.n_kv + 3) / 4), s.n_head_kv, s.n_stream);
    ggml_tensor * v    = ggml_new_tensor_4d(ctx, s.type, 256, 4 * ((s.n_kv + 3) / 4), s.n_head_kv, s.n_stream);
    ggml_tensor * sel  = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, s.n_sel, s.n_q, 1, s.n_stream);
    ggml_tensor * plan = ggml_sparse_attn_plan(ctx, sel, s.n_kv, s.G);
    ggml_tensor * attn = ggml_sparse_attn(ctx, q, k, v, plan, 0.0625f, 0, 0);
    nodes n = { q, k, v, sel, plan, attn };
    if (!mutate) {
        return n;
    }
    nodes c;
    c.q    = mk(ctx, q->type, q->ne[0], q->ne[1], q->ne[2], q->ne[3]);
    c.k    = mk(ctx, k->type, k->ne[0], k->ne[1], k->ne[2], k->ne[3]);
    c.v    = mk(ctx, v->type, v->ne[0], v->ne[1], v->ne[2], v->ne[3]);
    c.sel  = mk(ctx, sel->type, sel->ne[0], sel->ne[1], sel->ne[2], sel->ne[3]);
    c.plan = mk(ctx, plan->type, plan->ne[0], 1, 1, 1, plan);
    c.attn = mk(ctx, attn->type, attn->ne[0], attn->ne[1], attn->ne[2], attn->ne[3], attn);
    mutate(c);
    c.plan->src[0] = c.sel;
    c.attn->src[0] = c.q;
    c.attn->src[1] = c.k;
    c.attn->src[2] = c.v;
    c.attn->src[3] = c.plan;
    return c;
}

struct mutation {
    const char * name;
    bool plan_only;   // the malformed node is the plan (else the attend)
    std::function<void(ggml_context *, nodes &)> apply;
};

static int run(int, char **) {
    const std::vector<mutation> muts = {
        { "K view of 4 cells, plan n_kv 8192", false, [](ggml_context * ctx, nodes & n) {
            n.k = mk(ctx, n.k->type, 256, 4, n.k->ne[2], n.k->ne[3]); } },
        { "V view of 4 cells, plan n_kv 8192", false, [](ggml_context * ctx, nodes & n) {
            n.v = mk(ctx, n.v->type, 256, 4, n.v->ne[2], n.v->ne[3]); } },
        { "K / V views one quad short of 4*ceil(n_kv/4)", false, [](ggml_context * ctx, nodes & n) {
            n.k = mk(ctx, n.k->type, 256, n.k->ne[1] - 4, n.k->ne[2], n.k->ne[3]);
            n.v = mk(ctx, n.v->type, 256, n.v->ne[1] - 4, n.v->ne[2], n.v->ne[3]); } },
        { "plan n_kv 2^24 over 8192-cell K / V (plan sized for it)", false, [](ggml_context * ctx, nodes & n) {
            const int32_t G = op_i32(n.plan, 1);
            ggml_tensor * p = mk(ctx, GGML_TYPE_I8, (int64_t) ggml_sparse_attn_plan_nbytes(2051, 9, 1, (int64_t) 1 << 24, G), 1, 1, 1, n.plan);
            set_op_i32(p, 0, 1 << 24);
            n.plan = p; } },
        { "V with 1 head, K with 2", false, [](ggml_context * ctx, nodes & n) {
            n.v = mk(ctx, n.v->type, 256, n.v->ne[1], 1, n.v->ne[3]); } },
        { "V with 8 heads, K with 2 (V heads past K)", false, [](ggml_context * ctx, nodes & n) {
            n.k = mk(ctx, n.k->type, 256, n.k->ne[1], 2, n.k->ne[3]);
            n.v = mk(ctx, n.v->type, 256, n.v->ne[1], 8, n.v->ne[3]); } },
        { "K / V with 1 stream, q with 2", false, [](ggml_context * ctx, nodes & n) {
            // q, dst and the plan all name 2 streams: only K / V disagree
            const int32_t G = op_i32(n.plan, 1);
            n.q    = mk(ctx, GGML_TYPE_F32, 256, n.q->ne[1], n.q->ne[2], 2);
            n.attn = mk(ctx, GGML_TYPE_F32, 256, n.attn->ne[1], n.attn->ne[2], 2, n.attn);
            ggml_tensor * p = mk(ctx, GGML_TYPE_I8, (int64_t) ggml_sparse_attn_plan_nbytes(2051, 9, 2, 8192, G), 1, 1, 1, n.plan);
            set_op_i32(p, 4, 2);
            n.plan = p; } },
        { "plan op_params n_q 3, q n_q 9", false, [](ggml_context *, nodes & n) {
            set_op_i32(n.plan, 3, 3); } },
        { "plan smaller than its layout", false, [](ggml_context * ctx, nodes & n) {
            n.plan = mk(ctx, GGML_TYPE_I8, n.plan->ne[0] - 256, 1, 1, 1, n.plan); } },
        { "dst with 1 query, q with 9", false, [](ggml_context * ctx, nodes & n) {
            n.attn = mk(ctx, GGML_TYPE_F32, 256, n.attn->ne[1], 1, 1, n.attn); } },
        { "plan: sel n_sel 257, op_params 2051", true, [](ggml_context * ctx, nodes & n) {
            n.sel = mk(ctx, GGML_TYPE_I32, 257, n.sel->ne[1], 1, n.sel->ne[3]); } },
        { "plan: dst smaller than its layout", true, [](ggml_context * ctx, nodes & n) {
            n.plan = mk(ctx, GGML_TYPE_I8, n.plan->ne[0] - 256, 1, 1, 1, n.plan); } },
        { "plan: op_params n_kv 2^24, dst sized for 8192", true, [](ggml_context *, nodes & n) {
            set_op_i32(n.plan, 0, 1 << 24); } },
    };

    int n_checked = 0;
    for (ggml_backend_dev_t dev : sa_devices()) {
        printf("device %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        for (ggml_type t : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
            for (int32_t G : { 1, 4 }) {
                shape s;
                s.type = t;
                s.G = G;
                ggml_init_params ip = { 256 * ggml_tensor_overhead(), nullptr, true };
                ggml_context * ctx = ggml_init(ip);
                const nodes ok = build(ctx, s, nullptr);
                const bool ok_attn = ggml_backend_dev_supports_op(dev, ok.attn);
                const bool ok_plan = ggml_backend_dev_supports_op(dev, ok.plan);
                // the field-by-field copy itself is accepted wherever the built node is
                const nodes cp = build(ctx, s, [](nodes &) {});
                const bool cp_attn = ggml_backend_dev_supports_op(dev, cp.attn);
                const bool cp_plan = ggml_backend_dev_supports_op(dev, cp.plan);
                printf("  %-4s G %d: well-formed attend %s, plan %s (copy: %s, %s)\n", ggml_type_name(t), G,
                       ok_attn ? "supported" : "unsupported", ok_plan ? "supported" : "unsupported",
                       cp_attn ? "supported" : "unsupported", cp_plan ? "supported" : "unsupported");
                CHECK(cp_attn == ok_attn && cp_plan == ok_plan, "the copy of the well-formed node is not answered like the node");
                for (const mutation & m : muts) {
                    const bool base = m.plan_only ? ok_plan : ok_attn;
                    if (!base) {
                        continue;   // nothing to reject: the device does not take the well-formed node
                    }
                    const nodes bad = build(ctx, s, [&](nodes & n) { m.apply(ctx, n); });
                    const bool sup = ggml_backend_dev_supports_op(dev, m.plan_only ? bad.plan : bad.attn);
                    ++n_checked;
                    CHECK(!sup, "%s: supported (the kernels would read past the tensors)", m.name);
                }
                ggml_free(ctx);
            }
        }
    }
    printf("%d malformed nodes checked, %d failed: %s\n", n_checked, g_fail, g_fail == 0 && n_checked > 0 ? "OK" : "FAILED");
    return g_fail == 0 && n_checked > 0 ? 0 : 1;
}

} // namespace shape_guard

// ---- rpc-view --------------------------------------------------------------------------------------------------------
// An RPC server deserializes a node and its direct srcs only, so the plan (src[3]) arrives with src[0] == NULL. The
// SPARSE_ATTN alloc size must not depend on that missing source: on every device, for every shape it takes, the server's
// view must get the size of the fully linked node. Without a GPU device the CPU rows pass trivially.

namespace rpc_view {

// what deserialize_tensor rebuilds: type, ne, nb, op, op_params and flags; no src, no view, no data
static ggml_tensor * rpc_copy(ggml_context * ctx, const ggml_tensor * t) {
    ggml_tensor * c = ggml_new_tensor(ctx, t->type, GGML_MAX_DIMS, t->ne);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        c->nb[i] = t->nb[i];
    }
    c->op = t->op;
    memcpy(c->op_params, t->op_params, sizeof(c->op_params));
    c->flags = t->flags;
    return c;
}

// rpc_server::get_alloc_size: the node, then its direct srcs linked to it (and nothing below them)
static ggml_tensor * rpc_server_view(ggml_context * ctx, const ggml_tensor * t) {
    ggml_tensor * c = rpc_copy(ctx, t);
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (t->src[i]) {
            c->src[i] = rpc_copy(ctx, t->src[i]);
        }
    }
    return c;
}

static int run(int, char **) {
    int n_rows = 0;
    for (ggml_backend_dev_t dev : sa_devices()) {
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        printf("device %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        for (ggml_type t : { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 }) {
            for (int32_t G : { 1, 4 }) {
                for (int64_t n_q : { 3, 9, 512 }) {
                    ggml_init_params ip = { 64 * ggml_tensor_overhead(), nullptr, true };
                    ggml_context * ctx = ggml_init(ip);
                    const int64_t n_kv = 8192, n_sel = 2051, kv_head = 2, gqa = 12;
                    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, n_q, kv_head * gqa, 1);
                    ggml_tensor * k    = ggml_new_tensor_4d(ctx, t, 256, n_kv, kv_head, 1);
                    ggml_tensor * v    = ggml_new_tensor_4d(ctx, t, 256, n_kv, kv_head, 1);
                    ggml_tensor * sel  = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, n_sel, n_q, 1, 1);
                    ggml_tensor * plan = ggml_sparse_attn_plan(ctx, sel, n_kv, G);
                    ggml_tensor * attn = ggml_sparse_attn(ctx, q, k, v, plan, 0.0625f, 0, 0);
                    if (!ggml_backend_dev_supports_op(dev, attn) || !ggml_backend_dev_supports_op(dev, plan)) {
                        ggml_free(ctx);
                        continue;
                    }
                    const size_t a_linked = ggml_backend_buft_get_alloc_size(buft, attn);
                    const size_t p_linked = ggml_backend_buft_get_alloc_size(buft, plan);
                    ggml_tensor * attn_v = rpc_server_view(ctx, attn);
                    ggml_tensor * plan_v = rpc_server_view(ctx, plan);
                    const size_t a_view = ggml_backend_buft_get_alloc_size(buft, attn_v);
                    const size_t p_view = ggml_backend_buft_get_alloc_size(buft, plan_v);
                    printf("  %-4s G %d n_q %-3" PRId64 ": attend nbytes %zu, alloc linked %zu, server view %zu; plan alloc linked %zu, view %zu\n",
                           ggml_type_name(t), G, n_q, ggml_nbytes(attn), a_linked, a_view, p_linked, p_view);
                    CHECK(a_view == a_linked, "%s: attend alloc size of the server's view %zu != linked %zu", ggml_backend_dev_name(dev), a_view, a_linked);
                    CHECK(p_view == p_linked, "%s: plan alloc size of the server's view %zu != linked %zu", ggml_backend_dev_name(dev), p_view, p_linked);
                    n_rows++;
                    ggml_free(ctx);
                }
            }
        }
    }
    printf("%d rows, %d failed checks: %s\n", n_rows, g_fail, g_fail == 0 ? "OK" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}

} // namespace rpc_view

// ---- gate-scratch ----------------------------------------------------------------------------------------------------
// The output placement of the HIP gated SPARSE_ATTN fusion (sparse-attn-gate-scratch.h) on synthetic layouts of one
// compute buffer. No sparse kernel uses ctx.pool(), so the reserve sees every workspace: when an output lands on a tensor
// the kernel reads, it goes to graph memory the fusion leaves unwritten (dst, the CONT and SIGMOID outputs), or the fusion
// is refused. A scratch lies inside one candidate and overlaps no read, final output or other scratch. CPU only.

namespace gate_scratch {

static char arena[1 << 20];

struct layout {
    const char * name;
    // offsets (bytes) in the arena; n32 bytes per f32 tensor of the attend's shape, n32 / 2 for the 16-bit copy
    size_t q, k, v, plan, gate, ws;    // reads (the workspace right after dst: its offset is given)
    size_t dst, cont, unary;           // candidates
    long   mul, d16;                   // outputs, -1 when not written
    bool   expect_fused;               // a candidate fits
};

static int run(int, char **) {
    const size_t n32 = 4096, n16 = n32 / 2, small = 512;
    // tensor slots of n32 bytes: slot(i) = i * n32
    auto slot = [&](size_t i) { return i * n32; };
    const std::vector<layout> layouts = {
        // the MUL over Q (dead after the attend); no 16-bit copy
        { "mul over q",                slot(0), slot(1), slot(2), slot(3), slot(4), slot(5) + n32, slot(5), slot(7), slot(8), (long) slot(0), -1, true },
        // the MUL over Q and the 16-bit copy over the gate
        { "mul over q, d16 over gate", slot(0), slot(1), slot(2), slot(3), slot(4), slot(5) + n32, slot(5), slot(7), slot(8), (long) slot(0), (long) slot(4), true },
        // the 16-bit copy only (bf16-only reader), over K
        { "d16 over k",                slot(0), slot(1), slot(2), slot(3), slot(4), slot(5) + n32, slot(5), slot(7), slot(8), -1, (long) slot(1), true },
        // the MUL in place over the attend's dst (gallocr's MUL in-place): no hazard, written in place
        { "mul over dst (in place)",   slot(0), slot(1), slot(2), slot(3), slot(4), slot(5) + n32, slot(5), slot(7), slot(8), (long) slot(5), -1, true },
        // the MUL over Q and the SIGMOID in place over the CONT: only dst is left for the f32 output
        { "mul over q, unary on cont", slot(0), slot(1), slot(2), slot(3), slot(4), slot(5) + n32, slot(5), slot(7), slot(7), (long) slot(0), -1, true },
        // both outputs hazardous, the CONT / SIGMOID over Q (freed) as well: dst takes one, nothing is left for the other
        { "no room for both",          slot(0), slot(1), slot(2), slot(3), slot(4), slot(5) + n32, slot(5), slot(0), slot(0), (long) slot(0), (long) slot(4), false },
    };
    for (const layout & L : layouts) {
        const ggml_cuda_sa_span reads[6] = {
            { arena + L.q, n32 }, { arena + L.k, n32 }, { arena + L.v, n32 }, { arena + L.plan, small }, { arena + L.gate, n32 }, { arena + L.ws, small } };
        const ggml_cuda_sa_span cands[3] = { { arena + L.dst, n32 }, { arena + L.cont, n32 }, { arena + L.unary, n32 } };
        const ggml_cuda_sa_span o32 = { L.mul >= 0 ? arena + L.mul : nullptr, L.mul >= 0 ? n32 : 0 };
        const ggml_cuda_sa_span o16 = { L.d16 >= 0 ? arena + L.d16 : nullptr, L.d16 >= 0 ? n16 : 0 };
        const ggml_cuda_sa_gate_plan gp = ggml_cuda_sa_gate_plan_outputs(reads, 6, o32, o16, cands, 3);
        const char * wn[] = { "none", "direct", "pool", "scratch" };
        printf("  %-26s: fused %d, f32 %s, 16-bit %s\n", L.name, gp.ok, wn[gp.where32 & 3], wn[gp.where16 & 3]);
        CHECK(gp.ok == L.expect_fused, "%s: fused %d, expected %d", L.name, gp.ok, L.expect_fused);
        if (!gp.ok) {
            continue;
        }
        CHECK(gp.where32 != GGML_CUDA_SA_GATE_POOL && gp.where16 != GGML_CUDA_SA_GATE_POOL, "%s: an output goes to pool scratch (unreserved, D12)", L.name);
        CHECK((gp.where32 == GGML_CUDA_SA_GATE_NONE) == (o32.n == 0) && (gp.where16 == GGML_CUDA_SA_GATE_NONE) == (o16.n == 0), "%s: an output is dropped", L.name);
        // where each output is finally written by the kernel
        struct w { const void * p; size_t n; bool scratch; };
        std::vector<w> writes;
        for (int o = 0; o < 2; ++o) {
            const int where = o == 0 ? gp.where32 : gp.where16;
            const ggml_cuda_sa_span out = o == 0 ? o32 : o16;
            void * s = o == 0 ? gp.scratch32 : gp.scratch16;
            if (where == GGML_CUDA_SA_GATE_NONE) {
                continue;
            }
            if (where == GGML_CUDA_SA_GATE_DIRECT) {
                CHECK(!ggml_cuda_sa_overlaps_any(reads, 6, out.p, out.n), "%s: output %d written in place over a read", L.name, o);
                writes.push_back({ out.p, out.n, false });
                continue;
            }
            CHECK(s != nullptr, "%s: output %d has no scratch", L.name, o);
            bool inside = false;
            for (const auto & c : cands) {
                inside = inside || ((const char *) s >= (const char *) c.p && (const char *) s + out.n <= (const char *) c.p + c.n);
            }
            CHECK(inside, "%s: the scratch of output %d is not inside a candidate", L.name, o);
            CHECK(!ggml_cuda_sa_overlaps_any(reads, 6, s, out.n), "%s: the scratch of output %d overlaps a read", L.name, o);
            CHECK(!ggml_cuda_sa_overlap(s, out.n, o32.p, o32.n) && !ggml_cuda_sa_overlap(s, out.n, o16.p, o16.n),
                  "%s: the scratch of output %d overlaps a final output", L.name, o);
            writes.push_back({ s, out.n, true });
        }
        for (size_t a = 0; a < writes.size(); ++a) {
            for (size_t b = a + 1; b < writes.size(); ++b) {
                CHECK(!ggml_cuda_sa_overlap(writes[a].p, writes[a].n, writes[b].p, writes[b].n), "%s: two kernel outputs overlap", L.name);
            }
        }
    }
    printf("test-sparse-attn-gate-scratch: %d failed checks: %s\n", g_fail, g_fail == 0 ? "OK" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}

} // namespace gate_scratch

// ---- prefer-dense ----------------------------------------------------------------------------------------------------
// Every (K, V, n_q band) for which ggml_backend_sparse_attn_prefer_dense picks dense must be a FLASH_ATTN_EXT of the
// shortcut's shape that the device takes (e.g. not on a -DGGML_CUDA_FA=OFF build), or the QSA guard fails the context.
// A backend without the policy is skipped. Exit status 0 = every dense pick is runnable.

namespace prefer_dense {

static bool fa_supported(ggml_backend_dev_t dev, ggml_type tk, ggml_type tv, int64_t n_q, int64_t n_kv) {
    ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, n_q, 24, 1);
    ggml_tensor * k  = ggml_new_tensor_4d(ctx, tk, 256, n_kv, 2, 1);
    ggml_tensor * v  = ggml_new_tensor_4d(ctx, tv, 256, n_kv, 2, 1);
    ggml_tensor * m  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q, 1, 1);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, m, 0.0625f, 0.0f, 0.0f);
    const bool ok = ggml_backend_dev_supports_op(dev, fa);
    ggml_free(ctx);
    return ok;
}

static int run(int, char **) {
    int n_fail = 0, n_dev = 0;
    const ggml_type types[] = { GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL };
    for (ggml_backend_dev_t dev : sa_devices()) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto prefer = reg ? (ggml_backend_sparse_attn_prefer_dense_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_sparse_attn_prefer_dense") : nullptr;
        if (prefer == nullptr) {
            printf("%s (%s): no sparse-attention dense policy, skipped\n", ggml_backend_dev_name(dev), ggml_backend_reg_name(reg));
            continue;
        }
        n_dev++;
        int n_dense = 0, n_cells = 0, n_bad = 0;
        for (ggml_type tk : types) {
            for (ggml_type tv : types) {
                for (int64_t n_q : { 1, 8, 32, 127, 128, 511, 1024 }) {
                    for (int64_t n_kv : { 256, 2048 }) {
                        n_cells++;
                        if (!prefer(dev, tk, tv, n_q, n_kv)) {
                            continue;
                        }
                        n_dense++;
                        const bool runnable = fa_supported(dev, tk, tv, n_q, n_kv);
                        n_bad += !runnable;
                        CHECK(runnable, "%s: dense picked for %s/%s n_q %d n_kv %d, but this device's FLASH_ATTN_EXT rejects it",
                              ggml_backend_dev_name(dev), ggml_type_name(tk), ggml_type_name(tv), (int) n_q, (int) n_kv);
                    }
                }
            }
        }
        printf("%s (%s): dense %d of %d cells, %d not runnable; FLASH_ATTN_EXT f16/f16 %s, q8_0/q8_0 %s\n", ggml_backend_dev_name(dev),
               ggml_backend_reg_name(reg), n_dense, n_cells, n_bad, fa_supported(dev, GGML_TYPE_F16, GGML_TYPE_F16, 1, 2048) ? "supported" : "unsupported",
               fa_supported(dev, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 1, 2048) ? "supported" : "unsupported");
        n_fail += n_bad;
    }
    printf("test-qsa-prefer-dense: %s (%d devices with the policy, %d bad dense picks)\n", n_fail == 0 ? "OK" : "FAILED", n_dev, n_fail);
    return n_fail == 0 ? 0 : 1;
}

} // namespace prefer_dense

// ---- allvis-rows -----------------------------------------------------------------------------------------------------
// Rows built from the KQ mask with the ops of build_qsa_rows_av must equal {j : mask = 0} per (query row, stream), with the
// 4-D shape [n_kv, n_tps, 1, n_stream] the plan asserts, on each backend.
// usage: test-sparse-attn-checks allvis-rows [mask.bin n_kv n_q]...   (recorded f16 masks; the synthetic split-stream cases always run)

namespace allvis_rows {

static ggml_tensor * build_rows(ggml_context * ctx, ggml_tensor * m) {
    ggml_tensor * mf  = m->type == GGML_TYPE_F32 ? m : ggml_cast(ctx, m, GGML_TYPE_F32);
    ggml_tensor * vis = ggml_step(ctx, ggml_scale_bias(ctx, mf, 1.0f, 1.0f));
    ggml_tensor * ar  = ggml_scale_bias(ctx, ggml_arange(ctx, 0.0f, (float) m->ne[0], 1.0f), 1.0f, 1.0f);
    ggml_tensor * idf = ggml_scale_bias(ctx, ggml_mul(ctx, vis, ar), 1.0f, -1.0f);
    return ggml_cast(ctx, idf, GGML_TYPE_I32);
}

static void run_case(ggml_backend_t be, const char * bname, const char * cname, const std::vector<ggml_fp16_t> & mask,
                     int64_t n_kv, int64_t n_tps, int64_t n_stream) {
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_tps, 1, n_stream);
    ggml_tensor * rows = build_rows(ctx, m);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, rows);
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (!ggml_backend_supports_op(be, ggml_graph_node(gf, i))) {
            CHECK(false, "[%s %s] unsupported op %s", bname, cname, ggml_op_desc(ggml_graph_node(gf, i)));
            ggml_free(ctx);
            return;
        }
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(m, mask.data(), 0, mask.size()*sizeof(ggml_fp16_t));
    ggml_backend_graph_compute(be, gf);
    std::vector<int32_t> out(ggml_nelements(rows));
    ggml_backend_tensor_get(rows, out.data(), 0, out.size()*sizeof(int32_t));
    // row (q, s) lists j iff mask[s][q][j] == 0, at slot j; -1 elsewhere
    int64_t bad = 0, vis = 0;
    std::set<uint16_t> other;
    for (int64_t s = 0; s < n_stream; ++s) {
        for (int64_t q = 0; q < n_tps; ++q) {
            for (int64_t j = 0; j < n_kv; ++j) {
                const size_t e = (size_t) ((s*n_tps + q)*n_kv + j);
                const float mv = ggml_fp16_to_fp32(mask[e]);
                if (mv != 0.0f && !(std::isinf(mv) && mv < 0)) {
                    other.insert(mask[e]);
                }
                const int32_t want = mv == 0.0f ? (int32_t) j : -1;
                vis += want >= 0;
                bad += out[e] != want;
            }
        }
    }
    const bool shape_ok = rows->ne[0] == n_kv && rows->ne[1] == n_tps && rows->ne[2] == 1 && rows->ne[3] == n_stream;
    printf("  [%s %s] ne=[%lld,%lld,%lld,%lld] mismatch=%lld/%zu visible=%lld other_mask_values=%zu shape=%d\n", bname, cname,
           (long long) rows->ne[0], (long long) rows->ne[1], (long long) rows->ne[2], (long long) rows->ne[3], (long long) bad,
           out.size(), (long long) vis, other.size(), (int) shape_ok);
    CHECK(bad == 0 && shape_ok, "[%s %s] %lld rows differ from the mask, shape ok %d", bname, cname, (long long) bad, (int) shape_ok);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// split-stream causal mask: stream s holds len[s] cells (positions 0..len-1, its own sequence); the ubatch's n_tps queries of
// stream s are its last n_tps positions; cells past len (padding of the n_kv view) are -inf
static std::vector<ggml_fp16_t> synth(int64_t n_kv, int64_t n_tps, const std::vector<int64_t> & len) {
    std::vector<ggml_fp16_t> m(n_kv*n_tps*len.size());
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (size_t s = 0; s < len.size(); ++s) {
        for (int64_t q = 0; q < n_tps; ++q) {
            const int64_t p = len[s] - n_tps + q;
            for (int64_t j = 0; j < n_kv; ++j) {
                m[(s*n_tps + q)*n_kv + j] = j <= p ? z : ninf;
            }
        }
    }
    return m;
}

static int run(int argc, char ** argv) {
    struct be_t { ggml_backend_t be; std::string name; };
    std::vector<be_t> bes;
    for (ggml_backend_dev_t d : sa_devices()) {
        ggml_backend_t b = ggml_backend_dev_init(d, nullptr);
        if (b != nullptr) {
            bes.push_back({ b, ggml_backend_dev_name(d) });
        }
    }
    struct case_t { std::string name; int64_t n_kv, n_tps; std::vector<int64_t> len; };
    const std::vector<case_t> cases = {
        { "np2_ntps1_kv2048",   2048, 1,   { 2048, 1203 } },
        { "np2_ntps128_kv2048", 2048, 128, { 1977, 640 } },
        { "np2_ntps128_kv512",  512,  128, { 512, 128 } },
        { "np2_ntps1_kv256",    256,  1,   { 17, 256 } },
        { "np1_ntps512_kv2048", 2048, 512, { 2048 } },
    };
    for (const auto & b : bes) {
        printf("backend %s\n", b.name.c_str());
        for (const auto & c : cases) {
            run_case(b.be, b.name.c_str(), c.name.c_str(), synth(c.n_kv, c.n_tps, c.len), c.n_kv, c.n_tps, (int64_t) c.len.size());
        }
        for (int a = 1; a + 2 < argc; a += 3) {
            const int64_t n_kv = atoll(argv[a + 1]), n_q = atoll(argv[a + 2]);
            std::vector<ggml_fp16_t> m(n_kv*n_q);
            FILE * f = fopen(argv[a], "rb");
            if (f == nullptr || fread(m.data(), 2, m.size(), f) != m.size()) {
                printf("cannot read %s\n", argv[a]);
                return 2;
            }
            fclose(f);
            run_case(b.be, b.name.c_str(), argv[a], m, n_kv, n_q, 1);
        }
    }
    printf("RESULT %s (failures %d)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

} // namespace allvis_rows

struct section {
    const char * name;
    int (*run)(int argc, char ** argv);
};

static const section sections[] = {
    { "neginf-v",     neginf_v::run },
    { "shape-guard",  shape_guard::run },
    { "rpc-view",     rpc_view::run },
    { "gate-scratch", gate_scratch::run },
    { "prefer-dense", prefer_dense::run },
    { "allvis-rows",  allvis_rows::run },
};

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    int status = 0;
    bool found = false;
    for (const section & s : sections) {
        if (argc > 1 && strcmp(argv[1], s.name) != 0) {
            continue;
        }
        found = true;
        printf("== %s\n", s.name);
        g_fail = 0;
        char * no_args[] = { argv[0], nullptr };
        status |= argc > 1 ? s.run(argc - 1, argv + 1) : s.run(1, no_args);
    }
    if (!found) {
        printf("unknown section '%s'\n", argv[1]);
        return 2;
    }
    return status;
}
