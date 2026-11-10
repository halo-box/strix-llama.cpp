// supports_op of GGML_OP_SPARSE_ATTN / GGML_OP_SPARSE_ATTN_PLAN on nodes that ggml_sparse_attn(_plan) never built: an RPC
// server rebuilds each node from the wire (deserialize_tensor checks only that a tensor lies inside its buffer, not that
// the shapes of a node and its srcs agree), so a node whose plan n_kv exceeds its K / V views, whose V has fewer heads
// than K, whose streams or n_q disagree, or whose plan size does not match its op_params reached the kernels, which read
// K / V rows up to 4*ceil(n_kv/4) - 1 and heads up to k->ne[2] - 1 past the tensors. Every device of the build that takes
// the well-formed node must reject each malformed one (ggml_sparse_attn_shape_why / ggml_sparse_attn_plan_shape_why).
// Shapes only (no_alloc, no compute): runs on the CPU, and on HIP / Vulkan where the build has them.
#include "ggml.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

static int g_fail = 0;

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

int main() {
    ggml_backend_load_all();

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
    for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
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
                if (cp_attn != ok_attn || cp_plan != ok_plan) {
                    printf("    FAIL: the copy of the well-formed node is not answered like the node\n");
                    ++g_fail;
                }
                for (const mutation & m : muts) {
                    const bool base = m.plan_only ? ok_plan : ok_attn;
                    if (!base) {
                        continue;   // nothing to reject: the device does not take the well-formed node
                    }
                    const nodes bad = build(ctx, s, [&](nodes & n) { m.apply(ctx, n); });
                    const bool sup = ggml_backend_dev_supports_op(dev, m.plan_only ? bad.plan : bad.attn);
                    ++n_checked;
                    if (sup) {
                        printf("    FAIL: %s: supported (the kernels would read past the tensors)\n", m.name);
                        ++g_fail;
                    }
                }
                ggml_free(ctx);
            }
        }
    }
    printf("%d malformed nodes checked, %d failed: %s\n", n_checked, g_fail, g_fail == 0 && n_checked > 0 ? "OK" : "FAILED");
    return g_fail == 0 && n_checked > 0 ? 0 : 1;
}
