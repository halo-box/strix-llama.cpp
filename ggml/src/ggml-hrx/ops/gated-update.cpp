#include "ggml-hrx-ops.h"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <string>

// Gated update of x with m, as nodes:
//   n0 = RMS_NORM(x), n2 = RMS_NORM(m), n3 = CONCAT(n0, n2, 0), z = MUL_MAT(wd, n3),
//   f = MUL_MAT(wf, z), TANH, MUL(., x), SUB(x, .), g = MUL_MAT(wg, z), TANH, MUL(., m), y = ADD(., .)
// The range starts at z. n0, n2 and n3 are absorbed. An operand that is the CONT of a view is absorbed too: the kernels
// read the view. kernels/fused/gated_update_f32.loom has the math.

static constexpr int64_t GGML_HRX_GATED_RANK  = 64;
static constexpr int64_t GGML_HRX_GATED_CHUNK = 256;
// Up to this many tokens (the column limit of the vector matmul kernels) the decode tiles run and the apply kernel adds the
// partial sums itself. More tokens use the z kernel.
static constexpr int64_t GGML_HRX_GATED_PARTS_MAX_TOKENS = 8;
// From this many tokens the matrix-op kernels run (kernels/fused/gated_update_mma.loom). They round x, m, the weights and z
// to f16 for the products. gfx1151, Inkling 2 Mini, E 2304, 4096 tokens (model graph, device timestamps): down 835 + z 35 +
// apply 665 us against 953 + 95 + 956 us.
static constexpr int64_t GGML_HRX_GATED_MMA_MIN_TOKENS = 64;
// The absorbed nodes are at most this many nodes before the first node of the range.
static constexpr int     GGML_HRX_GATED_MAX_DIST = 256;

struct ggml_hrx_gated_operand {
    const ggml_tensor * t  = nullptr; // what the kernels read: the operand, or the view of its absorbed CONT
    int64_t             s1 = 0;       // row stride in elements
};

struct ggml_hrx_gated_match {
    const ggml_tensor *    wd = nullptr;
    const ggml_tensor *    wf = nullptr;
    const ggml_tensor *    wg = nullptr;
    const ggml_tensor *    y  = nullptr;
    ggml_hrx_gated_operand x;
    ggml_hrx_gated_operand m;
    float                  eps_x = 0.0f;
    float                  eps_m = 0.0f;
    int64_t                e     = 0;
    int64_t                n_t   = 0;
    int                    last  = -1;
};

static bool ggml_hrx_gated_f32_rows(const ggml_tensor * t, int64_t e, int64_t n_t) {
    return t->type == GGML_TYPE_F32 && t->ne[0] == e && t->ne[1] == n_t && t->ne[2] == 1 && t->ne[3] == 1 &&
           t->nb[0] == sizeof(float) && t->nb[1] % sizeof(float) == 0 && (n_t == 1 || t->nb[1] >= (size_t) e * sizeof(float));
}

static bool ggml_hrx_gated_weight(const ggml_tensor * w, int64_t ne0, int64_t ne1) {
    return w != nullptr && w->type == GGML_TYPE_F32 && w->ne[0] == ne0 && w->ne[1] == ne1 && w->ne[2] == 1 && w->ne[3] == 1 &&
           ggml_is_contiguous(w);
}

static bool ggml_hrx_gated_unary(const ggml_tensor * t, ggml_unary_op op) {
    return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op;
}

// The next node that runs after k must be op with the sources a and b (nullptr: any).
static const ggml_tensor * ggml_hrx_gated_next(const ggml_cgraph * g, int & k, ggml_op op, const ggml_tensor * a, const ggml_tensor * b) {
    k = ggml_hrx_next_node(g, k);
    if (k < 0) {
        return nullptr;
    }
    const ggml_tensor * t = g->nodes[k];
    if (t->op != op || t->type != GGML_TYPE_F32 || (a != nullptr && t->src[0] != a) || (b != nullptr && t->src[1] != b)) {
        return nullptr;
    }
    return t;
}

// A node between first and last that runs (not absorbed by this fusion) and writes memory of t.
static bool ggml_hrx_gated_clobbered(const ggml_cgraph * g, int first, int last, const ggml_tensor * t, const std::vector<int> & absorbed) {
    for (int k = first + 1; k < last; ++k) {
        const ggml_tensor * n = g->nodes[k];
        if (ggml_op_is_empty(n->op) || ggml_is_empty(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        bool own = false;
        for (int a : absorbed) {
            own = own || a == k;
        }
        if (!own && ggml_hrx_overlap(n, t)) {
            return true;
        }
    }
    return false;
}

// The operand of the norm and the update. n_uses: its reads by the matched nodes.
static bool ggml_hrx_gated_operand_init(const ggml_cgraph * g, int anchor, const ggml_tensor * t, int n_uses, const ggml_tensor * y,
                                        std::vector<int> & absorbed, ggml_hrx_gated_operand & op, int64_t e, int64_t n_t) {
    const ggml_tensor * v = t->src[0];
    if (t->op == GGML_OP_CONT && v != nullptr && v->type == GGML_TYPE_F32 && ggml_hrx_only_used_by(g, t, n_uses) &&
        ggml_hrx_gated_f32_rows(v, e, n_t) && !ggml_hrx_overlap(y, v)) {
        // The view source must stay unchanged until the fused kernels run.
        const int k = ggml_hrx_node_index(g, t, anchor - 1, GGML_HRX_GATED_MAX_DIST);
        if (k >= 0 && !ggml_hrx_gated_clobbered(g, k, anchor, v, absorbed)) {
            absorbed.push_back(k);
            op.t  = v;
            op.s1 = (int64_t) (v->nb[1] / sizeof(float));
            return true;
        }
    }
    if (!ggml_hrx_gated_f32_rows(t, e, n_t)) {
        return false;
    }
    // The update writes y in the same kernel. y can be the operand itself, else they must not share memory.
    if (ggml_hrx_overlap(y, t) && (y->data != t->data || y->nb[1] != t->nb[1])) {
        return false;
    }
    op.t  = t;
    op.s1 = (int64_t) (t->nb[1] / sizeof(float));
    return true;
}

static bool ggml_hrx_gated_match_at(const ggml_cgraph * g, int i, std::vector<int> & absorbed, ggml_hrx_gated_match & mt) {
    const ggml_tensor * z = g->nodes[i];
    if (z->op != GGML_OP_MUL_MAT || z->type != GGML_TYPE_F32 || z->ne[0] != GGML_HRX_GATED_RANK || z->ne[2] != 1 || z->ne[3] != 1 ||
        !ggml_hrx_only_used_by(g, z, 2)) {
        return false;
    }
    const ggml_tensor * cat = z->src[1];
    if (cat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat, 0) != 0 || !ggml_hrx_only_used_by(g, cat, 1)) {
        return false;
    }
    const ggml_tensor * nx = cat->src[0];
    const ggml_tensor * nm = cat->src[1];
    if (nx->op != GGML_OP_RMS_NORM || nm->op != GGML_OP_RMS_NORM || !ggml_hrx_only_used_by(g, nx, 1) || !ggml_hrx_only_used_by(g, nm, 1)) {
        return false;
    }
    const ggml_tensor * x = nx->src[0];
    const ggml_tensor * m = nm->src[0];
    const int64_t       e = x->ne[0];
    const int64_t     n_t = z->ne[1];
    if (e % GGML_HRX_GATED_CHUNK != 0 || e > 16777216 || !ggml_hrx_gated_f32_rows(nx, e, n_t) || !ggml_hrx_gated_f32_rows(nm, e, n_t) ||
        !ggml_hrx_gated_weight(z->src[0], 2 * e, GGML_HRX_GATED_RANK) || n_t >= INT_MAX / 256) {
        return false;
    }

    // f = TANH(MUL_MAT(wf, z)), SUB(x, MUL(f, x)), g = TANH(MUL_MAT(wg, z)), ADD(., MUL(g, m)).
    int k = i;
    const ggml_tensor * mf = ggml_hrx_gated_next(g, k, GGML_OP_MUL_MAT, nullptr, z);
    if (mf == nullptr || !ggml_hrx_gated_weight(mf->src[0], GGML_HRX_GATED_RANK, e)) {
        return false;
    }
    const ggml_tensor * tf = ggml_hrx_gated_next(g, k, GGML_OP_UNARY, mf, nullptr);
    const ggml_tensor * fx = tf != nullptr && ggml_hrx_gated_unary(tf, GGML_UNARY_OP_TANH) ? ggml_hrx_gated_next(g, k, GGML_OP_MUL, tf, x) : nullptr;
    const ggml_tensor * d  = fx != nullptr ? ggml_hrx_gated_next(g, k, GGML_OP_SUB, x, fx) : nullptr;
    const ggml_tensor * mg = d != nullptr ? ggml_hrx_gated_next(g, k, GGML_OP_MUL_MAT, nullptr, z) : nullptr;
    if (mg == nullptr || !ggml_hrx_gated_weight(mg->src[0], GGML_HRX_GATED_RANK, e)) {
        return false;
    }
    const ggml_tensor * tg = ggml_hrx_gated_next(g, k, GGML_OP_UNARY, mg, nullptr);
    const ggml_tensor * gm = tg != nullptr && ggml_hrx_gated_unary(tg, GGML_UNARY_OP_TANH) ? ggml_hrx_gated_next(g, k, GGML_OP_MUL, tg, m) : nullptr;
    const ggml_tensor * y  = gm != nullptr ? ggml_hrx_gated_next(g, k, GGML_OP_ADD, d, gm) : nullptr;
    if (y == nullptr || !ggml_is_contiguous(y) || !ggml_hrx_gated_f32_rows(y, e, n_t)) {
        return false;
    }
    for (const ggml_tensor * t : { mf, tf, fx, d, mg, tg, gm }) {
        if (!ggml_hrx_gated_f32_rows(t, e, n_t) || !ggml_hrx_only_used_by(g, t, 1)) {
            return false;
        }
    }

    for (const ggml_tensor * t : { nx, nm, cat }) {
        const int idx = ggml_hrx_node_index(g, t, i - 1, GGML_HRX_GATED_MAX_DIST);
        if (idx < 0) {
            return false;
        }
        absorbed.push_back(idx);
    }
    // x is read by n0, MUL and SUB, m by n2 and MUL.
    if (!ggml_hrx_gated_operand_init(g, i, x, 3, y, absorbed, mt.x, e, n_t) || !ggml_hrx_gated_operand_init(g, i, m, 2, y, absorbed, mt.m, e, n_t)) {
        return false;
    }
    mt.wd    = z->src[0];
    mt.wf    = mf->src[0];
    mt.wg    = mg->src[0];
    mt.y     = y;
    mt.eps_x = ggml_get_op_params_f32(nx, 0);
    mt.eps_m = ggml_get_op_params_f32(nm, 0);
    mt.e     = e;
    mt.n_t   = n_t;
    mt.last  = k;
    return true;
}

static int64_t ggml_hrx_gated_bits(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

static bool ggml_hrx_gated_use_mma(const ggml_backend_hrx_context * ctx, const ggml_hrx_gated_match & mt) {
    const ggml_hrx_device_context * dev = ctx->device;
    // The down kernel loads 4 values at a time.
    auto rows16 = [](const ggml_tensor * t) { return (uintptr_t) t->data % 16 == 0 && t->nb[1] % 16 == 0; };
    return mt.n_t >= GGML_HRX_GATED_MMA_MIN_TOKENS && dev->arch->matrix_op == GGML_HRX_MATRIX_OP_WMMA_GFX11 &&
           dev->arch->wave32 && mt.e % 64 == 0 && rows16(mt.x.t) && rows16(mt.m.t) && rows16(mt.y);
}

static bool ggml_hrx_gated_run_mma(ggml_backend_hrx_context * ctx, const ggml_hrx_gated_match & mt) {
    const int64_t e   = mt.e;
    const int64_t n_t = mt.n_t;
    // Input values of one down workgroup: the largest of these that divides E. Fewer chunks give fewer partial sums.
    // Neighbour workgroups read the chunks of the same tokens. gfx1151, E 2304, 4096 tokens, x and m rows 27648 and 36864 bytes
    // apart (test-backend-ops perf, device time): 776-932 us with 576 values, 904 with 384, 1328 with 256, 1003 with 1152,
    // 1245 us for one workgroup over all 2 E values of 64 tokens.
    int64_t kc = 64;
    for (int64_t c : { 576, 512, 448, 384, 320, 256, 192, 128 }) {
        if (e % c == 0) {
            kc = c;
            break;
        }
    }
    const int64_t n_chunks = 2 * e / kc;
    const int64_t sq_off   = n_t * n_chunks * GGML_HRX_GATED_RANK;
    const int64_t z_off    = sq_off + n_t * n_chunks;
    const int64_t n_part   = z_off + n_t * GGML_HRX_GATED_RANK;
    if (n_part >= INT_MAX) {
        return false;
    }
    const ggml_tensor * part = ggml_hrx_scratch(ctx, (size_t) n_part * sizeof(float));
    if (part == nullptr) {
        return false;
    }
    ggml_hrx_kernel_desc ddesc;
    ddesc.file   = "fused/gated_update_mma.loom";
    ddesc.symbol = "ggml_gated_update_down_mma";
    if (!ggml_hrx_dispatch(ctx, ggml_hrx_get_kernel(ctx->device, ddesc), { n_chunks, n_t },
            { e, n_t, mt.x.s1, mt.m.s1, kc, sq_off }, { mt.x.t, mt.m.t, mt.wd, part })) {
        return false;
    }
    ggml_hrx_kernel_desc zdesc;
    zdesc.file   = "fused/gated_update_f32.loom";
    zdesc.symbol = "ggml_gated_update_z";
    zdesc.config = { { "ggml.gated.half_chunks", std::to_string(n_chunks / 2) } };
    if (!ggml_hrx_dispatch(ctx, ggml_hrx_get_kernel(ctx->device, zdesc), { n_t },
            { e, sq_off, z_off, ggml_hrx_gated_bits((float) e), ggml_hrx_gated_bits(mt.eps_x), ggml_hrx_gated_bits(mt.eps_m) }, { part })) {
        return false;
    }
    ggml_hrx_kernel_desc adesc;
    adesc.file   = "fused/gated_update_mma.loom";
    adesc.symbol = "ggml_gated_update_apply_mma";
    return ggml_hrx_dispatch(ctx, ggml_hrx_get_kernel(ctx->device, adesc), { e, n_t },
        { e, n_t, mt.x.s1, mt.m.s1, (int64_t) (mt.y->nb[1] / sizeof(float)), z_off },
        { mt.x.t, mt.m.t, mt.wf, mt.wg, part, mt.y });
}

static bool ggml_hrx_gated_inkling_shape(const ggml_backend_hrx_context * ctx, const ggml_hrx_gated_match & mt) {
    return strcmp(ctx->device->arch->name, "gfx1151") == 0 && mt.e == 2304 && mt.n_t <= 8192 &&
           mt.x.s1 <= (INT_MAX - mt.e) / mt.n_t && mt.m.s1 <= (INT_MAX - mt.e) / mt.n_t &&
           strstr(mt.wd->name, ".loop_") != nullptr && strstr(mt.wd->name, "_ctrl_down.weight") != nullptr;
}

static bool ggml_hrx_gated_run_packed(ggml_backend_hrx_context * ctx, const ggml_hrx_gated_match & mt) {
    ggml_hrx_kernel_desc desc;
    desc.file = "fused/gated_update_native.loom";
    desc.symbol = "ggml_inkling_ctrl_prepare_native";
    ggml_hrx_kernel * prepare = ggml_hrx_get_kernel(ctx->device, desc);
    desc.symbol = "ggml_inkling_ctrl_down_native";
    ggml_hrx_kernel * down = ggml_hrx_get_kernel(ctx->device, desc);
    desc.symbol = "ggml_inkling_ctrl_apply_native";
    ggml_hrx_kernel * apply = ggml_hrx_get_kernel(ctx->device, desc);
    if (prepare == nullptr || down == nullptr || apply == nullptr) {
        return ggml_hrx_gated_run_mma(ctx, mt);
    }

    // Scratch holds normalized x and m, z, then the packed down, forget and gain weights.
    const int64_t n = mt.n_t;
    const size_t elements = n * (2 * mt.e + GGML_HRX_GATED_RANK * 2) +
                            ggml_nelements(mt.wd) + ggml_nelements(mt.wf) + ggml_nelements(mt.wg);
    const ggml_tensor * scratch = ggml_hrx_scratch(ctx, elements * sizeof(ggml_bf16_t));
    if (scratch == nullptr) {
        return ggml_hrx_gated_run_mma(ctx, mt);
    }

    return ggml_hrx_dispatch(ctx, prepare, { n },
               { n, mt.x.s1, mt.m.s1, ggml_hrx_gated_bits(mt.eps_x), ggml_hrx_gated_bits(mt.eps_m) },
               { mt.x.t, mt.m.t, mt.wd, mt.wf, mt.wg, scratch }) &&
           ggml_hrx_dispatch(ctx, down, { n }, { n }, { scratch }) &&
           ggml_hrx_dispatch(ctx, apply, { n },
               { n, mt.x.s1, mt.m.s1, (int64_t) (mt.y->nb[1] / sizeof(float)) },
               { mt.x.t, mt.m.t, scratch, mt.y });
}

static bool ggml_hrx_gated_run(ggml_backend_hrx_context * ctx, const ggml_hrx_gated_match & mt) {
    if (ggml_hrx_gated_use_mma(ctx, mt)) {
        if (mt.n_t % 128 == 0 && ggml_hrx_gated_inkling_shape(ctx, mt)) {
            return ggml_hrx_gated_run_packed(ctx, mt);
        }
        return ggml_hrx_gated_run_mma(ctx, mt);
    }
    const int64_t e        = mt.e;
    const int64_t n_t      = mt.n_t;
    const int64_t n_chunks = 2 * e / GGML_HRX_GATED_CHUNK;
    const int64_t sq_off   = n_t * n_chunks * GGML_HRX_GATED_RANK;
    const int64_t z_off    = sq_off + n_t * n_chunks;
    const bool    parts    = n_t <= GGML_HRX_GATED_PARTS_MAX_TOKENS;
    const int64_t n_part   = z_off + (parts ? 0 : n_t * GGML_HRX_GATED_RANK);
    if (n_part >= INT_MAX) {
        return false;
    }
    const ggml_tensor * part = ggml_hrx_scratch(ctx, (size_t) n_part * sizeof(float));
    if (part == nullptr) {
        return false;
    }

    // Prefill: a down workgroup keeps the weights of all 64 rows for 64 tokens, 4 rows in each lane (gfx1151, E 2304,
    // 2048 tokens, model timeline: 16, 32 or 64 tokens within 3 %). Few tokens: 16 rows for each workgroup and one row for
    // each lane give more lanes for the weight reads, and the lanes read x directly (1 token: 6.9 us against 9.4 us with
    // 64 rows and the tokens through workgroup memory). None of this changes the sums.
    const int64_t down_tokens = std::min<int64_t>(n_t, 64);
    ggml_hrx_kernel_desc ddesc;
    ddesc.file   = "fused/gated_update_f32.loom";
    ddesc.symbol = "ggml_gated_update_down";
    const std::pair<std::string, std::string> half_chunks = { "ggml.gated.half_chunks", std::to_string(n_chunks / 2) };
    ddesc.config = {
        half_chunks,
        { "ggml.gated.down_tokens", std::to_string(down_tokens) },
        { "ggml.gated.down_rows", parts ? "16" : "64" },
        { "ggml.gated.down_rpl", parts ? "1" : "4" },
        { "ggml.gated.down_direct", parts ? "1" : "0" },
    };
    if (!ggml_hrx_dispatch(ctx, ggml_hrx_get_kernel(ctx->device, ddesc), { n_chunks, n_t },
            { e, n_t, mt.x.s1, mt.m.s1, sq_off }, { mt.x.t, mt.m.t, mt.wd, part })) {
        return false;
    }

    const int64_t e_bits = ggml_hrx_gated_bits((float) e);
    const int64_t ex     = ggml_hrx_gated_bits(mt.eps_x);
    const int64_t em     = ggml_hrx_gated_bits(mt.eps_m);
    if (!parts) {
        ggml_hrx_kernel_desc zdesc;
        zdesc.file   = "fused/gated_update_f32.loom";
        zdesc.symbol = "ggml_gated_update_z";
        zdesc.config = { half_chunks };
        if (!ggml_hrx_dispatch(ctx, ggml_hrx_get_kernel(ctx->device, zdesc), { n_t }, { e, sq_off, z_off, e_bits, ex, em }, { part })) {
            return false;
        }
    }

    // Few tokens: 32 rows for each workgroup give more workgroups for the weight reads (gfx1151, E 2304, 1 token: 7.8 us
    // against 10.7 us with 64 rows and the partial sums added in a loop).
    const int64_t rows   = parts ? 32 : 256;
    const int64_t tokens = parts ? n_t : 32;
    ggml_hrx_kernel_desc adesc;
    adesc.file   = "fused/gated_update_f32.loom";
    adesc.symbol = "ggml_gated_update_apply";
    adesc.config = {
        half_chunks,
        { "ggml.gated.apply_rows", std::to_string(rows) },
        { "ggml.gated.apply_tokens", std::to_string(tokens) },
        { "ggml.gated.apply_parts", parts ? "1" : "0" },
    };
    return ggml_hrx_dispatch(ctx, ggml_hrx_get_kernel(ctx->device, adesc), { e, n_t },
        { e, n_t, mt.x.s1, mt.m.s1, (int64_t) (mt.y->nb[1] / sizeof(float)), sq_off, z_off, e_bits, ex, em },
        { mt.x.t, mt.m.t, mt.wf, mt.wg, part, mt.y });
}

int ggml_hrx_fuse_gated_update(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, std::vector<int> * absorbed) {
    std::vector<int>      own;
    ggml_hrx_gated_match  mt;
    if (!ggml_hrx_gated_match_at(cgraph, i, own, mt)) {
        return -1;
    }
    if (absorbed != nullptr) {
        absorbed->insert(absorbed->end(), own.begin(), own.end());
        return mt.last;
    }
    const std::vector<const ggml_tensor *> writes = { mt.y };
    ctx->dispatch_writes = &writes;
    if (!ggml_hrx_gated_run(ctx, mt)) {
        GGML_ABORT("%s: failed to run node %d (%s)", __func__, i, cgraph->nodes[i]->name);
    }
    ctx->dispatch_writes = nullptr;
    return mt.last;
}
