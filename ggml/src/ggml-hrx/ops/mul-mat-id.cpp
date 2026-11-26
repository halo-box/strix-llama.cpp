#include "ggml-hrx-ops.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <string>

// Weight type config value of kernels/matmul/mul_mat_id.loom. The value is -1 if the type is not supported.
static int ggml_hrx_mul_mat_id_weight_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:   return 1;
        case GGML_TYPE_Q8_0:  return 4;
        case GGML_TYPE_F32:   return 32;
        case GGML_TYPE_NVFP4: return 40;
        default:              return -1;
    }
}

// The kernels load F32 and NVFP4 weights in 4-byte words, F16 and Q8_0 weights in 2-byte halves.
static size_t ggml_hrx_mul_mat_id_weight_align(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_NVFP4 ? 4 : 2;
}

static bool ggml_hrx_mul_mat_id_aligned(const ggml_tensor * t, size_t align) {
    if (t->view_offs % align != 0) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->nb[i] % align != 0) {
            return false;
        }
    }
    return true;
}

// All kernels sum in F32. This meets each accumulator precision request (ggml_prec_get_acc).
bool ggml_hrx_supports_mul_mat_id(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    const ggml_tensor * w   = op->src[0];
    const ggml_tensor * x   = op->src[1];
    const ggml_tensor * ids = op->src[2];
    if (ggml_hrx_mul_mat_id_weight_type(w->type) < 0 || x->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
        op->type != GGML_TYPE_F32) {
        return false;
    }
    // graph_compute skips an empty dst.
    if (ggml_is_empty(op)) {
        return true;
    }
    // An empty tensor cannot be a buffer binding.
    if (ggml_is_empty(w) || ggml_is_empty(x)) {
        return false;
    }
    // The kernels read K in units of 16 values with contiguous weight and input rows.
    if (w->ne[0] % 16 != 0 || w->nb[0] != ggml_type_size(w->type) || x->nb[0] != sizeof(float) || op->nb[0] != sizeof(float)) {
        return false;
    }
    if (!ggml_hrx_mul_mat_id_aligned(w, ggml_hrx_mul_mat_id_weight_align(w->type)) || !ggml_hrx_mul_mat_id_aligned(x, 4) ||
        !ggml_hrx_mul_mat_id_aligned(ids, 4) || !ggml_hrx_mul_mat_id_aligned(op, 4)) {
        return false;
    }
    // One workgroup in y for each route.
    const int64_t n_routes = ids->ne[0] * ids->ne[1];
    if (n_routes > dev->arch->max_workgroups_yz) {
        return false;
    }
    // Indices, strides and spans are 32-bit launch scalars.
    return ggml_nbytes(w) <= INT_MAX && ggml_nbytes(x) <= INT_MAX && ggml_nbytes(ids) <= INT_MAX && ggml_nbytes(op) <= INT_MAX;
}

// Same arch check as the matrix-op kernel of mul_mat.
static bool ggml_hrx_mul_mat_id_wmma_arch(const ggml_hrx_device_context * dev) {
    const ggml_hrx_matrix_op op = dev->arch->matrix_op;
    return (op == GGML_HRX_MATRIX_OP_WMMA_GFX11 || op == GGML_HRX_MATRIX_OP_WMMA_GFX12) && dev->arch->wave32;
}

// Routes for each expert in 1/16.
static int64_t ggml_hrx_mul_mat_id_load(const ggml_tensor * node) {
    const ggml_tensor * ids = node->src[2];
    return 16 * ids->ne[0] * ids->ne[1] / node->src[0]->ne[2];
}

// The tiled kernel converts src1 to F16. The src1 precision request must allow this.
static bool ggml_hrx_mul_mat_id_use_wmma(const ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * w   = node->src[0];
    const ggml_prec prec_src1 = ggml_prec_get_src(node, 1);
    if (prec_src1 != GGML_PREC_UNDEFINED && prec_src1 < GGML_PREC_F16) {
        return false;
    }
    if (w->type != GGML_TYPE_F16 && w->type != GGML_TYPE_Q8_0 && w->type != GGML_TYPE_NVFP4) {
        return false;
    }
    return ggml_hrx_mul_mat_id_wmma_arch(ctx->device) && w->ne[0] % 32 == 0 &&
           ggml_hrx_mul_mat_id_load(node) >= ctx->device->tune->mul_mat_id_wmma_load;
}

// Groups the routes by expert into records of tile routes (kernels/matmul/mul_mat_id_routes.loom).
// A bucket of p routes has ceil(p / tile) records, and only buckets with routes have records.
// The table is at the start of the scratch. extra: scratch bytes after the table, from table_bytes on.
static const ggml_tensor * ggml_hrx_mul_mat_id_routes(ggml_backend_hrx_context * ctx, const ggml_tensor * node, int64_t tile, int64_t * records,
                                                      size_t extra = 0, size_t * table_bytes = nullptr) {
    const ggml_tensor * w   = node->src[0];
    const ggml_tensor * ids = node->src[2];
    const int64_t n_routes = ids->ne[0] * ids->ne[1];
    const int64_t n_buckets = w->ne[2] + 1;
    *records = std::min(n_routes, n_routes / tile + std::min(n_buckets, n_routes));
    const int64_t table_words = 4 + *records * (tile + 2);
    const size_t  bytes       = GGML_PAD((size_t) table_words * sizeof(int32_t), 256);
    const ggml_tensor * table = ggml_hrx_scratch(ctx, bytes + extra);
    if (table == nullptr || table_words >= INT_MAX) {
        return nullptr;
    }
    if (table_bytes != nullptr) {
        *table_bytes = bytes;
    }

    ggml_hrx_kernel_desc desc;
    desc.file   = "matmul/mul_mat_id_routes.loom";
    desc.symbol = "ggml_mul_mat_id_routes";
    desc.libs   = { "matmul/lib/mm_id.loom" };
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
    if (!ggml_hrx_dispatch(ctx, kernel, {},
            {
                n_routes, ids->ne[0], w->ne[2], tile, *records,
                (int64_t) (ids->nb[0] / 4), (int64_t) (ids->nb[1] / 4), (int64_t) (ggml_nbytes(ids) / 4),
            },
            { ids, table })) {
        return nullptr;
    }
    return table;
}

// The 128-route kernel reads an f16 copy of src1 and takes NVFP4 and Q8_0 weights with K a multiple of 64.
static bool ggml_hrx_mul_mat_id_use_wmma128(const ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * w = node->src[0];
    const ggml_tensor * x = node->src[1];
    if (w->type != GGML_TYPE_NVFP4 && w->type != GGML_TYPE_Q8_0) {
        return false;
    }
    const int64_t xh_halves = x->ne[0] * x->ne[1] * x->ne[2];
    return w->ne[0] % 64 == 0 && x->ne[3] == 1 && xh_halves < INT_MAX / 4 && ggml_hrx_mul_mat_id_use_wmma(ctx, node) &&
           ggml_hrx_mul_mat_id_load(node) >= ctx->device->tune->mul_mat_id_wmma128_load;
}

// Scratch layout of the 128-route kernel: the route table, then f16 rows of K halves (in halves from the scratch start).
struct ggml_hrx_mmid128_scratch {
    const ggml_tensor * scratch = nullptr;
    int64_t             records = 0;
    int64_t             xh_off  = 0;
};

// Groups the routes into records of 128 and copies src1 of node to f16 rows (token * ne11 + slot). extra: scratch halves
// after the f16 rows.
static bool ggml_hrx_mmid128_prepare(ggml_backend_hrx_context * ctx, const ggml_tensor * node, int64_t extra, ggml_hrx_mmid128_scratch & sc) {
    const ggml_tensor * x = node->src[1];
    const int64_t k      = x->ne[0];
    const int64_t x_rows = x->ne[1] * x->ne[2];
    size_t table_bytes = 0;
    sc.scratch = ggml_hrx_mul_mat_id_routes(ctx, node, 128, &sc.records, (size_t) (k * x_rows + extra) * sizeof(ggml_fp16_t), &table_bytes);
    if (sc.scratch == nullptr) {
        return false;
    }
    sc.xh_off = (int64_t) (table_bytes / sizeof(ggml_fp16_t));
    if (sc.xh_off + k * x_rows + extra >= INT_MAX) {
        return false;
    }
    ggml_hrx_kernel_desc cvt;
    cvt.file   = "matmul/mul_mat_id_wmma.loom";
    cvt.symbol = "ggml_mul_mat_id_x16";
    const int64_t n_items = k / 8 * x_rows;
    return ggml_hrx_dispatch(ctx, ggml_hrx_get_kernel(ctx->device, cvt), { n_items },
        {
            n_items, k / 8, x->ne[1], (int64_t) (x->nb[1] / 4), (int64_t) (x->nb[2] / 4), (int64_t) (ggml_nbytes(x) / 4),
            sc.xh_off, sc.xh_off + k * x_rows,
        },
        { x, sc.scratch });
}

// Activation of the GLU mode of ggml_mul_mat_id_wmma128: the expert scale (nullptr: none) and the SCALE parameters of the nodes.
struct ggml_hrx_mmid128_glu {
    const ggml_tensor * es = nullptr;
    float               p[8] = {};
};

// The 128-route kernel. Its input is the f16 rows at xh_off of sc (ne11 rows for each token). Without glu it writes node, or
// with out_off >= 0 f32 rows (out_f16: f16 rows) of w->ne[1] values at out_off (in floats, or halves) in the scratch (route * w->ne[1]).
// With glu it writes the f16 activations of the n_ff = w->ne[1] / 2 outputs at out_off (in halves) in the scratch (route * n_ff).
// To the scratch it can multiply each value by es[expert] (es != nullptr) and then by the weight rw[slot, token] (rw != nullptr).
static bool ggml_hrx_mmid128_run(ggml_backend_hrx_context * ctx, const ggml_tensor * node, const ggml_tensor * w, int64_t ne11,
                                 const ggml_hrx_mmid128_scratch & sc, int64_t xh_off, const ggml_hrx_mmid128_glu * glu, int64_t out_off,
                                 const ggml_tensor * es = nullptr, const ggml_tensor * rw = nullptr, bool out_f16 = false) {
    const ggml_tensor * ids = node->src[2];
    const int64_t n_ids  = ids->ne[0];
    const int64_t k      = w->ne[0];
    const int64_t xh_end = xh_off + k * ne11 * ids->ne[1];
    ggml_hrx_kernel_desc desc;
    desc.file   = "matmul/mul_mat_id_wmma.loom";
    desc.symbol = "ggml_mul_mat_id_wmma128";
    desc.libs   = { "motifs/common.loom", "motifs/float.loom", "motifs/q4_0.loom", "motifs/q8_0.loom", "motifs/nvfp4.loom",
                    "motifs/wmma.loom", "matmul/lib/mm_core.loom", "matmul/lib/mm_wmma.loom", "matmul/lib/mm_id_wmma.loom" };
    desc.config = { { "ggml.mul_mat_id.weight_type", std::to_string(ggml_hrx_mul_mat_id_weight_type(w->type)) },
                    { "ggml.mul_mat_id.glu", glu != nullptr ? "1" : "0" },
                    { "ggml.mul_mat_id.d_f16", out_f16 ? "1" : "0" } };
    std::vector<int64_t> scalars = {
        k, w->ne[1], w->ne[2], n_ids, ne11, sc.records,
        (int64_t) (w->nb[1] / 2), (int64_t) (w->nb[2] / 2), (int64_t) (ggml_nbytes(w) / 2),
        xh_off, xh_end,
    };
    if (glu != nullptr) {
        const int64_t n_ff = w->ne[1] / 2;
        scalars.insert(scalars.end(), { out_off, n_ff, n_ids * n_ff, out_off + n_ids * ids->ne[1] * n_ff, n_ff, glu->es != nullptr ? 1 : 0,
                                        0, 0, 0, 1 });
        for (float v : glu->p) {
            uint32_t bits;
            std::memcpy(&bits, &v, sizeof(bits));
            scalars.push_back((int64_t) bits);
        }
        return ggml_hrx_dispatch(ctx, ggml_hrx_get_kernel(ctx->device, desc), { w->ne[1], sc.records }, scalars,
                                 { w, sc.scratch, sc.scratch, glu->es != nullptr ? glu->es : w, w });
    }
    if (out_off >= 0) {
        scalars.insert(scalars.end(), { out_off, w->ne[1], n_ids * w->ne[1], out_off + n_ids * ids->ne[1] * w->ne[1], 1, es != nullptr ? 1 : 0 });
    } else {
        scalars.insert(scalars.end(), { 0, (int64_t) (node->nb[1] / 4), (int64_t) (node->nb[2] / 4), (int64_t) (ggml_nbytes(node) / 4), 1, 0 });
    }
    if (rw != nullptr) {
        scalars.insert(scalars.end(), { 1, (int64_t) (rw->nb[1] / 4), (int64_t) (rw->nb[2] / 4), (int64_t) (ggml_nbytes(rw) / 4) });
    } else {
        scalars.insert(scalars.end(), { 0, 0, 0, 1 });
    }
    scalars.insert(scalars.end(), 8, 0);
    return ggml_hrx_dispatch(ctx, ggml_hrx_get_kernel(ctx->device, desc), { w->ne[1], sc.records }, scalars,
                             { w, sc.scratch, out_off >= 0 ? sc.scratch : node, es != nullptr ? es : w, rw != nullptr ? rw : w });
}

bool ggml_hrx_op_mul_mat_id(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * w   = node->src[0];
    const ggml_tensor * x   = node->src[1];
    const ggml_tensor * ids = node->src[2];

    const int64_t n_ids    = ids->ne[0];
    const int64_t n_routes = n_ids * ids->ne[1];
    const std::string wtype = std::to_string(ggml_hrx_mul_mat_id_weight_type(w->type));
    // The routes kernel takes at most 1023 experts.
    const bool grouped = w->ne[2] < 1024;

    if (grouped && ggml_hrx_mul_mat_id_use_wmma(ctx, node)) {

        if (ggml_hrx_mul_mat_id_use_wmma128(ctx, node)) {
            ggml_hrx_mmid128_scratch sc;
            return ggml_hrx_mmid128_prepare(ctx, node, 0, sc) && ggml_hrx_mmid128_run(ctx, node, w, x->ne[1], sc, sc.xh_off, nullptr, -1);
        }
        int64_t records = 0;
        const ggml_tensor * table = ggml_hrx_mul_mat_id_routes(ctx, node, 64, &records);
        if (table == nullptr) {
            return false;
        }
        ggml_hrx_kernel_desc desc;
        desc.file   = "matmul/mul_mat_id_wmma.loom";
        desc.symbol = "ggml_mul_mat_id_wmma";
        desc.libs   = { "motifs/common.loom", "motifs/float.loom", "motifs/q4_0.loom", "motifs/q8_0.loom", "motifs/nvfp4.loom",
                        "motifs/wmma.loom", "matmul/lib/mm_core.loom", "matmul/lib/mm_wmma.loom" };
        desc.config = { { "ggml.mul_mat_id.weight_type", wtype } };
        ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
        return ggml_hrx_dispatch(ctx, kernel, { w->ne[1], records },
            {
                w->ne[0], w->ne[1], w->ne[2], n_ids, x->ne[1], records,
                (int64_t) (w->nb[1] / 2), (int64_t) (w->nb[2] / 2), (int64_t) (ggml_nbytes(w) / 2),
                (int64_t) (x->nb[1] / 4), (int64_t) (x->nb[2] / 4), (int64_t) (ggml_nbytes(x) / 4),
                (int64_t) (node->nb[1] / 4), (int64_t) (node->nb[2] / 4), (int64_t) (ggml_nbytes(node) / 4),
            },
            { w, x, table, node });
    }

    ggml_hrx_kernel_desc desc;
    desc.file   = "matmul/mul_mat_id.loom";
    desc.libs   = { "motifs/nvfp4.loom", "matmul/lib/mm_id.loom" };
    desc.config = { { "ggml.mul_mat_id.weight_type", wtype } };

    if (grouped && ggml_hrx_mul_mat_id_load(node) >= ctx->device->tune->mul_mat_id_rec_load) {
        int64_t records = 0;
        const ggml_tensor * table = ggml_hrx_mul_mat_id_routes(ctx, node, 8, &records);
        if (table == nullptr) {
            return false;
        }
        desc.symbol = "ggml_mul_mat_id_vec_rec";
        ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
        return ggml_hrx_dispatch(ctx, kernel, { w->ne[1], records },
            {
                w->ne[0], w->ne[1], w->ne[2], n_ids, x->ne[1], records,
                (int64_t) w->nb[1], (int64_t) w->nb[2],
                (int64_t) (x->nb[1] / 4), (int64_t) (x->nb[2] / 4),
                (int64_t) (node->nb[1] / 4), (int64_t) (node->nb[2] / 4), (int64_t) (ggml_nbytes(node) / 4),
            },
            { w, x, table, node });
    }

    desc.symbol = "ggml_mul_mat_id_vec";
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
    return ggml_hrx_dispatch(ctx, kernel, { w->ne[1], n_routes },
        {
            w->ne[0], w->ne[1], w->ne[2], n_ids, x->ne[1],
            (int64_t) w->nb[1], (int64_t) w->nb[2],
            (int64_t) (x->nb[1] / 4), (int64_t) (x->nb[2] / 4),
            (int64_t) (ids->nb[0] / 4), (int64_t) (ids->nb[1] / 4), (int64_t) (ggml_nbytes(ids) / 4),
            (int64_t) (node->nb[1] / 4), (int64_t) (node->nb[2] / 4), (int64_t) (ggml_nbytes(node) / 4),
        },
        { w, x, ids, node });
}

// Expert FFN with a merged gate/up projection and the SITU activation, as nodes:
//   gu = MUL_MAT_ID(w_gu, x, ids), sg = GET_ROWS(REPEAT(es), ids), gus = MUL(gu, sg), gate and up = views of gus,
//   act = MUL(MUL(SCALE(TANH(SCALE(gate))), SIGMOID(gate)), SCALE(TANH(SCALE(up)))), down = MUL_MAT_ID(w_down, act, ids).
// The expert scale (sg and the MUL) is optional. The routes are grouped once. gu and act are not stored: the gate/up kernel
// writes the f16 activations (the values the down kernel rounds act to) to the scratch. The range ends at down, or with the
// combine after it at the last ADD:
//   sd = GET_ROWS(REPEAT(es_down), ids) (optional), wd = MUL(MUL(down, sd), weights), out = ADD(...ADD(ADD(v0, v1), v2)..., vn)
// with v_k the view of slot k of wd. Then down goes to the scratch too. The REPEAT and GET_ROWS are absorbed when they come
// before gu.
struct ggml_hrx_moe_glu_match {
    const ggml_tensor *  gu      = nullptr;
    const ggml_tensor *  down    = nullptr;
    ggml_hrx_mmid128_glu glu;
    const ggml_tensor *  out     = nullptr; // last ADD of the combine, or nullptr
    const ggml_tensor *  weights = nullptr;
    const ggml_tensor *  es_down = nullptr;
    int                  last    = -1;
};

// The absorbed nodes are at most this many nodes before gu.
static constexpr int GGML_HRX_MOE_GLU_MAX_DIST = 64;
// Compute nodes from gu to down (the pattern has 14), and from down to the last ADD of the combine (n_used + 3).
static constexpr int GGML_HRX_MOE_GLU_MAX_NODES = 32;

// sg = GET_ROWS(REPEAT(es), ids) with es [1, n_as] and the repeat [1, n_as, T]. Returns es.
static const ggml_tensor * ggml_hrx_moe_glu_gathered_scale(const ggml_cgraph * g, const ggml_tensor * sg, const ggml_tensor * ids, int64_t n_as) {
    if (sg == nullptr || sg->op != GGML_OP_GET_ROWS || sg->src[1] != ids || sg->type != GGML_TYPE_F32 || sg->ne[0] != 1 ||
        sg->ne[1] != ids->ne[0] || sg->ne[2] != ids->ne[1] || !ggml_hrx_only_used_by(g, sg, 1)) {
        return nullptr;
    }
    const ggml_tensor * rep = sg->src[0];
    const ggml_tensor * es  = rep->src[0];
    if (rep->op != GGML_OP_REPEAT || rep->type != GGML_TYPE_F32 || rep->ne[0] != 1 || rep->ne[1] != n_as || rep->ne[2] != ids->ne[1] ||
        rep->ne[3] != 1 || es == nullptr || es->type != GGML_TYPE_F32 || ggml_nelements(es) != n_as || !ggml_is_contiguous(es) ||
        es->ne[0] != 1 || es->ne[1] != n_as || !ggml_hrx_only_used_by(g, rep, 1)) {
        return nullptr;
    }
    return es;
}

// The combine after down: fills mt.out, mt.weights, mt.es_down and adds its nodes to own. Returns the index of the last ADD.
static int ggml_hrx_moe_glu_combine(const ggml_cgraph * g, int d_idx, ggml_hrx_moe_glu_match & mt, std::vector<const ggml_tensor *> & own) {
    const ggml_tensor * down   = mt.down;
    const ggml_tensor * ids    = down->src[2];
    const int64_t       n_used = ids->ne[0];
    const int64_t       n_as   = down->src[0]->ne[2];
    if (n_used < 2 || n_used > 64 || down->ne[0] % 4 != 0 || down->ne[3] != 1) {
        return -1;
    }
    int k = d_idx;
    auto next = [&]() -> const ggml_tensor * {
        k = ggml_hrx_next_node(g, k);
        return k >= 0 && k - d_idx < 4 * GGML_HRX_MOE_GLU_MAX_NODES ? g->nodes[k] : nullptr;
    };
    std::vector<const ggml_tensor *> nodes;
    const ggml_tensor * x  = down;
    const ggml_tensor * es = nullptr;
    const ggml_tensor * wd = nullptr;
    // MUL(down, sd) and MUL(., weights), with the REPEAT and GET_ROWS of sd in between or before.
    while (wd == nullptr) {
        const ggml_tensor * n = next();
        if (n == nullptr) {
            return -1;
        }
        if (n->op != GGML_OP_MUL || n->src[0] != x || n->type != GGML_TYPE_F32 || !ggml_are_same_shape(n, down)) {
            continue;
        }
        const ggml_tensor * es_n = x == down ? ggml_hrx_moe_glu_gathered_scale(g, n->src[1], ids, n_as) : nullptr;
        if (es_n != nullptr) {
            if (!ggml_hrx_only_used_by(g, down, 1)) {
                return -1;
            }
            es = es_n;
            x  = n;
            nodes.insert(nodes.end(), { n, n->src[1], n->src[1]->src[0] });
            continue;
        }
        const ggml_tensor * w = n->src[1];
        if (w->type != GGML_TYPE_F32 || w->ne[0] != 1 || w->ne[1] != n_used || w->ne[2] != down->ne[2] || w->ne[3] != 1 ||
            !ggml_hrx_only_used_by(g, x, 1) || ggml_nbytes(w) > INT_MAX) {
            return -1;
        }
        mt.weights = w;
        wd         = n;
        nodes.push_back(n);
    }
    if (!ggml_hrx_only_used_by(g, wd, (int) n_used)) {
        return -1;
    }
    // ADD(v0, v1), then ADD(prev, v_k).
    auto slot_view = [&](const ggml_tensor * v, int64_t slot) {
        return v != nullptr && v->op == GGML_OP_VIEW && v->src[0] == wd && v->view_src == wd && v->view_offs == (size_t) slot * wd->nb[1] &&
               v->ne[0] == down->ne[0] && v->ne[1] == down->ne[2] && v->ne[2] == 1 && v->ne[3] == 1 && v->nb[0] == sizeof(float) &&
               v->nb[1] == wd->nb[2] && ggml_hrx_only_used_by(g, v, 1);
    };
    const ggml_tensor * sum = nullptr;
    for (int64_t slot = 1; slot < n_used; ++slot) {
        const ggml_tensor * n = next();
        if (n == nullptr || n->op != GGML_OP_ADD || n->type != GGML_TYPE_F32 || !slot_view(n->src[1], slot) ||
            (slot == 1 ? !slot_view(n->src[0], 0) : n->src[0] != sum) || (sum != nullptr && !ggml_hrx_only_used_by(g, sum, 1))) {
            return -1;
        }
        nodes.push_back(n);
        sum = n;
    }
    if (sum->ne[0] != down->ne[0] || sum->ne[1] != down->ne[2] || sum->ne[2] != 1 || sum->ne[3] != 1 || sum->nb[0] != sizeof(float) ||
        ggml_nbytes(sum) > INT_MAX) {
        return -1;
    }
    own.insert(own.end(), nodes.begin(), nodes.end());
    mt.out     = sum;
    mt.es_down = es;
    return k;
}

static bool ggml_hrx_moe_glu_unary(const ggml_tensor * t, ggml_unary_op op, const ggml_tensor * src) {
    return t != nullptr && t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op && t->src[0] == src && t->type == GGML_TYPE_F32;
}

static bool ggml_hrx_moe_glu_scale(const ggml_tensor * t, const ggml_tensor * src, float * p) {
    if (t == nullptr || t->op != GGML_OP_SCALE || t->src[0] != src || t->type != GGML_TYPE_F32) {
        return false;
    }
    p[0] = ggml_get_op_params_f32(t, 0);
    p[1] = ggml_get_op_params_f32(t, 1);
    return true;
}

// A view of gus with the shape of the gate or up half.
static bool ggml_hrx_moe_glu_half(const ggml_tensor * v, const ggml_tensor * gus, int64_t n_ff, size_t offs) {
    return v != nullptr && v->op == GGML_OP_VIEW && v->src[0] == gus && v->view_src == gus && v->view_offs == offs &&
           v->ne[0] == n_ff && v->ne[1] == gus->ne[1] && v->ne[2] == gus->ne[2] && v->ne[3] == 1 && v->nb[0] == gus->nb[0] &&
           v->nb[1] == gus->nb[1] && v->nb[2] == gus->nb[2];
}

static bool ggml_hrx_moe_glu_match_at(ggml_backend_hrx_context * ctx, const ggml_cgraph * g, int i, std::vector<int> & absorbed, ggml_hrx_moe_glu_match & mt) {
    const ggml_tensor * gu = g->nodes[i];
    if (gu->op != GGML_OP_MUL_MAT_ID || gu->type != GGML_TYPE_F32 || (gu->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || gu->ne[0] % 128 != 0 ||
        gu->ne[3] != 1 || !ggml_is_contiguous(gu) || ggml_hrx_mul_mat_id_weight_type(gu->src[0]->type) < 0 ||
        !ggml_hrx_supports_mul_mat_id(ctx->device, gu) || !ggml_hrx_mul_mat_id_use_wmma128(ctx, gu)) {
        return false;
    }
    const ggml_tensor * ids  = gu->src[2];
    const int64_t       n_ff = gu->ne[0] / 2;

    // down: the next MUL_MAT_ID with the same ids.
    int last = -1;
    for (int k = i, n = 0; n < GGML_HRX_MOE_GLU_MAX_NODES; ++n) {
        k = ggml_hrx_next_node(g, k);
        if (k < 0) {
            return false;
        }
        if (g->nodes[k]->op == GGML_OP_MUL_MAT_ID) {
            last = g->nodes[k]->src[2] == ids ? k : -1;
            break;
        }
    }
    if (last < 0) {
        return false;
    }
    const ggml_tensor * down = g->nodes[last];
    const ggml_tensor * act  = down->src[1];
    if (down->type != GGML_TYPE_F32 || down->src[0]->ne[0] != n_ff || down->src[0]->ne[2] != gu->src[0]->ne[2] ||
        !ggml_hrx_supports_mul_mat_id(ctx->device, down) || !ggml_hrx_mul_mat_id_use_wmma128(ctx, down)) {
        return false;
    }

    // act = MUL(MUL(g3, SIGMOID(gate)), u3)
    const ggml_tensor * a  = act->src[0];
    const ggml_tensor * u3 = act->src[1];
    if (act->op != GGML_OP_MUL || act->type != GGML_TYPE_F32 || a == nullptr || u3 == nullptr || a->op != GGML_OP_MUL ||
        a->type != GGML_TYPE_F32 || act->ne[0] != n_ff || act->ne[1] != gu->ne[1] || act->ne[2] != gu->ne[2] || act->ne[3] != 1) {
        return false;
    }
    const ggml_tensor * g3  = a->src[0];
    const ggml_tensor * sgm = a->src[1];
    float p[8];
    if (g3 == nullptr || sgm == nullptr || g3->op != GGML_OP_SCALE || u3->op != GGML_OP_SCALE) {
        return false;
    }
    const ggml_tensor * g2 = g3->src[0];
    const ggml_tensor * u2 = u3->src[0];
    if (g2 == nullptr || u2 == nullptr) {
        return false;
    }
    const ggml_tensor * g1 = g2->src[0];
    const ggml_tensor * u1 = u2->src[0];
    if (g1 == nullptr || u1 == nullptr || g1->src[0] == nullptr || u1->src[0] == nullptr) {
        return false;
    }
    const ggml_tensor * gate = g1->src[0];
    const ggml_tensor * up   = u1->src[0];
    const ggml_tensor * gus  = gate->src[0];
    if (gus == nullptr || !ggml_hrx_moe_glu_scale(g1, gate, p + 0) || !ggml_hrx_moe_glu_unary(g2, GGML_UNARY_OP_TANH, g1) ||
        !ggml_hrx_moe_glu_scale(g3, g2, p + 2) || !ggml_hrx_moe_glu_scale(u1, up, p + 4) ||
        !ggml_hrx_moe_glu_unary(u2, GGML_UNARY_OP_TANH, u1) || !ggml_hrx_moe_glu_scale(u3, u2, p + 6) ||
        sgm->op != GGML_OP_UNARY || ggml_get_unary_op(sgm) != GGML_UNARY_OP_SIGMOID || sgm->type != GGML_TYPE_F32 ||
        !ggml_hrx_moe_glu_half(gate, gus, n_ff, 0) || !ggml_hrx_moe_glu_half(sgm->src[0], gus, n_ff, 0) ||
        !ggml_hrx_moe_glu_half(up, gus, n_ff, (size_t) n_ff * sizeof(float))) {
        return false;
    }

    // Without the expert scale the views are of gu. Else gus = MUL(gu, GET_ROWS(REPEAT(es), ids)) with es [1, n_expert] and
    // the repeat [1, n_expert, T].
    const bool          scaled = gus != gu;
    const ggml_tensor * sg  = scaled ? gus->src[1] : nullptr;
    const ggml_tensor * rep = nullptr;
    const ggml_tensor * es  = nullptr;
    if (scaled) {
        if (gus->op != GGML_OP_MUL || gus->type != GGML_TYPE_F32 || gus->src[0] != gu || !ggml_are_same_shape(gus, gu) ||
            !ggml_is_contiguous(gus) || sg == nullptr || sg->op != GGML_OP_GET_ROWS || sg->src[1] != ids || sg->type != GGML_TYPE_F32 ||
            sg->ne[0] != 1 || sg->ne[1] != gu->ne[1] || sg->ne[2] != gu->ne[2]) {
            return false;
        }
        rep = sg->src[0];
        es  = rep->src[0];
        const int64_t n_as = gu->src[0]->ne[2];
        if (rep->op != GGML_OP_REPEAT || rep->type != GGML_TYPE_F32 || rep->ne[0] != 1 || rep->ne[1] != n_as || rep->ne[2] != gu->ne[2] ||
            rep->ne[3] != 1 || es == nullptr || es->type != GGML_TYPE_F32 || ggml_nelements(es) != n_as || !ggml_is_contiguous(es) ||
            es->ne[0] != 1 || es->ne[1] != n_as || !ggml_hrx_only_used_by(g, rep, 1) || !ggml_hrx_only_used_by(g, sg, 1)) {
            return false;
        }
    }

    // Uses: views count as readers. gus has the gate, sigmoid gate and up views.
    const int n_gate_views = sgm->src[0] == gate ? 1 : 2;
    const int gate_uses    = sgm->src[0] == gate ? 2 : 1;
    if ((scaled && !ggml_hrx_only_used_by(g, gu, 1)) || !ggml_hrx_only_used_by(g, gus, n_gate_views + 1) ||
        !ggml_hrx_only_used_by(g, gate, gate_uses) || (n_gate_views == 2 && !ggml_hrx_only_used_by(g, sgm->src[0], 1)) ||
        !ggml_hrx_only_used_by(g, up, 1)) {
        return false;
    }
    const ggml_tensor * chain[] = { g1, g2, g3, sgm, a, u1, u2, u3, act };
    for (const ggml_tensor * t : chain) {
        if (!ggml_hrx_only_used_by(g, t, 1)) {
            return false;
        }
    }

    // Every node that runs in the range is a pattern node. The REPEAT and GET_ROWS can also come before gu.
    // Every node that runs in the range is a pattern node. The nodes of the gathered scales that are not in the range come
    // before gu. Without a match with the combine, the range ends at down.
    std::vector<const ggml_tensor *> own = { gu, rep, sg, gus, g1, g2, g3, sgm, a, u1, u2, u3, act, down };
    mt.down = down;
    std::vector<const ggml_tensor *> own_combine = own;
    const int combine_last = ggml_hrx_moe_glu_combine(g, last, mt, own_combine);
    auto range_ok = [&](const std::vector<const ggml_tensor *> & nodes, int end, std::vector<int> & before) {
        for (int k = i + 1; k < end; ++k) {
            const ggml_tensor * n = g->nodes[k];
            if (ggml_op_is_empty(n->op) || ggml_is_empty(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                continue;
            }
            bool found = false;
            for (const ggml_tensor * t : nodes) {
                found = found || t == n;
            }
            if (!found) {
                return false;
            }
        }
        for (const ggml_tensor * t : nodes) {
            if (t == nullptr || ggml_hrx_node_index(g, t, end, end - i + 1) >= 0) {
                continue;
            }
            const int k = ggml_hrx_node_index(g, t, i - 1, GGML_HRX_MOE_GLU_MAX_DIST);
            if (k < 0 || (t->op != GGML_OP_REPEAT && t->op != GGML_OP_GET_ROWS)) {
                return false;
            }
            before.push_back(k);
        }
        return true;
    };
    std::vector<int> before;
    if (combine_last > last && range_ok(own_combine, combine_last, before)) {
        last = combine_last;
    } else {
        mt.out     = nullptr;
        mt.weights = nullptr;
        mt.es_down = nullptr;
        before.clear();
        if (!range_ok(own, last, before)) {
            return false;
        }
    }
    absorbed.insert(absorbed.end(), before.begin(), before.end());

    mt.gu     = gu;
    mt.glu.es = es;
    std::memcpy(mt.glu.p, p, sizeof(p));
    mt.last   = last;
    return true;
}

int ggml_hrx_fuse_moe_glu(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, std::vector<int> * absorbed) {

    std::vector<int>       own;
    ggml_hrx_moe_glu_match mt;
    if (!ggml_hrx_moe_glu_match_at(ctx, cgraph, i, own, mt)) {
        return -1;
    }
    if (absorbed != nullptr) {
        absorbed->insert(absorbed->end(), own.begin(), own.end());
        return mt.last;
    }
    const std::vector<const ggml_tensor *> writes = { mt.out != nullptr ? mt.out : mt.down };
    ctx->dispatch_writes = &writes;
    // Scratch after the route table (in halves): x as f16, the f16 activations, with the combine the f32 down rows.
    const ggml_tensor * ids     = mt.gu->src[2];
    const ggml_tensor * x       = mt.gu->src[1];
    const int64_t       n_ff    = mt.gu->ne[0] / 2;
    const int64_t       n_ids   = ids->ne[0];
    const int64_t       n_embd  = mt.down->ne[0];
    const int64_t       x_len   = x->ne[0] * x->ne[1] * ids->ne[1];
    const int64_t       act_len = GGML_PAD(n_ff * n_ids * ids->ne[1], 128);
    // With the combine the down rows are f16, as the activations: each weighted route row is rounded to f16 before the sum, and
    // the combine reads half the bytes. gfx1151, Inkling 2 Mini, 4096 tokens (model graph, device timestamps): combine 903 us
    // against 1642 us, down 128-route kernel about -1 %.
    const int64_t       dn_len  = mt.out != nullptr ? n_embd * n_ids * ids->ne[1] : 0;
    ggml_hrx_mmid128_scratch sc;
    bool ok = ggml_hrx_mmid128_prepare(ctx, mt.gu, act_len + dn_len, sc) && sc.xh_off + x_len + act_len + dn_len < INT_MAX;
    const int64_t act_off = sc.xh_off + x_len;
    const int64_t dn_off  = act_off + act_len;
    // With the combine the down kernel applies the down expert scale and the routing weights.
    ok = ok && ggml_hrx_mmid128_run(ctx, mt.gu, mt.gu->src[0], x->ne[1], sc, sc.xh_off, &mt.glu, act_off) &&
         ggml_hrx_mmid128_run(ctx, mt.down, mt.down->src[0], n_ids, sc, act_off, nullptr, mt.out != nullptr ? dn_off : -1, mt.es_down, mt.weights,
                              mt.out != nullptr);
    if (ok && mt.out != nullptr) {
        ggml_hrx_kernel_desc desc;
        desc.file   = "matmul/mul_mat_id_wmma.loom";
        desc.symbol = "ggml_moe_combine";
        desc.config = { { "ggml.mul_mat_id.d_f16", "1" } };
        const int64_t n_items = n_embd / 4 * ids->ne[1];
        ok = ggml_hrx_dispatch(ctx, ggml_hrx_get_kernel(ctx->device, desc), { n_items },
            { n_items, n_embd / 4, n_ids, dn_off, dn_off + n_embd * n_ids * ids->ne[1], (int64_t) (mt.out->nb[1] / 4), (int64_t) (ggml_nbytes(mt.out) / 4) },
            { sc.scratch, mt.out });
    }
    if (!ok) {
        GGML_ABORT("%s: failed to run node %d (%s)", __func__, i, cgraph->nodes[i]->name);
    }
    ctx->dispatch_writes = nullptr;
    return mt.last;
}
