#include "ggml-hrx-ops.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// The matrix-op kernel (kernels/matmul/mul_mat_wmma.loom) gets wave32 providers for gfx11 and gfx12 from motifs/wmma.loom.
// Other archs use the tiled kernel without matrix ops until their layout is validated.
static bool ggml_hrx_mul_mat_wmma_arch(const ggml_hrx_device_context * dev) {
    const ggml_hrx_matrix_op op = dev->arch->matrix_op;
    return (op == GGML_HRX_MATRIX_OP_WMMA_GFX11 || op == GGML_HRX_MATRIX_OP_WMMA_GFX12) && dev->arch->wave32;
}

// Weight type config value of kernels/matmul/lib/mm_core.loom. The value is -1 if the type is not supported.
static int ggml_hrx_mul_mat_weight_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return 0;
        case GGML_TYPE_F16:  return 1;
        case GGML_TYPE_Q4_0: return 2;
        case GGML_TYPE_BF16: return 3;
        case GGML_TYPE_Q8_0: return 4;
        default:             return -1;
    }
}

// Bytes of one weight unit: 32-bit words for F32, else halves. Q4_0 and Q8_0 blocks have only 2-byte alignment.
static size_t ggml_hrx_mul_mat_weight_unit(ggml_type type) {
    return type == GGML_TYPE_F32 ? 4 : 2;
}

// Weight strides and spans are in weight units. F32, F16 and BF16 rows must be contiguous.
static bool ggml_hrx_mul_mat_weight_strides(const ggml_tensor * w, int64_t s[GGML_MAX_DIMS]) {
    const size_t unit = ggml_hrx_mul_mat_weight_unit(w->type);
    if (w->view_offs % unit != 0) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (w->nb[i] % unit != 0) {
            return false;
        }
        s[i] = (int64_t) (w->nb[i] / unit);
    }
    return ggml_is_quantized(w->type) || s[0] == 1;
}

bool ggml_hrx_supports_mul_mat(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (ggml_hrx_mul_mat_weight_type(w->type) < 0 || x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    // graph_compute skips an empty dst (for example no output rows).
    if (ggml_is_empty(op)) {
        return true;
    }
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_mul_mat_weight_strides(w, s) || !ggml_hrx_elem_strides(x, s) || !ggml_hrx_elem_strides(op, s)) {
        return false;
    }
    if (x->view_offs % sizeof(float) != 0 || op->nb[0] != sizeof(float)) {
        return false;
    }
    // An empty tensor cannot be a buffer binding.
    if (ggml_is_empty(w) || ggml_is_empty(x)) {
        return false;
    }
    // The tiled kernels use one workgroup in y for each 32 rows or columns.
    const int64_t max_yz = dev->arch->max_workgroups_yz;
    if ((w->ne[1] + 31) / 32 > max_yz || (x->ne[1] + 31) / 32 > max_yz || x->ne[2] * x->ne[3] > max_yz) {
        return false;
    }
    // Indices, strides and spans are 32-bit launch scalars.
    return ggml_nbytes(w) <= INT_MAX && ggml_nbytes(x) <= INT_MAX && ggml_nbytes(op) <= INT_MAX;
}

// The matrix-op kernel needs K in steps of 32 and contiguous input rows.
// It converts src1 to F16. The src1 precision request must allow this. An F32 node keeps both operands in F32.
// F32 and BF16 weights are not converted to F16. The other kernels compute in F32.
static bool ggml_hrx_mul_mat_use_wmma(const ggml_tensor * node, const ggml_tensor * w, const int64_t xs[GGML_MAX_DIMS]) {
    if (w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_BF16 || ggml_prec_get_acc(node) == GGML_PREC_F32) {
        return false;
    }
    const ggml_prec prec_src1 = ggml_prec_get_src(node, 1);
    if (prec_src1 != GGML_PREC_UNDEFINED && prec_src1 < GGML_PREC_F16) {
        return false;
    }
    return w->ne[0] % 32 == 0 && xs[0] == 1;
}

// ggml_mul_mat_vec_f: plain weight types, few input columns and K in steps of 8 values. It needs no K split.
// gfx1151, n = 1, device time in a model graph: F32 [4608,64] 7.8 us against 28.5 us (split tiled + reduce), F32 [64,2304] 4.6 us against 8.4 us.
static bool ggml_hrx_mul_mat_use_vec_f(const ggml_tensor * w, const ggml_tensor * x, const int64_t xs[GGML_MAX_DIMS],
                                       const ggml_hrx_mul_mat_vec_tune & mv) {
    const bool plain = w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_BF16;
    return plain && x->ne[1] <= mv.max_cols && w->ne[0] % 8 == 0 && xs[0] == 1;
}

// Lanes for each row of ggml_mul_mat_vec_f. Each lane reads 8 K values in each step. Aim for at least 2 steps.
// The choice depends only on K. Thus a column gets the same result for each batch size.
static int64_t ggml_hrx_mul_mat_vec_f_lanes(int64_t k) {
    int64_t lanes = 16;
    while (lanes < 256 && lanes * 2 * 8 * 2 <= k) {
        lanes *= 2;
    }
    return lanes;
}

static const std::vector<const char *> ggml_hrx_mul_mat_libs = {
    "motifs/common.loom", "motifs/float.loom", "motifs/q4_0.loom", "motifs/q8_0.loom", "matmul/lib/mm_core.loom", "matmul/lib/mm_vec.loom",
};

// The tiled kernel writes the partial sums of each K range to scratch. The reduce kernel adds them in split order.
static bool ggml_hrx_mul_mat_split(ggml_backend_hrx_context * ctx, const ggml_tensor * node, int64_t chunk, int64_t n_split, bool tiled_f,
                                   const int64_t ws[GGML_MAX_DIMS], const int64_t xs[GGML_MAX_DIMS], const int64_t ds[GGML_MAX_DIMS]) {
    const ggml_tensor * w = node->src[0];
    const ggml_tensor * x = node->src[1];

    const int64_t n_rows       = w->ne[1];
    const int64_t n_cols       = x->ne[1];
    const int64_t n_batch      = x->ne[2] * x->ne[3];
    const int64_t split_stride = n_batch * n_rows * n_cols;
    const int64_t part_span    = n_split * split_stride;
    const ggml_tensor * part   = ggml_hrx_scratch(ctx, (size_t) part_span * sizeof(float));
    if (part == nullptr) {
        return false;
    }

    ggml_hrx_kernel_desc desc;
    desc.file   = "matmul/mul_mat.loom";
    desc.symbol = tiled_f ? "ggml_mul_mat_tiled_f" : "ggml_mul_mat_tiled_split";
    desc.libs   = ggml_hrx_mul_mat_libs;
    desc.config = { { "ggml.mul_mat.weight_type", std::to_string(ggml_hrx_mul_mat_weight_type(w->type)) } };
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    if (!ggml_hrx_dispatch(ctx, kernel, { n_rows, n_cols, n_batch * n_split },
            {
                w->ne[0], n_rows, n_cols, x->ne[2], x->ne[2] / w->ne[2], x->ne[3] / w->ne[3],
                ws[1], ws[2], ws[3], (int64_t) (ggml_nbytes(w) / ggml_hrx_mul_mat_weight_unit(w->type)),
                xs[0], xs[1], xs[2], xs[3], (int64_t) (ggml_nbytes(x) / sizeof(float)),
                n_rows, n_rows * n_cols, x->ne[2] * n_rows * n_cols, part_span,
                n_split, chunk, split_stride,
            },
            { w, x, part })) {
        return false;
    }

    ggml_hrx_kernel_desc rdesc;
    rdesc.file   = "matmul/mul_mat.loom";
    rdesc.symbol = "ggml_mul_mat_split_reduce";
    rdesc.libs   = ggml_hrx_mul_mat_libs;
    ggml_hrx_kernel * reduce = ggml_hrx_get_kernel(ctx->device, rdesc);
    return ggml_hrx_dispatch(ctx, reduce, { n_rows * n_cols, n_batch },
        { n_rows, n_cols, x->ne[2], n_split, split_stride, part_span, ds[1], ds[2], ds[3], (int64_t) (ggml_nbytes(node) / sizeof(float)) },
        { part, node });
}

// ggml_mul_mat_vec_xs keeps at most xs_max_k values of K. A longer K of a quantized weight runs as equal K ranges of whole
// superblocks into scratch, and the split reduce adds them in range order. Each column still has its own sums.
// gfx1151, Q8_0 [9216,2304] x [9216,6] in a model graph (device time): about 150 us for 3 ranges + reduce, 689 us for the tiled kernel.
static bool ggml_hrx_mul_mat_vec_split(ggml_backend_hrx_context * ctx, const ggml_tensor * node, int64_t chunk, int64_t n_split,
                                       const int64_t ds[GGML_MAX_DIMS]) {
    const ggml_tensor * w = node->src[0];
    const ggml_tensor * x = node->src[1];

    const int64_t n_rows       = w->ne[1];
    const int64_t n_cols       = x->ne[1];
    const int64_t split_stride = n_rows * n_cols * x->ne[2] * x->ne[3];
    const int64_t part_span    = n_split * split_stride;
    const ggml_tensor * part   = ggml_hrx_scratch(ctx, (size_t) part_span * sizeof(float));
    if (part == nullptr) {
        return false;
    }

    const std::vector<const ggml_tensor *> * writes = ctx->dispatch_writes;
    bool ok = true;
    for (int64_t s = 0; s < n_split && ok; ++s) {
        const int64_t k0 = s * chunk;
        ggml_tensor wv = *w;
        ggml_tensor xv = *x;
        wv.ne[0] = xv.ne[0] = std::min(chunk, w->ne[0] - k0);
        wv.data  = (char *) w->data + ggml_row_size(w->type, k0);
        xv.data  = (char *) x->data + k0 * sizeof(float);

        // The partial sums of range s, contiguous.
        ggml_tensor dv = *node;
        dv.buffer   = part->buffer;
        dv.view_src = nullptr;
        dv.data     = (char *) part->data + s * split_stride * sizeof(float);
        dv.nb[0]    = sizeof(float);
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            dv.nb[i] = dv.nb[i - 1] * dv.ne[i - 1];
        }
        dv.src[0] = &wv;
        dv.src[1] = &xv;

        const std::vector<const ggml_tensor *> dv_writes = { &dv };
        ctx->dispatch_writes = &dv_writes;
        ok = ggml_hrx_op_mul_mat(ctx, &dv);
        ctx->dispatch_writes = writes;
    }
    if (!ok) {
        return false;
    }

    ggml_hrx_kernel_desc rdesc;
    rdesc.file   = "matmul/mul_mat.loom";
    rdesc.symbol = "ggml_mul_mat_split_reduce";
    rdesc.libs   = ggml_hrx_mul_mat_libs;
    ggml_hrx_kernel * reduce = ggml_hrx_get_kernel(ctx->device, rdesc);
    return ggml_hrx_dispatch(ctx, reduce, { n_rows * n_cols, x->ne[2] * x->ne[3] },
        { n_rows, n_cols, x->ne[2], n_split, split_stride, part_span, ds[1], ds[2], ds[3], (int64_t) (ggml_nbytes(node) / sizeof(float)) },
        { part, node });
}

// ggml_mul_mat_small_k serves plain weight types up to this K. Its stage holds 32 K values of 8 input columns.
static constexpr int64_t GGML_HRX_MUL_MAT_SMALL_K = 32;

// K splits of the tiled kernel for plain weight types. Rows, K and batches give the split, not the columns.
// Thus a column gets the same result for each batch size. A split has at least 256 K values.
// The splits give at least one row tile for each CU. Measured on gfx1151 with F32 weights (test-backend-ops perf):
// [4608,64] x [4608,1] 17.5 us against 203 us unsplit, [2304,128] x [2304,32] 16 us against 108 us.
static int64_t ggml_hrx_mul_mat_split_chunk(const ggml_hrx_device_context * dev, int64_t n_rows, int64_t k, int64_t n_batch) {
    const int64_t tiles   = (n_rows + 63) / 64 * n_batch;
    const int64_t n_cu    = dev->n_cu;
    const int64_t n_split = std::min((n_cu + tiles - 1) / tiles, k / 256);
    if (n_split <= 1) {
        return k;
    }
    return ((k + n_split - 1) / n_split + 31) / 32 * 32;
}

bool ggml_hrx_op_mul_mat(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * w = node->src[0];
    const ggml_tensor * x = node->src[1];

    int64_t ws[GGML_MAX_DIMS];
    int64_t xs[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    ggml_hrx_mul_mat_weight_strides(w, ws);
    ggml_hrx_elem_strides(x, xs);
    ggml_hrx_elem_strides(node, ds);

    const ggml_hrx_mul_mat_vec_tune & mv = ctx->device->tune->mul_mat_vec[w->type];
    // GGML_HINT_EXACT_BATCH: each column must equal the n == 1 result bit for bit. Up to max_cols columns,
    // the kernels compute each column alone. Thus a larger batch runs in chunks of max_cols columns.
    if (x->ne[1] > mv.max_cols && ggml_get_op_params_i32(node, 1) == GGML_HINT_EXACT_BATCH) {
        for (int64_t col0 = 0; col0 < x->ne[1]; col0 += mv.max_cols) {
            ggml_tensor xc = *x;
            ggml_tensor dc = *node;
            xc.ne[1] = dc.ne[1] = std::min(mv.max_cols, x->ne[1] - col0);
            xc.data   = (char *) x->data + col0 * x->nb[1];
            dc.data   = (char *) node->data + col0 * node->nb[1];
            dc.src[1] = &xc;
            if (!ggml_hrx_op_mul_mat(ctx, &dc)) {
                return false;
            }
        }
        return true;
    }

    // gfx1151, DSpark four-tap mix (2304 channels, 4096 tokens): one lane per output took 1.0 ms against 16.4 ms with the tiled kernel.
    if (w->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
        ggml_nelements(w) == 4 && w->ne[0] == 4 && x->ne[1] >= 256 &&
        ggml_nelements(node) % 256 == 0 && ggml_nelements(node) <= INT32_MAX / 4 &&
        ggml_is_contiguous(w) && ggml_is_contiguous(x) && ggml_is_contiguous(node) && strcmp(ctx->device->arch->name, "gfx1151") == 0) {
        ggml_hrx_kernel_desc desc;
        desc.file   = "matmul/mul_mat.loom";
        desc.symbol = "ggml_mul_mat_taps4";
        return ggml_hrx_dispatch(ctx, ggml_hrx_get_kernel(ctx->device, desc), { ggml_nelements(node) / 256 }, {}, { w, x, node });
    }
    // The vector kernel reads K in 256-value superblocks with 4-wide activation loads.
    // The choice does not depend on the column count. Thus a column gets the same result for each batch size.
    // A partial last superblock needs whole 32-value blocks.
    const bool k_tail = w->ne[0] % 256 != 0;
    const bool vec    = x->ne[1] <= mv.max_cols && (!k_tail || ggml_blck_size(w->type) == 32) && w->ne[0] <= mv.xs_max_k &&
                        xs[0] == 1;
    const bool wmma   = x->ne[1] > mv.max_cols && ggml_hrx_mul_mat_wmma_arch(ctx->device) &&
                        ggml_hrx_mul_mat_use_wmma(node, w, xs);
    // ggml_mul_mat_vec_xs keeps fewer than 16 rows. It runs one workgroup for each row, thus more workgroups for many batches.
    // gfx1151, [2304,k,2048] x [2304,1,2048] with k = 2..5 (test-backend-ops perf, device time): 316-472 us against 402-560 us.
    const bool vec_f   = ggml_hrx_mul_mat_use_vec_f(w, x, xs, mv) && !(vec && w->ne[1] < 16);
    const bool plain   = !vec && !wmma && !ggml_is_quantized(w->type);
    const bool small_k = plain && w->ne[0] <= GGML_HRX_MUL_MAT_SMALL_K;
    // gfx1151, F32, 2048 columns (test-backend-ops perf, device time): [2304,128] 225 us against 377 us, [64,2304] 153 us
    // against 234 us, [4608,64] 345 us against 472 us (both with the same K splits).
    const bool tiled_f = plain && !small_k && w->ne[0] % 4 == 0 && xs[0] == 1;

    const int64_t n_batch = x->ne[2] * x->ne[3];

    if (!vec && !wmma && ggml_is_quantized(w->type) && x->ne[1] <= mv.max_cols && xs[0] == 1 && w->ne[0] > mv.xs_max_k) {
        const int64_t blck    = std::max<int64_t>(256, ggml_blck_size(w->type));
        const int64_t n_split = (w->ne[0] + mv.xs_max_k - 1) / mv.xs_max_k;
        const int64_t chunk   = ((w->ne[0] + n_split - 1) / n_split + blck - 1) / blck * blck;
        const int64_t last    = w->ne[0] - (n_split - 1) * chunk;
        if (chunk <= mv.xs_max_k && last > 0 && last % ggml_blck_size(w->type) == 0 && (last % 256 == 0 || ggml_blck_size(w->type) == 32)) {
            return ggml_hrx_mul_mat_vec_split(ctx, node, chunk, n_split, ds);
        }
    }

    if (vec_f) {
        ggml_hrx_kernel_desc desc;
        desc.file   = "matmul/mul_mat.loom";
        desc.symbol = "ggml_mul_mat_vec_f";
        desc.libs   = ggml_hrx_mul_mat_libs;
        desc.libs.push_back("motifs/reduce.loom");
        desc.config = {
            { "ggml.mul_mat.weight_type", std::to_string(ggml_hrx_mul_mat_weight_type(w->type)) },
            { "ggml.mul_mat.vec_cols", std::to_string(x->ne[1]) },
            { "ggml.mul_mat.vec_f_lanes", std::to_string(ggml_hrx_mul_mat_vec_f_lanes(w->ne[0])) },
        };
        ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
        return ggml_hrx_dispatch(ctx, kernel, { w->ne[1], x->ne[1], n_batch },
            {
                w->ne[0], w->ne[1], x->ne[1], x->ne[2], x->ne[2] / w->ne[2], x->ne[3] / w->ne[3],
                ws[1], ws[2], ws[3], (int64_t) (ggml_nbytes(w) / ggml_hrx_mul_mat_weight_unit(w->type)),
                xs[0], xs[1], xs[2], xs[3], (int64_t) (ggml_nbytes(x) / sizeof(float)),
                ds[1], ds[2], ds[3], (int64_t) (ggml_nbytes(node) / sizeof(float)),
            },
            { w, x, node });
    }

    int64_t chunk   = w->ne[0];
    int64_t n_split = 1;
    if (plain && !small_k) {
        chunk   = ggml_hrx_mul_mat_split_chunk(ctx->device, w->ne[1], w->ne[0], n_batch);
        n_split = (w->ne[0] + chunk - 1) / chunk;
        // The partial sums must fit the 32-bit indices and a scratch buffer of at most 256 MB.
        const int64_t part_span = n_split * n_batch * w->ne[1] * x->ne[1];
        if (n_batch * n_split > ctx->device->arch->max_workgroups_yz || part_span > (int64_t) (256u << 20) / 4) {
            chunk   = w->ne[0];
            n_split = 1;
        }
    }
    if (n_split > 1) {
        return ggml_hrx_mul_mat_split(ctx, node, chunk, n_split, tiled_f, ws, xs, ds);
    }
    if (tiled_f) {
        ggml_hrx_kernel_desc desc;
        desc.file   = "matmul/mul_mat.loom";
        desc.symbol = "ggml_mul_mat_tiled_f";
        desc.libs   = ggml_hrx_mul_mat_libs;
        desc.config = { { "ggml.mul_mat.weight_type", std::to_string(ggml_hrx_mul_mat_weight_type(w->type)) } };
        ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
        return ggml_hrx_dispatch(ctx, kernel, { w->ne[1], x->ne[1], n_batch },
            {
                w->ne[0], w->ne[1], x->ne[1], x->ne[2], x->ne[2] / w->ne[2], x->ne[3] / w->ne[3],
                ws[1], ws[2], ws[3], (int64_t) (ggml_nbytes(w) / ggml_hrx_mul_mat_weight_unit(w->type)),
                xs[0], xs[1], xs[2], xs[3], (int64_t) (ggml_nbytes(x) / sizeof(float)),
                ds[1], ds[2], ds[3], (int64_t) (ggml_nbytes(node) / sizeof(float)),
                1, w->ne[0], 0,
            },
            { w, x, node });
    }

    // Q8_0 has its own matrix-op kernels with the same sums. The wide one takes 128 input columns for each workgroup.
    // gfx1151, 2048 columns (test-backend-ops perf, 2 rounds): [2304,3072] 1166-1178 us against 1255-1363 us,
    // [2304,1024] 410-418 against 455-503, [3072,2304] 1639-1721 against 1588-1645.
    // The r256 one takes 256 rows for each workgroup. It needs K in steps of 64 and the gfx11 fragment layout.
    // gfx1151 (test-backend-ops perf, variants interleaved, 3 rounds, busy host): [2304,3072] x 4096 / 2048 columns 0.93 / 0.89
    // of the wide kernel, [3072,2304] 0.91 / 0.92; [2304,1024] 1.05 / 1.00, [2304,384] 1.12 / 1.09, [2304,3072] x 512 1.13.
    // gfx1151, Q8_0 [2304,384] x 4096 columns (Inkling 2 attn_rel, device timestamps): 535 us with 64 columns for each
    // workgroup against 706 us with 128. [2304,1024]: 989 us against 906 us. The sums are the same.
    const bool wmma_q8      = wmma && w->type == GGML_TYPE_Q8_0;
    const bool wmma_q8_wide = wmma_q8 && x->ne[1] >= 128 && w->ne[1] > 512;
    const bool wmma_q8_r256 = wmma_q8_wide && w->ne[0] % 64 == 0 && w->ne[1] >= 2048 && x->ne[1] >= 2048 &&
                              ctx->device->arch->matrix_op == GGML_HRX_MATRIX_OP_WMMA_GFX11;

    // F16 weights with at most 32 rows: 64 columns for each workgroup and the loads of the next K step during the matrix ops.
    // gfx1151, [2304,24] and [2304,32] x 4096 columns (Inkling 2 sink and tap projections, device timestamps): 250 us against
    // 339 us for ggml_mul_mat_wmma (128 rows for each workgroup). The sums are the same.
    const bool wmma_r32 = wmma && w->type == GGML_TYPE_F16 && w->ne[1] <= 32 && w->ne[0] % 64 == 0 && x->ne[1] >= 128 &&
                          ws[1] % 8 == 0 && w->view_offs % 16 == 0 && xs[1] % 4 == 0 && (uintptr_t) x->data % 16 == 0 &&
                          ctx->device->arch->matrix_op == GGML_HRX_MATRIX_OP_WMMA_GFX11;

    ggml_hrx_kernel_desc desc;
    desc.file   = wmma_q8 ? "matmul/mul_mat_wmma_q8_0.loom" : wmma ? "matmul/mul_mat_wmma.loom" : "matmul/mul_mat.loom";
    desc.symbol = wmma_q8_r256 ? "ggml_mul_mat_wmma_q8_0_r256" : wmma_q8_wide ? "ggml_mul_mat_wmma_q8_0_wide" : wmma_q8 ? "ggml_mul_mat_wmma_q8_0" : wmma_r32 ? "ggml_mul_mat_wmma_r32" : wmma ? "ggml_mul_mat_wmma" : vec ? "ggml_mul_mat_vec_xs" : small_k ? "ggml_mul_mat_small_k" : "ggml_mul_mat_tiled";
    desc.libs   = ggml_hrx_mul_mat_libs;
    if (wmma) {
        desc.libs.push_back("matmul/lib/mm_wmma.loom");
        desc.libs.push_back("motifs/wmma.loom");
    }
    if (wmma_q8_r256) {
        // On gfx1151, full-tile stores cut 3072-to-2304 time by 7-13%. Use the checked stores on the last partial tile.
        const bool full_cols = x->ne[1] % 128 == 0;
        const bool direct = strcmp(ctx->device->arch->name, "gfx1151") == 0 && ds[1] == w->ne[1] &&
                            ((w->ne[0] == 3072 && w->ne[1] == 2304) || (full_cols && w->ne[0] == 2304 && w->ne[1] == 3072));
        desc.config.push_back({ "ggml.mul_mat.q8_tile_stride", direct ? std::to_string(ds[1]) : "0" });
        desc.config.push_back({ "ggml.mul_mat.q8_partial_cols", direct && !full_cols ? "1" : "0" });
    }
    if (!wmma_q8) {
        desc.config = { { "ggml.mul_mat.weight_type", std::to_string(ggml_hrx_mul_mat_weight_type(w->type)) } };
    }
    if (vec) {
        desc.config.push_back({ "ggml.mul_mat.vec_cols", std::to_string(x->ne[1]) });
        desc.config.push_back({ "ggml.mul_mat.vec_xs_workgroups", std::to_string(mv.xs_wg_per_cu * ctx->device->n_cu) });
        desc.config.push_back({ "ggml.mul_mat.vec_xs_depth", std::to_string(mv.xs_depth) });
        desc.config.push_back({ "ggml.mul_mat.vec_k_tail", k_tail ? "1" : "0" });
    }
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    std::vector<int64_t> workload;
    if (vec) {
        workload = { w->ne[1], w->ne[0], n_batch };
    } else {
        workload = { w->ne[1], x->ne[1], n_batch };
    }

    return ggml_hrx_dispatch(ctx, kernel, workload,
        {
            w->ne[0], w->ne[1], x->ne[1], x->ne[2], x->ne[2] / w->ne[2], x->ne[3] / w->ne[3],
            ws[1], ws[2], ws[3], (int64_t) (ggml_nbytes(w) / ggml_hrx_mul_mat_weight_unit(w->type)),
            xs[0], xs[1], xs[2], xs[3], (int64_t) (ggml_nbytes(x) / sizeof(float)),
            ds[1], ds[2], ds[3], (int64_t) (ggml_nbytes(node) / sizeof(float)),
        },
        { w, x, node });
}

// Weights for each ggml_mul_mat_vec_group dispatch.
static constexpr int GGML_HRX_MUL_MAT_GROUP_MAX = 6;

// A 2D MUL_MAT with Q8_0 weights that runs ggml_mul_mat_vec_xs. Only these join a group.
static bool ggml_hrx_mul_mat_group_ok(const ggml_hrx_device_context * dev, const ggml_tensor * node) {
    if (node->op != GGML_OP_MUL_MAT || ggml_is_empty(node)) {
        return false;
    }
    const ggml_tensor * w = node->src[0];
    const ggml_tensor * x = node->src[1];
    if (w->type != GGML_TYPE_Q8_0 || !ggml_hrx_supports_mul_mat(dev, node)) {
        return false;
    }
    const ggml_hrx_mul_mat_vec_tune & mv = dev->tune->mul_mat_vec[w->type];
    return w->ne[2] * w->ne[3] == 1 && x->ne[2] * x->ne[3] == 1 && x->ne[1] <= mv.max_cols && w->ne[0] <= mv.xs_max_k &&
           x->nb[0] == sizeof(float);
}

// b can join the group of a: the same input, weight type and K.
static bool ggml_hrx_mul_mat_group_member(const ggml_hrx_device_context * dev, const ggml_tensor * a, const ggml_tensor * b) {
    return b->op == GGML_OP_MUL_MAT && b->src[1] == a->src[1] && b->src[0]->type == a->src[0]->type &&
           b->src[0]->ne[0] == a->src[0]->ne[0] && ggml_hrx_mul_mat_group_ok(dev, b);
}

void ggml_hrx_mul_mat_group_order(const ggml_hrx_device_context * dev, ggml_cgraph * cgraph) {
    static const bool disabled = getenv("GGML_HRX_DISABLE_FUSION") != nullptr;
    if (disabled) {
        return;
    }
    auto is_weight = [](const ggml_tensor * t) {
        return t->op == GGML_OP_NONE && t->view_src == nullptr && t->buffer != nullptr &&
               ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    };
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * a = cgraph->nodes[i];
        if (!ggml_hrx_mul_mat_group_ok(dev, a) || !is_weight(a->src[0])) {
            continue;
        }
        const ggml_tensor * x      = a->src[1];
        const ggml_tensor * x_base = x->view_src != nullptr ? x->view_src : x;
        int pos       = i + 1;
        int n_members = 1;
        for (int j = i + 1; j < cgraph->n_nodes && n_members < GGML_HRX_MUL_MAT_GROUP_MAX; ++j) {
            ggml_tensor * b = cgraph->nodes[j];
            if (ggml_hrx_mul_mat_group_member(dev, a, b) && is_weight(b->src[0])) {
                // b reads only a weight and x. Thus it can run right after the group.
                std::move_backward(cgraph->nodes + pos, cgraph->nodes + j, cgraph->nodes + j + 1);
                cgraph->nodes[pos++] = b;
                n_members++;
                continue;
            }
            // A later member must not move before a node that writes into the memory of x.
            if (b == x_base || b->view_src == x_base) {
                break;
            }
        }
        i = pos - 1;
    }
}

int ggml_hrx_fuse_mul_mat_vec_group(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, std::vector<int> * absorbed) {
    const ggml_hrx_device_context * dev   = ctx->device;
    const ggml_tensor *             first = cgraph->nodes[i];
    if (!ggml_hrx_mul_mat_group_ok(dev, first)) {
        return -1;
    }
    std::vector<const ggml_tensor *> members = { first };
    int last = i;
    for (int j = i + 1; j < cgraph->n_nodes && (int) members.size() < GGML_HRX_MUL_MAT_GROUP_MAX; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_is_empty(n) || ggml_op_is_empty(n->op) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (!ggml_hrx_mul_mat_group_member(dev, first, n)) {
            break;
        }
        members.push_back(n);
        last = j;
    }
    if (members.size() < 2) {
        return -1;
    }
    const ggml_tensor * x = first->src[1];
    for (size_t m = 0; m < members.size(); ++m) {
        if (ggml_hrx_overlap(members[m], x)) {
            return -1;
        }
        for (size_t l = 0; l < m; ++l) {
            if (ggml_hrx_overlap(members[m], members[l])) {
                return -1;
            }
        }
    }

    if (absorbed != nullptr) {
        return last;
    }

    const ggml_type                   type     = first->src[0]->type;
    const ggml_hrx_mul_mat_vec_tune & mv       = dev->tune->mul_mat_vec[type];
    const int64_t                     k        = first->src[0]->ne[0];
    const bool                        k_tail   = k % 256 != 0;
    // The same plan as in ggml_hrx_mv_xs_plan (kernels/matmul/mul_mat.loom), for the sum of the rows.
    const int64_t                     n_sb     = std::min<int64_t>(std::max<int64_t>((k + (k_tail ? 255 : 0)) / 256, 1), 16);
    const int64_t                     rows_par = 16 / n_sb;
    int64_t                           total_pass = 0;
    for (const ggml_tensor * m : members) {
        total_pass += (m->src[0]->ne[1] + rows_par - 1) / rows_par;
    }
    // The kernel keeps 256 / columns passes of partial sums, more than ggml_mul_mat_vec_xs. Thus a large group also gets about
    // one wave of workgroups. gfx1151, Inkling 2 decode with graph replay (device timestamps), 28 groups of 3072, 384 and 3072
    // rows and 20 groups that add 1024 and 1024 rows, K 2304: 4047-4111 us with 160 workgroups each against 4347 us with
    // at most 32 passes (204 and 268 workgroups), and about 4620 us for the same rows in single dispatches.
    const int64_t n_wg_target = mv.xs_wg_per_cu * dev->n_cu;
    int64_t       n_pass      = std::min<int64_t>((total_pass + n_wg_target - 1) / n_wg_target, 256 / x->ne[1]);
    n_pass                    = std::max<int64_t>(n_pass, total_pass >= 64 ? 4 : 1);

    std::vector<int64_t> starts(GGML_HRX_MUL_MAT_GROUP_MAX);
    int64_t n_wg = 0;
    for (int m = 0; m < GGML_HRX_MUL_MAT_GROUP_MAX; ++m) {
        starts[m] = n_wg;
        if (m < (int) members.size()) {
            n_wg += ((members[m]->src[0]->ne[1] + rows_par - 1) / rows_par + n_pass - 1) / n_pass;
        }
    }
    for (int m = (int) members.size(); m < GGML_HRX_MUL_MAT_GROUP_MAX; ++m) {
        starts[m] = n_wg;
    }

    ggml_hrx_kernel_desc desc;
    desc.file   = "matmul/mul_mat_vec_group.loom";
    desc.symbol = "ggml_mul_mat_vec_group";
    desc.libs   = ggml_hrx_mul_mat_libs;
    desc.config = {
        { "ggml.mul_mat.weight_type", std::to_string(ggml_hrx_mul_mat_weight_type(type)) },
        { "ggml.mul_mat.vec_cols", std::to_string(x->ne[1]) },
        { "ggml.mul_mat.vec_xs_depth", std::to_string(mv.xs_depth) },
        { "ggml.mul_mat.vec_k_tail", k_tail ? "1" : "0" },
    };
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    std::vector<int64_t> scalars = { k, x->ne[1], (int64_t) (x->nb[1] / sizeof(float)), (int64_t) (ggml_nbytes(x) / sizeof(float)), n_pass };
    scalars.insert(scalars.end(), starts.begin() + 1, starts.end());
    std::vector<const ggml_tensor *> weights;
    std::vector<const ggml_tensor *> dsts;
    for (int m = 0; m < GGML_HRX_MUL_MAT_GROUP_MAX; ++m) {
        const ggml_tensor * node = members[m < (int) members.size() ? m : 0];
        weights.push_back(node->src[0]);
        dsts.push_back(node);
    }
    const size_t unit = ggml_hrx_mul_mat_weight_unit(type);
    for (const ggml_tensor * w : weights) {
        scalars.push_back(w->ne[1]);
    }
    for (const ggml_tensor * w : weights) {
        scalars.push_back((int64_t) (w->nb[1] / unit));
    }
    for (const ggml_tensor * w : weights) {
        scalars.push_back((int64_t) (ggml_nbytes(w) / unit));
    }
    for (const ggml_tensor * d : dsts) {
        scalars.push_back((int64_t) (d->nb[1] / sizeof(float)));
    }
    for (const ggml_tensor * d : dsts) {
        scalars.push_back((int64_t) (ggml_nbytes(d) / sizeof(float)));
    }
    std::vector<const ggml_tensor *> buffers = { x };
    buffers.insert(buffers.end(), weights.begin(), weights.end());
    buffers.insert(buffers.end(), dsts.begin(), dsts.end());

    // The dispatch writes only the member outputs.
    std::vector<const ggml_tensor *> writes(members.begin(), members.end());
    ctx->dispatch_writes = &writes;
    const bool ok        = ggml_hrx_dispatch(ctx, kernel, { n_wg }, scalars, buffers);
    ctx->dispatch_writes = nullptr;
    if (!ok) {
        GGML_ABORT("%s: failed to run node %d (%s)", __func__, i, first->name);
    }
    return last;
}
