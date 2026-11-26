#include "ggml-hrx-ops.h"

#include <climits>
#include <string>

// Must match the dst type codes in copy/cpy_f32.loom.
static int ggml_hrx_cpy_dst_code(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return 0;
        case GGML_TYPE_F16: return 1;
        case GGML_TYPE_I32: return 2;
        default:            return -1;
    }
}

// Must match the src type codes in copy/cpy_f32.loom.
static int ggml_hrx_cpy_src_code(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return 0;
        case GGML_TYPE_F16: return 1;
        case GGML_TYPE_I32: return 2;
        default:            return -1;
    }
}

// Indices and spans are 32-bit launch scalars.
static bool ggml_hrx_fits_i32(const ggml_tensor * t) {
    return ggml_nelements(t) < INT_MAX && ggml_nbytes(t) / ggml_type_size(t->type) < INT_MAX;
}

// The kernel does not use the stride of a size-1 dim. Set it to 0.
static bool ggml_hrx_launch_strides(const ggml_tensor * t, int64_t s[GGML_MAX_DIMS]) {
    if (!ggml_hrx_elem_strides(t, s)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] == 1) {
            s[i] = 0;
        }
        if (s[i] >= INT_MAX) {
            return false;
        }
    }
    return true;
}

bool ggml_hrx_supports_cpy(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    if (src == nullptr || ggml_hrx_cpy_src_code(src->type) < 0 || ggml_hrx_cpy_dst_code(op->type) < 0 ||
        ggml_nelements(src) != ggml_nelements(op)) {
        return false;
    }
    // Values go through f32. Thus i32 -> i32 can lose bits and i32 -> f16 rounds two times.
    if (src->type == GGML_TYPE_I32 && op->type != GGML_TYPE_F32) {
        return false;
    }
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_launch_strides(src, s) || !ggml_hrx_launch_strides(op, s)) {
        return false;
    }
    return ggml_hrx_fits_i32(src) && ggml_hrx_fits_i32(op);
}

bool ggml_hrx_op_cpy(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src    = node->src[0];
    const ggml_tensor * dst    = node;
    const int64_t       n      = ggml_nelements(dst);
    const bool          stream = src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && n % 4 == 0 &&
                        ggml_is_contiguous(src) && ggml_is_contiguous(dst);
    const int64_t       groups = stream ? ggml_hrx_stream_groups(ctx->device, n) : 0;
    if (groups > 0) {
        ggml_hrx_kernel_desc desc;
        desc.file   = "copy/cpy_f32.loom";
        desc.symbol = "ggml_cpy_f32_stream";
        desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
        return ggml_hrx_dispatch(ctx, ggml_hrx_get_kernel(ctx->device, desc), { groups }, { n, groups }, { src, dst });
    }

    int64_t ss[GGML_MAX_DIMS];
    int64_t ts[GGML_MAX_DIMS];
    ggml_hrx_launch_strides(src, ss);
    ggml_hrx_launch_strides(dst, ts);

    ggml_hrx_kernel_desc desc;
    desc.file   = "copy/cpy_f32.loom";
    desc.symbol = "ggml_cpy_f32";
    desc.config = {
        { "ggml.cpy.dst_type", std::to_string(ggml_hrx_cpy_dst_code(dst->type)) },
        { "ggml.cpy.src_type", std::to_string(ggml_hrx_cpy_src_code(src->type)) },
    };
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    // src and dst can have different shapes. Both use logical element order.
    return ggml_hrx_dispatch(ctx, kernel, { n },
        {
            n,
            src->ne[0], src->ne[1], src->ne[2], ss[0], ss[1], ss[2], ss[3], (int64_t) (ggml_nbytes(src) / ggml_type_size(src->type)),
            dst->ne[0], dst->ne[1], dst->ne[2], ts[0], ts[1], ts[2], ts[3], (int64_t) (ggml_nbytes(dst) / ggml_type_size(dst->type)),
        },
        { src, dst });
}

// Source ranges of copy/concat_f32.loom. dst element i comes from src element i - lo if lo <= i < hi in each dim.
struct ggml_hrx_concat_src {
    int64_t lo[GGML_MAX_DIMS] = {};
    int64_t hi[GGML_MAX_DIMS] = {};
    int64_t s[GGML_MAX_DIMS]  = {};
    int64_t span              = 1;
};

static bool ggml_hrx_concat_src_init(const ggml_tensor * t, const int64_t lo[GGML_MAX_DIMS], ggml_hrx_concat_src * src) {
    if (t->type != GGML_TYPE_F32 || !ggml_hrx_fits_i32(t) || !ggml_hrx_launch_strides(t, src->s)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        src->lo[i] = lo[i];
        src->hi[i] = lo[i] + t->ne[i];
        if (lo[i] < 0 || src->hi[i] >= INT_MAX) {
            return false;
        }
    }
    src->span = (int64_t) (ggml_nbytes(t) / 4);
    return true;
}

// If b is nullptr, the kernel uses only a.
static bool ggml_hrx_concat_dispatch(ggml_backend_hrx_context * ctx, const ggml_tensor * node, const ggml_tensor * a,
                                     const ggml_hrx_concat_src & as, const ggml_tensor * b, const ggml_hrx_concat_src & bs) {
    const int64_t n = ggml_nelements(node);
    if (n == 0) {
        return true;
    }
    ggml_hrx_kernel_desc desc;
    desc.file   = "copy/concat_f32.loom";
    desc.symbol = "ggml_concat_f32";
    desc.config = { { "ggml.concat.has_b", b != nullptr ? "1" : "0" } };
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    return ggml_hrx_dispatch(ctx, kernel, { n },
        {
            n, node->ne[0], node->ne[1], node->ne[2],
            as.lo[0], as.lo[1], as.lo[2], as.lo[3], as.hi[0], as.hi[1], as.hi[2], as.hi[3],
            as.s[0], as.s[1], as.s[2], as.s[3], as.span,
            bs.lo[0], bs.lo[1], bs.lo[2], bs.lo[3], bs.hi[0], bs.hi[1], bs.hi[2], bs.hi[3],
            bs.s[0], bs.s[1], bs.s[2], bs.s[3], bs.span,
        },
        { a, b != nullptr ? b : a, node });
}

static bool ggml_hrx_concat_srcs(const ggml_tensor * op, ggml_hrx_concat_src * as, ggml_hrx_concat_src * bs) {
    const int     dim                 = ggml_get_op_params_i32(op, 0);
    const int64_t a_lo[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    int64_t       b_lo[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    if (dim < 0 || dim >= GGML_MAX_DIMS) {
        return false;
    }
    b_lo[dim] = op->src[0]->ne[dim];
    return ggml_hrx_concat_src_init(op->src[0], a_lo, as) && ggml_hrx_concat_src_init(op->src[1], b_lo, bs);
}

bool ggml_hrx_supports_concat(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    if (op->src[0] == nullptr || op->src[1] == nullptr || op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op) ||
        !ggml_hrx_fits_i32(op)) {
        return false;
    }
    ggml_hrx_concat_src as;
    ggml_hrx_concat_src bs;
    return ggml_hrx_concat_srcs(op, &as, &bs);
}

// From this many rows of b on, the tiled kernel runs. Fewer rows (decode) stay in the element kernel and in auto-fusion.
static constexpr int64_t GGML_HRX_CONCAT_TRANSPOSED_MIN_ROWS = 32;

bool ggml_hrx_concat_is_transposed(const ggml_tensor * op) {
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    return op->op == GGML_OP_CONCAT && a != nullptr && b != nullptr && ggml_get_op_params_i32(op, 0) == 0 && op->ne[3] == 1 &&
           b->nb[1] == sizeof(float) && b->nb[0] != sizeof(float) && b->ne[0] >= GGML_HRX_CONCAT_TRANSPOSED_MIN_ROWS;
}

bool ggml_hrx_op_concat(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    ggml_hrx_concat_src as;
    ggml_hrx_concat_src bs;
    ggml_hrx_concat_srcs(node, &as, &bs);
    if (!ggml_hrx_concat_is_transposed(node) || ggml_nelements(node) == 0) {
        return ggml_hrx_concat_dispatch(ctx, node, node->src[0], as, node->src[1], bs);
    }
    // The element kernel reads b with a large stride between neighbour lanes. gfx1151, test-backend-ops perf, b the
    // transpose of x: x [8192, 512] 78 us against 1733 us, x [10240, 2048] 2351 us against 14330 us.
    const int64_t n_t0 = (node->ne[0] + 63) / 64;
    const int64_t n_t1 = (node->ne[1] + 63) / 64;
    // The kernel does not read an empty a. b fills its binding.
    const ggml_tensor * a = ggml_is_empty(node->src[0]) ? node->src[1] : node->src[0];
    as.span = std::max<int64_t>(as.span, 1);
    ggml_hrx_kernel_desc desc;
    desc.file   = "copy/concat_f32.loom";
    desc.symbol = "ggml_concat_transposed_f32";
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
    return ggml_hrx_dispatch(ctx, kernel, { n_t0 * n_t1 * node->ne[2] },
        {
            node->ne[0], node->ne[1], n_t0, n_t1, node->src[0]->ne[0],
            as.s[0], as.s[1], as.s[2], as.span, bs.s[0], bs.s[1], bs.s[2], bs.span, ggml_nelements(node),
        },
        { a, node->src[1], node });
}

// Zero padding. lo is the left pad of each dim.
static bool ggml_hrx_pad_src(const ggml_tensor * op, ggml_hrx_concat_src * as) {
    int64_t lo[GGML_MAX_DIMS];
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        lo[i] = ggml_get_op_params_i32(op, 2 * i);
        if (ggml_get_op_params_i32(op, 2 * i + 1) < 0) {
            return false;
        }
    }
    return ggml_hrx_concat_src_init(op->src[0], lo, as);
}

bool ggml_hrx_supports_pad(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    // The kernel does not do the circular mode.
    if (op->src[0] == nullptr || op->type != GGML_TYPE_F32 || ggml_get_op_params_i32(op, 8) != 0 ||
        !ggml_is_contiguous(op) || !ggml_hrx_fits_i32(op)) {
        return false;
    }
    ggml_hrx_concat_src as;
    return ggml_hrx_pad_src(op, &as);
}

bool ggml_hrx_op_pad(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    ggml_hrx_concat_src as;
    ggml_hrx_pad_src(node, &as);
    return ggml_hrx_concat_dispatch(ctx, node, node->src[0], as, nullptr, ggml_hrx_concat_src());
}
