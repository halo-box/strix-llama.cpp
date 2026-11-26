#include "ggml-hrx-ops.h"

#include <climits>
#include <string>

// Counts and spans are 32-bit launch scalars.
static bool ggml_hrx_rows_fits(const ggml_tensor * t) {
    return ggml_nelements(t) < INT_MAX && ggml_nbytes(t) / ggml_type_size(t->type) < INT_MAX;
}

static int64_t ggml_hrx_rows_span(const ggml_tensor * t) {
    return (int64_t) (ggml_nbytes(t) / ggml_type_size(t->type));
}

// Strides and span of a get_rows src in the units of rows/get_rows_cvt.loom: elements, or 2-byte units for q8_0.
static bool ggml_hrx_get_rows_src_strides(const ggml_tensor * src, int64_t s[GGML_MAX_DIMS], int64_t * span) {
    if (src->type != GGML_TYPE_Q8_0) {
        *span = ggml_hrx_rows_span(src);
        return ggml_hrx_elem_strides(src, s);
    }
    if (src->nb[0] != ggml_type_size(src->type)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (src->nb[i] % 2 != 0) {
            return false;
        }
        s[i] = (int64_t) (src->nb[i] / 2);
    }
    *span = (int64_t) (ggml_nbytes(src) / 2);
    return true;
}

bool ggml_hrx_supports_get_rows(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    const ggml_tensor * ids = op->src[1];
    // i32 rows are copied as f32 bits.
    const bool i32 = src->type == GGML_TYPE_I32 && op->type == GGML_TYPE_I32;
    if ((src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16 && src->type != GGML_TYPE_BF16 && src->type != GGML_TYPE_Q8_0 && !i32) ||
        ids->type != GGML_TYPE_I32 || (op->type != GGML_TYPE_F32 && !i32)) {
        return false;
    }
    if (!ggml_is_contiguous(op)) {
        return false;
    }
    int64_t is[GGML_MAX_DIMS];
    int64_t ss[GGML_MAX_DIMS];
    int64_t span;
    if (!ggml_hrx_elem_strides(ids, is) || !ggml_hrx_rows_fits(ids) || !ggml_hrx_get_rows_src_strides(src, ss, &span)) {
        return false;
    }
    // q8_0 has only the 4-wide kernel. It needs the blocks of a row next to each other.
    if (src->type == GGML_TYPE_Q8_0 && (op->ne[0] % 4 != 0 || ss[0] != 17)) {
        return false;
    }
    // All indices must be below 2^31 - 8.
    return ggml_nelements(op) <= INT_MAX - 4096 && span <= INT_MAX - 4096;
}

bool ggml_hrx_op_get_rows(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];
    const ggml_tensor * ids = node->src[1];

    const int64_t n = ggml_nelements(node);
    if (n == 0) {
        return true;
    }

    int64_t ss[GGML_MAX_DIMS];
    int64_t is[GGML_MAX_DIMS];
    int64_t src_span;
    ggml_hrx_get_rows_src_strides(src, ss, &src_span);
    ggml_hrx_elem_strides(ids, is);

    const ggml_type src_type = src->type == GGML_TYPE_I32 ? GGML_TYPE_F32 : src->type;
    const bool      vec4     = node->ne[0] % 4 == 0 && (src_type == GGML_TYPE_Q8_0 || ss[0] == 1);
    if (src_type != GGML_TYPE_F32 || !vec4) {
        ggml_hrx_kernel_desc desc;
        desc.file   = "rows/get_rows_cvt.loom";
        desc.config = { { "ggml.get_rows.src_type", std::to_string((int) src_type) } };
        desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
        if (vec4) {
            desc.symbol = "ggml_get_rows_x4";
            desc.libs   = { "motifs/common.loom", "motifs/float.loom", "motifs/q8_0.loom" };
            ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
            return ggml_hrx_dispatch(ctx, kernel, { n },
                {
                    n, node->ne[0], node->ne[1], node->ne[2],
                    src->ne[1], ss[1], ss[2], ss[3], src_span,
                    is[0], is[1], is[2], ggml_hrx_rows_span(ids),
                },
                { src, ids, node });
        }
        desc.symbol = "ggml_get_rows_x1";
        ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
        return ggml_hrx_dispatch(ctx, kernel, { n },
            {
                n, node->ne[0], node->ne[1], node->ne[2],
                src->ne[1], ss[0], ss[1], ss[2], ss[3], src_span,
                is[0], is[1], is[2], ggml_hrx_rows_span(ids),
            },
            { src, ids, node });
    }

    const int64_t groups = ggml_hrx_stream_groups(ctx->device, n);

    ggml_hrx_kernel_desc desc;
    desc.file = "rows/get_rows_f32.loom";
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    if (groups > 0) {
        desc.symbol = "ggml_get_rows_f32_stream";
        ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
        return ggml_hrx_dispatch(ctx, kernel, { groups },
            {
                n, groups, node->ne[0], node->ne[1], node->ne[2],
                src->ne[1], ss[1], ss[2], ss[3], src_span,
                is[0], is[1], is[2], ggml_hrx_rows_span(ids),
            },
            { src, ids, node });
    }
    desc.symbol = "ggml_get_rows_f32_rows";
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    return ggml_hrx_dispatch(ctx, kernel, { n },
        {
            n, node->ne[0], node->ne[1], node->ne[2],
            src->ne[1], ss[0], ss[1], ss[2], ss[3], src_span,
            is[0], is[1], is[2], ggml_hrx_rows_span(ids),
        },
        { src, ids, node });
}

bool ggml_hrx_supports_set_rows(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    const ggml_tensor * ids = op->src[1];
    if (src->type != GGML_TYPE_F32 || (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) ||
        (ids->type != GGML_TYPE_I64 && ids->type != GGML_TYPE_I32)) {
        return false;
    }
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(src, s) || !ggml_hrx_elem_strides(ids, s) || !ggml_hrx_elem_strides(op, s)) {
        return false;
    }
    return ggml_hrx_rows_fits(op) && ggml_hrx_rows_fits(src) && ggml_hrx_rows_fits(ids);
}

bool ggml_hrx_op_set_rows(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];
    const ggml_tensor * ids = node->src[1];

    const int64_t n = ggml_nelements(src);
    if (n == 0) {
        return true;
    }

    int64_t ss[GGML_MAX_DIMS];
    int64_t is[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(src, ss);
    ggml_hrx_elem_strides(ids, is);
    ggml_hrx_elem_strides(node, ds);

    ggml_hrx_kernel_desc desc;
    desc.file   = "rows/set_rows_f32.loom";
    desc.symbol = "ggml_set_rows_f32";
    desc.config = {
        { "ggml.set_rows.dst_type", node->type == GGML_TYPE_F16 ? "1" : "0" },
        { "ggml.set_rows.ids_i32", ids->type == GGML_TYPE_I32 ? "1" : "0" },
    };
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    // node is the dst view. Its binding covers only the rows that the kernel can write.
    return ggml_hrx_dispatch(ctx, kernel, { n },
        {
            n, src->ne[0], src->ne[1], src->ne[2],
            ss[0], ss[1], ss[2], ss[3], ggml_hrx_rows_span(src),
            ids->ne[1], ids->ne[2], is[0], is[1], is[2], ggml_hrx_rows_span(ids),
            node->ne[1], ds[0], ds[1], ds[2], ds[3], ggml_hrx_rows_span(node),
        },
        { src, ids, node });
}
