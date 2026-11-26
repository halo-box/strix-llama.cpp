#include "ggml-hrx-ops.h"

#include <climits>
#include <string>

// Rows from this length use 1024 lanes (not tuned).
static constexpr int64_t GGML_HRX_ARGMAX_WIDE_NE0 = 16384;

bool ggml_hrx_supports_argmax(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_I32 || !ggml_is_matrix(src)) {
        return false;
    }
    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(src, ss) || !ggml_hrx_elem_strides(op, ds)) {
        return false;
    }
    return src->ne[1] < INT_MAX && (int64_t) (ggml_nbytes(src) / 4) < INT_MAX - 4096 && (int64_t) (ggml_nbytes(op) / 4) < INT_MAX;
}

bool ggml_hrx_op_argmax(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];
    if (ggml_nelements(node) == 0) {
        return true;
    }

    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(src, ss);
    ggml_hrx_elem_strides(node, ds);

    const int64_t span = (int64_t) (ggml_nbytes(src) / 4);
    const bool    vec4 = ss[0] == 1 && span >= 4;
    const int64_t wg   = src->ne[0] >= GGML_HRX_ARGMAX_WIDE_NE0 ? 1024 : ctx->device->tune->norm_wg_size;

    ggml_hrx_kernel_desc desc;
    desc.file   = "reduce/argmax_f32.loom";
    desc.symbol = "ggml_argmax_f32";
    desc.config = {
        { "ggml.argmax.vec4", vec4 ? "1" : "0" },
        { "ggml.argmax.wg_size", std::to_string(wg) },
    };
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    return ggml_hrx_dispatch(ctx, kernel, { src->ne[1] },
        { src->ne[0], ss[0], ss[1], span, ds[0], (int64_t) (ggml_nbytes(node) / 4) },
        { src, node });
}
