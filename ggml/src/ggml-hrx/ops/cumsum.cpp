#include "ggml-hrx-ops.h"

#include <climits>
#include <string>

// Rows from this length use 1024 lanes (not tuned).
static constexpr int64_t GGML_HRX_CUMSUM_WIDE_NE0 = 8192;

bool ggml_hrx_supports_cumsum(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || !ggml_are_same_shape(src, op)) {
        return false;
    }
    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(src, ss) || !ggml_hrx_elem_strides(op, ds) || ss[0] != 1 || ds[0] != 1) {
        return false;
    }
    // A tile can go 4096 values past ne0.
    return src->ne[0] < INT_MAX - 4096 && ggml_nrows(src) < INT_MAX && (int64_t) (ggml_nbytes(src) / 4) < INT_MAX &&
           (int64_t) (ggml_nbytes(op) / 4) < INT_MAX;
}

bool ggml_hrx_op_cumsum(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];
    if (ggml_nelements(node) == 0) {
        return true;
    }

    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(src, ss);
    ggml_hrx_elem_strides(node, ds);

    const int64_t wg = src->ne[0] >= GGML_HRX_CUMSUM_WIDE_NE0 ? 1024 : ctx->device->tune->norm_wg_size;

    ggml_hrx_kernel_desc desc;
    desc.file   = "reduce/cumsum_f32.loom";
    desc.symbol = "ggml_cumsum_f32";
    desc.config = { { "ggml.cumsum.wg_size", std::to_string(wg) } };
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    return ggml_hrx_dispatch(ctx, kernel, { ggml_nrows(src) },
        {
            src->ne[0], src->ne[1], src->ne[2],
            ss[1], ss[2], ss[3], (int64_t) (ggml_nbytes(src) / 4),
            ds[1], ds[2], ds[3], (int64_t) (ggml_nbytes(node) / 4),
        },
        { src, node });
}
