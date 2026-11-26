#include "ggml-hrx-ops.h"

#include <climits>
#include <string>

// Tensors from this size use 1024 lanes (not tuned).
static constexpr int64_t GGML_HRX_SUM_WIDE_N = 16384;

bool ggml_hrx_supports_sum(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    int64_t ss[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(src, ss)) {
        return false;
    }
    return ggml_nelements(src) > 0 && ggml_nelements(src) < INT_MAX - 4096 && (int64_t) (ggml_nbytes(src) / 4) < INT_MAX - 4096;
}

bool ggml_hrx_op_sum(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];
    const int64_t       n   = ggml_nelements(src);

    int64_t ss[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(src, ss);
    const int64_t span   = (int64_t) (ggml_nbytes(src) / 4);
    const bool    contig = ggml_is_contiguous(src) && span >= 4;

    ggml_hrx_kernel_desc desc;
    desc.file   = "reduce/sum_f32.loom";
    desc.symbol = "ggml_sum_f32";
    desc.config = {
        { "ggml.sum.contig", contig ? "1" : "0" },
        { "ggml.sum.wg_size", std::to_string(n >= GGML_HRX_SUM_WIDE_N ? 1024 : ctx->device->tune->norm_wg_size) },
    };
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    return ggml_hrx_dispatch(ctx, kernel, { 1 },
        { n, src->ne[0], src->ne[1], src->ne[2], ss[0], ss[1], ss[2], ss[3], span },
        { src, node });
}
