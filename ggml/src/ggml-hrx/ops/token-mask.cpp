#include "ggml-hrx-ops.h"
#include "ggml-token-mask.h"

#include <algorithm>
#include <climits>

bool ggml_hrx_supports_token_mask(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    if (!ggml_token_mask_valid_layout(op->src[0], op->src[1], op->src[2], op->src[3]) ||
        op->type != GGML_TYPE_F32 || !ggml_are_same_shape(op, op->src[0]) || !ggml_is_contiguous(op) ||
        ggml_nelements(op) >= INT_MAX - 1024) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        const auto * src = op->src[i];
        if (src && (ggml_nbytes(src) / 4 >= INT_MAX - 1024 || src->nb[1] / 4 >= INT_MAX - 1024)) {
            return false;
        }
    }
    return true;
}

bool ggml_hrx_op_token_mask(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const auto * logits = node->src[0];
    const auto * ids = node->src[1];
    const auto * payload = node->src[2];
    const auto * mask_desc = node->src[3];
    const int64_t payload_size = payload->ne[0];
    // Empty inputs use a valid binding; the kernel does not read them.
    const auto * payload_binding = payload_size ? payload : node;
    const auto * ids_binding = ids ? ids : node;
    ggml_hrx_kernel_desc desc;
    desc.file = "elementwise/token_mask_f32.loom";
    desc.symbol = "ggml_token_mask_f32";
    desc.config = { ggml_hrx_elementwise_wg_config(ctx->device), { "ggml.token_mask.ids", ids ? "1" : "0" } };
    auto * kernel = ggml_hrx_get_kernel(ctx->device, desc);
    return ggml_hrx_dispatch(ctx, kernel, { ggml_nelements(node) },
        { ggml_nelements(node), logits->ne[0], int64_t(logits->nb[1]/4), int64_t(ids_binding->nb[1]/4), int64_t(mask_desc->nb[1]/4),
          int64_t(ggml_nbytes(logits)/4), int64_t(ggml_nbytes(ids_binding)/4),
          int64_t(ggml_nbytes(mask_desc)/4), payload_size, std::max<int64_t>(payload_size, 1) },
        { logits, ids_binding, payload_binding, mask_desc, node });
}
