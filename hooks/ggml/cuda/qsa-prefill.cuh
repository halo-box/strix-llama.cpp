// hooks: ggml/src/ggml-cuda/qsa-prefill.cuh of the fork (25b1c175e, upstream 9048bad8d, new in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md
bool ggml_cuda_flash_attn_ext_qsa_prefill_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
bool ggml_cuda_flash_attn_ext_qsa_prefill_supported_dev(int device, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_qsa_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// sigmoid(gate) * attention in the epilogue; out32 / dst16 (either may be null) as ggml_cuda_fa7_gated
void ggml_cuda_flash_attn_ext_qsa_prefill_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * gate,
        float * out32, uint16_t * dst16, bool dst16_f16);
} // namespace ggml_fork
