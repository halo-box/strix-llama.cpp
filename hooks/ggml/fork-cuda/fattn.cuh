// hooks: ggml/src/ggml-cuda/fattn.cuh of the fork (1fe05fa45, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/fork-cuda/README.md

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Gated FA: fa7 prefill attention with the qwen4exp output gate sigmoid(gate) * O in its epilogue.
// gate: element (d, hq, t) at gate->data + t*nb[2] + hq*nb[1] + d*4. Writes dst16 (16-bit copy for the MMB GEMM)
// and, when out32 != nullptr, the F32 gated output [D*HQ, T].
bool ggml_cuda_fa7_gate_ok(const ggml_backend_cuda_context & ctx, const ggml_tensor * fa);
bool ggml_cuda_qsa_gate_ok(const ggml_backend_cuda_context & ctx, const ggml_tensor * fa);
void ggml_cuda_flash_attn_ext_qsa_prefill_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * gate,
        float * out32, uint16_t * dst16, bool dst16_f16);   // HIP only (qsa-prefill.cu)
void ggml_cuda_fa7_gated(ggml_backend_cuda_context & ctx, ggml_tensor * fa, const ggml_tensor * gate, float * out32,
        uint16_t * dst16, bool dst16_f16);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);
} // namespace ggml_fork
