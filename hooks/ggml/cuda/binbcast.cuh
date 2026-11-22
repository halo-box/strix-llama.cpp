// hooks: ggml/src/ggml-cuda/binbcast.cuh of the fork (1fe05fa45, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_fused_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);
void ggml_cuda_op_fused_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);
void ggml_cuda_op_weighted_expert_sum(ggml_backend_cuda_context & ctx, ggml_tensor * mul, ggml_tensor * dst, int n_expert_used);
void ggml_cuda_op_shared_mul_add(ggml_backend_cuda_context & ctx, ggml_tensor * mul, const ggml_tensor * other, const ggml_tensor * residual, ggml_tensor * dst);

void ggml_cuda_op_shared_gate_mul_add(
        ggml_backend_cuda_context & ctx, const ggml_tensor * gate_mm, const ggml_tensor * src, const ggml_tensor * other,
        const ggml_tensor * residual, ggml_tensor * dst);
} // namespace ggml_fork
