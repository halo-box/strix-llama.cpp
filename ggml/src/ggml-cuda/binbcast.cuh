#include "common.cuh"

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
