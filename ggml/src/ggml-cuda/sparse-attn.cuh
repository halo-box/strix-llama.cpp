#pragma once

// HIP sparse attention (GGML_OP_SPARSE_ATTN_PLAN, GGML_OP_SPARSE_ATTN; contracts in ggml.h, PLAN-v3 §2, §6.2).
// Host code of every sparse-attn-*.cu file is HIP only: on CUDA builds supports_op returns false for both ops.

#include "common.cuh"

// GGML_OP_SPARSE_ATTN_PLAN (sparse-attn-plan.cu)
bool   ggml_cuda_sparse_attn_plan_supported(int device, const ggml_tensor * op);
// ggml_nbytes(op) for the plan's own G, rounded up to the largest dst over every G of GGML_CUDA_SA_PLAN_GROUPS, plus the
// sort/dedup scratch (ggml_sparse_attn_plan_scratch_nbytes), which the kernels place right after ggml_nbytes(op)
size_t ggml_cuda_sparse_attn_plan_get_alloc_size(const ggml_tensor * op);
void   ggml_cuda_sparse_attn_plan(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
