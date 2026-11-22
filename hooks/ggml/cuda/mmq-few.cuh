// hooks: ggml/src/ggml-cuda/mmq-few.cuh of the fork (1fe05fa45, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

} // namespace ggml_fork
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

// ids != nullptr: MUL_MAT_ID
bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, int cc);

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
} // namespace ggml_fork
