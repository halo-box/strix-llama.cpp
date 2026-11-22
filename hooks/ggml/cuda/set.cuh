// hooks: ggml/src/ggml-cuda/set.cuh of the fork (3b76dee48, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

} // namespace ggml_fork
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

#define CUDA_SET_BLOCK_SIZE 256

void ggml_cuda_op_set(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
} // namespace ggml_fork
