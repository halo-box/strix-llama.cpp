// hooks: ggml/src/ggml-cuda/tsembd.cuh of the fork (1fe05fa45, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

#define CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE 256

void ggml_cuda_op_timestep_embedding(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
} // namespace ggml_fork
