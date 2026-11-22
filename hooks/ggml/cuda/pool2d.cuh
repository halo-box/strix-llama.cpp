// hooks: ggml/src/ggml-cuda/pool2d.cuh of the fork (3b76dee48, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

#define CUDA_POOL2D_BLOCK_SIZE 256
#define CUDA_POOL2D_WARP_KERNEL_MIN_WINDOW 32

void ggml_cuda_op_pool2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
} // namespace ggml_fork
