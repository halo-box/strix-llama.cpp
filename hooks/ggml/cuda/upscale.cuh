// hooks: ggml/src/ggml-cuda/upscale.cuh of the fork (25b1c175e, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

#define CUDA_UPSCALE_BLOCK_SIZE 256

void ggml_cuda_op_upscale(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
} // namespace ggml_fork
