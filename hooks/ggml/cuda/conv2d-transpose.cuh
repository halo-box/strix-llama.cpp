// hooks: ggml/src/ggml-cuda/conv2d-transpose.cuh of the fork (25b1c175e, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

#define CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE 256

void ggml_cuda_conv_2d_transpose_p0(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
} // namespace ggml_fork
