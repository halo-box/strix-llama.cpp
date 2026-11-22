// hooks: ggml/src/ggml-cuda/im2col.cuh of the fork (3b76dee48, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

#define CUDA_IM2COL_BLOCK_SIZE 256

void ggml_cuda_op_im2col(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_im2col_3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
} // namespace ggml_fork
