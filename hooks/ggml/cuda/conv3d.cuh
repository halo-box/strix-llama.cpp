// hooks: ggml/src/ggml-cuda/conv3d.cuh of the fork (3b76dee48, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md
#ifndef GGML_CUDA_CONV3D_CUH
#define GGML_CUDA_CONV3D_CUH

} // namespace ggml_fork
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

void ggml_cuda_op_conv3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#endif
} // namespace ggml_fork
