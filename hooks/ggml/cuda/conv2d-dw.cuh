// hooks: ggml/src/ggml-cuda/conv2d-dw.cuh of the fork (403dfcb03, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

#define CUDA_CONV2D_DW_BLOCK_SIZE 256
void ggml_cuda_op_conv2d_dw(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
} // namespace ggml_fork
