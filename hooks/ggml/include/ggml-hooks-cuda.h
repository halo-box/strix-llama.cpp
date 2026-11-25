#pragma once

// CUDA, HIP and MUSA hook kernels. Include from .cu files built in the backend (kernels/cuda/).
//
// kernel signatures for GGML_HOOK_BACKEND_CUDA (ggml_hook_cuda_op_fn, ggml_hook_cuda_fusion_fn):
//   op:     bool kernel(ggml_backend_cuda_context & ctx, ggml_tensor * dst);  return false to let upstream compute the node
//   fusion: void kernel(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int n_nodes);
// launch on ctx.stream(). Upstream computes the nodes before and after on the same stream.

#include "ggml-hooks.h"
