// hooks: ggml/src/ggml-cuda/opt-step-adamw.cuh of the fork (403dfcb03, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

#define CUDA_OPT_STEP_ADAMW_BLOCK_SIZE 256

void ggml_cuda_opt_step_adamw(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
} // namespace ggml_fork
