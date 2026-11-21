// hooks: ggml/src/ggml-cuda/opt-step-sgd.cuh of the fork (1fe05fa45, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/fork-cuda/README.md

#define CUDA_OPT_STEP_SGD_BLOCK_SIZE 256

void ggml_cuda_opt_step_sgd(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
} // namespace ggml_fork
