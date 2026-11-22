// hooks: ggml/src/ggml-cuda/sum.cuh of the fork (1fe05fa45, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

void sum_f32_cuda(ggml_cuda_pool & pool, const float * x, float * dst, const int64_t ne, cudaStream_t stream);

void ggml_cuda_op_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
} // namespace ggml_fork
