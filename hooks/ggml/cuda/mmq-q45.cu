// hooks: ggml/src/ggml-cuda/mmq-q45.cu of the fork (1fe05fa45, upstream 9048bad8d, new in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md
#define mul_mat_q mul_mat_q_q45
#define launch_mul_mat_q launch_mul_mat_q_q45
} // namespace ggml_fork
#include "mmq.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md
#undef launch_mul_mat_q
#undef mul_mat_q

void launch_mul_mat_q_q5_k_j32(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    launch_mul_mat_q_q45<GGML_TYPE_Q5_K, 32, false, GGML_PREC_Q8, false>(ctx, args, stream);
}
} // namespace ggml_fork
