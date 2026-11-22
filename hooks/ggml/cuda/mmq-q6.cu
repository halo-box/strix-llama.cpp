// hooks: ggml/src/ggml-cuda/mmq-q6.cu of the fork (25b1c175e, upstream 9048bad8d, new in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md
#define mul_mat_q mul_mat_q_q6
#define launch_mul_mat_q launch_mul_mat_q_q6
} // namespace ggml_fork
#include "mmq.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md
#undef launch_mul_mat_q
#undef mul_mat_q

void launch_mul_mat_q_q6_k_j32(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    launch_mul_mat_q_q6<GGML_TYPE_Q6_K, 32, false, GGML_PREC_Q8, false>(ctx, args, stream);
}
} // namespace ggml_fork
