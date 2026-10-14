#define mul_mat_q mul_mat_q_q6
#define launch_mul_mat_q launch_mul_mat_q_q6
#include "mmq.cuh"
#undef launch_mul_mat_q
#undef mul_mat_q

void launch_mul_mat_q_q6_k_j32(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    launch_mul_mat_q_q6<GGML_TYPE_Q6_K, 32, false>(ctx, args, stream);
}
