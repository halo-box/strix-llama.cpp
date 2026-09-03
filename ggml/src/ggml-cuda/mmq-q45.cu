#define mul_mat_q mul_mat_q_q45
#define launch_mul_mat_q launch_mul_mat_q_q45
#include "mmq.cuh"
#undef launch_mul_mat_q
#undef mul_mat_q

void launch_mul_mat_q_q5_k_j32(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    launch_mul_mat_q_q45<GGML_TYPE_Q5_K, 32, false>(ctx, args, stream);
}
