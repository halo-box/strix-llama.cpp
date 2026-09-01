#define mul_mat_q mul_mat_q_q8
#define launch_mul_mat_q launch_mul_mat_q_q8
#include "mmq.cuh"
#undef launch_mul_mat_q
#undef mul_mat_q

void launch_mul_mat_q_q8_0_j48(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    launch_mul_mat_q_q8<GGML_TYPE_Q8_0, 48, false>(ctx, args, stream);
}
