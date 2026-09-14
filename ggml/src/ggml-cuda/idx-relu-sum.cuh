#include "common.cuh"

struct ggml_cuda_idx_relu_sum_args {
    const ggml_tensor * score = nullptr;
    ggml_tensor *       dst   = nullptr;   // [n_blocks, n_tps, n_stream]
    int                 heads = 0;
};
void ggml_cuda_op_idx_relu_sum(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_relu_sum_args & args);
