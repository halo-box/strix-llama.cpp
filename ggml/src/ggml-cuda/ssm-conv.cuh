#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr,
                           ggml_tensor * silu_dst = nullptr, ggml_tensor * mul_dst = nullptr, uint16_t * mul_dst16 = nullptr);
