#include "common.cuh"

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst,
                                         const ggml_tensor *         merge = nullptr);

// reduction + SIGMOID(gate [1, T]) -> MUL(shexp, .) -> ADD(reduction, .) in one kernel; false if the shapes/types don't fit
bool ggml_cuda_op_moe_weighted_reduction_sgma(ggml_backend_cuda_context & ctx, const ggml_tensor * experts,
        const ggml_tensor * expert_scale, const ggml_tensor * weights, const ggml_tensor * gate, const ggml_tensor * shexp,
        ggml_tensor * dst);
