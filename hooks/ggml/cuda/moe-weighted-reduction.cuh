// hooks: ggml/src/ggml-cuda/moe-weighted-reduction.cuh of the fork (25b1c175e, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst,
                                         const ggml_tensor *         merge = nullptr);

// types/shapes/contiguity the fused kernel below needs (no data pointers, so graph_optimize can ask before allocation)
bool ggml_cuda_moe_weighted_reduction_sgma_supported(const ggml_tensor * experts, const ggml_tensor * expert_scale,
        const ggml_tensor * weights, const ggml_tensor * gate, const ggml_tensor * shexp, const ggml_tensor * dst);

// reduction + SIGMOID(gate [1, T]) -> MUL(shexp, .) -> ADD(reduction, .) in one kernel; false if the shapes/types don't fit
bool ggml_cuda_op_moe_weighted_reduction_sgma(ggml_backend_cuda_context & ctx, const ggml_tensor * experts,
        const ggml_tensor * expert_scale, const ggml_tensor * weights, const ggml_tensor * gate, const ggml_tensor * shexp,
        ggml_tensor * dst);
} // namespace ggml_fork
