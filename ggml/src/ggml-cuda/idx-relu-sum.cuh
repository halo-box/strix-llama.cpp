#include "common.cuh"

struct ggml_cuda_idx_relu_sum_args {
    const ggml_tensor * score = nullptr;
    ggml_tensor *       dst   = nullptr;   // [n_blocks, n_tps, n_stream]
    int                 heads = 0;
    // optional fused compact visibility (single sequence): dst += log(step(float(tails[t]) - float(starts[b])))
    const ggml_tensor * vis_tails  = nullptr;   // I32 [n_tps] contiguous view of the QSA limits
    const ggml_tensor * vis_starts = nullptr;   // I32 [n_blocks] contiguous view of the QSA limits
    // nodes of the visibility chain that the matcher checked itself (views/casts of the limits input); a caller that
    // widens the fusion must leave them out of ggml_can_fuse_subgraph_ext
    const ggml_tensor * vis_leafs[16] = {};
};
void ggml_cuda_op_idx_relu_sum(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_relu_sum_args & args);
