// hooks: ggml/src/ggml-cuda/mmid.cuh of the fork (1fe05fa45, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);

struct ggml_backend_cuda_context;
bool ggml_cuda_launch_mm_ids_bounded(ggml_backend_cuda_context & ctx,
        const int32_t * ids, int32_t * src_map, int32_t * dst_map, int32_t * bounds,
        int experts, int tokens, int used, int channels, int ids_stride, int token_stride,
        bool inverse, cudaStream_t stream);
} // namespace ggml_fork
