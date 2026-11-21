// hooks: ggml/src/ggml-cuda/norm-gated.cuh of the fork (1fe05fa45, upstream 9048bad8d, new in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/fork-cuda/README.md
struct ggml_cuda_norm_gated_match {
    const ggml_tensor * x;
    const ggml_tensor * w;
    const ggml_tensor * z;
    ggml_tensor * dst;
    float eps;
    int pre = -1;
    bool staged = false;
    uint16_t * dst16 = nullptr;   // optional BF16 copy of dst for the MMB GEMM that reads it next
    bool dst16_f16 = false;       // dst16 holds F16 (saturating) instead: the reader is an MMB_DENSE_F16 GEMM
};
int  ggml_cuda_norm_gated_match_at(const ggml_cgraph * cgraph, int i, ggml_cuda_norm_gated_match & m);
void ggml_cuda_op_norm_gated(ggml_backend_cuda_context & ctx, const ggml_cuda_norm_gated_match & m);
int  ggml_cuda_norm_rows_match_at(const ggml_cgraph * cgraph, int i, ggml_cuda_norm_gated_match & m);
} // namespace ggml_fork
