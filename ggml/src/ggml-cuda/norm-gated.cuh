#pragma once
#include "common.cuh"
struct ggml_cuda_norm_gated_match {
    const ggml_tensor * x;
    const ggml_tensor * w;
    const ggml_tensor * z;
    ggml_tensor * dst;
    float eps;
    int pre = -1;
    bool staged = false;
    uint16_t * dst16 = nullptr;   // optional BF16 copy of dst for the MMB GEMM that reads it next
};
int  ggml_cuda_norm_gated_match_at(const ggml_cgraph * cgraph, int i, ggml_cuda_norm_gated_match & m);
void ggml_cuda_op_norm_gated(ggml_backend_cuda_context & ctx, const ggml_cuda_norm_gated_match & m);
int  ggml_cuda_norm_rows_match_at(const ggml_cgraph * cgraph, int i, ggml_cuda_norm_gated_match & m);
