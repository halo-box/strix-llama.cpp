#pragma once

#include "common.cuh"

// Fused hyper-connection (DeepSeek-V4 / qwen4exp style) decode helpers.
//
// hc_mix_reduce:  mixed[e,t] = scale * sum_c ( xn[c*n_embd + e, t] * sigmoid(gate[c*n_embd + e, t]) ) + bias
//   replaces: UNARY(SIGMOID) -> MUL -> RESHAPE -> VIEW -> CONT -> (VIEW -> ADD) x (hc-1) -> SCALE
//
// hc_combine:     out[e,c,t] = residual[e,c,t] + block_out[e,t] * ( s2 * sigmoid(s1 * inject[c,t] + b1) + b2 )
//   replaces: SCALE -> UNARY(SIGMOID) -> SCALE -> RESHAPE* -> REPEAT -> MUL -> ADD
//
// Both replay the unfused rounding sequence exactly (separate mul and add roundings), so results are
// bit-identical to the individual kernels.

struct ggml_cuda_hc_mix_args {
    const ggml_tensor * xn;     // [hc*n_embd, T] F32 contiguous
    const ggml_tensor * gate;   // [hc*n_embd, T] F32 contiguous (pre-sigmoid)
    ggml_tensor *       dst;    // [n_embd, T]   F32 contiguous
    int                 hc;
    float               scale;
    float               bias;
};

struct ggml_cuda_hc_combine_args {
    const ggml_tensor * residual;   // [n_embd, hc, T] F32 contiguous
    const ggml_tensor * block_out;  // [n_embd, T]     F32 contiguous (or a reshape of it)
    const ggml_tensor * inject;     // [hc, T]         F32 contiguous (pre-scale/sigmoid)
    ggml_tensor *       dst;        // [n_embd, hc, T] F32 contiguous
    float               s1, b1;     // first SCALE
    float               s2, b2;     // second SCALE
};

void ggml_cuda_op_hc_mix_reduce(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_mix_args & args);
void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_combine_args & args);

// Hyper-connection combine followed by the grouped RMS norm of the next hc_mix (block per stream and token):
//   w[c]     = s2 * sigmoid(s1 * inject[c] + b1) + b2
//   res[e,c] = residual[e,c] + block_out[e] * w[c]        -> out_res (the layer residual)
//   xn[e,c]  = rms_norm(res[:,c]) * gamma[e,c]            -> out_xn  (next hc_norm)
// The norm replays rms_norm_f32<1024, true> (one block of 1024 threads per stream).
struct ggml_cuda_hc_combine_norm_args {
    const ggml_tensor * inject;     // [hc, T]  F32 contiguous (pre-scale/sigmoid)
    const ggml_tensor * residual;   // [n_embd, hc, T] F32 contiguous
    const ggml_tensor * block_out;  // [n_embd, 1, T]  F32 contiguous
    const ggml_tensor * gamma;      // [n_embd, hc] F32 contiguous
    ggml_tensor *       out_res;    // [n_embd, hc, T]
    ggml_tensor *       out_xn;     // [n_embd, hc, T]
    float               s1, b1, s2, b2;
    float               eps;
    bool                single_block = false; // alias-proof variant: one block per token, all inputs read before any store (hc <= 4)
    // BF16 stream options (HC16): a BF16 copy of xn for consumers that read it, and BF16 in-place residual / block_out
    uint16_t *          out_xn_bf16  = nullptr;
    bool                store_xn_f32 = true;     // false: every consumer reads the BF16 copy
    const uint16_t *    res_in_bf16  = nullptr;   // `residual` is BF16 in place (marked bf16-only)
    const uint16_t *    blk_in_bf16  = nullptr;
    uint16_t *          res_out_bf16 = nullptr;
    const ggml_tensor * w_inject     = nullptr;   // [hc * n_embd, hc] F32 weight
    ggml_tensor *       out_inject   = nullptr;   // [hc, T] F32, the skipped MUL_MAT node
};

bool ggml_cuda_hc_combine_norm_supported(const ggml_cuda_hc_combine_norm_args & args, int warp_size);
void ggml_cuda_op_hc_combine_norm(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_combine_norm_args & args);
