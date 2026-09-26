#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Chunked prefill (RDNA3.5): the q/k L2 norms that precede GATED_DELTA_NET (RMS_NORM(x, eps/n) -> SCALE(1/sqrt(n)),
// i.e. x * mul * rsqrt(sum(x^2) + eps) per 128-row) are folded into the chunk kernels. ggml_cuda_gdn_chunk_eligible
// is the exact dispatch condition of the chunked path for this GDN node with these q/k tensors; set_qk_norm registers
// the raw q/k views for the next evaluation of that node (consumed by it).
bool ggml_cuda_gdn_chunk_eligible(const ggml_tensor * gdn, const ggml_tensor * q, const ggml_tensor * k);
void ggml_cuda_gdn_set_qk_norm(const ggml_tensor * gdn, const ggml_tensor * q_raw, const ggml_tensor * k_raw, float eps, float mul);
// chunked prefill: the gated RMS norm of the output (RMS_NORM -> MUL(w) -> MUL(sigmoid(z))) written by the scan kernel
// straight into out (+ optional BF16 copy out16); z / out laid out like the GDN output rows [S_v, H, T]
void ggml_cuda_gdn_set_out_norm(const ggml_tensor * gdn, const float * w, const float * z, float * out, uint16_t * out16, float eps);
// chunked prefill: the causal conv + SiLU producing q/k/v is computed by the chunk kernels from the raw conv input
// x [T, C] (row stride C floats), history st [C, 3] and taps cw [C, 4]; qc/kc/vc = channel of element 0 of each view
void ggml_cuda_gdn_set_conv(const ggml_tensor * gdn, const float * x, const float * st, const float * cw,
        int64_t C, int64_t qc, int64_t kc, int64_t vc);
// ggml_cuda_gdn_chunk_eligible with the q/k views registered by set_qk_norm (if any)
bool ggml_cuda_gdn_chunk_eligible_node(const ggml_tensor * gdn);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// Fused Qwen3.5 Gated DeltaNet decode step (see ggml_cuda_try_gdn_decode_fusion). All strides in elements.
struct ggml_cuda_gdn_decode_args {
    // conv: conv_state_in [d_conv-1, C] (contiguous, per channel), qkv [C] with stride, conv_w [d_conv, C]
    const float * conv_state_in;
    const float * qkv;
    int64_t       qkv_stride;
    const float * conv_w;
    float         conv_bias;      // 0.0f (ssm_conv adds a runtime zero when there is no bias)
    float *       conv_state_out; // [d_conv-1, C] (contiguous) cache row
    // gate / beta
    const float * alpha;          // [H_v]
    const float * dt_bias;        // [H_v]
    const float * ssm_a;          // [H_v]
    const float * beta;           // [H_v]
    float         eps_l2;
    // recurrence
    const float *   state_cache;      // [S*S*H_v, n_slots] cache, row = state_ids[0] (transposed per head: [col][row])
    const int32_t * state_ids;
    int64_t         state_row_stride;
    float *       state_out;      // [S, S, H_v] cache row
    float         scale;          // 1/sqrt(S)
    // gated rms norm
    const float * norm_w;         // [S]
    const ggml_tensor * norm_w_tensor = nullptr;
    float         eps_rms;
    float *       out;            // [S, H_v]
    float *       attn_out;       // [S, H_v] pre-norm scratch (set by the host)
    int64_t       S, H_k, H_v, d_conv;
};

void ggml_cuda_op_gdn_decode_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_decode_args & args);
// first kernel only: writes the pre-norm attention output [S, H_v] to attn_scratch (the gated norm is applied by the consumer)
void ggml_cuda_op_gdn_decode_fused_prenorm(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_decode_args & args, float * attn_scratch);
