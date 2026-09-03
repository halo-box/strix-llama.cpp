#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

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
    float         eps_rms;
    float *       out;            // [S, H_v]
    float *       attn_out;       // [S, H_v] pre-norm scratch (set by the host)
    int64_t       S, H_k, H_v, d_conv;
};

void ggml_cuda_op_gdn_decode_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_decode_args & args);
