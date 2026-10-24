#include "gated_delta_net.cuh"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"
#include "unary.cuh"
#include "convert.cuh"

#include <cstdint>
#include <mutex>
#include <unordered_map>


// Warp reduction without the LDS crossbar: on RDNA the generic warp_reduce_sum compiles to 5 dependent
// ds_bpermute round trips (~100+ cycles each), which dominates the per-token latency of the recurrence.
// Same pairing/order of additions as warp_reduce_sum<32> (xor 16, 8, 4, 2, 1), so the result is bit-identical.
#if defined(GGML_USE_HIP) && (defined(RDNA3) || defined(RDNA4))
template <int mask>
static __device__ __forceinline__ float gdn_dpp_row_xmask(const float x) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x160 | mask, 0xf, 0xf, true));
}
static __device__ __forceinline__ float gdn_permlanex16_swap(const float x) {
    return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x), 0x76543210, 0xFEDCBA98, true, false));
}
static __device__ __forceinline__ float gdn_warp_reduce_sum32(float x) {
    x += gdn_permlanex16_swap(x);
    x += gdn_dpp_row_xmask<8>(x);
    x += gdn_dpp_row_xmask<4>(x);
    x += gdn_dpp_row_xmask<2>(x);
    x += gdn_dpp_row_xmask<1>(x);
    return x;
}
#define GDN_DPP_REDUCE 1
#endif // defined(GGML_USE_HIP) && (defined(RDNA3) || defined(RDNA4))

template <int width>
static __device__ __forceinline__ float gdn_warp_reduce_sum(const float x) {
#if defined(GDN_DPP_REDUCE)
    if constexpr (width == 32) {
        return gdn_warp_reduce_sum32(x);
    }
#endif // defined(GDN_DPP_REDUCE)
    return warp_reduce_sum<width>(x);
}

static constexpr int gated_delta_net_num_warps(int cc) {
    return GGML_CUDA_CC_IS_RDNA3_5(cc) ? 32 : 4;
}

static_assert(gated_delta_net_num_warps(GGML_CUDA_CC_RDNA3_5) == 32);
static_assert(gated_delta_net_num_warps(GGML_CUDA_CC_RDNA3) == 4);

#if defined(RDNA3_5)
static constexpr int gdn_num_warps = gated_delta_net_num_warps(GGML_CUDA_CC_RDNA3_5);
#else
static constexpr int gdn_num_warps = gated_delta_net_num_warps(GGML_CUDA_CC_RDNA3);
#endif

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * gdn_num_warps, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = gdn_warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = gdn_warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = gdn_warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = gdn_warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <int H>
__global__ void __launch_bounds__(512, 2)
gated_delta_net_kda_tiled_128_cuda(const float * q,
                                   const float * k,
                                   const float * v,
                                   const float * g,
                                   const float * beta,
                                   const float * curr_state,
                                   float *       dst,
                                   float *       state,
                                   int64_t       n_tokens,
                                   int64_t       sq1,
                                   int64_t       sq2,
                                   int64_t       sq3,
                                   int64_t       sv1,
                                   int64_t       sv2,
                                   int64_t       sv3,
                                   int64_t       sb1,
                                   int64_t       sb2,
                                   int64_t       sb3,
                                   const uint3   neqk1_magic,
                                   const uint3   rq3_magic,
                                   float         scale) {
    constexpr int S_v = 128;
    constexpr int token_tile = 16;
    constexpr int warp_size = 32;
    constexpr int rows_per_lane = S_v / warp_size;

    __shared__ float q_shared[token_tile][S_v];
    __shared__ float k_shared[token_tile][S_v];
    __shared__ float g_shared[token_tile][S_v];
    __shared__ float beta_shared[token_tile];

    const int h_idx = blockIdx.x;
    const int sequence = blockIdx.y;
    const int lane = threadIdx.x;
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    const int thread = threadIdx.y * warp_size + lane;
    const int nthreads = blockDim.y * warp_size;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_offset = (sequence * H + h_idx) * S_v * S_v;
    curr_state += state_offset + col * S_v;
    state += state_offset;
    dst += (sequence * n_tokens * H + h_idx) * S_v;

    float s_shard[rows_per_lane];
    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
        s_shard[r] = curr_state[r * warp_size + lane];
    }

    for (int t0 = 0; t0 < n_tokens; t0 += token_tile) {
        const int tile_size = min((int64_t) token_tile, n_tokens - t0);
        for (int idx = thread; idx < tile_size * S_v; idx += nthreads) {
            const int tt = idx / S_v;
            const int i = idx % S_v;
            const int t = t0 + tt;
            const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
            q_shared[tt][i] = q[iq3 * sq3 + t * sq2 + iq1 * sq1 + i];
            k_shared[tt][i] = k[iq3 * sq3 + t * sq2 + iq1 * sq1 + i];
            g_shared[tt][i] = g[gb_offset * S_v + i];
        }
        if (thread < tile_size) {
            beta_shared[thread] = beta[sequence * sb3 + (t0 + thread) * sb2 + h_idx * sb1];
        }
        __syncthreads();

        for (int tt = 0; tt < tile_size; ++tt) {
            const int t = t0 + tt;
            const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;
            float k_reg[rows_per_lane];
            float q_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                const int i = r * warp_size + lane;
                k_reg[r] = k_shared[tt][i];
                q_reg[r] = q_shared[tt][i];
            }

            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_shared[tt][i]) * s_shard[r] * k_reg[r];
            }
            const float kv_col = gdn_warp_reduce_sum<warp_size>(kv_shard);
            const float delta_col = (v_t[col] - kv_col) * beta_shared[tt];

            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                const int i = r * warp_size + lane;
                s_shard[r] = expf(g_shared[tt][i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }
            const float attn_col = gdn_warp_reduce_sum<warp_size>(attn_partial);
            if (lane == 0) {
                dst[(int64_t) t * S_v * H + col] = attn_col * scale;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
        state[col * S_v + r * warp_size + lane] = s_shard[r];
    }
}

// Non-KDA variant with the per-token inputs (q, k, g, beta, v) staged in shared memory for a tile of tokens,
// and COLS state columns per warp (k/q are loaded once per token for all columns, and the number of waves
// competing for issue slots shrinks). The recurrence itself is unchanged: for every column the same
// operations are executed in the same order as in gated_delta_net_cuda, so results are bit-identical.
template <int S_v, int NUM_WARPS, int COLS, int TOKEN_TILE, bool keep_rs_t>
__global__ void __launch_bounds__(32 * NUM_WARPS, 1)
gated_delta_net_tiled_cuda(const float * q,
                           const float * k,
                           const float * v,
                           const float * g,
                           const float * beta,
                           const float * curr_state,
                           float *       dst,
                           float *       state,
                           int64_t       H,
                           int64_t       n_tokens,
                           int64_t       sq1,
                           int64_t       sq2,
                           int64_t       sq3,
                           int64_t       sv1,
                           int64_t       sv2,
                           int64_t       sv3,
                           int64_t       sb1,
                           int64_t       sb2,
                           int64_t       sb3,
                           const uint3   neqk1_magic,
                           const uint3   rq3_magic,
                           float         scale,
                           int64_t       state_slot_stride,
                           int           K) {
    constexpr int warp_size     = 32;
    constexpr int rows_per_lane = S_v / warp_size;
    constexpr int block_cols    = NUM_WARPS * COLS;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of the warp size");
    static_assert(S_v % block_cols == 0, "block columns must divide S_v");

    __shared__ float q_shared[TOKEN_TILE][S_v];
    __shared__ float k_shared[TOKEN_TILE][S_v];
    __shared__ float v_shared[TOKEN_TILE][block_cols];
    __shared__ float g_shared[TOKEN_TILE];
    __shared__ float beta_shared[TOKEN_TILE];

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col0     = blockIdx.z * block_cols;          // first column of the block
    const int      colw     = threadIdx.y * COLS;               // first column of this warp inside the block
    const int      thread   = threadIdx.y * warp_size + lane;
    constexpr int  nthreads = NUM_WARPS * warp_size;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + (col0 + colw) * S_v;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v + col0 + colw;

    float s_shard[COLS][rows_per_lane];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int c = 0; c < COLS; c++) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[c][r] = curr_state[c * S_v + r * warp_size + lane];
        }
    }

    for (int t0 = 0; t0 < n_tokens; t0 += TOKEN_TILE) {
        const int tile_size = min((int64_t) TOKEN_TILE, n_tokens - t0);

        for (int idx = thread; idx < tile_size * S_v; idx += nthreads) {
            const int tt = idx / S_v;
            const int i  = idx % S_v;
            const int t  = t0 + tt;
            q_shared[tt][i] = q[iq3 * sq3 + t * sq2 + iq1 * sq1 + i];
            k_shared[tt][i] = k[iq3 * sq3 + t * sq2 + iq1 * sq1 + i];
        }
        for (int idx = thread; idx < tile_size * block_cols; idx += nthreads) {
            const int tt = idx / block_cols;
            const int c  = idx % block_cols;
            const int t  = t0 + tt;
            v_shared[tt][c] = v[sequence * sv3 + t * sv2 + h_idx * sv1 + col0 + c];
        }
        if (thread < tile_size) {
            const int64_t gb_offset = sequence * sb3 + (t0 + thread) * sb2 + h_idx * sb1;
            g_shared[thread]    = g[gb_offset];
            beta_shared[thread] = beta[gb_offset];
        }
        __syncthreads();

        for (int tt = 0; tt < tile_size; ++tt) {
            const int t = t0 + tt;

            float k_reg[rows_per_lane];
            float q_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                k_reg[r] = k_shared[tt][i];
                q_reg[r] = q_shared[tt][i];
            }

            const float g_val    = expf(g_shared[tt]);
            const float beta_val = beta_shared[tt];

            // The FMA contractions are spelled out so that the results are bit-identical to the
            // compiler's contraction of gated_delta_net_cuda (fma chains from 0, fma(-g, kv, v)*beta,
            // fma(g, s, k*delta)) independently of the surrounding code.
            float attn_col[COLS];
#pragma unroll
            for (int c = 0; c < COLS; c++) {
                // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard = fmaf(s_shard[c][r], k_reg[r], kv_shard);
                }
                const float kv_col = gdn_warp_reduce_sum<warp_size>(kv_shard);

                // delta[col] = (v[col] - g * kv[col]) * beta
                const float delta_col = fmaf(-g_val, kv_col, v_shared[tt][colw + c]) * beta_val;

                // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
                // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r] = fmaf(g_val, s_shard[c][r], k_reg[r] * delta_col);
                    attn_partial  = fmaf(s_shard[c][r], q_reg[r], attn_partial);
                }
                attn_col[c] = gdn_warp_reduce_sum<warp_size>(attn_partial);
            }

            if (lane < COLS) {
                float a = attn_col[0];
#pragma unroll
                for (int c = 1; c < COLS; c++) {
                    a = lane == c ? attn_col[c] : a;
                }
                attn_data[(int64_t) t * S_v * H + lane] = a * scale;
            }

            if constexpr (keep_rs_t) {
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * snapshot = state + target_slot * state_slot_stride + (col0 + colw) * S_v;
#pragma unroll
                    for (int c = 0; c < COLS; c++) {
#pragma unroll
                        for (int r = 0; r < rows_per_lane; r++) {
                            snapshot[c * S_v + r * warp_size + lane] = s_shard[c][r];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < COLS; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                state[(col0 + colw + c) * S_v + r * warp_size + lane] = s_shard[c][r];
            }
        }
    }
}


// ---------------------------------------------------------------------------------------------
// Chunked Gated DeltaNet prefill (RDNA3.5, S_v = 128, non-KDA, final state only, n_seqs == 1).
// Per chunk of CL tokens (G = in-chunk cumsum of g, D[r][s] = exp(G_r - G_s)):
//   A[r][s] = beta_r (k_r.k_s) D[r][s] (s < r),   T' = (I + A)^-1 diag(beta)
//   U  = T' (V - diag(e^G) K S0)                          (all delta-rule corrections, one product)
//   O  = scale diag(e^G) Q S0 + P U,   P[r][s] = scale (q_r.k_s) D[r][s] (s <= r)
//   S1 = e^{G_L} S0 + K^T diag(e^{G_L - G}) U
// gdn_chunk_prep_kernel: fully parallel, one block per (chunk, k-head) for its H/Hk value heads. Produces the chunk's
//     q/k (optionally through the fused causal conv + SiLU and the fused L2 norm) and v as fp16 rows, KK^T / QK^T on
//     WMMA, T by a barrier-free wave-level forward substitution, and writes T', P and the gate factors.
// gdn_chunk_scan_kernel: one block (8 waves) per value head, sequential over chunks; wave w owns 16 state columns
//     in fp32 WMMA accumulators. The chunk's q/k/T'/P are staged once per block in LDS (K^T transposed on the way in)
//     and the next chunk is prefetched into registers. Optionally applies the gated RMS norm of the output (fused).
// WMMA operands are fp16 with fp32 accumulation: not bit-identical to the sequential kernels (NMSE ~2e-7).
// GGML_GDN_CHUNK=0 disables the path (and every fusion that depends on it), GGML_GDN_CHUNK_MIN sets the token threshold.
#if defined(GGML_USE_HIP) && defined(RDNA3_5)
#define GDNC_ENABLED 1
#endif
#ifndef GDNC_SCAN_WAVES
#define GDNC_SCAN_WAVES 6
#endif
typedef _Float16 gdnc_v16h __attribute__((ext_vector_type(16)));
typedef float    gdnc_v8f  __attribute__((ext_vector_type(8)));

#if defined(GDNC_ENABLED)
#define GDNC_WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a), (b), (c))
// A/B operand fragment from a row-major fp16 matrix: lane holds row (lane & 15), 16 consecutive k
static __device__ __forceinline__ gdnc_v16h gdnc_frag(const _Float16 * base, const int stride, const int row0, const int k0, const int lane) {
    const _Float16 * p = base + (row0 + (lane & 15)) * stride + k0;
    gdnc_v16h f;
    const uint4 a = *(const uint4 *) p, b = *(const uint4 *) (p + 8);
    __builtin_memcpy(&f, &a, 16); __builtin_memcpy(((char *) &f) + 16, &b, 16);
    return f;
}
// same, from 16 consecutive fp32 values
static __device__ __forceinline__ gdnc_v16h gdnc_frag_f32(const float * p) {
    gdnc_v16h f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 x = ((const float4 *) p)[j];
        f[4 * j + 0] = (_Float16) x.x; f[4 * j + 1] = (_Float16) x.y;
        f[4 * j + 2] = (_Float16) x.z; f[4 * j + 3] = (_Float16) x.w;
    }
    return f;
}
// accumulator tile (lane: column lane&15, rows 2e + (lane>>4)) -> B fragment (lane: column lane&15, rows 0..15)
static __device__ __forceinline__ gdnc_v16h gdnc_acc_to_frag(const gdnc_v8f & acc, const int lane) {
    gdnc_v16h f;
    const bool hi = lane >= 16;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float own = acc[e];
        const float oth = __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(own), __float_as_int(own), 0x76543210, 0xFEDCBA98, true, false));
        f[2 * e]     = (_Float16) (hi ? oth : own);
        f[2 * e + 1] = (_Float16) (hi ? own : oth);
    }
    return f;
}
#endif

// inputs of the chunk kernels: the q/k/v views of the GDN op, or (fused conv) the raw conv input x [T, C] with the
// conv history [C, 3] and taps [C, 4]: q/k/v element (t, c) is then silu(sum_j w[c][j] * in(t - 3 + j, c)), the
// ssm_conv_f32 + SiLU arithmetic of gdn_conv_direct_kernel
struct gdnc_in {
    const float * q;  const float * k;  const float * v;
    int64_t       sq1, sq2, sv1, sv2;
    const float * x;  const float * st; const float * cw;   // x == nullptr: read the views
    int64_t       C, qc, kc, vc;                            // x row stride, channel of element 0 of the q / k / v views
};

#if defined(GDNC_ENABLED)
// channels c..c+3 of token t through the causal conv + SiLU; wt[ch][j] = tap j of channel c + ch
static __device__ __forceinline__ float4 gdnc_conv4(const gdnc_in & in, const float (&wt)[4][4], const int64_t t, const int64_t c) {
    float s[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int64_t u = t - 3 + j;
        float xv[4];
        if (u >= 0) {
            const float4 v4 = *(const float4 *) (in.x + u * in.C + c);
            xv[0] = v4.x; xv[1] = v4.y; xv[2] = v4.z; xv[3] = v4.w;
        } else {
#pragma unroll
            for (int ch = 0; ch < 4; ++ch) xv[ch] = in.st[(c + ch) * 3 + (u + 3)];
        }
#pragma unroll
        for (int ch = 0; ch < 4; ++ch) s[ch] = fmaf(xv[ch], wt[ch][j], s[ch]);
    }
    float4 o;
    o.x = ggml_cuda_op_silu_single(s[0] + 0.0f); o.y = ggml_cuda_op_silu_single(s[1] + 0.0f);
    o.z = ggml_cuda_op_silu_single(s[2] + 0.0f); o.w = ggml_cuda_op_silu_single(s[3] + 0.0f);
    return o;
}
static __device__ __forceinline__ void gdnc_load_taps(const gdnc_in & in, const int64_t c, float (&wt)[4][4]) {
#pragma unroll
    for (int ch = 0; ch < 4; ++ch) {
        const float4 w4 = *(const float4 *) (in.cw + (c + ch) * 4);
        wt[ch][0] = w4.x; wt[ch][1] = w4.y; wt[ch][2] = w4.z; wt[ch][3] = w4.w;
    }
}
static __device__ __forceinline__ uint2 gdnc_pack4h(const float4 v) {
    const _Float16 h[4] = { (_Float16) v.x, (_Float16) v.y, (_Float16) v.z, (_Float16) v.w };
    uint2 u;
    __builtin_memcpy(&u, h, 8);
    return u;
}
#endif

// K1: one block per (chunk, k-head), handling the H/Hk value heads that share it. Produces the (normalized) q/k and v
// of the chunk as fp16 rows (Qh/Kh [Hk, T, 128], Vh [H, T, 128]) for the scan, plus T', P and the gate factors.
template <int CL>
__global__ void __launch_bounds__(256)
gdn_chunk_prep_kernel(const gdnc_in in, const float * __restrict__ g,
        const float * __restrict__ beta, _Float16 * __restrict__ Tb, _Float16 * __restrict__ Pb, float2 * __restrict__ Gb,
        _Float16 * __restrict__ Qh, _Float16 * __restrict__ Kh, _Float16 * __restrict__ Vh,
        const int H, const int Hk, const int n_tokens, const int nch,
        const int64_t sb1, const int64_t sb2, const float scale,
        const float qk_eps, const float qk_mul) {
#if defined(GDNC_ENABLED)
    constexpr int SK   = 136;
    constexpr int NT   = CL / 16;
    constexpr int NTRI = NT * (NT + 1) / 2;
    constexpr int MR   = CL / 32;
    constexpr int CPW  = CL / 8;
    constexpr int NQK  = CL * 32 / 256;
    constexpr int RMAX = 4;
    __shared__ __align__(16) _Float16 Ks[CL * SK];
    __shared__ __align__(16) _Float16 Qs[CL * SK];
    __shared__ float KK[CL][CL + 1];
    __shared__ float QK[CL][CL + 1];
    __shared__ float A[CL][CL + 1];
    __shared__ float Gs[RMAX][CL], bs[RMAX][CL];
    const int hk = blockIdx.x, ch = blockIdx.y, tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int cl = lane & 15, rh = lane >> 4;
    const int R = H / Hk;
    const int t0 = ch * CL, Lc = min(CL, n_tokens - t0);
    const int i4 = (tid & 31) * 4;   // this thread's 4 channels of every row it handles (row = (idx >> 5))

    {
        float4 kv[NQK], qv[NQK];
        if (in.x) {
            float wq[4][4], wk[4][4];
            gdnc_load_taps(in, in.qc + (int64_t) hk * in.sq1 + i4, wq);
            gdnc_load_taps(in, in.kc + (int64_t) hk * in.sq1 + i4, wk);
#pragma unroll
            for (int j = 0; j < NQK; ++j) {
                const int r = (tid + 256 * j) >> 5;
                const int64_t t = t0 + min(r, Lc - 1);
                qv[j] = gdnc_conv4(in, wq, t, in.qc + (int64_t) hk * in.sq1 + i4);
                kv[j] = gdnc_conv4(in, wk, t, in.kc + (int64_t) hk * in.sq1 + i4);
            }
        } else {
#pragma unroll
            for (int j = 0; j < NQK; ++j) {
                const int r = (tid + 256 * j) >> 5;
                const int64_t o = (int64_t) (t0 + min(r, Lc - 1)) * in.sq2 + (int64_t) hk * in.sq1 + i4;
                kv[j] = *(const float4 *) (in.k + o);
                qv[j] = *(const float4 *) (in.q + o);
            }
        }
        if (qk_eps >= 0.0f) {
            // fused q/k L2 norm: a row's 128 values are the float4s of one wave (row = idx >> 5)
#pragma unroll
            for (int j = 0; j < NQK; ++j) {
                const float sk = gdn_warp_reduce_sum<32>(kv[j].x * kv[j].x + kv[j].y * kv[j].y + kv[j].z * kv[j].z + kv[j].w * kv[j].w);
                const float sq = gdn_warp_reduce_sum<32>(qv[j].x * qv[j].x + qv[j].y * qv[j].y + qv[j].z * qv[j].z + qv[j].w * qv[j].w);
                const float fk = qk_mul * rsqrtf(sk + qk_eps), fq = qk_mul * rsqrtf(sq + qk_eps);
                kv[j].x *= fk; kv[j].y *= fk; kv[j].z *= fk; kv[j].w *= fk;
                qv[j].x *= fq; qv[j].y *= fq; qv[j].z *= fq; qv[j].w *= fq;
            }
        }
#pragma unroll
        for (int j = 0; j < NQK; ++j) {
            const int r = (tid + 256 * j) >> 5;
            const bool ok = r < Lc;
            const float4 z4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const uint2 kh = gdnc_pack4h(ok ? kv[j] : z4), qh = gdnc_pack4h(ok ? qv[j] : z4);
            *(uint2 *) (Ks + r * SK + i4) = kh;
            *(uint2 *) (Qs + r * SK + i4) = qh;
            if (ok) {
                const int64_t o = ((int64_t) hk * n_tokens + t0 + r) * 128 + i4;
                *(uint2 *) (Kh + o) = kh;
                *(uint2 *) (Qh + o) = qh;
            }
        }
        // v of the R value heads of this k-head -> Vh (fp16)
        for (int jh = 0; jh < R; ++jh) {
            const int h = hk + jh * Hk;
            float wv[4][4];
            if (in.x) gdnc_load_taps(in, in.vc + (int64_t) h * in.sv1 + i4, wv);
#pragma unroll
            for (int j = 0; j < NQK; ++j) {
                const int r = (tid + 256 * j) >> 5;
                if (r >= Lc) continue;
                const int64_t t = t0 + r;
                const float4 vv = in.x ? gdnc_conv4(in, wv, t, in.vc + (int64_t) h * in.sv1 + i4)
                                       : *(const float4 *) (in.v + t * in.sv2 + (int64_t) h * in.sv1 + i4);
                *(uint2 *) (Vh + ((int64_t) h * n_tokens + t) * 128 + i4) = gdnc_pack4h(vv);
            }
        }
    }
    for (int idx = tid; idx < R * CL; idx += 256) {
        const int j = idx / CL, r = idx % CL, h = hk + j * Hk;
        const int64_t o = (int64_t) (t0 + r) * sb2 + (int64_t) h * sb1;
        Gs[j][r] = r < Lc ? g[o]    : 0.0f;
        bs[j][r] = r < Lc ? beta[o] : 0.0f;
    }
    __syncthreads();
    if (tid < R) {
        float gv[CL];
#pragma unroll
        for (int r = 0; r < CL; ++r) gv[r] = Gs[tid][r];
#pragma unroll
        for (int r = 1; r < CL; ++r) gv[r] += gv[r - 1];
#pragma unroll
        for (int r = 0; r < CL; ++r) Gs[tid][r] = gv[r];
    }
    // raw lower-triangular tiles of K K^T and Q K^T (shared by all value heads of this k-head)
    for (int job = wave; job < 2 * NTRI; job += 8) {
        const bool isq = job >= NTRI;
        int jj = isq ? job - NTRI : job, rt = 0;
        while (jj >= rt + 1) { jj -= rt + 1; ++rt; }
        const int st = jj;
        const _Float16 * As = isq ? Qs : Ks;
        gdnc_v8f acc = {};
#pragma unroll
        for (int kk = 0; kk < 128; kk += 16) {
            acc = GDNC_WMMA(gdnc_frag(As, SK, rt * 16, kk, lane), gdnc_frag(Ks, SK, st * 16, kk, lane), acc);
        }
        float (*D)[CL + 1] = isq ? QK : KK;
#pragma unroll
        for (int e = 0; e < 8; ++e) D[rt * 16 + 2 * e + rh][st * 16 + cl] = acc[e];
    }
    __syncthreads();

    for (int j = 0; j < R; ++j) {
        const int h = hk + j * Hk;
        _Float16 * Pp = Pb + ((int64_t) h * nch + ch) * CL * CL;
        for (int idx = tid; idx < CL * CL; idx += 256) {
            const int r = idx / CL, s = idx % CL;
            const float d = s <= r ? expf(Gs[j][r] - Gs[j][s]) : 0.0f;
            A[r][s] = s < r ? bs[j][r] * KK[r][s] * d : 0.0f;
            Pp[idx] = (_Float16) (s <= r ? scale * QK[r][s] * d : 0.0f);
        }
        if (tid < CL) {
            Gb[((int64_t) h * nch + ch) * CL + tid] = make_float2(expf(Gs[j][tid]), expf(Gs[j][CL - 1] - Gs[j][tid]));
        }
        __syncthreads();
        // T = (I + A)^-1 column by column, barrier-free: lane owns rows lane + 32 m, the wave CPW columns
        {
            const int j0 = wave * CPW;
            float x[MR][CPW];
#pragma unroll
            for (int m = 0; m < MR; ++m)
#pragma unroll
                for (int c = 0; c < CPW; ++c) x[m][c] = (lane + 32 * m == j0 + c) ? 1.0f : 0.0f;
#pragma unroll
            for (int qq = 0; qq < CL - 1; ++qq) {
                if (qq < j0) continue;
                float a[MR];
#pragma unroll
                for (int m = 0; m < MR; ++m) a[m] = A[lane + 32 * m][qq];
#pragma unroll
                for (int c = 0; c < CPW; ++c) {
                    const float xq = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x[qq / 32][c]), qq % 32));
#pragma unroll
                    for (int m = 0; m < MR; ++m) x[m][c] = fmaf(-a[m], xq, x[m][c]);
                }
            }
            _Float16 * Tp = Tb + ((int64_t) h * nch + ch) * CL * CL;
#pragma unroll
            for (int m = 0; m < MR; ++m) {
                const int r = lane + 32 * m;
                _Float16 tmp[CPW];
#pragma unroll
                for (int c = 0; c < CPW; ++c) tmp[c] = (_Float16) (x[m][c] * bs[j][j0 + c]);
                __builtin_memcpy(Tp + r * CL + j0, tmp, sizeof(tmp));
            }
        }
        __syncthreads();
    }
#else
    NO_DEVICE_CODE;
#endif
}

// K2: one block (8 waves) per value head, wave w owns state columns [16 w, 16 w + 16). The chunk's q/k/T'/P are
// staged once per block in LDS (K^T transposed on the way in); the next chunk is prefetched into registers.
template <int CL>
__global__ void __launch_bounds__(256, GDNC_SCAN_WAVES)
gdn_chunk_scan_kernel(const _Float16 * __restrict__ Qh, const _Float16 * __restrict__ Kh, const _Float16 * __restrict__ Vh,
        const float * __restrict__ curr_state, float * __restrict__ dst, float * __restrict__ state,
        const _Float16 * __restrict__ Tb, const _Float16 * __restrict__ Pb, const float2 * __restrict__ Gb,
        const int H, const int Hk, const int n_tokens, const int nch, const float scale,
        const float * __restrict__ on_w, const float * __restrict__ on_z, float * __restrict__ on_out, uint16_t * __restrict__ on_out16,
        const float on_eps, const bool on_out16_f16) {
#if defined(GDNC_ENABLED)
    constexpr int NT  = CL / 16;
    constexpr int SQ  = 136;
    constexpr int ST  = CL + 8;
    constexpr int NQK = CL * 32 / 256;            // 4-half groups of q (and of k) per thread per chunk
    constexpr int NTP = 2 * CL * CL / 8 / 256;         // uint4 of T'/P per thread per chunk
    static_assert(NTP * 256 == 2 * CL * CL / 8, "T'/P staging must divide evenly");
    __shared__ __align__(16) _Float16 Qs[CL * SQ];
    __shared__ __align__(16) _Float16 Ks[CL * SQ];
    __shared__ __align__(16) _Float16 KTs[128 * ST];
    __shared__ __align__(16) _Float16 Ts[CL * ST];
    __shared__ __align__(16) _Float16 Ps[CL * ST];
    __shared__ float2 Gs[CL];
    __shared__ float  red[CL][9];   // fused gated norm: per-row partial sums of squares of the 8 waves
    const int h = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int hk = h % Hk;
    const int cl = lane & 15, rh = lane >> 4, col = wave * 16 + cl;

    const float * s_in = curr_state + (int64_t) h * 128 * 128;
    gdnc_v8f S[8];
#pragma unroll
    for (int it = 0; it < 8; ++it)
#pragma unroll
        for (int e = 0; e < 8; ++e) S[it][e] = s_in[col * 128 + it * 16 + 2 * e + rh];

    const _Float16 * kh = Kh + (int64_t) hk * n_tokens * 128;   // prep's fp16 rows: normalized q / k, v
    const _Float16 * qh = Qh + (int64_t) hk * n_tokens * 128;
    const _Float16 * vh = Vh + (int64_t) h * n_tokens * 128 + col;

    uint2 pk[NQK], pq[NQK];
    uint4  ptp[NTP];
    float2 pg = make_float2(0.0f, 0.0f);
    float  pv[NT][8];
    float  pz[NT][8];   // fused output norm: the gate z of this lane's rows / column, prefetched with v
    const float * zh = on_z ? on_z + (int64_t) h * 128 + col : nullptr;
    auto prefetch = [&](const int ch) {
        const int t0 = ch * CL;
#pragma unroll
        for (int j = 0; j < NQK; ++j) {
            const int idx = tid + 256 * j, r = idx >> 5, i = (idx & 31) * 4;
            const int64_t o = (int64_t) min(t0 + r, n_tokens - 1) * 128 + i;
            pk[j] = *(const uint2 *) (kh + o);
            pq[j] = *(const uint2 *) (qh + o);
        }
        const _Float16 * Tp = Tb + ((int64_t) h * nch + ch) * CL * CL;
        const _Float16 * Pp = Pb + ((int64_t) h * nch + ch) * CL * CL;
#pragma unroll
        for (int j = 0; j < NTP; ++j) {
            // 2 CL^2 / 8 is a multiple of 256 for CL = 32 / 64: no guard (a divergent guard forces vmcnt(0) at the join)
            const int idx = tid + 256 * j;
            const int w = idx % (CL * CL / 8);
            ptp[j] = *(const uint4 *) ((idx < CL * CL / 8 ? Tp : Pp) + w * 8);
        }
        pg = Gb[((int64_t) h * nch + ch) * CL + (tid % CL)];
    };
    // v / z of a chunk: issued right after the previous chunk consumed its own (one register copy each)
    auto prefetch_v = [&](const int ch) {
        const int t0 = ch * CL;
#pragma unroll
        for (int rt = 0; rt < NT; ++rt)
#pragma unroll
            for (int e = 0; e < 8; ++e) pv[rt][e] = (float) vh[(int64_t) min(t0 + rt * 16 + 2 * e + rh, n_tokens - 1) * 128];
    };
    auto prefetch_z = [&](const int ch) {
        const int t0 = ch * CL;
#pragma unroll
        for (int rt = 0; rt < NT; ++rt)
#pragma unroll
            for (int e = 0; e < 8; ++e) pz[rt][e] = zh[(int64_t) min(t0 + rt * 16 + 2 * e + rh, n_tokens - 1) * H * 128];
    };
    prefetch(0);
    prefetch_v(0);
    if (zh) prefetch_z(0);

    for (int ch = 0; ch < nch; ++ch) {
        const int t0 = ch * CL;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NQK; ++j) {
            const int idx = tid + 256 * j, r = idx >> 5, i = (idx & 31) * 4;
            *(uint2 *) (Ks + r * SQ + i) = pk[j];
            *(uint2 *) (Qs + r * SQ + i) = pq[j];
            _Float16 kk4[4];
            __builtin_memcpy(kk4, &pk[j], 8);
            KTs[(i + 0) * ST + r] = kk4[0]; KTs[(i + 1) * ST + r] = kk4[1];
            KTs[(i + 2) * ST + r] = kk4[2]; KTs[(i + 3) * ST + r] = kk4[3];
        }
#pragma unroll
        for (int j = 0; j < NTP; ++j) {
            const int idx = tid + 256 * j;
            const int w = idx % (CL * CL / 8), r = w / (CL / 8), c8 = (w % (CL / 8)) * 8;
            *(uint4 *) ((idx < CL * CL / 8 ? Ts : Ps) + r * ST + c8) = ptp[j];
        }
        if (tid < CL) Gs[tid] = pg;
        __syncthreads();
        // unconditional (clamped): a branch here makes the compiler drain the prefetch before the compute
        prefetch(min(ch + 1, nch - 1));

        // K S0, Q S0
        gdnc_v8f ks[NT], qs[NT];
#pragma unroll
        for (int rt = 0; rt < NT; ++rt) { ks[rt] = gdnc_v8f{}; qs[rt] = gdnc_v8f{}; }
#pragma unroll
        for (int it = 0; it < 8; ++it) {
            const gdnc_v16h sb = gdnc_acc_to_frag(S[it], lane);
#pragma unroll
            for (int rt = 0; rt < NT; ++rt) {
                ks[rt] = GDNC_WMMA(gdnc_frag(Ks, SQ, rt * 16, it * 16, lane), sb, ks[rt]);
                qs[rt] = GDNC_WMMA(gdnc_frag(Qs, SQ, rt * 16, it * 16, lane), sb, qs[rt]);
            }
        }
        // U = T' (V - diag(e^G) K S0)
        gdnc_v16h YB[NT];
#pragma unroll
        for (int rt = 0; rt < NT; ++rt) {
            gdnc_v8f y;
#pragma unroll
            for (int e = 0; e < 8; ++e) y[e] = fmaf(-Gs[rt * 16 + 2 * e + rh].x, ks[rt][e], pv[rt][e]);
            YB[rt] = gdnc_acc_to_frag(y, lane);
        }
        prefetch_v(min(ch + 1, nch - 1));
        gdnc_v8f U[NT];
        gdnc_v16h UB[NT];
#pragma unroll
        for (int rt = 0; rt < NT; ++rt) {
            gdnc_v8f u = {};
#pragma unroll
            for (int st = 0; st <= rt; ++st) u = GDNC_WMMA(gdnc_frag(Ts, ST, rt * 16, st * 16, lane), YB[st], u);
            U[rt]  = u;
            UB[rt] = gdnc_acc_to_frag(u, lane);
        }
        // S1 = e^{G_L} S0 + K^T diag(e^{G_L - G}) U
        gdnc_v16h UB2[NT];
#pragma unroll
        for (int st = 0; st < NT; ++st) {
            gdnc_v8f u;
#pragma unroll
            for (int e = 0; e < 8; ++e) u[e] = U[st][e] * Gs[st * 16 + 2 * e + rh].y;
            UB2[st] = gdnc_acc_to_frag(u, lane);
        }
        const float eGL = Gs[CL - 1].x;
#pragma unroll
        for (int it = 0; it < 8; ++it) {
            gdnc_v8f acc;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] = S[it][e] * eGL;
#pragma unroll
            for (int st = 0; st < NT; ++st) acc = GDNC_WMMA(gdnc_frag(KTs, ST, it * 16, st * 16, lane), UB2[st], acc);
            S[it] = acc;
        }
        // O = scale diag(e^G) Q S0 + P U
        gdnc_v8f O[NT];
#pragma unroll
        for (int rt = 0; rt < NT; ++rt) {
            gdnc_v8f o;
#pragma unroll
            for (int e = 0; e < 8; ++e) o[e] = qs[rt][e] * (Gs[rt * 16 + 2 * e + rh].x * scale);
#pragma unroll
            for (int st = 0; st <= rt; ++st) o = GDNC_WMMA(gdnc_frag(Ps, ST, rt * 16, st * 16, lane), UB[st], o);
            O[rt] = o;
        }
        if (on_z == nullptr) {
#pragma unroll
            for (int rt = 0; rt < NT; ++rt)
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int t = t0 + rt * 16 + 2 * e + rh;
                    if (t < n_tokens) dst[((int64_t) t * H + h) * 128 + col] = O[rt][e];
                }
        } else {
            // fused gated RMS norm (rms_rows_f32<true>): v = rsqrt(mean(o^2) + eps) * o * w * sigmoid(z)
#pragma unroll
            for (int rt = 0; rt < NT; ++rt)
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    float p = O[rt][e] * O[rt][e];
                    p += __shfl_xor(p, 1, 32); p += __shfl_xor(p, 2, 32); p += __shfl_xor(p, 4, 32); p += __shfl_xor(p, 8, 32);
                    if (cl == 0) red[rt * 16 + 2 * e + rh][wave] = p;
                }
            __syncthreads();
            const float wc = on_w[col];
#pragma unroll
            for (int rt = 0; rt < NT; ++rt)
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int r = rt * 16 + 2 * e + rh, t = t0 + r;
                    float ss = 0.0f;
#pragma unroll
                    for (int w8 = 0; w8 < 8; ++w8) ss += red[r][w8];
                    if (t < n_tokens) {
                        const int64_t oi = ((int64_t) t * H + h) * 128 + col;
                        const float sc = rsqrtf(ss / 128.0f + on_eps);
                        const float v  = (sc * O[rt][e] * wc) * (1.0f / (1.0f + expf(-pz[rt][e])));
                        if (on_out) on_out[oi] = v;
                        if (on_out16) on_out16[oi] = ggml_cuda_f32_to_act16(v, on_out16_f16);
                    }
                }
            prefetch_z(min(ch + 1, nch - 1));
        }
    }

    float * s_out = state + (int64_t) h * 128 * 128;
#pragma unroll
    for (int it = 0; it < 8; ++it)
#pragma unroll
        for (int e = 0; e < 8; ++e) s_out[col * 128 + it * 16 + 2 * e + rh] = S[it][e];
#else
    NO_DEVICE_CODE;
#endif
}

// ---------------------------------------------------------------------------------------------
// GDN chunked prefill v2 (verified against an fp64 reference):
// prep2 (q/k only) + scan2 VROW (v conv + SiLU inside the scan, no Vh round trip). Used when the conv, q/k L2 norm
// and gated output norm (bf16 out only) are all fused; GGML_GDN_V2=0 restores prep + scan.
#if defined(GDNC_ENABLED)
#define S2_BAR() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")

// accumulator tile (lane: column lane&15, rows 2e + (lane>>4)) -> B fragment (lane: column lane&15, rows 0..15)
static __device__ __forceinline__ gdnc_v16h s2_acc_to_frag(const gdnc_v8f & acc, const int lane) {
    const bool hi = lane >= 16;
    const uint32_t s0 = hi ? 0x01000504u : 0x05040100u, s1 = hi ? 0x03020706u : 0x07060302u;
    uint32_t w[8];
#pragma unroll
    for (int e = 0; e < 8; e += 2) {
        const _Float16 h2[2] = { (_Float16) acc[e], (_Float16) acc[e + 1] };
        uint32_t A; __builtin_memcpy(&A, h2, 4);
        const uint32_t B = __builtin_amdgcn_permlanex16(A, A, 0x76543210, 0xFEDCBA98, true, false);
        w[e]     = __builtin_amdgcn_perm(B, A, s0);
        w[e + 1] = __builtin_amdgcn_perm(B, A, s1);
    }
    gdnc_v16h f;
    __builtin_memcpy(&f, w, 32);
    return f;
}
static __device__ __forceinline__ float s2_silu(float x) { return x * __builtin_amdgcn_rcpf(1.0f + __expf(-x)); }
static __device__ __forceinline__ float s2_sigm(float x) { return __builtin_amdgcn_rcpf(1.0f + __expf(-x)); }

// conv + SiLU of 4 channels of token t from an XT-typed x
template <typename XT>
static __device__ __forceinline__ float4 s2_conv4(const XT * x, const float * st, const int64_t C, const float (&wt)[4][4], const int64_t t, const int64_t c) {
    float s[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int64_t u = t - 3 + j;
        float xv[4];
        if (u >= 0) {
            if constexpr (sizeof(XT) == 4) {
                const float4 v4 = *(const float4 *) (x + u * C + c);
                xv[0] = v4.x; xv[1] = v4.y; xv[2] = v4.z; xv[3] = v4.w;
            } else {
                const uint2 v2 = *(const uint2 *) (x + u * C + c);
                _Float16 hv[4]; __builtin_memcpy(hv, &v2, 8);
                xv[0] = hv[0]; xv[1] = hv[1]; xv[2] = hv[2]; xv[3] = hv[3];
            }
        } else {
#pragma unroll
            for (int ch = 0; ch < 4; ++ch) xv[ch] = st[(c + ch) * 3 + (u + 3)];
        }
#pragma unroll
        for (int ch = 0; ch < 4; ++ch) s[ch] = fmaf(xv[ch], wt[ch][j], s[ch]);
    }
    return make_float4(s2_silu(s[0]), s2_silu(s[1]), s2_silu(s[2]), s2_silu(s[3]));
}
#endif // GDNC_ENABLED

template <int CL, typename XT = float>
__global__ void __launch_bounds__(256)
gdn_prep2_kernel(const gdnc_in in, const XT * __restrict__ xin, const float * __restrict__ g,
        const float * __restrict__ beta, _Float16 * __restrict__ Tb, _Float16 * __restrict__ Pb, float2 * __restrict__ Gb,
        _Float16 * __restrict__ Qh, _Float16 * __restrict__ Kh,
        const int H, const int Hk, const int n_tokens, const int nch,
        const int64_t sb1, const int64_t sb2, const float scale, const float qk_eps, const float qk_mul) {
#if defined(GDNC_ENABLED)
    constexpr int SK   = 136;
    constexpr int NT   = CL / 16;
    constexpr int NTRI = NT * (NT + 1) / 2;
    constexpr int MR   = CL / 32;
    constexpr int CPW  = CL / 8;
    constexpr int NQK  = CL * 32 / 256;
    constexpr int RMAX = 4;
    __shared__ __align__(16) _Float16 KQs[2 * CL * SK];   // Ks, Qs (until the KK^T / QK^T jobs) | A (after)
    _Float16 * Ks = KQs, * Qs = KQs + CL * SK;
    float (*A)[CL][CL + 1] = (float (*)[CL][CL + 1]) KQs;
    static_assert(RMAX * CL * (CL + 1) * 4 <= 2 * CL * SK * 2, "A alias");
    __shared__ float KK[CL][CL + 1];
    __shared__ float QK[CL][CL + 1];
    __shared__ float Gs[RMAX][CL], bs[RMAX][CL];
    const int hk = blockIdx.x, ch = blockIdx.y, tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int cl = lane & 15, rh = lane >> 4;
    const int R = H / Hk;
    const int t0 = ch * CL, Lc = min(CL, n_tokens - t0);
    const int i4 = (tid & 31) * 4;
    {
        float4 kv[NQK], qv[NQK];
        float wq[4][4], wk[4][4];
        gdnc_load_taps(in, in.qc + (int64_t) hk * in.sq1 + i4, wq);
        gdnc_load_taps(in, in.kc + (int64_t) hk * in.sq1 + i4, wk);
#pragma unroll
        for (int j = 0; j < NQK; ++j) {
            const int r = (tid + 256 * j) >> 5;
            const int64_t t = t0 + min(r, Lc - 1);
            qv[j] = s2_conv4<XT>(xin, in.st, in.C, wq, t, in.qc + (int64_t) hk * in.sq1 + i4);
            kv[j] = s2_conv4<XT>(xin, in.st, in.C, wk, t, in.kc + (int64_t) hk * in.sq1 + i4);
        }
#pragma unroll
        for (int j = 0; j < NQK; ++j) {
            const float sk = gdn_warp_reduce_sum<32>(kv[j].x * kv[j].x + kv[j].y * kv[j].y + kv[j].z * kv[j].z + kv[j].w * kv[j].w);
            const float sq = gdn_warp_reduce_sum<32>(qv[j].x * qv[j].x + qv[j].y * qv[j].y + qv[j].z * qv[j].z + qv[j].w * qv[j].w);
            const float fk = qk_mul * rsqrtf(sk + qk_eps), fq = qk_mul * rsqrtf(sq + qk_eps);
            kv[j].x *= fk; kv[j].y *= fk; kv[j].z *= fk; kv[j].w *= fk;
            qv[j].x *= fq; qv[j].y *= fq; qv[j].z *= fq; qv[j].w *= fq;
        }
#pragma unroll
        for (int j = 0; j < NQK; ++j) {
            const int r = (tid + 256 * j) >> 5;
            const bool ok = r < Lc;
            const float4 z4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const uint2 kh = gdnc_pack4h(ok ? kv[j] : z4), qh = gdnc_pack4h(ok ? qv[j] : z4);
            *(uint2 *) (Ks + r * SK + i4) = kh;
            *(uint2 *) (Qs + r * SK + i4) = qh;
            if (ok) {
                const int64_t o = ((int64_t) hk * n_tokens + t0 + r) * 128 + i4;
                *(uint2 *) (Kh + o) = kh;
                *(uint2 *) (Qh + o) = qh;
            }
        }
    }
    for (int idx = tid; idx < R * CL; idx += 256) {
        const int j = idx / CL, r = idx % CL, h = hk + j * Hk;
        const int64_t o = (int64_t) (t0 + r) * sb2 + (int64_t) h * sb1;
        Gs[j][r] = r < Lc ? g[o]    : 0.0f;
        bs[j][r] = r < Lc ? beta[o] : 0.0f;
    }
    __syncthreads();
    if (tid < R * 32) {
        // inclusive prefix sum of g over the chunk, one wave per value head
        const int j = tid >> 5;
        float G = Gs[j][lane];
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const float u = __shfl_up(G, d, 32);
            if (lane >= d) G += u;
        }
        Gs[j][lane] = G;
    }
    for (int job = wave; job < 2 * NTRI; job += 8) {
        const bool isq = job >= NTRI;
        int jj = isq ? job - NTRI : job, rt = 0;
        while (jj >= rt + 1) { jj -= rt + 1; ++rt; }
        const int st = jj;
        const _Float16 * As = isq ? Qs : Ks;
        gdnc_v8f acc = {};
#pragma unroll
        for (int kk = 0; kk < 128; kk += 16) {
            acc = GDNC_WMMA(gdnc_frag(As, SK, rt * 16, kk, lane), gdnc_frag(Ks, SK, st * 16, kk, lane), acc);
        }
        float (*D)[CL + 1] = isq ? QK : KK;
#pragma unroll
        for (int e = 0; e < 8; ++e) D[rt * 16 + 2 * e + rh][st * 16 + cl] = acc[e];
    }
    __syncthreads();
    // A and P for all R value heads at once
    for (int idx = tid; idx < R * CL * CL; idx += 256) {
        const int j = idx / (CL * CL), rs = idx % (CL * CL), r = rs / CL, s = rs % CL, h = hk + j * Hk;
        const float d = s <= r ? __expf(Gs[j][r] - Gs[j][s]) : 0.0f;
        A[j][r][s] = s < r ? bs[j][r] * KK[r][s] * d : 0.0f;
        Pb[((int64_t) h * nch + ch) * CL * CL + rs] = (_Float16) (s <= r ? scale * QK[r][s] * d : 0.0f);
    }
    if (tid < R * CL) {
        const int j = tid / CL, r = tid % CL, h = hk + j * Hk;
        Gb[((int64_t) h * nch + ch) * CL + r] = make_float2(__expf(Gs[j][r]), __expf(Gs[j][CL - 1] - Gs[j][r]));
    }
    __syncthreads();
    // T = (I + A)^-1 column by column, barrier-free (A of all R heads is in LDS): lane = row, wave = CPW columns
    for (int j = 0; j < R; ++j) {
        const int h = hk + j * Hk;
        const int j0 = wave * CPW;
        float x[CPW];
#pragma unroll
        for (int c = 0; c < CPW; ++c) x[c] = (lane == j0 + c) ? 1.0f : 0.0f;
#pragma unroll
        for (int qq = 0; qq < CL - 1; ++qq) {
            if (qq < j0) continue;
            const float a = A[j][lane][qq];
#pragma unroll
            for (int c = 0; c < CPW; ++c) {
                const float xq = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x[c]), qq));
                x[c] = fmaf(-a, xq, x[c]);
            }
        }
        _Float16 tmp[CPW];
#pragma unroll
        for (int c = 0; c < CPW; ++c) tmp[c] = (_Float16) (x[c] * bs[j][j0 + c]);
        __builtin_memcpy(Tb + ((int64_t) h * nch + ch) * CL * CL + lane * CL + j0, tmp, sizeof(tmp));
    }
#else
    NO_DEVICE_CODE;
#endif
}

template <int CL, int MODE = 0, typename XT = float, int NPF = 0, int PFD = 3, bool VROW = false, bool OUT_F16 = false, bool ZBF16 = false>
__global__ void __launch_bounds__(256, 6)
gdn_scan2_kernel(int * __restrict__ prog, unsigned * __restrict__ sink, const _Float16 * __restrict__ Qh, const _Float16 * __restrict__ Kh,
        const XT * __restrict__ X, const float * __restrict__ cst, const float * __restrict__ cw, const int C, const int vc,
        const float * __restrict__ curr_state, float * __restrict__ state,
        const _Float16 * __restrict__ Tb, const _Float16 * __restrict__ Pb, const float2 * __restrict__ Gb,
        const int H, const int Hk, const int n_tokens, const int nch, const float scale,
        const float * __restrict__ on_w, const void * __restrict__ on_z_, uint16_t * __restrict__ out16, const float on_eps,
        const int xv_hs, const int xv_c0, const int z_ts, const int z_hs) {
#if defined(GDNC_ENABLED)
    static_assert(CL == 32, "scan2: CL 32");
    static_assert(!(ZBF16 && NPF > 0), "scan2: the MALL prefetcher reads z as XT");
    const XT * __restrict__ on_z = (const XT *) on_z_;
    constexpr int ZB = ZBF16 ? 2 : (int) sizeof(XT);   // bytes per z value (ZBF16: BF16 z written by the z GEMM, GGML_GDN_Z16)
    constexpr int NT  = CL / 16;
    constexpr int SQ  = 136;
    constexpr int ST  = CL + 8;
    constexpr int NQK = CL * 32 / 256;
    constexpr int NTP = 2 * CL * CL / 8 / 256;
    constexpr int NX  = CL + 2;                 // x rows per lane: t0 + rh - 3 + k, k < NX
    __shared__ __align__(16) _Float16 Qs[CL * SQ];
    __shared__ __align__(16) _Float16 Ks[CL * SQ];
    __shared__ __align__(16) _Float16 KTs[128 * ST];
    __shared__ __align__(16) _Float16 Ts[CL * ST];
    __shared__ __align__(16) _Float16 Ps[CL * ST];
    __shared__ __align__(16) _Float16 Os[CL * SQ];
    __shared__ float2 Gs[CL];
    __shared__ __align__(16) float Wn[128];
    if constexpr (NPF > 0) {
        if ((int) blockIdx.x >= H) {
            // MALL prefetcher: streams x_v / z of chunk c into the caches once the scan blocks have finished
            // chunk c - PFD (cumulative progress counter), so the scan's own (latency-bound) loads hit L2 / MALL
            const int p = blockIdx.x - H, tid = threadIdx.x;
            unsigned acc = 0;
            const int total = H * nch;
            if (tid == 0) sink[1 + p] = __hip_atomic_load(prog, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            for (int c = 0; c < nch; ++c) {
                if (threadIdx.x == 0) {
                    int pr;
                    while ((pr = __hip_atomic_load(prog, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) < H * (c - PFD) && pr < total)
                        __builtin_amdgcn_s_sleep(8);
                }
                __syncthreads();
                if (__hip_atomic_load(prog, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >= total) break;
                for (int hh = p; hh < H; hh += NPF) {
                    // 32 rows x 128 values of x_v and of z for head hh: 8 threads per row, 16 values each
                    const int r = tid >> 3, q = tid & 7;
                    const int t = min(c * CL + r, n_tokens - 1);
                    const XT * xp = X + (int64_t) t * C + hh * xv_hs + xv_c0 + q * 16;
                    const XT * zp = on_z + (int64_t) t * z_ts + hh * z_hs + q * 16;
                    acc ^= *(const volatile unsigned *) xp;
                    acc ^= *(const volatile unsigned *) zp;
                    if constexpr (sizeof(XT) == 4) {
                        acc ^= *(const volatile unsigned *) (xp + 8);
                        acc ^= *(const volatile unsigned *) (zp + 8);
                    }
                }
            }
            if (acc == 0x12345678u) sink[0] = acc;
            return;
        }
    }
    const int h = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int hk = h % Hk;
    const int cl = lane & 15, rh = lane >> 4, col = wave * 16 + cl;
    if (tid < 128) Wn[tid] = on_w[tid];
    const int nr = tid >> 3, n16 = tid & 7;     // norm stage: row nr, columns 16 n16 .. +15

    const float * s_in = curr_state + (int64_t) h * 128 * 128;
    gdnc_v8f S[8];
#pragma unroll
    for (int it = 0; it < 8; ++it)
#pragma unroll
        for (int e = 0; e < 8; ++e) S[it][e] = s_in[col * 128 + it * 16 + 2 * e + rh];
    const __amdgpu_buffer_rsrc_t rz = __builtin_amdgcn_make_buffer_rsrc((void *) on_z, 0, 0x7fffffff, 0x31004000);
    const __amdgpu_buffer_rsrc_t rx = __builtin_amdgcn_make_buffer_rsrc((void *) X, 0, 0x7fffffff, 0x31004000);
    const __amdgpu_buffer_rsrc_t rout = __builtin_amdgcn_make_buffer_rsrc((void *) out16, 0, n_tokens * H * 128 * 2, 0x31004000);

    const _Float16 * kh = Kh + (int64_t) hk * n_tokens * 128;
    const _Float16 * qh = Qh + (int64_t) hk * n_tokens * 128;
    const int vch = vc + h * 128 + col;
    const int xoff = h * xv_hs + xv_c0 + col;
    float wv[4];
    { const float4 b = *(const float4 *) (cw + (int64_t) vch * 4); wv[0] = b.x; wv[1] = b.y; wv[2] = b.z; wv[3] = b.w; }

    uint2 pk[NQK], pq[NQK];
    uint4 ptp[NTP];
    float2 pg;
    float xr[NX];
    float xq4[7][4];      // VROW: rows t0 + 4 wave - 3 + j, channels 4 lane .. +3
    float wq4[4][4];
    if constexpr (VROW) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float4 b = *(const float4 *) (cw + (int64_t) (vc + h * 128 + lane * 4 + c) * 4);
            wq4[c][0] = b.x; wq4[c][1] = b.y; wq4[c][2] = b.z; wq4[c][3] = b.w;
        }
    }
    constexpr int ZN = ZB;                  // uint4 of z per thread (16 values)
    uint4 pz[ZN];
    auto prefetch = [&](const int ch) {
        const int t0 = ch * CL;
#pragma unroll
        for (int j = 0; j < NQK; ++j) {
            const int idx = tid + 256 * j, r = idx >> 5, i = (idx & 31) * 4;
            const int64_t o = (int64_t) min(t0 + r, n_tokens - 1) * 128 + i;
            pk[j] = *(const uint2 *) (kh + o);
            pq[j] = *(const uint2 *) (qh + o);
        }
        const _Float16 * Tp = Tb + ((int64_t) h * nch + ch) * CL * CL;
        const _Float16 * Pp = Pb + ((int64_t) h * nch + ch) * CL * CL;
#pragma unroll
        for (int j = 0; j < NTP; ++j) {
            const int idx = tid + 256 * j;
            const int w = idx % (CL * CL / 8);
            ptp[j] = *(const uint4 *) ((idx < CL * CL / 8 ? Tp : Pp) + w * 8);
        }
        pg = Gb[((int64_t) h * nch + ch) * CL + (tid % CL)];
        if constexpr (VROW) {
            const int xo4 = h * xv_hs + xv_c0 + lane * 4;
#pragma unroll
            for (int j = 0; j < 7; ++j) {
                const int t = min(max(t0 + wave * 4 - 3 + j, 0), n_tokens - 1);
                if constexpr (sizeof(XT) == 4) {
                    typedef unsigned v4u __attribute__((ext_vector_type(4)));
                    const v4u v = __builtin_amdgcn_raw_buffer_load_b128(rx, (t * C + xo4) * 4, 0, 0);
                    xq4[j][0] = __uint_as_float(v.x); xq4[j][1] = __uint_as_float(v.y); xq4[j][2] = __uint_as_float(v.z); xq4[j][3] = __uint_as_float(v.w);
                } else {
                    typedef unsigned v2u __attribute__((ext_vector_type(2)));
                    const v2u v = __builtin_amdgcn_raw_buffer_load_b64(rx, (t * C + xo4) * 2, 0, 0);
                    _Float16 hv[4]; __builtin_memcpy(hv, &v, 8);
                    xq4[j][0] = hv[0]; xq4[j][1] = hv[1]; xq4[j][2] = hv[2]; xq4[j][3] = hv[3];
                }
            }
        } else
#pragma unroll
        for (int k = 0; k < NX; ++k) {
            const int t = (MODE & 256) ? (rh + k) : min(max(t0 + rh - 3 + k, 0), n_tokens - 1);
            if constexpr (sizeof(XT) == 4) {
                xr[k] = (MODE & 128) ? (float) k : __uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(rx, (t * C + xoff) * 4, 0, 0));
            } else {
                xr[k] = (MODE & 128) ? (float) k : (float) __builtin_bit_cast(_Float16, __builtin_amdgcn_raw_buffer_load_b16(rx, (t * C + xoff) * 2, 0, 0));
            }
        }
    };
    auto prefetch_z = [&](const int ch) {
        if constexpr ((MODE & 512) != 0) return;   // late gate: sigmoid(z) is applied by gdn_gate_kernel after the scan
        const int t = (MODE & 256) ? nr : min(ch * CL + nr, n_tokens - 1);
        const int zb = (t * z_ts + h * z_hs + n16 * 16) * ZB;
        typedef unsigned v4u __attribute__((ext_vector_type(4)));
#pragma unroll
        for (int j = 0; j < ZN; ++j) {
            if (MODE & 64) { pz[j] = make_uint4(j, 2, 3, 4); continue; }
            const v4u v = __builtin_amdgcn_raw_buffer_load_b128(rz, zb + 16 * j, 0, 0);
            pz[j] = make_uint4(v.x, v.y, v.z, v.w);
        }
    };
    prefetch(0);
    prefetch_z(0);
    // conv history (rows t < 0) from the conv state
#pragma unroll
    for (int k = 0; k < 3; ++k) if (rh - 3 + k < 0) xr[k] = cst[(int64_t) vch * 3 + rh + k];
    if constexpr (VROW) {
        if (wave == 0) {
#pragma unroll
            for (int j = 0; j < 3; ++j)
#pragma unroll
                for (int c = 0; c < 4; ++c) xq4[j][c] = cst[(int64_t) (vc + h * 128 + lane * 4 + c) * 3 + j];
        }
    }

    for (int ch = 0; ch < nch; ++ch) {
        const int t0 = ch * CL;
        S2_BAR();
#pragma unroll
        for (int j = 0; j < NQK; ++j) {
            const int idx = tid + 256 * j, r = idx >> 5, i = (idx & 31) * 4;
            *(uint2 *) (Ks + r * SQ + i) = pk[j];
            *(uint2 *) (Qs + r * SQ + i) = pq[j];
            _Float16 kk4[4];
            __builtin_memcpy(kk4, &pk[j], 8);
            KTs[(i + 0) * ST + r] = kk4[0]; KTs[(i + 1) * ST + r] = kk4[1];
            KTs[(i + 2) * ST + r] = kk4[2]; KTs[(i + 3) * ST + r] = kk4[3];
        }
#pragma unroll
        for (int j = 0; j < NTP; ++j) {
            const int idx = tid + 256 * j;
            const int w = idx % (CL * CL / 8), r = w / (CL / 8), c8 = (w % (CL / 8)) * 8;
            *(uint4 *) ((idx < CL * CL / 8 ? Ts : Ps) + r * ST + c8) = ptp[j];
        }
        if (tid < CL) Gs[tid] = pg;
        // v = silu(conv(x)) in accumulator layout: row r = 16 rt + 2e + rh uses xr[r - rh + j], j = 0..3
        gdnc_v8f vacc[NT];
        if constexpr (VROW) {
            // row layout: thread = rows 4 wave .. +3, channels 4 lane .. +3 -> fp16 rows in Os (free until the O stage)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                _Float16 hv[4];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    float sacc = xq4[i][c] * wq4[c][0];
                    sacc = fmaf(xq4[i + 1][c], wq4[c][1], sacc); sacc = fmaf(xq4[i + 2][c], wq4[c][2], sacc); sacc = fmaf(xq4[i + 3][c], wq4[c][3], sacc);
                    hv[c] = (_Float16) s2_silu(sacc);
                }
                *(uint2 *) (Os + (wave * 4 + i) * SQ + lane * 4) = *(const uint2 *) hv;
            }
        } else
#pragma unroll
        for (int rt = 0; rt < NT; ++rt)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int k = rt * 16 + 2 * e;
                float s = xr[k] * wv[0];
                s = fmaf(xr[k + 1], wv[1], s); s = fmaf(xr[k + 2], wv[2], s); s = fmaf(xr[k + 3], wv[3], s);
                vacc[rt][e] = (MODE & 32) ? xr[k] : s2_silu(s);
            }
        S2_BAR();
        const int nx = min(ch + 1, nch - 1);
        if (!(MODE & 16)) prefetch(nx);

        gdnc_v8f ks[NT], qs[NT];
#pragma unroll
        for (int rt = 0; rt < NT; ++rt) { ks[rt] = gdnc_v8f{}; qs[rt] = gdnc_v8f{}; }
#pragma unroll
        for (int it = 0; it < 8; ++it) {
            if (MODE & 1) break;
            const gdnc_v16h sb = s2_acc_to_frag(S[it], lane);
#pragma unroll
            for (int rt = 0; rt < NT; ++rt) {
                ks[rt] = GDNC_WMMA(gdnc_frag(Ks, SQ, rt * 16, it * 16, lane), sb, ks[rt]);
                qs[rt] = GDNC_WMMA(gdnc_frag(Qs, SQ, rt * 16, it * 16, lane), sb, qs[rt]);
            }
        }
        if constexpr (VROW) {
#pragma unroll
            for (int rt = 0; rt < NT; ++rt)
#pragma unroll
                for (int e = 0; e < 8; ++e) vacc[rt][e] = (float) Os[(rt * 16 + 2 * e + rh) * SQ + col];
        }
        gdnc_v16h YB[NT];
#pragma unroll
        for (int rt = 0; rt < NT; ++rt) {
            gdnc_v8f y;
#pragma unroll
            for (int e = 0; e < 8; ++e) y[e] = fmaf(-Gs[rt * 16 + 2 * e + rh].x, ks[rt][e], vacc[rt][e]);
            YB[rt] = s2_acc_to_frag(y, lane);
        }
        gdnc_v16h UB[NT], UB2[NT];
#pragma unroll
        for (int rt = 0; rt < NT; ++rt) {
            gdnc_v8f u = {};
#pragma unroll
            for (int st = 0; st <= rt; ++st) u = GDNC_WMMA(gdnc_frag(Ts, ST, rt * 16, st * 16, lane), YB[st], u);
            UB[rt] = s2_acc_to_frag(u, lane);
            gdnc_v8f u2;
#pragma unroll
            for (int e = 0; e < 8; ++e) u2[e] = u[e] * Gs[rt * 16 + 2 * e + rh].y;
            UB2[rt] = s2_acc_to_frag(u2, lane);
        }
        const float eGL = Gs[CL - 1].x;
#pragma unroll
        for (int it = 0; it < 8; ++it) {
            gdnc_v8f acc;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] = S[it][e] * eGL;
#pragma unroll
            for (int st = 0; st < NT && !(MODE & 2); ++st) acc = GDNC_WMMA(gdnc_frag(KTs, ST, it * 16, st * 16, lane), UB2[st], acc);
            S[it] = acc;
        }
        if constexpr (VROW) S2_BAR();   // every wave has read its v from Os
#pragma unroll
        for (int rt = 0; rt < NT; ++rt) {
            gdnc_v8f o;
#pragma unroll
            for (int e = 0; e < 8; ++e) o[e] = qs[rt][e] * (Gs[rt * 16 + 2 * e + rh].x * scale);
#pragma unroll
            for (int st = 0; st <= rt; ++st) o = GDNC_WMMA(gdnc_frag(Ps, ST, rt * 16, st * 16, lane), UB[st], o);
#pragma unroll
            for (int e = 0; e < 8; ++e) Os[(rt * 16 + 2 * e + rh) * SQ + col] = (_Float16) o[e];
        }
        S2_BAR();
        // gated RMS norm: thread = row nr, columns 16 n16 .. +15
        if (!(MODE & 4)) {
            float o[16];
            {
                const uint4 a = *(const uint4 *) (Os + nr * SQ + n16 * 16), b = *(const uint4 *) (Os + nr * SQ + n16 * 16 + 8);
                _Float16 hv[16];
                __builtin_memcpy(hv, &a, 16); __builtin_memcpy(hv + 8, &b, 16);
#pragma unroll
                for (int c = 0; c < 16; ++c) o[c] = (float) hv[c];
            }
            float ss = 0.0f;
#pragma unroll
            for (int c = 0; c < 16; ++c) ss = fmaf(o[c], o[c], ss);
            ss += gdn_dpp_row_xmask<1>(ss); ss += gdn_dpp_row_xmask<2>(ss); ss += gdn_dpp_row_xmask<4>(ss);
            const float sc = rsqrtf(ss * (1.0f / 128.0f) + on_eps);
            float zz[16];
            if constexpr ((MODE & 512) != 0) {
#pragma unroll
                for (int c = 0; c < 16; ++c) zz[c] = 0.0f;
            } else if constexpr (ZBF16) {
                uint32_t zw[8];
                __builtin_memcpy(zw, pz, 32);
#pragma unroll
                for (int c = 0; c < 8; ++c) { zz[2 * c] = __uint_as_float(zw[c] << 16); zz[2 * c + 1] = __uint_as_float(zw[c] & 0xffff0000u); }
            } else if constexpr (sizeof(XT) == 4) {
                __builtin_memcpy(zz, pz, 64);
            } else {
                _Float16 zh[16];
                __builtin_memcpy(zh, pz, 32);
#pragma unroll
                for (int c = 0; c < 16; ++c) zz[c] = (float) zh[c];
            }
            float wn[16];
#pragma unroll
            for (int j = 0; j < 4; ++j) { const float4 b = *(const float4 *) (Wn + n16 * 16 + 4 * j); wn[4*j] = b.x; wn[4*j+1] = b.y; wn[4*j+2] = b.z; wn[4*j+3] = b.w; }
            uint32_t pk2[8];
#pragma unroll
            for (int c = 0; c < 16; c += 2) {
                if constexpr ((MODE & 512) != 0) {   // un-gated y = rmsnorm(o) * w as F16 (the gate kernel converts in place)
                    const float v0 = sc * o[c] * wn[c], v1 = sc * o[c + 1] * wn[c + 1];
                    pk2[c / 2] = (uint32_t) ggml_cuda_f32_to_act16(v0, true) | ((uint32_t) ggml_cuda_f32_to_act16(v1, true) << 16);
                } else {
                    const float v0 = (sc * o[c] * wn[c]) * s2_sigm(zz[c]);
                    const float v1 = (sc * o[c + 1] * wn[c + 1]) * s2_sigm(zz[c + 1]);
                    pk2[c / 2] = (uint32_t) ggml_cuda_f32_to_act16(v0, OUT_F16) | ((uint32_t) ggml_cuda_f32_to_act16(v1, OUT_F16) << 16);
                }
            }
            // unconditional buffer store: rows t >= n_tokens fall outside num_records and are dropped by the hardware
            // (a guard branch makes the compiler sink the z prefetch into it and drain every load with vmcnt(0))
            {
                const int ob = (((t0 + nr) * H + h) * 128 + n16 * 16) * 2;
                typedef unsigned v4u __attribute__((ext_vector_type(4)));
                __builtin_amdgcn_raw_buffer_store_b128(v4u{pk2[0], pk2[1], pk2[2], pk2[3]}, rout, ob, 0, 0);
                __builtin_amdgcn_raw_buffer_store_b128(v4u{pk2[4], pk2[5], pk2[6], pk2[7]}, rout, ob + 16, 0, 0);
            }
            if (!(MODE & 16)) prefetch_z(nx);
            if constexpr (NPF > 0) { if (tid == 0) __hip_atomic_fetch_add(prog, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
        }
    }

    float * s_out = state + (int64_t) h * 128 * 128;
#pragma unroll
    for (int it = 0; it < 8; ++it)
#pragma unroll
        for (int e = 0; e < 8; ++e) s_out[col * 128 + it * 16 + 2 * e + rh] = S[it][e];
#else
    NO_DEVICE_CODE;
#endif
}

// GGML_GDN_LATEJOIN: out16 (F16 y = rmsnorm(o) * w from scan2 MODE 512) *= sigmoid(z) in place, 8 values per thread
template <bool OUT_F16, bool ZBF16>
__global__ void __launch_bounds__(256)
gdn_gate_kernel(uint16_t * __restrict__ out16, const void * __restrict__ z, const int64_t n8) {
    const int64_t i = (int64_t) blockIdx.x * 256 + threadIdx.x;
    if (i >= n8) return;
    const uint4 y = ((const uint4 *) out16)[i];
    float zz[8];
    if constexpr (ZBF16) {
        const uint4 zb = ((const uint4 *) z)[i];
        const uint32_t zw[4] = { zb.x, zb.y, zb.z, zb.w };
#pragma unroll
        for (int c = 0; c < 4; ++c) { zz[2 * c] = __uint_as_float(zw[c] << 16); zz[2 * c + 1] = __uint_as_float(zw[c] & 0xffff0000u); }
    } else {
        const float4 z0 = ((const float4 *) z)[2 * i], z1 = ((const float4 *) z)[2 * i + 1];
        zz[0] = z0.x; zz[1] = z0.y; zz[2] = z0.z; zz[3] = z0.w; zz[4] = z1.x; zz[5] = z1.y; zz[6] = z1.z; zz[7] = z1.w;
    }
    _Float16 yh[8];
    __builtin_memcpy(yh, &y, 16);
    uint32_t o[4];
#pragma unroll
    for (int c = 0; c < 8; c += 2) {
        const float v0 = (float) yh[c] / (1.0f + __expf(-zz[c]));
        const float v1 = (float) yh[c + 1] / (1.0f + __expf(-zz[c + 1]));
        o[c / 2] = (uint32_t) ggml_cuda_f32_to_act16(v0, OUT_F16) | ((uint32_t) ggml_cuda_f32_to_act16(v1, OUT_F16) << 16);
    }
    ((uint4 *) out16)[i] = make_uint4(o[0], o[1], o[2], o[3]);
}

#ifndef GDN_V2_DEFAULT
#define GDN_V2_DEFAULT 1   // pp2048 1683 -> 1702 t/s; logits KLD vs the v1 chunk path at the KV-bf16 noise level
#endif
static bool gdn_chunk_enabled() {
    static const int v = getenv("GGML_GDN_CHUNK") ? atoi(getenv("GGML_GDN_CHUNK")) : 1;
    return v != 0;
}

static int gdn_chunk_min_tokens() {
    static const int v = getenv("GGML_GDN_CHUNK_MIN") ? atoi(getenv("GGML_GDN_CHUNK_MIN")) : 256;
    return v;
}

// the dispatch condition of the chunked prefill path (single source of truth for the op and the q/k-norm fusion)
static bool gdn_chunk_eligible_raw(const int cc, const bool kda, const bool keep_rs, const int64_t S_v, const int64_t H,
        const int64_t Hk, const int64_t n_tokens, const int64_t n_seqs, const int64_t sq1, const int64_t sq2,
        const void * q_d, const void * k_d) {
#if defined(GGML_USE_HIP)
    const bool aligned = sq1 % 4 == 0 && sq2 % 4 == 0 && ((uintptr_t) q_d) % 16 == 0 && ((uintptr_t) k_d) % 16 == 0;
    // n_seqs > 1: the chunk kernels run once per sequence on its slices (each exactly a one-sequence prefill)
    return !kda && !keep_rs && GGML_CUDA_CC_IS_RDNA3_5(cc) && S_v == 128 && n_seqs >= 1 && n_tokens >= gdn_chunk_min_tokens() &&
        aligned && Hk > 0 && H % Hk == 0 && H / Hk <= 4 && gdn_chunk_enabled();
#else
    GGML_UNUSED(cc); GGML_UNUSED(kda); GGML_UNUSED(keep_rs); GGML_UNUSED(S_v); GGML_UNUSED(H); GGML_UNUSED(Hk);
    GGML_UNUSED(n_tokens); GGML_UNUSED(n_seqs); GGML_UNUSED(sq1); GGML_UNUSED(sq2); GGML_UNUSED(q_d); GGML_UNUSED(k_d);
    return false;
#endif
}

bool ggml_cuda_gdn_chunk_eligible(const ggml_tensor * gdn, const ggml_tensor * q, const ggml_tensor * k) {
    const ggml_tensor * v = gdn->src[2];
    if (gdn->op != GGML_OP_GATED_DELTA_NET || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(q, k) || !ggml_are_same_stride(q, k) || !ggml_is_contiguous_rows(q) || q->ne[1] != gdn->src[0]->ne[1]) {
        return false;
    }
    const int  cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool kda     = gdn->src[3]->ne[0] == v->ne[0];
    const bool keep_rs = ggml_get_op_params_i32(gdn, 0) > 1;
    if (q->ne[3] != v->ne[3] || q->ne[2] != v->ne[2]) {
        return false;
    }
    return gdn_chunk_eligible_raw(cc, kda, keep_rs, v->ne[0], v->ne[1], q->ne[1], v->ne[2], v->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)), (int64_t) (q->nb[2] / sizeof(float)), q->data, k->data);
}

struct gdn_qk_norm_ovr {
    const ggml_tensor * q;
    const ggml_tensor * k;
    float eps;
    float mul;
};
static std::mutex                                                   g_gdn_qk_mtx;
static std::unordered_map<const ggml_tensor *, gdn_qk_norm_ovr>     g_gdn_qk;

struct gdn_conv_ovr {
    const float * x;
    const float * st;
    const float * cw;
    int64_t       C, qc, kc, vc;
};
static std::unordered_map<const ggml_tensor *, gdn_conv_ovr> g_gdn_conv;

void ggml_cuda_gdn_set_conv(const ggml_tensor * gdn, const float * x, const float * st, const float * cw,
        int64_t C, int64_t qc, int64_t kc, int64_t vc) {
    std::lock_guard<std::mutex> lk(g_gdn_qk_mtx);
    g_gdn_conv[gdn] = { x, st, cw, C, qc, kc, vc };
}

struct gdn_out_norm_ovr {
    const float * w;
    const float * z;
    float *       out;
    uint16_t *    out16;
    float         eps;
    bool          out16_f16;
    bool          z_bf16;     // z holds BF16 (its GEMM skipped the F32 store, GGML_GDN_Z16): only the v2 scan reads that
};
static std::unordered_map<const ggml_tensor *, gdn_out_norm_ovr> g_gdn_on;

void ggml_cuda_gdn_set_out_norm(const ggml_tensor * gdn, const float * w, const float * z, float * out, uint16_t * out16, float eps,
        bool out16_f16, bool z_bf16) {
    std::lock_guard<std::mutex> lk(g_gdn_qk_mtx);
    g_gdn_on[gdn] = { w, z, out, out16, eps, out16_f16, z_bf16 };
}

static bool gdn_v2_enabled() {
    static const bool v = getenv("GGML_GDN_V2") ? atoi(getenv("GGML_GDN_V2")) != 0 : GDN_V2_DEFAULT;
    return v;
}

// GGML_GDN_Z16 (default 1): the z GEMM of a GDN layer that takes the v2 chunked path stores BF16 only (no F32 z), read by
// the scan2 output-norm epilogue. Graph-time predicate (shapes / env only); the op aborts if the v2 scan is not taken after all.
bool ggml_cuda_gdn_z16_ok(const ggml_tensor * gdn) {
#if defined(GGML_USE_HIP)
    static const bool env = getenv("GGML_GDN_Z16") ? atoi(getenv("GGML_GDN_Z16")) != 0 : true;
    static const bool nofuse = getenv("GGML_CUDA_DISABLE_GDN_OUTNORM") != nullptr ||
        (getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && atoi(getenv("GGML_CUDA_DISABLE_FUSION")) != 0);
    if (!env || nofuse || !gdn || gdn->op != GGML_OP_GATED_DELTA_NET || !gdn_v2_enabled() || !gdn_chunk_enabled()) return false;
    const ggml_tensor * v = gdn->src[2];
    const int  cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool kda     = gdn->src[3]->ne[0] == v->ne[0];
    const bool keep_rs = ggml_get_op_params_i32(gdn, 0) > 1;
    return GGML_CUDA_CC_IS_RDNA3_5(cc) && !kda && !keep_rs && v->ne[0] == 128 && v->ne[2] >= gdn_chunk_min_tokens() &&
        v->ne[1] * 128 % 8 == 0;
#else
    GGML_UNUSED(gdn);
    return false;
#endif
}

bool ggml_cuda_gdn_chunk_eligible_node(const ggml_tensor * gdn) {
    const ggml_tensor * q = gdn->src[0], * k = gdn->src[1];
    {
        std::lock_guard<std::mutex> lk(g_gdn_qk_mtx);
        auto it = g_gdn_qk.find(gdn);
        if (it != g_gdn_qk.end()) { q = it->second.q; k = it->second.k; }
    }
    return ggml_cuda_gdn_chunk_eligible(gdn, q, k);
}

void ggml_cuda_gdn_set_qk_norm(const ggml_tensor * gdn, const ggml_tensor * q_raw, const ggml_tensor * k_raw, float eps, float mul) {
    std::lock_guard<std::mutex> lk(g_gdn_qk_mtx);
    g_gdn_qk[gdn] = { q_raw, k_raw, eps, mul };
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    // Qwen3.6 PP2048 measured faster on gfx1151 on 2026-09-02. Retest if occupancy changes.
    const int num_warps = GGML_CUDA_CC_IS_RDNA3_5(cc) && S_v == 128 && (H == 32 || H == 48 || H == 64) && !KDA ? 32 :
        (GGML_CUDA_CC_IS_RDNA3_5(cc) ? 16 : gated_delta_net_num_warps(cc));
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    if constexpr (KDA && !keep_rs_t) {
        if (GGML_CUDA_CC_IS_RDNA3_5(cc) && S_v == 128 && (H == 16 || H == 32) && n_tokens >= 16 && n_seqs == 1) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
            auto kernel = H == 16 ? gated_delta_net_kda_tiled_128_cuda<16> : gated_delta_net_kda_tiled_128_cuda<32>;
            ggml_cuda_kernel_launch(kernel, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, n_tokens,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale);
            return;
        }
    }

    if constexpr (!KDA) {
        if (GGML_CUDA_CC_IS_RDNA3_5(cc) && S_v == 128 && num_warps == 32 && n_tokens >= 16) {
            // 16 warps x 4 columns per block, 16-token tiles: best of the swept configurations on gfx1151
            //     (2048 tokens, H=32: 5.03 ms -> 2.2 ms; H=64: 9.29 ms -> 4.5 ms).
            const int cfg = getenv("GGML_GDN_TILE_CFG") ? atoi(getenv("GGML_GDN_TILE_CFG")) : (H == 48 ? 1 : 0);
            auto launch_tiled = [&](auto kernel, int warps, int cols) {
                const dim3 tiled_grid(H, n_seqs, S_v / (warps * cols));
                const dim3 tiled_block(warp_size, warps, 1);
                const ggml_cuda_kernel_launch_params launch_params(tiled_grid, tiled_block, 0, stream);
                ggml_cuda_kernel_launch(kernel, launch_params,
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                    sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                    neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            };
            switch (cfg) {
                case 1: launch_tiled(gated_delta_net_tiled_cuda<128, 8,  8, 16, keep_rs_t>, 8,  8); break;
                case 2: launch_tiled(gated_delta_net_tiled_cuda<128, 32, 2, 16, keep_rs_t>, 32, 2); break;
                case 3: launch_tiled(gated_delta_net_tiled_cuda<128, 16, 4,  8, keep_rs_t>, 16, 4); break;
                default: launch_tiled(gated_delta_net_tiled_cuda<128, 16, 4, 16, keep_rs_t>, 16, 4); break;
            }
            return;
        }
    }

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
    }
}

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    return {
        (const float *) q->data,
        (const float *) k->data,
        (const float *) v->data,
        (const float *) g->data,
        (const float *) beta->data,
        (const float *) state->data,
        (float *) dst->data,
        nullptr,
        H,
        q->ne[1],
        v->ne[2],
        v->ne[3],
        v->ne[3] / q->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)),
        (int64_t) (q->nb[2] / sizeof(float)),
        (int64_t) (q->nb[3] / sizeof(float)),
        (int64_t) (v->nb[1] / sizeof(float)),
        (int64_t) (v->nb[2] / sizeof(float)),
        (int64_t) (v->nb[3] / sizeof(float)),
        (int64_t) (beta->nb[1] / sizeof(float)),
        (int64_t) (beta->nb[2] / sizeof(float)),
        (int64_t) (beta->nb[3] / sizeof(float)),
        1.0f / sqrtf((float) S_v),
        g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) < v->ne[2] && S_v == 128 && q->ne[0] == 128,
    };
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    // q/k-norm fusion: the normalized q/k were never computed; read the raw views and normalize in the chunk kernels
    gdn_qk_norm_ovr qk_ovr  = { nullptr, nullptr, -1.0f, 1.0f };
    {
        std::lock_guard<std::mutex> lk(g_gdn_qk_mtx);
        auto it = g_gdn_qk.find(dst);
        if (it != g_gdn_qk.end()) {
            qk_ovr = it->second;
            g_gdn_qk.erase(it);
        }
    }
    gdn_out_norm_ovr on_ovr = { nullptr, nullptr, nullptr, nullptr, 0.0f, false, false };
    {
        std::lock_guard<std::mutex> lk(g_gdn_qk_mtx);
        auto it = g_gdn_on.find(dst);
        if (it != g_gdn_on.end()) {
            on_ovr = it->second;
            g_gdn_on.erase(it);
        }
    }
    gdn_conv_ovr cv_ovr = { nullptr, nullptr, nullptr, 0, 0, 0, 0 };
    {
        std::lock_guard<std::mutex> lk(g_gdn_qk_mtx);
        auto it = g_gdn_conv.find(dst);
        if (it != g_gdn_conv.end()) {
            cv_ovr = it->second;
            g_gdn_conv.erase(it);
        }
    }
    ggml_tensor * src_q     = qk_ovr.q ? (ggml_tensor *) qk_ovr.q : dst->src[0];
    ggml_tensor * src_k     = qk_ovr.k ? (ggml_tensor *) qk_ovr.k : dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    GGML_ASSERT(neq3 > 0 && nev3 >= neq3 && nev3 % neq3 == 0);
    GGML_ASSERT(nek3 == neq3);
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

#if defined(GGML_USE_HIP)
    {
        const int dev = ggml_cuda_get_device();
        const int cc  = ggml_cuda_info().devices[dev].cc;
        const float qk_eps = qk_ovr.q ? qk_ovr.eps : -1.0f;
        const float qk_mul = qk_ovr.q ? qk_ovr.mul : 1.0f;
        const bool eligible = neq3 == n_seqs && gdn_chunk_eligible_raw(cc, kda, keep_rs, S_v, H, neqk1, n_tokens, n_seqs, sq1, sq2, q_d, k_d);
        if ((qk_ovr.q || on_ovr.z || cv_ovr.x) && !eligible) {
            GGML_ABORT("gated_delta_net: conv / q/k / output norms were fused for %s but the chunked path is not taken", dst->name);
        }
        if (eligible) {
            const gdnc_in in0 = { q_d, k_d, v_d, sq1, sq2, sv1, sv2, cv_ovr.x, cv_ovr.st, cv_ovr.cw, cv_ovr.C, cv_ovr.qc, cv_ovr.kc, cv_ovr.vc };
            const bool gdn_v2 = gdn_v2_enabled();
            const bool use_v2 = gdn_v2 && cv_ovr.x && qk_ovr.q && on_ovr.z && on_ovr.out16 && !on_ovr.out && sv1 == 128 &&
                ((uintptr_t) cv_ovr.x) % 16 == 0 && (cv_ovr.C % 4) == 0 && (cv_ovr.vc % 4) == 0 &&
                (int64_t) n_tokens * cv_ovr.C * 4 < INT32_MAX && (int64_t) n_tokens * H * 128 * 2 < INT32_MAX;
            // z stored BF16-only (GGML_GDN_Z16, a graph-time choice) is read natively only by the v2 epilogue: when the
            // run-time checks above still refuse v2, the chunked scan below reads an F32 copy instead
            const bool z_widen = on_ovr.z_bf16 && !use_v2;
            auto run = [&](auto CLc, const int64_t is) {
                // sequence `is`: its q/k/v (or conv input and history), g/beta, state in/out, output rows and z / norm output
                const size_t ro = (size_t) is * n_tokens * H * 128;   // output rows, z and the norm output of the sequence
                gdnc_in in = in0;
                in.q += is * sq3; in.k += is * sq3; in.v += is * sv3;
                const float * cx  = cv_ovr.x  ? cv_ovr.x  + (size_t) is * n_tokens * cv_ovr.C : nullptr;
                const float * cst = cv_ovr.st ? cv_ovr.st + (size_t) is * 3 * cv_ovr.C : nullptr;
                in.x = cx; in.st = cst;
                const float * gs = g_d + is * sb3, * bs = b_d + is * sb3;
                const float * s0 = s_d + (size_t) is * S_v * S_v * H;
                float * so = state_d + (size_t) is * S_v * S_v * H;
                float * dsts = dst_d + ro;
                const float * zs = on_ovr.z ? (const float *) ((const char *) on_ovr.z + ro * (on_ovr.z_bf16 ? 2 : 4)) : nullptr;
                uint16_t * o16s = on_ovr.out16 ? on_ovr.out16 + ro : nullptr;
                float * o32s = on_ovr.out ? on_ovr.out + ro : nullptr;
                constexpr int CL = decltype(CLc)::value;
                const int nch = (int) ((n_tokens + CL - 1) / CL);
                if (use_v2) {
                    ggml_cuda_pool_alloc<_Float16> Tb(ctx.pool(), (size_t) H * nch * CL * CL);
                    ggml_cuda_pool_alloc<_Float16> Pb(ctx.pool(), (size_t) H * nch * CL * CL);
                    ggml_cuda_pool_alloc<float2>   Gb(ctx.pool(), (size_t) H * nch * CL);
                    ggml_cuda_pool_alloc<_Float16> Qh(ctx.pool(), (size_t) neqk1 * n_tokens * 128);
                    ggml_cuda_pool_alloc<_Float16> Kh(ctx.pool(), (size_t) neqk1 * n_tokens * 128);
                    gdn_prep2_kernel<CL, float><<<dim3(neqk1, nch), 256, 0, stream>>>(in, cx, gs, bs,
                        Tb.get(), Pb.get(), Gb.get(), Qh.get(), Kh.get(), (int) H, (int) neqk1, (int) n_tokens, nch,
                        sb1, sb2, scale, qk_eps, qk_mul);
                    // GGML_GDN_LATEJOIN=1: the scan runs without z, concurrently with the z GEMM (GGML_CUDA_OVL side stream);
                    // sigmoid(z) is applied afterwards by gdn_gate_kernel
                    static const bool late = getenv("GGML_GDN_LATEJOIN") ? atoi(getenv("GGML_GDN_LATEJOIN")) != 0 : false;
                    if (late) {
                        gdn_scan2_kernel<CL, 512, float, 0, 3, true, true><<<dim3(H), 256, 0, stream>>>(nullptr, nullptr, Qh.get(), Kh.get(),
                            cx, cst, cv_ovr.cw, (int) cv_ovr.C, (int) cv_ovr.vc, s0, so,
                            Tb.get(), Pb.get(), Gb.get(), (int) H, (int) neqk1, (int) n_tokens, nch, scale,
                            on_ovr.w, zs, o16s, on_ovr.eps, 128, (int) cv_ovr.vc, (int) (H * 128), 128);
                        ggml_cuda_ovl_join(ctx);
                        const int64_t n8 = (int64_t) n_tokens * H * 128 / 8;
                        auto gk = on_ovr.z_bf16 ? (on_ovr.out16_f16 ? gdn_gate_kernel<true, true> : gdn_gate_kernel<false, true>)
                                                : (on_ovr.out16_f16 ? gdn_gate_kernel<true, false> : gdn_gate_kernel<false, false>);
                        gk<<<(n8 + 255) / 256, 256, 0, stream>>>(o16s, zs, n8);
                        CUDA_CHECK(cudaGetLastError());
                        return;
                    }
                    ggml_cuda_ovl_join(ctx);   // GGML_CUDA_OVL: z (read by the output-norm epilogue) ran on the side stream
                    // MMB_DENSE_F16: the F16 out16 copy is a template flag (no runtime select in the epilogue)
                    (on_ovr.z_bf16 ? (on_ovr.out16_f16 ? gdn_scan2_kernel<CL, 0, float, 0, 3, true, true, true> : gdn_scan2_kernel<CL, 0, float, 0, 3, true, false, true>)
                                   : (on_ovr.out16_f16 ? gdn_scan2_kernel<CL, 0, float, 0, 3, true, true> : gdn_scan2_kernel<CL, 0, float, 0, 3, true, false>))
                        <<<dim3(H), 256, 0, stream>>>(nullptr, nullptr, Qh.get(), Kh.get(),
                        cx, cst, cv_ovr.cw, (int) cv_ovr.C, (int) cv_ovr.vc, s0, so,
                        Tb.get(), Pb.get(), Gb.get(), (int) H, (int) neqk1, (int) n_tokens, nch, scale,
                        on_ovr.w, zs, o16s, on_ovr.eps, 128, (int) cv_ovr.vc, (int) (H * 128), 128);
                    CUDA_CHECK(cudaGetLastError());
                    return;
                }
                ggml_cuda_pool_alloc<_Float16> Tb(ctx.pool(), (size_t) H * nch * CL * CL);
                ggml_cuda_pool_alloc<_Float16> Pb(ctx.pool(), (size_t) H * nch * CL * CL);
                ggml_cuda_pool_alloc<float2>   Gb(ctx.pool(), (size_t) H * nch * CL);
                ggml_cuda_pool_alloc<_Float16> Qh(ctx.pool(), (size_t) neqk1 * n_tokens * 128);
                ggml_cuda_pool_alloc<_Float16> Kh(ctx.pool(), (size_t) neqk1 * n_tokens * 128);
                ggml_cuda_pool_alloc<_Float16> Vh(ctx.pool(), (size_t) H * n_tokens * 128);
                gdn_chunk_prep_kernel<CL><<<dim3(neqk1, nch), 256, 0, stream>>>(in, gs, bs,
                    Tb.get(), Pb.get(), Gb.get(), Qh.get(), Kh.get(), Vh.get(), (int) H, (int) neqk1, (int) n_tokens, nch,
                    sb1, sb2, scale, qk_eps, qk_mul);
                ggml_cuda_ovl_join(ctx);   // z may come from the side stream
                ggml_cuda_pool_alloc<float> z32(ctx.pool());
                if (z_widen) {
                    z32.alloc((size_t) n_tokens * H * 128);
                    ggml_get_to_fp32_cuda(GGML_TYPE_BF16)(zs, z32.get(), (int64_t) n_tokens * H * 128, stream);
                    zs = z32.get();
                }
                gdn_chunk_scan_kernel<CL><<<dim3(H), 256, 0, stream>>>(Qh.get(), Kh.get(), Vh.get(), s0, dsts, so,
                    Tb.get(), Pb.get(), Gb.get(), (int) H, (int) neqk1, (int) n_tokens, nch, scale,
                    on_ovr.w, zs, o32s, o16s, on_ovr.eps, on_ovr.out16_f16);
                CUDA_CHECK(cudaGetLastError());
            };
            for (int64_t is = 0; is < n_seqs; ++is) {
                run(std::integral_constant<int, 32>{}, is);  //CL64 fails LDS 90KB > 64KB limit
            }
            return;
        }
    }
#endif

    // upstream MMA path for the cases the fork chunked path does not take (no fused norms reach it)
    {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        args.state_out = state_d;
        if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
            return;
        }
    }
    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}

// ---------------------------------------------------------------------------------------------
// Fused Qwen3.5/3.6 Gated DeltaNet decode step (n_tokens == 1, n_seqs == 1, S == 128): one block per value
// head computes the causal conv + SiLU of its q/k/v channels (writing the shifted conv state back into the
// cache), the q/k L2 norms, the gate (softplus) and beta (sigmoid), the recurrence with the new state written
// straight into the cache, and the gated RMS norm of the attention output. This replaces 14 graph nodes
// (11 kernels plus their ~2 us launch gaps on gfx1151) that together take ~50 us per layer in decode.
//
// Every op is spelled with the same operation order and FMA contraction as the kernels it replaces
// (ssm_conv_f32, l2_norm_dual_f32_s128, unary/binary elementwise, gated_delta_net_cuda, rms_norm_f32<128>),
// so the outputs are bit-identical to the unfused graph. When args.qk_rms_scale is set, the q/k norms
// replay rms_norm_f32<256, false> + scale_f32 (the qwen4exp build_gdn_l2_norm form) with the same thread
// layout, reduction tree and single-rounding products.

static __device__ __forceinline__ float gdn_decode_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __device__ __forceinline__ float gdn_decode_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

template <int S, int D_CONV>
__global__ void __launch_bounds__(32 * 32, 1)
gdn_decode_fused_cuda(const ggml_cuda_gdn_decode_args args) {
    constexpr int warp_size     = 32;
    constexpr int nwarps        = 32;
    constexpr int rows_per_lane = S / warp_size;
    constexpr int cols_per_warp = S / nwarps;
    static_assert(S == warp_size * rows_per_lane && S == nwarps * cols_per_warp);

    __shared__ float q_c[S];
    __shared__ float k_c[S];
    __shared__ float v_c[S];
    __shared__ float q_n[S];
    __shared__ float k_n[S];
    __shared__ float attn[S];
    __shared__ float qk_red[8]; // q/k rms norm: the 4+4 per-warp partial sums of the standalone 256-thread block

    const int h      = blockIdx.x;                 // value head
    const int lane   = threadIdx.x;
    const int warp   = threadIdx.y;
    const int thread = warp * warp_size + lane;
    const int hk     = h % args.H_k;               // key/query head (fastmodulo(h_idx, neqk1) in gated_delta_net_cuda)

    // 1. causal conv + SiLU for this head's q, k and v channels; shift the conv state (CPY conv_state_last)
    if (thread < 3 * S) {
        const int part = thread / S;               // 0: q, 1: k, 2: v
        const int i    = thread % S;
        const int c    = part == 0 ? hk * S : part == 1 ? args.H_k * S + hk * S : 2 * args.H_k * S + h * S;
        const int ch   = c + i;

        float x[D_CONV];
        float w[D_CONV];
#pragma unroll
        for (int j = 0; j < D_CONV - 1; j++) {
            x[j] = args.conv_state_in[ch * (D_CONV - 1) + j];
        }
        x[D_CONV - 1] = args.qkv[ch * args.qkv_stride];
#pragma unroll
        for (int j = 0; j < D_CONV; j++) {
            w[j] = args.conv_w[ch * D_CONV + j];
        }

        // ssm_conv_f32: fma chain from 0, then the (zero) bias add, then SiLU
        float sumf = fmaf(x[0], w[0], 0.0f);
#pragma unroll
        for (int j = 1; j < D_CONV; j++) {
            sumf = fmaf(x[j], w[j], sumf);
        }
        sumf += args.conv_bias;
        const float y = ggml_cuda_op_silu_single(sumf);

        if (part == 0) {
            q_c[i] = y;
        } else if (part == 1) {
            k_c[i] = y;
        } else {
            v_c[i] = y;
        }

        // q/k channels are shared by H_v/H_k value heads: the first one writes the conv state
        if (part == 2 || h == hk) {
#pragma unroll
            for (int j = 0; j < D_CONV - 1; j++) {
                args.conv_state_out[ch * (D_CONV - 1) + j] = x[j + 1];
            }
        }
    }
    __syncthreads();

    // 2. q/k norm: l2_norm_dual_f32_s128 for q (warp 0) and k (warp 1), or, when the graph uses the
    //    qwen4exp build_gdn_l2_norm representation RMS_NORM(x, eps) -> SCALE(mul, add), the exact
    //    replay of the dispatched rms_norm_f32<256, false> + scale_f32 pair
    if (!args.qk_rms_scale) {
        if (warp < 2) {
            const float * x = warp == 0 ? q_c : k_c;
            float *       y = warp == 0 ? q_n : k_n;
            float tmp = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const float xi = x[r * warp_size + lane];
                tmp = fmaf(xi, xi, tmp);
            }
            tmp = gdn_warp_reduce_sum<warp_size>(tmp);
            const float scale = rsqrtf(fmaxf(tmp, args.eps_l2 * args.eps_l2));
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                y[r * warp_size + lane] = scale * x[r * warp_size + lane];
            }
        }
    } else {
        // rms_norm_f32<256, false>: 128 elements over 128 lanes (4 warps, one element per lane; warps 4..7
        // of the standalone 256-thread block hold exact zeros that drop out of the tree), the warp
        // butterfly (xor 16/8/4/2/1), the cross-warp tree ((v0 + v2) + (v1 + v3)), mean = sum / 128,
        // scale = rsqrtf(mean + eps), dst = scale * x; then scale_f32: dst = mul * dst + add. The standalone
        // scale_f32 compiles its mul+add to a single FMA (v_fma_f32 in the RDNA3.5 build), so the same
        // contraction is spelled explicitly here.
        const int iqk = warp < 4 ? 0 : 1;
        if (warp < 8) {
            const float * x  = iqk == 0 ? q_c : k_c;
            const float   xi = x[(warp & 3) * warp_size + lane];
            float part = fmaf(xi, xi, 0.0f);
            part = warp_reduce_sum<warp_size>(part);
            if (lane == 0) {
                qk_red[warp] = part;
            }
        }
        __syncthreads();
        const float total = iqk == 0 ? (qk_red[0] + qk_red[2]) + (qk_red[1] + qk_red[3])
                                     : (qk_red[4] + qk_red[6]) + (qk_red[5] + qk_red[7]);
        const float scale = rsqrtf(total / (float) args.S + args.qk_rms_eps[iqk]);
        if (warp < 8) {
            const float * x = iqk == 0 ? q_c : k_c;
            float *       y = iqk == 0 ? q_n : k_n;
            const int     i = (warp & 3) * warp_size + lane;
            const float rms = scale * x[i];
            y[i] = fmaf(args.qk_scale_mul[iqk], rms, args.qk_scale_add[iqk]);
        }
    }

    // gate = softplus(alpha + dt_bias) * A, beta = sigmoid(beta): scalar per head, computed by every thread
    const float alpha_biased = args.alpha[h] + args.dt_bias[h];
    const float gate         = gdn_decode_softplus(alpha_biased) * args.ssm_a[h];
    const float beta_val     = gdn_decode_sigmoid(args.beta[h]);
    const float g_val        = expf(gate);
    __syncthreads();

    // 3. recurrence (gated_delta_net_cuda), cols_per_warp state columns per warp
    {
        const float * state_in  = args.state_cache + (int64_t) args.state_ids[0] * args.state_row_stride + (int64_t) h * S * S;
        float *       state_out = args.state_out + (int64_t) h * S * S;
        const int     colw      = warp * cols_per_warp;

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            k_reg[r] = k_n[r * warp_size + lane];
            q_reg[r] = q_n[r * warp_size + lane];
        }

        float s_shard[cols_per_warp][rows_per_lane];
#pragma unroll
        for (int c = 0; c < cols_per_warp; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[c][r] = state_in[(colw + c) * S + r * warp_size + lane];
            }
        }

        // same explicit contractions as gated_delta_net_tiled_cuda (bit-identical to gated_delta_net_cuda)
        float attn_col[cols_per_warp];
#pragma unroll
        for (int c = 0; c < cols_per_warp; c++) {
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard = fmaf(s_shard[c][r], k_reg[r], kv_shard);
            }
            const float kv_col = gdn_warp_reduce_sum<warp_size>(kv_shard);

            const float delta_col = fmaf(-g_val, kv_col, v_c[colw + c]) * beta_val;

            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[c][r] = fmaf(g_val, s_shard[c][r], k_reg[r] * delta_col);
                attn_partial  = fmaf(s_shard[c][r], q_reg[r], attn_partial);
            }
            attn_col[c] = gdn_warp_reduce_sum<warp_size>(attn_partial);
        }

        if (lane < cols_per_warp) {
            float a = attn_col[0];
#pragma unroll
            for (int c = 1; c < cols_per_warp; c++) {
                a = lane == c ? attn_col[c] : a;
            }
            attn[colw + lane] = a * args.scale;
        }

#pragma unroll
        for (int c = 0; c < cols_per_warp; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                state_out[(colw + c) * S + r * warp_size + lane] = s_shard[c][r];
            }
        }
    }
    __syncthreads();

    // 4. pre-norm attention output to the scratch buffer (the gated rms norm runs as a second kernel: its
    //    destination may alias inputs of this kernel that other blocks are still reading)
    if (thread < S) {
        args.attn_out[(int64_t) h * S + thread] = attn[thread];
    }
}

// rms_norm_f32<128, true>: per-warp partial sums, then the second-level butterfly of block_reduce over
// lanes 0..3 (which reduces to (s0 + s2) + (s1 + s3)), times the norm weight
template <int S>
__global__ void __launch_bounds__(S, 1)
gdn_decode_norm_cuda(const float * attn, const float * norm_w, float * out, const float eps, const float * z) {
    constexpr int warp_size = 32;
    static_assert(S == 4 * warp_size);
    __shared__ float red[4];

    const int h    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int warp = tid / warp_size;
    const int lane = tid % warp_size;

    const float xi   = attn[(int64_t) h * S + tid];
    const float part = gdn_warp_reduce_sum<warp_size>(fmaf(xi, xi, 0.0f));
    if (lane == 0) {
        red[warp] = part;
    }
    __syncthreads();
    const float tmp   = (red[0] + red[2]) + (red[1] + red[3]);
    const float mean  = tmp / (float) S;
    const float scale = rsqrtf(mean + eps);
    const float normed = scale * xi * norm_w[tid];
    if (z != nullptr) {
        // the following sigmoid(z) * normed of unary_gated_op_kernel<op_sigmoid>, same expression
        out[(int64_t) h * S + tid] = (1.0f / (1.0f + expf(-z[(int64_t) h * S + tid]))) * normed;
    } else {
        out[(int64_t) h * S + tid] = normed;
    }
}

void ggml_cuda_op_gdn_decode_fused_prenorm(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_decode_args & args_in, float * attn_scratch) {
    GGML_ASSERT(args_in.S == 128 && args_in.d_conv == 4);
    ggml_cuda_gdn_decode_args args = args_in;
    args.attn_out = attn_scratch;

    const dim3 grid(args.H_v, 1, 1);
    const dim3 block(32, 32, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, block, 0, ctx.stream());
    if (args.n_tokens == 1 && args.K == 1) {
        ggml_cuda_kernel_launch(gdn_decode_fused_cuda<128, 4>, launch_params, args);
        return;
    }

    GGML_ASSERT(args.n_tokens >= 1 && args.n_tokens <= args.K && args.K <= 8);
    const int64_t conv_elems = (args.d_conv - 1) * args.S * (2 * args.H_k + args.H_v);
    for (int64_t slot = args.n_tokens; slot < args.K; ++slot) {
        CUDA_CHECK(cudaMemcpyAsync(args.conv_state_out + slot * args.conv_slot_stride, args.conv_state_in,
            conv_elems * sizeof(float), cudaMemcpyDeviceToDevice, ctx.stream()));
    }
    for (int64_t t = 0; t < args.n_tokens; ++t) {
        const int64_t slot = args.n_tokens - 1 - t;
        ggml_cuda_gdn_decode_args step = args;
        step.qkv += t * args.qkv_token_stride;
        step.alpha += t * args.H_v;
        step.beta += t * args.H_v;
        step.conv_state_out += slot * args.conv_slot_stride;
        step.state_out += slot * args.state_slot_stride;
        step.attn_out += t * args.S * args.H_v;
        if (t > 0) {
            step.conv_state_in = args.conv_state_out + (slot + 1) * args.conv_slot_stride;
            step.state_cache = args.state_out + (slot + 1) * args.state_slot_stride;
            step.state_row_stride = 0;
        }
        ggml_cuda_kernel_launch(gdn_decode_fused_cuda<128, 4>, launch_params, step);
    }
}

void ggml_cuda_op_gdn_decode_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_decode_args & args_in) {
    ggml_cuda_op_gdn_decode_fused_gated(ctx, args_in, nullptr, nullptr);
}

void ggml_cuda_op_gdn_decode_fused_gated(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_decode_args & args_in,
        const float * z, float * gated_out) {
    GGML_ASSERT(args_in.S == 128 && args_in.d_conv == 4);
    ggml_cuda_pool_alloc<float> attn(ctx.pool(), args_in.S * args_in.H_v * args_in.n_tokens);
    ggml_cuda_gdn_decode_args args = args_in;
    args.attn_out = attn.get();
    ggml_cuda_op_gdn_decode_fused_prenorm(ctx, args, attn.get());

    const dim3 grid(args.H_v, 1, 1);
    const dim3 norm_block(128, 1, 1);
    const ggml_cuda_kernel_launch_params norm_params = ggml_cuda_kernel_launch_params(grid, norm_block, 0, ctx.stream());
    // Read all token inputs before the norm output can reuse their storage.
    for (int64_t t = 0; t < args.n_tokens; ++t) {
        const int64_t offset = t * args.S * args.H_v;
        ggml_cuda_kernel_launch(gdn_decode_norm_cuda<128>, norm_params, attn.get() + offset, args.norm_w,
            (z ? gated_out : args.out) + offset, args.eps_rms, z ? z + offset : nullptr);
    }
}
