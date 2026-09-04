#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"
#include "unary.cuh"


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
    constexpr int H = 32;
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
                dst[t * S_v * H + col] = attn_col * scale;
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
        if (GGML_CUDA_CC_IS_RDNA3_5(cc) && S_v == 128 && H == 32 && n_tokens >= 16 && n_seqs == 1) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
            ggml_cuda_kernel_launch(gated_delta_net_kda_tiled_128_cuda, launch_params,
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
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
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
// so the outputs are bit-identical to the unfused graph.

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
    __shared__ float red[4];

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

    // 2. l2_norm_dual_f32_s128 for q (warp 0) and k (warp 1)
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
gdn_decode_norm_cuda(const float * attn, const float * norm_w, float * out, const float eps) {
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
    out[(int64_t) h * S + tid] = scale * xi * norm_w[tid];
}

void ggml_cuda_op_gdn_decode_fused_prenorm(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_decode_args & args_in, float * attn_scratch) {
    GGML_ASSERT(args_in.S == 128 && args_in.d_conv == 4);
    ggml_cuda_gdn_decode_args args = args_in;
    args.attn_out = attn_scratch;

    const dim3 grid(args.H_v, 1, 1);
    const dim3 block(32, 32, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, block, 0, ctx.stream());
    ggml_cuda_kernel_launch(gdn_decode_fused_cuda<128, 4>, launch_params, args);
}

void ggml_cuda_op_gdn_decode_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_decode_args & args_in) {
    GGML_ASSERT(args_in.S == 128 && args_in.d_conv == 4);
    ggml_cuda_pool_alloc<float> attn(ctx.pool(), args_in.S * args_in.H_v);
    ggml_cuda_gdn_decode_args args = args_in;
    args.attn_out = attn.get();
    ggml_cuda_op_gdn_decode_fused_prenorm(ctx, args, attn.get());

    const dim3 grid(args.H_v, 1, 1);
    const dim3 norm_block(128, 1, 1);
    const ggml_cuda_kernel_launch_params norm_params = ggml_cuda_kernel_launch_params(grid, norm_block, 0, ctx.stream());
    ggml_cuda_kernel_launch(gdn_decode_norm_cuda<128>, norm_params, attn.get(), args.norm_w, args.out, args.eps_rms);
}
