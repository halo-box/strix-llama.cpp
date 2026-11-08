#pragma once

// The fast Walsh-Hadamard transform shared by fwht.cu (the GGML_HINT_SRC0_IS_HADAMARD mul_mat) and the in-op rotations
// of GGML_OP_SPARSE_ATTN (rot_q prologue, rot_o epilogue; PLAN-v3 §8, D13, W10).
//
// Arithmetic (both forms below): every element is first multiplied by `scale` (the caller passes 1.0f / sqrtf(N), the
// value fwht.cu's host code and the CPU op compute), then the butterflies run from stride 1 up to N / 2; at stride h the
// pair (x_lo, x_hi), lo with bit h clear, becomes (x_lo + x_hi, x_lo - x_hi). This is the order and the operations of
// ggml-cpu's sa_fwht (ggml-cpu/sparse-attn.cpp) and of the hinted mul_mat kernels, so the result is the orthonormal
// Sylvester-Hadamard product H x. FP contraction is off inside both functions: no multiply-add is fused at any call site,
// so the bits of the result depend only on x and scale, which is what makes the in-kernel rotations bitwise equal to
// fwht.cu (tests/test-sparse-attn-rot.cpp).

#include "common.cuh"

// FP contraction off for the enclosing compound statement (it must open it). clang-based compilers (HIP, MUSA, clang
// CUDA) honour the pragma; nvcc does not know it and keeps its default (fwht.cu as before; the sparse-attention kernels
// are HIP only)
#if defined(__clang__)
#define SA_FP_CONTRACT_OFF _Pragma("clang fp contract(off)")
#else
#define SA_FP_CONTRACT_OFF
#endif

// fwht_block<N, WS>: one block of N elements held by the WS lanes of a warp, N / WS values per lane: value i of lane l
// is element i * WS + l. Strides below WS are exchanged between lanes (__shfl_xor_sync), the others stay in the lane's
// registers. All WS lanes of the warp must call it.
template <int N, int WS>
static __device__ __forceinline__ void fwht_block(float * x, const int lane, const float scale) {
SA_FP_CONTRACT_OFF
    static_assert(N >= WS && N % WS == 0, "fwht_block: N must be a multiple of the warp size");
    constexpr int EL = N / WS;

#pragma unroll
    for (int i = 0; i < EL; ++i) {
        x[i] = x[i] * scale;
    }

#pragma unroll
    for (int h = 1; h < WS; h *= 2) {
#pragma unroll
        for (int j = 0; j < EL; j++) {
            const float val  = x[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, WS);

            x[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = WS; h < N; h *= 2) {
        const int step = h / WS;
#pragma unroll
        for (int j = 0; j < EL; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float a = x[j + k];
                const float b = x[j + k + step];

                x[j + k]        = a + b;
                x[j + k + step] = a - b;
            }
        }
    }
}

// fwht_row<D, WS>: one row of D elements held by a warp in the fwht_block layout (value i of lane l = element i * WS + l),
// transformed over its consecutive blocks of rot elements (rot 64, 128 or 256, a multiple of WS dividing D). Block b is
// values [b * rot / WS, (b + 1) * rot / WS) of every lane, so each block is one fwht_block call. rot is uniform over the
// warp; any other value leaves x unchanged (the callers reject it before launch).
template <int D, int WS>
static __device__ __forceinline__ void fwht_row(float * x, const int lane, const int rot, const float scale) {
    static_assert(D % 256 == 0, "fwht_row: D must be a multiple of 256");
    switch (rot) {
        case 64:
            if constexpr (64 >= WS) {
#pragma unroll
                for (int b = 0; b < D / 64; ++b) {
                    fwht_block<64, WS>(x + b * (64 / WS), lane, scale);
                }
            }
            break;
        case 128:
#pragma unroll
            for (int b = 0; b < D / 128; ++b) {
                fwht_block<128, WS>(x + b * (128 / WS), lane, scale);
            }
            break;
        case 256:
#pragma unroll
            for (int b = 0; b < D / 256; ++b) {
                fwht_block<256, WS>(x + b * (256 / WS), lane, scale);
            }
            break;
        default:
            break;
    }
}

// sa_fwht_smem<D>: n_rows rows of D floats in shared memory (row r at x + r * ld), each transformed in place over its
// consecutive blocks of rot elements, by the nt threads of the block (thread t takes elements t, t + nt, ... for the
// scaling, and pairs t, t + nt, ... of every stage). The arithmetic is fwht_block's: x * scale, then at stride h,
// lo = x_lo + x_hi, hi = x_lo - x_hi, from h = 1 up. Every thread of the block must call it with the same arguments
// (it synchronises before the first stage and after every stage, so the rows are complete on return). For the tiers
// without a fixed warp layout: the portable tier (any wave size), the G4 fixup and the G1 split merge.
template <int D>
static __device__ __forceinline__ void sa_fwht_smem(float * x, const int n_rows, const int ld, const int rot, const float scale,
                                                    const int t, const int nt) {
SA_FP_CONTRACT_OFF
    const int n = n_rows * D;
    for (int e = t; e < n; e += nt) {
        float * p = x + (e / D) * ld + e % D;
        *p = *p * scale;
    }
    __syncthreads();
    for (int h = 1; h < rot; h *= 2) {
        // pair p of the stage: row p / (D / 2), lo = the (p % (D / 2))-th index of the row with bit h clear; h < rot and
        // the blocks are aligned, so lo + h is in the same block
        for (int p = t; p < n / 2; p += nt) {
            const int r  = p / (D / 2), pp = p % (D / 2);
            const int lo = (pp / h) * 2 * h + pp % h;
            float * y = x + r * ld;
            const float a = y[lo];
            const float b = y[lo + h];
            y[lo]     = a + b;
            y[lo + h] = a - b;
        }
        __syncthreads();
    }
}
