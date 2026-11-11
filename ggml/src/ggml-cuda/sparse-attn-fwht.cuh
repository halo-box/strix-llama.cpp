#pragma once

// Fast Walsh-Hadamard transform shared by fwht.cu and the in-op rotations of GGML_OP_SPARSE_ATTN (rot_q, rot_o).
// x * scale (scale = 1 / sqrtf(N)), then butterflies from stride 1 up: (x_lo, x_hi) -> (x_lo + x_hi, x_lo - x_hi), the
// order of the CPU op. FP contraction is off so the result bits depend only on x and scale: the in-kernel rotations are
// bitwise equal to fwht.cu.

#include "common.cuh"

// FP contraction off for the enclosing compound statement (must open it); nvcc ignores the pragma (fwht.cu as before)
#if defined(__clang__)
#define SA_FP_CONTRACT_OFF _Pragma("clang fp contract(off)")
#else
#define SA_FP_CONTRACT_OFF
#endif

// fwht_block<N, WS>: N elements over the WS lanes of a warp, value i of lane l is element i * WS + l. Strides below WS
// go through shuffles. All WS lanes must call it.
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

// fwht_row<D, WS>: a row of D elements in the fwht_block layout, transformed over consecutive blocks of rot (64, 128 or
// 256) elements; any other rot leaves x unchanged (rejected before launch). rot is uniform over the warp.
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

// sa_fwht_smem<D>: n_rows rows of D floats in shared memory (row r at x + r * ld), transformed in place over blocks of
// rot by the nt threads of the block, with fwht_block's arithmetic. Every thread must call it with the same arguments; it
// syncs after every stage. For the tiers without a fixed warp layout (portable, G4 fixup, G1 merge).
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
        // pair p: row p / (D / 2), lo the (p % (D / 2))-th index with bit h clear; lo + h stays in the same block
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
