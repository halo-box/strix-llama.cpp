// EXL3 (exllamav3) linear on RDNA3 / RDNA4: GGML_OP_EXL3_MUL_MAT, see ggml/exl3/README.md.
// y = H(H(x * suh) W_hat) * svh / 128, H the unnormalized 128-point Hadamard, W_hat decoded from the trellis.
//
// Up to 32 rows (decode, MTP verification): two launches per op, the input rotation (which also zeroes this call's
// arrival counters), then the dot kernel. A block of the dot kernel computes T = 2 adjacent 16-column tiles for up to 8
// rows, its warps splitting K; the last block to finish a 128-column segment applies the output rotation and the
// output scale to it.
// More rows (prefill, RDNA3): exl3_rotate_split writes H(x * suh) as f16 hi + lo parts with a power-of-two scale per
// row, then exl3_gemm_kernel decodes 128-column x 32-k slabs of W_hat into LDS once per block and k step and runs f16
// WMMAs on both parts, with the output rotation in registers. This follows exllamav3's dispatch (direct quantized
// kernels for few rows; from 144 rows exllamav3 reconstructs W in fp16 and calls hgemm), but reconstructs per tile
// inside the GEMM instead of the whole weight, and the tile loop is the one of the MMB GEMM in mmb.cu.
//
// Adapted from turboderp-org/exllamav3 at a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a (MIT licence, below):
//   exllamav3/exllamav3_ext/rocm/quant/exl3_gemv_tiles_rdna.cuh   per-lane load plan, tile decoders, the 16-bit
//                                                                  hash multiply, uniform addressing
//   exllamav3/exllamav3_ext/rocm/quant/exl3_gemv_multirow_rdna.cu split K over the warps of a block, M rows per
//                                                                  warp, the output rotation by the last of the 8
//                                                                  tiles of a segment
//   exllamav3/exllamav3_ext/rocm/quant/exl3_gemv_kernel_rdna.cuh  lane layout and quad reduction of the dot core
// No code from QTIP (GPL-3.0) is used. Changes against those files:
// - the activations stay f32: each product is one v_fma_mix_f32 of an f32 activation and the exact fp16 codebook
//   value, where exllamav3 rounds the rotated activations to fp16 for v_dot2_f32_f16. The op stays at f32 rounding
//   against ggml_mul_mat on the f32 W.
// - the Hadamards are unnormalized and the 1/128 is folded into the output scale. The input rotation is bit for
//   bit the one of ggml-cpu/exl3-ops.cpp.
// - the input rotation zeroes the arrival counters of every call; exllamav3 keeps self-resetting counters in a
//   persistent per-device block. No graph patch sites.
// - x may be a strided 4-D tensor. In the dot kernel rows past 8 run in groups of 8, one pass over the weights per
//   group.
// - always T = 2 tiles per wave; the number of warps per block is chosen at launch.
// - the GEMM path keeps f32-grade accuracy (exllamav3 rounds the activations to fp16 for its GEMMs): W_hat is exact in
//   f16 and xh = hi + lo carries ~22 bits, so the op stays at NMSE ~1e-13 against ggml_mul_mat on the f32 W. The two
//   products double the WMMA work of a single f16 product; the WMMA f32 accumulate truncates, so each k step
//   accumulates from zero and is added in f32 (see exl3_gemm_kernel).
//
// exllamav3 licence:
//
// MIT License
//
// Copyright (c) 2025 Turboderp
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "exl3.cuh"

#if defined(GGML_USE_EXL3)

#include "../../exl3/exl3.h"

#include <algorithm>

// the rotations must not contract into FMAs, so that they round as the CPU op does
#pragma clang fp contract(off)

#define EXL3_HAD       128 // Hadamard block
#define EXL3_T         2   // adjacent 16-column tiles per wave
#define EXL3_WARPS_MAX 16  // warps per block of the dot kernel

// half-integer rates K + 0.5 (mul1) ride the bits template slot as 16 + floor(K)
template <int bits>
struct exl3_width {
    static constexpr bool half       = bits > 16;
    static constexpr int  ka         = half ? bits - 16 : bits;
    static constexpr int  tile_bytes = half ? 16*(2*ka + 1) : 32*bits;
};

// the dwords lane L reads from a tile (byte offsets) and the funnel shifts that decode its 8 weights
//   bits 1, 2, 4: 2 dwords, one shift; bits 3: 2 dwords; bits 5, 6, 8: 4; bits 7: 8; half rates: 4
template <int bits>
struct exl3_lane_plan {
    static constexpr int R = exl3_width<bits>::half ? 4 : (bits <= 4 ? 2 : (bits == 7 ? 8 : 4));
    uint32_t off[R];
    int      sh[R/2];
};

static __device__ __forceinline__ int exl3_uni(const int x) {
    return __builtin_amdgcn_readfirstlane(x);
}

template <typename T>
static __device__ __forceinline__ T * exl3_uni_ptr(T * p) {
    const uint64_t v  = (uint64_t) p;
    const uint32_t lo = __builtin_amdgcn_readfirstlane((uint32_t) v);
    const uint32_t hi = __builtin_amdgcn_readfirstlane((uint32_t) (v >> 32));
    return (T *) (((uint64_t) hi << 32) | lo);
}

// keeps a loop-invariant lane offset in a VGPR inside the loop, so the loads use the SGPR base + VGPR offset form
static __device__ __forceinline__ void exl3_opaque(uint32_t & v) {
    asm volatile("" : "+v"(v));
}

typedef float exl3_f2 __attribute__((ext_vector_type(2)));

typedef __attribute__((address_space(1))) const char     exl3_g_char;
typedef __attribute__((address_space(1))) const uint32_t exl3_g_u32;
typedef __attribute__((address_space(1))) const exl3_f2  exl3_g_f2;

static __device__ __forceinline__ uint32_t exl3_ld_u32(exl3_g_char * base, const uint32_t off) {
    return *(exl3_g_u32 *) (base + off);
}

static __device__ __forceinline__ exl3_f2 exl3_ld_f2(exl3_g_char * base, const uint32_t off) {
    return *(exl3_g_f2 *) (base + off);
}

static __device__ __forceinline__ half2 exl3_as_half2(const uint32_t u) {
    return __builtin_bit_cast(half2, u);
}

// x*C (+ D) mod 2^32 for the 16-bit x in the low (HI = false) or high half of src, as two full-rate ops:
// x*C = x*C_lo16 + ((x*C_hi16) mod 2^16) << 16 (v_mul_lo_u32 is quarter rate)
template <uint32_t C, uint32_t D, bool HI>
static __device__ __forceinline__ uint32_t exl3_hmul16(const uint32_t src) {
#if defined(RDNA3) || defined(RDNA4)
    constexpr uint32_t k_hi = C & 0xffff0000u;
    constexpr uint32_t c_lo = C & 0xffffu;
    uint32_t r;
    uint32_t p;
    if constexpr (D == 0) {
        if constexpr (HI) {
            asm("v_pk_mul_lo_u16 %0, %1, %2" : "=v"(r) : "v"(src), "s"(k_hi));
        } else {
            asm("v_pk_mul_lo_u16 %0, %1, %2 op_sel_hi:[0,1]" : "=v"(r) : "v"(src), "s"(k_hi));
        }
    } else {
        if constexpr (HI) {
            asm("v_pk_mad_u16 %0, %1, %2, %3" : "=v"(r) : "v"(src), "s"(k_hi), "s"(D));
        } else {
            asm("v_pk_mad_u16 %0, %1, %2, %3 op_sel_hi:[0,1,1]" : "=v"(r) : "v"(src), "s"(k_hi), "s"(D));
        }
    }
    if constexpr (HI) {
        asm("v_mad_u32_u16 %0, %1, %2, %3 op_sel:[1,0,0,0]" : "=v"(p) : "v"(src), "s"(c_lo), "v"(r));
    } else {
        asm("v_mad_u32_u16 %0, %1, %2, %3" : "=v"(p) : "v"(src), "s"(c_lo), "v"(r));
    }
    return p;
#else
    return (HI ? src >> 16 : src & 0xffffu)*C + D;
#endif // defined(RDNA3) || defined(RDNA4)
}

template <int cb, bool HI>
static __device__ __forceinline__ uint32_t exl3_hash16(const uint32_t src) {
    if constexpr (cb == GGML_EXL3_CODEBOOK_3INST) {
        return exl3_hmul16<89226354u, 64248484u, HI>(src);
    } else if constexpr (cb == GGML_EXL3_CODEBOOK_MCG) {
        return exl3_hmul16<GGML_EXL3_MCG_MULT, 0u, HI>(src);
    } else {
        return exl3_hmul16<GGML_EXL3_MUL1_MULT, 0u, HI>(src);
    }
}

// two hashed states -> their codebook values (lo, hi), each with the single fp16 rounding of ggml_exl3_codebook_value
template <int cb>
static __device__ __forceinline__ half2 exl3_decode_pair(uint32_t p0, uint32_t p1) {
    if constexpr (cb == GGML_EXL3_CODEBOOK_MUL1) {
        // [0x6400 + bytesum(p0) | 0x6400 + bytesum(p1)] = [1024 + s0 | 1024 + s1] as fp16
        uint32_t s = __builtin_amdgcn_sad_u8(p0, 0u, 0x64006400u);
        s = __builtin_amdgcn_sad_hi_u8(p1, 0u, s);
        const half2 k_inv  = __half2half2(__ushort_as_half(0x1eee));
        const half2 k_bias = __half2half2(__ushort_as_half(0xc931));
        return __hfma2(exl3_as_half2(s), k_inv, k_bias);
    } else {
        const half2 h0 = exl3_as_half2((p0 & 0x8fff8fffu) ^ 0x3b603b60u);
        const half2 h1 = exl3_as_half2((p1 & 0x8fff8fffu) ^ 0x3b603b60u);
        return __hadd2(__lows2half2(h0, h1), __highs2half2(h0, h1));
    }
}

// the states are the low 16 bits of src0 and src1 (or the high half, per HI0 / HI1)
template <int cb, bool HI0, bool HI1>
static __device__ __forceinline__ half2 exl3_dq_pair(const uint32_t src0, const uint32_t src1) {
    return exl3_decode_pair<cb>(exl3_hash16<cb, HI0>(src0), exl3_hash16<cb, HI1>(src1));
}

// low 32 bits of the 64-bit a:b shifted right, shift < 64
static __device__ __forceinline__ uint32_t exl3_fshift(const uint32_t b, const uint32_t a, const int shift) {
    return (uint32_t) ((((uint64_t) a << 32) | b) >> shift);
}

template <int bits>
static __device__ __forceinline__ void exl3_dq8_idx(const int t_offset, int & i0, int & i2, int & s2) {
    const int b1 = (t_offset + 257)*bits;
    const int b0 = b1 - 16;
    const int b2 = b1 + bits*7;
    i0 = (b0/32) % (bits*256/32);
    s2 = ((b2 - 1)/32 + 1)*32 - b2;
    i2 = ((b2 - 1)/32) % (bits*256/32);
}

template <int bits>
static __device__ __forceinline__ void exl3_dq4_idx(const int t_offset, int & i0, int & i2, int & s2) {
    const int b0 = (t_offset + 257)*bits - 16;
    const int b2 = b0 + 3*bits + 16;
    i0 = (b0/32) % (bits*256/32);
    s2 = ((b2 - 1)/32 + 1)*32 - b2;
    i2 = ((b2 - 1)/32) % (bits*256/32);
}

template <int bits>
static __device__ __forceinline__ void exl3_dq2_idx(const int t_offset, int & i0, int & i2, int & s2) {
    const int b0 = (t_offset + 257)*bits - 16;
    const int b2 = b0 + bits + 16;
    i0 = (b0/32) % (bits*256/32);
    s2 = ((b2 - 1)/32 + 1)*32 - b2;
    i2 = ((b2 - 1)/32) % (bits*256/32);
}

// lane L decodes ring positions 8L .. 8L + 7: tile rows r0, r0 + 1, r0 + 8, r0 + 9 (r0 = (L%4)*2) of the columns
// L/4 (f0) and L/4 + 8 (f1)
template <int bits>
static __device__ __forceinline__ exl3_lane_plan<bits> exl3_make_lane_plan(const int lane) {
    exl3_lane_plan<bits> p;
    const int t_offset = lane << 3;
    if constexpr (exl3_width<bits>::half) {
        // windows 4..7 and 0..3, two dwords and one funnel shift each
        constexpr int KA    = exl3_width<bits>::ka;
        constexpr int bits2 = 2*KA + 1;
        constexpr int words = 4*bits2;
        constexpr int gspan = 18 + 3*KA;
        const int e7  = ((t_offset >> 1) + 4)*bits2 + 128*bits2;
        const int e3  = e7 - 2*bits2;
        const int hi7 = (e7 - 1)/32;
        const int lo7 = (e7 - gspan)/32;
        const int hi3 = (e3 - 1)/32;
        const int lo3 = (e3 - gspan)/32;
        p.off[0] = (lo7 % words)*4;
        p.off[1] = (hi7 % words)*4;
        p.sh[0]  = (hi7 + 1)*32 - e7;
        p.off[2] = (lo3 % words)*4;
        p.off[3] = (hi3 % words)*4;
        p.sh[1]  = (hi3 + 1)*32 - e3;
    } else if constexpr (bits == 1) {
        const int i1 = t_offset >> 5;
        p.off[1] = i1*4;
        p.off[0] = ((i1 + 7) & 7)*4;
        p.sh[0]  = (~t_offset) & 24;
    } else if constexpr (bits == 2) {
        const int i1 = t_offset >> 4;
        p.off[1] = i1*4;
        p.off[0] = ((i1 + 15) & 15)*4;
        p.sh[0]  = ((~t_offset) & 8) << 1;
    } else if constexpr (bits == 4) {
        const int i1 = t_offset >> 3;
        p.off[1] = i1*4;
        p.off[0] = ((i1 + 31) & 31)*4;
        p.sh[0]  = 20;
    } else if constexpr (bits == 3) {
        int i0, i2, s2;
        exl3_dq8_idx<bits>(t_offset, i0, i2, s2);
        p.off[0] = i0*4;
        p.off[1] = i2*4;
        p.sh[0]  = s2;
    } else if constexpr (bits == 7) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int i0, i2, s2;
                exl3_dq2_idx<bits>(t_offset + 4*h + 2*i, i0, i2, s2);
                p.off[(h*2 + i)*2]     = i0*4;
                p.off[(h*2 + i)*2 + 1] = i2*4;
                p.sh[h*2 + i]          = s2;
            }
        }
    } else {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int i0, i2, s2;
            exl3_dq4_idx<bits>(t_offset + 4*h, i0, i2, s2);
            p.off[h*2]     = i0*4;
            p.off[h*2 + 1] = i2*4;
            p.sh[h]        = s2;
        }
    }
    return p;
}

// positions t, t + 1 from the dwords a (lower index) and b of one window pair (bits 7)
template <int bits, int cb>
static __device__ __forceinline__ half2 exl3_dq2(const uint32_t a, const uint32_t b, const int s2) {
    const uint32_t w1 = exl3_fshift(b, a, s2);
    const uint32_t w0 = exl3_fshift(b, a, s2 + bits);
    return exl3_dq_pair<cb, false, false>(w0, w1);
}

// positions t .. t + 3 (bits 5, 6, 8)
template <int bits, int cb>
static __device__ __forceinline__ void exl3_dq4(const uint32_t a, const uint32_t b, const int s2, half2 * f) {
    const uint32_t w3 = exl3_fshift(b, a, s2);
    const uint32_t w2 = exl3_fshift(b, a, s2 + bits);
    const uint32_t w1 = exl3_fshift(b, a, s2 + bits*2);
    const uint32_t w0 = exl3_fshift(b, a, s2 + bits*3);
    f[0] = exl3_dq_pair<cb, false, false>(w0, w1);
    f[1] = exl3_dq_pair<cb, false, false>(w2, w3);
}

// the 8 weights of a lane from its plan's dwords (raw[2i] is the lower-index dword, the high half of the funnel):
// f0 = (r0, r0 + 1), (r0 + 8, r0 + 9) of column L/4, f1 the same rows of column L/4 + 8
template <int bits, int cb>
static __device__ __forceinline__ void exl3_decode_tile(const uint32_t * raw, const exl3_lane_plan<bits> & lp,
        half2 * f0, half2 * f1) {
    if constexpr (exl3_width<bits>::half) {
        constexpr int KA = exl3_width<bits>::ka;
        const uint32_t w7 = __funnelshift_r(raw[1], raw[0], lp.sh[0]);
        const uint32_t w6 = w7 >> (KA + 1);
        const uint32_t w5 = w6 >> KA;
        const uint32_t w4 = w5 >> (KA + 1);
        const uint32_t w3 = __funnelshift_r(raw[3], raw[2], lp.sh[1]);
        const uint32_t w2 = w3 >> (KA + 1);
        const uint32_t w1 = w2 >> KA;
        const uint32_t w0 = w1 >> (KA + 1);
        f0[0] = exl3_dq_pair<cb, false, false>(w0, w1);
        f0[1] = exl3_dq_pair<cb, false, false>(w2, w3);
        f1[0] = exl3_dq_pair<cb, false, false>(w4, w5);
        f1[1] = exl3_dq_pair<cb, false, false>(w6, w7);
    } else if constexpr (bits == 1) {
        const uint32_t b = __funnelshift_r(raw[1], raw[0], lp.sh[0]);
        f0[0] = exl3_dq_pair<cb, false, false>(b >> 7, b >> 6);
        f0[1] = exl3_dq_pair<cb, false, false>(b >> 5, b >> 4);
        f1[0] = exl3_dq_pair<cb, false, false>(b >> 3, b >> 2);
        f1[1] = exl3_dq_pair<cb, false, false>(b >> 1, b);
    } else if constexpr (bits == 2) {
        const uint32_t b = __funnelshift_r(raw[1], raw[0], lp.sh[0]);
        f0[0] = exl3_dq_pair<cb, false, false>(b >> 14, b >> 12);
        f0[1] = exl3_dq_pair<cb, false, false>(b >> 10, b >> 8);
        f1[0] = exl3_dq_pair<cb, false, false>(b >> 6,  b >> 4);
        f1[1] = exl3_dq_pair<cb, false, false>(b >> 2,  b);
    } else if constexpr (bits == 4) {
        // w0..w2 from s (bits 8, 4, 0), w3..w7 from b (bits 16, 12, 8, 4, 0); w3 is the high half of b
        const uint32_t b = raw[1];
        const uint32_t s = __funnelshift_r(b, raw[0], 20);
        f0[0] = exl3_dq_pair<cb, false, false>(s >> 8,  s >> 4);
        f0[1] = exl3_dq_pair<cb, false, true >(s,       b);
        f1[0] = exl3_dq_pair<cb, false, false>(b >> 12, b >> 8);
        f1[1] = exl3_dq_pair<cb, false, false>(b >> 4,  b);
    } else if constexpr (bits == 3) {
        const uint32_t w7 = exl3_fshift(raw[1], raw[0], lp.sh[0]);
        const uint32_t w6 = w7 >> bits;
        const uint32_t w5 = w6 >> bits;
        const uint32_t w4 = w5 >> bits;
        const uint32_t w3 = exl3_fshift(raw[1], raw[0], lp.sh[0] + bits*4);
        const uint32_t w2 = w3 >> bits;
        const uint32_t w1 = w2 >> bits;
        const uint32_t w0 = w1 >> bits;
        f0[0] = exl3_dq_pair<cb, false, false>(w0, w1);
        f0[1] = exl3_dq_pair<cb, false, false>(w2, w3);
        f1[0] = exl3_dq_pair<cb, false, false>(w4, w5);
        f1[1] = exl3_dq_pair<cb, false, false>(w6, w7);
    } else if constexpr (bits == 7) {
        f0[0] = exl3_dq2<bits, cb>(raw[0], raw[1], lp.sh[0]);
        f0[1] = exl3_dq2<bits, cb>(raw[2], raw[3], lp.sh[1]);
        f1[0] = exl3_dq2<bits, cb>(raw[4], raw[5], lp.sh[2]);
        f1[1] = exl3_dq2<bits, cb>(raw[6], raw[7], lp.sh[3]);
    } else {
        exl3_dq4<bits, cb>(raw[0], raw[1], lp.sh[0], f0);
        exl3_dq4<bits, cb>(raw[2], raw[3], lp.sh[1], f1);
    }
}

// k-tiles whose loads are issued before the first is decoded: exllamav3's table for T = 2 (exl3_tiles_u_splitk),
// not re-measured here
template <int bits, int M>
static constexpr __device__ int exl3_u() {
    if constexpr (M >= 8) {
        return 1;
    } else if constexpr (exl3_width<bits>::half) {
        return 2;
    } else {
        return bits <= 3 ? 2 : 1;
    }
}

// U k-tiles of T tiles for M rows: acc_a[t*M + r] (column L/4) and acc_b (column L/4 + 8) of lane L
template <int bits, int cb, int M, int U>
static __device__ __forceinline__ void exl3_dot_step(
        exl3_g_char * b_c, const uint32_t b_stride, uint32_t * off,
        exl3_g_char * x_c, const uint32_t * x_row, uint32_t & x_off,
        const exl3_lane_plan<bits> & lp, float * acc_a, float * acc_b) {
    constexpr int R          = exl3_lane_plan<bits>::R;
    constexpr int tile_bytes = exl3_width<bits>::tile_bytes;
#pragma unroll
    for (int i = 0; i < R; ++i) {
        exl3_opaque(off[i]);
    }
    exl3_opaque(x_off);

    uint32_t raw[U][EXL3_T][R];
    exl3_f2  xa[U][M];
    exl3_f2  xb[U][M];
#pragma unroll
    for (int u = 0; u < U; ++u) {
#pragma unroll
        for (int t = 0; t < EXL3_T; ++t) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                raw[u][t][i] = exl3_ld_u32(b_c + (size_t) u*b_stride + t*tile_bytes, off[i]);
            }
        }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
#pragma unroll
        for (int r = 0; r < M; ++r) {
            exl3_g_char * xr = x_c + x_row[r] + u*16*sizeof(float);
            xa[u][r] = exl3_ld_f2(xr, x_off);
            xb[u][r] = exl3_ld_f2(xr + 8*sizeof(float), x_off);
        }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
#pragma unroll
        for (int t = 0; t < EXL3_T; ++t) {
            half2 f0[2];
            half2 f1[2];
            exl3_decode_tile<bits, cb>(raw[u][t], lp, f0, f1);
#pragma unroll
            for (int r = 0; r < M; ++r) {
                float & a = acc_a[t*M + r];
                float & b = acc_b[t*M + r];
                a = fmaf(xa[u][r].x, __low2float (f0[0]), a);
                a = fmaf(xa[u][r].y, __high2float(f0[0]), a);
                a = fmaf(xb[u][r].x, __low2float (f0[1]), a);
                a = fmaf(xb[u][r].y, __high2float(f0[1]), a);
                b = fmaf(xa[u][r].x, __low2float (f1[0]), b);
                b = fmaf(xa[u][r].y, __high2float(f1[0]), b);
                b = fmaf(xb[u][r].x, __low2float (f1[1]), b);
                b = fmaf(xb[u][r].y, __high2float(f1[1]), b);
            }
        }
    }
}

// the k-tiles [kb_begin, kb_end) of the T tiles from tile_n0 for M rows of xh (rows_valid of them hold data; the
// others re-read the last one). out[t*M + r]: lane l < 16 holds column l of tile t, row r
template <int bits, int cb, int M>
static __device__ __forceinline__ void exl3_dot(
        const float * xh, const int k, const int rows_valid,
        const int16_t * trellis, const int n_tiles, const int tile_n0,
        const int kb_begin, const int kb_end, const int lane, float * out) {
    xh      = exl3_uni_ptr(xh);
    trellis = exl3_uni_ptr(trellis);
    const int k_u        = exl3_uni(k);
    const int n_tiles_u  = exl3_uni(n_tiles);
    const int tile_n0_u  = exl3_uni(tile_n0);
    const int kb_begin_u = exl3_uni(kb_begin);
    const int kb_end_u   = exl3_uni(kb_end);
    const int row_last   = exl3_uni(rows_valid) - 1;

    constexpr int R          = exl3_lane_plan<bits>::R;
    constexpr int tile_bytes = exl3_width<bits>::tile_bytes;
    constexpr int U          = exl3_u<bits, M>();

    const exl3_lane_plan<bits> lp = exl3_make_lane_plan<bits>(lane);
    uint32_t off[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        off[i] = lp.off[i];
    }
    // (x[r0], x[r0 + 1]) within a k-tile; (x[r0 + 8], x[r0 + 9]) 8 floats further
    uint32_t x_off = (uint32_t) (lane & 3)*2*sizeof(float);

    float acc_a[EXL3_T*M];
    float acc_b[EXL3_T*M];
#pragma unroll
    for (int i = 0; i < EXL3_T*M; ++i) {
        acc_a[i] = 0.0f;
        acc_b[i] = 0.0f;
    }

    const uint32_t b_stride = (uint32_t) n_tiles_u*tile_bytes;
    exl3_g_char * b_c = (exl3_g_char *) (const char *) trellis + ((size_t) kb_begin_u*n_tiles_u + tile_n0_u)*tile_bytes;
    exl3_g_char * x_c = (exl3_g_char *) (const char *) (xh + (size_t) kb_begin_u*16);
    uint32_t x_row[M];
#pragma unroll
    for (int r = 0; r < M; ++r) {
        x_row[r] = (uint32_t) ((r < row_last ? r : row_last)*k_u*sizeof(float));
    }

    int kt = kb_begin_u;
    if constexpr (U > 1) {
        for (; kt + U <= kb_end_u; kt += U) {
            exl3_dot_step<bits, cb, M, U>(b_c, b_stride, off, x_c, x_row, x_off, lp, acc_a, acc_b);
            b_c += (size_t) U*b_stride;
            x_c += U*16*sizeof(float);
        }
    }
    for (; kt < kb_end_u; ++kt) {
        exl3_dot_step<bits, cb, M, 1>(b_c, b_stride, off, x_c, x_row, x_off, lp, acc_a, acc_b);
        b_c += b_stride;
        x_c += 16*sizeof(float);
    }

    // quad q holds the 16 tile rows of columns q (acc_a) and q + 8 (acc_b) between its four lanes
#pragma unroll
    for (int i = 0; i < EXL3_T*M; ++i) {
        acc_a[i] += __shfl_xor(acc_a[i], 1, 32);
        acc_a[i] += __shfl_xor(acc_a[i], 2, 32);
        acc_b[i] += __shfl_xor(acc_b[i], 1, 32);
        acc_b[i] += __shfl_xor(acc_b[i], 2, 32);
        const float va = __shfl(acc_a[i], (lane & 7)*4, 32);
        const float vb = __shfl(acc_b[i], (lane & 7)*4, 32);
        out[i] = (lane & 8) ? vb : va;
    }
}

// unnormalized 128-point Hadamard of v[i] = element i*32 + lane: the stages h = 1, 2, ..., 64 in the order and with
// the roundings of fwht128 in ggml-cpu/exl3-ops.cpp
static __device__ __forceinline__ void exl3_fwht128(float * v, const int lane) {
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o = __shfl_xor(v[i], h, 32);
            v[i] = (lane & h) ? o - v[i] : v[i] + o;
        }
    }
    const float a0 = v[0] + v[1];
    const float a1 = v[0] - v[1];
    const float a2 = v[2] + v[3];
    const float a3 = v[2] - v[3];
    v[0] = a0 + a2;
    v[1] = a1 + a3;
    v[2] = a0 - a2;
    v[3] = a1 - a3;
}

// xh[r] = H(x[r]*suh), one warp per (row, 128 block); zeroes the dot kernel's arrival counters
static __global__ void __launch_bounds__(256) exl3_rotate_in(
        const float * __restrict__ x, const half * __restrict__ suh, float * __restrict__ xh,
        int * __restrict__ counters, const int n_counters, const int k, const int rows,
        const int ne1, const int ne2, const int64_t s1, const int64_t s2, const int64_t s3) {
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for (int i = tid; i < n_counters; i += gridDim.x*blockDim.x) {
        counters[i] = 0;
    }

    const int lane   = threadIdx.x % 32;
    const int w      = tid / 32;
    const int k_segs = k / EXL3_HAD;
    if (w >= rows*k_segs) {
        return;
    }
    const int r   = w / k_segs;
    const int seg = w % k_segs;
    const int i1  = r % ne1;
    const int i2  = (r / ne1) % ne2;
    const int i3  = r / (ne1*ne2);

    const float * xr = x + i1*s1 + i2*s2 + i3*s3 + seg*EXL3_HAD;
    const half  * su = suh + seg*EXL3_HAD;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = xr[i*32 + lane]*__half2float(su[i*32 + lane]);
    }
    exl3_fwht128(v, lane);
    float * dr = xh + (int64_t) r*k + seg*EXL3_HAD;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        dr[i*32 + lane] = v[i];
    }
}

// y = H(y) * svh/128 for one 128-column segment of one row, one warp
static __device__ __forceinline__ void exl3_rotate_out(float * y, const half * svh, const int lane) {
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = y[i*32 + lane];
    }
    exl3_fwht128(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        y[i*32 + lane] = v[i]*(__half2float(svh[i*32 + lane])*(1.0f/EXL3_HAD));
    }
}

// grid (n/16/T, row groups), blockDim.x = 32*warps. counters: one per (row group, 128-column segment), zeroed
template <int bits, int cb, int M>
static __global__ void __launch_bounds__(EXL3_WARPS_MAX*32) exl3_mul_mat_kernel(
        const float * __restrict__ xh, const int16_t * __restrict__ trellis, const half * __restrict__ svh,
        float * dst, int * counters, const int k, const int n, const int rows) {
#if defined(RDNA3) || defined(RDNA4)
    extern __shared__ float sh_red[]; // [warps][T*M][16], with more than one warp

    const int n_warps    = blockDim.x / 32;
    const int warp_id    = exl3_uni(threadIdx.x / 32);
    const int lane       = threadIdx.x % 32;
    const int tile_n0    = blockIdx.x*EXL3_T;
    const int group      = blockIdx.y;
    const int row0       = group*M;
    const int rows_valid = min(M, rows - row0);
    const int k_tiles    = k / 16;
    const int chunk      = (k_tiles + n_warps - 1) / n_warps;
    const int kb0        = min(warp_id*chunk, k_tiles);
    const int kb1        = min(kb0 + chunk, k_tiles);

    float acc[EXL3_T*M];
#pragma unroll
    for (int i = 0; i < EXL3_T*M; ++i) {
        acc[i] = 0.0f;
    }
    if (kb0 < kb1) {
        exl3_dot<bits, cb, M>(xh + (size_t) row0*k, k, rows_valid, trellis, n / 16, tile_n0, kb0, kb1, lane, acc);
    }

    if (n_warps > 1) {
        if (lane < 16) {
#pragma unroll
            for (int i = 0; i < EXL3_T*M; ++i) {
                sh_red[(warp_id*EXL3_T*M + i)*16 + lane] = acc[i];
            }
        }
        __syncthreads();
        if (warp_id != 0) {
            return;
        }
        if (lane < 16) {
#pragma unroll
            for (int i = 0; i < EXL3_T*M; ++i) {
                float s = 0.0f;
                for (int w = 0; w < n_warps; ++w) {
                    s += sh_red[(w*EXL3_T*M + i)*16 + lane];
                }
                acc[i] = s;
            }
        }
    }

    if (lane < 16) {
#pragma unroll
        for (int t = 0; t < EXL3_T; ++t) {
#pragma unroll
            for (int r = 0; r < M; ++r) {
                if (r < rows_valid) {
                    dst[(size_t) (row0 + r)*n + (tile_n0 + t)*16 + lane] = acc[t*M + r];
                }
            }
        }
    }

    // the last block of the segment's 8 tiles rotates it
    const int seg = tile_n0 / (EXL3_HAD/16);
    __threadfence();
    int old = 0;
    if (lane == 0) {
        old = atomicAdd(counters + (size_t) group*(n/EXL3_HAD) + seg, EXL3_T);
    }
    old = __shfl(old, 0, 32);
    if (old + EXL3_T != EXL3_HAD/16) {
        return;
    }
    __threadfence();
    for (int r = 0; r < rows_valid; ++r) {
        exl3_rotate_out(dst + (size_t) (row0 + r)*n + seg*EXL3_HAD, svh + seg*EXL3_HAD, lane);
    }
#else
    GGML_UNUSED_VARS(xh, trellis, svh, dst, counters, k, n, rows);
    NO_DEVICE_CODE;
#endif // defined(RDNA3) || defined(RDNA4)
}

// warps per block (split K): exllamav3's exl3_gemv_splitk_warps rule, not re-measured here
static int exl3_warps(const int64_t k_tiles, const int64_t blocks) {
    if (k_tiles <= 128 || blocks >= 1024) {
        return 4;
    }
    if (blocks < 128 || k_tiles >= 1024) {
        return 16;
    }
    return 8;
}

template <int bits, int cb, int M>
static void exl3_launch_m(const float * xh, const int16_t * trellis, const half * svh, float * dst, int * counters,
        const int64_t k, const int64_t n, const int64_t rows, const int64_t n_groups, cudaStream_t stream) {
    const int64_t n_blocks = n/16/EXL3_T;
    const int     warps    = exl3_warps(k/16, n_blocks*n_groups);
    const size_t  smem     = warps > 1 ? (size_t) warps*EXL3_T*M*16*sizeof(float) : 0;
    const dim3    grid((unsigned) n_blocks, (unsigned) n_groups);
    exl3_mul_mat_kernel<bits, cb, M><<<grid, warps*32, smem, stream>>>(
        xh, trellis, svh, dst, counters, (int) k, (int) n, (int) rows);
}

template <int bits, int cb>
static void exl3_launch(const int M, const float * xh, const int16_t * trellis, const half * svh, float * dst,
        int * counters, const int64_t k, const int64_t n, const int64_t rows, const int64_t n_groups, cudaStream_t stream) {
    switch (M) {
        case 1: exl3_launch_m<bits, cb, 1>(xh, trellis, svh, dst, counters, k, n, rows, n_groups, stream); break;
        case 2: exl3_launch_m<bits, cb, 2>(xh, trellis, svh, dst, counters, k, n, rows, n_groups, stream); break;
        case 4: exl3_launch_m<bits, cb, 4>(xh, trellis, svh, dst, counters, k, n, rows, n_groups, stream); break;
        case 8: exl3_launch_m<bits, cb, 8>(xh, trellis, svh, dst, counters, k, n, rows, n_groups, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

template <int bits>
static void exl3_launch_cb(const int cb, const int M, const float * xh, const int16_t * trellis, const half * svh,
        float * dst, int * counters, const int64_t k, const int64_t n, const int64_t rows, const int64_t n_groups,
        cudaStream_t stream) {
    switch (cb) {
        case GGML_EXL3_CODEBOOK_3INST:
            exl3_launch<bits, GGML_EXL3_CODEBOOK_3INST>(M, xh, trellis, svh, dst, counters, k, n, rows, n_groups, stream);
            break;
        case GGML_EXL3_CODEBOOK_MCG:
            exl3_launch<bits, GGML_EXL3_CODEBOOK_MCG>(M, xh, trellis, svh, dst, counters, k, n, rows, n_groups, stream);
            break;
        case GGML_EXL3_CODEBOOK_MUL1:
            exl3_launch<bits, GGML_EXL3_CODEBOOK_MUL1>(M, xh, trellis, svh, dst, counters, k, n, rows, n_groups, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// ---- GEMM path for more than EXL3_GEMV_ROWS rows (prefill), RDNA3 only ----

// rows up to this run the dot kernel above (a pass over the weights per 8 rows), more rows the GEMM below. 200 linears
// of Qwen3.5-4B (K = 4), gfx1151, op bench ms per call set: 32 rows dot 53.9 / GEMM 65.8, 48 rows 78.6 / 67.0
#define EXL3_GEMV_ROWS 32
#define EXL3_SPLIT_K   32768 // longest k of the GEMM path (exl3_rotate_split holds a row in registers)
#define EXL3_GEMM_BM   128 // output columns per block: one Hadamard segment
#define EXL3_GEMM_BN   128 // rows per block, 16 per warp (32 per warp: 386 ms vs 254 below, spills and half the blocks)
// k per main-loop step, 2 k-tiles. Op bench sum over the 200 linears at 512 rows: 254 ms; k step 64: 267 ms (256 VGPRs,
// spills)
#define EXL3_GEMM_BK   32
#define EXL3_GEMM_LS   (EXL3_GEMM_BK + 8) // LDS row stride in halves

typedef _Float16 exl3_v16h __attribute__((ext_vector_type(16)));
typedef float    exl3_v8f  __attribute__((ext_vector_type(8)));
typedef uint32_t exl3_u4   __attribute__((ext_vector_type(4))); // an array of HIP's uint4 (a union) was kept in scratch

// xh = H(x*suh) per 128 block, then per row a power of two s with max|xh|*s in [2^14, 2^15) (s = 1 for a row without
// a finite nonzero value): hi = f16(xh*s), lo = f16(xh*s - hi), inv[r] = 1/s. The product with W_hat (exact in f16) of
// hi + lo carries ~22 significant bits of xh. One block of 8 warps per row, warp w holds the segments w, w + 8, ...
template <int S>
static __global__ void __launch_bounds__(256) exl3_rotate_split(
        const float * __restrict__ x, const half * __restrict__ suh, half * __restrict__ xhi, half * __restrict__ xlo,
        float * __restrict__ inv, const int k, const int ne1, const int ne2, const int64_t s1, const int64_t s2,
        const int64_t s3) {
    __shared__ float s_max[8];
    const int r    = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int segs = k / EXL3_HAD;
    const int i1   = r % ne1;
    const int i2   = (r / ne1) % ne2;
    const int i3   = r / (ne1*ne2);

    const float * xr = x + i1*s1 + i2*s2 + i3*s3;
    float v[S][4];
    float m = 0.0f;
#pragma unroll
    for (int s = 0; s < S; ++s) {
        const int seg = warp + 8*s;
        if (seg < segs) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v[s][i] = xr[seg*EXL3_HAD + i*32 + lane]*__half2float(suh[seg*EXL3_HAD + i*32 + lane]);
            }
            exl3_fwht128(v[s], lane);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                m = fmaxf(m, fabsf(v[s][i]));
            }
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor(m, o, 32));
    }
    if (lane == 0) {
        s_max[warp] = m;
    }
    __syncthreads();
    m = s_max[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) {
        m = fmaxf(m, s_max[w]);
    }
    // m = f*2^e with f in [0.5, 1): m*2^(15 - e) is in [2^14, 2^15)
    int sh = 0;
    if (m > 0.0f && m <= FLT_MAX) {
        const int e = (int) ((__float_as_uint(m) >> 23) & 0xff) - 126;
        sh = min(max(15 - e, -126), 126);
    }
    const float sc = __uint_as_float((uint32_t) (127 + sh) << 23);

    half * hr = xhi + (int64_t) r*k;
    half * lr = xlo + (int64_t) r*k;
#pragma unroll
    for (int s = 0; s < S; ++s) {
        const int seg = warp + 8*s;
        if (seg < segs) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float t = v[s][i]*sc;
                const half  h = __float2half_rn(t);
                hr[seg*EXL3_HAD + i*32 + lane] = h;
                lr[seg*EXL3_HAD + i*32 + lane] = __float2half_rn(t - __half2float(h));
            }
        }
    }
    if (threadIdx.x == 0) {
        inv[r] = __uint_as_float((uint32_t) (127 - sh) << 23);
    }
}

// v of lane ^ H within each row of 16 lanes (DPP row_xmask, no LDS)
template <int H>
static __device__ __forceinline__ float exl3_xor16(const float v) {
    static_assert(H >= 1 && H < 16, "row_xmask");
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), 0x160 + H, 0xf, 0xf, false));
}

// one butterfly stage of exl3_fwht128 across lanes l, l ^ H (H < 16)
template <int H>
static __device__ __forceinline__ void exl3_fwht_lanes(exl3_v8f * acc, const int lane) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float o = exl3_xor16<H>(acc[i][e]);
            acc[i][e] = (lane & H) ? o - acc[i][e] : acc[i][e] + o;
        }
    }
}

// 16 halves of one LDS row: an f16 WMMA operand (lane l holds row l % 16)
static __device__ __forceinline__ exl3_v16h exl3_frag(const uint16_t * p) {
    const uint4 a = *(const uint4 *) p;
    const uint4 b = *(const uint4 *) (p + 8);
    return __builtin_bit_cast(exl3_v16h, (uint4[2]){a, b});
}

// y = H(xh W_hat)*svh/128 for BM = 128 output columns (one Hadamard segment) and BN = 128 rows per block, with the
// structure of the MMB tile GEMM in mmb.cu (register prefetch of the next k step, one LDS stage, two barriers per
// step). Per k step warp w decodes column tile w of the step's 2 k-tiles into the LDS weight tile, as f16
// [column][k]; the [hi; lo] rows of xh go to LDS as they are. Each warp then holds all 128 columns of its 16 rows, so
// the output Hadamard runs in registers (stages 1..8 across lanes, 16..64 across fragments, the order and roundings of
// exl3_fwht128). Both products, W_hat with xh hi and with xh lo, are f16 WMMAs with f32 accumulation.
template <int bits, int cb>
static __global__ void __launch_bounds__(256) exl3_gemm_kernel(
        const int16_t * __restrict__ trellis, const half * __restrict__ xhi, const half * __restrict__ xlo,
        const float * __restrict__ inv, const half * __restrict__ svh, float * __restrict__ dst, const int k,
        const int n, const int rows) {
#if defined(RDNA3)
    constexpr int BM         = EXL3_GEMM_BM;
    constexpr int BN         = EXL3_GEMM_BN;
    constexpr int BK         = EXL3_GEMM_BK;
    constexpr int LS         = EXL3_GEMM_LS;
    constexpr int R          = exl3_lane_plan<bits>::R;
    constexpr int tile_bytes = exl3_width<bits>::tile_bytes;
    constexpr int KT         = BK/16;          // k-tiles per step
    constexpr int B_ITEMS    = 2*BN*BK/8/256;  // 16-byte chunks of xh hi and lo per thread per step
    static_assert(BM == 8*16 && BN == 8*16, "one column tile and 16 rows per warp");

    __shared__ __align__(16) uint16_t As[BM*LS];
    __shared__ __align__(16) uint16_t Bs[2*BN*LS];

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int n0   = blockIdx.x*BM;
    const int t0   = blockIdx.y*BN;
    const int n_steps = k / BK;
    const uint32_t kt_stride = (uint32_t) (n/16)*tile_bytes;

    const exl3_lane_plan<bits> lp = exl3_make_lane_plan<bits>(lane);
    exl3_g_char * tq = (exl3_g_char *) exl3_uni_ptr((const char *) trellis + (size_t) (n0/16 + warp)*tile_bytes);

    // weight stores: lane L holds rows r0, r0 + 1, r0 + 8, r0 + 9 (k) of columns c and c + 8 of its tile
    const int c  = lane / 4;
    const int r0 = (lane % 4)*2;
    uint32_t * a_st0 = (uint32_t *) (As + (warp*16 + c    )*LS + r0);
    uint32_t * a_st1 = (uint32_t *) (As + (warp*16 + c + 8)*LS + r0);

    // activations: item i of a thread is 8 halves of row (tid + 256*i) / (BK/8) of the [hi; lo] tile. Rows past the end
    // re-read the last row (their outputs are not stored): the loads stay unguarded, so that the compiler does not
    // drain them at a divergent join
    const exl3_u4 * xsrc[B_ITEMS];
    exl3_u4       * xdst[B_ITEMS];
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) {
        const int it  = tid + i*256;
        const int row = min(t0 + (it / (BK/8)) % BN, rows - 1);
        xsrc[i] = (const exl3_u4 *) ((i < B_ITEMS/2 ? xhi : xlo) + (int64_t) row*k + (it % (BK/8))*8);
        xdst[i] = (exl3_u4 *) (Bs + (it / (BK/8))*LS + (it % (BK/8))*8);
    }

    uint32_t raw[KT][R];
    exl3_u4  bst[B_ITEMS];
#define EXL3_GEMM_LOAD(ks_)                                                       \
    do {                                                                          \
        _Pragma("unroll")                                                         \
        for (int j = 0; j < KT; ++j) {                                            \
            exl3_g_char * tp = tq + (size_t) ((ks_)*KT + j)*kt_stride;            \
            _Pragma("unroll")                                                     \
            for (int i = 0; i < R; ++i) {                                         \
                raw[j][i] = exl3_ld_u32(tp, lp.off[i]);                           \
            }                                                                     \
        }                                                                         \
        _Pragma("unroll")                                                         \
        for (int i = 0; i < B_ITEMS; ++i) {                                       \
            bst[i] = xsrc[i][(ks_)*(BK/8)];                                       \
        }                                                                         \
    } while (0)
#define EXL3_GEMM_STORE()                                                         \
    do {                                                                          \
        _Pragma("unroll")                                                         \
        for (int j = 0; j < KT; ++j) {                                            \
            half2 f0[2];                                                          \
            half2 f1[2];                                                          \
            exl3_decode_tile<bits, cb>(raw[j], lp, f0, f1);                       \
            a_st0[j*8]     = __builtin_bit_cast(uint32_t, f0[0]);                 \
            a_st0[j*8 + 4] = __builtin_bit_cast(uint32_t, f0[1]);                 \
            a_st1[j*8]     = __builtin_bit_cast(uint32_t, f1[0]);                 \
            a_st1[j*8 + 4] = __builtin_bit_cast(uint32_t, f1[1]);                 \
        }                                                                         \
        _Pragma("unroll")                                                         \
        for (int i = 0; i < B_ITEMS; ++i) {                                       \
            *xdst[i] = bst[i];                                                    \
        }                                                                         \
    } while (0)

    // The WMMA f32 accumulate truncates (KB 0fb3ba63: -7e-6 relative bias over k = 2560 when a whole dot product
    // accumulates inside it). Here the WMMAs of one k step accumulate from zero into p, and p is added to acc with a
    // round-to-nearest f32 add. Container shapes, rows 16..513, NMSE against ggml_mul_mat (CPU, f32 W) when all of k
    // accumulated inside the WMMA: 2.5e-10 (k = 2560) .. 3.2e-9 (k = 9216), growing as k^2.
    exl3_v8f acc[8];
    const exl3_v8f zero = {};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc[i] = zero;
    }

    EXL3_GEMM_LOAD(0);
    EXL3_GEMM_STORE();
    __syncthreads();
    for (int ks = 0; ks < n_steps; ++ks) {
        // the last step reloads itself: no branch around the prefetch. The scheduling barrier keeps the loads ahead
        // of the WMMAs: left alone the compiler sank them to the end of the step and then waited for them (sum over
        // the 200 linears 265 ms, 254 with both barriers)
        EXL3_GEMM_LOAD(min(ks + 1, n_steps - 1));
        __builtin_amdgcn_sched_barrier(0);
        // the x fragments of the step, then two column fragments at a time over the whole step: only two partial sums
        // are live, and no WMMA waits on the one just issued
        const int r = lane % 16;
        exl3_v16h bh[KT];
        exl3_v16h bl[KT];
#pragma unroll
        for (int j = 0; j < KT; ++j) {
            bh[j] = exl3_frag(Bs + (     warp*16 + r)*LS + j*16);
            bl[j] = exl3_frag(Bs + (BN + warp*16 + r)*LS + j*16);
        }
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            exl3_v8f p0 = zero;
            exl3_v8f p1 = zero;
#pragma unroll
            for (int j = 0; j < KT; ++j) {
                const exl3_v16h a0 = exl3_frag(As + ( i     *16 + r)*LS + j*16);
                const exl3_v16h a1 = exl3_frag(As + ((i + 1)*16 + r)*LS + j*16);
                p0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bh[j], a0, p0);
                p1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bh[j], a1, p1);
                p0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bl[j], a0, p0);
                p1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bl[j], a1, p1);
            }
            acc[i]     += p0;
            acc[i + 1] += p1;
        }
        // and the decode of the next step after them, when its loads have landed
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();
        EXL3_GEMM_STORE();
        __syncthreads();
    }
#undef EXL3_GEMM_LOAD
#undef EXL3_GEMM_STORE

    // acc[i][e]: row t0 + warp*16 + 2*e + lane/16, column n0 + i*16 + lane%16
    exl3_fwht_lanes<1>(acc, lane);
    exl3_fwht_lanes<2>(acc, lane);
    exl3_fwht_lanes<4>(acc, lane);
    exl3_fwht_lanes<8>(acc, lane);
#pragma unroll
    for (int hb = 1; hb < 8; hb <<= 1) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i & hb) {
                continue;
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float a = acc[i][e];
                const float b = acc[i | hb][e];
                acc[i][e]      = a + b;
                acc[i | hb][e] = a - b;
            }
        }
    }
    float sv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sv[i] = __half2float(svh[n0 + i*16 + lane % 16])*(1.0f/EXL3_HAD);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int t = t0 + warp*16 + 2*e + lane/16;
        if (t < rows) {
            const float s = inv[t];
            float * y = dst + (int64_t) t*n + n0 + lane % 16;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                y[i*16] = acc[i][e]*(sv[i]*s);
            }
        }
    }
#else
    GGML_UNUSED_VARS(trellis, xhi, xlo, inv, svh, dst, k, n, rows);
    NO_DEVICE_CODE;
#endif // defined(RDNA3)
}

template <int bits, int cb>
static void exl3_gemm_launch(const int16_t * trellis, const half * xhi, const half * xlo, const float * inv,
        const half * svh, float * dst, const int64_t k, const int64_t n, const int64_t rows, cudaStream_t stream) {
    const dim3 grid((unsigned) (n/EXL3_GEMM_BM), (unsigned) ((rows + EXL3_GEMM_BN - 1)/EXL3_GEMM_BN));
    exl3_gemm_kernel<bits, cb><<<grid, 256, 0, stream>>>(trellis, xhi, xlo, inv, svh, dst, (int) k, (int) n, (int) rows);
}

template <int bits>
static void exl3_gemm_launch_cb(const int cb, const int16_t * trellis, const half * xhi, const half * xlo,
        const float * inv, const half * svh, float * dst, const int64_t k, const int64_t n, const int64_t rows,
        cudaStream_t stream) {
    switch (cb) {
        case GGML_EXL3_CODEBOOK_3INST:
            exl3_gemm_launch<bits, GGML_EXL3_CODEBOOK_3INST>(trellis, xhi, xlo, inv, svh, dst, k, n, rows, stream);
            break;
        case GGML_EXL3_CODEBOOK_MCG:
            exl3_gemm_launch<bits, GGML_EXL3_CODEBOOK_MCG>(trellis, xhi, xlo, inv, svh, dst, k, n, rows, stream);
            break;
        case GGML_EXL3_CODEBOOK_MUL1:
            exl3_gemm_launch<bits, GGML_EXL3_CODEBOOK_MUL1>(trellis, xhi, xlo, inv, svh, dst, k, n, rows, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static void exl3_rotate_split_launch(const ggml_tensor * x, const half * suh, half * xhi, half * xlo, float * inv,
        const int64_t k, const int64_t rows, cudaStream_t stream) {
    const int64_t segs_per_warp = (k/EXL3_HAD + 7)/8;
    const float * xd = (const float *) x->data;
    const int     ne1 = (int) x->ne[1];
    const int     ne2 = (int) x->ne[2];
    const int64_t s1  = x->nb[1]/sizeof(float);
    const int64_t s2  = x->nb[2]/sizeof(float);
    const int64_t s3  = x->nb[3]/sizeof(float);
    const dim3 grid((unsigned) rows);
    if (segs_per_warp <= 4) {
        exl3_rotate_split<4><<<grid, 256, 0, stream>>>(xd, suh, xhi, xlo, inv, (int) k, ne1, ne2, s1, s2, s3);
    } else if (segs_per_warp <= 8) {
        exl3_rotate_split<8><<<grid, 256, 0, stream>>>(xd, suh, xhi, xlo, inv, (int) k, ne1, ne2, s1, s2, s3);
    } else if (segs_per_warp <= 16) {
        exl3_rotate_split<16><<<grid, 256, 0, stream>>>(xd, suh, xhi, xlo, inv, (int) k, ne1, ne2, s1, s2, s3);
    } else {
        GGML_ASSERT(segs_per_warp <= 32);
        exl3_rotate_split<32><<<grid, 256, 0, stream>>>(xd, suh, xhi, xlo, inv, (int) k, ne1, ne2, s1, s2, s3);
    }
}

bool ggml_cuda_exl3_supports_op(const int device, const ggml_tensor * op) {
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_RDNA4(cc)) {
        return false;
    }
    const ggml_tensor * trellis = op->src[0];
    const ggml_tensor * x       = op->src[1];
    const int           cb      = ggml_get_op_params_i32(op, 0);
    return op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && x->type == GGML_TYPE_F32 &&
           x->nb[0] == sizeof(float) && x->nb[1] % sizeof(float) == 0 && x->nb[2] % sizeof(float) == 0 &&
           x->nb[3] % sizeof(float) == 0 && ggml_exl3_rate_supported(trellis->ne[0], (ggml_exl3_codebook) cb);
}

void ggml_cuda_op_exl3_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * trellis = dst->src[0];
    const ggml_tensor * x       = dst->src[1];
    const ggml_tensor * suh     = dst->src[2];
    const ggml_tensor * svh     = dst->src[3];

    GGML_ASSERT(ggml_cuda_exl3_supports_op(ctx.device, dst));

    const int     cb         = ggml_get_op_params_i32(dst, 0);
    const int64_t tile_words = trellis->ne[0];
    const int64_t n          = trellis->ne[1]*16;
    const int64_t k          = trellis->ne[2]*16;
    const int64_t rows       = ggml_nrows(x);
    if (rows == 0) {
        return;
    }

    cudaStream_t stream = ctx.stream();

    const int16_t * tq   = (const int16_t *) trellis->data;
    const half    * sv   = (const half *) svh->data;
    float         * y    = (float *) dst->data;
    const bool      half_rate = tile_words % 16 != 0;
    const int       bits = half_rate ? 16 + (int) (tile_words - 8) / 16 : (int) tile_words / 16;

    if (rows > EXL3_GEMV_ROWS && k <= EXL3_SPLIT_K && GGML_CUDA_CC_IS_RDNA3(ggml_cuda_info().devices[ctx.device].cc)) {
        GGML_ASSERT((rows + EXL3_GEMM_BN - 1)/EXL3_GEMM_BN <= 65535);
        ggml_cuda_pool_alloc<half>  xhi(ctx.pool(), rows*k);
        ggml_cuda_pool_alloc<half>  xlo(ctx.pool(), rows*k);
        ggml_cuda_pool_alloc<float> inv(ctx.pool(), rows);
        exl3_rotate_split_launch(x, (const half *) suh->data, xhi.get(), xlo.get(), inv.get(), k, rows, stream);
        switch (bits) {
            case 1: exl3_gemm_launch_cb<1>(cb, tq, xhi.get(), xlo.get(), inv.get(), sv, y, k, n, rows, stream); break;
            case 2: exl3_gemm_launch_cb<2>(cb, tq, xhi.get(), xlo.get(), inv.get(), sv, y, k, n, rows, stream); break;
            case 3: exl3_gemm_launch_cb<3>(cb, tq, xhi.get(), xlo.get(), inv.get(), sv, y, k, n, rows, stream); break;
            case 4: exl3_gemm_launch_cb<4>(cb, tq, xhi.get(), xlo.get(), inv.get(), sv, y, k, n, rows, stream); break;
            case 5: exl3_gemm_launch_cb<5>(cb, tq, xhi.get(), xlo.get(), inv.get(), sv, y, k, n, rows, stream); break;
            case 6: exl3_gemm_launch_cb<6>(cb, tq, xhi.get(), xlo.get(), inv.get(), sv, y, k, n, rows, stream); break;
            case 7: exl3_gemm_launch_cb<7>(cb, tq, xhi.get(), xlo.get(), inv.get(), sv, y, k, n, rows, stream); break;
            case 8: exl3_gemm_launch_cb<8>(cb, tq, xhi.get(), xlo.get(), inv.get(), sv, y, k, n, rows, stream); break;
            case 17: exl3_gemm_launch<17, GGML_EXL3_CODEBOOK_MUL1>(tq, xhi.get(), xlo.get(), inv.get(), sv, y, k, n, rows, stream); break;
            case 18: exl3_gemm_launch<18, GGML_EXL3_CODEBOOK_MUL1>(tq, xhi.get(), xlo.get(), inv.get(), sv, y, k, n, rows, stream); break;
            case 19: exl3_gemm_launch<19, GGML_EXL3_CODEBOOK_MUL1>(tq, xhi.get(), xlo.get(), inv.get(), sv, y, k, n, rows, stream); break;
            default: GGML_ABORT("fatal error");
        }
        return;
    }

    // row tile: the smallest of 1, 2, 4, 8 that holds the rows; more rows run in groups of 8
    const int     M        = rows == 1 ? 1 : rows == 2 ? 2 : rows <= 4 ? 4 : 8;
    const int64_t n_groups = (rows + M - 1) / M;
    const int64_t n_segs   = n / EXL3_HAD;
    GGML_ASSERT(n_groups <= 65535);

    ggml_cuda_pool_alloc<float> xh(ctx.pool(), rows*k);
    ggml_cuda_pool_alloc<int>   counters(ctx.pool(), n_groups*n_segs);

    const int64_t threads_in = rows*(k/EXL3_HAD)*32;
    exl3_rotate_in<<<(unsigned) ((threads_in + 255) / 256), 256, 0, stream>>>(
        (const float *) x->data, (const half *) suh->data, xh.get(), counters.get(), (int) (n_groups*n_segs), (int) k,
        (int) rows, (int) x->ne[1], (int) x->ne[2], x->nb[1] / sizeof(float), x->nb[2] / sizeof(float),
        x->nb[3] / sizeof(float));

    switch (bits) {
        case 1: exl3_launch_cb<1>(cb, M, xh.get(), tq, sv, y, counters.get(), k, n, rows, n_groups, stream); break;
        case 2: exl3_launch_cb<2>(cb, M, xh.get(), tq, sv, y, counters.get(), k, n, rows, n_groups, stream); break;
        case 3: exl3_launch_cb<3>(cb, M, xh.get(), tq, sv, y, counters.get(), k, n, rows, n_groups, stream); break;
        case 4: exl3_launch_cb<4>(cb, M, xh.get(), tq, sv, y, counters.get(), k, n, rows, n_groups, stream); break;
        case 5: exl3_launch_cb<5>(cb, M, xh.get(), tq, sv, y, counters.get(), k, n, rows, n_groups, stream); break;
        case 6: exl3_launch_cb<6>(cb, M, xh.get(), tq, sv, y, counters.get(), k, n, rows, n_groups, stream); break;
        case 7: exl3_launch_cb<7>(cb, M, xh.get(), tq, sv, y, counters.get(), k, n, rows, n_groups, stream); break;
        case 8: exl3_launch_cb<8>(cb, M, xh.get(), tq, sv, y, counters.get(), k, n, rows, n_groups, stream); break;
        // half rates K + 0.5 exist for mul1 only
        case 17: exl3_launch<17, GGML_EXL3_CODEBOOK_MUL1>(M, xh.get(), tq, sv, y, counters.get(), k, n, rows, n_groups, stream); break;
        case 18: exl3_launch<18, GGML_EXL3_CODEBOOK_MUL1>(M, xh.get(), tq, sv, y, counters.get(), k, n, rows, n_groups, stream); break;
        case 19: exl3_launch<19, GGML_EXL3_CODEBOOK_MUL1>(M, xh.get(), tq, sv, y, counters.get(), k, n, rows, n_groups, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

#endif // defined(GGML_USE_EXL3)
