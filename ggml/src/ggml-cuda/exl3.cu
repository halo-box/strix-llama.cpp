// EXL3 (exllamav3) linear on RDNA3 / RDNA4: GGML_OP_EXL3_MUL_MAT, see ggml/exl3/README.md.
// y = H(H(x * suh) W_hat) * svh / 128, H the unnormalized 128-point Hadamard, W_hat decoded from the trellis.
//
// Up to 32 rows (decode, MTP verification): two launches per op, the input rotation (which also zeroes this call's
// arrival counters), then the dot kernel. A block of the dot kernel computes T = 2 adjacent 16-column tiles for up to 8
// rows, its warps splitting K; the last block to finish a 128-column segment applies the output rotation and the
// output scale to it.
// More rows (prefill, RDNA3): exl3_rotate_split writes H(x * suh) in f16 (hi, plus the lo part when two products are
// needed) with a power-of-two scale per row, then a GEMM decodes 128-column x 32-k slabs of W_hat into LDS once per
// block and k step and runs f16 WMMAs, with the output rotation in registers: exl3_gemm1_kernel (one product, 256 rows
// per block) or exl3_gemm_kernel (hi + lo, f32 grade); exl3_gemm_products picks one. This follows exllamav3's dispatch
// (direct quantized kernels for few rows; from 144 rows exllamav3 reconstructs W in fp16 and calls hgemm), but decodes
// per tile inside the GEMM instead of the whole weight, and the tile loop is the one of the MMB GEMM in mmb.cu.
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
// - the GEMM path rounds only the rotated activations (W_hat is exact in f16), and only where this backend context's
//   own F32-weight GEMM rounds both operands to f16 (exl3_gemm_products); elsewhere xh = hi + lo carries ~22 bits and
//   the op stays at NMSE ~1e-13 against ggml_mul_mat on the f32 W, at twice the WMMA work. In that path the WMMA f32
//   accumulate truncates, so each k step accumulates from zero and is added in f32 (see exl3_gemm_kernel).
// - exllamav3's fused expert kernel merges the gate/up output transform, the SwiGLU and the down input transform; here
//   only the SwiGLU moves into the down projection's input rotation (graph fusion swiglu -> EXL3_MUL_MAT_ID in
//   ggml-cuda.cu, GEMM path only), computed with ggml's gated kernel's f32 roundings, so the rows are unchanged.
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
#include "mmb.cuh"
#include "unary.cuh"

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

// EXL3_MUL_MAT_ID: the rows of the op are the (slot, token) pairs p = slot + n_used*token, in expert order. A group is
// consecutive rows of one expert; grp (the group table of one launch) holds (expert, first row, rows) per group, rows = 0
// past the table's last group. Without grp (one token) row r is pair r and group r, its expert ids[r].
struct exl3_mmid {
    const char * ids;       // I32 [n_used, n_tokens]
    int64_t      ids_nb0;
    int64_t      ids_nb1;
    int          n_used;
    int          ne11;      // x rows per token: 1 (broadcast over the slots) or n_used
    int64_t      tq_stride; // int16 words of one expert's trellis
    const int  * order;     // row -> pair
    const int  * grp;
};

static __device__ __forceinline__ int exl3_mmid_expert(const exl3_mmid & mm, const int p) {
    return *(const int *) (mm.ids + (p % mm.n_used)*mm.ids_nb0 + (p / mm.n_used)*mm.ids_nb1);
}

static __device__ __forceinline__ int exl3_mmid_pair(const exl3_mmid & mm, const int r) {
    return mm.order ? mm.order[r] : r;
}

static __device__ __forceinline__ void exl3_mmid_group(const exl3_mmid & mm, const int g, int & e, int & row0, int & rows) {
    if (mm.grp) {
        e    = mm.grp[3*g];
        row0 = mm.grp[3*g + 1];
        rows = mm.grp[3*g + 2];
    } else {
        e    = exl3_mmid_expert(mm, g);
        row0 = g;
        rows = 1;
    }
}

// x row of pair p
static __device__ __forceinline__ const float * exl3_mmid_x(const exl3_mmid & mm, const float * x, const int p,
        const int64_t s1, const int64_t s2) {
    return x + ((p % mm.n_used) % mm.ne11)*s1 + (p / mm.n_used)*s2;
}

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

// xh[r] = H(x[r]*suh), one warp per (row, 128 block); zeroes the dot kernel's arrival counters. ID: row r is the x row
// of its pair, with the suh of the pair's expert
template <bool ID>
static __global__ void __launch_bounds__(256) exl3_rotate_in(
        const float * __restrict__ x, const half * __restrict__ suh, float * __restrict__ xh,
        int * __restrict__ counters, const int n_counters, const int k, const int rows,
        const int ne1, const int ne2, const int64_t s1, const int64_t s2, const int64_t s3, const exl3_mmid mm) {
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

    const float * xr;
    const half  * su;
    if constexpr (ID) {
        const int p = exl3_mmid_pair(mm, r);
        xr = exl3_mmid_x(mm, x, p, s1, s2) + seg*EXL3_HAD;
        su = suh + (int64_t) exl3_mmid_expert(mm, p)*k + seg*EXL3_HAD;
    } else {
        const int i1 = r % ne1;
        const int i2 = (r / ne1) % ne2;
        const int i3 = r / (ne1*ne2);
        xr = x + i1*s1 + i2*s2 + i3*s3 + seg*EXL3_HAD;
        su = suh + seg*EXL3_HAD;
    }
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

// grid (n/16/T, row groups), blockDim.x = 32*warps. counters: one per (row group, 128-column segment), zeroed.
// ID: the groups of mm, each with its expert's trellis and svh; row r is stored to the dst row of its pair
template <int bits, int cb, int M, bool ID>
static __global__ void __launch_bounds__(EXL3_WARPS_MAX*32) exl3_mul_mat_kernel(
        const float * __restrict__ xh, const int16_t * __restrict__ trellis, const half * __restrict__ svh,
        float * dst, int * counters, const int k, const int n, const int rows, const exl3_mmid mm) {
#if defined(RDNA3) || defined(RDNA4)
    extern __shared__ float sh_red[]; // [warps][T*M][16], with more than one warp

    const int n_warps    = blockDim.x / 32;
    const int warp_id    = exl3_uni(threadIdx.x / 32);
    const int lane       = threadIdx.x % 32;
    const int tile_n0    = blockIdx.x*EXL3_T;
    const int group      = blockIdx.y;
    int row0       = group*M;
    int rows_valid = min(M, rows - row0);
    if constexpr (ID) {
        int e;
        exl3_mmid_group(mm, group, e, row0, rows_valid);
        if (rows_valid == 0) {
            return;
        }
        trellis += e*mm.tq_stride;
        svh     += (int64_t) e*n;
    }
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
                    const int64_t dr = ID ? exl3_mmid_pair(mm, row0 + r) : row0 + r;
                    dst[(size_t) dr*n + (tile_n0 + t)*16 + lane] = acc[t*M + r];
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
        const int64_t dr = ID ? exl3_mmid_pair(mm, row0 + r) : row0 + r;
        exl3_rotate_out(dst + (size_t) dr*n + seg*EXL3_HAD, svh + seg*EXL3_HAD, lane);
    }
#else
    GGML_UNUSED_VARS(xh, trellis, svh, dst, counters, k, n, rows, mm);
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

// mm: the groups of an EXL3_MUL_MAT_ID (nullptr for EXL3_MUL_MAT)
template <int bits, int cb, int M>
static void exl3_launch_m(const float * xh, const int16_t * trellis, const half * svh, float * dst, int * counters,
        const int64_t k, const int64_t n, const int64_t rows, const int64_t n_groups, const exl3_mmid * mm,
        cudaStream_t stream) {
    const int64_t n_blocks = n/16/EXL3_T;
    const int     warps    = exl3_warps(k/16, n_blocks*n_groups);
    const size_t  smem     = warps > 1 ? (size_t) warps*EXL3_T*M*16*sizeof(float) : 0;
    const dim3    grid((unsigned) n_blocks, (unsigned) n_groups);
    if (mm != nullptr) {
        exl3_mul_mat_kernel<bits, cb, M, true><<<grid, warps*32, smem, stream>>>(
            xh, trellis, svh, dst, counters, (int) k, (int) n, (int) rows, *mm);
    } else {
        exl3_mul_mat_kernel<bits, cb, M, false><<<grid, warps*32, smem, stream>>>(
            xh, trellis, svh, dst, counters, (int) k, (int) n, (int) rows, exl3_mmid {});
    }
}

template <int bits, int cb>
static void exl3_launch(const int M, const float * xh, const int16_t * trellis, const half * svh, float * dst,
        int * counters, const int64_t k, const int64_t n, const int64_t rows, const int64_t n_groups,
        const exl3_mmid * mm, cudaStream_t stream) {
    switch (M) {
        case 1: exl3_launch_m<bits, cb, 1>(xh, trellis, svh, dst, counters, k, n, rows, n_groups, mm, stream); break;
        case 2: exl3_launch_m<bits, cb, 2>(xh, trellis, svh, dst, counters, k, n, rows, n_groups, mm, stream); break;
        case 4: exl3_launch_m<bits, cb, 4>(xh, trellis, svh, dst, counters, k, n, rows, n_groups, mm, stream); break;
        case 8: exl3_launch_m<bits, cb, 8>(xh, trellis, svh, dst, counters, k, n, rows, n_groups, mm, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

template <int bits>
static void exl3_launch_cb(const int cb, const int M, const float * xh, const int16_t * trellis, const half * svh,
        float * dst, int * counters, const int64_t k, const int64_t n, const int64_t rows, const int64_t n_groups,
        const exl3_mmid * mm, cudaStream_t stream) {
    switch (cb) {
        case GGML_EXL3_CODEBOOK_3INST:
            exl3_launch<bits, GGML_EXL3_CODEBOOK_3INST>(M, xh, trellis, svh, dst, counters, k, n, rows, n_groups, mm, stream);
            break;
        case GGML_EXL3_CODEBOOK_MCG:
            exl3_launch<bits, GGML_EXL3_CODEBOOK_MCG>(M, xh, trellis, svh, dst, counters, k, n, rows, n_groups, mm, stream);
            break;
        case GGML_EXL3_CODEBOOK_MUL1:
            exl3_launch<bits, GGML_EXL3_CODEBOOK_MUL1>(M, xh, trellis, svh, dst, counters, k, n, rows, n_groups, mm, stream);
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
// rows per block, 16 per warp (32 per warp: 386 ms vs 254 below, spills and half the blocks). The hi + lo GEMM takes
// blocks of half that (W = 4 warps) when its last block would be at most half full. 200 linears of Qwen3.5-4B (K = 4),
// gfx1151, op bench ms with 128- vs 64-row blocks: 33 rows 62.9 vs 40.4, 64 65.3 vs 44.1, 96 68.0 vs 75.5, 128 70.4 vs
// 77.6, 160 127.0 vs 106.1, 192 135.5 vs 112.8, 256 141.3 vs 149.3, 320 198.6 vs 176.6, 384 209.2 vs 212.3, 448 264.3 vs
// 254.7, 511 279.3 vs 284.8 (32-row blocks: 46.5 at 33, 50.5 at 64; q4_K MMQ 21.8 at 33, 30.4 at 64, 178.5 at 511)
#define EXL3_GEMM_BN   128
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
// ID: row r is the x row of its pair, with the suh of the pair's expert
// GLU (ID only): x is the gate and up the up tensor of a standard SwiGLU (same strides) whose output the op reads; the row is
// silu(gate)*up computed as ggml's gated kernel computes it (unary.cu, one f32 rounding of each product) and then times suh,
// the operations of the unfused graph without the f32 store and load in between
template <int S, bool LO, bool ID, bool GLU = false>
static __global__ void __launch_bounds__(256) exl3_rotate_split(
        const float * __restrict__ x, const half * __restrict__ suh, half * __restrict__ xhi, half * __restrict__ xlo,
        float * __restrict__ inv, const int k, const int ne1, const int ne2, const int64_t s1, const int64_t s2,
        const int64_t s3, const exl3_mmid mm, const float * __restrict__ up = nullptr) {
    static_assert(!GLU || ID, "the SwiGLU input is an EXL3_MUL_MAT_ID input");
    __shared__ float s_max[8];
    const int r    = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int segs = k / EXL3_HAD;

    const float * xr;
    const float * ur = nullptr;
    if constexpr (ID) {
        const int p = exl3_mmid_pair(mm, r);
        xr   = exl3_mmid_x(mm, x, p, s1, s2);
        if constexpr (GLU) {
            ur = exl3_mmid_x(mm, up, p, s1, s2);
        }
        suh += (int64_t) exl3_mmid_expert(mm, p)*k;
    } else {
        const int i1 = r % ne1;
        const int i2 = (r / ne1) % ne2;
        const int i3 = r / (ne1*ne2);
        xr = x + i1*s1 + i2*s2 + i3*s3;
    }
    float v[S][4];
    float m = 0.0f;
#pragma unroll
    for (int s = 0; s < S; ++s) {
        const int seg = warp + 8*s;
        if (seg < segs) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if constexpr (GLU) {
                    const float g = ggml_cuda_op_silu_single(xr[seg*EXL3_HAD + i*32 + lane])*ur[seg*EXL3_HAD + i*32 + lane];
                    v[s][i] = g*__half2float(suh[seg*EXL3_HAD + i*32 + lane]);
                } else {
                    v[s][i] = xr[seg*EXL3_HAD + i*32 + lane]*__half2float(suh[seg*EXL3_HAD + i*32 + lane]);
                }
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
    half * lr = LO ? xlo + (int64_t) r*k : nullptr;
#pragma unroll
    for (int s = 0; s < S; ++s) {
        const int seg = warp + 8*s;
        if (seg < segs) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float t = v[s][i]*sc;
                const half  h = __float2half_rn(t);
                hr[seg*EXL3_HAD + i*32 + lane] = h;
                if constexpr (LO) {
                    lr[seg*EXL3_HAD + i*32 + lane] = __float2half_rn(t - __half2float(h));
                }
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

// y = H(xh W_hat)*svh/128 for BM = 128 output columns (one Hadamard segment) and BN = 16*W rows per block (W warps),
// with the structure of the MMB tile GEMM in mmb.cu (register prefetch of the next k step, one LDS stage, two barriers
// per step). Per k step warp w decodes column tiles w, w + W, ... of the step's 2 k-tiles into the LDS weight tile, as
// f16 [column][k]; the [hi; lo] rows of xh go to LDS as they are. Each warp then holds all 128 columns of its 16 rows, so
// the output Hadamard runs in registers (stages 1..8 across lanes, 16..64 across fragments, the order and roundings of
// exl3_fwht128). Both products, W_hat with xh hi and with xh lo, are f16 WMMAs with f32 accumulation.
// ID: block row g is group g of mm (up to BN rows of one expert), with that expert's trellis and svh; row t is stored to
// the dst row of its pair. A warp whose 16 rows are all past the group's last row only decodes its share of the weight
// tile: it skips the WMMAs and the output. P = 1 (ID only): one f16 product, W_hat with xh hi, accumulated across k in
// place in the order of exl3_gemm1_kernel, so a row equals that kernel's row bit for bit
template <int bits, int cb, bool ID, int W, int P = 2>
static __global__ void __launch_bounds__(32*W) exl3_gemm_kernel(
        const int16_t * __restrict__ trellis, const half * __restrict__ xhi, const half * __restrict__ xlo,
        const float * __restrict__ inv, const half * __restrict__ svh, float * __restrict__ dst, const int k,
        const int n, int rows, const exl3_mmid mm) {
#if defined(RDNA3)
    constexpr int BM         = EXL3_GEMM_BM;
    constexpr int BN         = 16*W;
    constexpr int NT         = 32*W;           // threads
    constexpr int CT         = 8/W;            // column tiles a warp decodes per k-tile
    constexpr int BK         = EXL3_GEMM_BK;
    constexpr int LS         = EXL3_GEMM_LS;
    constexpr int R          = exl3_lane_plan<bits>::R;
    constexpr int tile_bytes = exl3_width<bits>::tile_bytes;
    constexpr int KT         = BK/16;          // k-tiles per step
    constexpr int B_ITEMS    = P*BN*BK/8/NT;   // 16-byte chunks of xh hi (and lo) per thread per step
    static_assert(BM == 8*16 && 8 % W == 0, "8/W column tiles and 16 rows per warp");
    static_assert(P == 2 || (P == 1 && ID), "one product without ID is exl3_gemm1_kernel");

    __shared__ __align__(16) uint16_t As[BM*LS];
    __shared__ __align__(16) uint16_t Bs[P*BN*LS];

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int n0   = blockIdx.x*BM;
    int       t0   = blockIdx.y*BN;
    int       rt   = warp; // the warp's 16 rows: t0 + rt*16 ..
    bool      active = true; // they hold a row of the block
    if constexpr (ID) {
        // rows [t0, rows) of the expert's group. The row tiles rotate over the warps from block to block, so that the
        // warps of the partial groups that skip their WMMAs are not always the last ones (the same SIMDs)
        int e;
        int nr;
        exl3_mmid_group(mm, blockIdx.y, e, t0, nr);
        if (nr == 0) {
            return;
        }
        rows     = t0 + nr;
        rt       = (warp + blockIdx.x + blockIdx.y) % W;
        active   = rt*16 < nr;
        trellis += e*mm.tq_stride;
        svh     += (int64_t) e*n;
    }
    const int n_steps = k / BK;
    const uint32_t kt_stride = (uint32_t) (n/16)*tile_bytes;

    const exl3_lane_plan<bits> lp = exl3_make_lane_plan<bits>(lane);
    exl3_g_char * tq = (exl3_g_char *) exl3_uni_ptr((const char *) trellis + (size_t) (n0/16 + warp)*tile_bytes);

    // weight stores: lane L holds rows r0, r0 + 1, r0 + 8, r0 + 9 (k) of columns c and c + 8 of its tiles
    const int c  = lane / 4;
    const int r0 = (lane % 4)*2;
    uint32_t * a_st0[CT];
    uint32_t * a_st1[CT];
#pragma unroll
    for (int q = 0; q < CT; ++q) {
        a_st0[q] = (uint32_t *) (As + ((warp + W*q)*16 + c    )*LS + r0);
        a_st1[q] = (uint32_t *) (As + ((warp + W*q)*16 + c + 8)*LS + r0);
    }

    // activations: item i of a thread is 8 halves of row (tid + NT*i) / (BK/8) of the [hi; lo] tile. Rows past the end
    // re-read the last row (their outputs are not stored): the loads stay unguarded, so that the compiler does not
    // drain them at a divergent join
    const exl3_u4 * xsrc[B_ITEMS];
    exl3_u4       * xdst[B_ITEMS];
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) {
        const int it  = tid + i*NT;
        const int row = min(t0 + (it / (BK/8)) % BN, rows - 1);
        xsrc[i] = (const exl3_u4 *) ((i < B_ITEMS/P ? xhi : xlo) + (int64_t) row*k + (it % (BK/8))*8);
        xdst[i] = (exl3_u4 *) (Bs + (it / (BK/8))*LS + (it % (BK/8))*8);
    }

    uint32_t raw[CT][KT][R];
    exl3_u4  bst[B_ITEMS];
#define EXL3_GEMM_LOAD(ks_)                                                       \
    do {                                                                          \
        _Pragma("unroll")                                                         \
        for (int q = 0; q < CT; ++q) {                                            \
            _Pragma("unroll")                                                     \
            for (int j = 0; j < KT; ++j) {                                        \
                exl3_g_char * tp = tq + (size_t) q*W*tile_bytes + (size_t) ((ks_)*KT + j)*kt_stride; \
                _Pragma("unroll")                                                 \
                for (int i = 0; i < R; ++i) {                                     \
                    raw[q][j][i] = exl3_ld_u32(tp, lp.off[i]);                    \
                }                                                                 \
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
        for (int q = 0; q < CT; ++q) {                                            \
            _Pragma("unroll")                                                     \
            for (int j = 0; j < KT; ++j) {                                        \
                half2 f0[2];                                                      \
                half2 f1[2];                                                      \
                exl3_decode_tile<bits, cb>(raw[q][j], lp, f0, f1);                \
                a_st0[q][j*8]     = __builtin_bit_cast(uint32_t, f0[0]);          \
                a_st0[q][j*8 + 4] = __builtin_bit_cast(uint32_t, f0[1]);          \
                a_st1[q][j*8]     = __builtin_bit_cast(uint32_t, f1[0]);          \
                a_st1[q][j*8 + 4] = __builtin_bit_cast(uint32_t, f1[1]);          \
            }                                                                     \
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
        if (active) {
            // the x fragments of the step, then two column fragments at a time over the whole step: only two partial
            // sums are live, and no WMMA waits on the one just issued
            const int r = lane % 16;
            exl3_v16h bh[KT];
            exl3_v16h bl[KT];
#pragma unroll
            for (int j = 0; j < KT; ++j) {
                bh[j] = exl3_frag(Bs + (     rt*16 + r)*LS + j*16);
                if constexpr (P == 2) {
                    bl[j] = exl3_frag(Bs + (BN + rt*16 + r)*LS + j*16);
                }
            }
            if constexpr (P == 2) {
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
            } else {
#pragma unroll
                for (int j = 0; j < KT; ++j) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const exl3_v16h a = exl3_frag(As + (i*16 + r)*LS + j*16);
                        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bh[j], a, acc[i]);
                    }
                }
            }
        }
        // and the decode of the next step after them, when its loads have landed
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();
        EXL3_GEMM_STORE();
        __syncthreads();
    }
#undef EXL3_GEMM_LOAD
#undef EXL3_GEMM_STORE
    if (!active) {
        return;
    }

    // acc[i][e]: row t0 + rt*16 + 2*e + lane/16, column n0 + i*16 + lane%16
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
        const int t = t0 + rt*16 + 2*e + lane/16;
        if (t < rows) {
            const float s = inv[t];
            float * y = dst + (int64_t) (ID ? exl3_mmid_pair(mm, t) : t)*n + n0 + lane % 16;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                y[i*16] = acc[i][e]*(sv[i]*s);
            }
        }
    }
#else
    GGML_UNUSED_VARS(trellis, xhi, xlo, inv, svh, dst, k, n, rows, mm);
    NO_DEVICE_CODE;
#endif // defined(RDNA3)
}

// One f16 product: y = H(xh_hi W_hat)*svh/128 with BM = 128 output columns and BN = 256 rows per block; warp w holds
// rows w*32 .. w*32 + 31 as two 16-row tiles over all 128 columns, so each A fragment read from LDS feeds two WMMAs and
// each decoded tile serves 256 rows. Op bench, 200 linears of Qwen3.5-4B at 512 rows, same clock: 140 ms, against 163
// for one product in the loop of exl3_gemm_kernel (16 rows per warp) and 277 for its two products. The WMMAs accumulate
// across k in place: the truncation (2.5e-10 at k 2560 .. 3.2e-9 at k 9216 in exl3_gemm_kernel) stays small against
// the f16 rounding of xh.
template <int bits, int cb>
static __global__ void __launch_bounds__(256) exl3_gemm1_kernel(
        const int16_t * __restrict__ trellis, const half * __restrict__ xhi, const float * __restrict__ inv,
        const half * __restrict__ svh, float * __restrict__ dst, const int k, const int n, const int rows) {
#if defined(RDNA3)
    constexpr int BM         = EXL3_GEMM_BM;
    constexpr int BN         = 2*EXL3_GEMM_BN;
    constexpr int BK         = EXL3_GEMM_BK;
    constexpr int LS         = EXL3_GEMM_LS;
    constexpr int R          = exl3_lane_plan<bits>::R;
    constexpr int tile_bytes = exl3_width<bits>::tile_bytes;
    constexpr int KT         = BK/16;
    constexpr int B_ITEMS    = BN*BK/8/256;
    static_assert(BM == 8*16 && BN == 8*32, "one column tile and 32 rows per warp");

    __shared__ __align__(16) uint16_t As[BM*LS];
    __shared__ __align__(16) uint16_t Bs[BN*LS];

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int n0   = blockIdx.x*BM;
    const int t0   = blockIdx.y*BN;
    const int n_steps = k / BK;
    const uint32_t kt_stride = (uint32_t) (n/16)*tile_bytes;

    const exl3_lane_plan<bits> lp = exl3_make_lane_plan<bits>(lane);
    exl3_g_char * tq = (exl3_g_char *) exl3_uni_ptr((const char *) trellis + (size_t) (n0/16 + warp)*tile_bytes);

    const int c  = lane / 4;
    const int r0 = (lane % 4)*2;
    uint32_t * a_st0 = (uint32_t *) (As + (warp*16 + c    )*LS + r0);
    uint32_t * a_st1 = (uint32_t *) (As + (warp*16 + c + 8)*LS + r0);

    const exl3_u4 * xsrc[B_ITEMS];
    exl3_u4       * xdst[B_ITEMS];
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) {
        const int it  = tid + i*256;
        const int row = min(t0 + it / (BK/8), rows - 1);
        xsrc[i] = (const exl3_u4 *) (xhi + (int64_t) row*k + (it % (BK/8))*8);
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

    exl3_v8f acc[2][8];
    const exl3_v8f zero = {};
#pragma unroll
    for (int t = 0; t < 2; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[t][i] = zero;
        }
    }

    EXL3_GEMM_LOAD(0);
    EXL3_GEMM_STORE();
    __syncthreads();
    for (int ks = 0; ks < n_steps; ++ks) {
        // as in exl3_gemm_kernel: the last step reloads itself, the loads stay ahead of the WMMAs. Two LDS stages (one
        // barrier per step, 60 KB, 241 VGPRs) measured slower on every container shape: 139.3 vs 137.4 ms per 200
        // linears at 512 rows and the same clock, k = 9216 912 vs 814 us (two blocks per WGP instead of three)
        EXL3_GEMM_LOAD(min(ks + 1, n_steps - 1));
        __builtin_amdgcn_sched_barrier(0);
        const int r = lane % 16;
#pragma unroll
        for (int j = 0; j < KT; ++j) {
            const exl3_v16h b0 = exl3_frag(Bs + (warp*32      + r)*LS + j*16);
            const exl3_v16h b1 = exl3_frag(Bs + (warp*32 + 16 + r)*LS + j*16);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const exl3_v16h a = exl3_frag(As + (i*16 + r)*LS + j*16);
                acc[0][i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b0, a, acc[0][i]);
                acc[1][i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b1, a, acc[1][i]);
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();
        EXL3_GEMM_STORE();
        __syncthreads();
    }
#undef EXL3_GEMM_LOAD
#undef EXL3_GEMM_STORE

    float sv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sv[i] = __half2float(svh[n0 + i*16 + lane % 16])*(1.0f/EXL3_HAD);
    }
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        exl3_fwht_lanes<1>(acc[t], lane);
        exl3_fwht_lanes<2>(acc[t], lane);
        exl3_fwht_lanes<4>(acc[t], lane);
        exl3_fwht_lanes<8>(acc[t], lane);
#pragma unroll
        for (int hb = 1; hb < 8; hb <<= 1) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (i & hb) {
                    continue;
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float a = acc[t][i][e];
                    const float b = acc[t][i | hb][e];
                    acc[t][i][e]      = a + b;
                    acc[t][i | hb][e] = a - b;
                }
            }
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int row = t0 + warp*32 + t*16 + 2*e + lane/16;
            if (row < rows) {
                const float s = inv[row];
                float * y = dst + (int64_t) row*n + n0 + lane % 16;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    y[i*16] = acc[t][i][e]*(sv[i]*s);
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(trellis, xhi, inv, svh, dst, k, n, rows);
    NO_DEVICE_CODE;
#endif // defined(RDNA3)
}

// hi + lo GEMM with W warps (16*W rows per block); P = 1: the one-product variant (EXL3_MUL_MAT_ID only)
template <int bits, int cb, bool ID>
static void exl3_gemm2_launch(const int W, const int P, const dim3 grid, const int16_t * trellis, const half * xhi,
        const half * xlo, const float * inv, const half * svh, float * dst, const int64_t k, const int64_t n,
        const int64_t rows, const exl3_mmid & mm, cudaStream_t stream) {
    if constexpr (ID) {
        if (P == 1) {
            GGML_ASSERT(W == 8);
            exl3_gemm_kernel<bits, cb, true, 8, 1><<<grid, 256, 0, stream>>>(trellis, xhi, xlo, inv, svh, dst, (int) k, (int) n, (int) rows, mm);
            return;
        }
    }
    GGML_ASSERT(P == 2);
    switch (W) {
        case 2:
            // not for K = 7: four column tiles of 8 dwords per lane spill (260 bytes of scratch)
            if constexpr (bits != 7) {
                exl3_gemm_kernel<bits, cb, ID, 2><<<grid, 64, 0, stream>>>(trellis, xhi, xlo, inv, svh, dst, (int) k, (int) n, (int) rows, mm);
            } else {
                GGML_ABORT("fatal error");
            }
            break;
        case 4: exl3_gemm_kernel<bits, cb, ID, 4><<<grid, 128, 0, stream>>>(trellis, xhi, xlo, inv, svh, dst, (int) k, (int) n, (int) rows, mm); break;
        case 8: exl3_gemm_kernel<bits, cb, ID, 8><<<grid, 256, 0, stream>>>(trellis, xhi, xlo, inv, svh, dst, (int) k, (int) n, (int) rows, mm); break;
        default: GGML_ABORT("fatal error");
    }
}

// mm: the groups of one table of an EXL3_MUL_MAT_ID, n_groups of up to 16*W rows, P products; nullptr for EXL3_MUL_MAT.
// W: warps of the hi + lo GEMM (16*W rows per block)
template <int bits, int cb>
static void exl3_gemm_launch(const int16_t * trellis, const half * xhi, const half * xlo, const float * inv,
        const half * svh, float * dst, const int64_t k, const int64_t n, const int64_t rows, const exl3_mmid * mm,
        const int64_t n_groups, const int W, const int P, cudaStream_t stream) {
    if (mm != nullptr) {
        GGML_ASSERT(xlo != nullptr);
        const dim3 grid((unsigned) (n/EXL3_GEMM_BM), (unsigned) n_groups);
        exl3_gemm2_launch<bits, cb, true>(W, P, grid, trellis, xhi, xlo, inv, svh, dst, k, n, rows, *mm, stream);
    } else if (xlo == nullptr) {
        const dim3 grid((unsigned) (n/EXL3_GEMM_BM), (unsigned) ((rows + 2*EXL3_GEMM_BN - 1)/(2*EXL3_GEMM_BN)));
        exl3_gemm1_kernel<bits, cb><<<grid, 256, 0, stream>>>(trellis, xhi, inv, svh, dst, (int) k, (int) n, (int) rows);
    } else {
        const dim3 grid((unsigned) (n/EXL3_GEMM_BM), (unsigned) ((rows + 16*W - 1)/(16*W)));
        exl3_gemm2_launch<bits, cb, false>(W, 2, grid, trellis, xhi, xlo, inv, svh, dst, k, n, rows, exl3_mmid {}, stream);
    }
}

template <int bits>
static void exl3_gemm_launch_cb(const int cb, const int16_t * trellis, const half * xhi, const half * xlo,
        const float * inv, const half * svh, float * dst, const int64_t k, const int64_t n, const int64_t rows,
        const exl3_mmid * mm, const int64_t n_groups, const int W, const int P, cudaStream_t stream) {
    switch (cb) {
        case GGML_EXL3_CODEBOOK_3INST:
            exl3_gemm_launch<bits, GGML_EXL3_CODEBOOK_3INST>(trellis, xhi, xlo, inv, svh, dst, k, n, rows, mm, n_groups, W, P, stream);
            break;
        case GGML_EXL3_CODEBOOK_MCG:
            exl3_gemm_launch<bits, GGML_EXL3_CODEBOOK_MCG>(trellis, xhi, xlo, inv, svh, dst, k, n, rows, mm, n_groups, W, P, stream);
            break;
        case GGML_EXL3_CODEBOOK_MUL1:
            exl3_gemm_launch<bits, GGML_EXL3_CODEBOOK_MUL1>(trellis, xhi, xlo, inv, svh, dst, k, n, rows, mm, n_groups, W, P, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

template <bool LO, bool ID, bool GLU = false>
static void exl3_rotate_split_launch_lo(const ggml_tensor * x, const half * suh, half * xhi, half * xlo, float * inv,
        const int64_t k, const int64_t rows, const exl3_mmid & mm, cudaStream_t stream, const float * gate = nullptr,
        const float * up = nullptr) {
    const int64_t segs_per_warp = (k/EXL3_HAD + 7)/8;
    const float * xd = GLU ? gate : (const float *) x->data;
    const int     ne1 = (int) x->ne[1];
    const int     ne2 = (int) x->ne[2];
    const int64_t s1  = x->nb[1]/sizeof(float);
    const int64_t s2  = x->nb[2]/sizeof(float);
    const int64_t s3  = x->nb[3]/sizeof(float);
    const dim3 grid((unsigned) rows);
    if (segs_per_warp <= 4) {
        exl3_rotate_split<4, LO, ID, GLU><<<grid, 256, 0, stream>>>(xd, suh, xhi, xlo, inv, (int) k, ne1, ne2, s1, s2, s3, mm, up);
    } else if (segs_per_warp <= 8) {
        exl3_rotate_split<8, LO, ID, GLU><<<grid, 256, 0, stream>>>(xd, suh, xhi, xlo, inv, (int) k, ne1, ne2, s1, s2, s3, mm, up);
    } else if (segs_per_warp <= 16) {
        exl3_rotate_split<16, LO, ID, GLU><<<grid, 256, 0, stream>>>(xd, suh, xhi, xlo, inv, (int) k, ne1, ne2, s1, s2, s3, mm, up);
    } else {
        GGML_ASSERT(segs_per_warp <= 32);
        exl3_rotate_split<32, LO, ID, GLU><<<grid, 256, 0, stream>>>(xd, suh, xhi, xlo, inv, (int) k, ne1, ne2, s1, s2, s3, mm, up);
    }
}

// mm: the rows of an EXL3_MUL_MAT_ID (two products only); nullptr for EXL3_MUL_MAT
static void exl3_rotate_split_launch(const ggml_tensor * x, const half * suh, half * xhi, half * xlo, float * inv,
        const int64_t k, const int64_t rows, const exl3_mmid * mm, cudaStream_t stream) {
    if (mm != nullptr) {
        GGML_ASSERT(xlo != nullptr);
        exl3_rotate_split_launch_lo<true, true>(x, suh, xhi, xlo, inv, k, rows, *mm, stream);
    } else if (xlo != nullptr) {
        exl3_rotate_split_launch_lo<true, false>(x, suh, xhi, xlo, inv, k, rows, exl3_mmid {}, stream);
    } else {
        exl3_rotate_split_launch_lo<false, false>(x, suh, xhi, xlo, inv, k, rows, exl3_mmid {}, stream);
    }
}

// the rate of a trellis: K, or 16 + floor(K) for the half rates K + 0.5 (mul1 only)
static int exl3_bits(const int64_t tile_words) {
    return tile_words % 16 != 0 ? 16 + (int) (tile_words - 8) / 16 : (int) tile_words / 16;
}

static void exl3_gemm_bits(const int bits, const int cb, const int16_t * tq, const half * xhi, const half * xlo,
        const float * inv, const half * sv, float * y, const int64_t k, const int64_t n, const int64_t rows,
        const exl3_mmid * mm, const int64_t n_groups, const int W, const int P, cudaStream_t stream) {
    switch (bits) {
        case 1: exl3_gemm_launch_cb<1>(cb, tq, xhi, xlo, inv, sv, y, k, n, rows, mm, n_groups, W, P, stream); break;
        case 2: exl3_gemm_launch_cb<2>(cb, tq, xhi, xlo, inv, sv, y, k, n, rows, mm, n_groups, W, P, stream); break;
        case 3: exl3_gemm_launch_cb<3>(cb, tq, xhi, xlo, inv, sv, y, k, n, rows, mm, n_groups, W, P, stream); break;
        case 4: exl3_gemm_launch_cb<4>(cb, tq, xhi, xlo, inv, sv, y, k, n, rows, mm, n_groups, W, P, stream); break;
        case 5: exl3_gemm_launch_cb<5>(cb, tq, xhi, xlo, inv, sv, y, k, n, rows, mm, n_groups, W, P, stream); break;
        case 6: exl3_gemm_launch_cb<6>(cb, tq, xhi, xlo, inv, sv, y, k, n, rows, mm, n_groups, W, P, stream); break;
        case 7: exl3_gemm_launch_cb<7>(cb, tq, xhi, xlo, inv, sv, y, k, n, rows, mm, n_groups, W, P, stream); break;
        case 8: exl3_gemm_launch_cb<8>(cb, tq, xhi, xlo, inv, sv, y, k, n, rows, mm, n_groups, W, P, stream); break;
        case 17: exl3_gemm_launch<17, GGML_EXL3_CODEBOOK_MUL1>(tq, xhi, xlo, inv, sv, y, k, n, rows, mm, n_groups, W, P, stream); break;
        case 18: exl3_gemm_launch<18, GGML_EXL3_CODEBOOK_MUL1>(tq, xhi, xlo, inv, sv, y, k, n, rows, mm, n_groups, W, P, stream); break;
        case 19: exl3_gemm_launch<19, GGML_EXL3_CODEBOOK_MUL1>(tq, xhi, xlo, inv, sv, y, k, n, rows, mm, n_groups, W, P, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

static void exl3_dot_bits(const int bits, const int cb, const int M, const float * xh, const int16_t * tq,
        const half * sv, float * y, int * counters, const int64_t k, const int64_t n, const int64_t rows,
        const int64_t n_groups, const exl3_mmid * mm, cudaStream_t stream) {
    switch (bits) {
        case 1: exl3_launch_cb<1>(cb, M, xh, tq, sv, y, counters, k, n, rows, n_groups, mm, stream); break;
        case 2: exl3_launch_cb<2>(cb, M, xh, tq, sv, y, counters, k, n, rows, n_groups, mm, stream); break;
        case 3: exl3_launch_cb<3>(cb, M, xh, tq, sv, y, counters, k, n, rows, n_groups, mm, stream); break;
        case 4: exl3_launch_cb<4>(cb, M, xh, tq, sv, y, counters, k, n, rows, n_groups, mm, stream); break;
        case 5: exl3_launch_cb<5>(cb, M, xh, tq, sv, y, counters, k, n, rows, n_groups, mm, stream); break;
        case 6: exl3_launch_cb<6>(cb, M, xh, tq, sv, y, counters, k, n, rows, n_groups, mm, stream); break;
        case 7: exl3_launch_cb<7>(cb, M, xh, tq, sv, y, counters, k, n, rows, n_groups, mm, stream); break;
        case 8: exl3_launch_cb<8>(cb, M, xh, tq, sv, y, counters, k, n, rows, n_groups, mm, stream); break;
        // half rates K + 0.5 exist for mul1 only
        case 17: exl3_launch<17, GGML_EXL3_CODEBOOK_MUL1>(M, xh, tq, sv, y, counters, k, n, rows, n_groups, mm, stream); break;
        case 18: exl3_launch<18, GGML_EXL3_CODEBOOK_MUL1>(M, xh, tq, sv, y, counters, k, n, rows, n_groups, mm, stream); break;
        case 19: exl3_launch<19, GGML_EXL3_CODEBOOK_MUL1>(M, xh, tq, sv, y, counters, k, n, rows, n_groups, mm, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

// Products per k step on the GEMM path. W_hat is exact in f16, so the one rounding is that of xh = H(x*suh): one f16
// product (exl3_gemm1_kernel, NMSE 3.5e-8 .. 4.7e-8 against a double sum on the container shapes at 512 rows) or hi + lo
// (exl3_gemm_kernel, f32 grade, <= 1.7e-13, twice the WMMA work). The default follows the GEMM this backend context runs
// for an F32 weight of the same shape on the same x: one product where mmb takes it (mmb_f32split_kernel, one f16 product
// with both operands rounded: 7.4e-8 .. 8.5e-8 on those shapes at 512 and 513 rows), two where it is f32 (MMVF, MMF or
// rocBLAS: 1e-14 .. 3e-12 at 1 .. 511 rows, and for strided x). GGML_EXL3_GEMM_PRODUCTS=1 or 2 overrides the choice.
static int exl3_gemm_products(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * dst,
        const int64_t k, const int64_t n) {
    static const int forced = getenv("GGML_EXL3_GEMM_PRODUCTS") ? atoi(getenv("GGML_EXL3_GEMM_PRODUCTS")) : 0;
    if (forced == 1 || forced == 2) {
        return forced;
    }
    ggml_tensor w = {};
    w.type  = GGML_TYPE_F32;
    w.ne[0] = k;
    w.ne[1] = n;
    w.ne[2] = 1;
    w.ne[3] = 1;
    w.nb[0] = sizeof(float);
    w.nb[1] = k*sizeof(float);
    w.nb[2] = n*w.nb[1];
    w.nb[3] = w.nb[2];
    return ggml_cuda_mmb_supported_mm(ctx, &w, x, dst) ? 1 : 2;
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
    const int       bits = exl3_bits(tile_words);

    if (rows > EXL3_GEMV_ROWS && k <= EXL3_SPLIT_K && GGML_CUDA_CC_IS_RDNA3(ggml_cuda_info().devices[ctx.device].cc)) {
        GGML_ASSERT((rows + EXL3_GEMM_BN/2 - 1)/(EXL3_GEMM_BN/2) <= 65535);
        // one f16 product or hi + lo, following the F32-weight GEMM of this context (exl3_gemm_products)
        const bool two = exl3_gemm_products(ctx, x, dst, k, n) == 2;
        ggml_cuda_pool_alloc<half>  xhi(ctx.pool(), rows*k);
        ggml_cuda_pool_alloc<half>  xlo(ctx.pool());
        ggml_cuda_pool_alloc<float> inv(ctx.pool(), rows);
        if (two) {
            xlo.alloc(rows*k);
        }
        half * xl = two ? xlo.get() : nullptr;
        exl3_rotate_split_launch(x, (const half *) suh->data, xhi.get(), xl, inv.get(), k, rows, nullptr, stream);
        // hi + lo: blocks of 64 rows (W = 4) when the last block of EXL3_GEMM_BN rows would be at most half full
        const int64_t tail = rows % EXL3_GEMM_BN;
        const int     W    = tail > 0 && tail <= EXL3_GEMM_BN/2 ? EXL3_GEMM_BN/32 : EXL3_GEMM_BN/16;
        exl3_gemm_bits(bits, cb, tq, xhi.get(), xl, inv.get(), sv, y, k, n, rows, nullptr, 0, W, two ? 2 : 1, stream);
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
    exl3_rotate_in<false><<<(unsigned) ((threads_in + 255) / 256), 256, 0, stream>>>(
        (const float *) x->data, (const half *) suh->data, xh.get(), counters.get(), (int) (n_groups*n_segs), (int) k,
        (int) rows, (int) x->ne[1], (int) x->ne[2], x->nb[1] / sizeof(float), x->nb[2] / sizeof(float),
        x->nb[3] / sizeof(float), exl3_mmid {});
    exl3_dot_bits(bits, cb, M, xh.get(), tq, sv, y, counters.get(), k, n, rows, n_groups, nullptr, stream);
}

// ---- EXL3_MUL_MAT_ID ----

// The path by the average rows per expert, n_pairs/n_exp: the dot kernel in groups of up to 8 rows of one expert (a pass
// over the expert's weights per group) below EXL3_MMID_GEMM_ROWS, else the GEMM (RDNA3) in groups of up to 32 rows of one
// expert (W = 2 warps) up to EXL3_MMID_W2_ROWS, 64 rows (W = 4) above; the GEMM rows of an expert with at least
// exl3_mmid_one_rows rows run one product in groups of EXL3_MMID_ONE_W warps. A warp whose 16 rows are past its group's
// last row skips its WMMAs (exl3_gemm_kernel), so a group costs the WMMA work of its rows rounded up to 16: routing in a
// model is far from uniform (LFM2.5-8B-A1B at ubatch 512, llama-bench's prompt: median 16 rows per expert, 18% of the
// experts empty, 15% with >= 128 rows holding 68% of the rows), and whole groups did 1.53x the WMMA work of the rows,
// 16-row tiles 1.11x (.claude/exl3-s8/results/routing). LFM2.5 stacks (2048 -> 1792 x broadcast / 1792 -> 2048 per slot,
// 32 experts, top-4, K = 2 mul1), gfx1151, us per op:
//   dot vs GEMM 32-row groups, uniform routing (.claude/exl3-s7/results/mmid): 88 tokens (11 rows per expert) 638 / 717
//   vs 679 / 722, 96 tokens (12) 682 / 690 vs 621 / 634
//   32- vs 64-row groups with the skipped tiles, uniform (.claude/exl3-s8/results/mmid/tok-*): 96 tokens 503 / 496 vs
//   532 / 532, 128: 557 / 606 vs 656 / 658, 192: 652 / 774 vs 782 / 819, 256: 840 / 991 vs 932 / 931, 384: 1294 / 1366
//   vs 1149 / 1132
//   512 tokens, 64-row groups with the skipped tiles vs whole groups (ABBA, .claude/exl3-s8/results/mmid/fin-*): uniform
//   1426 / 1445 vs 1523 / 1544, llama-bench's routing 1351 / 1371 vs 1550 / 1573, prose 1370 / 1390 vs 1514 / 1576,
//   code 1373 / 1399 vs 1579 / 1584. Rejected (llama-bench's routing): 128-row groups 1345 / 1367 vs 1315 / 1344
//   (s8b-*), a separate launch of 32-row blocks for the last partial groups of up to 32 rows 1369 / 1382 (llama-bench
//   pp2048 4114 / 4059 tok/s vs 4229, results/screen-w)
#define EXL3_MMID_GEMM_ROWS   12
#define EXL3_MMID_W2_ROWS     32
// 128-row groups for the one-product experts; 64-row groups measured the same (llama-bench pp2048 at ubatch 2048, where
// 66% of the rows take one product: 5111 vs 5094 tok/s, .claude/exl3-s8/results/screen-onew)
#define EXL3_MMID_ONE_W       8
#define EXL3_MMID_EXPERTS_MAX 1024 // exl3_mmid_route counts the rows of each expert in LDS

// group tables of an EXL3_MUL_MAT_ID: table 0 groups of up to s[0] rows (two products on the GEMM path), table 1 groups of
// up to s[1] rows of the experts with at least one_rows rows (one product). The dot path uses table 0 only
struct exl3_mmid_tables {
    int * grp[2];
    int   n_grp[2];
    int   s[2];
};

// the rows of an EXL3_MUL_MAT_ID in expert order (order[r] = pair of row r) and its group tables (exl3_mmid). One block;
// the atomics fill the rows of an expert in any order, which no result depends on (each row is computed on its own)
static __global__ void __launch_bounds__(1024) exl3_mmid_route(
        const char * __restrict__ ids, const int64_t nb0, const int64_t nb1, const int n_used, const int n_pairs,
        const int n_exp, const int one_rows, int * __restrict__ order, const exl3_mmid_tables tab) {
    extern __shared__ int s_route[]; // [n_exp] rows per expert, then [n_exp] fill positions
    int * s_cnt = s_route;
    int * s_pos = s_route + n_exp;
    for (int e = threadIdx.x; e < n_exp; e += blockDim.x) {
        s_cnt[e] = 0;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < n_pairs; p += blockDim.x) {
        atomicAdd(&s_cnt[*(const int *) (ids + (p % n_used)*nb0 + (p / n_used)*nb1)], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int row  = 0;
        int g[2] = { 0, 0 };
        for (int e = 0; e < n_exp; ++e) {
            const int c = s_cnt[e];
            const int t = c >= one_rows ? 1 : 0;
            s_pos[e] = row;
            for (int j = 0; j < c; j += tab.s[t]) {
                int * q = tab.grp[t] + 3*g[t]++;
                q[0] = e;
                q[1] = row + j;
                q[2] = min(tab.s[t], c - j);
            }
            row += c;
        }
        for (int t = 0; t < 2; ++t) {
            for (; g[t] < tab.n_grp[t]; ++g[t]) {
                int * q = tab.grp[t] + 3*g[t];
                q[0] = 0;
                q[1] = 0;
                q[2] = 0;
            }
        }
    }
    __syncthreads();
    for (int p = threadIdx.x; p < n_pairs; p += blockDim.x) {
        const int e = *(const int *) (ids + (p % n_used)*nb0 + (p / n_used)*nb1);
        order[atomicAdd(&s_pos[e], 1)] = p;
    }
}

// Products of the GEMM rows, per expert. The F32 control, ggml_mul_mat_id on an F32 stack, runs on this backend the rows
// of each expert as one ggml_mul_mat of a view of the stack on a contiguous copy of the rows: one f16 product
// (mmb_f32split_kernel, both operands rounded) where ggml_cuda_mmb_supported_mm takes that shape (from 512 rows for a
// view source), f32 (MMVF or rocBLAS) below. An expert's rows run one product iff the control runs one for them. Returns
// the fewest rows of an expert that do (INT_MAX: none can, an expert holds at most one row per token).
// GGML_EXL3_GEMM_PRODUCTS=1 or 2 overrides the choice as for EXL3_MUL_MAT.
static int exl3_mmid_one_rows(ggml_backend_cuda_context & ctx, const int64_t k, const int64_t n, const int64_t n_tok) {
    static const int forced = getenv("GGML_EXL3_GEMM_PRODUCTS") ? atoi(getenv("GGML_EXL3_GEMM_PRODUCTS")) : 0;
    if (forced == 1 || forced == 2) {
        return forced == 1 ? 1 : INT_MAX;
    }
    ggml_tensor w = {};
    w.type  = GGML_TYPE_F32;
    w.op    = GGML_OP_VIEW;
    w.ne[0] = k;
    w.ne[1] = n;
    w.ne[2] = 1;
    w.ne[3] = 1;
    w.nb[0] = sizeof(float);
    w.nb[1] = k*sizeof(float);
    w.nb[2] = n*w.nb[1];
    w.nb[3] = w.nb[2];
    ggml_tensor x = w;
    ggml_tensor d = w;
    x.op = GGML_OP_NONE;
    d.op = GGML_OP_NONE;
    auto one = [&](const int64_t t) {
        x.ne[1] = t;
        x.nb[2] = x.nb[3] = t*x.nb[1];
        d.ne[0] = n;
        d.ne[1] = t;
        d.nb[1] = n*sizeof(float);
        d.nb[2] = d.nb[3] = t*d.nb[1];
        return ggml_cuda_mmb_supported_mm(ctx, &w, &x, &d);
    };
    if (!one(n_tok)) {
        return INT_MAX;
    }
    int64_t lo = 0;
    int64_t hi = n_tok;
    while (hi - lo > 1) {
        const int64_t mid = (lo + hi)/2;
        if (one(mid)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return (int) hi;
}

bool ggml_cuda_exl3_mul_mat_id_supports_op(const int device, const ggml_tensor * op) {
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_RDNA4(cc)) {
        return false;
    }
    const ggml_tensor * trellis = op->src[0];
    const ggml_tensor * x       = op->src[1];
    const ggml_tensor * ids     = op->src[2];
    const int           cb      = ggml_get_op_params_i32(op, 0);
    return op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && x->type == GGML_TYPE_F32 &&
           x->nb[0] == sizeof(float) && x->nb[1] % sizeof(float) == 0 && x->nb[2] % sizeof(float) == 0 &&
           ids->type == GGML_TYPE_I32 && trellis->ne[3] <= EXL3_MMID_EXPERTS_MAX &&
           ggml_exl3_rate_supported(trellis->ne[0], (ggml_exl3_codebook) cb);
}

// the GEMM path (exl3_rotate_split + exl3_gemm_kernel) of EXL3_MUL_MAT_ID
static bool exl3_mmid_gemm(const int device, const ggml_tensor * dst) {
    const ggml_tensor * trellis = dst->src[0];
    const int64_t n_pairs = dst->src[2]->ne[0]*dst->src[2]->ne[1];
    return n_pairs >= EXL3_MMID_GEMM_ROWS*trellis->ne[3] && trellis->ne[2]*16 <= EXL3_SPLIT_K &&
           GGML_CUDA_CC_IS_RDNA3(ggml_cuda_info().devices[device].cc);
}

// each row (pair) is computed as EXL3_MUL_MAT computes a row of its expert: the dot kernel with groups of rows of one
// expert, or the GEMM, hi + lo (exl3_gemm_kernel, two products) or one product (its P = 1 variant, rows equal to those
// of exl3_gemm1_kernel) per expert as the F32 control does (exl3_mmid_one_rows). No host synchronization: the routing
// is computed on the device.
// glu: the standard F32 SwiGLU node that dst reads as x, computed by the input rotation of the GEMM path (exl3_mmid_gemm)
static void exl3_op_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * glu) {
    const ggml_tensor * trellis = dst->src[0];
    const ggml_tensor * x       = dst->src[1];
    const ggml_tensor * ids     = dst->src[2];
    const ggml_tensor * suh     = dst->src[3];
    const ggml_tensor * svh     = dst->src[4];

    GGML_ASSERT(ggml_cuda_exl3_mul_mat_id_supports_op(ctx.device, dst));

    const int     cb      = ggml_get_op_params_i32(dst, 0);
    const int64_t n       = trellis->ne[1]*16;
    const int64_t k       = trellis->ne[2]*16;
    const int64_t n_exp   = trellis->ne[3];
    const int64_t n_used  = ids->ne[0];
    const int64_t n_tok   = ids->ne[1];
    const int64_t n_pairs = n_used*n_tok;
    if (n_pairs == 0) {
        return;
    }

    cudaStream_t stream = ctx.stream();

    const int16_t * tq   = (const int16_t *) trellis->data;
    const half    * su   = (const half *) suh->data;
    const half    * sv   = (const half *) svh->data;
    float         * y    = (float *) dst->data;
    const int       bits = exl3_bits(trellis->ne[0]);

    exl3_mmid mm = {};
    mm.ids       = (const char *) ids->data;
    mm.ids_nb0   = ids->nb[0];
    mm.ids_nb1   = ids->nb[1];
    mm.n_used    = (int) n_used;
    mm.ne11      = (int) x->ne[1];
    mm.tq_stride = trellis->ne[0]*trellis->ne[1]*trellis->ne[2];

    const bool gemm = exl3_mmid_gemm(ctx.device, dst);
    GGML_ASSERT(glu == nullptr || gemm);
    // an expert has at most one row per token
    const int M = n_tok == 1 ? 1 : n_tok == 2 ? 2 : n_tok <= 4 ? 4 : 8;
    // warps of the two-product groups, then of the one-product ones
    const int W[2] = { n_pairs <= EXL3_MMID_W2_ROWS*n_exp && bits != 7 ? 2 : 4, EXL3_MMID_ONE_W };

    // one token: a group per pair; more: the routing and its group tables
    exl3_mmid_tables tab = {};
    int64_t n_groups = n_pairs;
    ggml_cuda_pool_alloc<int> order(ctx.pool());
    ggml_cuda_pool_alloc<int> grp(ctx.pool());
    if (n_tok > 1) {
        const int one_rows = gemm ? exl3_mmid_one_rows(ctx, k, n, n_tok) : INT_MAX;
        int64_t n_grp[2];
        tab.s[0] = gemm ? 16*W[0] : M;
        tab.s[1] = 16*W[1];
        n_grp[0] = n_pairs/tab.s[0] + std::min(n_exp, n_pairs);
        n_grp[1] = one_rows <= n_tok ? n_pairs/tab.s[1] + n_pairs/one_rows : 0;
        order.alloc(n_pairs);
        grp.alloc(3*(n_grp[0] + n_grp[1]));
        for (int t = 0; t < 2; ++t) {
            GGML_ASSERT(n_grp[t] <= 65535);
            tab.grp[t]   = n_grp[t] > 0 ? grp.get() + 3*(t == 0 ? 0 : n_grp[0]) : nullptr;
            tab.n_grp[t] = (int) n_grp[t];
        }
        exl3_mmid_route<<<1, 1024, 2*n_exp*sizeof(int), stream>>>(mm.ids, mm.ids_nb0, mm.ids_nb1, (int) n_used,
            (int) n_pairs, (int) n_exp, one_rows, order.get(), tab);
        mm.order = order.get();
        mm.grp   = tab.grp[0];
        n_groups = n_grp[0];
    }
    GGML_ASSERT(n_groups <= 65535);

    if (gemm) {
        ggml_cuda_pool_alloc<half>  xhi(ctx.pool(), n_pairs*k);
        ggml_cuda_pool_alloc<half>  xlo(ctx.pool(), n_pairs*k);
        ggml_cuda_pool_alloc<float> inv(ctx.pool(), n_pairs);
        if (glu != nullptr) {
            exl3_rotate_split_launch_lo<true, true, true>(x, su, xhi.get(), xlo.get(), inv.get(), k, n_pairs, mm, stream,
                (const float *) glu->src[0]->data, (const float *) glu->src[1]->data);
        } else {
            exl3_rotate_split_launch(x, su, xhi.get(), xlo.get(), inv.get(), k, n_pairs, &mm, stream);
        }
        for (int t = 0; t < 2; ++t) {
            if (tab.n_grp[t] > 0) {
                mm.grp = tab.grp[t];
                exl3_gemm_bits(bits, cb, tq, xhi.get(), xlo.get(), inv.get(), sv, y, k, n, n_pairs, &mm, tab.n_grp[t], W[t],
                               t == 0 ? 2 : 1, stream);
            }
        }
        return;
    }

    const int64_t n_segs = n / EXL3_HAD;
    ggml_cuda_pool_alloc<float> xh(ctx.pool(), n_pairs*k);
    ggml_cuda_pool_alloc<int>   counters(ctx.pool(), n_groups*n_segs);

    const int64_t threads_in = n_pairs*(k/EXL3_HAD)*32;
    exl3_rotate_in<true><<<(unsigned) ((threads_in + 255) / 256), 256, 0, stream>>>(
        (const float *) x->data, su, xh.get(), counters.get(), (int) (n_groups*n_segs), (int) k, (int) n_pairs, 1, 1,
        x->nb[1] / sizeof(float), x->nb[2] / sizeof(float), 0, mm);
    exl3_dot_bits(bits, cb, M, xh.get(), tq, sv, y, counters.get(), k, n, n_pairs, n_groups, &mm, stream);
}

void ggml_cuda_op_exl3_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    exl3_op_mul_mat_id(ctx, dst, nullptr);
}

bool ggml_cuda_exl3_mul_mat_id_swiglu_fusable(const int device, const ggml_tensor * glu, const ggml_tensor * dst) {
    const ggml_tensor * gate = glu->src[0];
    const ggml_tensor * up   = glu->src[1];
    return glu->op == GGML_OP_GLU && ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU && up != nullptr && dst->src[1] == glu &&
           glu->type == GGML_TYPE_F32 && gate->type == GGML_TYPE_F32 && up->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(glu) && ggml_is_contiguous(gate) && ggml_is_contiguous(up) &&
           ggml_are_same_shape(glu, gate) && ggml_are_same_shape(glu, up) && exl3_mmid_gemm(device, dst);
}

void ggml_cuda_op_exl3_mul_mat_id_swiglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * glu) {
    GGML_ASSERT(ggml_cuda_exl3_mul_mat_id_swiglu_fusable(ctx.device, glu, dst));
    exl3_op_mul_mat_id(ctx, dst, glu);
}

// ---- EXL3_GET_ROWS ----

// One warp per (row, ring of 256), lane l holding positions l + 32*q, q = 0..7: the symbols of the ring's K-bit stream
// (LSB first) go to LDS, the state of position i stacks the symbols of positions i, i - 1, ... from bit 0 up, and its
// mul1 value times the ring's fp16 scale is exact in f32; then the orthonormal H256 in the butterfly order of
// ggml_exl3_decode_embd_row (stages 1 .. 16 across lanes, 32 .. 128 across q), the 1/16 and the signs, each one f32
// rounding of the same operands as that decoder, so the rows are its rows bit for bit
static __global__ void __launch_bounds__(256) exl3_get_rows_kernel(
        const int16_t * __restrict__ table, const int64_t row_words, const char * __restrict__ ids, const int64_t ids_nb0,
        const half * __restrict__ signs, float * __restrict__ dst, const int64_t dst_s1, const int n_ids, const int n_out,
        const int K) {
    __shared__ uint8_t s_sym[8][256];
    const int  lane   = threadIdx.x % 32;
    const int  warp   = threadIdx.x / 32;
    const int  rings  = (n_out + 255)/256;
    const int  w      = blockIdx.x*8 + warp;
    const bool active = w < n_ids*rings;
    const int  i      = active ? w / rings : 0;
    const int  g      = active ? w % rings : 0;
    const int  words  = 16*K;

    const int16_t * ring = table + (int64_t) *(const int *) (ids + i*ids_nb0)*row_words + (int64_t) g*(1 + words);
    uint8_t * sym = s_sym[warp];
    if (active) {
        const uint16_t * stream = (const uint16_t *) (ring + 1);
        const uint32_t   mask   = (1u << K) - 1;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int      bit    = (lane*8 + j)*K;
            const uint32_t window = (uint32_t) stream[bit >> 4] | ((uint32_t) stream[((bit >> 4) + 1) % words] << 16);
            sym[lane*8 + j] = (uint8_t) ((window >> (bit & 15)) & mask);
        }
    }
    __syncthreads();
    if (!active) {
        return;
    }

    const float scale = __half2float(__ushort_as_half((unsigned short) ring[0]));
    const int   nj    = (15 + K)/K;
    float v[8];
#pragma unroll
    for (int q = 0; q < 8; q += 2) {
        uint32_t st[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int pos = lane + 32*(q + h);
            uint32_t s = 0;
            for (int j = 0; j < nj; ++j) {
                s |= (uint32_t) sym[(pos - j) & 255] << (j*K);
            }
            st[h] = s & 0xffffu;
        }
        const half2 c = exl3_dq_pair<GGML_EXL3_CODEBOOK_MUL1, false, false>(st[0], st[1]);
        v[q]     = __low2float(c)*scale;
        v[q + 1] = __high2float(c)*scale;
    }
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const float o = __shfl_xor(v[q], h, 32);
            v[q] = (lane & h) ? o - v[q] : v[q] + o;
        }
    }
#pragma unroll
    for (int hq = 1; hq < 8; hq <<= 1) {
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            if (q & hq) {
                continue;
            }
            const float a = v[q];
            const float b = v[q | hq];
            v[q]      = a + b;
            v[q | hq] = a - b;
        }
    }

    float      * out = dst + (int64_t) i*dst_s1 + g*256;
    const half * sg  = signs + g*256;
    const int    n_g = min(256, n_out - g*256);
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        const int pos = lane + 32*q;
        if (pos < n_g) {
            out[pos] = (v[q]*0.0625f)*__half2float(sg[pos]);
        }
    }
}

bool ggml_cuda_exl3_get_rows_supports_op(const int device, const ggml_tensor * op) {
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_RDNA4(cc)) {
        return false;
    }
    const ggml_tensor * table = op->src[0];
    const ggml_tensor * ids   = op->src[1];
    const ggml_tensor * signs = op->src[2];
    return op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && ids->type == GGML_TYPE_I32 &&
           ggml_exl3_embd_k(table->ne[0], signs->ne[1]) > 0;
}

void ggml_cuda_op_exl3_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * table = dst->src[0];
    const ggml_tensor * ids   = dst->src[1];
    const ggml_tensor * signs = dst->src[2];

    GGML_ASSERT(ggml_cuda_exl3_get_rows_supports_op(ctx.device, dst));

    const int     n_out = ggml_get_op_params_i32(dst, 0);
    const int     K     = ggml_exl3_embd_k(table->ne[0], signs->ne[1]);
    const int64_t n_ids = ids->ne[0];
    if (n_ids == 0) {
        return;
    }
    const int64_t warps = n_ids*((n_out + 255)/256);
    exl3_get_rows_kernel<<<(unsigned) ((warps + 7)/8), 256, 0, ctx.stream()>>>(
        (const int16_t *) table->data, table->nb[1]/sizeof(int16_t), (const char *) ids->data, ids->nb[0],
        (const half *) signs->data, (float *) dst->data, dst->nb[1]/sizeof(float), (int) n_ids, n_out, K);
}

#endif // defined(GGML_USE_EXL3)
