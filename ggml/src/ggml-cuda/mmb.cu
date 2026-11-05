#include <functional>
#include "mmb.cuh"
#include "unary.cuh"
#include <unordered_map>
#include <map>
#include <atomic>
#include <mutex>
#include <utility>
#include "mmid.cuh"
#include <cstdlib>
#include <vector>
#include <unordered_set>
#include <mutex>
#include <set>

namespace {

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f  __attribute__((ext_vector_type(8)));
constexpr int MMB_BK = 64, MMB_NT = 256, MMB_LDS_STRIDE = MMB_BK + 8;

__device__ __forceinline__ uint16_t mmb_f2bf(float f) { uint32_t u = __float_as_uint(f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
// same RNE as mmb_f2bf on both halves, the two high halves joined by one v_perm_b32 (no mov_b16 + and_or). Unguarded
// like the other __builtin_amdgcn_perm uses in this file: device code here is AMD-only.
__device__ __forceinline__ uint32_t mmb_pack2(float a, float b) {
    uint32_t ua = __float_as_uint(a), ub = __float_as_uint(b);
    ua += 0x7fffu + ((ua >> 16) & 1u); ub += 0x7fffu + ((ub >> 16) & 1u);
    return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}
__constant__ int8_t mmb_kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__device__ __forceinline__ float mmb_h2f(uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }
// One Q5_1 block {d|m, qh, qs[16]} (a = {dm, qh, qs0, qs1}, b.xy = {qs2, qs3}) -> 32 BF16 in out[0..16): the per-block
// body of mmb_dq_q51_pair (same fp32 expression and RNE pack, bit-identical), so two threads can share a row.
__device__ __forceinline__ void mmb_dq_q51_one(const uint4 a, const uint4 b, uint32_t * out) {
    const uint32_t dm = a.x, high = a.y;
    const uint32_t qs[4] = {a.z, a.w, b.x, b.y};
    const float d = mmb_h2f((uint16_t)dm), m = mmb_h2f((uint16_t)(dm >> 16));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t q = qs[j];
        float lo[4], hi[4];
#pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            const int low = ((q >> (8 * lane)) & 15) | (((high >> (4 * j + lane)) & 1) << 4);
            const int upper = ((q >> (8 * lane + 4)) & 15) | (((high >> (16 + 4 * j + lane)) & 1) << 4);
            lo[lane] = d * low + m; hi[lane] = d * upper + m;
        }
        out[2*j] = mmb_pack2(lo[0], lo[1]);
        out[2*j + 1] = mmb_pack2(lo[2], lo[3]);
        out[8 + 2*j] = mmb_pack2(hi[0], hi[1]);
        out[8 + 2*j + 1] = mmb_pack2(hi[2], hi[3]);
    }
}

__global__ void mmb_cvt_f32_bf16(const float * __restrict__ x, uint16_t * __restrict__ y, const size_t n) {
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i + 8 <= n) {
        const float4 a = *(const float4 *)(x + i), b = *(const float4 *)(x + i + 4);
        uint4 o; o.x = mmb_pack2(a.x, a.y); o.y = mmb_pack2(a.z, a.w); o.z = mmb_pack2(b.x, b.y); o.w = mmb_pack2(b.z, b.w);
        *(uint4 *)(y + i) = o;
    } else {
        for (; i < n; ++i) y[i] = mmb_f2bf(x[i]);
    }
}

}
#include "mmb-quant.cuh"
namespace {

// dequantize one weight row's two consecutive IQ4_NL blocks (36 bytes) into 64 bf16 in LDS.
// LUT held in registers as (kv + 128) bytes and applied with v_perm_b32 (4 nibbles per op pair) instead of a per-lane
// indexed constant array (which lowers to one scalar-byte memory load per element). The value kv*d is produced as
// fma(kv+128, d, -128*d): -128*d is exact, so the single rounding equals RN(kv*d) -> bitwise the same BF16 as before.
__device__ __forceinline__ void mmb_dq_row36(const uint4 w0, const uint4 w1, const uint32_t w2, uint32_t * arow) {
    const uint32_t ws[9] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2};
    const float d0 = mmb_h2f((uint16_t)(ws[0] & 0xffff)), d1 = mmb_h2f((uint16_t)(ws[4] >> 16));
    const uint32_t q0[4] = { (ws[0] >> 16) | (ws[1] << 16), (ws[1] >> 16) | (ws[2] << 16), (ws[2] >> 16) | (ws[3] << 16), (ws[3] >> 16) | (ws[4] << 16) };
    const uint32_t q1[4] = { ws[5], ws[6], ws[7], ws[8] };
    // kv + 128 = {1,24,45,63,79,93,106,118,129,141,153,166,181,197,217,241} packed little-endian, 4 per dword
    const uint32_t L0 = 0x3f2d1801u, L1 = 0x766a5d4fu, L2 = 0xa6998d81u, L3 = 0xf1d9c5b5u;
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
        const float d = blk ? d1 : d0; const float md = -128.0f * d; const uint32_t * q = blk ? q1 : q0; uint32_t * out = arow + blk * 16;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t v = q[w];
            const uint32_t nib[2] = { v & 0x0F0F0F0Fu, (v >> 4) & 0x0F0F0F0Fu };
            float x[2][4];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t n = nib[h];
                const uint32_t sel = n & 0x07070707u;
                const uint32_t pA = __builtin_amdgcn_perm(L1, L0, sel);   // entries 0..7
                const uint32_t pB = __builtin_amdgcn_perm(L3, L2, sel);   // entries 8..15
                const uint32_t m  = ((n >> 3) & 0x01010101u) * 0xFFu;      // 0xFF where the nibble >= 8
                const uint32_t u  = (pA & ~m) | (pB & m);
                x[h][0] = fmaf((float)(u & 0xFFu), d, md);
                x[h][1] = fmaf((float)((u >> 8) & 0xFFu), d, md);
                x[h][2] = fmaf((float)((u >> 16) & 0xFFu), d, md);
                x[h][3] = fmaf((float)(u >> 24), d, md);
            }
            out[2*w] = mmb_pack2(x[0][0], x[0][1]); out[2*w + 1] = mmb_pack2(x[0][2], x[0][3]);
            out[8 + 2*w] = mmb_pack2(x[1][0], x[1][1]); out[8 + 2*w + 1] = mmb_pack2(x[1][2], x[1][3]);
        }
    }
}

// dequantize one weight row's two consecutive Q8_0 blocks (68 bytes: d0 qs0[32] d1 qs1[32]) into 64 bf16 in LDS
__device__ __forceinline__ void mmb_dq_row68(const uint4 w0, const uint4 w1, const uint4 w2, const uint4 w3, const uint32_t w4, uint32_t * arow) {
    const uint32_t ws[17] = {w0.x,w0.y,w0.z,w0.w, w1.x,w1.y,w1.z,w1.w, w2.x,w2.y,w2.z,w2.w, w3.x,w3.y,w3.z,w3.w, w4};
    const float d0 = mmb_h2f((uint16_t)(ws[0] & 0xffff)), d1 = mmb_h2f((uint16_t)(ws[8] >> 16));
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
        const float d = blk ? d1 : d0; uint32_t * out = arow + blk * 16;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint32_t v = blk ? ws[9 + w] : ((ws[w] >> 16) | (ws[w + 1] << 16));
            const float e0 = d * (float)(int8_t)(v      ), e1 = d * (float)(int8_t)(v >>  8);
            const float e2 = d * (float)(int8_t)(v >> 16), e3 = d * (float)(int8_t)(v >> 24);
            out[2*w] = mmb_pack2(e0, e1); out[2*w + 1] = mmb_pack2(e2, e3);
        }
    }
}

template <typename DRowFn>
__device__ __forceinline__ void mmb_store_tile(const v8f & acc, float * __restrict__ stg, float * __restrict__ D, uint16_t * __restrict__ Dh,
        const bool store_f32, const int M, DRowFn drow, const int n_base, const int m_base, const int lane) {
    const int cm = lane & 15, cn = lane >> 4;
#pragma unroll
    for (int e = 0; e < 8; ++e) { stg[(2 * e + cn) * 16 + cm] = acc[e]; }
    __syncthreads();
    const int n = lane >> 1, half = lane & 1;
    const int dr = drow(n_base + n);
    if (dr >= 0) {
        const float4 v0 = *(const float4 *)(stg + n * 16 + half * 8), v1 = *(const float4 *)(stg + n * 16 + half * 8 + 4);
        const size_t base = (size_t)dr * M + m_base + half * 8;
        const bool full = m_base + 16 <= M;
        if (store_f32) {
            if (full && (M & 3) == 0) { *(float4 *)(D + base) = v0; *(float4 *)(D + base + 4) = v1; }
            else { const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
                   for (int k = 0; k < 8; ++k) { if (m_base + half * 8 + k < M) D[base + k] = vv[k]; } }
        }
        if (Dh) {
            if (full && (M & 7) == 0) { *(uint4 *)(Dh + base) = make_uint4(mmb_pack2(v0.x, v0.y), mmb_pack2(v0.z, v0.w), mmb_pack2(v1.x, v1.y), mmb_pack2(v1.z, v1.w)); }
            else { const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
                   for (int k = 0; k < 8; ++k) { if (m_base + half * 8 + k < M) Dh[base + k] = mmb_f2bf(vv[k]); } }
        }
    }
    __syncthreads();
}
template <int BM, int BN, int WTM, int WTN, int WTYPE, bool TAIL, bool F16 = false, typename XRowFn, typename DRowFn>
__device__ __forceinline__ void mmb_tile_gemm(const uint8_t * __restrict__ Wbase, const size_t wrow_bytes, const int a_rows,
        const uint16_t * __restrict__ Xh, const int K, XRowFn xrow, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, DRowFn drow, const int m0,
        const int n_cols, uint16_t * As, uint16_t * Bs) {
    constexpr int WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    constexpr int A_ITEMS = (BM + MMB_NT - 1) / MMB_NT;
    constexpr int B_ITEMS = (BN * 8) / MMB_NT;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;
    uint4 a0[A_ITEMS], a1[A_ITEMS], a3[A_ITEMS], a4[A_ITEMS], a5[A_ITEMS], a6[A_ITEMS], a7[A_ITEMS], a8[A_ITEMS]; uint32_t a2[A_ITEMS];
    uint4 bst[B_ITEMS];
    int brow[B_ITEMS];
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; brow[i] = xrow(c >> 3); }

    // Q2_0 (routed down): a lane prefetches its 2 qs bytes and d in the WMMA loop, store_lds decodes its 8 weights
    constexpr bool Q20 = WTYPE == 32 + GGML_TYPE_Q2_0;
    constexpr int Q2R = Q20 ? (BM * 8) / MMB_NT : 1;
    uint32_t q20[Q2R];
    // Q5_1 at BM=128 (routed down): two threads per row, one 24-byte block (32 weights) each. One thread per row left
    // half the CTA idle while the other half decoded 64 weights (~600 VALU per stage against 16 WMMA per wave).
    constexpr bool Q51S = false && WTYPE == 32 + GGML_TYPE_Q5_1 && BM * 2 == MMB_NT && MMB_BK == 64;   // off: measured neutral-to-negative (1678 -> 1668 t/s pp2048)

    // gfx1151/ROCm 10.1 clang 24: S119 has seven saves/32 bytes with loop-head BF16 row prefetch (2026-10-24); load the last three vectors near the final WMMAs.
    // Remove this schedule when full-row prefetch is spill-free at the measured throughput.
#if defined(__gfx1151__)
    constexpr bool STAGGER_A = !TAIL && BM == 128 && BN == 256 && WTM == 64 && WTN == 64 && WTYPE == 2;
#else
    constexpr bool STAGGER_A = false;
#endif
    int weight_ks = 0;
    auto load_regs = [&](const int ks) {
        weight_ks = ks;
        if constexpr (Q51S) {
            const int row = tid >> 1;
            if (row < a_rows) {
                const uint2 * p = (const uint2 *)(Wbase + (size_t)row * wrow_bytes + (size_t)(ks * 2 + (tid & 1)) * 24);
                const uint2 w0 = p[0], w1 = p[1], w2 = p[2];
                a0[0] = make_uint4(w0.x, w0.y, w1.x, w1.y); a1[0] = make_uint4(w2.x, w2.y, 0u, 0u);
            } else { a0[0] = a1[0] = make_uint4(0,0,0,0); }   // d = m = 0: zero rows (never stored)
        }
        if constexpr (Q20) {
#pragma unroll
            for (int r = 0; r < Q2R; ++r) {
                const int row = (tid >> 3) + r * (MMB_NT / 8);
                const block_q2_0 * x = (const block_q2_0 *)(Wbase + (size_t)row * wrow_bytes) + ks;
                q20[r] = row < a_rows ? *(const uint16_t *)(x->qs + 2 * (tid & 7)) | ((uint32_t)*(const uint16_t *)&x->d << 16) : 0u;
            }
        }
        if constexpr (!Q51S) {
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) {
            const int row = tid + i * MMB_NT;
            if (row < BM && row < a_rows) {
                if constexpr (WTYPE == 0) { const uint8_t * p = Wbase + (size_t)row * wrow_bytes + (size_t)ks * 36;
                    a0[i] = *(const uint4 *)(p); a1[i] = *(const uint4 *)(p + 16); a2[i] = *(const uint32_t *)(p + 32); }
                else if constexpr (WTYPE == 1) { const uint8_t * p = Wbase + (size_t)row * wrow_bytes + (size_t)ks * 68;
                    a0[i] = *(const uint4 *)(p); a1[i] = *(const uint4 *)(p + 16); a3[i] = *(const uint4 *)(p + 32); a4[i] = *(const uint4 *)(p + 48); a2[i] = *(const uint32_t *)(p + 64); }
                else if constexpr (WTYPE == 32 + GGML_TYPE_Q5_1) {
                    const uint4 * p = (const uint4 *)(Wbase + (size_t)row * wrow_bytes + (size_t)ks * 48);
                    a0[i] = p[0]; a1[i] = p[1]; a3[i] = p[2];
                }
                else if constexpr (WTYPE == 2) { const uint4 * p = (const uint4 *)(Wbase + (size_t)row * wrow_bytes + (size_t)ks * 128);
                    a0[i] = p[0]; a1[i] = p[1]; a3[i] = p[2]; a4[i] = p[3]; a5[i] = p[4];
                    if constexpr (!STAGGER_A) { a6[i] = p[5]; a7[i] = p[6]; a8[i] = p[7]; } }
            } else { a0[i] = make_uint4(0,0,0,0); a1[i] = make_uint4(0,0,0,0); a3[i] = make_uint4(0,0,0,0); a4[i] = make_uint4(0,0,0,0); a5[i] = a6[i] = a7[i] = a8[i] = make_uint4(0,0,0,0); a2[i] = 0; }
        }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {
            const int c = tid + i * MMB_NT; const int off = (c & 7) * 8;
            bst[i] = (brow[i] >= 0) ? *(const uint4 *)(Xh + (size_t)brow[i] * K + ks * MMB_BK + off) : make_uint4(0,0,0,0);
        }
    };
    auto load_tail_regs = [&](const int ks) {
        if constexpr (STAGGER_A) {
#pragma unroll
            for (int i = 0; i < A_ITEMS; ++i) {
                const int row = tid + i * MMB_NT;
                if (row < BM && row < a_rows) {
                    const uint4 * p = (const uint4 *)(Wbase + (size_t)row * wrow_bytes + (size_t)ks * 128);
                    a6[i] = p[5]; a7[i] = p[6]; a8[i] = p[7];
                } else { a6[i] = a7[i] = a8[i] = make_uint4(0,0,0,0); }
            }
        }
    };
    auto store_lds = [&]() {
        if constexpr (Q20) {
#pragma unroll
            for (int r = 0; r < Q2R; ++r) {
                const int row = (tid >> 3) + r * (MMB_NT / 8);
                const float d = mmb_h2f((uint16_t)(q20[r] >> 16));
                float v[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) v[j] = ((int) ((q20[r] >> (2 * j)) & 3) - 1) * d;
                mmb_store8(As + row * MMB_LDS_STRIDE + 8 * (tid & 7), v);
            }
        } else if constexpr (WTYPE >= 32 && WTYPE != 32 + GGML_TYPE_Q5_1) mmb_load_quant_tile<WTYPE, BM, MMB_LDS_STRIDE>(Wbase, wrow_bytes, a_rows, weight_ks, As);
        if constexpr (Q51S) {
            mmb_dq_q51_one(a0[0], a1[0], (uint32_t *)(As + (tid >> 1) * MMB_LDS_STRIDE + (tid & 1) * 32));
        } else {
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) { const int row = tid + i * MMB_NT; if (row < BM) {
            if constexpr (WTYPE == 0) mmb_dq_row36(a0[i], a1[i], a2[i], (uint32_t *)(As + row * MMB_LDS_STRIDE));
            else if constexpr (WTYPE == 1) mmb_dq_row68(a0[i], a1[i], a3[i], a4[i], a2[i], (uint32_t *)(As + row * MMB_LDS_STRIDE));
            else if constexpr (WTYPE == 32 + GGML_TYPE_Q5_1) mmb_dq_q51_pair(a0[i], a1[i], a3[i], (uint32_t *)(As + row * MMB_LDS_STRIDE));
            else if constexpr (WTYPE == 2) { uint4 * d = (uint4 *)(As + row * MMB_LDS_STRIDE); d[0] = a0[i]; d[1] = a1[i]; d[2] = a3[i]; d[3] = a4[i]; d[4] = a5[i]; d[5] = a6[i]; d[6] = a7[i]; d[7] = a8[i]; } } }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; *(uint4 *)(Bs + (c >> 3) * MMB_LDS_STRIDE + (c & 7) * 8) = bst[i]; }
    };

    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

    bool jact[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) jact[j] = !TAIL || (wn * WTN + j * 16) < n_cols;   // whole fragments past the valid columns are never stored
    const int nks = K / MMB_BK;
    load_regs(0); load_tail_regs(0); store_lds(); __syncthreads();
    for (int ks = 0; ks < nks; ++ks) {
        if (ks + 1 < nks) load_regs(ks + 1);
#pragma unroll
        for (int kk = 0; kk < MMB_BK; kk += 16) {
            // gfx1151/ROCm 10.1 clang 24: streaming B clears three dense spill names.
            // Remove this when the full fragment bank is spill-free.
#if defined(__gfx1151__)
            constexpr bool STREAM = !TAIL && ((BM == 128 && BN == 128 && WTM == 32 && WTN == 64 && (WTYPE == 0 || WTYPE == 176 || WTYPE == 181)) || (BM == 128 && BN == 256 && WTM == 64 && WTN == 64 && WTYPE == 2));
#else
            constexpr bool STREAM = false;
#endif
            if constexpr (STREAM) {
                v16s a[TM]; const int r = lane & 15;
#pragma unroll
                for (int i = 0; i < TM; ++i) { const uint16_t * p = As + (wm * WTM + i * 16 + r) * MMB_LDS_STRIDE + kk;
                    a[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)p, *(const uint4 *)(p + 8)}); }
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const uint16_t * p = Bs + (wn * WTN + j * 16 + r) * MMB_LDS_STRIDE + kk;
                    const v16s b = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)p, *(const uint4 *)(p + 8)});
#pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        if constexpr (F16) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b, a[i], acc[i][j]);
                        else               acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b, a[i], acc[i][j]);
                        // gfx1151/ROCm 10.1 clang 24: S119 needs seven saves/32 bytes at loop head (2026-10-24); remove these bounds when full prefetch is spill-free.
#if defined(__gfx1151__)
                        if constexpr (STAGGER_A) {
                            if (kk + 16 == MMB_BK && j + 1 == TN && i + 1 == TM / 2) {
                                __builtin_amdgcn_sched_barrier(0);
                                load_tail_regs(ks + 1 < nks ? ks + 1 : ks);
                                __builtin_amdgcn_sched_barrier(0);
                            }
                        }
#endif
                    }
                }
            } else {
                v16s a[TM], b[TN]; const int r = lane & 15;
#pragma unroll
                for (int i = 0; i < TM; ++i) { const uint16_t * p = As + (wm * WTM + i * 16 + r) * MMB_LDS_STRIDE + kk;
                    const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); a[i] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
                for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; } const uint16_t * p = Bs + (wn * WTN + j * 16 + r) * MMB_LDS_STRIDE + kk;
                    const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); b[j] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; }
                        if constexpr (F16) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b[j], a[i], acc[i][j]);
                        else               acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], a[i], acc[i][j]); }
            }
        }
        __syncthreads();
        if (ks + 1 < nks) store_lds();
        __syncthreads();
    }
    // epilogue through LDS (per-wave 1 KB stage in the now-free A/B tile area); tiles are 16 rows, a_rows is a
    // multiple of 32 in this model, so whole tiles are either valid or beyond a_rows
    float * stg = (float *)((BN * MMB_LDS_STRIDE * 2 >= MMB_NT / 32 * 1024) ? Bs : As) + wave * 256;
    int store_wm = wm;
    // gfx1151/ROCm 10.1 clang 24: the early epilogue offset costs one save/eight private bytes (2026-10-24); remove this bound when the compiler rematerializes it after compute.
#if defined(__gfx1151__)
    if constexpr (STAGGER_A) {
        int store_tid = tid;
        asm volatile("" : "+v"(store_tid));
        store_wm = (store_tid >> 5) % WAVES_M;
    }
#endif
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int ml = store_wm * WTM + i * 16; const bool ok = ml < a_rows;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            if (ok && jact[j]) mmb_store_tile(acc[i][j], stg, D, Dh, store_f32, M, drow, wn * WTN + j * 16, m0 + ml, lane);
            else { __syncthreads(); __syncthreads(); }
        }
    }
}

template <int BM, int BN, int WTM, int WTN, int WTYPE, bool F16 = false>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_dense_kernel(const uint8_t * __restrict__ W, const uint16_t * __restrict__ Xh, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, const int K, const int T) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3)
    NO_DEVICE_CODE; // WMMA kernels are RDNA3-only; the host gate keeps other devices off this path
#else
    __shared__ __align__(16) uint16_t As[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    const int m0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    const size_t wrow_bytes = mmb_row_bytes<WTYPE>(K);
    mmb_tile_gemm<BM, BN, WTM, WTN, WTYPE, false, F16>(W + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (t0 + i < T) ? t0 + i : -1; }, D, Dh, store_f32, M, [&](int i) { return (t0 + i < T) ? t0 + i : -1; }, m0, T - t0, As, Bs);
#endif
}

// ===== Tile-repacked Q8_0 dense GEMM (dense R2; ported from team/alt-dense-isa k7/k8, bf16 path) =====
// The weight is rewritten IN PLACE (same bytes, see ggml_cuda_mmb_dense_r2_prepare) to the layout: per (128-row tile, 64-K step) chunk of 8704 B: qs[128][64] int8 then d[128][2] f16 (same 34 B
// per 32 weights as ggml Q8_0). Every weight load is a coalesced b128 per lane; buffer loads with scalar step offsets;
// barrier without gl0_inv. PF=1: k7 schedule (one register stage); PF=2: k8 schedule (global prefetch distance 2,
// two register stages, nks must be even). Grid is 1-D; GM-grouped raster (GM >= nM: plain M-fastest order).
// Requires M % 128 == 0, K % 64 == 0. bf16 dequant is the same fp32 fma + RNE pack as mmb_dq_row68, same WMMA
// K order -> bit-identical to the WTYPE=1 dense kernel.
constexpr int MMB_TQ8_CHUNK = 8704;
__device__ __forceinline__ void mmb_tq8_lds_barrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
__device__ __forceinline__ void mmb_tq8_dq16(const uint32_t dw, const uint4 q, uint32_t * out /*8 dwords*/) {
    const uint32_t qa[4] = {q.x ^ 0x80808080u, q.y ^ 0x80808080u, q.z ^ 0x80808080u, q.w ^ 0x80808080u};
    const float d = mmb_h2f((uint16_t)(dw & 0xffff)), md = -128.0f * d;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
        const uint32_t v = qa[p >> 1]; const int b = (p & 1) * 2;
        const float u0 = (float)((v >> (8 * b)) & 0xffu), u1 = (float)((v >> (8 * b + 8)) & 0xffu);
        out[p] = mmb_pack2(fmaf(u0, d, md), fmaf(u1, d, md));
    }
}
// dense F16: f16 dequant (team/alt-dense-isa dq16<true>): byte b -> f16 bits 0x64bb = 1024 + (q + 128), minus
// 1152 = q exactly, times d: RN_f16(q * d) with one rounding (0.91 VALU per WMMA vs 2.41 for the fp32 fma + bf16 pack).
typedef _Float16 tq8_h2  __attribute__((ext_vector_type(2)));
typedef _Float16 tq8_v16h __attribute__((ext_vector_type(16)));
__device__ __forceinline__ void mmb_tq8_dq16h(const uint32_t dw, const uint4 q, uint32_t * out /*8 dwords*/) {
    const uint32_t qa[4] = {q.x ^ 0x80808080u, q.y ^ 0x80808080u, q.z ^ 0x80808080u, q.w ^ 0x80808080u};
    const tq8_h2 d2 = __builtin_bit_cast(tq8_h2, __builtin_amdgcn_perm(dw, dw, 0x01000100u));
    const tq8_h2 bias = {(_Float16) 1152.0f, (_Float16) 1152.0f};
#pragma unroll
    for (int p = 0; p < 8; ++p) {
        const uint32_t sel = (p & 1) ? 0x04030402u : 0x04010400u;   // bytes {b, 0x64, b+1, 0x64}
        const uint32_t hv = __builtin_amdgcn_perm(0x64646464u, qa[p >> 1], sel);
        out[p] = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(tq8_h2, hv) - bias) * d2);
    }
}
// F32 -> F16 (RNE), saturated to +-65504: the dense F16 activation format (NaN stays NaN)
__device__ __forceinline__ uint16_t mmb_f2h_sat(const float f) {
    const float c = f != f ? f : fminf(fmaxf(f, -65504.0f), 65504.0f);
    return __builtin_bit_cast(uint16_t, (_Float16) c);
}
__global__ void mmb_cvt_f32_f16_sat(const float * __restrict__ x, uint16_t * __restrict__ y, const size_t n) {
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i + 8 <= n) {
        const float4 a = *(const float4 *)(x + i), b = *(const float4 *)(x + i + 4);
        uint4 o;
        o.x = mmb_f2h_sat(a.x) | ((uint32_t) mmb_f2h_sat(a.y) << 16); o.y = mmb_f2h_sat(a.z) | ((uint32_t) mmb_f2h_sat(a.w) << 16);
        o.z = mmb_f2h_sat(b.x) | ((uint32_t) mmb_f2h_sat(b.y) << 16); o.w = mmb_f2h_sat(b.z) | ((uint32_t) mmb_f2h_sat(b.w) << 16);
        *(uint4 *)(y + i) = o;
    } else {
        for (; i < n; ++i) y[i] = mmb_f2h_sat(x[i]);
    }
}
template <int BM, int BN, int WTM, int WTN, int PF, int LS = 72, bool F16 = false>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_tq8_dense_kernel(const uint8_t * __restrict__ W, const uint16_t * __restrict__ Xh, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32,
        const int M, const int K, const int T, const int GM) {
#if !defined(RDNA3)   // also the host pass: the buffer-resource builtins are device-only and need a recent clang
    GGML_UNUSED_VARS(W, Xh, D, Dh, store_f32, M, K, T, GM);
    NO_DEVICE_CODE;
#else
    static_assert(BM == 128, "row mapping assumes BM=128");
    static_assert(PF == 1 || PF == 2, "prefetch depth 1 or 2");
    constexpr int BK = 64;
    __shared__ __align__(16) uint16_t As[BM * LS];
    __shared__ __align__(16) uint16_t Bs[BN * LS];
    const int nM = (M + BM - 1) / BM, nT = (T + BN - 1) / BN, L = blockIdx.x;
    const int GMc = min(GM, nM);   // GM can be 1<<20 (one group): GM*nT would overflow int
    const int gsz_all = GMc * nT, gid = L / gsz_all, first_m = gid * GMc, gsz = min(nM - first_m, GMc), li = L - gid * gsz_all;
    const int m0 = (first_m + li % gsz) * BM, t0 = (li / gsz) * BN;
    (void) nM;
    constexpr int WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    constexpr int B_ITEMS = (BN * 8) / MMB_NT;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;
    const __amdgpu_buffer_rsrc_t wr = __builtin_amdgcn_make_buffer_rsrc((void *)(W + (size_t)(m0 / BM) * (K / BK) * MMB_TQ8_CHUNK), (short)0, (K / BK) * MMB_TQ8_CHUNK, 0x31004000);
    const __amdgpu_buffer_rsrc_t xr = __builtin_amdgcn_make_buffer_rsrc((void *)(Xh + (size_t)t0 * K), (short)0, min(BN, T - t0) * K * 2, 0x31004000);
    const int xv = ((tid >> 3) * K + (tid & 7) * 8) * 2;           // item i adds i*32 rows (in voffset: soffset is not bounds-checked)
    const int qv = tid * 16, dv = 8192 + ((tid >> 2) * 2 + ((tid & 3) >> 1)) * 2;
    struct Stage { uint4 q0, q1; uint32_t d0, d1; uint4 bst[B_ITEMS]; };
    auto load = [&](Stage & S, const int ks) {
        const int so = ks * MMB_TQ8_CHUNK;
        const auto a0 = __builtin_amdgcn_raw_buffer_load_b128(wr, qv, so, 0);
        const auto a1 = __builtin_amdgcn_raw_buffer_load_b128(wr, qv, so + 4096, 0);
        S.d0 = __builtin_amdgcn_raw_buffer_load_b16(wr, dv, so, 0);
        S.d1 = __builtin_amdgcn_raw_buffer_load_b16(wr, dv, so + 256, 0);
        S.q0 = make_uint4(a0[0], a0[1], a0[2], a0[3]); S.q1 = make_uint4(a1[0], a1[1], a1[2], a1[3]);
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const auto x = __builtin_amdgcn_raw_buffer_load_b128(xr, xv + i * 32 * K * 2, ks * BK * 2, 0); S.bst[i] = make_uint4(x[0], x[1], x[2], x[3]); }
    };
    const int a_st = ((tid >> 2) * LS + (tid & 3) * 16) * 2;     // row r0 = tid>>2; row r1 = r0 + 64
    const int b_st = ((tid >> 3) * LS + (tid & 7) * 8) * 2;
    auto store = [&](const Stage & S) {
        uint32_t o[8], p[8];
        if constexpr (F16) { mmb_tq8_dq16h(S.d0, S.q0, o); mmb_tq8_dq16h(S.d1, S.q1, p); }
        else               { mmb_tq8_dq16(S.d0, S.q0, o); mmb_tq8_dq16(S.d1, S.q1, p); }
        uint8_t * a = (uint8_t *)As + a_st;
        *(uint4 *)(a) = make_uint4(o[0], o[1], o[2], o[3]); *(uint4 *)(a + 16) = make_uint4(o[4], o[5], o[6], o[7]);
        *(uint4 *)(a + 64 * LS * 2) = make_uint4(p[0], p[1], p[2], p[3]); *(uint4 *)(a + 64 * LS * 2 + 16) = make_uint4(p[4], p[5], p[6], p[7]);
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) *(uint4 *)((uint8_t *)Bs + b_st + i * 32 * LS * 2) = S.bst[i];
    };
    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;
    const int nks = K / BK;
    const int r = lane & 15;
    const uint8_t * Ab = (const uint8_t *)As + ((wm * WTM + r) * LS) * 2;
    const uint8_t * Bb = (const uint8_t *)Bs + ((wn * WTN + r) * LS) * 2;
    auto compute = [&]() {
        v16s a[TM], b[TN];
        auto lda = [&](int i, int kk) { const uint8_t * p = Ab + (i * 16 * LS + kk) * 2;
            const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 16); a[i] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); };
        auto ldb = [&](int j, int kk) { const uint8_t * p = Bb + (j * 16 * LS + kk) * 2;
            const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 16); b[j] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); };
#pragma unroll
        for (int i = 0; i < TM; ++i) lda(i, 0);
#pragma unroll
        for (int j = 0; j < TN; ++j) ldb(j, 0);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    if constexpr (F16) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(tq8_v16h, b[j]), __builtin_bit_cast(tq8_v16h, a[i]), acc[i][j]);
                    else               acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], a[i], acc[i][j]);
                    if (i == TM - 1 && kk + 16 < BK) { __builtin_amdgcn_sched_barrier(0); ldb(j, kk + 16); __builtin_amdgcn_sched_barrier(0); }
                }
                if (kk + 16 < BK) { __builtin_amdgcn_sched_barrier(0); lda(i, kk + 16); __builtin_amdgcn_sched_barrier(0); }
            }
        }
    };
    if constexpr (PF == 1) {
        Stage s;
        load(s, 0); store(s); mmb_tq8_lds_barrier();
        for (int ks = 0; ks < nks; ++ks) {
            load(s, min(ks + 1, nks - 1));
            compute();
            mmb_tq8_lds_barrier(); store(s); mmb_tq8_lds_barrier();
        }
    } else {
        Stage sA, sB;
        load(sA, 0); load(sB, min(1, nks - 1)); store(sA); mmb_tq8_lds_barrier();
        for (int ks = 0; ks < nks; ks += 2) {     // nks even
            load(sA, min(ks + 2, nks - 1));
            compute();
            mmb_tq8_lds_barrier(); store(sB); mmb_tq8_lds_barrier();
            load(sB, min(ks + 3, nks - 1));
            compute();
            mmb_tq8_lds_barrier(); store(sA); mmb_tq8_lds_barrier();
        }
    }
    const bool full = (m0 + BM <= M) && (t0 + BN <= T);
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int m = m0 + wm * WTM + i * 16 + (lane & 15);
            const size_t off = (size_t)(t0 + wn * WTN + j * 16 + (lane >> 4)) * M + m;
            if (store_f32) {
                float * dp = D + off;
                if (full) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) dp[(size_t)2 * e * M] = acc[i][j][e];
                } else {
#pragma unroll
                    for (int e = 0; e < 8; ++e) { const int t = t0 + wn * WTN + j * 16 + 2 * e + (lane >> 4); if (t < T && m < M) dp[(size_t)2 * e * M] = acc[i][j][e]; }
                }
            }
            if (Dh) {   // bf16 copy, same RNE as mmb_store_tile
                uint16_t * hp = Dh + off;
                if (full) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) hp[(size_t)2 * e * M] = mmb_f2bf(acc[i][j][e]);
                } else {
#pragma unroll
                    for (int e = 0; e < 8; ++e) { const int t = t0 + wn * WTN + j * 16 + 2 * e + (lane >> 4); if (t < T && m < M) hp[(size_t)2 * e * M] = mmb_f2bf(acc[i][j][e]); }
                }
            }
        }
#endif
}
// ggml Q8_0 rows (K/32 blocks of {f16 d, int8 qs[32]}) -> tile-repacked layout above; one thread per Q8_0 block.
__global__ void mmb_tq8_repack_kernel(const uint8_t * __restrict__ W, uint8_t * __restrict__ out, const int M, const int K) {
    const int nkb = K / 32;
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) M * nkb) return;
    const int m = (int) (i / nkb), b = (int) (i % nkb);
    const int mt = m / 128, r = m % 128, ks = b / 2, bb = b % 2;
    const uint8_t * src = W + (size_t) m * nkb * 34 + (size_t) b * 34;
    uint8_t * c = out + ((size_t) mt * (K / 64) + ks) * MMB_TQ8_CHUNK;
    c[8192 + (r * 2 + bb) * 2] = src[0]; c[8192 + (r * 2 + bb) * 2 + 1] = src[1];
#pragma unroll
    for (int k = 0; k < 32; ++k) c[r * 64 + bb * 32 + k] = src[2 + k];
}

#if defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float gm_mul_rn(const float a, const float b) { float r; asm("v_mul_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r; }
__device__ __forceinline__ float gm_add_rn(const float a, const float b) { float r; asm("v_add_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r; }
#else
__device__ __forceinline__ float gm_mul_rn(const float a, const float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float gm_add_rn(const float a, const float b) { return __fadd_rn(a, b); }
#endif
__device__ __forceinline__ float gm_sigmoid(const float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float gm_bf2f(const uint16_t h) { return __uint_as_float(((uint32_t) h) << 16); }

// CH channels x BN tokens per CTA, every wave 16 channels (all HC streams) x 32 tokens. CH=64 (BN=64): the epilogue's
// Xn reads of one token row are a whole 128-byte line inside one CTA (CH=32 split each line across two CTAs that are
// rarely co-resident: 149 MB fetched per call for a 42 MB Xn on gfx1151 pp2048).
typedef unsigned int mmb_u4v __attribute__((ext_vector_type(4)));   // not uint4: a copied struct array stays in scratch
template <int HC, int WTYPE = 0, int CH = 32, int TOKW = 32>
__global__ void __launch_bounds__(MMB_NT, 2)
hc_gate_mix_kernel(const uint8_t * __restrict__ W, const uint16_t * __restrict__ Lo, const uint16_t * __restrict__ Xn, float * __restrict__ Out,
        uint16_t * __restrict__ OutH, uint16_t * __restrict__ OutF, const bool store_f32,
        const int E, const int K, const int T, const float scale, const float bias) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3)
    // WMMA (wave32, 16x16x16 bf16/f16 into f32) exists only on RDNA3; the host gate keeps other devices off this path
    NO_DEVICE_CODE;
#else
    constexpr int WAVES_M = CH / 16, WAVES_N = (MMB_NT / 32) / WAVES_M, BN = WAVES_N * TOKW, BM = HC * CH, JT = TOKW / 16;
    static_assert(BM <= MMB_NT, "one A row per thread");
    static_assert(WAVES_M * WAVES_N == MMB_NT / 32 && WAVES_M * 16 == CH && WAVES_N * TOKW == BN, "full channel x token coverage");
    __shared__ __align__(16) uint16_t As[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;     // wave: 16 channels (all HC streams) x 32 tokens
    const int e0 = blockIdx.x * CH, t0 = blockIdx.y * BN;
    const size_t wrow_bytes = mmb_row_bytes<WTYPE>(K);
    constexpr int B_ITEMS = (BN * 8) / MMB_NT;
    uint4 a0 = make_uint4(0,0,0,0), a1 = make_uint4(0,0,0,0); uint32_t a2 = 0; mmb_u4v bst[B_ITEMS]; int brow[B_ITEMS];
    mmb_u4v aw[WTYPE == 2 ? 8 : 1];
    const uint8_t * arow = W;
    if (tid < BM) { const int c = tid / CH, i = tid - c * CH; arow = W + (size_t)(c * E + e0 + i) * wrow_bytes; }
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; const int t = t0 + (c >> 3); brow[i] = t < T ? t : -1; }
    int weight_ks = 0;
    auto load_regs = [&](const int ks) {
        weight_ks = ks;
        if constexpr (WTYPE == 0) if (tid < BM) { const uint8_t * p = arow + (size_t)ks * 36; a0 = *(const uint4 *)(p); a1 = *(const uint4 *)(p + 16); a2 = *(const uint32_t *)(p + 32); }
        if constexpr (WTYPE == 2) if (tid < BM) { const mmb_u4v * p = (const mmb_u4v *)(arow + (size_t)ks * MMB_BK * 2);
#pragma unroll
            for (int q = 0; q < 8; ++q) aw[q] = p[q]; }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; const int off = (c & 7) * 8;
            bst[i] = (brow[i] >= 0) ? *(const mmb_u4v *)(Lo + (size_t)brow[i] * K + ks * MMB_BK + off) : mmb_u4v{0, 0, 0, 0}; }
    };
    auto store_lds = [&]() {
        if constexpr (WTYPE == 0) { if (tid < BM) mmb_dq_row36(a0, a1, a2, (uint32_t *)(As + tid * MMB_LDS_STRIDE)); }
        else if constexpr (WTYPE == 2) { if (tid < BM) {
#pragma unroll
            for (int q = 0; q < 8; ++q) *(mmb_u4v *)(As + tid * MMB_LDS_STRIDE + q * 8) = aw[q]; } }
        else {
            constexpr int TYPE = WTYPE == 1 ? GGML_TYPE_Q8_0 : WTYPE - 32;
            for (int row = tid / 8; row < BM; row += MMB_NT / 8) {
                const int c = row / CH, ch = row % CH;
                mmb_decode_slice<(ggml_type)TYPE>(W + (size_t)(c * E + e0 + ch) * wrow_bytes, weight_ks * 64, As + row * MMB_LDS_STRIDE, tid % 8);
            }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; *(mmb_u4v *)(Bs + (c >> 3) * MMB_LDS_STRIDE + (c & 7) * 8) = bst[i]; }
    };
    v8f acc[HC][JT];
#pragma unroll
    for (int c = 0; c < HC; ++c)
#pragma unroll
        for (int j = 0; j < JT; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[c][j][e] = 0.f;
    const int nks = K / MMB_BK;
    load_regs(0); store_lds(); __syncthreads();
    for (int ks = 0; ks < nks; ++ks) {
        // unconditional prefetch/store (the last ones are unused): a conditional def kept the staged registers out of
        // SROA (144 B/lane scratch, vmcnt(0) right after each load on gfx1151)
        load_regs(min(ks + 1, nks - 1));
#pragma unroll
        for (int kk = 0; kk < MMB_BK; kk += 16) {
            v16s a[HC], b[JT]; const int r = lane & 15;
#pragma unroll
            for (int c = 0; c < HC; ++c) { const uint16_t * p = As + (c * CH + wm * 16 + r) * MMB_LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); a[c] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
            for (int j = 0; j < JT; ++j) { const uint16_t * p = Bs + (wn * TOKW + j * 16 + r) * MMB_LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); b[j] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
            for (int c = 0; c < HC; ++c)
#pragma unroll
                for (int j = 0; j < JT; ++j) acc[c][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], a[c], acc[c][j]);
        }
        __syncthreads();
        store_lds();
        __syncthreads();
    }
    // epilogue: lane holds channel (lane & 15) of the wave's 16 and tokens 2e + (lane >> 4) of each 16-token fragment
    const int cm = lane & 15, cn = lane >> 4; const int ch = e0 + wm * 16 + cm;
#pragma unroll
    for (int j = 0; j < JT; ++j) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int t = t0 + wn * TOKW + j * 16 + 2 * e + cn;
            if (t >= T) continue;
            const uint16_t * xr = Xn + (size_t)t * ((size_t)HC * E) + ch;
            float s = 0.f;
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                const float g = __uint_as_float(((uint32_t) mmb_f2bf(acc[c][j][e])) << 16);   // the gate GEMM's BF16 epilogue rounding
                const float term = gm_mul_rn(gm_bf2f(xr[(size_t)c * E]), gm_sigmoid(g));
                s = (c == 0) ? term : gm_add_rn(s, term);
            }
            const float o = scale * s + bias;
            if (store_f32) Out[(size_t)t * E + ch] = o;
            if (OutH) OutH[(size_t)t * E + ch] = mmb_f2bf(o);
            if (OutF) OutF[(size_t)t * E + ch] = mmb_f2h_sat(o);   // dense F16 copy
        }
    }
#endif
}

template <int BM, int BN, int WTM, int WTN, int WTYPE = 0>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_routed_kernel(const uint8_t * __restrict__ W, const size_t expert_bytes, const uint16_t * __restrict__ Xh, float * __restrict__ D,
        uint16_t * __restrict__ Dh, const bool store_f32,
        const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const uint32_t * __restrict__ desc, const int M, const int K) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3)
    NO_DEVICE_CODE; // WMMA kernels are RDNA3-only; the host gate keeps other devices off this path
#else
    __shared__ __align__(16) uint16_t As[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    const uint32_t dsc = desc[blockIdx.y];
    if (dsc == UINT32_MAX) return;   // uniform across the block, before any barrier
    const int e = dsc & 0xffff, jt = dsc >> 16;
    const int r0 = bounds[e] + jt * BN, cnt = bounds[e + 1] - r0;
    const int m0 = blockIdx.x * BM;
    const size_t wrow_bytes = mmb_row_bytes<WTYPE>(K);
    mmb_tile_gemm<BM, BN, WTM, WTN, WTYPE, true>(W + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (i < cnt) ? ids_src[r0 + i] : -1; }, D, Dh, store_f32, M, [&](int i) { return (i < cnt) ? ids_dst[r0 + i] : -1; }, m0, cnt, As, Bs);
#endif
}

template <int BM, int BN, int WTM, int WTN, int WTYPE, bool TAIL, bool Q4K_HALF, typename XRowFn, typename DRowFn>
__device__ __forceinline__ void mmb_tile_gemm_glu(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t wrow_bytes, const int a_rows,
        const uint16_t * __restrict__ Xh, const int K, XRowFn xrow, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, DRowFn drow, const int m0,
        const int n_cols, uint16_t * Ag, uint16_t * Au, uint16_t * Bs) {
    constexpr int WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    constexpr int A_ITEMS = (BM + MMB_NT - 1) / MMB_NT;
    constexpr int B_ITEMS = (BN * 8) / MMB_NT;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;
    uint4 g0[A_ITEMS], g1[A_ITEMS], u0[A_ITEMS], u1[A_ITEMS]; uint32_t g2[A_ITEMS], u2[A_ITEMS];
    uint4 gm[A_ITEMS], um[A_ITEMS];
    uint4 bst[B_ITEMS];
    int brow[B_ITEMS];
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; brow[i] = xrow(c >> 3); }
    // grid-based types (mmb_lb): prefetch each lane's block fields into registers during the WMMA loop, grid in
    // LDS, decode in store_lds. Same fp32 math as dequantize_* -> bit-identical.
    using LBT = mmb_lb<WTYPE>;
    constexpr bool LB = LBT::ok;
    constexpr int LBR = LB ? (BM * 8) / MMB_NT : 1;
    uint32_t lbg[LBR][2], lbu[LBR][2];
    __shared__ typename LBT::grid_t lbgrid[LBT::N];
    if constexpr (LB) { for (int i = tid; i < LBT::N; i += MMB_NT) lbgrid[i] = LBT::entry(i); }
    // gfx1151/ROCm 10.1 clang 24: direct activation staging clears 20 wide-GLU spill names without changing the 64x256 tile.
    // Remove this when activation prefetch keeps these tiles spill-free.
#if defined(__gfx1151__)
    constexpr bool DEFER_B = BM == 64 && BN == 256 && WTM == 32 && WTN == 64;
#else
    constexpr bool DEFER_B = false;
#endif
    int weight_ks = 0;
    auto load_regs = [&](const int ks) {
        weight_ks = ks;
        if constexpr (LB) {
            const int l8 = tid & 7, sub = l8 & 1, il = l8 >> 1;
#pragma unroll
            for (int r = 0; r < LBR; ++r) {
                const int row = (tid >> 3) + r * (MMB_NT / 8);
                if (row < a_rows) {
                    LBT::fetch(Wg + (size_t)row * wrow_bytes, ks, sub, il, lbg[r][0], lbg[r][1]);
                    LBT::fetch(Wu + (size_t)row * wrow_bytes, ks, sub, il, lbu[r][0], lbu[r][1]);
                } else { lbg[r][0] = lbg[r][1] = lbu[r][0] = lbu[r][1] = 0u; } // d = 0 -> zero rows (never stored)
            }
        }
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) {
            const int row = Q4K_HALF ? (tid + i * MMB_NT) / 2 : tid + i * MMB_NT;
            if constexpr (WTYPE == 0) {
            if (row < BM && row < a_rows) {
                const uint8_t * pg = Wg + (size_t)row * wrow_bytes + (size_t)ks * 36;
                const uint8_t * pu = Wu + (size_t)row * wrow_bytes + (size_t)ks * 36;
                g0[i] = *(const uint4 *)(pg); g1[i] = *(const uint4 *)(pg + 16); g2[i] = *(const uint32_t *)(pg + 32);
                u0[i] = *(const uint4 *)(pu); u1[i] = *(const uint4 *)(pu + 16); u2[i] = *(const uint32_t *)(pu + 32);
            } else { g0[i] = g1[i] = u0[i] = u1[i] = make_uint4(0,0,0,0); g2[i] = u2[i] = 0; }
            } else if constexpr (WTYPE == 32 + GGML_TYPE_Q4_K) {
                if (row < BM && row < a_rows) {
                    const uint8_t * pg = Wg + (size_t)row * wrow_bytes + (size_t)(ks / 4) * sizeof(block_q4_K);
                    const uint8_t * pu = Wu + (size_t)row * wrow_bytes + (size_t)(ks / 4) * sizeof(block_q4_K);
                    gm[i] = *(const uint4 *)pg; um[i] = *(const uint4 *)pu;
                    const int offset = 16 + (ks & 3) * 32 + (Q4K_HALF ? (tid & 1) * 16 : 0);
                    g0[i] = *(const uint4 *)(pg + offset); u0[i] = *(const uint4 *)(pu + offset);
                    if constexpr (!Q4K_HALF) {
                        g1[i] = *(const uint4 *)(pg + offset + 16); u1[i] = *(const uint4 *)(pu + offset + 16);
                    }
                } else { gm[i] = um[i] = g0[i] = g1[i] = u0[i] = u1[i] = make_uint4(0,0,0,0); }
            }
        }
        if constexpr (!DEFER_B) {
#pragma unroll
            for (int i = 0; i < B_ITEMS; ++i) {
                const int c = tid + i * MMB_NT; const int off = (c & 7) * 8;
                bst[i] = (brow[i] >= 0) ? *(const uint4 *)(Xh + (size_t)brow[i] * K + ks * MMB_BK + off) : make_uint4(0,0,0,0);
            }
        }
    };
    auto store_lds = [&]() {
        if constexpr (LB) {
            // branchless (padding rows decode to zero); same fp32 arithmetic and the same RNE + sign-XOR packing
            // as dequantize_*, so the BF16 weights are bit-identical to the generic decoder path
            const int l8 = tid & 7, sub = l8 & 1, il = l8 >> 1;
#pragma unroll
            for (int r = 0; r < LBR; ++r) {
                const int row = (tid >> 3) + r * (MMB_NT / 8);
                *(uint4 *)(Ag + row * MMB_LDS_STRIDE + 32 * sub + 8 * il) = LBT::decode(lbgrid, lbg[r][0], lbg[r][1], il);
                *(uint4 *)(Au + row * MMB_LDS_STRIDE + 32 * sub + 8 * il) = LBT::decode(lbgrid, lbu[r][0], lbu[r][1], il);
            }
        } else if constexpr (WTYPE != 0 && WTYPE != 32 + GGML_TYPE_Q4_K) {
            constexpr int LOAD_TYPE = WTYPE == 1 ? 32 + GGML_TYPE_Q8_0 : WTYPE;
            mmb_load_quant_tile<LOAD_TYPE, BM, MMB_LDS_STRIDE>(Wg, wrow_bytes, a_rows, weight_ks, Ag);
            mmb_load_quant_tile<LOAD_TYPE, BM, MMB_LDS_STRIDE>(Wu, wrow_bytes, a_rows, weight_ks, Au);
        }

#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) {
            const int row = Q4K_HALF ? (tid + i * MMB_NT) / 2 : tid + i * MMB_NT;
            if (row < BM) {
                if constexpr (WTYPE == 0) {
                    mmb_dq_row36(g0[i], g1[i], g2[i], (uint32_t *)(Ag + row * MMB_LDS_STRIDE));
                    mmb_dq_row36(u0[i], u1[i], u2[i], (uint32_t *)(Au + row * MMB_LDS_STRIDE));
                } else if constexpr (WTYPE == 32 + GGML_TYPE_Q4_K) {
                    mmb_dq_q4k_slice<Q4K_HALF>(g0[i], Q4K_HALF ? g0[i] : g1[i], gm[i], weight_ks & 3, (uint32_t *)(Ag + row * MMB_LDS_STRIDE), tid & 1);
                    mmb_dq_q4k_slice<Q4K_HALF>(u0[i], Q4K_HALF ? u0[i] : u1[i], um[i], weight_ks & 3, (uint32_t *)(Au + row * MMB_LDS_STRIDE), tid & 1);
                }
            }
        }
        if constexpr (DEFER_B) {
#pragma unroll 2
            for (int i = 0; i < B_ITEMS; ++i) {
                const int c = tid + i * MMB_NT, row = xrow(c >> 3);
                *(uint4 *)(Bs + (c >> 3) * MMB_LDS_STRIDE + (c & 7) * 8) = row >= 0 ? *(const uint4 *)(Xh + (size_t)row * K + weight_ks * MMB_BK + (c & 7) * 8) : make_uint4(0,0,0,0);
            }
        } else {
#pragma unroll
            for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; *(uint4 *)(Bs + (c >> 3) * MMB_LDS_STRIDE + (c & 7) * 8) = bst[i]; }
        }
    };
    v8f accg[TM][TN], accu[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) { accg[i][j][e] = 0.f; accu[i][j][e] = 0.f; }
    bool jact[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) jact[j] = !TAIL || (wn * WTN + j * 16) < n_cols;
    const int nks = K / MMB_BK;
    load_regs(0); if constexpr (LB) __syncthreads(); store_lds(); __syncthreads();
    for (int ks = 0; ks < nks; ++ks) {
        if (ks + 1 < nks) load_regs(ks + 1);
#pragma unroll
        for (int kk = 0; kk < MMB_BK; kk += 16) {
            // gfx1151/ROCm 10.1 clang 24: streaming gate/up fragments clears 20 GLU spill names.
            // Remove this when both fragment banks are spill-free.
#if defined(__gfx1151__)
            constexpr bool STREAM = (BM == 128 && BN == 128 && WTM == 64 && WTN == 32) || (BM == 64 && BN == 256 && WTM == 32 && WTN == 64);
#else
            constexpr bool STREAM = false;
#endif
            if constexpr (STREAM) {
                v16s b[TN]; const int r = lane & 15;
#pragma unroll
                for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; } const uint16_t * p = Bs + (wn * WTN + j * 16 + r) * MMB_LDS_STRIDE + kk;
                    b[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)p, *(const uint4 *)(p + 8)}); }
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const int off = (wm * WTM + i * 16 + r) * MMB_LDS_STRIDE + kk;
                    {
                        const v16s a = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Ag + off), *(const uint4 *)(Ag + off + 8)});
#pragma unroll
                        for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; }
                            accg[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], a, accg[i][j]); }
                        }
                    {
                        const v16s a = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Au + off), *(const uint4 *)(Au + off + 8)});
#pragma unroll
                        for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; }
                            accu[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], a, accu[i][j]); }
                        }
                }
            } else {
                v16s ag[TM], au[TM], b[TN]; const int r = lane & 15;
#pragma unroll
                for (int i = 0; i < TM; ++i) { const int off = (wm * WTM + i * 16 + r) * MMB_LDS_STRIDE + kk;
                    ag[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Ag + off), *(const uint4 *)(Ag + off + 8)});
                    au[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Au + off), *(const uint4 *)(Au + off + 8)}); }
#pragma unroll
                for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; } const uint16_t * p = Bs + (wn * WTN + j * 16 + r) * MMB_LDS_STRIDE + kk;
                    b[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)p, *(const uint4 *)(p + 8)}); }
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        if constexpr (TAIL) { if (!jact[j]) continue; }
                        accg[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], ag[i], accg[i][j]);
                        accu[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], au[i], accu[i][j]);
                    }
            }
        }
        __syncthreads();
        if (ks + 1 < nks) store_lds();
        __syncthreads();
    }
    float * stg = (float *)((BN * MMB_LDS_STRIDE * 2 >= MMB_NT / 32 * 1024) ? Bs : Ag) + wave * 256;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int ml = wm * WTM + i * 16; const bool ok = ml < a_rows;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            if (ok && jact[j]) {
                v8f v;
#pragma unroll
                for (int e = 0; e < 8; ++e) { v[e] = ggml_cuda_op_silu_single(accg[i][j][e]) * accu[i][j][e]; }
                mmb_store_tile(v, stg, D, Dh, store_f32, M, drow, wn * WTN + j * 16, m0 + ml, lane);
            } else { __syncthreads(); __syncthreads(); }
        }
    }
}

template <int BM, int BN, int WTM, int WTN, int WTYPE = 0, bool Q4K_HALF = false>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_routed_glu_kernel(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t expert_bytes, const uint16_t * __restrict__ Xh,
        float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32,
        const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const uint32_t * __restrict__ desc, const int M, const int K) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3)
    NO_DEVICE_CODE; // WMMA kernels are RDNA3-only; the host gate keeps other devices off this path
#else
    __shared__ __align__(16) uint16_t Ag[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Au[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    const uint32_t dsc = desc[blockIdx.y];
    if (dsc == UINT32_MAX) return;
    const int e = dsc & 0xffff, jt = dsc >> 16;
    const int r0 = bounds[e] + jt * BN, cnt = bounds[e + 1] - r0;
    const int m0 = blockIdx.x * BM;
    const size_t wrow_bytes = mmb_row_bytes<WTYPE>(K);
    mmb_tile_gemm_glu<BM, BN, WTM, WTN, WTYPE, true, Q4K_HALF>(Wg + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, Wu + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (i < cnt) ? ids_src[r0 + i] : -1; }, D, Dh, store_f32, M, [&](int i) { return (i < cnt) ? ids_dst[r0 + i] : -1; }, m0, cnt, Ag, Au, Bs);
#endif
}

// F32 x F32 -> F32 GEMM on WMMA: operands split into F16 hi + lo at tile load; TWO: Xhi*Whi + Xlo*Whi, !TWO adds Xhi*Wlo, !XSPLIT keeps only Xhi*Whi.
__device__ __forceinline__ void mmb_split2(float x, uint16_t & hi, uint16_t & lo) {
    hi = __builtin_bit_cast(uint16_t, (_Float16) x); lo = __builtin_bit_cast(uint16_t, (_Float16) (x - (float) __builtin_bit_cast(_Float16, hi)));
}
// EPI == 1 (qwen4exp QSA indexer scorer): the T columns are (query, head) pairs with 4 heads, and D is [M, T/4] with
// D[q*M + m] = ((relu(h0) + relu(h1)) + relu(h2)) + relu(h3) (+ log(step(float(tails[q]) - float(starts[m]))) when
// tails != nullptr): the exact expression the separate GEMM, idx_relu_sum and visibility chain evaluate.
template <int BM, int BN, int WTM, int WTN, bool TWO, bool XSPLIT, int EPI = 0>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_f32split_kernel(const float * __restrict__ W, const float * __restrict__ X, float * __restrict__ D, const int M, const int K, const int T,
        const float * __restrict__ W2 = nullptr, float * __restrict__ D2 = nullptr, const int M1 = 0,
        const int32_t * __restrict__ tails = nullptr, const int32_t * __restrict__ starts = nullptr) {
    // W2 != nullptr: two GEMMs on the same X, rows [0, M1) from W -> D (ld M1) and rows [M1, M) from W2 -> D2 (ld M - M1)
#if defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3)
    // WMMA (wave32, 16x16x16 bf16/f16 into f32) exists only on RDNA3; the host gate keeps other devices off this path
    NO_DEVICE_CODE;
#else
    constexpr int BKs = 32, LS = BKs + 8, WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    static_assert(WAVES_M * (BN / WTN) * 32 == MMB_NT, "tile must use every wave");
    __shared__ __align__(16) uint16_t Ah[BM * LS], Al[TWO ? 8 : BM * LS], Bh[BN * LS], Bl[XSPLIT ? BN * LS : 8];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, wm = wave % WAVES_M, wn = wave / WAVES_M;
    const int m0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;
    constexpr int A_CH = BM * BKs / 4, B_CH = BN * BKs / 4;
    // one register stage: the next K slice is fetched while the current one is multiplied (unguarded loads with
    // clamped rows, zeroed at the LDS store: a divergent guard would make the compiler drain them at the join)
    constexpr int A_IT = (A_CH + MMB_NT - 1) / MMB_NT, B_IT = (B_CH + MMB_NT - 1) / MMB_NT;
    static_assert(A_CH % MMB_NT == 0 && B_CH % MMB_NT == 0, "f32split staging assumes whole passes");
    float4 ra[A_IT], rb[B_IT];
    auto gload = [&](const int k0) {
#pragma unroll
        for (int i = 0; i < A_IT; ++i) { const int idx = tid + i * MMB_NT, row = idx >> 3, c4 = (idx & 7) * 4;
            const int gr = min(m0 + row, M - 1);
            const float * wr = (W2 && gr >= M1) ? W2 + (size_t) (gr - M1) * K : W + (size_t) gr * K;
            ra[i] = *(const float4 *)(wr + k0 + c4); }
#pragma unroll
        for (int i = 0; i < B_IT; ++i) { const int idx = tid + i * MMB_NT, row = idx >> 3, c4 = (idx & 7) * 4;
            rb[i] = *(const float4 *)(X + (size_t) min(t0 + row, T - 1) * K + k0 + c4); }
    };
    auto lstore = [&]() {
#pragma unroll
        for (int i = 0; i < A_IT; ++i) { const int idx = tid + i * MMB_NT, row = idx >> 3, c4 = (idx & 7) * 4;
            const float4 v = m0 + row < M ? ra[i] : make_float4(0.f,0.f,0.f,0.f);
            uint16_t h[4], l[4]; mmb_split2(v.x,h[0],l[0]); mmb_split2(v.y,h[1],l[1]); mmb_split2(v.z,h[2],l[2]); mmb_split2(v.w,h[3],l[3]);
            *(uint2 *)(Ah + row * LS + c4) = make_uint2((uint32_t)h[0] | ((uint32_t)h[1] << 16), (uint32_t)h[2] | ((uint32_t)h[3] << 16));
            if constexpr (!TWO) *(uint2 *)(Al + row * LS + c4) = make_uint2((uint32_t)l[0] | ((uint32_t)l[1] << 16), (uint32_t)l[2] | ((uint32_t)l[3] << 16)); }
#pragma unroll
        for (int i = 0; i < B_IT; ++i) { const int idx = tid + i * MMB_NT, row = idx >> 3, c4 = (idx & 7) * 4;
            const float4 v = t0 + row < T ? rb[i] : make_float4(0.f,0.f,0.f,0.f);
            uint16_t h[4], l[4]; mmb_split2(v.x,h[0],l[0]); mmb_split2(v.y,h[1],l[1]); mmb_split2(v.z,h[2],l[2]); mmb_split2(v.w,h[3],l[3]);
            *(uint2 *)(Bh + row * LS + c4) = make_uint2((uint32_t)h[0] | ((uint32_t)h[1] << 16), (uint32_t)h[2] | ((uint32_t)h[3] << 16));
            if constexpr (XSPLIT) *(uint2 *)(Bl + row * LS + c4) = make_uint2((uint32_t)l[0] | ((uint32_t)l[1] << 16), (uint32_t)l[2] | ((uint32_t)l[3] << 16)); }
    };
    gload(0);
    for (int k0 = 0; k0 < K; k0 += BKs) {
        lstore();
        __syncthreads();
        gload(min(k0 + BKs, K - BKs));
        const int r = lane & 15;
#pragma unroll
        for (int kk = 0; kk < BKs; kk += 16) {
            v16s ah[TM], al[TM], bh[TN], bl[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) { const int off = (wm * WTM + i * 16 + r) * LS + kk;
                ah[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Ah + off), *(const uint4 *)(Ah + off + 8)});
                if constexpr (!TWO) al[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Al + off), *(const uint4 *)(Al + off + 8)}); }
#pragma unroll
            for (int j = 0; j < TN; ++j) { const int off = (wn * WTN + j * 16 + r) * LS + kk;
                bh[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Bh + off), *(const uint4 *)(Bh + off + 8)});
                if constexpr (XSPLIT) bl[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Bl + off), *(const uint4 *)(Bl + off + 8)}); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bh[j], ah[i], acc[i][j]);
                    if constexpr (XSPLIT) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bl[j], ah[i], acc[i][j]);
                    if constexpr (!TWO) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bh[j], al[i], acc[i][j]);
                }
        }
        __syncthreads();
    }
    const int cm = lane & 15, cn = lane >> 4;
    if constexpr (EPI == 1) {
        // lane holds t = base + 4p + cn (e = 2p) and base + 4p + 2 + cn (e = 2p + 1): heads cn and 2 + cn of query
        // base/4 + p; lane ^ 16 holds the other two heads of the same (m, query). The shuffles run on every lane.
        const int Q = T >> 2;
#pragma unroll
        for (int i = 0; i < TM; ++i) { const int m = m0 + wm * WTM + i * 16 + cm;
            const float sm = (tails && m < M) ? (float) starts[m] : 0.0f;
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                float o[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) o[e] = __shfl_xor(acc[i][j][e], 16, 32);
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float h0 = cn == 0 ? acc[i][j][2 * p]     : o[2 * p];
                    const float h1 = cn == 0 ? o[2 * p]             : acc[i][j][2 * p];
                    const float h2 = cn == 0 ? acc[i][j][2 * p + 1] : o[2 * p + 1];
                    const float h3 = cn == 0 ? o[2 * p + 1]         : acc[i][j][2 * p + 1];
                    float s = fmaxf(h0, 0.0f);
                    s = s + fmaxf(h1, 0.0f);
                    s = s + fmaxf(h2, 0.0f);
                    s = s + fmaxf(h3, 0.0f);
                    const int q = ((t0 + wn * WTN + j * 16) >> 2) + p;
                    if ((p >> 1) == cn && m < M && q < Q) {
                        if (tails) {
                            const float d    = (float) tails[q] - sm;
                            const float step = d > 0.0f;
                            s = s + logf(step);
                        }
                        D[(size_t) q * M + m] = s;
                    }
                }
            }
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) { const int m = m0 + wm * WTM + i * 16 + cm; if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) { const int t = t0 + wn * WTN + j * 16 + 2 * e + cn; if (t < T) {
                if (W2) { if (m < M1) D[(size_t) t * M1 + m] = acc[i][j][e]; else D2[(size_t) t * (M - M1) + (m - M1)] = acc[i][j][e]; }
                else D[(size_t) t * M + m] = acc[i][j][e]; } } }
#endif
}


// qwen4exp QSA indexer scorer with the keys stationary. mmb_f32split_kernel<..., EPI=1> re-reads its key tile for every
// 128-column tile of the 2048 (query, head) columns, which spills the ~32 MB of pooled F32 keys per layer (250k depth)
// out of the Infinity Cache. Here each workgroup rounds its BM key rows to F16 into LDS once and walks all T columns,
// staging each 128-column chunk of the queries (pre-rounded to F16 once per call) in LDS. Every output tile issues the
// same WMMAs on the same F16 fragments in the same K order as the split kernel (TWO, !XSPLIT: hi parts only), and the
// epilogue computes the same values, so the result matches it bit for bit.
__global__ void mmb_f32_to_f16_kernel(const float * __restrict__ x, uint16_t * __restrict__ y, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __builtin_bit_cast(uint16_t, (_Float16) x[i]);
}

template <int BM, int BN, int WAVES_M>
__global__ void __launch_bounds__(MMB_NT, 1)
mmb_idx_score_kernel(const float * __restrict__ W, const uint16_t * __restrict__ Xh, float * __restrict__ D,
        const int M, const int K, const int T, const int32_t * __restrict__ tails, const int32_t * __restrict__ starts) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3)
    NO_DEVICE_CODE;
#else
    constexpr int KMAX = 128, LS = KMAX + 8;
    constexpr int WAVES_N = MMB_NT / 32 / WAVES_M, WTM = BM / WAVES_M, WTN = BN / WAVES_N, TM = WTM / 16, TN = WTN / 16;
    static_assert(WAVES_M * WAVES_N * 32 == MMB_NT && TM >= 1 && TN >= 1 && WTM % 16 == 0 && WTN % 16 == 0, "tile must use every wave");
    __shared__ __align__(16) uint16_t Ah[BM * LS];
    __shared__ __align__(16) uint16_t Bh[BN * LS];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, wm = wave % WAVES_M, wn = wave / WAVES_M;
    const int m0 = blockIdx.x * BM;
    const int K4 = K / 4, K8 = K / 8;

    // keys -> F16 (the hi part of mmb_split2), rows past M are zero
    for (int idx = tid; idx < BM * K4; idx += MMB_NT) {
        const int row = idx / K4, c4 = (idx % K4) * 4;
        const float4 v = m0 + row < M ? *(const float4 *)(W + (size_t) (m0 + row) * K + c4) : make_float4(0.f, 0.f, 0.f, 0.f);
        const uint32_t h0 = __builtin_bit_cast(uint16_t, (_Float16) v.x), h1 = __builtin_bit_cast(uint16_t, (_Float16) v.y);
        const uint32_t h2 = __builtin_bit_cast(uint16_t, (_Float16) v.z), h3 = __builtin_bit_cast(uint16_t, (_Float16) v.w);
        *(uint2 *)(Ah + row * LS + c4) = make_uint2(h0 | (h1 << 16), h2 | (h3 << 16));
    }

    // query chunks: BN rows x K halfs, 8 halfs per load; columns past T are zero
    constexpr int B_IT = BN * KMAX / 8 / MMB_NT;
    uint4 rb[B_IT];
    const int b_loads = BN * K8;
    auto gload = [&](const int t0) {
#pragma unroll
        for (int it = 0; it < B_IT; ++it) {
            const int idx = tid + it * MMB_NT;
            const int row = idx / K8, c8 = (idx % K8) * 8;
            rb[it] = (idx < b_loads && t0 + row < T) ? *(const uint4 *)(Xh + (size_t) (t0 + row) * K + c8) : make_uint4(0, 0, 0, 0);
        }
    };
    auto lstore = [&]() {
#pragma unroll
        for (int it = 0; it < B_IT; ++it) {
            const int idx = tid + it * MMB_NT;
            if (idx < b_loads) { const int row = idx / K8, c8 = (idx % K8) * 8; *(uint4 *)(Bh + row * LS + c8) = rb[it]; }
        }
    };

    const int r = lane & 15, cm = lane & 15, cn = lane >> 4;
    const int Q = T >> 2;
    float sm[TM];
#pragma unroll
    for (int i = 0; i < TM; ++i) { const int m = m0 + wm * WTM + i * 16 + cm; sm[i] = (tails && m < M) ? (float) starts[m] : 0.0f; }

    gload(0);
    for (int t0 = 0; t0 < T; t0 += BN) {
        __syncthreads();   // the previous chunk is consumed (and the keys are stored on the first pass)
        lstore();
        __syncthreads();
        if (t0 + BN < T) gload(t0 + BN);
        v8f acc[TM][TN];
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j)
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;
        for (int k0 = 0; k0 < K; k0 += 16) {
            v16s ah[TM], bh[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) { const int off = (wm * WTM + i * 16 + r) * LS + k0;
                ah[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Ah + off), *(const uint4 *)(Ah + off + 8)}); }
#pragma unroll
            for (int j = 0; j < TN; ++j) { const int off = (wn * WTN + j * 16 + r) * LS + k0;
                bh[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Bh + off), *(const uint4 *)(Bh + off + 8)}); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bh[j], ah[i], acc[i][j]);
        }
        // epilogue of mmb_f32split_kernel<..., EPI=1>
#pragma unroll
        for (int i = 0; i < TM; ++i) { const int m = m0 + wm * WTM + i * 16 + cm;
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                float o[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) o[e] = __shfl_xor(acc[i][j][e], 16, 32);
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float h0 = cn == 0 ? acc[i][j][2 * p]     : o[2 * p];
                    const float h1 = cn == 0 ? o[2 * p]             : acc[i][j][2 * p];
                    const float h2 = cn == 0 ? acc[i][j][2 * p + 1] : o[2 * p + 1];
                    const float h3 = cn == 0 ? o[2 * p + 1]         : acc[i][j][2 * p + 1];
                    float s = fmaxf(h0, 0.0f);
                    s = s + fmaxf(h1, 0.0f);
                    s = s + fmaxf(h2, 0.0f);
                    s = s + fmaxf(h3, 0.0f);
                    const int q = ((t0 + wn * WTN + j * 16) >> 2) + p;
                    if ((p >> 1) == cn && m < M && q < Q) {
                        if (tails) {
                            // log(step(d)) is exactly 0 or -inf, and s >= +0, so this is s + logf(step)
                            const float d = (float) tails[q] - sm[i];
                            s = d > 0.0f ? s + 0.0f : -INFINITY;
                        }
                        D[(size_t) q * M + m] = s;
                    }
                }
            }
        }
    }
#endif
}

// ---- HC down W8A8 (int8): Q8_0 weights x int8 activations, int8 WMMA, per-32-K-block scales ----
// The int8 WMMA (wmma_i32_16x16x16_iu8_w32) computes a whole 32-K block (one Q8_0 block, one scale) in two
// 16-K WMMA ops; the int32 partial is then scaled by (dw * dx) into the f32 accumulator. Layout of the iu8
// result: lane holds c[l] = (row 2l + (lane>>4), col lane&15), so the fork's mmb_store_tile staging maps the
// tile rows to the output rows unchanged. Activations come quantized in the [kb][token][32] layout the GEMM
// reads: 32 int8 per (32-K block, token) plus one f32 scale per (block, token).
__global__ void mmb_quantize_q8_act_kernel(const uint16_t * __restrict__ X, uint8_t * __restrict__ Xq8,
        float * __restrict__ Xs, const int K, const int T, const bool allow_vec = false) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int kb = blockIdx.y;
    if (allow_vec) {   // consecutive lanes take consecutive 32-K blocks of one token: coalesced row reads
        const int nkb = K / 32;
        const int g = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
        t = g / nkb; kb = g - t * nkb;
    }
    if (t >= T) return;
    const int base = kb * 32;
    float v[32]; float amax = 0.0f;
    // vec: 16-byte loads / stores (the scalar 16-bit loads and byte stores ran at ~60 GB/s); same values
    static_assert(sizeof(uint4) == 16, "uint4");
    const bool vec = allow_vec && (((uintptr_t) X | (uintptr_t) Xq8) & 15) == 0 && (K & 7) == 0;
    if (vec) {
        const uint4 * xp = (const uint4 *) (X + (size_t) t * K + base);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint4 u = xp[q];
            const uint32_t w[4] = { u.x, u.y, u.z, u.w };
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                v[q * 8 + 2 * h]     = __uint_as_float(w[h] << 16);
                v[q * 8 + 2 * h + 1] = __uint_as_float(w[h] & 0xffff0000u);
            }
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
    } else {
#pragma unroll
    // FIX (inference-path): X is the BF16 activation in [T][K] row-major (was read as f16 bits in a [K][T] layout)
    for (int i = 0; i < 32; ++i) { v[i] = __uint_as_float(((uint32_t) X[(size_t) t * K + base + i]) << 16); amax = fmaxf(amax, fabsf(v[i])); }
    }
    const float d = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    uint8_t * out = Xq8 + ((size_t) kb * T + t) * 32;
    if (vec) {
        uint32_t pk[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            pk[q] = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b) pk[q] |= ((uint32_t) (uint8_t) (int8_t) roundf(v[q * 4 + b] * id)) << (8 * b);
        }
        ((uint4 *) out)[0] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        ((uint4 *) out)[1] = make_uint4(pk[4], pk[5], pk[6], pk[7]);
    } else {
#pragma unroll
    for (int i = 0; i < 32; ++i) out[i] = (uint8_t)(int8_t) roundf(v[i] * id);
    }
    Xs[(size_t) kb * T + t] = d;
}

// HC down W8A8 GEMM: BM rows x BN tokens per block, BK 32-K blocks per stage, WM x WN waves. The iu8 result
// layout is c[l] = (row 2l + (lane>>4), col lane&15), i.e. (weight, token) with the operands (W, X); the
// staging below transposes it to the [token][weight] form mmb_store_tile's read-back expects.
typedef int mmb_vi4 __attribute__((ext_vector_type(4)));
typedef int mmb_vi8 __attribute__((ext_vector_type(8)));
struct mmb_tok_identity { int T; __device__ int operator()(int i) const { return i < T ? i : -1; } };
template <int BM, int BN, int BK, int WM, int WN, typename DRowFn>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_hcd_w8a8_kernel(const uint8_t * __restrict__ W, const uint8_t * __restrict__ Xq8, const float * __restrict__ Xs,
        float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, const int K, const int T, DRowFn drow,
        const int x_gs = -1, const bool row_major = false) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3)
    NO_DEVICE_CODE;
#else
    constexpr int ROW_TILES = BM / 16, TOK_TILES = BN / 16;
    constexpr int WROWS = ROW_TILES / WM, WTOKS = TOK_TILES / WN;
    static_assert(WM * WN == 8 && ROW_TILES % WM == 0 && TOK_TILES % WN == 0, "wave layout");
    __shared__ __align__(16) mmb_vi4 s_a[BK][ROW_TILES][32];
    __shared__ float s_dw[BK][ROW_TILES][16];
    __shared__ __align__(16) mmb_vi4 s_b[BK][TOK_TILES][32];
    __shared__ float s_dx[BK][TOK_TILES][16];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wave_row = wave / WN, wave_tok = wave % WN;
    const int sub_lane = lane & 15, half_id = lane >> 4;
    // row_major: grid (row blocks, token blocks), the row blocks of one activation tile back to back
    const int m0 = (row_major ? blockIdx.x : blockIdx.y) * BM, t0 = (row_major ? blockIdx.y : blockIdx.x) * BN;
    const int nkb = K / 32;
    const size_t wrow_bytes = (size_t) nkb * 34;
    constexpr int A_PER = (BM * BK + MMB_NT - 1) / MMB_NT;
    mmb_vi4 r_q0[A_PER], r_q1[A_PER];
    float r_dw[A_PER];
    const uint8_t * a_ptr[A_PER];
    bool a_live[A_PER];
#pragma unroll
    for (int p = 0; p < A_PER; ++p) {
        const int idx = p * MMB_NT + tid;
        const int row = idx / BK;
        const int r = m0 + row;
        a_live[p] = idx < BM * BK && r < M;
        a_ptr[p] = W + (size_t)(a_live[p] ? r : (M - 1)) * wrow_bytes;
    }
    const int b_ts = wave;   // one wave per token tile
    // activation layout: [kb][t][32] + scales [kb][t] (x_gs < 0), or token-grouped [t/G][kb][G][32] + [t/G][kb][G]
    // with G = 1 << x_gs (<= 16): the group block index of (t, kb) is ((t >> gs) * nkb + kb) * G + (t & (G-1))
    // a token past T (the tail of the last tile) reads token T-1: its row is never stored
    const int    b_t  = min(t0 + b_ts * 16 + sub_lane, T - 1);
    const size_t b_bi = x_gs < 0 ? (size_t) b_t : (((size_t) (b_t >> x_gs) * nkb) << x_gs) + (b_t & ((1 << x_gs) - 1));
    const size_t x_kb = x_gs < 0 ? (size_t) T : ((size_t) 1 << x_gs);   // block-index stride of one kb
    const size_t s_kb = x_kb;
    const uint8_t * b_base = Xq8 + b_bi * 32 + (size_t) half_id * 16;
    const float *   b_scale = Xs + b_bi;
    const bool b_live = t0 + b_ts * 16 < T;
    mmb_vi4 r_b[BK];
    float r_dx[BK];
    const auto fetch_stage = [&](const int kb0) {
#pragma unroll
        for (int p = 0; p < A_PER; ++p) {
            const int kb_in = (p * MMB_NT + tid) % BK;
            const int kb = kb0 + kb_in;
            const int kb_c = kb < nkb ? kb : nkb - 1;
            const uint8_t * blk = a_ptr[p] + (size_t) kb_c * 34;
            const bool live = a_live[p] && kb < nkb;
            __builtin_memcpy(&r_q0[p], blk + 2, 16);
            __builtin_memcpy(&r_q1[p], blk + 18, 16);
            r_dw[p] = live ? mmb_h2f(*(const uint16_t *) blk) : 0.0f;
        }
        if (b_live) {
#pragma unroll
            for (int i = 0; i < BK; ++i) {
                const int kb = kb0 + i;
                const int kb_c = kb < nkb ? kb : nkb - 1;
                r_b[i] = *(const mmb_vi4 *)(b_base + (size_t) kb_c * x_kb * 32);
                r_dx[i] = kb < nkb ? b_scale[(size_t) kb_c * s_kb] : 0.0f;
            }
        }
    };
    const auto commit_stage = [&]() {
#pragma unroll
        for (int p = 0; p < A_PER; ++p) {
            const int idx = p * MMB_NT + tid;
            if (idx >= BM * BK) continue;
            const int row = idx / BK, kk = idx % BK;
            const int rs = row / 16, rl = row % 16;
            s_a[kk][rs][rl] = r_q0[p];
            s_a[kk][rs][16 + rl] = r_q1[p];
            s_dw[kk][rs][(rl & 1) == 0 ? rl / 2 : 8 + rl / 2] = r_dw[p];
        }
        if (b_live) {
#pragma unroll
            for (int i = 0; i < BK; ++i) {
                s_b[i][b_ts][lane] = r_b[i];
                if (lane < 16) s_dx[i][b_ts][lane] = r_dx[i];
            }
        }
    };
    float acc[WROWS][WTOKS][8];
#pragma unroll
    for (int i = 0; i < WROWS; ++i)
#pragma unroll
        for (int j = 0; j < WTOKS; ++j)
#pragma unroll
            for (int l = 0; l < 8; ++l) acc[i][j][l] = 0.0f;
    // unroll the 32-K blocks of a stage only for short stages (BK <= 2)
    constexpr int KB_UNROLL = BK <= 2 ? BK : 1;
    fetch_stage(0);
    for (int kb0 = 0; kb0 < nkb; kb0 += BK) {
        commit_stage();
        __syncthreads();
        if (kb0 + BK < nkb) fetch_stage(kb0 + BK);
#pragma unroll KB_UNROLL
        for (int kb = 0; kb < BK; ++kb) {
            mmb_vi4 a0[WROWS], a1[WROWS];
            float dw[WROWS][8];
#pragma unroll
            for (int i = 0; i < WROWS; ++i) {
                const int rs = wave_row * WROWS + i;
                a0[i] = s_a[kb][rs][sub_lane];
                a1[i] = s_a[kb][rs][16 + sub_lane];
                const float4 lo = *(const float4 *)&s_dw[kb][rs][half_id * 8];
                const float4 up = *(const float4 *)&s_dw[kb][rs][half_id * 8 + 4];
                dw[i][0] = lo.x; dw[i][1] = lo.y; dw[i][2] = lo.z; dw[i][3] = lo.w;
                dw[i][4] = up.x; dw[i][5] = up.y; dw[i][6] = up.z; dw[i][7] = up.w;
            }
#pragma unroll
            for (int j = 0; j < WTOKS; ++j) {
                const int ts = wave_tok * WTOKS + j;
                const mmb_vi4 b0 = s_b[kb][ts][sub_lane];
                const mmb_vi4 b1 = s_b[kb][ts][16 + sub_lane];
                const float dx = s_dx[kb][ts][sub_lane];
#pragma unroll
                for (int i = 0; i < WROWS; ++i) {
                    mmb_vi8 c = {0, 0, 0, 0, 0, 0, 0, 0};
                    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0[i], true, b0, c, true);
                    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1[i], true, b1, c, true);
#pragma unroll
                    for (int l = 0; l < 8; ++l) acc[i][j][l] += (dw[i][l] * dx) * (float) c[l];
                }
            }
            __builtin_amdgcn_sched_barrier(0);
        }
        __syncthreads();
    }
    // epilogue: stage [token][weight] (the iu8 c[l] is (weight 2l+half, token sub_lane)), then the same
    // transposed read-back as mmb_store_tile: thread (n, half) writes 8 weights of token n.
    float * stg = (float *) s_b + wave * 256;
    const int cm = lane & 15, cn = lane >> 4;
#pragma unroll
    for (int i = 0; i < WROWS; ++i) {
        const int ml = (wave_row * WROWS + i) * 16;
        const bool ok = m0 + ml < M;
#pragma unroll
        for (int j = 0; j < WTOKS; ++j) {
            const int tl = (wave_tok * WTOKS + j) * 16;
            if (ok && t0 + tl < T) {
#pragma unroll
                for (int e = 0; e < 8; ++e) stg[cm * 16 + 2 * e + cn] = acc[i][j][e];
                __syncthreads();
                const int n = lane >> 1, half = lane & 1;
                const int dr = drow(t0 + tl + n);
                if (dr >= 0) {
                    const float4 v0 = *(const float4 *)(stg + n * 16 + half * 8), v1 = *(const float4 *)(stg + n * 16 + half * 8 + 4);
                    const size_t base = (size_t)dr * M + m0 + ml + half * 8;
                    const bool full = m0 + ml + 16 <= M;
                    if (store_f32) {
                        if (full && (M & 3) == 0) { *(float4 *)(D + base) = v0; *(float4 *)(D + base + 4) = v1; }
                        else { const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
                               for (int k = 0; k < 8; ++k) { if (m0 + ml + half * 8 + k < M) D[base + k] = vv[k]; } }
                    }
                    if (Dh) {
                        if (full && (M & 7) == 0) { *(uint4 *)(Dh + base) = make_uint4(mmb_pack2(v0.x, v0.y), mmb_pack2(v0.z, v0.w), mmb_pack2(v1.x, v1.y), mmb_pack2(v1.z, v1.w)); }
                        else { const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
                               for (int k = 0; k < 8; ++k) { if (m0 + ml + half * 8 + k < M) Dh[base + k] = mmb_f2bf(vv[k]); } }
                    }
                }
                __syncthreads();
            } else { __syncthreads(); __syncthreads(); }
        }
    }
#endif
}

// two tile classes: experts with >= thresh rows get BN_BIG-row tiles, the rest BN_SMALL-row tiles (fewer wasted rows on tiny experts)
__global__ void mmb_build_desc2(const int32_t * __restrict__ bounds, uint32_t * __restrict__ desc_big, uint32_t * __restrict__ desc_small,
        const int E, const int nbig_max, const int nsmall_max, const int BN_BIG, const int BN_SMALL, const int thresh) {
    __shared__ int sb[1024], ss[1024];
    const int e = threadIdx.x;
    for (int i = e; i < nbig_max;   i += blockDim.x) desc_big[i]   = UINT32_MAX;
    for (int i = e; i < nsmall_max; i += blockDim.x) desc_small[i] = UINT32_MAX;
    int cnt = (e < E) ? bounds[e + 1] - bounds[e] : 0;
    const bool big = cnt >= thresh;
    const int tb = big ? (cnt + BN_BIG - 1) / BN_BIG : 0;
    const int ts = big ? 0 : (cnt + BN_SMALL - 1) / BN_SMALL;
    sb[e] = tb; ss[e] = ts;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        const int vb = (e >= off) ? sb[e - off] : 0, vs = (e >= off) ? ss[e - off] : 0;
        __syncthreads();
        sb[e] += vb; ss[e] += vs;
        __syncthreads();
    }
    const int bb = sb[e] - tb, bs = ss[e] - ts;
    for (int jt = 0; jt < tb; ++jt) { const int idx = bb + jt; if (idx < nbig_max)   desc_big[idx]   = (uint32_t)e | ((uint32_t)jt << 16); }
    for (int jt = 0; jt < ts; ++jt) { const int idx = bs + jt; if (idx < nsmall_max) desc_small[idx] = (uint32_t)e | ((uint32_t)jt << 16); }
}

__global__ void mmb_build_desc3(const int32_t * __restrict__ bounds, uint32_t * __restrict__ desc_small,
        uint32_t * __restrict__ desc_mid, uint32_t * __restrict__ desc_big,
        const int E, const int nsmall_max, const int nmid_max, const int nbig_max, const int BN_BIG, const int BN_SMALL) {
    __shared__ int su[1024], sm[1024], sb[1024];
    const int e = threadIdx.x;
    for (int i = e; i < nbig_max;   i += blockDim.x) desc_big[i]   = UINT32_MAX;
    for (int i = e; i < nmid_max;   i += blockDim.x) desc_mid[i]   = UINT32_MAX;
    for (int i = e; i < nsmall_max; i += blockDim.x) desc_small[i] = UINT32_MAX;
    int cnt = (e < E) ? bounds[e + 1] - bounds[e] : 0;
    const int ts = cnt < 32 ? (cnt + BN_SMALL - 1) / BN_SMALL : 0;
    const int tm = (cnt >= 32 && cnt < 64) ? 1 : 0;
    const int tb = cnt >= 64 ? (cnt + BN_BIG - 1) / BN_BIG : 0;
    su[e] = ts; sm[e] = tm; sb[e] = tb;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        const int vu = (e >= off) ? su[e - off] : 0, vm = (e >= off) ? sm[e - off] : 0, vb = (e >= off) ? sb[e - off] : 0;
        __syncthreads();
        su[e] += vu; sm[e] += vm; sb[e] += vb;
        __syncthreads();
    }
    const int bu = su[e] - ts, bm = sm[e] - tm, bb = sb[e] - tb;
    for (int jt = 0; jt < ts; ++jt) { const int idx = bu + jt; if (idx < nsmall_max) desc_small[idx] = (uint32_t)e | ((uint32_t)jt << 16); }
    for (int jt = 0; jt < tm; ++jt) { const int idx = bm + jt; if (idx < nmid_max)   desc_mid[idx]   = (uint32_t)e | ((uint32_t)jt << 16); }
    for (int jt = 0; jt < tb; ++jt) { const int idx = bb + jt; if (idx < nbig_max)   desc_big[idx]   = (uint32_t)e | ((uint32_t)jt << 16); }
}

} // namespace

struct mmb_cache_entry { const ggml_tensor * root; const void * data; size_t n; bool f16; ggml_cuda_pool_alloc<uint16_t> * buf; };
// Routing maps of the last fused GLU, reused by the routed down that reads that GLU's output (same ids tensor).
struct mmb_ids_record {
    const void * ids = nullptr;
    const ggml_tensor * glu = nullptr;
    const void * glu_data = nullptr;
    int T = 0, n_used = 0, E = 0, si1 = 0;
    const int32_t * ids_dst = nullptr;
    const int32_t * bounds = nullptr;
};

struct ggml_cuda_mmb_context {
    struct workspace {
        std::vector<mmb_cache_entry> cache;
        mmb_cache_entry slots[5] = {};   // 4: F16 copy of the gate-mix output (dense F16)
        size_t slot_cap[5] = {};
    };
    workspace streams[GGML_CUDA_MAX_STREAMS];
    std::unordered_set<const ggml_tensor *> bf16_only;
    std::unordered_map<const ggml_tensor *, int> f16_marks;   // dense F16 reader marks (MMB_F16_WANT / MMB_F16_ONLY)
    std::unordered_map<const void *, uint16_t *> shadow;
    std::map<std::pair<const void *, const void *>, uint16_t *> shadow_pair;
    size_t shadow_bytes = 0;
    // int8 copy of the normalized streams (W8A8 HC down): written by the combine that produces xn, read by the
    // next HCD GEMM; keyed by the xn root tensor and its data pointer. A grown buffer is retired, not freed, until
    // release_all: captured graphs keep the old pointers baked into their kernel arguments.
    void *   xn_q8 = nullptr;
    void *   xn_q8_scale = nullptr;
    size_t   xn_q8_cap = 0;
    size_t   xn_q8_scale_cap = 0;
    const ggml_tensor * xn_q8_for = nullptr;
    const void * xn_q8_for_data = nullptr;
    std::vector<void *> xn_q8_retired;
    int      xn_q8_gs = -1;   // q8 layout: -1 = [kb][t][32] / [kb][t]; g >= 0: token groups of 1<<g ([t/G][kb][G][32]), hc_combine_norm_inject_v2
    // routed tier streams: the GLU pool buffers kept alive for the down, its routing maps, the pending side and the
    // fork/prep/join events
    std::vector<std::function<void()>> glu_hold;
    mmb_ids_record ids_rec;
    bool side_pending = false;
    cudaEvent_t fork_ev[2] = {};
    cudaEvent_t prep_ev = nullptr;
    cudaEvent_t join_ev = nullptr;
};

namespace {

static ggml_cuda_mmb_context & mmb_state(ggml_backend_cuda_context & ctx) {
    if (!ctx.mmb) {
        ctx.mmb = new ggml_cuda_mmb_context;
    }
    return *ctx.mmb;
}

static ggml_cuda_mmb_context::workspace & mmb_workspace(ggml_backend_cuda_context & ctx) {
    return mmb_state(ctx).streams[ctx.curr_stream_no];
}

static size_t mmb_cache_max() { return 4; }
static const ggml_tensor * mmb_root(const ggml_tensor * t) { return t->view_src ? t->view_src : t; }
static uint16_t * mmb_cache_insert(ggml_backend_cuda_context & ctx, const ggml_tensor * t, const size_t n, const bool f16 = false) {
    if (mmb_workspace(ctx).cache.size() >= mmb_cache_max()) { delete mmb_workspace(ctx).cache.front().buf; mmb_workspace(ctx).cache.erase(mmb_workspace(ctx).cache.begin()); }
    auto * buf = new ggml_cuda_pool_alloc<uint16_t>(ctx.pool(), n);
    mmb_workspace(ctx).cache.push_back({mmb_root(t), t->data, n, f16, buf});
    return buf->get();
}
// 16-bit copy of src1 (n elements) inside a slot or cache entry of its root: the entry itself, or a contiguous row range
// of it (a view at an offset, e.g. the last-token strip of a trimmed layer). The copy has the root's element layout.
static const uint16_t * mmb_copy_find(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const size_t n, const bool f16) {
    const ggml_tensor * root = mmb_root(src1);
    const bool sub = src1->view_src != nullptr && ggml_is_contiguous(src1) && ggml_is_contiguous(root) && root->type == GGML_TYPE_F32;
    auto at = [&](const void * e_data, const size_t e_n, const uint16_t * buf) -> const uint16_t * {
        if (e_data == src1->data && e_n == n) return buf;
        if (!sub || e_data != root->data) return nullptr;
        const size_t off = (size_t) ((const char *) src1->data - (const char *) e_data);
        if ((const char *) src1->data < (const char *) e_data || off % sizeof(float) != 0 || off / sizeof(float) + n > e_n) return nullptr;
        return buf + off / sizeof(float);
    };
    for (int si = 0; si < GGML_CUDA_MAX_STREAMS; ++si) {
        auto & work = mmb_state(ctx).streams[si];
        for (int k = 0; k < (int) std::size(work.slots); ++k) {
            auto & e = work.slots[k];
            if (!e.buf || e.f16 != f16 || e.root != root) continue;
            if (const uint16_t * p = at(e.data, e.n, e.buf->get())) { ggml_cuda_ovl_note_slot_read(ctx, si, k); return p; }
        }
    }
    for (auto & e : mmb_workspace(ctx).cache) {
        if (e.f16 != f16 || e.root != root) continue;
        if (const uint16_t * p = at(e.data, e.n, e.buf->get())) return p;
    }
    return nullptr;
}
static const uint16_t * mmb_bf16_activation(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const size_t n, cudaStream_t stream) {
    // Slot producers have graph dependencies; temporary conversions belong to their stream.
    if (const uint16_t * p = mmb_copy_find(ctx, src1, n, false)) return p;
    uint16_t * buf = mmb_cache_insert(ctx, src1, n);
    mmb_cvt_f32_bf16<<<(unsigned)((n / 8 + 255) / 256), 256, 0, stream>>>((const float *) src1->data, buf, n);
    return buf;
}

// F16 (RNE) copy of an F32 activation: the exact value mmb_f32split_kernel stores as the hi half, so the f16hi
// dense GEMM on this copy is bit-identical to the split kernel's hi*hi accumulation.
static const uint16_t * mmb_f16_activation(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const size_t n, cudaStream_t stream) {
    if (const uint16_t * p = mmb_copy_find(ctx, src1, n, true)) return p;
    uint16_t * buf = mmb_cache_insert(ctx, src1, n, true);
    const int64_t nb = (int64_t) n;
    mmb_f32_to_f16_kernel<<<(unsigned) ((nb + 255) / 256), 256, 0, stream>>>((const float *) src1->data, buf, nb);
    return buf;
}
// cached 16-bit copy of src1 written by its producer (f16: the dense F16 copy), nullptr if none: no conversion
static const uint16_t * mmb_act_lookup(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const size_t n, const bool f16) {
    return mmb_copy_find(ctx, src1, n, f16);
}

// Shadow BF16 copies of IQ4_NL dense weights: dequantised once (same LUT*scale -> BF16 RNE as mmb_dq_row36, so the
// WMMA inputs are bitwise identical) so the dense GEMM runs the dequant-free WTYPE=2 path.
__global__ void mmb_dq_q6k_bf16_kernel(const uint8_t * __restrict__ W, uint16_t * __restrict__ out, const size_t nblocks) {
    const size_t b = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const uint8_t * p  = W + b * 210;
    const uint8_t * ql = p, * qh = p + 128;
    const int8_t  * sc = (const int8_t *) (p + 192);
    const float d = mmb_h2f(*(const uint16_t *) (p + 208));
    uint16_t * o = out + b * 256;
    for (int n = 0; n < 2; ++n) {
        const uint8_t * QL = ql + 64 * n; const uint8_t * QH = qh + 32 * n; const int8_t * S = sc + 8 * n; uint16_t * Y = o + 128 * n;
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = (int8_t)((QL[l +  0] & 0xF) | (((QH[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((QL[l + 32] & 0xF) | (((QH[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((QL[l +  0] >>  4) | (((QH[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((QL[l + 32] >>  4) | (((QH[l] >> 6) & 3) << 4)) - 32;
            Y[l +  0] = mmb_f2bf(d * S[is + 0] * q1);
            Y[l + 32] = mmb_f2bf(d * S[is + 2] * q2);
            Y[l + 64] = mmb_f2bf(d * S[is + 4] * q3);
            Y[l + 96] = mmb_f2bf(d * S[is + 6] * q4);
        }
    }
}

__global__ void mmb_dq_q8_0_bf16_kernel(const uint8_t * __restrict__ W, uint16_t * __restrict__ out, const size_t nblocks) {
    const size_t b = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const uint8_t * p = W + b * 34;
    const float d = mmb_h2f((uint16_t)(p[0] | (p[1] << 8)));
    uint32_t * o = (uint32_t *) (out + b * 32);
#pragma unroll
    for (int j = 0; j < 16; ++j) o[j] = mmb_pack2(d * (float)(int8_t) p[2 + 2 * j], d * (float)(int8_t) p[3 + 2 * j]);
}

// F32 weights eligible for the f16-hi route (the M=512 MoE router [2560 -> 512]): (f16)W shadow, see the dispatch.
static bool mmb_is_f16hi_f32(const ggml_tensor * w) {
    return w && w->type == GGML_TYPE_F32 && w->op == GGML_OP_NONE && w->data && w->buffer &&
        w->ne[2] == 1 && w->ne[3] == 1 && ggml_is_contiguous(w) &&
        w->ne[0] >= 512 && w->ne[0] % 64 == 0 && w->ne[1] > 64 && w->ne[1] <= 1024;
}

// HC Q8_0 matrices (hc_*_down [10240 -> 320], hc_*_up [320 -> 10240]) get a BF16 shadow: their GEMMs re-decode the
// whole matrix per token tile. MMB_SHADOW_Q8=0 disables.
static bool mmb_is_hc_q8(const ggml_tensor * w) {
    static const bool on = getenv("MMB_SHADOW_Q8") ? atoi(getenv("MMB_SHADOW_Q8")) != 0 : true;
    // HC down [10240 -> 320] reads its weights once per token tile: Q8_0 (68 B / 64 weights) beats the BF16 shadow
    // (128 B) there, so only the HC up / gate weights get a shadow unless MMB_HCD_SHADOW=1
    static const bool down = getenv("MMB_HCD_SHADOW") ? atoi(getenv("MMB_HCD_SHADOW")) != 0 : false;
    if (!on || !w || w->type != GGML_TYPE_Q8_0 || w->op != GGML_OP_NONE || !w->data || !w->buffer || w->ne[2] != 1 || w->ne[3] != 1 || !ggml_is_contiguous(w)) return false;
    return (down && w->ne[0] >= 4096 && w->ne[1] <= 384) || (w->ne[0] <= 384 && w->ne[1] >= 4096);
}

__global__ void mmb_dq_iq4nl_bf16_kernel(const uint8_t * __restrict__ W, uint16_t * __restrict__ out, const size_t nblocks) {
    const size_t b = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const uint8_t * p = W + b * 18;
    const float d = mmb_h2f(*(const uint16_t *) p);
    const uint32_t * q = (const uint32_t *) (p + 2);
    uint32_t * o = (uint32_t *) (out + b * 32);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const uint32_t v = q[w];
        const float l0 = d * mmb_kv_iq4nl[(v      ) & 0xF], h0 = d * mmb_kv_iq4nl[(v >>  4) & 0xF];
        const float l1 = d * mmb_kv_iq4nl[(v >>  8) & 0xF], h1 = d * mmb_kv_iq4nl[(v >> 12) & 0xF];
        const float l2 = d * mmb_kv_iq4nl[(v >> 16) & 0xF], h2 = d * mmb_kv_iq4nl[(v >> 20) & 0xF];
        const float l3 = d * mmb_kv_iq4nl[(v >> 24) & 0xF], h3 = d * mmb_kv_iq4nl[(v >> 28) & 0xF];
        o[2*w] = mmb_pack2(l0, l1); o[2*w + 1] = mmb_pack2(l2, l3); o[8 + 2*w] = mmb_pack2(h0, h1); o[8 + 2*w + 1] = mmb_pack2(h2, h3);
    }
}
int    mmb_shadow_mode(){ return 2; }
bool   mmb_shadow()    { return mmb_shadow_mode() != 0; }
bool   mmb_shadow_q6k(){ return mmb_shadow_mode() >= 1; }
size_t mmb_shadow_cap(){ return (size_t) 6144 << 20; }
static bool mmb_is_resident_q6k(const ggml_tensor * w) { return w && w->type == GGML_TYPE_Q6_K && w->op == GGML_OP_NONE && w->data && w->buffer && w->ne[2] == 1 && w->ne[3] == 1 && ggml_is_contiguous(w) && w->ne[0] % 256 == 0 && w->ne[1] <= 32768; }
static bool mmb_is_resident_iq4(const ggml_tensor * w) { return w && w->type == GGML_TYPE_IQ4_NL && w->op == GGML_OP_NONE && w->data && w->buffer && w->ne[2] == 1 && w->ne[3] == 1 && ggml_is_contiguous(w); }
static bool mmb_is_row_concat(const ggml_tensor * w) {
    return w && w->op == GGML_OP_CONCAT && w->type == GGML_TYPE_IQ4_NL && ggml_get_op_params_i32(w, 0) == 1 && mmb_is_resident_iq4(w->src[0]) && mmb_is_resident_iq4(w->src[1]) &&
           w->src[0]->ne[0] == w->src[1]->ne[0] && w->ne[0] == w->src[0]->ne[0] && w->ne[1] == w->src[0]->ne[1] + w->src[1]->ne[1];
}
static const uint16_t * mmb_shadow_lookup(ggml_backend_cuda_context & ctx, const ggml_tensor * w) {
    if (w->op == GGML_OP_CONCAT) { auto it = mmb_state(ctx).shadow_pair.find({w->src[0]->data, w->src[1]->data}); return it == mmb_state(ctx).shadow_pair.end() ? nullptr : it->second; }
    auto it = mmb_state(ctx).shadow.find(w->data); return it == mmb_state(ctx).shadow.end() ? nullptr : it->second;
}

// RDNA3.5 (gfx1151) only, tuned for qwen4exp shapes. On gfx1151 (ROCm 7.2.1) it lost to MMQ on other archs: dense qwen35 prefill 3.4-3.9x slower, MoE 14-28% (PR #75).
// llama now opts in every backend context it creates, limited to the weight types in mmb_quant_type_dense() / mmb_quant_type_routed(); only the 32-row small-batch gate (mmb_min_t) depends on the arch.
bool mmb_enabled(const ggml_backend_cuda_context & ctx) {
    return ctx.mmb_opt_in && GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[ctx.device].cc);
}
// Smallest GEMM row count for the MMB consumers: 32 in a context with the small-batch opt-in (qwen4exp), else 512. The QSA indexer score keeps 512, see below.
// gfx1151 / ROCm 10, Qwen3.8-Flash-Next UD-Q4_K_XL and UD-IQ4_XS prefill: MMB is about 5% slower than MMQ at 16 tokens, 11-14% faster at 32.
// Retest when the MMQ or MMB tiles or the mmb_quant_type_*() gates change.
int  mmb_min_t(const ggml_backend_cuda_context & ctx) { return ctx.mmb_small_batch ? 32 : 512; }
int  mmb_f32split_mode(){ return 2; }
int  mmb_tall_mode(){ return 2; }
bool mmb_tall()    { return mmb_tall_mode() != 0; }
bool mmb_glu()     { static const bool v = getenv("MMB_GLU") ? atoi(getenv("MMB_GLU")) != 0 : true; return v; }

} // namespace

// Drops the cache entry whose buffer is p (handed out by ggml_cuda_mmb_cache_produce) when its producer did not
// run after all, so no reader finds a copy that was never written.
void ggml_cuda_mmb_cache_unproduce(ggml_backend_cuda_context & ctx, const uint16_t * p) {
    if (!p || !ctx.mmb) return;
    auto & cache = mmb_workspace(ctx).cache;
    for (auto it = cache.begin(); it != cache.end(); ++it) {
        if (it->buf && it->buf->get() == p) { delete it->buf; cache.erase(it); return; }
    }
}
const uint16_t * ggml_cuda_mmb_cache_lookup(ggml_backend_cuda_context & ctx, const ggml_tensor * t) {
    const ggml_tensor * root = mmb_root(t);
    for (int si = 0; si < GGML_CUDA_MAX_STREAMS; ++si) {
        auto & work = mmb_state(ctx).streams[si];
        for (int k = 0; k < (int) std::size(work.slots); ++k) { auto & e = work.slots[k]; if (e.buf && !e.f16 && e.root == root && e.data == t->data) { ggml_cuda_ovl_note_slot_read(ctx, si, k); return e.buf->get(); } }
    }
    for (auto & e : mmb_workspace(ctx).cache) if (!e.f16 && e.root == root && e.data == t->data) return e.buf->get();
    return nullptr;
}
uint16_t * ggml_cuda_mmb_slot_reserve(ggml_backend_cuda_context & ctx, int slot, const ggml_tensor * t, size_t n, bool f16) {
    ggml_cuda_ovl_slot_write(ctx, slot);
    mmb_cache_entry & e = mmb_workspace(ctx).slots[slot];
    if (e.buf && mmb_workspace(ctx).slot_cap[slot] < n) { delete e.buf; e.buf = nullptr; }
    if (!e.buf) { e.buf = new ggml_cuda_pool_alloc<uint16_t>(ctx.pool(), n); mmb_workspace(ctx).slot_cap[slot] = n; }
    e.root = mmb_root(t); e.data = t->data; e.n = n; e.f16 = f16;
    return e.buf->get();
}
void ggml_cuda_mmb_marks_clear(ggml_backend_cuda_context & ctx) { if (ctx.mmb) { ctx.mmb->bf16_only.clear(); ctx.mmb->f16_marks.clear(); } }
void ggml_cuda_mmb_mark_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * t, int bits) { mmb_state(ctx).f16_marks[t] |= bits; }
int ggml_cuda_mmb_f16_marks(ggml_backend_cuda_context & ctx, const ggml_tensor * t) {
    if (!ctx.mmb) return 0;
    auto it = ctx.mmb->f16_marks.find(t->view_src ? t->view_src : t);
    return it == ctx.mmb->f16_marks.end() ? 0 : it->second;
}
size_t ggml_cuda_mmb_marks_count(ggml_backend_cuda_context & ctx) { return ctx.mmb ? ctx.mmb->bf16_only.size() : 0; }
void ggml_cuda_mmb_mark_bf16_only(ggml_backend_cuda_context & ctx, const ggml_tensor * t) {
    // a graph output is read by the host after the graph (embeddings, MTP hidden state): it keeps its F32 store
    if (t->flags & GGML_TENSOR_FLAG_OUTPUT) return;
    mmb_state(ctx).bf16_only.insert(t);
}
bool ggml_cuda_mmb_is_bf16_only(ggml_backend_cuda_context & ctx, const ggml_tensor * t) { return ctx.mmb && ctx.mmb->bf16_only.count(t) > 0; }
void ggml_cuda_mmb_begin_graph(ggml_backend_cuda_context & ctx) {
    if (!ctx.mmb) return;
    // the int8 xn copy is keyed by data pointer only: an xn of this graph at the same address must not find the
    // previous graph's copy (the allocator reuses addresses across graphs)
    ctx.mmb->xn_q8_for = nullptr;
    ctx.mmb->xn_q8_for_data = nullptr;
    // holds left by a GLU whose down did not run through mmb (another kernel or backend)
    for (auto it = ctx.mmb->glu_hold.rbegin(); it != ctx.mmb->glu_hold.rend(); ++it) (*it)();
    ctx.mmb->glu_hold.clear();
    ctx.mmb->ids_rec = mmb_ids_record{};
    ctx.mmb->side_pending = false;
    for (auto & work : mmb_state(ctx).streams) {
        for (auto & e : work.cache) delete e.buf;
        work.cache.clear();
        for (auto & e : work.slots) { e.root = nullptr; e.data = nullptr; e.n = 0; }
    }
}
void ggml_cuda_mmb_release_all(ggml_backend_cuda_context & ctx) {
    if (!ctx.mmb) return;
    ggml_cuda_mmb_begin_graph(ctx);
    for (auto & work : ctx.mmb->streams) {
        for (auto & e : work.slots) delete e.buf;
    }
    for (auto & e : ctx.mmb->shadow) CUDA_CHECK(cudaFree(e.second));
    for (auto & e : ctx.mmb->shadow_pair) CUDA_CHECK(cudaFree(e.second));
    if (ctx.mmb->xn_q8) CUDA_CHECK(cudaFree(ctx.mmb->xn_q8));
    if (ctx.mmb->xn_q8_scale) CUDA_CHECK(cudaFree(ctx.mmb->xn_q8_scale));
    for (void * p : ctx.mmb->xn_q8_retired) CUDA_CHECK(cudaFree(p));
    for (cudaEvent_t e : { ctx.mmb->fork_ev[0], ctx.mmb->fork_ev[1], ctx.mmb->prep_ev, ctx.mmb->join_ev }) {
        if (e) CUDA_CHECK(cudaEventDestroy(e));
    }
    delete ctx.mmb;
    ctx.mmb = nullptr;
}
// A producer writes the BF16 copy of t itself: the entry is found by the MMB GEMM that reads t next.
uint16_t * ggml_cuda_mmb_cache_produce(ggml_backend_cuda_context & ctx, const ggml_tensor * t, size_t n, bool f16) {
    if (!mmb_enabled(ctx)) return nullptr;
    return mmb_cache_insert(ctx, t, n, f16);
}

// int8 copy of the normalized streams: the producing combine calls xnq8_buf (allocating/recording the key), the
// W8A8 HC down GEMM calls xnq8_get with its src1 root; a mismatch (single-block combine, different graph) means
// the buffer does not hold this xn and the GEMM falls back to its own quantization.
uint8_t * ggml_cuda_mmb_xnq8_buf(ggml_backend_cuda_context & ctx, const ggml_tensor * xn_root, size_t bytes, size_t scale_bytes) {
    ggml_cuda_mmb_context & st = mmb_state(ctx);
    if (st.xn_q8_cap < bytes) {
        if (st.xn_q8) st.xn_q8_retired.push_back(st.xn_q8);
        bytes = std::max(bytes, 2 * st.xn_q8_cap);   // geometric: the retired buffers stay bounded by the live one
        CUDA_CHECK(cudaMalloc(&st.xn_q8, bytes));
        st.xn_q8_cap = bytes;
    }
    if (st.xn_q8_scale_cap < scale_bytes) {
        if (st.xn_q8_scale) st.xn_q8_retired.push_back(st.xn_q8_scale);
        scale_bytes = std::max(scale_bytes, 2 * st.xn_q8_scale_cap);
        CUDA_CHECK(cudaMalloc(&st.xn_q8_scale, scale_bytes));
        st.xn_q8_scale_cap = scale_bytes;
    }
    st.xn_q8_for = xn_root;
    st.xn_q8_for_data = xn_root->data;
    st.xn_q8_gs = -1;
    return (uint8_t *) st.xn_q8;
}
void ggml_cuda_mmb_xnq8_set_layout(ggml_backend_cuda_context & ctx, const int gs) {
    mmb_state(ctx).xn_q8_gs = gs;
}
float * ggml_cuda_mmb_xnq8_scale(ggml_backend_cuda_context & ctx) {
    return (float *) mmb_state(ctx).xn_q8_scale;
}
const uint8_t * ggml_cuda_mmb_xnq8_get(ggml_backend_cuda_context & ctx, const ggml_tensor * xn_root, const float ** scale_out, int * gs_out) {
    ggml_cuda_mmb_context & st = mmb_state(ctx);
    if (!st.xn_q8 || st.xn_q8_for != xn_root || st.xn_q8_for_data != xn_root->data) return nullptr;
    *scale_out = (const float *) st.xn_q8_scale;
    if (gs_out) *gs_out = st.xn_q8_gs;
    return (const uint8_t *) st.xn_q8;
}
uint16_t * ggml_cuda_mmb_cache_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * t, size_t n) {
    if (!mmb_enabled(ctx) || ggml_nrows(t) < mmb_min_t(ctx)) return nullptr;
    return ggml_cuda_mmb_slot_reserve(ctx, 0, t, n);
}

#ifdef GGML_USE_TRELLIS
// Smallest token count mmb takes for a weight type. mmb_min_t(ctx) is where it beats MMQ; the trellis types have no MMQ
// (their alternative is dequant + BLAS, per expert and host-synchronised for MUL_MAT_ID), so they switch
// over right after MMVQ stops (MMVQ_MAX_BATCH_SIZE). GGML_CUDA_MMB_TQ_MIN_T overrides.
static int64_t mmb_min_t_for(const ggml_backend_cuda_context & ctx, ggml_type type) {
    if (ggml_cuda_type_is_tq(type)) {
        static const int v = getenv("GGML_CUDA_MMB_TQ_MIN_T") ? atoi(getenv("GGML_CUDA_MMB_TQ_MIN_T")) : 9;   // MMVQ_MAX_BATCH_SIZE + 1
        return v;
    }
    return mmb_min_t(ctx);
}
#endif // GGML_USE_TRELLIS

bool ggml_cuda_mmb_supported_mm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (!mmb_enabled(ctx)) return false;
    const bool quant = mmb_quant_type_dense(src0->type);
    const bool bf16w = src0->type == GGML_TYPE_BF16;
    const bool f32w  = src0->type == GGML_TYPE_F32;
    if (quant && src0->ne[0] % ggml_blck_size(src0->type) != 0) return false;
    if ((!quant && !bf16w && !f32w) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return false;
    const int64_t K = src0->ne[0], M = src0->ne[1];
    if ((f32w ? K % 32 : K % 64) != 0 || src1->ne[0] != K || dst->ne[0] != M) return false;
    const int64_t T = src1->ne[1] * src1->ne[2] * src1->ne[3];
    // graph-computed F32 src0 (not a weight), such as the QSA indexer score with heads x tokens GEMM rows: keep the 512-row gate.
    // At 8-31 tokens and 32K depth, MMB changed the scores with no measured speed gain.
    const bool graph_src0 = f32w && src0->op != GGML_OP_NONE;
#ifdef GGML_USE_TRELLIS
    if (T < (graph_src0 ? 512 : mmb_min_t_for(ctx, src0->type)) || T > INT32_MAX / 4) return false;
#else
    if (T < (graph_src0 ? 512 : mmb_min_t(ctx)) || T > INT32_MAX / 4) return false;
#endif
    return ggml_nrows(dst) == T;
}

// true when act is (or derives from) a gate/up view of one fused gate_up MUL_MAT_ID output (ne[0] == 2*n_ff). Split-
// tensor models build gate and up as their own mul_mat_id, so they are not matched and keep the GLU fusion.
static bool mmb_act_from_fused_gateup(const ggml_tensor * act, const int n_ff, const int depth = 0) {
    if (!act || depth > 3) return false;
    if (act->op == GGML_OP_VIEW) {
        const ggml_tensor * s = act->view_src;
        return s && s->op == GGML_OP_MUL_MAT_ID && s->ne[0] == 2 * n_ff;
    }
    if (act->op == GGML_OP_MUL_MAT_ID) return false;   // split path: gate / up are their own projections
    for (int i = 0; i < 4; ++i) {
        if (!act->src[i]) break;
        if (mmb_act_from_fused_gateup(act->src[i], n_ff, depth + 1)) return true;
    }
    return false;
}

bool ggml_cuda_mmb_supported_mmid(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst) {
    if (!mmb_enabled(ctx)) return false;
    // Fused gate_up is split into gate/up views, so the SwiGLU fusion never fires and MMB pays BF16 precision with no
    // fused-kernel payoff. gemma-4-26B-A4B / gfx1151 / ROCm 10.0: KL vs master 0.19-0.31, top-1 90-93% (split-tensor
    // models 0.002-0.065, 97-99%); pp4096 +0.9%. Keep on MMQ; Q8_0 / IQ4_NL already eligible, so exempt.
    // Trellis types have no MMQ, so here they fall to the per-expert dequant + BLAS loop (see mmb_tq_enabled).
    const char * nm = ggml_get_name(dst);
    if (nm && strncmp(nm, "ffn_moe_gate_up", 15) == 0 &&
            src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_IQ4_NL) return false;
    // down projection of a fused gate_up model: its activation is the SwiGLU of two gate_up views, so the whole
    // expert FFN stays on MMQ (same Q8_0 / IQ4_NL exemption).
    if (mmb_act_from_fused_gateup(src1, (int) src0->ne[0]) &&
            src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_IQ4_NL) return false;
    if (!mmb_quant_type_routed(src0->type) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return false;
    const int64_t K = src0->ne[0], M = src0->ne[1], E = src0->ne[2];
    if (src0->ne[3] != 1 || K % 64 != 0 || E < 1 || E > 1024) return false;
    if (K % ggml_blck_size(src0->type) != 0) return false;
    const int64_t n_used = ids->ne[0], T = ids->ne[1];
    if (src1->ne[0] != K || src1->ne[3] != 1 || src1->ne[2] != T) return false;
    if (src1->ne[1] != 1 && src1->ne[1] != n_used) return false;
    if (dst->ne[0] != M || dst->ne[1] != n_used || dst->ne[2] != T || dst->ne[3] != 1) return false;
    if (ids->nb[0] != sizeof(int32_t) || ids->ne[2] != 1 || ids->ne[3] != 1) return false;
#ifdef GGML_USE_TRELLIS
    if (T < mmb_min_t_for(ctx, src0->type) || n_used > 64 || (T * n_used) >> 16 >= 1024) return false;   // tile index must fit in 16 bits per expert
#else
    if (T < mmb_min_t(ctx) || n_used > 64 || (T * n_used) >> 16 >= 1024) return false;   // tile index must fit in 16 bits per expert
#endif
    return true;
}

// geometry ggml_cuda_mul_mat_mmb picks for an F32 weight with M > 64; the fused scorer only takes 0 and 5
static int mmb_f32_tile_for(int M) {
    static const int FT_env = getenv("MMB_F32_TILE") ? atoi(getenv("MMB_F32_TILE")) : -1;
    return FT_env >= 0 ? FT_env : (M <= 1024 ? 5 : 0);
}

bool ggml_cuda_mmb_idx_score_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * mm, int heads) {
    if (heads != 4 || src0->type != GGML_TYPE_F32 || !ggml_cuda_mmb_supported_mm(ctx, src0, src1, mm)) return false;
    const int64_t M = src0->ne[1], T = src1->ne[1] * src1->ne[2] * src1->ne[3];
    if (M <= 64 || T % 4 != 0 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    const int FT = mmb_f32_tile_for((int) M);
    return FT == 0 || FT == 5;
}

void ggml_cuda_mmb_idx_score(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int32_t * tails, const int32_t * starts) {
    cudaStream_t stream = ctx.stream();
    const int K = (int) src0->ne[0], M = (int) src0->ne[1];
    const int T = (int) (src1->ne[1] * src1->ne[2] * src1->ne[3]);
    const float * W = (const float *) src0->data, * X = (const float *) src1->data; float * D = (float *) dst->data;
    // keys-stationary scorer for the wide case (MMB_IDX_SCORE_V2=0 keeps the split kernel; same result)
    static const bool v2 = getenv("MMB_IDX_SCORE_V2") == nullptr || atoi(getenv("MMB_IDX_SCORE_V2")) != 0;
    if (v2 && mmb_f32_tile_for(M) == 0 && K <= 128 && K % 16 == 0) {
        ggml_cuda_pool_alloc<uint16_t> xh(ctx.pool(), (size_t) T * K);
        const int64_t n = (int64_t) T * K;
        mmb_f32_to_f16_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(X, xh.get(), n);
        // 64 key rows x 64 query columns per step (2 x 4 waves): fastest of the shapes tried on gfx1151 at 250k depth
        mmb_idx_score_kernel<64, 64, 2><<<(M + 63) / 64, MMB_NT, 0, stream>>>(W, xh.get(), D, M, K, T, tails, starts);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (mmb_f32_tile_for(M) == 5) {
        dim3 g((M + 31) / 32, (T + 63) / 64);
        mmb_f32split_kernel<32, 64, 16, 16, true, false, 1><<<g, MMB_NT, 0, stream>>>(W, X, D, M, K, T, nullptr, nullptr, 0, tails, starts);
    } else {
        dim3 g((M + 127) / 128, (T + 127) / 128);
        mmb_f32split_kernel<128, 128, 32, 64, true, false, 1><<<g, MMB_NT, 0, stream>>>(W, X, D, M, K, T, nullptr, nullptr, 0, tails, starts);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Dense R2: Q8_0 dense weights with a k7/k8-winning shape are repacked IN PLACE at graph_optimize to the
// tile layout of mmb_tq8_dense_kernel (no extra memory). Every reader of a repacked tensor must understand that layout:
// the MMB GEMM (any T, ggml_cuda_mul_mat_dq), the RDNA3.5 fused-quantize matvecs (decode, mmvq.cu: x_pk) and the
// grouped matvec. ggml-cuda.cu routes every MUL_MAT on a repacked weight there; the MMVQ/MMQ fusions stay off it.
//   K=2560, M>=8192 (M=10240, 12288): k7 128x256 (8 waves 64x64), PF=1
//   M=6144 K=2560 and K=6144 M=2560: k8 128x128 (waves 64x32), PF=2
// The small dense Q8_0 shapes (shared expert gate/up [2560 -> 640], down [640 -> 2560], attention K/V [2560 -> 512])
// are repacked to the k8 tile too, so they run on mmb_tq8_dense_kernel (F16) instead of mmb_dense_kernel<128,128>.
static bool mmb_tq8_shape(const int64_t M, const int64_t K) {
    if (M % 128 != 0 || K % 64 != 0 || M > 65536 || K > 16384) return false;
    if (M >= 512 && K >= 512 && (M < 2560 || K < 2560)) return (K / 64) % 2 == 0;   // k8 only
    if (M < 2560 || K < 2560) return false;   // not vocab-sized (tied embeddings)
    if (K <= 4096 && M >= 8192) return true;                          // k7
    if ((K / 64) % 2 != 0) return false;                              // k8: nks even
    return M >= 6144 || K > 4096;
}
static bool mmb_tq8_k7(const int M, const int K) { return K <= 4096 && M >= 8192; }
// Dense F16: the k7/k8 kernels read F16 activations (f16 dequant + f16 WMMA) instead of BF16. The producers of
// these GEMM inputs (HC gate mix, GDN gated output norm, gated attention output) write the F16 copy themselves
// (saturating RNE) when the graph marks say a reader wants it; any other input is converted here (F32 -> F16), or
// the BF16 kernel runs on a producer's BF16 copy.
static const uint16_t * mmb_f16sat_activation(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const size_t n, cudaStream_t stream) {
    uint16_t * buf = mmb_cache_insert(ctx, src1, n, true);
    mmb_cvt_f32_f16_sat<<<(unsigned)((n / 8 + 255) / 256), 256, 0, stream>>>((const float *) src1->data, buf, n);
    return buf;
}
// Layout record, process-wide like the routed R2 one: data pointer -> bytes for dense Q8_0 tensors rewritten in place.
static std::mutex g_dq_mutex;
// serializes the in-place repacks (check, rewrite, record) across backend contexts sharing the weights; the device is
// synchronized before a rewrite so no other context's kernel still reads the old layout
static std::mutex g_repack_mutex;
static std::map<const char *, size_t> g_dq_packed;
static std::atomic<int> g_dq_count{0};
bool ggml_cuda_mmb_dq_packed(const ggml_tensor * w) {
    if (g_dq_count.load(std::memory_order_relaxed) == 0 || !w || w->type != GGML_TYPE_Q8_0 || !w->data) return false;
    std::lock_guard<std::mutex> lock(g_dq_mutex);
    return g_dq_packed.count((const char *) w->data) > 0;
}
static void mmb_dq_forget(const void * base, const size_t size) {
    if (g_dq_count.load() == 0) return;
    std::lock_guard<std::mutex> lock(g_dq_mutex);
    const char * lo = (const char *) base, * hi = lo + size;
    for (auto it = g_dq_packed.begin(); it != g_dq_packed.end();) {
        if (it->first < hi && it->first + it->second > lo) {
            // the bytes outside the write would stay in the tile layout while the record says plain Q8_0
            if (it->first < lo || it->first + it->second > hi) {
                GGML_ABORT("a %zu-byte write covers part of a weight repacked in place (dense R2): partial writes to it are not supported", size);
            }
            it = g_dq_packed.erase(it);
        } else ++it;
    }
    g_dq_count.store((int) g_dq_packed.size());
}
static bool mmb_dq_overlaps(const void * base, const size_t size) {
    if (g_dq_count.load() == 0) return false;
    std::lock_guard<std::mutex> lock(g_dq_mutex);
    const char * lo = (const char *) base, * hi = lo + size;
    for (const auto & p : g_dq_packed) if (p.first < hi && p.first + p.second > lo) return true;
    return false;
}
static void mmb_tq8_launch(const uint8_t * wt, const uint16_t * xhp, float * D, uint16_t * Dh, const bool store_f32,
        const int M, const int K, const int T, const bool k7, const bool f16, cudaStream_t stream) {
    GGML_ASSERT(M % 128 == 0 && K % 64 == 0 && (k7 || (K / 64) % 2 == 0));
    // raster: GM row tiles per group (1 << 20 = plain M-fastest). gfx1151 pp2048, median us per call, M-fastest vs GM=1 /
    // GM=8: [K=6144, M=2560] 1770 vs 2157 / 1934, [K=2560, M=10240] 2661 vs - / 2861.
    const int GM = 1 << 20;
    if (f16) {
        if (k7) mmb_tq8_dense_kernel<128, 256, 64, 64, 1, 72, true><<<(M / 128) * ((T + 255) / 256), MMB_NT, 0, stream>>>(wt, xhp, D, Dh, store_f32, M, K, T, GM);
        else {
            // up to 1024 tokens: 128x64 tiles when the 128x128 grid has fewer than 160 blocks,
            // e.g. M=2560 at T=512 (80 blocks): +1.3% pp512 on gfx1151; 320 is worse at ub1024. Same per-output sums.
            if (T <= 1024 && (M / 128) * ((T + 127) / 128) < 160)
                mmb_tq8_dense_kernel<128, 64, 64, 16, 2, 72, true><<<(M / 128) * ((T + 63) / 64), MMB_NT, 0, stream>>>(wt, xhp, D, Dh, store_f32, M, K, T, GM);
            else
                mmb_tq8_dense_kernel<128, 128, 64, 32, 2, 72, true><<<(M / 128) * ((T + 127) / 128), MMB_NT, 0, stream>>>(wt, xhp, D, Dh, store_f32, M, K, T, GM);
        }
    } else {
        if (k7) mmb_tq8_dense_kernel<128, 256, 64, 64, 1><<<(M / 128) * ((T + 255) / 256), MMB_NT, 0, stream>>>(wt, xhp, D, Dh, store_f32, M, K, T, GM);
        else    mmb_tq8_dense_kernel<128, 128, 64, 32, 2><<<(M / 128) * ((T + 127) / 128), MMB_NT, 0, stream>>>(wt, xhp, D, Dh, store_f32, M, K, T, GM);
    }
    CUDA_CHECK(cudaGetLastError());
}
// Any-T GEMM on a repacked weight outside the MMB batch range (2..mmb_min_t-1 rows, or MMB off for this context):
// private f16 activation copy, F32 output, k8 tile (one token tile up to 128 rows: the weight is streamed once).
void ggml_cuda_mul_mat_dq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_mmb_dq_packed(src0) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src1) && ggml_is_contiguous(dst) && src1->ne[0] == src0->ne[0] && dst->ne[0] == src0->ne[1]);
    const int K = (int) src0->ne[0], M = (int) src0->ne[1];
    const int64_t T64 = ggml_nrows(src1);
    GGML_ASSERT(T64 <= INT32_MAX / 4 && ggml_nrows(dst) == T64);
    const int T = (int) T64;
    cudaStream_t stream = ctx.stream();
    const size_t n = (size_t) T * K;
    ggml_cuda_pool_alloc<uint16_t> xh(ctx.pool(), n);
    mmb_cvt_f32_f16_sat<<<(unsigned)((n / 8 + 255) / 256), 256, 0, stream>>>((const float *) src1->data, xh.get(), n);
    mmb_tq8_launch((const uint8_t *) src0->data, xh.get(), (float *) dst->data, nullptr, true, M, K, T, mmb_tq8_k7(M, K), true, stream);
}
int ggml_cuda_mmb_f16_reader(ggml_backend_cuda_context & ctx, const ggml_tensor * n) {
    if (!n || n->op != GGML_OP_MUL_MAT || !n->src[0] || !n->src[1]) return 0;
    const ggml_tensor * w = n->src[0], * x = n->src[1];
    if (x->view_src && x->view_offs != 0) return 0;   // the copies are keyed by root + data pointer
    if (ggml_cuda_mmb_dq_packed(w)) return ggml_cuda_mmb_supported_mm(ctx, w, x, n) ? 1 : 0;
    // the f16hi F32-weight GEMM (MoE router) converts its own F32 -> F16 copy: gfx1151 pp2048, the router conversions a
    // gate-mix F16 copy would save (47 x ~160 us) are eaten by the second 16-bit store in those gate mixes (no net gain)
    return 0;
}
void ggml_cuda_mul_mat_mmb(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();
    const int K = (int) src0->ne[0], M = (int) src0->ne[1];
    const int T = (int) (src1->ne[1] * src1->ne[2] * src1->ne[3]);
    if (src0->type == GGML_TYPE_F32) {
        const float * W = (const float *) src0->data, * X = (const float *) src1->data; float * D = (float *) dst->data;
        // MMB_F32_TILE narrows the M>64 F32 GEMM (e.g. the M=512 MoE router) for more blocks per ubatch chunk.
        // 0 keeps master's 128x128 tile. Valid geometries satisfy (8/WAVES_M)*WTN == BN (the kernel's static_assert).
        // Default: 5 (32x64) for M <= 1024 (the M=512 router: 87 -> 77 ms per pp4096 on gfx1151, about -0.4% GPU time),
        // 128x128 above, where the narrow tile would reload the activations ~3x more per output. Every geometry keeps the
        // per-output K order, so the output is the same.
        static const int FT_env = getenv("MMB_F32_TILE") ? atoi(getenv("MMB_F32_TILE")) : -1;
        const int FT = FT_env >= 0 ? FT_env : (M <= 1024 ? 5 : 0);
        // MMB_F32_SMALL: M <= 64 (ssm_alpha / ssm_beta [2560 -> 48]) launches only T/128 blocks at 64x128 (32 at 4096 tokens
        // on 40 CUs); narrower token tiles give 2-4x the blocks with the same per-output K order (bit-identical)
        static const int FS = getenv("MMB_F32_SMALL") ? atoi(getenv("MMB_F32_SMALL")) : 2;
        // f16-hi route: a shadow holds (f16)W and an f16 copy of X holds (f16)X, so the WMMA f16 GEMM replays the
        // split kernel's hi*hi accumulation bit for bit (same K order) without the per-tile F32 split or the F32
        // re-reads. Geometry <64,128,32,32>: 8 waves, TM=2, TN=2.
        const uint16_t * w16 = (M > 64 && M <= 1024 && K % 64 == 0 && T >= 512) ? mmb_shadow_lookup(ctx, src0) : nullptr;
        if (w16) {
            const uint16_t * x16 = mmb_f16_activation(ctx, src1, (size_t) T * K, stream);
            // 128x128 tile for M >= 256, else 64x128 (after asf0 fc94088ee's taller F32 router tile). Same per-output K
            // order for every tile (bit-identical). gfx1151, qwen3.8-flash-next router (M=512, K=2560), us per GEMM
            // 64x128 -> 128x128: T=512 70 -> 63, T=1024 130 -> 97, T=2048 270 -> 207 (256x64: 218, 128x64: 233).
            if (M >= 256) {
                dim3 g((M + 127) / 128, (T + 127) / 128);
                mmb_dense_kernel<128, 128, 32, 64, 2, true><<<g, MMB_NT, 0, stream>>>((const uint8_t *) w16, x16, D, (uint16_t *) nullptr, true, M, K, T);
            } else {
            dim3 grid((M + 63) / 64, (T + 127) / 128);
            mmb_dense_kernel<64, 128, 32, 32, 2, true><<<grid, MMB_NT, 0, stream>>>((const uint8_t *) w16, x16, D, (uint16_t *) nullptr, true, M, K, T);
            }
            CUDA_CHECK(cudaGetLastError()); return;
        }
        if (M <= 64 && FS == 2) {
            dim3 grid((M + 63) / 64, (T + 31) / 32);
            mmb_f32split_kernel<64, 32, 16, 16, true, false><<<grid, MMB_NT, 0, stream>>>(W, X, D, M, K, T);
        } else if (M <= 64 && FS == 1) {
            dim3 grid((M + 63) / 64, (T + 63) / 64);
            mmb_f32split_kernel<64, 64, 16, 32, true, false><<<grid, MMB_NT, 0, stream>>>(W, X, D, M, K, T);
        } else if (M <= 64) {
            dim3 grid((M + 63) / 64, (T + 127) / 128);
            mmb_f32split_kernel<64, 128, 16, 64, true, false><<<grid, MMB_NT, 0, stream>>>(W, X, D, M, K, T);
        } else if (FT == 1) { dim3 g((M + 63) / 64,  (T + 127) / 128); mmb_f32split_kernel< 64, 128, 32, 32, true, false><<<g, MMB_NT, 0, stream>>>(W, X, D, M, K, T); }
        else if (FT == 4)   { dim3 g((M + 31) / 32,  (T + 127) / 128); mmb_f32split_kernel< 32, 128, 32, 16, true, false><<<g, MMB_NT, 0, stream>>>(W, X, D, M, K, T); }
        else if (FT == 5)   { dim3 g((M + 31) / 32,  (T + 63) / 64);   mmb_f32split_kernel< 32,  64, 16, 16, true, false><<<g, MMB_NT, 0, stream>>>(W, X, D, M, K, T); }
        else if (FT == 6)   { dim3 g((M + 63) / 64,  (T + 63) / 64);   mmb_f32split_kernel< 64,  64, 32, 16, true, false><<<g, MMB_NT, 0, stream>>>(W, X, D, M, K, T); }
        // 7/8: from asf0/strix-llama.cpp fc94088ee (256x64: a block holds up to 256 experts, the activations are read once
        // per 256 rows). Only reached outside the f16hi gate.
        else if (FT == 7)   { dim3 g((M + 255) / 256,(T + 63) / 64);   mmb_f32split_kernel<256,  64, 64, 32, true, false><<<g, MMB_NT, 0, stream>>>(W, X, D, M, K, T); }
        else if (FT == 8)   { dim3 g((M + 127) / 128,(T + 63) / 64);   mmb_f32split_kernel<128,  64, 32, 32, true, false><<<g, MMB_NT, 0, stream>>>(W, X, D, M, K, T); }
        else                { dim3 g((M + 127) / 128,(T + 127) / 128); mmb_f32split_kernel<128, 128, 32, 64, true, false><<<g, MMB_NT, 0, stream>>>(W, X, D, M, K, T);
        }
        CUDA_CHECK(cudaGetLastError()); return;
    }
    // dense F16 on a repacked weight: the producer's F16 copy, else its BF16 copy (BF16 kernel), else convert to F16
    const uint16_t * xf16 = nullptr;
    if (ggml_cuda_mmb_dq_packed(src0)) {
        const size_t n = (size_t) T * K;
        xf16 = mmb_act_lookup(ctx, src1, n, true);
        if (!xf16 && !mmb_act_lookup(ctx, src1, n, false)) xf16 = mmb_f16sat_activation(ctx, src1, n, stream);
    }
    const uint16_t * xhp = xf16 ? xf16 : mmb_bf16_activation(ctx, src1, (size_t) T * K, stream);
    const uint8_t * W = (const uint8_t *) src0->data; float * D = (float *) dst->data;
    if (mmb_tall() && src0->type == GGML_TYPE_IQ4_NL && M <= 384 && K >= 4096 && T >= 2048) {   // tall-M tile: HC down|inject [10240 -> 324], activations read once
        static const int wide = mmb_tall_mode() >= 2;
        if (wide) { dim3 grid(1, (T + 63) / 64); mmb_dense_kernel<384, 64, 96, 32, 0><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, (uint16_t *) nullptr, true, M, K, T); }
        else      { dim3 grid(1, (T + 31) / 32); mmb_dense_kernel<384, 32, 96, 16, 0><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, (uint16_t *) nullptr, true, M, K, T); }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Q8_0 HC down [10240 -> 320]: at 128x128 tiles only 3x(T/128) blocks with a K=10240 loop (occupancy-bound).
    // Narrower tiles keep the per-output K accumulation order (bit-identical) but launch 3-10x more blocks.
    // MMB_HCD_TILE: 0 = generic path, 1 <64,128,32,32>, 2 <32,128,32,16>, 3 <64,64,32,16>, 4 <32,64,16,16>, 5 <64,32,16,16>
    const uint16_t * hc_shadow = mmb_is_hc_q8(src0) ? mmb_shadow_lookup(ctx, src0) : nullptr;
    static const int HCD = getenv("MMB_HCD_TILE") ? atoi(getenv("MMB_HCD_TILE")) : 3;
    if (HCD && src0->type == GGML_TYPE_Q8_0 && M <= 384 && K >= 4096 && T >= 512) {
        const bool hbf = ggml_cuda_mmb_blk16() && ggml_cuda_mmb_is_bf16_only(ctx, dst) && (M & 7) == 0;
        uint16_t * Dh2 = hbf ? (uint16_t *) dst->data : nullptr; const bool sf = !hbf;
        // W8A8 int8 route: Q8_0 weights read as int8, activations quantized per 32-K block into the [kb][t][32]
        // layout; int8 WMMA with the per-block scale in the f32 accumulator. Other shapes take the bf16 tiles.
        if ((M % 64) == 0 && (K % 128) == 0) {
            const float * xsc = nullptr;
            int xtm = -1;
            // the int8 copy covers the whole xn root: only a reader of all of it may use it
            const uint8_t * xq = src1->data == mmb_root(src1)->data && ggml_nelements(src1) == ggml_nelements(mmb_root(src1)) ?
                ggml_cuda_mmb_xnq8_get(ctx, mmb_root(src1), &xsc, &xtm) : nullptr;
            bool fused = xq != nullptr;
            ggml_cuda_pool_alloc<uint8_t> xq_buf(ctx.pool());
            ggml_cuda_pool_alloc<float> xsc_buf(ctx.pool());
            if (!fused) {   // no fused int8 copy (single-block combine / first graph): quantize here
                const int nkb = K / 32;
                xq_buf.alloc((size_t) nkb * T * 32);
                xsc_buf.alloc((size_t) nkb * T);
                dim3 gq((T + 255) / 256, nkb);
                mmb_quantize_q8_act_kernel<<<gq, 256, 0, stream>>>(xhp, xq_buf.get(), xsc_buf.get(), K, T, true);
                xq = xq_buf.get(); xsc = xsc_buf.get();
            }
            // row blocks fastest, so the row blocks of an activation tile share it in L2.
            // 32-row tile up to 1024 tokens: M=320 at T=512 gives only 20 blocks of 64 rows for 40 CUs; 32-row blocks
            // are 0.4-0.7% faster end to end at ub 512/1024 (same sums)
            if (T <= 1024 && M % 32 == 0) {
                mmb_hcd_w8a8_kernel<32, 128, 4, 2, 4><<<dim3(M / 32, (T + 127) / 128), MMB_NT, 0, stream>>>(W, xq, xsc, D, Dh2, sf, M, K, T,
                    mmb_tok_identity{T}, fused ? xtm : -1, true);
            } else
            mmb_hcd_w8a8_kernel<64, 128, 4, 4, 2><<<dim3(M / 64, (T + 127) / 128), MMB_NT, 0, stream>>>(W, xq, xsc, D, Dh2, sf, M, K, T,
                mmb_tok_identity{T}, fused ? xtm : -1, true);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        auto go = [&](auto tag, const uint8_t * WP) {
            constexpr int WT = decltype(tag)::value;
            if      (HCD == 1) { dim3 g((M + 63) / 64, (T + 127) / 128); mmb_dense_kernel<64, 128, 32, 32, WT><<<g, MMB_NT, 0, stream>>>(WP, xhp, D, Dh2, sf, M, K, T); }
            else if (HCD == 2) { dim3 g((M + 31) / 32, (T + 127) / 128); mmb_dense_kernel<32, 128, 32, 16, WT><<<g, MMB_NT, 0, stream>>>(WP, xhp, D, Dh2, sf, M, K, T); }
            else if (HCD == 3) { dim3 g((M + 63) / 64, (T + 63) / 64);   mmb_dense_kernel<64, 64, 32, 16, WT><<<g, MMB_NT, 0, stream>>>(WP, xhp, D, Dh2, sf, M, K, T); }
            else if (HCD == 4) { dim3 g((M + 31) / 32, (T + 63) / 64);   mmb_dense_kernel<32, 64, 16, 16, WT><<<g, MMB_NT, 0, stream>>>(WP, xhp, D, Dh2, sf, M, K, T); }
            else if (HCD == 6) { dim3 g(1, (T + 31) / 32);               mmb_dense_kernel<384, 32, 96, 16, WT><<<g, MMB_NT, 0, stream>>>(WP, xhp, D, Dh2, sf, M, K, T); }
            else if (HCD == 7) { dim3 g((M + 63) / 64, (T + 127) / 128); mmb_dense_kernel<64, 128, 32, 32, WT><<<g, MMB_NT, 0, stream>>>(WP, xhp, D, Dh2, sf, M, K, T); }
            else if (HCD == 8) { dim3 g((M + 127) / 128, (T + 63) / 64); mmb_dense_kernel<128, 64, 64, 16, WT><<<g, MMB_NT, 0, stream>>>(WP, xhp, D, Dh2, sf, M, K, T); }
            else if (HCD == 9) { dim3 g((M + 127) / 128, (T + 127) / 128); mmb_dense_kernel<128, 128, 64, 32, WT><<<g, MMB_NT, 0, stream>>>(WP, xhp, D, Dh2, sf, M, K, T); }
            else if (HCD == 10) { dim3 g((M + 255) / 256, (T + 31) / 32); mmb_dense_kernel<256, 32, 64, 16, WT><<<g, MMB_NT, 0, stream>>>(WP, xhp, D, Dh2, sf, M, K, T); }
            else if (HCD == 11) { dim3 g((M + 255) / 256, (T + 63) / 64); mmb_dense_kernel<256, 64, 64, 32, WT><<<g, MMB_NT, 0, stream>>>(WP, xhp, D, Dh2, sf, M, K, T); }
            else               { dim3 g((M + 63) / 64, (T + 31) / 32);   mmb_dense_kernel<64, 32, 16, 16, WT><<<g, MMB_NT, 0, stream>>>(WP, xhp, D, Dh2, sf, M, K, T); }
        };
        if (hc_shadow) go(std::integral_constant<int, 2>{}, (const uint8_t *) hc_shadow);
        else           go(std::integral_constant<int, 1>{}, W);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const uint16_t * shadow_pre = ((src0->type == GGML_TYPE_IQ4_NL && mmb_shadow()) || src0->type == GGML_TYPE_Q6_K) ? mmb_shadow_lookup(ctx, src0) : nullptr;
    const bool big = (M >= 6144 && K >= 2560) || (shadow_pre && K >= 2560 && T >= 4096);
    uint16_t * Dh = (K == 320 && M == 10240) ? ggml_cuda_mmb_slot_reserve(ctx, 1, dst, (size_t) T * M) : nullptr;
    bool store_f32 = !(Dh && ggml_cuda_mmb_is_bf16_only(ctx, dst));
    if (ggml_cuda_mmb_blk16() && !Dh && ggml_cuda_mmb_is_bf16_only(ctx, dst) && (M & 7) == 0) {
        Dh = (uint16_t *) dst->data; store_f32 = false;
    }
    if (ggml_cuda_mmb_dq_packed(src0)) {   // dense R2 in-place tile layout: only mmb_tq8_dense_kernel reads it here
        mmb_tq8_launch(W, xhp, D, Dh, store_f32, M, K, T, mmb_tq8_k7(M, K), xf16 != nullptr, stream);
        return;
    }
    dim3 grid((M + 63) / 64, big ? (T + 127) / 128 : (T + 127) / 128);
    // MMB_DENSE_TILE narrows the shared-expert / dense GEMM tiles for more blocks per ubatch chunk.
    // The big shapes take the 128x256 tile (LDS 54 KB, VGPR 256, one CTA/SM).
#define MMB_SMALL_DENSE(WT, WP) do { \
    static const int DT_ = getenv("MMB_DENSE_TILE") ? atoi(getenv("MMB_DENSE_TILE")) : 0; \
    if (DT_ == 1) { dim3 gd((M + 63) / 64, (T + 127) / 128); mmb_dense_kernel<64, 128, 32, 32, WT><<<gd, MMB_NT, 0, stream>>>(WP, xhp, D, Dh, store_f32, M, K, T); } \
    else if (DT_ == 2) { dim3 gd((M + 31) / 32, (T + 127) / 128); mmb_dense_kernel<32, 128, 32, 16, WT><<<gd, MMB_NT, 0, stream>>>(WP, xhp, D, Dh, store_f32, M, K, T); } \
    else if (DT_ == 3) { dim3 gd((M + 63) / 64, (T + 63) / 64); mmb_dense_kernel<64, 64, 32, 16, WT><<<gd, MMB_NT, 0, stream>>>(WP, xhp, D, Dh, store_f32, M, K, T); } \
    else { dim3 gd((M + 127) / 128, (T + 127) / 128); mmb_dense_kernel<128, 128, 32, 64, WT><<<gd, MMB_NT, 0, stream>>>(WP, xhp, D, Dh, store_f32, M, K, T); } \
} while (0)
#define MMB_BIG_DENSE(WT, WP) do { \
    dim3 gd((M + 127) / 128, (T + 255) / 256); mmb_dense_kernel<128, 256, 64, 64, WT><<<gd, MMB_NT, 0, stream>>>(WP, xhp, D, Dh, store_f32, M, K, T); \
} while (0)

    const uint16_t * shadow = shadow_pre;
    if (shadow) {
        if (big) MMB_BIG_DENSE(2, (const uint8_t *) shadow);
        else     { MMB_SMALL_DENSE(2, (const uint8_t *) shadow); }
    } else if (src0->type == GGML_TYPE_IQ4_NL) {
        if (big) MMB_BIG_DENSE(0, W);
        else     { MMB_SMALL_DENSE(0, W); }
    } else if (src0->type == GGML_TYPE_Q8_0) {
        if (big) MMB_BIG_DENSE(1, W);
        else     { MMB_SMALL_DENSE(1, W); }
    } else if (mmb_quant_type_dense(src0->type)) {
        mmb_dispatch_quant(src0->type, [&](auto tag) {
            constexpr int WT = decltype(tag)::value;
            if constexpr (WT < 32 || mmb_quant_type_dense_supported(static_cast<ggml_type>(WT - 32))) {
                if (big) MMB_BIG_DENSE(WT, W);
                else     { MMB_SMALL_DENSE(WT, W); }
            }
        });
    } else {
        if (big) MMB_BIG_DENSE(2, W);
        else     { MMB_SMALL_DENSE(2, W); }
    }
    CUDA_CHECK(cudaGetLastError());
}

// two F32-weight GEMMs [K -> M1] and [K -> M2] (M1 + M2 <= 128) on the same F32 activations, one pass over X
bool ggml_cuda_mmb_f32_dual(ggml_backend_cuda_context & ctx, const ggml_tensor * w1, const ggml_tensor * w2, const ggml_tensor * x,
        ggml_tensor * d1, ggml_tensor * d2) {
    const int K = (int) w1->ne[0], M1 = (int) w1->ne[1], M2 = (int) w2->ne[1];
    const int T = (int) (x->ne[1] * x->ne[2] * x->ne[3]);
    if (w1->type != GGML_TYPE_F32 || w2->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || w2->ne[0] != K || x->ne[0] != K ||
            M1 + M2 > 128 || M1 % 16 != 0 || K % 32 != 0 || !ggml_is_contiguous(w1) || !ggml_is_contiguous(w2) || !ggml_is_contiguous(x) ||
            !ggml_is_contiguous(d1) || !ggml_is_contiguous(d2) || ggml_nelements(d1) != (int64_t) M1 * T || ggml_nelements(d2) != (int64_t) M2 * T) {
        return false;
    }
    const int M = M1 + M2;
    dim3 grid((M + 63) / 64, (T + 31) / 32);
    mmb_f32split_kernel<64, 32, 16, 16, true, false><<<grid, MMB_NT, 0, ctx.stream()>>>((const float *) w1->data, (const float *) x->data,
        (float *) d1->data, M, K, T, (const float *) w2->data, (float *) d2->data, M1);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_mmb_gatemix() { return true; }
bool ggml_cuda_mmb_down16() { return true; }
bool ggml_cuda_mmb_blk16() { return true; }
bool ggml_cuda_mmb_res16()  { return true; }
bool ggml_cuda_hc_gate_mix_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * lo, const ggml_tensor * xn, const ggml_tensor * dst, const int hc) {
    if (ggml_cuda_mmb_dq_packed(w)) return false;   // the kernel decodes the row-major Q8_0 layout, not the dense R2 tiles
    if (hc != 4 || !mmb_quant_type_dense(w->type) || lo->type != GGML_TYPE_F32 || !ggml_is_contiguous(lo) || !ggml_is_contiguous(dst)) return false;
    // the fused kernel is instantiated for IQ4_NL and the BF16 shadow of a Q8_0 gate only: the direct Q8_0 and trellis
    // decodes spill on gfx1151 (CI VGPR check), so those gates take the unfused GEMM + mix. A shadow is never dropped,
    // so a graph planned with the fused route also runs it.
    if (w->type != GGML_TYPE_IQ4_NL && !(mmb_is_hc_q8(w) && mmb_shadow_lookup(ctx, w))) return false;
    const int K = (int) w->ne[0], M = (int) w->ne[1], E = (int) dst->ne[0]; const int T = (int) ggml_nrows(dst);
    if (K % ggml_blck_size(w->type) != 0) return false;
    return !(K % MMB_BK != 0 || M != hc * E || E % 32 != 0 || lo->ne[0] != K || ggml_nrows(lo) != T || xn->ne[0] != M || ggml_nrows(xn) != T || T < mmb_min_t(ctx));
}
bool ggml_cuda_hc_gate_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * lo, const ggml_tensor * xn, ggml_tensor * dst,
        const int hc, const float scale, const float bias) {
    if (!ggml_cuda_hc_gate_mix_supported(ctx, w, lo, xn, dst, hc)) return false;
    const int K = (int) w->ne[0], E = (int) dst->ne[0]; const int T = (int) ggml_nrows(dst);
    const uint16_t * xn16 = ggml_cuda_mmb_cache_lookup(ctx, xn);
    if (!xn16) return false;
    cudaStream_t stream = ctx.stream();
    const uint16_t * lo16 = mmb_bf16_activation(ctx, lo, (size_t) T * K, stream);
    // dense F16: an F16 copy (slot 4) for the dense-r2 readers; the BF16 copy is skipped when only
    // dense-r2 GEMMs read the output
    const int fm = ggml_cuda_mmb_f16_marks(ctx, dst);
    uint16_t * outf = (fm & MMB_F16_WANT) ? ggml_cuda_mmb_slot_reserve(ctx, 4, dst, (size_t) T * E, true) : nullptr;
    uint16_t * outh = (outf && (fm & MMB_F16_ONLY)) ? nullptr : ggml_cuda_mmb_slot_reserve(ctx, 3, dst, (size_t) T * E);
    const bool store_f32 = !((outh || outf) && ggml_cuda_mmb_is_bf16_only(ctx, dst));
    // 32 channels x 128 tokens per CTA. gfx1151 pp2048 b/ub 4096: 64 x 64 cuts the Xn over-fetch but halves the token
    // tile (twice the weight re-reads per channel block): 1671 -> 1649 t/s.
    const dim3 grid(E / 32, (T + 127) / 128);
    const uint16_t * hc_shadow = mmb_is_hc_q8(w) ? mmb_shadow_lookup(ctx, w) : nullptr;
    if (hc_shadow) {
        hc_gate_mix_kernel<4, 2, 32><<<grid, MMB_NT, 0, stream>>>((const uint8_t *) hc_shadow, lo16, xn16, (float *) dst->data, outh, outf, store_f32, E, K, T, scale, bias);
    } else {
        GGML_ASSERT(w->type == GGML_TYPE_IQ4_NL);   // see ggml_cuda_hc_gate_mix_supported
        hc_gate_mix_kernel<4, 0, 32><<<grid, MMB_NT, 0, stream>>>((const uint8_t *) w->data, lo16, xn16, (float *) dst->data, outh, outf, store_f32, E, K, T, scale, bias);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}


// gufo's RoutedF16 helpers, verbatim (HeaderByte/GatherBit/SpreadHighBits/CodesToHalves).
__device__ __forceinline__ uint4 mmb_gluk_bf2f16(uint4 v) {
    uint32_t o[4];
    for (int p = 0; p < 4; ++p) {
        const uint32_t w = (&v.x)[p];
        // FIX (inference-path): the activation is BF16 - widen bf16 bits to f32, then round to f16.
        // (was mmb_h2f(): reinterpreting bf16 bits as f16, i.e. garbage activations into the f16 WMMA)
        const _Float16 a = (_Float16) __uint_as_float(w << 16);
        const _Float16 b = (_Float16) __uint_as_float(w & 0xffff0000u);
        o[p] = (uint32_t) __builtin_bit_cast(uint16_t, a) | ((uint32_t) __builtin_bit_cast(uint16_t, b) << 16);
    }
    return make_uint4(o[0], o[1], o[2], o[3]);
}
typedef short v16hk_mmb __attribute__((ext_vector_type(16)));
__device__ __forceinline__ uint32_t mmb_gluk_header_byte(const uint4 & h, uint32_t i) {
    const uint32_t word = i < 4 ? h.x : i < 8 ? h.y : i < 12 ? h.z : h.w;
    return (word >> (8U * (i & 3U))) & 0xFFU;
}
__device__ __forceinline__ uint32_t mmb_gluk_gbit(uint32_t w, int s) {
    return (((w >> s) & 0x01010101U) * 0x01020408U) >> 24U;
}
__device__ __forceinline__ uint32_t mmb_gluk_shbits(uint32_t bits) {
    return (__umul24(bits, 0x00204081U) & 0x01010101U) << 4U;
}
__device__ __forceinline__ void mmb_gluk_c2h(uint32_t codes, __half2 idx_lo, __half2 idx_hi, __half2 scale2, __half2 bias2, __half2 & lo, __half2 & hi) {
    const uint32_t p0 = __builtin_amdgcn_perm(codes, 0x64646464U, 0x01050004U);
    const uint32_t p1 = __builtin_amdgcn_perm(codes, 0x64646464U, 0x03070206U);
    lo = __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), idx_lo), scale2, bias2);
    hi = __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), idx_hi), scale2, bias2);
}
__device__ __forceinline__ v8f mmb_gluk_wmma(v16s a, v16s b, v8f c) {
#if defined(RDNA3)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    GGML_UNUSED_VARS(a, b);
    NO_DEVICE_CODE;
    return c;
#endif // defined(RDNA3)
}
template <int WTYPE, int BN, bool PAIR = true> __global__ void __launch_bounds__(256) mmb_routed_gluk_kernel(const uint8_t * __restrict__ w_g, const uint8_t * __restrict__ w_u,
        const uint16_t * __restrict__ x, float * __restrict__ out, uint16_t * __restrict__ out_half, const bool store_f32,
        const std::int32_t * __restrict__ rows_in, const std::int32_t * __restrict__ rows_out,
        const std::int32_t * __restrict__ pad_bounds, const uint32_t * __restrict__ tiles,
        const std::size_t expert_bytes, std::size_t m, std::size_t k) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3)
  NO_DEVICE_CODE;   // RDNA3 WMMA and wave32 inline asm
#else
  constexpr int BM = 128, BK = 2;
  constexpr bool kPair = PAIR;

  static_assert(BM == 128 || BM == 256, "eight waves, 16-row tiles");
  static_assert(!kPair || BM == 128);
  static_assert(BN % 16 == 0 && BN / 16 <= 8);
  static_assert(BK == 2, "one stage is one 32-byte Q4_K nibble group");
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = BM / 128;
  constexpr bool kQ5 = WTYPE == 32 + GGML_TYPE_Q5_1;
  constexpr bool kQ5K = WTYPE == 32 + GGML_TYPE_Q5_K;
  constexpr bool kQ8 = WTYPE == 32 + GGML_TYPE_Q8_0;
  constexpr bool kKQuant = WTYPE == 32 + GGML_TYPE_Q4_K || kQ5K;
  // 16-byte code chunks per row and stage: Q4_K's nibble pair and Q5_1's
  // two nibble blocks are two, Q8_0's two byte blocks are four.
  constexpr int kChunks = kQ8 ? 2 * BK : BK;

  // LDS plan (bytes): the code plane holds BM rows x kChunks 16-byte chunks
  // with the chunks of nearby rows permuted so a fragment read (one row per
  // lane) covers all bank groups; the activation plane is
  // [kb][16-element quarter][token][16 B] so a fragment read is 256
  // contiguous bytes; the epilogue reuses it all.
  constexpr int kCodeBytes = BM * kChunks * 16;
  constexpr int kHighBytes = (kQ5 || kQ5K) ? BK * BM * 4 : 0;
  constexpr int kScaleBytes = BK * BM * 4;
  // One slot of padding per activation quarter plane: the eight chunks of
  // a token then land on eight bank groups when they are written.
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  // The epilogue transposes one 16x16 tile per wave through the same
  // bytes (8 KB), which the narrow tile's stages do not reach.
  constexpr int kStageBytes = kCodeBytes + kHighBytes + kScaleBytes + kActBytes;
  constexpr int kLdsBytes = kStageBytes > 8 * 1024 ? kStageBytes : 8 * 1024;
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_codes = reinterpret_cast<uint4*>(lds);
  auto* s_high = reinterpret_cast<std::uint32_t*>(lds + kCodeBytes);
  auto* s_scale =
      reinterpret_cast<std::uint32_t*>(lds + kCodeBytes + kHighBytes);
  auto* s_act =
      reinterpret_cast<uint4*>(lds + kCodeBytes + kHighBytes + kScaleBytes);

  const std::int32_t tile = tiles[blockIdx.y];
  if (tile == UINT32_MAX) return;
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * BN;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const int live_tok_tiles =
      kTokTiles < (bucket_rows - t_local + 15) / 16
          ? kTokTiles
          : ((bucket_rows - t_local + 15) / 16);
  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const std::size_t row_bytes = mmb_row_bytes<WTYPE>(k);
  const auto* w_expert = static_cast<const std::uint8_t*>(w_g) +
                         static_cast<std::size_t>(expert) * expert_bytes;
  const auto* w_up_base = static_cast<const std::uint8_t*>(w_u);
  if constexpr (kPair) {
    w_up_base += static_cast<std::size_t>(expert) * expert_bytes;
  }
  (void) kPair;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  constexpr int kRows = kPair ? BM / 2 : BM;
  const int r_block = static_cast<int>(blockIdx.x) * kRows;

  // Weight fetch: unit u of a thread is (row = tid / 2 + 128 u, chunk c =
  // tid % 2). Q4_K: the two 16-byte halves of one 32-byte nibble group (two
  // K blocks, low and high nibbles); Q5_1: one 24-byte K block each.
  const int f_c = tid & 1;
  const std::uint8_t* f_ptr[kWaveRowTiles];
  bool f_live[kWaveRowTiles];
  uint4 f_header[kWaveRowTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r =
        r_block + (kPair ? (tid >> 1) % kRows : (tid >> 1) + (u * 128));
    f_live[u] = r < m_i;
    const std::uint8_t* weights = w_expert;
    if constexpr (kPair) {
      if ((tid >> 1) >= kRows) {
        weights = w_up_base;
      }
    }
    f_ptr[u] = weights +
               static_cast<std::size_t>(f_live[u] ? r : (m_i - 1)) * row_bytes;
    f_header[u] = make_uint4(0u, 0u, 0u, 0u);
  }
  // The next stage's weights and activations, fetched one stage ahead. The
  // (scale, bias) pair is derived from the raw header word only when the
  // stage is committed, so nothing waits on the loads before the compute.
  uint4 f_codes[kWaveRowTiles];
  // Paired gate/up tiles reuse the full quantized block over four stages.
  // Keep its remaining codes in registers alongside the cached header.
  uint4 code_cache[kWaveRowTiles][4];
  uint4 f_codes_hi[kWaveRowTiles];  ///< Q8_0: the block's second 16 codes
  uint4 f_qh[kWaveRowTiles][2];     ///< Q5_K: the superblock's high bits
  std::uint32_t f_high[kWaveRowTiles];
  std::uint32_t f_dm[kWaveRowTiles];  ///< Q5_1: d | m; Q8_0: d
  int f_sb32[kWaveRowTiles];          ///< Q4_K: the K block in its superblock
  constexpr int kActFetch = BN <= 64 ? 2 : 4;
  uint4 a_data[kActFetch];

  // Activation fetch: BN tokens x (BK * 64) bytes per stage in 16-byte
  // chunks, eight per token; each thread fetches consecutive 256-chunk
  // strides, up to four for a 128-token tile.
  constexpr int kActChunks = BN * BK * 4;
  static_assert(kActChunks <= kActFetch * 256);
  const uint16_t* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + (i * 256);
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    const int c_row = t_local + t;
    const std::int32_t src = (chunk < kActChunks && c_row < bucket_rows)
                                 ? rows_in[bucket_begin + c_row]
                                 : -1;
    a_src[i] = src >= 0 ? x + (static_cast<std::size_t>(src) * k) + (sub >> 2) * 32 + (sub & 3) * 8
                        : nullptr;
    // s_act[(kb * 4 + quarter) * kActStride + t]
    a_slot[i] = chunk < kActChunks ? (sub * kActStride) + t : -1;
  }

  const auto swizzle = [](int row, int c) {
    return (row * kChunks) +
           (c ^ (kChunks == 4 ? ((row >> 1) & 3) : ((row >> 2) & 1)));
  };

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int u = 0; u < kWaveRowTiles; ++u) {
      if constexpr (kQ5) {
        const int kb = kb0 + f_c;
        const auto* words = reinterpret_cast<const uint2*>(f_ptr[u]) + (kb * 3);
        const uint2 w0 = words[0];
        const uint2 w1 = words[1];
        const uint2 w2 = words[2];
        f_codes[u] = make_uint4(w1.x, w1.y, w2.x, w2.y);
        f_high[u] = w0.y;
        f_dm[u] = w0.x;
      } else if constexpr (kQ8) {
        // block_q8_0 is 34 bytes, so the code loads are 2-byte aligned.
        const auto* blk = f_ptr[u] + ((kb0 + f_c) * 34);
        f_dm[u] = *reinterpret_cast<const std::uint16_t*>(blk);
        __builtin_memcpy(&f_codes[u], blk + 2, 16);
        __builtin_memcpy(&f_codes_hi[u], blk + 18, 16);
      } else {
        constexpr int kBlockChunks = kQ5K ? 11 : 9;
        constexpr int kCodeChunk = kQ5K ? 3 : 1;
        const int block = kb0 / 8;
        const auto* blk =
            reinterpret_cast<const uint4*>(f_ptr[u]) + (block * kBlockChunks);
        // The K sweep enters a new superblock every eight Q8-sized blocks.
        if (kb0 % 8 == 0) {
          f_header[u] = blk[0];
          if constexpr (kPair) {
#pragma unroll
            for (int group = 0; group < 4; ++group)
              code_cache[u][group] = blk[kCodeChunk + group * 2 + f_c];
          }
          if constexpr (kQ5K) {
            f_qh[u][0] = blk[1];
            f_qh[u][1] = blk[2];
          }
        }
        const int sb32 = (kb0 % 8) + f_c;
        if constexpr (kPair) {
          const int group = sb32 / 2;
          // Select register values explicitly: a runtime array index or C++
          // conditional can become an address select and spill the cache.
          // Three cndmask instructions per word also avoid four ANDs and
          // three ORs. These are wave32 predicates; group is uniform.
          const std::uint32_t low = 0u - (group & 1u);
          const std::uint32_t high = 0u - ((group >> 1u) & 1u);
          const auto select = [](std::uint32_t a, std::uint32_t b,
                                 std::uint32_t mask) {
            std::uint32_t value;
            asm("v_cndmask_b32_e64 %0, %1, %2, %3"
                : "=v"(value)
                : "v"(a), "v"(b), "s"(mask));
            return value;
          };
          const uint4& c0 = code_cache[u][0];
          const uint4& c1 = code_cache[u][1];
          const uint4& c2 = code_cache[u][2];
          const uint4& c3 = code_cache[u][3];
          f_codes[u] = make_uint4(
              select(select(c0.x, c1.x, low), select(c2.x, c3.x, low), high),
              select(select(c0.y, c1.y, low), select(c2.y, c3.y, low), high),
              select(select(c0.z, c1.z, low), select(c2.z, c3.z, low), high),
              select(select(c0.w, c1.w, low), select(c2.w, c3.w, low), high));
        } else {
          f_codes[u] = blk[kCodeChunk + (sb32 / 2) * 2 + f_c];
        }
        f_sb32[u] = sb32;
        if constexpr (kQ5K) {
          // Bit sb32 of the 32 high-bit bytes, packed as the Q5_1 word.
          const std::uint32_t qh[8] = {f_qh[u][0].x, f_qh[u][0].y, f_qh[u][0].z,
                                       f_qh[u][0].w, f_qh[u][1].x, f_qh[u][1].y,
                                       f_qh[u][1].z, f_qh[u][1].w};
          std::uint32_t high = 0;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            high |= mmb_gluk_gbit(qh[i], sb32) << (4 * i);
          }
          f_high[u] = high;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      a_data[i] = a_src[i] != nullptr
                      ? mmb_gluk_bf2f16(*reinterpret_cast<const uint4*>(a_src[i] + (kb0 * 32)))
                      : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int u = 0; u < kWaveRowTiles; ++u) {
      const int row = (tid >> 1) + (u * 128);
      std::uint32_t scale_bias = 0;
      if constexpr (kQ8) {
        s_codes[swizzle(row, 2 * f_c)] = f_codes[u];
        s_codes[swizzle(row, (2 * f_c) + 1)] = f_codes_hi[u];
        scale_bias = f_live[u] ? f_dm[u] : 0U;  // half2 (d, 0)
      } else {
        s_codes[swizzle(row, f_c)] = f_codes[u];
      }
      if constexpr (kQ5K) {
        s_high[(f_c * BM) + row] = f_high[u];
      }
      if constexpr (kQ8) {
      } else if constexpr (kQ5) {
        s_high[(f_c * BM) + row] = f_high[u];
        const __half2 dm = __builtin_bit_cast(__half2, f_dm[u]);
        const float d = f_live[u] ? __low2float(dm) : 0.0F;
        const float mn = f_live[u] ? __high2float(dm) : 0.0F;
        scale_bias =
            __builtin_bit_cast(std::uint32_t, __floats2half2_rn(d, mn));
      } else {
        const int sb32 = f_sb32[u];
        std::uint32_t sc = 0;
        std::uint32_t mn = 0;
        if (sb32 < 4) {
          sc = mmb_gluk_header_byte(f_header[u], 4 + sb32) & 0x3FU;
          mn = mmb_gluk_header_byte(f_header[u], 8 + sb32) & 0x3FU;
        } else {
          sc = (mmb_gluk_header_byte(f_header[u], 8 + sb32) & 0x0FU) |
               ((mmb_gluk_header_byte(f_header[u], sb32) >> 6U) << 4U);
          mn = (mmb_gluk_header_byte(f_header[u], 8 + sb32) >> 4U) |
               ((mmb_gluk_header_byte(f_header[u], 4 + sb32) >> 6U) << 4U);
        }
        const __half2 dm = __builtin_bit_cast(__half2, f_header[u].x);
        const float scale =
            f_live[u] ? __low2float(dm) * static_cast<float>(sc) : 0.0F;
        const float offset =
            f_live[u] ? __high2float(dm) * static_cast<float>(mn) : 0.0F;
        scale_bias = __builtin_bit_cast(std::uint32_t,
                                        __floats2half2_rn(scale, -offset));
      }
      s_scale[(f_c * BM) + row] = scale_bias;
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      if (a_slot[i] >= 0) {
        s_act[a_slot[i]] = a_data[i];
      }
    }
  };

  v8f acc[kWaveRowTiles][kTokTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      acc[u][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  const __half2 magic =
      __floats2half2_rn(kQ8 ? -1152.0F : -1024.0F, kQ8 ? -1152.0F : -1024.0F);
  const auto compute_stage = [&]() {
    uint4 raw[kWaveRowTiles][BK];
    if constexpr (!kQ8) {
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
#pragma unroll
        for (int c = 0; c < BK; ++c) {
          raw[u][c] = s_codes[swizzle(row, c)];
        }
      }
    }
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16s a_lo[kWaveRowTiles];
      v16s a_hi[kWaveRowTiles];
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
        const __half2 sb =
            __builtin_bit_cast(__half2, s_scale[(kb * BM) + row]);
        const __half2 scale2 = __low2half2(sb);
        const __half2 bias2 = __high2half2(sb);
        std::uint32_t nib[8];
        if constexpr (kQ8) {
          // Q8_0: the block's 32 signed bytes are chunks 2 kb and 2 kb + 1;
          // flipping the sign bit carries q + 128, which the 1152 magic
          // takes back out.
          const uint4 c0 = s_codes[swizzle(row, 2 * kb)];
          const uint4 c1 = s_codes[swizzle(row, (2 * kb) + 1)];
          const std::uint32_t words[8] = {c0.x, c0.y, c0.z, c0.w,
                                          c1.x, c1.y, c1.z, c1.w};
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            nib[i] = words[i] ^ 0x80808080U;
          }
        } else if constexpr (kQ5) {
          // Q5_1: K block kb's 16 bytes are chunk kb; elements 0-15 take
          // the low nibbles, 16-31 the high, plus bit j of the high-bit
          // word.
          const uint4 r = raw[u][kb];
          const std::uint32_t high = s_high[(kb * BM) + row];
          const std::uint32_t words[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            nib[i] = (words[i] & 0x0F0F0F0FU) |
                     mmb_gluk_shbits((high >> (4 * i)) & 0xFU);
            nib[4 + i] = ((words[i] >> 4U) & 0x0F0F0F0FU) |
                         mmb_gluk_shbits((high >> (16 + 4 * i)) & 0xFU);
          }
        } else {
          // Q4_K / Q5_K: elements 0-15 of K block kb0 + kb are the low
          // (kb = 0) or high (kb = 1) nibbles of chunk 0, elements 16-31
          // of chunk 1; Q5_K adds bit j of the staged high-bit word.
          const unsigned shift = 4U * static_cast<unsigned>(kb);
          const std::uint32_t words[8] = {raw[u][0].x, raw[u][0].y, raw[u][0].z,
                                          raw[u][0].w, raw[u][1].x, raw[u][1].y,
                                          raw[u][1].z, raw[u][1].w};
          const std::uint32_t high = kQ5K ? s_high[(kb * BM) + row] : 0U;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            nib[i] = (words[i] >> shift) & 0x0F0F0F0FU;
            if constexpr (kQ5K) {
              nib[i] |= mmb_gluk_shbits((high >> (4 * i)) & 0xFU);
            }
          }
        }
        __half2 h[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          mmb_gluk_c2h(nib[i], magic, magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
        }
        __builtin_memcpy(&a_lo[u], &h[0], 32);
        __builtin_memcpy(&a_hi[u], &h[8], 32);
      }
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        if constexpr (kPair || ((kQ5 || kQ8) && BN >= 48)) {
          // Keep one token tile's LDS fragments live at a time. Hoisting
          // all eight tiles spills registers and defeats the wider tile's
          // reuse of each weight decode. This is a compiler barrier only.
          asm volatile("" ::: "memory");
        }
        // A short expert bucket has no output in the remaining token
        // tiles, so omit their WMMA work.
        if constexpr ((kPair || ((kQ5 || kQ8) && BN >= 48)) && kTokTiles > 1) {
          if (j >= live_tok_tiles)
            continue;
        }
        const uint4* frag =
            s_act + ((kb * 4) * kActStride) + (j * 16) + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          b[q] = frag[q * kActStride];
        }
        v16s b_lo;
        v16s b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
          acc[u][j] = mmb_gluk_wmma(a_lo[u], b_lo, acc[u][j]);
          acc[u][j] = mmb_gluk_wmma(a_hi[u], b_hi, acc[u][j]);
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
    compute_stage();
    __syncthreads();
  }

  if constexpr (kPair) {
    // Four waves hold gate rows, four the matching up rows: silu(gate)*up.
    constexpr unsigned stride = 18;
    constexpr unsigned plane = 16 * stride;
    float* base = reinterpret_cast<float*>(lds);
    float* scratch = base + wave_id * plane;
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        scratch[sub_lane * stride + 2 * l + half_id] = acc[0][j][l];
      }
      __syncthreads();
#pragma unroll
      for (int unit = 0; unit < 2; ++unit) {
        const int flat = (unit * 256 + tid) * 2;
        const int t = t_local + j * 16 + flat / kRows;
        const int r = flat % kRows;
        if (t < bucket_rows && r_block + r < m_i) {
          const std::int32_t dst = rows_out[pad_bounds[expert] + t];
          if (dst >= 0) {
            const int idx = (r / 16) * plane + (flat / kRows) * stride + r % 16;
            for (int v = 0; v < 2; ++v) {
              const float g = base[idx + v];
              const float up = base[idx + v + 4 * plane];
              const float value = ggml_cuda_op_silu_single(g) * up;
              const std::size_t o = (std::size_t)dst * m + r_block + r + v;
              if (store_f32) out[o] = value;
              if (out_half != nullptr) out_half[o] = mmb_f2bf(value);
            }
          }
        }
      }
      __syncthreads();
    }
    return;
  }

  {
    // Single-matrix direct epilogue (!kPair): wave w's C fragment is the
    // 16-row tile [r_block + 16w .. +15] x tokens [t_local + 16j .. +15]:
    // acc[u][j][l] = C[row = 16w + 2l + half_id, token = 16j + sub_lane].
    // Same (t, r) math as the pair epilogue minus the pairing: out[o],
    // o = dst*m + r_block + r, with r the fragment's own row.
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      const int t = t_local + j * 16 + sub_lane;
      if (t >= bucket_rows) continue;                    // padded tail tokens
      const std::int32_t dst = rows_out[pad_bounds[expert] + t];
      if (dst < 0) continue;
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {           // = 1 for BM=128
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          const int r = r_block + (wave_id * 16) + (u * 128) + 2 * l + half_id;
          if (r >= m_i) continue;                         // M-tile tail rows
          const float value = acc[u][j][l];
          const std::size_t o = (std::size_t)dst * m + r;
          if (store_f32) out[o] = value;
          if (out_half != nullptr) out_half[o] = mmb_f2bf(value);
        }
      }
    }
    return;
  }
#endif
}

// Tier streams: the routed small tier (hundreds of experts under THRESH rows: weight-streaming bound, ~360 GB/s
// from DRAM+MALL) runs on a side stream concurrently with the big tier (a hundred experts holding ~85% of the rows:
// WMMA bound), so the weight stream hides behind the big tier's compute. The tiers write disjoint output rows and
// read only the desc/ids built before the fork. Prefill runs under HIP graph capture here, so the fork/join is
// captured as graph edges (event record + wait, the side stream rejoins before the op returns). Declare after the
// pool allocations the tiers read: the destructor joins the side stream before they are freed.
// Cross-op join: the GLU op leaves its small tier running on the side stream (no join);
// the following routed-down op builds its ids/desc on the side stream (after small GLU), main waits only for that,
// then big down (main, after big GLU) and small down (side, after small GLU) overlap; join at the end of down.
// pp2048 1675 -> 1684 t/s vs a per-op join, PPL unchanged.
// The pending side, the holds, the routing record and the events are per backend context (ggml_cuda_mmb_context).
template <typename T> static ggml_cuda_pool_alloc<T> & mmb_hold_alloc(ggml_backend_cuda_context & ctx, size_t n) {
    auto * a = new ggml_cuda_pool_alloc<T>(ctx.pool(), n);
    mmb_state(ctx).glu_hold.push_back([a]() { delete a; });
    return *a;
}
// The routed down reuses the routing maps of the GLU before it (same ids tensor). For the GLU
// (src1 [K, 1, T]) the helper writes ids_dst = token*n_used + slot, which is exactly the down's ids_src1 (src1 [K, n_used, T])
// and ids_dst; the bounds are the same. The GLU maps are held (mmb_hold_alloc) until the down releases them.
static void mmb_hold_release(ggml_backend_cuda_context & ctx) {
    ggml_cuda_mmb_context & st = mmb_state(ctx);
    for (auto it = st.glu_hold.rbegin(); it != st.glu_hold.rend(); ++it) (*it)();
    st.glu_hold.clear();
    st.ids_rec = mmb_ids_record{};
}
static cudaEvent_t mmb_event(cudaEvent_t & e) {
    if (!e) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    return e;
}
struct mmb_tier_fork {
    cudaStream_t main = nullptr, side = nullptr;
    cudaEvent_t join = nullptr;
    bool nojoin = false;
    // mode 0: normal fork/join; 1: fork, leave side pending (no join); 2: continue a pending side (no entry wait), join
    explicit mmb_tier_fork(ggml_backend_cuda_context & ctx, int mode = 0) {
        constexpr int SIDE = GGML_CUDA_MAX_STREAMS - 1;
        main = ctx.stream();
        if (ctx.curr_stream_no == SIDE || !(ggml_cuda_ovl_mask() & 8)) return;   // GGML_CUDA_OVL bit 8: tier side stream
        ggml_cuda_mmb_context & st = mmb_state(ctx);
        side = ctx.stream(ctx.device, SIDE);
        if (mode == 2 && st.side_pending) {
            st.side_pending = false;      // side already ordered after its own GLU work
        } else {
            CUDA_CHECK(cudaEventRecord(mmb_event(st.fork_ev[0]), main));
            CUDA_CHECK(cudaStreamWaitEvent(side, st.fork_ev[0], 0));
        }
        if (mode == 1) { nojoin = true; st.side_pending = true; }
        join = mmb_event(st.fork_ev[1]);
    }
    cudaStream_t small() const { return side ? side : main; }
    ~mmb_tier_fork() {
        if (side && !nojoin) {
            CUDA_CHECK(cudaEventRecord(join, side));
            CUDA_CHECK(cudaStreamWaitEvent(main, join, 0));
        }
    }
};
static void mmb_join_pending(ggml_backend_cuda_context & ctx) {
    ggml_cuda_mmb_context & st = mmb_state(ctx);
    if (!st.side_pending) return;
    st.side_pending = false;
    const cudaEvent_t ej = mmb_event(st.join_ev);
    CUDA_CHECK(cudaEventRecord(ej, ctx.stream(ctx.device, GGML_CUDA_MAX_STREAMS - 1)));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), ej, 0));
}

// ---------------------------------------------------------------------------------------------------------------------
// Down R2: routed MoE down projection (Q5_1, K=640) on a repacked shadow of the expert weights (team/moe-down md4,
// variant "L2 256x64 KC128 interleaved", MODE 0 + the existing weighted reduction).
// Shadow layout R2, same 24 B per 32 weights as Q5_1: [expert][M/16 row tile][K/32 block][16 rows][24 B]
//   D0..D4 (20 B): pair P = 3j+k (P<15) holds q[2P] at bits [5k,5k+5) and q[2P+1] at [16+5k,21+5k) of D_j;
//                  pair 15: bit j of q30 is bit 15 of D_j, bit j of q31 is bit 31 of D_j
//   dm (4 B): {f16(d * 2^19), f16(m)}.  q at f16 mantissa bits [5,10) is the subnormal q*2^-19 exactly, so
//   v_pk_fma_f16(w, d*2^19, m) = RN_f16(q*d + m): one pk_fma per 2 weights, no LDS for A.
// The original Q5_1 tensor is untouched (decode / MMVQ keep reading it); the shadow is a separate allocation.
// Activations: the bf16 X copy is widened to f32 and rounded (saturating) to f16 in the B-tile load, see
// mmb_gluk_bf2f16 (the bf16-activation correctness fix) for why it must not be reinterpreted.
namespace {
typedef _Float16 r2_v16h __attribute__((ext_vector_type(16)));
typedef uint32_t r2_v8u  __attribute__((ext_vector_type(8)));
constexpr int R2_BM = 256, R2_BN = 64;

__global__ void mmb_r2_repack_kernel(const uint8_t * __restrict__ W, uint8_t * __restrict__ R, const size_t nblk,
        const int M, const int NB) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk) return;
    const size_t per_e = (size_t) M * NB;
    const size_t e = i / per_e, rem = i % per_e;
    const int row = (int) (rem / NB), b = (int) (rem % NB);
    const uint2 * p = (const uint2 *) (W + i * 24);
    const uint2 x0 = p[0], x1 = p[1], x2 = p[2];
    const uint32_t dm = x0.x, qh = x0.y;
    const uint32_t qs[4] = {x1.x, x1.y, x2.x, x2.y};
    auto q = [&](int j) -> uint32_t {
        const uint32_t byte = (qs[(j & 15) >> 2] >> (8 * (j & 3))) & 0xffu;
        const uint32_t lo = j < 16 ? (byte & 15u) : (byte >> 4);
        return lo | (((qh >> j) & 1u) << 4);
    };
    uint32_t D[5] = {0, 0, 0, 0, 0};
#pragma unroll
    for (int P = 0; P < 15; ++P) {
        const int j = P / 3, k = P % 3;
        D[j] |= q(2 * P) << (5 * k);
        D[j] |= q(2 * P + 1) << (16 + 5 * k);
    }
    const uint32_t q30 = q(30), q31 = q(31);
#pragma unroll
    for (int j = 0; j < 5; ++j) { D[j] |= ((q30 >> j) & 1u) << 15; D[j] |= ((q31 >> j) & 1u) << 31; }
    const float ds = (float) __builtin_bit_cast(_Float16, (uint16_t) (dm & 0xffffu)) * 524288.0f;
    const _Float16 dsh = (_Float16) ds;
    const uint32_t dmo = (uint32_t) __builtin_bit_cast(uint16_t, dsh) | (dm & 0xffff0000u);
    uint32_t * o = (uint32_t *) (R + ((((e * (size_t) (M / 16) + row / 16) * NB + b) * 16 + row % 16) * 24));
    o[0] = D[0]; o[1] = D[1]; o[2] = D[2]; o[3] = D[3]; o[4] = D[4]; o[5] = dmo;
}

// d * 2^19 must be an exact finite f16 for every block, else the tensor keeps its Q5_1 layout
__global__ void mmb_r2_check_kernel(const uint8_t * __restrict__ W, const size_t nblk, int * __restrict__ bad) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk) return;
    const uint16_t dh = *(const uint16_t *) (W + i * 24);
    const float ds = (float) __builtin_bit_cast(_Float16, dh) * 524288.0f;
    if (!((float) (_Float16) ds == ds) || !(fabsf(ds) <= 65504.0f)) atomicOr(bad, 1);
}

// Tile list for the R2 kernel: experts by row count (desc, stable), each expert's [256-row tile][64-token tile]s;
// full tiles (>= 32 tokens) and partial tiles are interleaved proportionally (md4 "interleaved") so the partial
// tail does not run alone at the end. Unused entries get cnt = 0 (the CTA exits). One block, E <= 1024.
__global__ void mmb_r2_build_desc(const int32_t * __restrict__ bounds, uint4 * __restrict__ desc, const int E, const int n_rt,
        const int n_max) {
    __shared__ int cnt[1024], order[1024], bpre[1024], spre[1024];
    __shared__ int tot[2];
    const int t = threadIdx.x;
    if (t < E) cnt[t] = bounds[t + 1] - bounds[t];
    __syncthreads();
    if (t < E) {
        const int c = cnt[t]; int rank = 0;
        for (int j = 0; j < E; ++j) { const int cj = cnt[j]; rank += (cj > c) || (cj == c && j < t); }
        order[rank] = t;
    }
    __syncthreads();
    {   // exclusive prefixes over the rank order: inclusive scan of (full, partial) tile counts, then shift
        int nb = 0, ns = 0;
        if (t < E) {
            const int c = cnt[order[t]];
            const int nfull = c / R2_BN + ((c % R2_BN) >= 32 ? 1 : 0), npart = (c + R2_BN - 1) / R2_BN - nfull;
            nb = n_rt * nfull; ns = n_rt * npart;
        }
        bpre[t] = nb; spre[t] = ns;
        __syncthreads();
        for (int off = 1; off < 1024; off <<= 1) {
            const int a = t >= off ? bpre[t - off] : 0, b = t >= off ? spre[t - off] : 0;
            __syncthreads();
            bpre[t] += a; spre[t] += b;
            __syncthreads();
        }
        if (t == 1023) { tot[0] = bpre[t]; tot[1] = spre[t]; }
        bpre[t] -= nb; spre[t] -= ns;
    }
    __syncthreads();
    const long long nB = tot[0], nS = tot[1];
    if (t < E) {
        const int e = order[t], c = cnt[e], r0e = bounds[e];
        const int njt = (c + R2_BN - 1) / R2_BN, nfull = c / R2_BN + ((c % R2_BN) >= 32 ? 1 : 0), npart = njt - nfull;
        for (int rt = 0; rt < n_rt; ++rt) {
            for (int jt = 0; jt < njt; ++jt) {
                const int tc = min(R2_BN, c - jt * R2_BN);
                long long pos;
                if (tc >= 32) {   // full tiles are jt < nfull (only the last tile can be short)
                    const long long ib = bpre[t] + (long long) rt * nfull + jt;
                    const long long before = nB > 0 ? (ib * nS + nB - 1) / nB : 0;
                    pos = ib + (before < nS ? before : nS);
                } else {
                    const long long is = spre[t] + (long long) rt * npart;
                    const long long before = nS > 0 ? (is * nB) / nS + 1 : 0;
                    pos = is + (before < nB ? before : nB);
                }
                desc[pos] = make_uint4((uint32_t) e, (uint32_t) rt, (uint32_t) (r0e + jt * R2_BN), (uint32_t) tc);
            }
        }
    }
    for (long long p = nB + nS + t; p < n_max; p += blockDim.x) desc[p] = make_uint4(0, 0, 0, 0);
}

__device__ __forceinline__ uint32_t r2_and_or(uint32_t a, uint32_t m, uint32_t c) {
    uint32_t r; asm("v_and_or_b32 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(m), "v"(c)); return r;
}
__device__ __forceinline__ void r2_dec(const uint32_t * D, const uint32_t dd, const uint32_t mm, r2_v8u & f0, r2_v8u & f1) {
    uint32_t w[16];
#pragma unroll
    for (int P = 0; P < 15; ++P) {
        const int k = P % 3; const uint32_t Dj = D[P / 3];
        w[P] = (k == 0 ? (Dj << 5) : k == 1 ? Dj : (Dj >> 5)) & 0x03E003E0u;
    }
    uint32_t w15 = (D[0] >> 10) & 0x00200020u;
#pragma unroll
    for (int j = 1; j < 5; ++j) w15 = r2_and_or(D[j] >> (10 - j), 0x00200020u << j, w15);
    w[15] = w15;
#pragma unroll
    for (int P = 0; P < 16; ++P) {
        uint32_t v;
        asm("v_pk_fma_f16 %0, %1, %2, %3" : "=v"(v) : "v"(w[P]), "v"(dd), "v"(mm));
        if (P < 8) f0[P] = v; else f1[P - 8] = v;
    }
}
// bf16 -> f32 -> f16 (RNE), saturated to +-65504 so an out-of-range activation cannot become inf
__device__ __forceinline__ uint32_t r2_bf2h2(const uint32_t w) {
    const float a = fminf(fmaxf(__uint_as_float(w << 16), -65504.0f), 65504.0f);
    const float b = fminf(fmaxf(__uint_as_float(w & 0xffff0000u), -65504.0f), 65504.0f);
    return (uint32_t) __builtin_bit_cast(uint16_t, (_Float16) a) | ((uint32_t) __builtin_bit_cast(uint16_t, (_Float16) b) << 16);
}

// 8 waves x (TM*16 rows) = 256 rows, TN*16 = 64 tokens; K in KC chunks (B double-buffered in LDS), A straight from
// global into registers (PD blocks deep) and decoded per wave. Output: bf16 Dh[ids_dst][M].
template <int K, int TM = 2, int TN = 4, int KC = 128, int PD = 2>
__global__ void __launch_bounds__(256)
mmb_r2_down_kernel(const uint8_t * __restrict__ R, const size_t eb, const uint16_t * __restrict__ X, const int32_t * __restrict__ ids_src,
        const int32_t * __restrict__ ids_dst, const uint4 * __restrict__ descs, uint16_t * __restrict__ Dh, float * __restrict__ D, const int M) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3)
    NO_DEVICE_CODE;
#else
    constexpr int NB = K / 32, BM = 8 * 16 * TM, BN = 16 * TN, NCH = K / KC, LSB = KC + 8, BPC = KC / 32;
    constexpr int BPI = BN * KC / 8 / 256;
    static_assert(BPI >= 1 && K % KC == 0 && BM == R2_BM && BN == R2_BN, "");
    constexpr int BSZ = (2 * BN * LSB > 8 * BN * TM * 16) ? BN * LSB : 4 * BN * TM * 16;
    __shared__ __align__(16) _Float16 Bs[2][BSZ];
    const uint4 dsc = descs[blockIdx.x];
    const int cnt = (int) dsc.w;
    if (cnt == 0) return;
    const int e = (int) dsc.x, rt = (int) dsc.y, r0 = (int) dsc.z;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, r = lane & 15, hf = lane >> 4;
    const uint8_t * Ap[TM];
#pragma unroll
    for (int i = 0; i < TM; ++i) Ap[i] = R + (size_t) e * eb + (size_t) ((rt * BM) / 16 + wave * TM + i) * NB * 384 + r * 24;
    const uint16_t * bsrc[BPI]; int bdst[BPI];
#pragma unroll
    for (int i = 0; i < BPI; ++i) {
        const int c = tid + i * 256, tok = c / (KC / 8), kp = c % (KC / 8);
        const int sr = tok < cnt ? ids_src[r0 + tok] : -1;
        bsrc[i] = sr >= 0 ? X + (size_t) sr * K + kp * 8 : nullptr;
        bdst[i] = tok * LSB + kp * 8;
    }
    uint4 breg[BPI];
    auto loadB = [&](int ch) {
#pragma unroll
        for (int i = 0; i < BPI; ++i) breg[i] = bsrc[i] ? *(const uint4 *) (bsrc[i] + ch * KC) : make_uint4(0, 0, 0, 0);
    };
    auto storeB = [&](int buf) {
#pragma unroll
        for (int i = 0; i < BPI; ++i) {
            const uint4 v = breg[i];
            *(uint4 *) (&Bs[buf][bdst[i]]) = make_uint4(r2_bf2h2(v.x), r2_bf2h2(v.y), r2_bf2h2(v.z), r2_bf2h2(v.w));
        }
    };
    uint4 alo[PD][TM]; uint2 ahd[PD][TM];
    auto loadA = [&](int slot, int b) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const uint8_t * p = Ap[i] + (size_t) b * 384;
            const uint2 x0 = *(const uint2 *) p, x1 = *(const uint2 *) (p + 8), x2 = *(const uint2 *) (p + 16);
            alo[slot][i] = make_uint4(x0.x, x0.y, x1.x, x1.y); ahd[slot][i] = x2;
        }
    };
    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    const int njt = (cnt + 15) >> 4;

    loadB(0); storeB(0);
#pragma unroll
    for (int b = 0; b < PD; ++b) loadA(b, b);
    __syncthreads();
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch) {
        if (ch + 1 < NCH) loadB(ch + 1);
#pragma unroll
        for (int bb = 0; bb < BPC; ++bb) {
            const int b = ch * BPC + bb, slot = b % PD;
            uint4 lo[TM]; uint2 hd[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) { lo[i] = alo[slot][i]; hd[i] = ahd[slot][i]; }
            if (b + PD < NB) loadA(slot, b + PD);
            r2_v8u afs[2][TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const uint32_t dm = hd[i].y;
                const uint32_t dd = __builtin_amdgcn_perm(dm, dm, 0x01000100u);
                const uint32_t mm = __builtin_amdgcn_perm(dm, dm, 0x03020302u);
                const uint32_t D[5] = {lo[i].x, lo[i].y, lo[i].z, lo[i].w, hd[i].x};
                r2_dec(D, dd, mm, afs[0][i], afs[1][i]);
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const _Float16 * bp = &Bs[ch & 1][r * LSB + bb * 32 + s * 16];
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    if (j < njt) {
                        const r2_v16h bf = *(const r2_v16h *) (bp + j * 16 * LSB);
#pragma unroll
                        for (int i = 0; i < TM; ++i) asm volatile("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0" : "+v"(acc[i][j]) : "v"(bf), "v"(afs[s][i]));
                    }
                }
            }
        }
        if (ch + 1 < NCH) { storeB((ch + 1) & 1); __syncthreads(); }
    }
    const int m0 = rt * BM + wave * TM * 16;
    if (D) {   // f32 output (dst not marked bf16-only): direct stores from the accumulators
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int tok = j * 16 + 2 * v + hf;
                if (tok < cnt) {
                    float * o = D + (size_t) ids_dst[r0 + tok] * M + m0 + r;
#pragma unroll
                    for (int i = 0; i < TM; ++i) o[i * 16] = acc[i][j][v];
                }
            }
        return;
    }
    __syncthreads();
    uint16_t * stg = (uint16_t *) &Bs[0][0] + wave * BN * TM * 16;   // per-wave [BN tokens][TM*16 ch] bf16
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v) stg[(j * 16 + 2 * v + hf) * (TM * 16) + i * 16 + r] = mmb_f2bf(acc[i][j][v]);
    constexpr int PPR = TM * 2, TPI = 32 / PPR;
#pragma unroll
    for (int it = 0; it < BN / TPI; ++it) {
        const int tok = it * TPI + lane / PPR, part = lane % PPR;
        if (tok < cnt) {
            const uint4 v = *(const uint4 *) (stg + tok * TM * 16 + part * 8);
            const int dr = ids_dst[r0 + tok];
            *(uint4 *) (Dh + (size_t) dr * M + m0 + part * 8) = v;
        }
    }
#endif
}

// f32 activations, one wave per (16-row tile, routed row): GEMV over the in-place R2 layout for every MUL_MAT_ID the R2
// WMMA kernel does not take (decode / small batches / MMB off). dst[:, i, t] = W[ids[i, t]] @ src1[:, i % ne11, t].
// 4 waves per block share one (i, t): the 640-float activation row is staged in LDS once. Each lane owns row r of the
// tile and every other K block (h = lane >> 4); the two halves are summed with one xor-16 shuffle.
template <int K>
__global__ void __launch_bounds__(128)
mmb_r2_gemv_kernel(const uint8_t * __restrict__ R, const size_t eb, const char * __restrict__ x, const size_t nb11, const size_t nb12,
        const int ne11, const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1, char * __restrict__ dst,
        const size_t nb1, const size_t nb2, const int M, const int E) {
    constexpr int NB = K / 32;
    __shared__ float xs[K];
    const int i = blockIdx.y, t = blockIdx.z;
    const float * xr = (const float *) (x + (size_t) t * nb12 + (size_t) (i % ne11) * nb11);
    for (int k = threadIdx.x; k < K; k += blockDim.x) xs[k] = xr[k];
    __syncthreads();
    const int e = *(const int32_t *) (ids + (size_t) t * ids_nb1 + (size_t) i * ids_nb0);
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, r = lane & 15, h = lane >> 4;
    const int rt = blockIdx.x * 4 + wave;
    if (rt * 16 >= M) return;
    const uint8_t * base = R + (size_t) e * eb + ((size_t) rt * NB * 16 + r) * 24;
    float acc = 0.0f;
    for (int b = h; b < NB; b += 2) {
        const uint2 * p = (const uint2 *) (base + (size_t) b * 384);
        const uint2 x0 = p[0], x1 = p[1], x2 = p[2];
        const uint32_t D[5] = {x0.x, x0.y, x1.x, x1.y, x2.x};
        const uint32_t dm = x2.y;
        const float d = (float) __builtin_bit_cast(_Float16, (uint16_t) (dm & 0xffffu)) * (1.0f / 524288.0f);
        const float m = (float) __builtin_bit_cast(_Float16, (uint16_t) (dm >> 16));
        const float * xb = xs + b * 32;
        float sq = 0.0f, sx = 0.0f;
#pragma unroll
        for (int P = 0; P < 15; ++P) {
            const uint32_t Dj = D[P / 3] >> (5 * (P % 3));
            const float a0 = xb[2 * P], a1 = xb[2 * P + 1];
            sq += (float) (Dj & 31u) * a0 + (float) ((Dj >> 16) & 31u) * a1;
            sx += a0 + a1;
        }
        uint32_t q30 = 0, q31 = 0;
#pragma unroll
        for (int j = 0; j < 5; ++j) { q30 |= ((D[j] >> 15) & 1u) << j; q31 |= ((D[j] >> 31) & 1u) << j; }
        sq += (float) q30 * xb[30] + (float) q31 * xb[31];
        sx += xb[30] + xb[31];
        acc += d * sq + m * sx;
    }
    acc += __shfl_xor(acc, 16, 32);
    if (h == 0) *(float *) (dst + (size_t) t * nb2 + (size_t) i * nb1 + (size_t) (rt * 16 + r) * sizeof(float)) = acc;
}

// ---------------------------------------------------------------------------------------------------------------------
// GLU R2: routed MoE gate/up + SwiGLU (Q4_K experts) on an in-place repacked layout ("G2"), one launch over a
// device-built tile list (big and small experts interleaved), A (weights) straight from global into registers.
// G2 layout, per tensor, same 144 B per (row, 256-K superblock) as Q4_K: [expert][M/16 row tile][K/256 sb][2304 B]
//   bytes    0..255 : [16 rows][16 B] the row's Q4_K header (d, dmin, scales[12]), unchanged
//   bytes 256..2303 : [group g 0..3][piece p 0..1][16 rows][16 B] = qs[32 g + 16 p .. +16) of the row
// so a wave's code load is 512 contiguous bytes (16 rows x both pieces, see mmb_g2_glu_kernel for the lane mapping).
// Decode: nibble q placed at f16 mantissa bits [6,10) is the subnormal q*2^-18 exactly, so
//   v_pk_fma_f16(w, f16(d*sc)*2^18, f16(-dmin*mn)) = RN_f16(q*f16(d*sc) - f16(dmin*mn)),
// bit-identical to the gluk decode (magic add then fma) as long as f16(d*sc)*2^18 is a finite f16, i.e. d*sc < 0.25
// (checked at repack; a pair that fails keeps the Q4_K layout and the gluk path).
// The repack swaps bytes 1 and 2 of every code dword, so the two-nibble extraction (bytes 0+2, then 1+3) yields the
// K elements in natural order: same operands and accumulation order as the gluk kernel, bit-identical results.
typedef _Float16 g2_v16h __attribute__((ext_vector_type(16)));
typedef uint32_t g2_v8u  __attribute__((ext_vector_type(8)));
constexpr int G2_TILE = 2304;   // bytes per (16-row tile, superblock)
constexpr int G2_BM = 128;      // output columns per block: 8 waves x 16

__global__ void mmb_g2_repack_kernel(const uint8_t * __restrict__ W, uint8_t * __restrict__ R, const size_t n, const int M, const int NSB) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;   // (expert, row, superblock) in Q4_K order
    if (i >= n) return;
    const size_t per_e = (size_t) M * NSB;
    const size_t e = i / per_e, rem = i % per_e;
    const int row = (int) (rem / NSB), s = (int) (rem % NSB);
    const uint4 * src = (const uint4 *) (W + i * 144);
    uint8_t * dst = R + e * per_e * 144 + ((size_t) (row / 16) * NSB + s) * G2_TILE + (row % 16) * 16;
    uint4 v[9];
#pragma unroll
    for (int c = 0; c < 9; ++c) v[c] = src[c];
    *(uint4 *) dst = v[0];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        const uint4 q = v[1 + c];   // bytes {0,1,2,3} -> {0,2,1,3} in every dword
        *(uint4 *) (dst + 256 + c * 256) = make_uint4(__builtin_amdgcn_perm(q.x, q.x, 0x03010200u), __builtin_amdgcn_perm(q.y, q.y, 0x03010200u),
                                                      __builtin_amdgcn_perm(q.z, q.z, 0x03010200u), __builtin_amdgcn_perm(q.w, q.w, 0x03010200u));
    }
}

__device__ __forceinline__ uint32_t g2_byte(const uint4 h, const int i) {
    const int k = i >> 2;
    const uint32_t w = k == 0 ? h.x : k == 1 ? h.y : k == 2 ? h.z : h.w;   // no dynamic register indexing
    return (w >> (8 * (i & 3))) & 0xffu;
}
// 6-bit scale / min of sub-block j (ggml get_scale_min_k4)
__device__ __forceinline__ void g2_scmn(const uint4 h, const int j, uint32_t & sc, uint32_t & mn) {
    if (j < 4) { sc = g2_byte(h, 4 + j) & 63u; mn = g2_byte(h, 8 + j) & 63u; }
    else {
        sc = (g2_byte(h, 8 + j) & 15u) | ((g2_byte(h, j) >> 6) << 4);
        mn = (g2_byte(h, 8 + j) >> 4) | ((g2_byte(h, 4 + j) >> 6) << 4);
    }
}
// f16(d*sc)*2^18 must stay a finite f16 for every sub-block, else the tensor keeps its Q4_K layout
__global__ void mmb_g2_check_kernel(const uint8_t * __restrict__ W, const size_t n, int * __restrict__ bad) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint4 h = *(const uint4 *) (W + i * 144);
    bool ok = true;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        uint32_t sc, mn; g2_scmn(h, j, sc, mn);
        const float d = (float) __builtin_bit_cast(_Float16, (uint16_t) (h.x & 0xffffu));
        const float sh = (float) (_Float16) (d * (float) sc);
        ok = ok && fabsf(sh * 262144.0f) <= 65504.0f;
    }
    if (!ok) atomicOr(bad, 1);
}

__device__ __forceinline__ uint32_t g2_fma(const uint32_t a, const uint32_t b, const uint32_t c) {
    uint32_t r; asm("v_pk_fma_f16 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c)); return r;
}
// 16 K elements of one row (4 repacked code dwords, low or high nibbles) -> WMMA fragment (natural K order)
template <bool HI> __device__ __forceinline__ g2_v8u g2_dec(const uint4 c, const uint32_t S2, const uint32_t B2) {
    const uint32_t w[4] = {c.x, c.y, c.z, c.w};
    g2_v8u f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t x = HI ? ((w[j] << 2) & 0x03C003C0u) : ((w[j] << 6) & 0x03C003C0u);
        const uint32_t y = HI ? ((w[j] >> 6) & 0x03C003C0u) : ((w[j] >> 2) & 0x03C003C0u);
        f[2 * j] = g2_fma(x, S2, B2);
        f[2 * j + 1] = g2_fma(y, S2, B2);
    }
    return f;
}
// bf16 -> f32 -> f16 (RNE), the conversion of the gluk kernel (mmb_gluk_bf2f16)
__device__ __forceinline__ uint32_t g2_h(const uint32_t fbits) {
    return (uint32_t) __builtin_bit_cast(uint16_t, (_Float16) __uint_as_float(fbits));
}
// 8 bf16 activations -> f16
__device__ __forceinline__ uint4 g2_cvt8(const uint4 v) {
    return make_uint4(g2_h(v.x << 16) | (g2_h(v.x & 0xffff0000u) << 16), g2_h(v.y << 16) | (g2_h(v.y & 0xffff0000u) << 16),
                      g2_h(v.z << 16) | (g2_h(v.z & 0xffff0000u) << 16), g2_h(v.w << 16) | (g2_h(v.w & 0xffff0000u) << 16));
}

// (gate, up) f16 (scale*2^18, bias) of sub-block j for this lane's row, packed math on both tiles at once:
// RN_f16(d*sc) and RN_f16(-dmin*mn) are the f32-then-round values of the gluk kernel (the exact products fit in 24
// bits), *2^18 is two exact multiplies (1024, 256). D2 = (d_g, d_u), M2 = (-dmin_g, -dmin_u).
__device__ __forceinline__ uint32_t g2_pk_mul(const uint32_t a, const uint32_t b) {
    uint32_t r; asm("v_pk_mul_f16 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r;
}
__device__ __forceinline__ uint32_t g2_pk_add(const uint32_t a, const uint32_t b) {
    uint32_t r; asm("v_pk_add_f16 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r;
}
// bytes i of a (low half) and of b (high half), each in the low byte of its half
__device__ __forceinline__ uint32_t g2_pair(const uint32_t a, const uint32_t b, const int i) {
    return __builtin_amdgcn_perm(b, a, 0x0c000c00u | ((uint32_t) (4 + i) << 16) | (uint32_t) i);
}
__device__ __forceinline__ g2_v8u g2_xhalf(const g2_v8u x) {
    g2_v8u y;
#pragma unroll
    for (int i = 0; i < 8; ++i) y[i] = __builtin_amdgcn_permlanex16(x[i], x[i], 0x76543210u, 0xfedcba98u, false, false);
    return y;
}

// sub-block j (uniform) of this lane's row
__device__ __forceinline__ void g2_sb2(const uint4 hg, const uint4 hu, const uint32_t D2, const uint32_t M2, const int j,
        uint32_t & Sg, uint32_t & Bg, uint32_t & Su, uint32_t & Bu) {
    const int jj = j & 3;
    const uint32_t sel = 0x0c000c00u | ((uint32_t) (4 + jj) << 16) | (uint32_t) jj;
    uint32_t sc, mn;
    if (j < 4) {
        sc = (__builtin_amdgcn_perm(hu.y, hg.y, sel) & 0x003F003Fu) | 0x64006400u;
        mn = (__builtin_amdgcn_perm(hu.z, hg.z, sel) & 0x003F003Fu) | 0x64006400u;
    } else {
        const uint32_t w = __builtin_amdgcn_perm(hu.w, hg.w, sel);
        sc = (w & 0x000F000Fu) | ((__builtin_amdgcn_perm(hu.y, hg.y, sel) >> 2) & 0x00300030u) | 0x64006400u;
        mn = ((w >> 4) & 0x000F000Fu) | ((__builtin_amdgcn_perm(hu.z, hg.z, sel) >> 2) & 0x00300030u) | 0x64006400u;
    }
    const uint32_t m1024 = 0xE400E400u, k1024 = 0x64006400u, k256 = 0x5C005C00u;
    uint32_t S = g2_pk_mul(D2, g2_pk_add(sc, m1024));
    S = g2_pk_mul(g2_pk_mul(S, k1024), k256);
    const uint32_t B = g2_pk_mul(M2, g2_pk_add(mn, m1024));
    Sg = __builtin_amdgcn_perm(S, S, 0x01000100u); Su = __builtin_amdgcn_perm(S, S, 0x03020302u);
    Bg = __builtin_amdgcn_perm(B, B, 0x01000100u); Bu = __builtin_amdgcn_perm(B, B, 0x03020302u);
}

// Block = (expert, 128 output columns, up to BN = 16 TN tokens); wave w owns output columns 16 w .. +15 of the block:
// its gate and up 16-row tiles (A, decoded in registers) x all live 16-token tiles (B, f16 in LDS, one 64-K group per
// chunk, double-buffered). silu(gate) * up in registers; bf16 output staged through LDS for 16 B row stores.
// Weights are the first WMMA operand: gfx11 reads operand-A row m only from lane m (m even) or lane 16 + m (m odd),
// so lane (r, h) loads and decodes piece (r & 1) ^ h of its row - the used lanes get piece 0 (K 0..15) in place and
// piece 1 (K 16..31) through one permlanex16, one decode per lane. Accumulator lane (n, h) holds weight rows 2 v + h,
// token n. TN = 4 keeps 176 VGPRs: two blocks per CU, so streaming small-expert tiles share CUs with WMMA-bound ones.
template <int K, int TN, int NW = 8>
__global__ void __launch_bounds__(NW * 32) __attribute__((amdgpu_waves_per_eu(TN <= 4 ? 8 : 1)))
mmb_g2_glu_kernel(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t eb, const uint16_t * __restrict__ X,
        const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const uint4 * __restrict__ descs,
        uint16_t * __restrict__ Dh, float * __restrict__ D, const int M) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3)
    NO_DEVICE_CODE;
#else
    constexpr int KC = 64, NSB = K / 256, NG = K / 64, BN = 16 * TN, NCH = K / KC, LSB = KC + 8;
    constexpr int NT = NW * 32, BM = NW * 16;   // threads, output columns per block
    constexpr int BPI = BN * KC / 8 / NT;
    constexpr int SST = BM + 8;
    constexpr int BSZ = BN * LSB;
    static_assert(K % 256 == 0 && BPI >= 1 && BN * KC % (8 * NT) == 0, "");
    static_assert(2 * BSZ >= BN * SST, "");
    __shared__ __align__(16) _Float16 Bs[2][BSZ];
    const uint4 dsc = descs[blockIdx.x];
    const int cnt = (int) dsc.w;
    if (cnt == 0) return;
    const int e = (int) dsc.x, cb = (int) dsc.y, r0 = (int) dsc.z;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, r = lane & 15, hf = lane >> 4;
    const size_t hoff = (size_t) e * eb + (size_t) (cb * NW + wave) * NSB * G2_TILE + r * 16;
    const size_t coff = hoff + 256 + ((r & 1) ^ hf) * 256;
    const uint8_t * Hg = Wg + hoff, * Hu = Wu + hoff;
    const uint8_t * Cg = Wg + coff, * Cu = Wu + coff;

    int bsrc[BPI], bdst[BPI];   // element offsets into X (-1: zero row)
#pragma unroll
    for (int i = 0; i < BPI; ++i) {
        const int c = tid + i * NT, tok = c / (KC / 8), kp = c % (KC / 8);
        const int sr = tok < cnt ? ids_src[r0 + tok] : -1;
        bsrc[i] = sr >= 0 ? sr * K + kp * 8 : -1;
        bdst[i] = tok * LSB + kp * 8;
    }
    uint4 breg[BPI];
    auto loadB = [&](int ch) {
#pragma unroll
        for (int i = 0; i < BPI; ++i) breg[i] = bsrc[i] >= 0 ? *(const uint4 *) (X + bsrc[i] + ch * KC) : make_uint4(0, 0, 0, 0);
    };
    auto storeB = [&](int buf) {
#pragma unroll
        for (int i = 0; i < BPI; ++i) *(uint4 *) (&Bs[buf][bdst[i]]) = g2_cvt8(breg[i]);
    };
    uint4 ac[1][2];   // next group's codes [gate, up]
    auto loadA = [&](uint4 (&a)[2], int gi) {
        const size_t off = (size_t) (gi >> 2) * G2_TILE + (gi & 3) * 512;   // 64-K group gi
        a[0] = *(const uint4 *) (Cg + off); a[1] = *(const uint4 *) (Cu + off);
    };
    v8f acc[2][TN];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    const int njt = (cnt + 15) >> 4;

    uint4 hn0 = *(const uint4 *) Hg, hn1 = *(const uint4 *) Hu;
    uint4 hc0 = hn0, hc1 = hn1;
    uint32_t D2 = 0, M2 = 0;
    loadB(0); storeB(0);
    loadA(ac[0], 0);
    __syncthreads();
#pragma unroll 2
    for (int gi = 0; gi < NG; ++gi) {
        const int gg = gi & 3;
        if (gg == 0) {
            hc0 = hn0; hc1 = hn1;
            if (gi + 4 < NG) { hn0 = *(const uint4 *) (Hg + (size_t) ((gi >> 2) + 1) * G2_TILE); hn1 = *(const uint4 *) (Hu + (size_t) ((gi >> 2) + 1) * G2_TILE); }
            D2 = __builtin_amdgcn_perm(hc1.x, hc0.x, 0x05040100u);
            M2 = __builtin_amdgcn_perm(hc1.x, hc0.x, 0x07060302u) ^ 0x80008000u;
        }
        if (gi + 1 < NCH) loadB(gi + 1);
        const uint4 cg = ac[0][0], cu = ac[0][1];
        if (gi + 1 < NG) loadA(ac[0], gi + 1);
        const _Float16 * bbase = &Bs[gi & 1][r * LSB];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint32_t Sg, Bg, Su, Bu;
            g2_sb2(hc0, hc1, D2, M2, 2 * gg + h, Sg, Bg, Su, Bu);
            const g2_v8u xg = h ? g2_dec<true>(cg, Sg, Bg) : g2_dec<false>(cg, Sg, Bg);
            const g2_v8u xu = h ? g2_dec<true>(cu, Su, Bu) : g2_dec<false>(cu, Su, Bu);
            const g2_v8u yg = g2_xhalf(xg), yu = g2_xhalf(xu);
            const _Float16 * bp = bbase + h * 32;
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                asm volatile("" ::: "memory");   // one token tile's B fragments live at a time
                if (j < njt) {
                    const g2_v16h b0 = *(const g2_v16h *) (bp + j * 16 * LSB);
                    const g2_v16h b1 = *(const g2_v16h *) (bp + j * 16 * LSB + 16);
                    asm volatile("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0" : "+v"(acc[0][j]) : "v"(xg), "v"(b0));
                    asm volatile("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0" : "+v"(acc[1][j]) : "v"(xu), "v"(b0));
                    asm volatile("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0" : "+v"(acc[0][j]) : "v"(yg), "v"(b1));
                    asm volatile("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0" : "+v"(acc[1][j]) : "v"(yu), "v"(b1));
                }
            }
        }
        if (gi + 1 < NCH) { storeB((gi + 1) & 1); __syncthreads(); }
    }
    const int col = cb * BM + wave * 16 + hf;   // + 2 v
    if (D) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int tok = j * 16 + r;
            if (tok < cnt) {
                const int dr = ids_dst[r0 + tok];
                if (dr >= 0) {
#pragma unroll
                    for (int v = 0; v < 8; ++v) D[(size_t) dr * M + col + 2 * v] = ggml_cuda_op_silu_single(acc[0][j][v]) * acc[1][j][v];
                }
            }
        }
    }
    if (Dh) {
        __syncthreads();
        uint16_t * stg = (uint16_t *) &Bs[0][0];
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v)
                stg[(j * 16 + r) * SST + wave * 16 + 2 * v + hf] = mmb_f2bf(ggml_cuda_op_silu_single(acc[0][j][v]) * acc[1][j][v]);
        __syncthreads();
        for (int idx = tid; idx < cnt * (BM / 8); idx += NT) {
            const int tok = idx / (BM / 8), part = idx % (BM / 8);
            const int dr = ids_dst[r0 + tok];
            if (dr >= 0) *(uint4 *) (Dh + (size_t) dr * M + cb * BM + part * 8) = *(const uint4 *) (stg + tok * SST + part * 8);
        }
    }
#endif
}

// Tile list: experts sorted by row count (desc), each expert's [128-column block][BN-token tile]s, column-block major.
// Full tiles (>= BN/2 tokens) and partial tiles are interleaved proportionally (the md4 order) so the weight-streaming
// partial tiles run alongside the WMMA-bound full tiles. Unused entries get cnt = 0. One block of 1024, E <= 1024.
__global__ void __launch_bounds__(1024) mmb_g2_build_desc(const int32_t * __restrict__ bounds, uint4 * __restrict__ desc, const int E,
        const int n_cb, const int BN, const int n_max) {
    __shared__ uint32_t key[1024];
    __shared__ int pf[1024], ps[1024];
    const int t = threadIdx.x;
    {
        const int c = t < E ? bounds[t + 1] - bounds[t] : 0;
        key[t] = t < E ? ((uint32_t) (0x3FFFFF - min(c, 0x3FFFFF)) << 10) | (uint32_t) t : 0xFFFFFFFFu;
    }
    __syncthreads();
    for (int k = 2; k <= 1024; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int ixj = t ^ j;
            if (ixj > t) {
                const uint32_t a = key[t], b = key[ixj];
                if ((a > b) == ((t & k) == 0)) { key[t] = b; key[ixj] = a; }
            }
            __syncthreads();
        }
    }
    auto ntiles = [&](int k, int & nf, int & np, int & e, int & c) {
        const uint32_t kk = key[k];
        e = kk == 0xFFFFFFFFu ? -1 : (int) (kk & 1023u);
        c = e >= 0 ? bounds[e + 1] - bounds[e] : 0;
        const int njt = (c + BN - 1) / BN;
        nf = c / BN + ((c % BN) >= BN / 2 ? 1 : 0); np = njt - nf;
    };
    {
        int nf, np, e, c; ntiles(t, nf, np, e, c);
        pf[t] = nf * n_cb; ps[t] = np * n_cb;
    }
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        const int a = t >= off ? pf[t - off] : 0, b = t >= off ? ps[t - off] : 0;
        __syncthreads();
        pf[t] += a; ps[t] += b;
        __syncthreads();
    }
    const int nB = pf[1023], nS = ps[1023];
    for (int p = t; p < nB; p += 1024) {
        int lo = 0, hi = 1023;
        while (lo < hi) { const int mid = (lo + hi) >> 1; if (pf[mid] > p) hi = mid; else lo = mid + 1; }
        int nf, np, e, c; ntiles(lo, nf, np, e, c);
        const int local = p - (lo ? pf[lo - 1] : 0), cb = local / nf, jt = local % nf;
        const int before = nB > 0 ? (int) (((long long) p * nS + nB - 1) / nB) : 0;
        const int pos = p + min(before, nS);
        desc[pos] = make_uint4((uint32_t) e, (uint32_t) cb, (uint32_t) (bounds[e] + jt * BN), (uint32_t) min(BN, c - jt * BN));
    }
    for (int q = t; q < nS; q += 1024) {
        int lo = 0, hi = 1023;
        while (lo < hi) { const int mid = (lo + hi) >> 1; if (ps[mid] > q) hi = mid; else lo = mid + 1; }
        int nf, np, e, c; ntiles(lo, nf, np, e, c);
        const int cb = q - (lo ? ps[lo - 1] : 0), jt = nf;   // the partial tile is the expert's last
        const int before = nS > 0 ? (int) ((long long) q * nB / nS) + 1 : 0;
        const int pos = q + min(before, nB);
        desc[pos] = make_uint4((uint32_t) e, (uint32_t) cb, (uint32_t) (bounds[e] + jt * BN), (uint32_t) (c - jt * BN));
    }
    for (int p = nB + nS + t; p < n_max; p += 1024) desc[p] = make_uint4(0, 0, 0, 0);
}

// f32 activations: GEMV over the G2 layout for every MUL_MAT_ID on a repacked tensor that the G2 WMMA kernel does
// not take (decode, small batches). One block per (16-row tile, routed row): 16 lane groups of 16 split the 4 K
// groups of every superblock, the activation row is staged in LDS. GLU: gate and up tiles, dst = silu(gate) * up.
template <bool GLU>
__global__ void __launch_bounds__(256)
mmb_g2_gemv_kernel(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t eb, const char * __restrict__ x,
        const size_t nb11, const size_t nb12, const int ne11, const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
        char * __restrict__ dst, const size_t nb1, const size_t nb2, const int M, const int E, const int K) {
    extern __shared__ float g2_xs[];
    __shared__ float red[2][16][17];
    const int ct = blockIdx.x, i = blockIdx.y, t = blockIdx.z;
    const float * xr = (const float *) (x + (size_t) t * nb12 + (size_t) (i % ne11) * nb11);
    for (int k = threadIdx.x; k < K; k += blockDim.x) g2_xs[k] = xr[k];
    __syncthreads();
    const int e = *(const int32_t *) (ids + (size_t) t * ids_nb1 + (size_t) i * ids_nb0);
    if (e < 0 || e >= E) return;
    const int r = threadIdx.x & 15, q = threadIdx.x >> 4;
    const int NSB = K / 256, NU = NSB * 4;
    const size_t aoff = (size_t) e * eb + (size_t) ct * NSB * G2_TILE + r * 16;
    auto unit = [&](const uint8_t * base, const int s, const int g) -> float {
        const uint4 h = *(const uint4 *) base;
        const uint4 c0 = *(const uint4 *) (base + 256 + g * 512), c1 = *(const uint4 *) (base + 512 + g * 512);
        const float d = (float) __builtin_bit_cast(_Float16, (uint16_t) (h.x & 0xffffu));
        const float dm = (float) __builtin_bit_cast(_Float16, (uint16_t) (h.x >> 16));
        const float * xb = g2_xs + s * 256 + g * 64;
        float sl = 0.0f, sh = 0.0f, xl = 0.0f, xh = 0.0f;
        auto word = [&](const uint32_t w, const int b0) {
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const uint32_t byte = (w >> (8 * b)) & 0xffu;
                const int k = b0 + (b == 1 ? 2 : b == 2 ? 1 : b);   // repacked byte order {0,2,1,3}
                const float a0 = xb[k], a1 = xb[32 + k];
                sl += (float) (byte & 15u) * a0; xl += a0;
                sh += (float) (byte >> 4) * a1; xh += a1;
            }
        };
        word(c0.x, 0); word(c0.y, 4); word(c0.z, 8); word(c0.w, 12);
        word(c1.x, 16); word(c1.y, 20); word(c1.z, 24); word(c1.w, 28);
        uint32_t sc0, mn0, sc1, mn1;
        g2_scmn(h, 2 * g, sc0, mn0); g2_scmn(h, 2 * g + 1, sc1, mn1);
        return d * ((float) sc0 * sl + (float) sc1 * sh) - dm * ((float) mn0 * xl + (float) mn1 * xh);
    };
    float accg = 0.0f, accu = 0.0f;
    for (int u = q; u < NU; u += 16) {
        const int s = u >> 2, g = u & 3;
        accg += unit(Wg + aoff + (size_t) s * G2_TILE, s, g);
        if (GLU) accu += unit(Wu + aoff + (size_t) s * G2_TILE, s, g);
    }
    red[0][q][r] = accg; red[1][q][r] = accu;
    __syncthreads();
    if (threadIdx.x < 16) {
        float a = 0.0f, b = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) { a += red[0][j][r]; b += red[1][j][r]; }
        *(float *) (dst + (size_t) t * nb2 + (size_t) i * nb1 + (size_t) (ct * 16 + r) * sizeof(float)) = GLU ? ggml_cuda_op_silu_single(a) * b : a;
    }
}

static bool mmb_r2_eligible(const ggml_tensor * w) {
    return w && w->type == GGML_TYPE_Q5_1 && w->op == GGML_OP_NONE && w->view_src == nullptr && w->data && w->buffer &&
           ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && ggml_is_contiguous(w) && w->ne[3] == 1 &&
           w->ne[0] == 640 && w->ne[1] % R2_BM == 0 && w->ne[2] >= 1 && w->ne[2] <= 1024 && w->nb[2] == (size_t) w->ne[1] * (w->ne[0] / 32) * 24;
}
// Layout record, process-wide (the weights outlive backend contexts): data pointer -> bytes for tensors rewritten in
// place to R2; data pointers that failed the scale check stay in Q5_1 and are not retried. Entries are dropped when
// their memory is freed or overwritten through the buffer interface (ggml_cuda_mmb_r2_forget).
static std::mutex g_r2_mutex;
static std::map<const char *, size_t> g_r2_packed;
static std::set<const char *> g_r2_failed;
// GLU R2 layout record (same lifetime rules as g_r2_packed): Q4_K gate/up expert tensors rewritten in place to G2
static std::map<const char *, size_t> g_g2_packed;
static std::set<const char *> g_g2_failed;
static bool mmb_g2_eligible(const ggml_tensor * w) {
    return w && w->type == GGML_TYPE_Q4_K && w->op == GGML_OP_NONE && w->view_src == nullptr && w->data && w->buffer &&
           ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && ggml_is_contiguous(w) && w->ne[3] == 1 &&
           (w->ne[0] == 2048 || w->ne[0] == 2560 || w->ne[0] == 4096) && w->ne[1] % G2_BM == 0 && w->ne[2] >= 1 && w->ne[2] <= 1024 &&
           w->nb[2] == (size_t) w->ne[1] * (w->ne[0] / 256) * 144;
}
} // namespace

bool ggml_cuda_mmb_r2_packed(const ggml_tensor * w) {   // any routed repacked layout (R2 Q5_1 down, G2 Q4_K gate/up)
    if (!w || !w->data || (w->type != GGML_TYPE_Q5_1 && w->type != GGML_TYPE_Q4_K)) return false;
    std::lock_guard<std::mutex> lock(g_r2_mutex);
    const auto & m = w->type == GGML_TYPE_Q5_1 ? g_r2_packed : g_g2_packed;
    return !m.empty() && m.count((const char *) w->data) > 0;
}

bool ggml_cuda_mmb_g2_packed(const ggml_tensor * w) {
    if (!w || w->type != GGML_TYPE_Q4_K || !w->data) return false;
    std::lock_guard<std::mutex> lock(g_r2_mutex);
    return !g_g2_packed.empty() && g_g2_packed.count((const char *) w->data) > 0;
}

void ggml_cuda_mmb_r2_forget(const void * base, const size_t size) {
    mmb_dq_forget(base, size);   // dense R2 records share the buffer hooks
    std::lock_guard<std::mutex> lock(g_r2_mutex);
    const char * lo = (const char *) base, * hi = lo + size;
    for (auto * packed : {&g_r2_packed, &g_g2_packed}) {
        for (auto it = packed->begin(); it != packed->end();) {
            if (it->first < hi && it->first + it->second > lo) {
                if (it->first < lo || it->first + it->second > hi) {
                    GGML_ABORT("a %zu-byte write covers part of a weight repacked in place (%s): partial writes to it are not supported",
                               size, packed == &g_r2_packed ? "down R2" : "GLU R2");
                }
                it = packed->erase(it);
            } else ++it;
        }
    }
    for (auto * failed : {&g_r2_failed, &g_g2_failed}) {
        for (auto it = failed->begin(); it != failed->end();) {
            if (*it >= lo && *it < hi) it = failed->erase(it); else ++it;
        }
    }
}

const char * ggml_cuda_mmb_r2_layout(const void * base, const size_t size) {
    if (mmb_dq_overlaps(base, size)) return "dense R2";
    std::lock_guard<std::mutex> lock(g_r2_mutex);
    const char * lo = (const char *) base, * hi = lo + size;
    for (const auto & p : g_r2_packed) if (p.first < hi && p.first + p.second > lo) return "down R2";
    for (const auto & p : g_g2_packed) if (p.first < hi && p.first + p.second > lo) return "GLU R2";
    return nullptr;
}

// In-place repack of a routed Q5_1 down weight [K=640, M, E] to the R2 layout (same size). Read-only scale check over
// the whole tensor first (a tensor with any d*2^19 outside f16 stays Q5_1), then experts are copied in chunks to a
// small temporary buffer and repacked from it back into the tensor's own memory.
void ggml_cuda_mmb_down_r2_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w) {
    if (!mmb_enabled(ctx) || !mmb_r2_eligible(w)) return;
    std::lock_guard<std::mutex> repack_lock(g_repack_mutex);
    const char * key = (const char *) w->data;
    {
        std::lock_guard<std::mutex> lock(g_r2_mutex);
        if (g_r2_packed.count(key) || g_r2_failed.count(key)) return;
    }
    const int M = (int) w->ne[1], NB = (int) (w->ne[0] / 32), E = (int) w->ne[2];
    const size_t eb = w->nb[2], bytes = (size_t) E * eb, nblk = bytes / 24;
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int> bad(ctx.pool(), 1);
    CUDA_CHECK(cudaMemsetAsync(bad.get(), 0, sizeof(int), stream));
    mmb_r2_check_kernel<<<(unsigned) ((nblk + 255) / 256), 256, 0, stream>>>((const uint8_t *) w->data, nblk, bad.get());
    CUDA_CHECK(cudaGetLastError());
    int hbad = 0;
    CUDA_CHECK(cudaMemcpyAsync(&hbad, bad.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (hbad) {
        GGML_LOG_WARN("mmb_down_r2: %s has a Q5_1 scale outside the f16 d*2^19 range; it keeps the Q5_1 layout\n", w->name);
        std::lock_guard<std::mutex> lock(g_r2_mutex);
        g_r2_failed.insert(key);
        return;
    }
    constexpr int CHUNK = 16;   // experts per temporary copy (16 x 1.2 MB)
    ggml_cuda_set_device(ctx.device);
    CUDA_CHECK(cudaDeviceSynchronize());
    {
        ggml_cuda_pool_alloc<uint8_t> tmp(ctx.pool(), (size_t) CHUNK * eb);
        for (int e0 = 0; e0 < E; e0 += CHUNK) {
            const int ne = std::min(CHUNK, E - e0);
            uint8_t * dst = (uint8_t *) w->data + (size_t) e0 * eb;
            CUDA_CHECK(cudaMemcpyAsync(tmp.get(), dst, (size_t) ne * eb, cudaMemcpyDeviceToDevice, stream));
            const size_t nb = (size_t) ne * M * NB;
            mmb_r2_repack_kernel<<<(unsigned) ((nb + 255) / 256), 256, 0, stream>>>(tmp.get(), dst, nb, M, NB);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::lock_guard<std::mutex> lock(g_r2_mutex);
    g_r2_packed[key] = bytes;
}

// R2 routed down (MMB path). Returns false (nothing launched) only for a tensor that is not repacked.
static bool mmb_down_r2_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const uint16_t * xhp, uint16_t * Dh, float * D,
        const int32_t * ids_src1, const int32_t * ids_dst, const int32_t * bounds, const int E, const int M, const int K,
        const int n_rows, cudaStream_t prep_stream, const bool xp) {
    if (!ggml_cuda_mmb_r2_packed(src0)) return false;
    GGML_ASSERT(K == 640 && M % R2_BM == 0 && E <= 1024 && (Dh || D));
    const int n_rt = M / R2_BM;
    const int n_max = n_rt * (n_rows / R2_BN + E + 1);
    ggml_cuda_pool_alloc<uint4> desc(ctx.pool(), n_max);
    mmb_r2_build_desc<<<1, 1024, 0, prep_stream>>>(bounds, desc.get(), E, n_rt, n_max);
    cudaStream_t stream = prep_stream;
    if (xp) {
        const cudaEvent_t eprep = mmb_event(mmb_state(ctx).prep_ev);
        CUDA_CHECK(cudaEventRecord(eprep, prep_stream));
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), eprep, 0));
        stream = ctx.stream();
    }
    mmb_r2_down_kernel<640><<<n_max, 256, 0, stream>>>((const uint8_t *) src0->data, (size_t) src0->nb[2], xhp, ids_src1, ids_dst, desc.get(), Dh, D, M);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

void ggml_cuda_mul_mat_id_r2_gemv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_mmb_r2_packed(src0) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(src0->ne[0] == 640 && src1->ne[0] == 640 && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));
    const int M = (int) src0->ne[1], E = (int) src0->ne[2];
    const int n_used = (int) ids->ne[0], T = (int) ids->ne[1];
    GGML_ASSERT(dst->ne[0] == M && dst->ne[1] == n_used && dst->ne[2] == T && T <= 65535 && n_used <= 65535);
    const dim3 grid((M / 16 + 3) / 4, n_used, T);
    mmb_r2_gemv_kernel<640><<<grid, 128, 0, ctx.stream()>>>((const uint8_t *) src0->data, (size_t) src0->nb[2], (const char *) src1->data,
        src1->nb[1], src1->nb[2], (int) src1->ne[1], (const char *) ids->data, ids->nb[0], ids->nb[1], (char *) dst->data, dst->nb[1], dst->nb[2], M, E);
    CUDA_CHECK(cudaGetLastError());
}

// GLU R2: in-place repack of a routed Q4_K gate/up pair [K, M, E] to the G2 layout (same size, no extra memory
// beyond a 16-expert temporary). Both tensors pass the scale check or neither is repacked.
void ggml_cuda_mmb_glu_r2_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw) {
    if (!mmb_enabled(ctx)) return;
    if (!mmb_g2_eligible(gw) || !mmb_g2_eligible(uw) || gw->data == uw->data || !ggml_are_same_shape(gw, uw) || gw->nb[2] != uw->nb[2]) return;
    std::lock_guard<std::mutex> repack_lock(g_repack_mutex);
    {
        std::lock_guard<std::mutex> lock(g_r2_mutex);
        const char * kg = (const char *) gw->data, * ku = (const char *) uw->data;
        if (g_g2_failed.count(kg) || g_g2_failed.count(ku)) return;
        if (g_g2_packed.count(kg) && g_g2_packed.count(ku)) return;
        GGML_ASSERT(!g_g2_packed.count(kg) && !g_g2_packed.count(ku));   // a pair is packed together
    }
    const int M = (int) gw->ne[1], NSB = (int) (gw->ne[0] / 256), E = (int) gw->ne[2];
    const size_t eb = gw->nb[2], bytes = (size_t) E * eb, n = bytes / 144;
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int> bad(ctx.pool(), 1);
    CUDA_CHECK(cudaMemsetAsync(bad.get(), 0, sizeof(int), stream));
    for (const ggml_tensor * w : {gw, uw}) {
        mmb_g2_check_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>((const uint8_t *) w->data, n, bad.get());
        CUDA_CHECK(cudaGetLastError());
    }
    int hbad = 0;
    CUDA_CHECK(cudaMemcpyAsync(&hbad, bad.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (hbad) {
        GGML_LOG_WARN("mmb_glu_r2: %s / %s have a Q4_K sub-block scale d*sc >= 0.25; they keep the Q4_K layout\n", gw->name, uw->name);
        std::lock_guard<std::mutex> lock(g_r2_mutex);
        g_g2_failed.insert((const char *) gw->data); g_g2_failed.insert((const char *) uw->data);
        return;
    }
    constexpr int CHUNK = 16;   // experts per temporary copy
    ggml_cuda_set_device(ctx.device);
    CUDA_CHECK(cudaDeviceSynchronize());
    {
        ggml_cuda_pool_alloc<uint8_t> tmp(ctx.pool(), (size_t) CHUNK * eb);
        for (const ggml_tensor * w : {gw, uw}) {
            for (int e0 = 0; e0 < E; e0 += CHUNK) {
                const int ne = std::min(CHUNK, E - e0);
                uint8_t * dst = (uint8_t *) w->data + (size_t) e0 * eb;
                CUDA_CHECK(cudaMemcpyAsync(tmp.get(), dst, (size_t) ne * eb, cudaMemcpyDeviceToDevice, stream));
                const size_t nb = (size_t) ne * eb / 144;
                mmb_g2_repack_kernel<<<(unsigned) ((nb + 255) / 256), 256, 0, stream>>>(tmp.get(), dst, nb, M, NSB);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::lock_guard<std::mutex> lock(g_r2_mutex);
    g_g2_packed[(const char *) gw->data] = bytes;
    g_g2_packed[(const char *) uw->data] = bytes;
}

// G2 routed gate/up + SwiGLU (MMB path): one launch over the device-built tile list, any T.
static void mmb_glu_r2_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw, const uint16_t * xhp,
        uint16_t * Dh, float * D, const int32_t * ids_src1, const int32_t * ids_dst, const int32_t * bounds, const int E, const int M,
        const int K, const int n_rows, cudaStream_t stream) {
    GGML_ASSERT(K % 256 == 0 && M % G2_BM == 0 && E <= 1024 && (Dh || D));
    constexpr int TN = 4;
    const int BN = 16 * TN, n_cb = M / G2_BM;
    const int n_max = n_cb * (n_rows / BN + E + 1);
    ggml_cuda_pool_alloc<uint4> desc(ctx.pool(), n_max);
    mmb_g2_build_desc<<<1, 1024, 0, stream>>>(bounds, desc.get(), E, n_cb, BN, n_max);
    const uint8_t * Wg = (const uint8_t *) gw->data, * Wu = (const uint8_t *) uw->data;
    const size_t eb = gw->nb[2];
    auto launch = [&](auto kt) {
        constexpr int KK = decltype(kt)::value;
        mmb_g2_glu_kernel<KK, TN><<<n_max, 256, 0, stream>>>(Wg, Wu, eb, xhp, ids_src1, ids_dst, desc.get(), Dh, D, M);
    };
    if (K == 2560) launch(std::integral_constant<int, 2560>{});
    else if (K == 2048) launch(std::integral_constant<int, 2048>{});
    else if (K == 4096) launch(std::integral_constant<int, 4096>{});
    else GGML_ABORT("mmb_glu_r2: unsupported K=%d", K);
    CUDA_CHECK(cudaGetLastError());
}

// f32 GEMV on G2-packed experts (decode / batches the MMB GLU path does not take). glu: fused silu(gate) * up.
static void mmb_g2_gemv_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);
    const int K = (int) gw->ne[0], M = (int) gw->ne[1], E = (int) gw->ne[2];
    GGML_ASSERT(src1->ne[0] == K && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) && K % 256 == 0 && M % 16 == 0);
    const int n_used = (int) ids->ne[0], T = (int) ids->ne[1];
    GGML_ASSERT(dst->ne[0] == M && dst->ne[1] == n_used && dst->ne[2] == T && T <= 65535 && n_used <= 65535);
    const dim3 grid(M / 16, n_used, T);
    const size_t shm = (size_t) K * sizeof(float);
    if (uw) mmb_g2_gemv_kernel<true><<<grid, 256, shm, ctx.stream()>>>((const uint8_t *) gw->data, (const uint8_t *) uw->data, gw->nb[2],
        (const char *) src1->data, src1->nb[1], src1->nb[2], (int) src1->ne[1], (const char *) ids->data, ids->nb[0], ids->nb[1],
        (char *) dst->data, dst->nb[1], dst->nb[2], M, E, K);
    else    mmb_g2_gemv_kernel<false><<<grid, 256, shm, ctx.stream()>>>((const uint8_t *) gw->data, (const uint8_t *) gw->data, gw->nb[2],
        (const char *) src1->data, src1->nb[1], src1->nb[2], (int) src1->ne[1], (const char *) ids->data, ids->nb[0], ids->nb[1],
        (char *) dst->data, dst->nb[1], dst->nb[2], M, E, K);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_id_g2_gemv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_mmb_g2_packed(src0));
    mmb_g2_gemv_launch(ctx, src0, nullptr, src1, ids, dst);
}

// decode fusion: gate/up MUL_MAT_ID + SWIGLU on a G2-packed pair
bool ggml_cuda_mmb_g2_glu_gemv_ok(const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * glu) {
    if (!ggml_cuda_mmb_g2_packed(gw) || !ggml_cuda_mmb_g2_packed(uw) || !src1 || !ids || !glu) return false;
    if (glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0) return false;
    if (glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu) || !glu->src[0] || !glu->src[1]) return false;
    if (glu->src[0]->src[0] != gw || glu->src[1]->src[0] != uw || glu->src[0]->src[1] != src1 || glu->src[1]->src[1] != src1 ||
        glu->src[0]->src[2] != ids || glu->src[1]->src[2] != ids) return false;
    if (!ggml_are_same_shape(gw, uw) || gw->nb[2] != uw->nb[2] || src1->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) return false;
    return ggml_are_same_shape(glu, glu->src[0]) && src1->nb[0] == sizeof(float) && src1->ne[0] == gw->ne[0];
}

void ggml_cuda_mul_mat_id_g2_gemv_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * glu) {
    mmb_g2_gemv_launch(ctx, gw, uw, src1, ids, glu);
}

void ggml_cuda_mul_mat_id_mmb(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    const bool xp = mmb_state(ctx).side_pending;
    cudaStream_t stream = xp ? ctx.stream(ctx.device, GGML_CUDA_MAX_STREAMS - 1) : ctx.stream();   // prep stream
    const int K = (int) src0->ne[0], M = (int) src0->ne[1], E = (int) src0->ne[2];
    const int ne11 = (int) src1->ne[1], T = (int) src1->ne[2], n_used = (int) ids->ne[0];
    const int n_rows_x = ne11 * T, n_rows = n_used * T;
    constexpr int BN = 128;

    const uint16_t * xhp = mmb_bf16_activation(ctx, src1, (size_t) n_rows_x * K, stream);

    const int si1  = (int) (ids->nb[1] / sizeof(int32_t));
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);
    const mmb_ids_record & rec = mmb_state(ctx).ids_rec;
    const bool reuse = !xp && rec.ids == ids->data && rec.glu == mmb_root(src1) && rec.glu_data == mmb_root(src1)->data && rec.T == T && rec.n_used == n_used && rec.E == E &&
        rec.si1 == si1 && ne11 == n_used && sis1 == n_used && ggml_cuda_mmb_r2_packed(src0);
    ggml_cuda_pool_alloc<int32_t> ids_src1_a(ctx.pool()), ids_dst_a(ctx.pool()), bounds_a(ctx.pool());
    const int32_t * ids_src1_p, * ids_dst_p, * bounds_p;
    if (reuse) {
        ids_src1_p = rec.ids_dst; ids_dst_p = rec.ids_dst; bounds_p = rec.bounds;
    } else {
        ids_src1_a.alloc(n_rows); ids_dst_a.alloc(n_rows); bounds_a.alloc(E + 1);
        if (!ggml_cuda_launch_mm_ids_bounded(ctx, (const int32_t *) ids->data, ids_src1_a.get(), ids_dst_a.get(), bounds_a.get(),
                E, T, n_used, ne11, si1, sis1, /*inverse=*/false, stream)) {
            ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1_a.get(), ids_dst_a.get(), bounds_a.get(),
                E, T, n_used, ne11, si1, sis1, /*write_inverse=*/false, stream);
        }
        ids_src1_p = ids_src1_a.get(); ids_dst_p = ids_dst_a.get(); bounds_p = bounds_a.get();
    }
    {   // down R2: Q5_1 down experts with a repacked shadow -> one interleaved launch (no big/small tier split)
        uint16_t * Dh_r2 = ggml_cuda_mmb_is_bf16_only(ctx, dst) ? (uint16_t *) dst->data : nullptr;
        if (mmb_down_r2_launch(ctx, src0, xhp, Dh_r2, Dh_r2 ? nullptr : (float *) dst->data, ids_src1_p, ids_dst_p, bounds_p, E, M, K, n_rows, stream, xp)) {
            mmb_join_pending(ctx);   // the pending side (small GLU tier) must be joined before the reduction
            mmb_hold_release(ctx);
            return;
        }
    }
    constexpr int BN_SMALL = 32;
    // THRESH=32 (was 128): more MoE experts use the wide BN tile during prefill.
    // Measured on qwen3.8-flash-next / gfx1151, -p 2048 -d 0,12000,32000,64000:
    // +3.6% d0, +4.7% d12k, +4.0% d32k, +1.9% d64k vs THRESH=128; byte-identical
    // output, test-backend-ops 29643/29643. The descriptor arrays are sized for it (no override).
    constexpr int THRESH = 32;
    // RT 2/3 use a 64-token big tile (half the padding waste on the ~40-row experts); the desc must be built with
    // the same width the kernel reads.
    // gfx1151 / qwen3.8-flash-next pp2048 d0, RT=0 -> 2: 1598 -> 1821 t/s (less logging this time, tighter tails):
    // the down stack (43 Q5_1 big experts in <128,128,32,64,39>) ran at 1 CTA/SM with 36 KB LDS; <128,64,32,32>
    // fits 2 CTAs/SM and wins x1.14 wall despite 2x the tile count. pp4096 1851, pp8192 1818, tg128 unchanged.
    static const int RT = getenv("MMB_ROUTED_TILE") ? atoi(getenv("MMB_ROUTED_TILE")) : 2;
    // RT 2/3 use a 64-token big tile (half the padding waste on the ~40-row experts); the desc must be built with
    // the same width the kernel reads. Measured: RT=0 stays best (the wider weight re-read outweighs the padding).
    const int BN_rt = (RT == 2 || RT == 3) ? 64 : BN;
    // RT 1/2/3 launch only the big+small tiers, so the desc must be 2-tier: mmb_build_desc3's mid tier would
    // hold experts no launched kernel consumes, silently dropping them from the down projection (empty output).
    // Only the RT=0 else-branch launches the mid tier.
    const bool dmid = RT != 1 && RT != 2 && RT != 3;
    const int nbig_max   = n_rows / 64 + E + 2;
    const int nsmall_max = E * ((THRESH + BN_SMALL - 1) / BN_SMALL) + 1;
    ggml_cuda_pool_alloc<uint32_t> desc_big(ctx.pool(), nbig_max);
    ggml_cuda_pool_alloc<uint32_t> desc_small(ctx.pool(), nsmall_max);
    ggml_cuda_pool_alloc<uint32_t> desc_mid(ctx.pool(), !dmid ? 1 : E + 1);
    if (dmid)
        mmb_build_desc3<<<1, 1024, 0, stream>>>(bounds_p, desc_small.get(), desc_mid.get(), desc_big.get(), E, nsmall_max, E + 1, nbig_max, BN_rt, BN_SMALL);
    else
        mmb_build_desc2<<<1, 1024, 0, stream>>>(bounds_p, desc_big.get(), desc_small.get(), E, nbig_max, nsmall_max, BN_rt, BN_SMALL, THRESH);

    if (xp) {
        const cudaEvent_t eprep = mmb_event(mmb_state(ctx).prep_ev);
        CUDA_CHECK(cudaEventRecord(eprep, stream));
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), eprep, 0));
        stream = ctx.stream();
    }
    const uint8_t * W = (const uint8_t *) src0->data; float * D = (float *) dst->data; const size_t eb = (size_t) src0->nb[2];
    uint16_t * Dh = ggml_cuda_mmb_is_bf16_only(ctx, dst) ? (uint16_t *) dst->data : nullptr;
    const bool store_f32 = Dh == nullptr;
    // small tile mapped WAVES_M=8 x WAVES_N=1 (<128,32,16,32>), not 4 x 2 (<128,32,32,16>): a <=16-token tile (the
    // common small-expert case) leaves 4 waves WMMA-idle under the old shape. Bit-identical: same (m,n), same K order.
    dim3 gbig((M + 127) / 128, nbig_max), gsmall((M + 127) / 128, nsmall_max);
    mmb_dispatch_quant(src0->type, [&](auto tag) {
        constexpr int WT = decltype(tag)::value;
    if (RT == 1) {
        const dim3 gb((M + 63) / 64, nbig_max), gs((M + 63) / 64, nsmall_max);
        mmb_routed_kernel<64, BN, 32, 32, WT><<<gb, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1_p, ids_dst_p, bounds_p, desc_big.get(), M, K);
        mmb_routed_kernel<64, BN_SMALL, 16, 16, WT><<<gs, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1_p, ids_dst_p, bounds_p, desc_small.get(), M, K);
    } else if (RT == 2) {
        const dim3 gb((M + 127) / 128, nbig_max), gs((M + 127) / 128, nsmall_max);
        mmb_tier_fork fork(ctx, 2);
        mmb_routed_kernel<128, BN_SMALL, 16, 32, WT><<<gs, MMB_NT, 0, fork.small()>>>(W, eb, xhp, D, Dh, store_f32, ids_src1_p, ids_dst_p, bounds_p, desc_small.get(), M, K); // 8x2 = full 16-wave coverage (was 8x1 = 8/16 idle)
        mmb_routed_kernel<128, 64, 32, 32, WT><<<gb, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1_p, ids_dst_p, bounds_p, desc_big.get(), M, K); // WAVES_M 4x WAVES_N 2, TM=2 TN=2 = full 128x64 coverage (was <32,16> = 32/64 tokens, half the tile lost)
    } else if (RT == 3) {
        const dim3 gb((M + 255) / 256, nbig_max), gs((M + 255) / 256, nsmall_max);
        mmb_routed_kernel<256, 64, 64, 32, WT><<<gb, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1_p, ids_dst_p, bounds_p, desc_big.get(), M, K);
        mmb_routed_kernel<256, BN_SMALL, 64, 16, WT><<<gs, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1_p, ids_dst_p, bounds_p, desc_small.get(), M, K);
    } else {
    if (dmid) mmb_routed_kernel<128, 64, 32, 32, WT><<<dim3((M + 127) / 128, E + 1), MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1_p, ids_dst_p, bounds_p, desc_mid.get(), M, K);
    mmb_routed_kernel<128, BN, 32, 64, WT><<<gbig, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1_p, ids_dst_p, bounds_p, desc_big.get(), M, K);
    mmb_routed_kernel<128, BN_SMALL, 16, 32, WT><<<gsmall, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1_p, ids_dst_p, bounds_p, desc_small.get(), M, K);
    }
    });
    mmb_join_pending(ctx);   // a path without fork: join the pending side now
    mmb_hold_release(ctx);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_mmb_supported_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * glu) {
    if (!mmb_enabled(ctx) || !mmb_glu() || !gw || !uw || !src1 || !ids || !glu) return false;
    if (!mmb_quant_type_routed(gw->type) || uw->type != gw->type) return false;
    if (!ggml_are_same_shape(gw, uw) || gw->nb[1] != uw->nb[1] || gw->nb[2] != uw->nb[2]) return false;
    if (glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0) return false;
    if (glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu) || !glu->src[0] || !glu->src[1]) return false;
    if (glu->src[0]->op != GGML_OP_MUL_MAT_ID || glu->src[1]->op != GGML_OP_MUL_MAT_ID) return false;
    if (glu->src[0]->src[0] != gw || glu->src[1]->src[0] != uw || glu->src[0]->src[1] != src1 || glu->src[1]->src[1] != src1 || glu->src[0]->src[2] != ids || glu->src[1]->src[2] != ids) return false;
    if (ggml_nelements(glu) != ggml_nelements(glu->src[0]) || glu->ne[0] != gw->ne[1]) return false;
    return ggml_cuda_mmb_supported_mmid(ctx, gw, src1, ids, glu->src[0]) && ggml_cuda_mmb_supported_mmid(ctx, uw, src1, ids, glu->src[1]);
}

void ggml_cuda_mul_mat_id_mmb_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * glu) {
    mmb_hold_release(ctx);   // stale holds (should be empty)
    cudaStream_t stream = ctx.stream();
    const int K = (int) gw->ne[0], M = (int) gw->ne[1], E = (int) gw->ne[2];
    const int ne11 = (int) src1->ne[1], T = (int) src1->ne[2], n_used = (int) ids->ne[0];
    const int n_rows_x = ne11 * T, n_rows = n_used * T;
    constexpr int BN_sel = 128;   // the big-tier token tile; nbig_cap and launch_big are sized for 64 / 128 / 256 only
    // gluk mode: gufo-style raw-code kPair GEMM for the routed GLU; desc built
    // at BN=128 token tiles for the big tier.
    // Gated at T>=128: it wins the prefill (pp2048 +40) but costs decode-width
    // rows (T=1..8) through tile padding (tg128 -0.7).
    constexpr int BN_gluk = 128;
    // gluk decodes Q4_K / Q5_K superblocks and Q5_1 only: the other routed types keep the legacy tiers
    const bool glk_this = T >= 128 && M % 2 == 0 && (gw->type == GGML_TYPE_Q4_K || gw->type == GGML_TYPE_Q5_K || gw->type == GGML_TYPE_Q5_1);   // the pair epilogue stores row pairs
    const int BN_rt = glk_this ? BN_gluk : BN_sel;
    const uint16_t * xhp = mmb_bf16_activation(ctx, src1, (size_t) n_rows_x * K, stream);
    auto & ids_src1 = mmb_hold_alloc<int32_t>(ctx, n_rows);
    auto & ids_dst = mmb_hold_alloc<int32_t>(ctx, n_rows);
    auto & bounds = mmb_hold_alloc<int32_t>(ctx, E + 1);
    const int si1  = (int) (ids->nb[1] / sizeof(int32_t));
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);
    if (!ggml_cuda_launch_mm_ids_bounded(ctx, (const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            E, T, n_used, ne11, si1, sis1, /*inverse=*/false, stream)) {
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            E, T, n_used, ne11, si1, sis1, /*write_inverse=*/false, stream);
    }
    if (ne11 == 1 && sis1 == 1) {
        mmb_state(ctx).ids_rec = mmb_ids_record{ids->data, glu, glu->data, T, n_used, E, si1, ids_dst.get(), bounds.get()};
    }
    if (ggml_cuda_mmb_g2_packed(gw) && ggml_cuda_mmb_g2_packed(uw)) {   // GLU R2: one launch, no tier split / side stream
        uint16_t * Dh = ggml_cuda_mmb_slot_reserve(ctx, 2, glu, (size_t) n_rows * M);
        const bool store_f32 = !ggml_cuda_mmb_is_bf16_only(ctx, glu);
        mmb_glu_r2_launch(ctx, gw, uw, xhp, Dh, store_f32 ? (float *) glu->data : nullptr, ids_src1.get(), ids_dst.get(), bounds.get(), E, M, K, n_rows, stream);
        return;
    }
    constexpr int BN_SMALL = 32;
    // THRESH=32 (was 128): more MoE experts use the wide BN tile during prefill.
    // Measured on qwen3.8-flash-next / gfx1151, -p 2048 -d 0,12000,32000,64000:
    // +3.6% d0, +4.7% d12k, +4.0% d32k, +1.9% d64k vs THRESH=128; byte-identical
    // output, test-backend-ops 29643/29643. The descriptor arrays are sized for it (no override).
    constexpr int THRESH = 32;
    const int nbig_cap   = glk_this ? n_rows / BN_gluk + E + 2 : n_rows / 64 + E + 2;
    const int nbig_max   = nbig_cap;   // legacy name used by the non-gluk branches
    const int nsmall_max = E * ((THRESH + 31) / 32) + 1;   // small experts take ceil(cnt / 32) tiles
    auto & desc_big = mmb_hold_alloc<uint32_t>(ctx, nbig_cap);
    auto & desc_small = mmb_hold_alloc<uint32_t>(ctx, nsmall_max);
    auto & desc_mid = mmb_hold_alloc<uint32_t>(ctx, glk_this ? 1 : E + 1);
    if (glk_this)
        mmb_build_desc2<<<1, 1024, 0, stream>>>(bounds.get(), desc_big.get(), desc_small.get(), E, nbig_cap, nsmall_max, BN_rt, BN_SMALL, THRESH);
    else
        mmb_build_desc3<<<1, 1024, 0, stream>>>(bounds.get(), desc_small.get(), desc_mid.get(), desc_big.get(), E, nsmall_max, E + 1, nbig_cap, BN_rt, BN_SMALL);
    uint16_t * Dh = ggml_cuda_mmb_slot_reserve(ctx, 2, glu, (size_t) n_rows * M);
    const bool store_f32 = !ggml_cuda_mmb_is_bf16_only(ctx, glu);
    const uint8_t * Wg = (const uint8_t *) gw->data, * Wu = (const uint8_t *) uw->data; float * D = (float *) glu->data; const size_t eb = (size_t) gw->nb[2];
    // MMB_BM_BIG / MMB_BM_SMALL widen the M tile (n_ff_exp rows per block) to halve the M-tile count and the
    // B-tile reloads at small rows/expert.  BM=128 needs a matching (WTM,WTN) to keep the 8 waves full.
    static const int BM_BIG   = getenv("MMB_BM_BIG")   ? atoi(getenv("MMB_BM_BIG"))   : 64;
    static const int BM_SMALL = getenv("MMB_BM_SMALL") ? atoi(getenv("MMB_BM_SMALL")) : 64;
    // gfx1151 / ROCm 10, Qwen3.8 Q4_K pp4096: routed GLU 510 -> 476 ms in a GPU trace.
    // Retest if quant decode or lane layout changes; MMB_Q4K_HALF=0 restores one lane per row.
    static const bool q4k_half = getenv("MMB_Q4K_HALF") ? atoi(getenv("MMB_Q4K_HALF")) != 0 : true;
    // gluk (GK) entirely replaces the big/mid/small legacy kernels for the K-block quant types:
    // its desc_big covers every expert >= THRESH and desc_small every expert below, so the mid
    // tier (whose desc is never built in gluk mode - pool garbage as faults) and the legacy
    // small kernel (duplicate desc_small consumers) must not launch.
    mmb_dispatch_quant(gw->type, [&](auto tag) {
        constexpr int WT = decltype(tag)::value;
        if constexpr (WT < 32 || mmb_quant_type_routed_supported(static_cast<ggml_type>(WT - 32))) {
            if (glk_this) {
                // joined here: an op between this GLU and the down (a router-weight MUL, a down off MMB) reads the output
                mmb_tier_fork fork(ctx);
                {
                    const dim3 gs((M + 63) / 64, nsmall_max);
                    mmb_routed_gluk_kernel<WT, 32><<<gs, MMB_NT, 0, fork.small()>>>(Wg, Wu, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), eb, (size_t) M, (size_t) K);
                }
                {
                    const dim3 gb((M + 63) / 64, nbig_cap);
                    mmb_routed_gluk_kernel<WT, BN_gluk><<<gb, MMB_NT, 0, stream>>>(Wg, Wu, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), eb, (size_t) M, (size_t) K);
                }
                CUDA_CHECK(cudaGetLastError());
                return;
            }
            // 128 x 64 spills on gfx1151 (CI VGPR check): a 64-token selection takes the 64-row tiles below
            if (BM_BIG == 128 && BN_sel != 256 && BN_sel != 64) {
                const dim3 gbig((M + 127) / 128, nbig_max);
                mmb_routed_glu_kernel<128, 128, 64, 32, WT><<<gbig, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
            } else {
                auto launch_big = [&](auto bn_tag) {
                    constexpr int BN = decltype(bn_tag)::value;
                    const dim3 gbig((M + 63) / 64, nbig_max);
                    if constexpr (WT == 32 + GGML_TYPE_Q4_K && BN == 128) {
                        if (q4k_half) {
                            mmb_routed_glu_kernel<64, BN, 32, BN / 4, WT, true><<<gbig, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
                        } else {
                            mmb_routed_glu_kernel<64, BN, 32, BN / 4, WT><<<gbig, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
                        }
                    } else if constexpr (BN == 64) {
                        mmb_routed_glu_kernel<64, 64, 16, 32, WT><<<gbig, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
                    } else {
                        mmb_routed_glu_kernel<64, BN, 32, BN / 4, WT><<<gbig, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
                    }
                };
                if (BN_sel == 256) launch_big(std::integral_constant<int,256>{});
                else if (BN_sel == 64) launch_big(std::integral_constant<int,64>{});
                else               launch_big(std::integral_constant<int,128>{});
            }
            if (!glk_this) mmb_routed_glu_kernel<64, 64, 16, 32, WT><<<dim3((M + 63) / 64, E + 1), MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_mid.get(), M, K);
            if (glk_this) return;
            if (BM_SMALL == 128) {
                const dim3 gsmall((M + 127) / 128, nsmall_max);
                mmb_routed_glu_kernel<128, BN_SMALL, 32, 16, WT><<<gsmall, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
            } else {
                const dim3 gsmall((M + 63) / 64, nsmall_max);
                if constexpr (WT == 32 + GGML_TYPE_Q4_K) {
                    if (q4k_half) {
                        mmb_routed_glu_kernel<64, BN_SMALL, 16, 16, WT, true><<<gsmall, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
                    } else {
                        mmb_routed_glu_kernel<64, BN_SMALL, 16, 16, WT><<<gsmall, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
                    }
                } else {
                    mmb_routed_glu_kernel<64, BN_SMALL, 16, 16, WT><<<gsmall, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
                }
            }
        }
    });
    CUDA_CHECK(cudaGetLastError());
}

// Called from graph_optimize (outside stream capture): create the shadow for an eligible IQ4_NL dense weight.
void ggml_cuda_mmb_shadow_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w) {
    if (!w) return;
    if (mmb_is_f16hi_f32(w)) {
        if (mmb_state(ctx).shadow.count(w->data) > 0) return;
        const size_t n = (size_t) w->ne[0] * w->ne[1];
        if (mmb_state(ctx).shadow_bytes + n * 2 > mmb_shadow_cap()) return;
        uint16_t * buf = nullptr;
        if (cudaMalloc((void **) &buf, n * 2) != cudaSuccess) { GGML_LOG_WARN("MMB_SHADOW alloc failed (%zu bytes)\n", n * 2); return; }
        const int64_t nb = (int64_t) n;
        mmb_f32_to_f16_kernel<<<(unsigned) ((nb + 255) / 256), 256, 0, ctx.stream()>>>((const float *) w->data, buf, nb);
        CUDA_CHECK(cudaGetLastError());
        mmb_state(ctx).shadow[w->data] = buf; mmb_state(ctx).shadow_bytes += n * 2;
        return;
    }
    if (mmb_is_hc_q8(w)) {
        if (mmb_state(ctx).shadow.count(w->data) > 0) return;
        const size_t n = (size_t) w->ne[0] * w->ne[1], bytes = n * 2;
        if (mmb_state(ctx).shadow_bytes + bytes > mmb_shadow_cap()) return;
        uint16_t * buf = nullptr;
        if (cudaMalloc((void **) &buf, bytes) != cudaSuccess) { GGML_LOG_WARN("MMB_SHADOW alloc failed (%zu bytes)\n", bytes); return; }
        mmb_dq_q8_0_bf16_kernel<<<(unsigned) ((n / 32 + 255) / 256), 256, 0, ctx.stream()>>>((const uint8_t *) w->data, buf, n / 32);
        CUDA_CHECK(cudaGetLastError());
        mmb_state(ctx).shadow[w->data] = buf; mmb_state(ctx).shadow_bytes += bytes;
        return;
    }
    if (mmb_is_resident_q6k(w)) {
        if (!mmb_shadow_q6k() || mmb_state(ctx).shadow.count(w->data) > 0) return;
        const size_t n = (size_t) w->ne[0] * w->ne[1], bytes = n * 2;
        if (mmb_state(ctx).shadow_bytes + bytes > mmb_shadow_cap()) { GGML_LOG_INFO("MMB_SHADOW cap reached; %s stays Q6_K\n", w->name); return; }
        uint16_t * buf = nullptr;
        if (cudaMalloc((void **) &buf, bytes) != cudaSuccess) { GGML_LOG_WARN("MMB_SHADOW alloc failed (%zu bytes)\n", bytes); return; }
        mmb_dq_q6k_bf16_kernel<<<(unsigned) ((n / 256 + 255) / 256), 256, 0, ctx.stream()>>>((const uint8_t *) w->data, buf, n / 256);
        CUDA_CHECK(cudaGetLastError());
        mmb_state(ctx).shadow[w->data] = buf; mmb_state(ctx).shadow_bytes += bytes;
        return;
    }
    if (mmb_shadow_mode() != 1) return;               // mode 2: Q6_K only
    const bool concat = mmb_is_row_concat(w);
    if (!concat && !mmb_is_resident_iq4(w)) return;
    if (concat ? mmb_state(ctx).shadow_pair.count({w->src[0]->data, w->src[1]->data}) > 0 : mmb_state(ctx).shadow.count(w->data) > 0) return;
    const size_t n = (size_t) w->ne[0] * w->ne[1];
    const size_t bytes = n * 2;
    if (mmb_state(ctx).shadow_bytes + bytes > mmb_shadow_cap()) { static bool warned = false; if (!warned) { GGML_LOG_INFO("MMB_SHADOW cap reached at %.1f MB; further weights stay IQ4_NL\n", mmb_state(ctx).shadow_bytes / 1048576.0); warned = true; } return; }
    uint16_t * buf = nullptr;
    if (cudaMalloc((void **) &buf, bytes) != cudaSuccess) { GGML_LOG_WARN("MMB_SHADOW alloc failed (%zu bytes)\n", bytes); return; }
    if (concat) {
        const size_t n0 = (size_t) w->src[0]->ne[0] * w->src[0]->ne[1], n1 = (size_t) w->src[1]->ne[0] * w->src[1]->ne[1];
        mmb_dq_iq4nl_bf16_kernel<<<(unsigned) ((n0 / 32 + 255) / 256), 256, 0, ctx.stream()>>>((const uint8_t *) w->src[0]->data, buf, n0 / 32);
        mmb_dq_iq4nl_bf16_kernel<<<(unsigned) ((n1 / 32 + 255) / 256), 256, 0, ctx.stream()>>>((const uint8_t *) w->src[1]->data, buf + n0, n1 / 32);
        mmb_state(ctx).shadow_pair[{w->src[0]->data, w->src[1]->data}] = buf;
    } else {
        mmb_dq_iq4nl_bf16_kernel<<<(unsigned) ((n / 32 + 255) / 256), 256, 0, ctx.stream()>>>((const uint8_t *) w->data, buf, n / 32);
        mmb_state(ctx).shadow[w->data] = buf;
    }
    CUDA_CHECK(cudaGetLastError());
    mmb_state(ctx).shadow_bytes += bytes;
}

// Dense R2: in-place repack of a dense Q8_0 weight to the tile layout of mmb_tq8_dense_kernel. Each 128-row tile
// occupies the same bytes in both layouts, so tiles are staged through a small temporary buffer and rewritten in
// place. Only weights whose every reader in the graph is a MUL_MAT src0 are passed here (ggml-cuda.cu graph_optimize).
void ggml_cuda_mmb_dense_r2_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w) {
    if (!mmb_enabled(ctx)) return;
    // the token embedding is read by GET_ROWS in other graphs (and is the LM head when tied): the in-place layout would
    // be read as Q8_0 rows there, and the graph-local reader scan cannot see those graphs
    if (w && strstr(w->name, "token_embd") != nullptr) return;
    if (!w || w->type != GGML_TYPE_Q8_0 || w->op != GGML_OP_NONE || w->view_src != nullptr || !w->data || !w->buffer ||
            ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS || !ggml_is_contiguous(w) ||
            w->ne[2] != 1 || w->ne[3] != 1 || !mmb_tq8_shape(w->ne[1], w->ne[0])) return;
    std::lock_guard<std::mutex> repack_lock(g_repack_mutex);
    const char * key = (const char *) w->data;
    if (ggml_cuda_mmb_dq_packed(w)) return;
    ggml_cuda_set_device(ctx.device);
    CUDA_CHECK(cudaDeviceSynchronize());
    const int K = (int) w->ne[0], M = (int) w->ne[1];
    const size_t tile = (size_t) (K / 64) * MMB_TQ8_CHUNK, bytes = (size_t) (M / 128) * tile;
    GGML_ASSERT(bytes == ggml_nbytes(w) && tile == (size_t) 128 * (K / 32) * 34);
    cudaStream_t stream = ctx.stream();
    const int G = std::max(1, (int) ((4u << 20) / tile));   // tiles per staging copy (<= ~4 MB)
    {
        ggml_cuda_pool_alloc<uint8_t> tmp(ctx.pool(), (size_t) G * tile);
        for (int t0 = 0; t0 < M / 128; t0 += G) {
            const int nt = std::min(G, M / 128 - t0);
            uint8_t * dst = (uint8_t *) w->data + (size_t) t0 * tile;
            CUDA_CHECK(cudaMemcpyAsync(tmp.get(), dst, (size_t) nt * tile, cudaMemcpyDeviceToDevice, stream));
            const size_t nb = (size_t) nt * 128 * (K / 32);
            mmb_tq8_repack_kernel<<<(unsigned) ((nb + 255) / 256), 256, 0, stream>>>(tmp.get(), dst, nt * 128, K);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::lock_guard<std::mutex> lock(g_dq_mutex);
    g_dq_packed[key] = bytes;
    g_dq_count.store((int) g_dq_packed.size());
}
