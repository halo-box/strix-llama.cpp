#pragma once
#include "dequantize.cuh"
#include <type_traits>

struct mmb_quant_slice {
    uint16_t * dst;
    int begin;
    int offset = 0;
    struct element {
        uint16_t * dst;
        int index;
        __device__ void operator=(float value) const {
            if (index >= 0 && index < 64) dst[index] = mmb_f2bf(value);
        }
    };
    __device__ mmb_quant_slice operator+(int64_t n) const { return {dst, begin, offset + (int) n}; }
    __device__ element operator[](int64_t n) const { return {dst, offset + (int) n - begin}; }
};

__device__ __forceinline__ void mmb_store8(uint16_t * dst, const float * v) {
    uint4 o; o.x = mmb_pack2(v[0], v[1]); o.y = mmb_pack2(v[2], v[3]); o.z = mmb_pack2(v[4], v[5]); o.w = mmb_pack2(v[6], v[7]);
    *(uint4 *) dst = o;
}

// 8 weights of an IQ2/IQ3_XXS grid group: fp32 d * grid * sign as in dequantize_*, RNE to bf16, one 16-byte store
__device__ __forceinline__ void mmb_store_grid8(uint16_t * dst, const uint64_t grid, const float d, const uint32_t signs) {
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = d * (float)((grid >> (8 * j)) & 0xff) * (signs & (1u << j) ? -1.f : 1.f);
    mmb_store8(dst, v);
}

// Routed GLU kernel, grid-based types: fetch() loads the block fields of a lane (8 weights) into 2 uint32 in the WMMA loop.
// decode() makes the 8 bf16 weights from them and the LDS grid, same math as dequantize_*, signs as XOR on the bf16 sign bit.
__device__ __forceinline__ uint4 mmb_grid8_bf16(const uint32_t lo, const uint32_t hi, const float d, const uint32_t sg) {
    auto rb = [](const float x) { const uint32_t u = __float_as_uint(x); return u + 0x7fffu + ((u >> 16) & 1u); };
    auto pk = [&](const uint32_t g, const int j) {
        return __builtin_amdgcn_perm(rb(d * (float)((g >> (8 * j + 8)) & 0xff)), rb(d * (float)((g >> (8 * j)) & 0xff)), 0x07060302u); };
    uint4 o;
    o.x = pk(lo, 0) ^ ((sg &  1u) << 15 | (sg &   2u) << 30);
    o.y = pk(lo, 2) ^ ((sg &  4u) << 13 | (sg &   8u) << 28);
    o.z = pk(hi, 0) ^ ((sg & 16u) << 11 | (sg &  32u) << 26);
    o.w = pk(hi, 2) ^ ((sg & 64u) <<  9 | (sg & 128u) << 24);
    return o;
}

// ksigns_iq2xs[i] without the table: 7 sign bits plus an even-parity bit 7
__device__ __forceinline__ uint32_t mmb_ksigns(const uint32_t i) { return i | ((__popc(i) & 1u) << 7); }

template <int WTYPE> struct mmb_lb { static constexpr bool ok = false; using grid_t = uint32_t; static constexpr int N = 1; };

template <> struct mmb_lb<32 + GGML_TYPE_IQ3_S> {
    static constexpr bool ok = true; using grid_t = uint32_t; static constexpr int N = 512;
    static __device__ __forceinline__ grid_t entry(const int i) { return iq3s_grid[i]; }
    static __device__ __forceinline__ void fetch(const uint8_t * row, const int ks, const int sub, const int il, uint32_t & w0, uint32_t & w1) {
        const block_iq3_s * x = (const block_iq3_s *) row + (ks * 64) / QK_K;
        const int ib = ((ks * 64) % QK_K) / 32 + sub;
        w0 = *(const uint16_t *)(x->qs + 8 * ib + 2 * il) | ((uint32_t) x->qh[ib] << 16) | ((uint32_t) x->signs[4 * ib + il] << 24);
        w1 = (uint32_t) *(const uint16_t *) &x->d | ((uint32_t) ((x->scales[ib / 2] >> 4 * (ib % 2)) & 0xf) << 16);
    }
    static __device__ __forceinline__ uint4 decode(const grid_t * g, const uint32_t w0, const uint32_t w1, const int il) {
        const uint32_t qh = (w0 >> 16) & 0xff;
        return mmb_grid8_bf16(g[(w0 & 0xff) | ((qh << (8 - 2 * il)) & 256)], g[((w0 >> 8) & 0xff) | ((qh << (7 - 2 * il)) & 256)],
            mmb_h2f((uint16_t) w1) * (1 + 2 * (int)(w1 >> 16)), w0 >> 24);
    }
};
template <> struct mmb_lb<32 + GGML_TYPE_IQ2_XXS> {
    static constexpr bool ok = true; using grid_t = uint64_t; static constexpr int N = 256;
    static __device__ __forceinline__ grid_t entry(const int i) { return iq2xxs_grid[i]; }
    static __device__ __forceinline__ void fetch(const uint8_t * row, const int ks, const int sub, const int il, uint32_t & w0, uint32_t & w1) {
        const block_iq2_xxs * x = (const block_iq2_xxs *) row + (ks * 64) / QK_K;
        const uint16_t * q2 = x->qs + 4 * (((ks * 64) % QK_K) / 32 + sub);
        w0 = q2[2] | ((uint32_t) q2[3] << 16);
        w1 = (uint32_t) *(const uint16_t *) &x->d | ((uint32_t) ((const uint8_t *) q2)[il] << 16);
    }
    static __device__ __forceinline__ uint4 decode(const grid_t * g, const uint32_t w0, const uint32_t w1, const int il) {
        const uint64_t grid = g[w1 >> 16];
        return mmb_grid8_bf16((uint32_t) grid, (uint32_t) (grid >> 32), mmb_h2f((uint16_t) w1) * (0.5f + (w0 >> 28)) * 0.25f, mmb_ksigns((w0 >> 7 * il) & 127));
    }
};
template <> struct mmb_lb<32 + GGML_TYPE_IQ2_XS> {
    static constexpr bool ok = true; using grid_t = uint64_t; static constexpr int N = 512;
    static __device__ __forceinline__ grid_t entry(const int i) { return iq2xs_grid[i]; }
    static __device__ __forceinline__ void fetch(const uint8_t * row, const int ks, const int sub, const int il, uint32_t & w0, uint32_t & w1) {
        const block_iq2_xs * x = (const block_iq2_xs *) row + (ks * 64) / QK_K;
        const int ib = ((ks * 64) % QK_K) / 32 + sub;
        w0 = x->qs[4 * ib + il] | ((uint32_t) ((x->scales[ib] >> 4 * (il / 2)) & 0xf) << 16);
        w1 = *(const uint16_t *) &x->d;
    }
    static __device__ __forceinline__ uint4 decode(const grid_t * g, const uint32_t w0, const uint32_t w1, const int il) {
        GGML_UNUSED(il);
        const uint64_t grid = g[w0 & 511];
        return mmb_grid8_bf16((uint32_t) grid, (uint32_t) (grid >> 32), mmb_h2f((uint16_t) w1) * (0.5f + (w0 >> 16)) * 0.25f, mmb_ksigns((w0 >> 9) & 127));
    }
};
// IQ2_S: all grid bytes are 8, 25 or 43, so the LDS table holds 2-bit codes (2 KB instead of 8 KB keeps 3 blocks per WGP)
template <> struct mmb_lb<32 + GGML_TYPE_IQ2_S> {
    static constexpr bool ok = true; using grid_t = uint16_t; static constexpr int N = 1024;
    static __device__ __forceinline__ grid_t entry(const int i) {
        const uint64_t g = iq2s_grid[i]; uint32_t c = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) { const uint32_t b = (g >> (8 * j)) & 0xff; c |= ((uint32_t) (b > 8) + (uint32_t) (b > 25)) << (2 * j); }
        return (grid_t) c;
    }
    static __device__ __forceinline__ void fetch(const uint8_t * row, const int ks, const int sub, const int il, uint32_t & w0, uint32_t & w1) {
        const block_iq2_s * x = (const block_iq2_s *) row + (ks * 64) / QK_K;
        const int ib = ((ks * 64) % QK_K) / 32 + sub;
        w0 = (x->qs[4 * ib + il] | ((x->qh[ib] << (8 - 2 * il)) & 0x300)) | ((uint32_t) x->qs[QK_K / 8 + 4 * ib + il] << 16)
           | ((uint32_t) ((x->scales[ib] >> 4 * (il / 2)) & 0xf) << 24);
        w1 = *(const uint16_t *) &x->d;
    }
    static __device__ __forceinline__ uint32_t sel4(const uint32_t c8) {   // 4 two-bit codes -> one per byte
        const uint32_t t = (c8 | (c8 << 12)) & 0x000f000fu;
        return (t | (t << 6)) & 0x03030303u;
    }
    static __device__ __forceinline__ uint4 decode(const grid_t * g, const uint32_t w0, const uint32_t w1, const int il) {
        GGML_UNUSED(il);
        const uint32_t c = g[w0 & 0x3ff];
        return mmb_grid8_bf16(__builtin_amdgcn_perm(0u, 0x002b1908u, sel4(c & 0xff)), __builtin_amdgcn_perm(0u, 0x002b1908u, sel4(c >> 8)),
            mmb_h2f((uint16_t) w1) * (0.5f + (w0 >> 24)) * 0.25f, (w0 >> 16) & 0xff);
    }
};
template <> struct mmb_lb<32 + GGML_TYPE_IQ3_XXS> {
    static constexpr bool ok = true; using grid_t = uint32_t; static constexpr int N = 256;
    static __device__ __forceinline__ grid_t entry(const int i) { return iq3xxs_grid[i]; }
    static __device__ __forceinline__ void fetch(const uint8_t * row, const int ks, const int sub, const int il, uint32_t & w0, uint32_t & w1) {
        const block_iq3_xxs * x = (const block_iq3_xxs *) row + (ks * 64) / QK_K;
        const int ib = ((ks * 64) % QK_K) / 32 + sub;
        const uint16_t * gas = (const uint16_t *) (x->qs + QK_K / 4) + 2 * ib;
        w0 = gas[0] | ((uint32_t) gas[1] << 16);
        w1 = (uint32_t) *(const uint16_t *) &x->d | ((uint32_t) *(const uint16_t *) (x->qs + 8 * ib + 2 * il) << 16);
    }
    static __device__ __forceinline__ uint4 decode(const grid_t * g, const uint32_t w0, const uint32_t w1, const int il) {
        return mmb_grid8_bf16(g[(w1 >> 16) & 0xff], g[w1 >> 24], mmb_h2f((uint16_t) w1) * (0.5f + (w0 >> 28)) * 0.5f, mmb_ksigns((w0 >> 7 * il) & 127));
    }
};

template <ggml_type TYPE>
__device__ __forceinline__ void mmb_decode_slice(const void * row, const int k0, uint16_t * dst, const int lane) {
    constexpr int QK = ggml_cuda_type_traits<TYPE>::qk;
    if constexpr (TYPE == GGML_TYPE_Q1_0) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q1_0(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q2_0) {
        // QK = 64: the slice is one block, lane -> 8 consecutive weights (2 bytes of qs), (code - 1) * d as dequantize_q2_0
        static_assert(QK == 64, "Q2_0 slice decode assumes one block per 64-wide K slice");
        const block_q2_0 * x = (const block_q2_0 *) row + k0 / QK;
        const float d = x->d;
        const uint32_t q = x->qs[2 * lane] | (x->qs[2 * lane + 1] << 8);
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = ((int) ((q >> (2 * j)) & 3) - 1) * d;
        mmb_store8(dst + 8 * lane, v);
    }
    else if constexpr (TYPE == GGML_TYPE_Q4_0) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q4_0(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q4_1) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q4_1(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q5_0) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q5_0(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q5_1) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q5_1(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q8_0) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q8_0(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q2_K) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 64; tid += 8) dequantize_q2_K<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_Q3_K) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 64; tid += 8) dequantize_q3_K<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_Q4_K) {
        const mmb_quant_slice out{dst, k0 % QK};
        const int tid = (k0 % QK) / 64 * 8 + lane;
        dequantize_q4_K<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_Q5_K) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 64; tid += 8) dequantize_q5_K<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_Q6_K) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 64; tid += 8) dequantize_q6_K<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ1_S) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 32; tid += 8) dequantize_iq1_s<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ1_M) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 32; tid += 8) dequantize_iq1_m<float>(row, k0 / QK, out, tid);
    }
    // IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS: a lane decodes 8-weight group (lane & 3) of 32-block ib0 + (lane >> 2)
    else if constexpr (TYPE == GGML_TYPE_IQ2_XXS) {
        const block_iq2_xxs * x = (const block_iq2_xxs *) row + k0 / QK;
        const int sub = lane >> 2, il = lane & 3, ib = (k0 % QK) / 32 + sub;
        const uint16_t * q2 = x->qs + 4 * ib;
        const uint32_t aux32 = q2[2] | (q2[3] << 16);
        const float d = (float) x->d * (0.5f + (aux32 >> 28)) * 0.25f;
        mmb_store_grid8(dst + 32 * sub + 8 * il, iq2xxs_grid[((const uint8_t *) q2)[il]], d, ksigns_iq2xs[(aux32 >> 7 * il) & 127]);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ2_XS) {
        const block_iq2_xs * x = (const block_iq2_xs *) row + k0 / QK;
        const int sub = lane >> 2, il = lane & 3, ib = (k0 % QK) / 32 + sub;
        const uint16_t q2 = x->qs[4 * ib + il];
        const float d = (float) x->d * (0.5f + ((x->scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
        mmb_store_grid8(dst + 32 * sub + 8 * il, iq2xs_grid[q2 & 511], d, ksigns_iq2xs[q2 >> 9]);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ2_S) {
        const block_iq2_s * x = (const block_iq2_s *) row + k0 / QK;
        const int sub = lane >> 2, il = lane & 3, ib = (k0 % QK) / 32 + sub;
        const float d = (float) x->d * (0.5f + ((x->scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
        mmb_store_grid8(dst + 32 * sub + 8 * il, iq2s_grid[x->qs[4 * ib + il] | ((x->qh[ib] << (8 - 2 * il)) & 0x300)], d,
            x->qs[QK_K / 8 + 4 * ib + il]);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ3_XXS) {
        const block_iq3_xxs * x = (const block_iq3_xxs *) row + k0 / QK;
        const int sub = lane >> 2, il = lane & 3, ib = (k0 % QK) / 32 + sub;
        const uint8_t * q3 = x->qs + 8 * ib;
        const uint16_t * gas = (const uint16_t *) (x->qs + QK_K / 4) + 2 * ib;
        const uint32_t aux32 = gas[0] | (gas[1] << 16);
        const float d = (float) x->d * (0.5f + (aux32 >> 28)) * 0.5f;
        const uint64_t grid = iq3xxs_grid[q3[2 * il + 0]] | ((uint64_t) iq3xxs_grid[q3[2 * il + 1]] << 32);
        mmb_store_grid8(dst + 32 * sub + 8 * il, grid, d, ksigns_iq2xs[(aux32 >> 7 * il) & 127]);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ3_S) {
        // lane -> (il = lane>>1, ib = ib0 + (lane&1)): 8 consecutive weights, one 16-byte LDS store.
        // Same arithmetic as dequantize_iq3_s (d * grid * sign in fp32, then RNE bf16) -> bit-identical.
        const block_iq3_s * x = (const block_iq3_s *) row + k0 / QK;
        const int ib0 = (k0 % QK) / 32, sub = lane & 1, il = lane >> 1, ib = ib0 + sub;
        const uint16_t q2 = *(const uint16_t *)(x->qs + 8 * ib + 2 * il);
        const int qh = x->qh[ib];
        const uint32_t g1 = iq3s_grid[(q2 & 0xff) | ((qh << (8 - 2 * il)) & 256)];
        const uint32_t g2 = iq3s_grid[(q2 >> 8)   | ((qh << (7 - 2 * il)) & 256)];
        const float d = (float) x->d * (1 + 2 * ((x->scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
        const int signs = x->signs[4 * ib + il];
        float v[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j + 0] = d * (float)((g1 >> (8 * j)) & 0xff) * (signs & (1 << (j + 0)) ? -1.f : 1.f);
            v[j + 4] = d * (float)((g2 >> (8 * j)) & 0xff) * (signs & (1 << (j + 4)) ? -1.f : 1.f);
        }
        uint4 o; o.x = mmb_pack2(v[0], v[1]); o.y = mmb_pack2(v[2], v[3]); o.z = mmb_pack2(v[4], v[5]); o.w = mmb_pack2(v[6], v[7]);
        *(uint4 *)(dst + 32 * sub + 8 * il) = o;
    }
    else if constexpr (TYPE == GGML_TYPE_IQ4_XS) {
        // decode only the two 32-blocks of this 64-wide K slice (tid = il*8 + ib): 1 call/lane instead of 4
        const mmb_quant_slice out{dst, k0 % QK};
        const int ib0 = (k0 % QK) / 32;
        dequantize_iq4_xs<float>(row, k0 / QK, out, (lane >> 1) * 8 + ib0 + (lane & 1));
    }
    else if constexpr (TYPE == GGML_TYPE_IQ4_NL) {
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / 32, q = (pos % 32) / 2;
            const block_iq4_nl & b = ((const block_iq4_nl *) row)[ib];
            const int o = ib * 32 + q - k0;
            dst[o] = mmb_f2bf((float)b.d * kvalues_iq4nl[b.qs[q] & 15]);
            dst[o + 16] = mmb_f2bf((float)b.d * kvalues_iq4nl[b.qs[q] >> 4]);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_MXFP4) {
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, q = (pos % QK) / 2;
            const block_mxfp4 & b = ((const block_mxfp4 *) row)[ib];
            const float d = ggml_cuda_e8m0_to_fp32(b.e);
            const int o = ib * QK + q - k0;
            dst[o] = mmb_f2bf(d * kvalues_mxfp4[b.qs[q] & 15] * 0.5f);
            dst[o + QK / 2] = mmb_f2bf(d * kvalues_mxfp4[b.qs[q] >> 4] * 0.5f);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_NVFP4) {
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, in = pos % QK;
            const int sub = in / QK_NVFP4_SUB, j = (in % QK_NVFP4_SUB) / 2;
            const block_nvfp4 & b = ((const block_nvfp4 *) row)[ib];
            const float d = ggml_cuda_ue4m3_to_fp32(b.d[sub]);
            const uint8_t q = b.qs[sub * (QK_NVFP4_SUB / 2) + j];
            const int o = ib * QK + sub * QK_NVFP4_SUB + j - k0;
            dst[o] = mmb_f2bf(d * kvalues_mxfp4[q & 15]);
            dst[o + QK_NVFP4_SUB / 2] = mmb_f2bf(d * kvalues_mxfp4[q >> 4]);
        }
    }
}

template <int WTYPE>
__host__ __device__ constexpr size_t mmb_row_bytes(int k) {
    if constexpr (WTYPE == 0) return (size_t)(k / 32) * 18;
    else if constexpr (WTYPE == 1) return (size_t)(k / 32) * 34;
    else if constexpr (WTYPE == 2) return (size_t)k * 2;
    else {
        using traits = ggml_cuda_type_traits<(ggml_type)(WTYPE - 32)>;
        return (size_t)(k / traits::qk) * traits::bs;
    }
}

template <int WTYPE, int BM, int STRIDE>
__device__ __forceinline__ void mmb_load_quant_tile(const uint8_t * weights, size_t row_bytes, int rows, int ks, uint16_t * tile) {
    for (int row = threadIdx.x / 8; row < BM; row += blockDim.x / 8) {
        uint16_t * dst = tile + row * STRIDE;
        const int lane = threadIdx.x % 8;
        if (row < rows) mmb_decode_slice<(ggml_type)(WTYPE - 32)>(weights + row * row_bytes, ks * 64, dst, lane);
        else {
#pragma unroll
            for (int k = lane; k < 64; k += 8) dst[k] = 0;
        }
    }
}

// routed: a MUL_MAT_ID expert weight (the decoders PR #91 tuned: Q4_K / Q5_K / Q5_1 ...). Dense GEMMs keep the two types
// with decoders measured on every arch (#123).
static bool mmb_quant_type(ggml_type type, bool routed = false) {
    if (!routed && type != GGML_TYPE_Q8_0 && type != GGML_TYPE_IQ4_NL) return false;
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            return true;
        default: return false;
    }
}

template <typename Fn>
static void mmb_dispatch_quant(ggml_type type, Fn fn) {
    switch (type) {
        case GGML_TYPE_Q1_0: fn(std::integral_constant<int, 32 + GGML_TYPE_Q1_0>{}); break;
        case GGML_TYPE_Q2_0: fn(std::integral_constant<int, 32 + GGML_TYPE_Q2_0>{}); break;
        case GGML_TYPE_Q4_0: fn(std::integral_constant<int, 32 + GGML_TYPE_Q4_0>{}); break;
        case GGML_TYPE_Q4_1: fn(std::integral_constant<int, 32 + GGML_TYPE_Q4_1>{}); break;
        case GGML_TYPE_Q5_0: fn(std::integral_constant<int, 32 + GGML_TYPE_Q5_0>{}); break;
        case GGML_TYPE_Q5_1: fn(std::integral_constant<int, 32 + GGML_TYPE_Q5_1>{}); break;
        case GGML_TYPE_Q8_0: fn(std::integral_constant<int, 1>{}); break;
        case GGML_TYPE_Q2_K: fn(std::integral_constant<int, 32 + GGML_TYPE_Q2_K>{}); break;
        case GGML_TYPE_Q3_K: fn(std::integral_constant<int, 32 + GGML_TYPE_Q3_K>{}); break;
        case GGML_TYPE_Q4_K: fn(std::integral_constant<int, 32 + GGML_TYPE_Q4_K>{}); break;
        case GGML_TYPE_Q5_K: fn(std::integral_constant<int, 32 + GGML_TYPE_Q5_K>{}); break;
        case GGML_TYPE_Q6_K: fn(std::integral_constant<int, 32 + GGML_TYPE_Q6_K>{}); break;
        case GGML_TYPE_IQ1_S: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ1_S>{}); break;
        case GGML_TYPE_IQ1_M: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ1_M>{}); break;
        case GGML_TYPE_IQ2_XXS: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ2_XXS>{}); break;
        case GGML_TYPE_IQ2_XS: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ2_XS>{}); break;
        case GGML_TYPE_IQ2_S: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ2_S>{}); break;
        case GGML_TYPE_IQ3_XXS: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ3_XXS>{}); break;
        case GGML_TYPE_IQ3_S: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ3_S>{}); break;
        case GGML_TYPE_IQ4_XS: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ4_XS>{}); break;
        case GGML_TYPE_IQ4_NL: fn(std::integral_constant<int, 0>{}); break;
        case GGML_TYPE_MXFP4: fn(std::integral_constant<int, 32 + GGML_TYPE_MXFP4>{}); break;
        case GGML_TYPE_NVFP4: fn(std::integral_constant<int, 32 + GGML_TYPE_NVFP4>{}); break;
        default: GGML_ABORT("unsupported MMB quant type");
    }
}

template <bool HALF = false>
__device__ __forceinline__ void mmb_dq_q4k_slice(uint4 q0, uint4 q1, uint4 meta, int slice, uint32_t * out, int half = 0) {
    const float d = mmb_h2f((uint16_t) meta.x), dm = mmb_h2f((uint16_t)(meta.x >> 16));
    auto byte = [&](int i) { const uint32_t word = i < 4 ? meta.y : i < 8 ? meta.z : meta.w; return (word >> (8 * (i & 3))) & 255; };
    auto scale_min = [&](int i, float & ds, float & ms) {
        const uint32_t sc = i < 4 ? byte(i) & 63 : (byte(i + 4) & 15) | ((byte(i - 4) >> 6) << 4);
        const uint32_t mn = i < 4 ? byte(i + 4) & 63 : (byte(i + 4) >> 4) | ((byte(i) >> 6) << 4);
        ds = d * sc; ms = dm * mn;
    };
    float d0, d1, m0, m1;
    scale_min(2 * slice, d0, m0); scale_min(2 * slice + 1, d1, m1);
    const uint32_t words[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
    for (int j = 0; j < (HALF ? 4 : 8); ++j) {
        const int wj = HALF ? j + 4 * half : j;
        const uint32_t q = words[j];
        out[2*wj] = mmb_pack2(d0 * (q & 15) - m0, d0 * ((q >> 8) & 15) - m0);
        out[2*wj + 1] = mmb_pack2(d0 * ((q >> 16) & 15) - m0, d0 * ((q >> 24) & 15) - m0);
        out[16 + 2*wj] = mmb_pack2(d1 * ((q >> 4) & 15) - m1, d1 * ((q >> 12) & 15) - m1);
        out[16 + 2*wj + 1] = mmb_pack2(d1 * ((q >> 20) & 15) - m1, d1 * (q >> 28) - m1);
    }
}


__device__ __forceinline__ void mmb_dq_q51_pair(uint4 a, uint4 b, uint4 c, uint32_t * out) {
    const uint32_t words[12] = {a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w,c.x,c.y,c.z,c.w};
#pragma unroll
    for (int block = 0; block < 2; ++block) {
        const uint32_t dm = words[block * 6], high = words[block * 6 + 1];
        const float d = mmb_h2f((uint16_t)dm), m = mmb_h2f((uint16_t)(dm >> 16));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q = words[block * 6 + 2 + j];
            float lo[4], hi[4];
#pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
                const int low = ((q >> (8 * lane)) & 15) | (((high >> (4 * j + lane)) & 1) << 4);
                const int upper = ((q >> (8 * lane + 4)) & 15) | (((high >> (16 + 4 * j + lane)) & 1) << 4);
                lo[lane] = d * low + m; hi[lane] = d * upper + m;
            }
            out[block * 16 + 2*j] = mmb_pack2(lo[0],lo[1]);
            out[block * 16 + 2*j + 1] = mmb_pack2(lo[2],lo[3]);
            out[block * 16 + 8 + 2*j] = mmb_pack2(hi[0],hi[1]);
            out[block * 16 + 8 + 2*j + 1] = mmb_pack2(hi[2],hi[3]);
        }
    }
}
