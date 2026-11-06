#pragma once

// K/V loaders of the HIP sparse-attention WMMA kernels (PLAN-v3 §6.2, D8, D10) and the per-type loader traits.
//
// Every loader delivers, for the 16 cells of one plan chunk (4 entries x 4 cells) and the 32 dims of block w of a K/V
// row (wave w of the 8-wave G4 block owns dims 32w .. 32w+31), the f16 halves of the op contract (sparse-attn-q8.cuh):
// the WMMA fragments are identical whichever loader produced them (C18 "Correctness": 4905 bitwise checks between
// loader variants of the same type pair).
//
// K loaders (D8, C18 decision 1):
//   SA_K_DIRECT  every lane loads the 32 dims of its key (lanes r and r + 16 duplicate the load), the PR load_k
//   SA_K_SPLIT   every type in the G4 attend (sa_k_loader_of): lane half p loads and converts 16 of the 32 dims once ("piece" p: dims 8p..8p+7 and
//                16+8p..16+8p+7), the halves are exchanged with v_permlanex16; no LDS. C18: split is -6.7 % against the
//                PR's per-lane q8 K loader on wiki d131k, and +3.9 % / +5.5 % (q8_0), +2.2 % / +3.7 % (q4_0),
//                +1.4 % / +4.3 % (bf16) against f16 K at d4k / d131k text
// V loaders:
//   SA_V_PACKED      the PR's packed f16 V [stream][kv head][quad][dim][4 cells] written by sa_pack_v (sparse-attn-pack.cu)
//                    into the attend dst over-allocation; one uint2 per (dim, entry)
//   SA_V_SPLITGATHER direct read of the cache rows: lane half p gathers dim 16p + r of block w for the 16 cells (one
//                    element each), converts them after barrier 2 and exchanges the halves with v_permlanex16. C18: the
//                    fastest direct V at 262k / 1M for q8_0 / q4_0 (X+X 20.34 / 24.60 ms vs packed 23.18 / 37.32), 0
//                    scratch with split K; for f16 / bf16 it is the direct V of the CAPPED trait (below)
//
// V pack-limit trait (D8, OWNER decision 2026-11-05, AT-O deferred): SA_VPACK_CAPPED = packed iff n_kv < 64 * n_q, else
// direct (PR #219's n_q * 64 < n_kv switch, C18 decision 3); SA_VPACK_ALWAYS = packed at every n_kv (the owner's
// alternative for f16 / bf16: +512 / +1024 MiB at c524288 / c1M, R1-C). Changing a type's row of sa_vpack_trait is the
// one-line switch.

#include <cstdint>
#include <cstddef>

// the type ids of the WMMA tiers (S_fast, PLAN-v3 §6.1); also the order of the census keys
enum sa_type_id {
    SA_T_F16  = 0,
    SA_T_BF16 = 1,
    SA_T_Q8_0 = 2,
    SA_T_Q4_0 = 3,
    SA_T_COUNT,
};

enum sa_k_loader { SA_K_DIRECT = 0, SA_K_SPLIT = 1 };
enum sa_v_loader { SA_V_PACKED = 0, SA_V_SPLITGATHER = 1, SA_V_COUNT };
enum sa_vpack    { SA_VPACK_CAPPED = 0, SA_VPACK_ALWAYS = 1 };

// the tokens of the census instance tables (sparse-attn-census.h) as kernel template arguments
#define SA_TID_f16     SA_T_F16
#define SA_TID_bf16    SA_T_BF16
#define SA_TID_q8_0    SA_T_Q8_0
#define SA_TID_q4_0    SA_T_Q4_0
#define SA_VL_packed   SA_V_PACKED
#define SA_VL_direct   SA_V_SPLITGATHER
#define SA_EPI_plain   SA_EPI_PLAIN
#define SA_EPI_gated   SA_EPI_GATED

// K loader of the G4 attend: split for every type. D8 names direct for f16; C18 measured split and direct equal for
// f16 K (Kf16X-Vf16P vs Kf16D-Vf16P: 12.43 / 12.60 ms wiki d4k, 16.32 / 16.33 wiki d131k, 38.30 / 38.80 syn 262k,
// 48.78 / 49.04 syn 1M), and split holds 8 raw VGPRs of the next chunk's K across the barriers where direct holds the 16
// of the converted fragment: with direct f16 K, the GQA 12 instances with a direct quantized V do not fit 192 VGPRs
// (W3 resource checks). SA_K_DIRECT stays for the G1 tier (D9).
static constexpr int sa_k_loader_of(const int t) {
    return t == SA_T_COUNT ? SA_K_DIRECT : SA_K_SPLIT;
}

// D8 V pack-limit trait, per V type (default CAPPED for every type)
static constexpr int sa_vpack_trait(const int t) {
    return t == SA_T_F16  ? SA_VPACK_CAPPED :
           t == SA_T_BF16 ? SA_VPACK_CAPPED :
                            SA_VPACK_CAPPED;
}

// bytes of one K/V row block of 32 dims, and of a 256-dim row
static constexpr int sa_block_bytes(const int t) {
    return t == SA_T_Q8_0 ? 34 : t == SA_T_Q4_0 ? 18 : 64;
}

#if defined(GGML_USE_HIP)

#include "sparse-attn-q8.cuh"

// value of lane i ^ 16
static __device__ __forceinline__ uint32_t sa_xchg16(const uint32_t x) {
    return __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xfedcba98u, false, false);
}

// raw dwords of one piece (dims 8p..8p+7 and 16+8p..16+8p+7 of block w of the row)
template <int T> struct sa_piece { static constexpr int N = T == SA_T_Q8_0 ? 7 : T == SA_T_Q4_0 ? 4 : 8; };

template <int T>
static __device__ __forceinline__ void sa_piece_load(const char * row, const int w, const int p, uint32_t * raw) {
    if constexpr (T == SA_T_Q8_0) {
        const char * base = row + 34 * w;
        raw[6] = *reinterpret_cast<const uint16_t *>(base);
        if (w & 1) {   // qs dword aligned (rows are 16-byte aligned, 34 * w + 2 = 4 mod 4 for odd w)
            const uint32_t * x = reinterpret_cast<const uint32_t *>(base + 2 + 8 * p);
            const uint32_t * y = reinterpret_cast<const uint32_t *>(base + 18 + 8 * p);
            raw[0] = x[0]; raw[1] = x[1]; raw[3] = y[0]; raw[4] = y[1]; raw[2] = 0; raw[5] = 0;
        } else {       // qs 2 bytes past a dword
            const uint32_t * x = reinterpret_cast<const uint32_t *>(base + 8 * p);
            const uint32_t * y = reinterpret_cast<const uint32_t *>(base + 16 + 8 * p);
            raw[0] = x[0]; raw[1] = x[1]; raw[2] = x[2]; raw[3] = y[0]; raw[4] = y[1]; raw[5] = y[2];
        }
    } else if constexpr (T == SA_T_Q4_0) {
        const char * base = row + 18 * w;
        raw[3] = *reinterpret_cast<const uint16_t *>(base);
        if (w & 1) {
            const uint32_t * x = reinterpret_cast<const uint32_t *>(base + 2 + 8 * p);
            raw[0] = x[0]; raw[1] = x[1]; raw[2] = 0;
        } else {
            const uint32_t * x = reinterpret_cast<const uint32_t *>(base + 8 * p);
            raw[0] = x[0]; raw[1] = x[1]; raw[2] = x[2];
        }
    } else {
        const uint4 x = *reinterpret_cast<const uint4 *>(row + 64 * w + 16 * p);
        const uint4 y = *reinterpret_cast<const uint4 *>(row + 64 * w + 32 + 16 * p);
        raw[0] = x.x; raw[1] = x.y; raw[2] = x.z; raw[3] = x.w; raw[4] = y.x; raw[5] = y.y; raw[6] = y.z; raw[7] = y.w;
    }
}

// out[0..3] = halves of dims 8p..8p+7, out[4..7] = dims 16+8p..16+8p+7
template <int T>
static __device__ __forceinline__ void sa_piece_cvt(const uint32_t * raw, const int w, uint32_t * out) {
    if constexpr (T == SA_T_Q8_0) {
        const _Float16 d = __builtin_bit_cast(_Float16, (uint16_t) raw[6]);
        const sa_half2 d2 = sa_half2{d, d};
        uint32_t qa[2], qb[2];
        if (w & 1) {
            qa[0] = raw[0]; qa[1] = raw[1]; qb[0] = raw[3]; qb[1] = raw[4];
        } else {
            qa[0] = __builtin_amdgcn_alignbyte(raw[1], raw[0], 2); qa[1] = __builtin_amdgcn_alignbyte(raw[2], raw[1], 2);
            qb[0] = __builtin_amdgcn_alignbyte(raw[4], raw[3], 2); qb[1] = __builtin_amdgcn_alignbyte(raw[5], raw[4], 2);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            uint32_t e = sa_q8x2_to_h2(qa[i], d2), o = sa_q8x2_to_h2(qa[i] >> 8, d2);
            out[2*i] = __builtin_amdgcn_perm(o, e, 0x05040100u); out[2*i+1] = __builtin_amdgcn_perm(o, e, 0x07060302u);
            e = sa_q8x2_to_h2(qb[i], d2); o = sa_q8x2_to_h2(qb[i] >> 8, d2);
            out[4+2*i] = __builtin_amdgcn_perm(o, e, 0x05040100u); out[4+2*i+1] = __builtin_amdgcn_perm(o, e, 0x07060302u);
        }
    } else if constexpr (T == SA_T_Q4_0) {
        const _Float16 d = __builtin_bit_cast(_Float16, (uint16_t) raw[3]);
        const sa_half2 d2 = sa_half2{d, d};
        uint32_t qq[2];
        if (w & 1) {
            qq[0] = raw[0]; qq[1] = raw[1];
        } else {
            qq[0] = __builtin_amdgcn_alignbyte(raw[1], raw[0], 2); qq[1] = __builtin_amdgcn_alignbyte(raw[2], raw[1], 2);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t L = qq[i], H = qq[i] >> 4;
            uint32_t e = sa_q4x2_to_h2(L, d2), o = sa_q4x2_to_h2(L >> 8, d2);
            out[2*i] = __builtin_amdgcn_perm(o, e, 0x05040100u); out[2*i+1] = __builtin_amdgcn_perm(o, e, 0x07060302u);
            e = sa_q4x2_to_h2(H, d2); o = sa_q4x2_to_h2(H >> 8, d2);
            out[4+2*i] = __builtin_amdgcn_perm(o, e, 0x05040100u); out[4+2*i+1] = __builtin_amdgcn_perm(o, e, 0x07060302u);
        }
    } else if constexpr (T == SA_T_BF16) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out[i] = sa_bf2_to_h2(raw[i]);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out[i] = raw[i];
        }
    }
}

// one V element (dim dd of block w) of a row as one raw dword for the gather loader: f16 / bf16 bits in the low half,
// or the q8_0 byte / the q4_0 nibble's byte (pre-shifted) in the low half and the block scale d in the high half
template <int T>
static __device__ __forceinline__ uint32_t sa_gather_load(const char * row, const int w, const int dd) {
    if constexpr (T == SA_T_Q8_0) {
        const char * blk = row + 34 * w;
        return (uint32_t) *reinterpret_cast<const uint8_t *>(blk + 2 + dd) | ((uint32_t) *reinterpret_cast<const uint16_t *>(blk) << 16);
    } else if constexpr (T == SA_T_Q4_0) {
        const char * blk = row + 18 * w;
        return ((uint32_t) *reinterpret_cast<const uint8_t *>(blk + 2 + (dd & 15)) >> (dd & 16 ? 4 : 0)) |
               ((uint32_t) *reinterpret_cast<const uint16_t *>(blk) << 16);
    } else {
        return *reinterpret_cast<const uint16_t *>(row + 2 * (32 * w + dd));
    }
}

// two gathered elements (cells k, k + 1) to one dword of halves
template <int T>
static __device__ __forceinline__ uint32_t sa_gather_pair(const uint32_t g0, const uint32_t g1) {
    if constexpr (T == SA_T_Q8_0) {
        return sa_q8x2_to_h2((g0 & 0xffu) | ((g1 & 0xffu) << 16), __builtin_bit_cast(sa_half2, __builtin_amdgcn_perm(g1, g0, 0x07060302u)));
    } else if constexpr (T == SA_T_Q4_0) {
        return sa_q4x2_to_h2((g0 & 15u) | ((g1 & 15u) << 16), __builtin_bit_cast(sa_half2, __builtin_amdgcn_perm(g1, g0, 0x07060302u)));
    } else if constexpr (T == SA_T_BF16) {
        return (uint32_t) sa_bf1_to_h(g0 & 0xffffu) | ((uint32_t) sa_bf1_to_h(g1 & 0xffffu) << 16);
    } else {
        return (g0 & 0xffffu) | (g1 << 16);
    }
}

// f16 halves of the 4 cells of entry j of a chunk, each zeroed (select, not multiply) when no query of the group
// attends the cell: mk is the entry's 16-bit mask, bit 4 * query + cell
static __device__ __forceinline__ uint2 sa_mask_cells(const uint2 v, const uint32_t mk) {
    uint32_t lo = 0, hi = 0;
    if (mk & 0x1111u) lo |= 0x0000ffffu;
    if (mk & 0x2222u) lo |= 0xffff0000u;
    if (mk & 0x4444u) hi |= 0x0000ffffu;
    if (mk & 0x8888u) hi |= 0xffff0000u;
    return make_uint2(v.x & lo, v.y & hi);
}

#endif // defined(GGML_USE_HIP)
