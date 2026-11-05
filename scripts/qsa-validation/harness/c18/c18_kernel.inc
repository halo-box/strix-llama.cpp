// C18: the PR's G4 attention kernel (pr_attn.inc = PR 94e086847 qsa-prefill.cu, qsa3_attn_kernel) with the K and V
// loaders replaced by per-type, per-strategy policies. Everything outside the loaders (Q prologue, WMMA order, online
// softmax, barriers, epilogue) is the PR code line by line, so outputs of loaders that deliver the same halves are
// bit-identical to each other and to the PR kernel.
// Types : TF16 f16; TQ8 q8_0; TQ4 q4_0; TBF bf16 -> f16 (saturating) then f16 WMMA; TBW bf16 bits fed to bf16 WMMA
//         (Q, resp. P, rounded to bf16 for that product).
// K loaders: LD  = direct, per lane, dequant at the load (PR load_k: every lane loads and dequantizes its key's 32 dims,
//                 lanes r and r+16 duplicate the work);
//            LDD = LD with the dequant deferred to the end of the iteration (raw bytes held in VGPRs across the barriers);
//            LX  = split: lane half p loads/dequantizes 16 of the 32 dims, exchange with v_permlanex16, deferred;
//            LS  = LDS-staged: lane l takes key l>>1, half l&1 (coalesced), dequantizes once, writes an f16 tile in a
//                 wave-private LDS slot (aliasing part[0][w], free between barrier 2 and the next partials), reads the
//                 WMMA fragment back; deferred.
// V loaders: LP  = the PR packed V (pv = f16/bf16 [kvh][quad][dim][4 keys] filled by a pack kernel, timed separately);
//            LD  = direct gather per lane (PR DIRECT_V for f16/q8; same scheme for q4/bf16), dequant at the load;
//            LX  = split gather: lane half p gathers dim 16p+r only, exchange with v_permlanex16, dequant deferred
//                 to after barrier 2;
//            LS  = LDS-staged: coalesced piece loads (as K LS), dequantized once, written transposed into a
//                 wave-private LDS slot (aliasing part[1][w]), fragment read back with 2 x b128; deferred.
enum { TF16 = 0, TQ8 = 1, TQ4 = 2, TBF = 3, TBW = 4 };
enum { LD = 0, LDD = 1, LX = 2, LS = 3, LP = 4 };

#define QK4_0 32
typedef struct { uint16_t d; uint8_t qs[QK4_0 / 2]; } block_q4_0;

static __device__ __forceinline__ uint32_t c18_q4x2_to_h2(const uint32_t x, const qsa3_half2 d2) {
    // nibbles in byte 0 and byte 2 of x: 0x6400 | q is 1024 + q in half, exact; minus 1032 gives q - 8 exactly
    const qsa3_half2 v = __builtin_bit_cast(qsa3_half2, (x & 0x000F000Fu) | 0x64006400u);
    return __builtin_bit_cast(uint32_t, (v - qsa3_half2{(_Float16) 1032, (_Float16) 1032}) * d2);
}
static __device__ __forceinline__ uint32_t c18_bf2_to_h2(const uint32_t x) {
    const float lo = __builtin_bit_cast(float, x << 16), hi = __builtin_bit_cast(float, x & 0xffff0000u);
    const _Float16 a = (_Float16) __builtin_amdgcn_fmed3f(lo, -65504.f, 65504.f);
    const _Float16 b = (_Float16) __builtin_amdgcn_fmed3f(hi, -65504.f, 65504.f);
    return (uint32_t) __builtin_bit_cast(uint16_t, a) | ((uint32_t) __builtin_bit_cast(uint16_t, b) << 16);
}
static __device__ __forceinline__ uint16_t c18_bf1_to_h(const uint32_t x16) {
    return __builtin_bit_cast(uint16_t, (_Float16) __builtin_amdgcn_fmed3f(__builtin_bit_cast(float, x16 << 16), -65504.f, 65504.f));
}
static __device__ __forceinline__ uint16_t c18_f2bf(const float f) {
    uint32_t u = __builtin_bit_cast(uint32_t, f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
static __device__ __forceinline__ uint32_t c18_xchg(const uint32_t x) {   // value of lane i^16
    return __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xfedcba98u, false, false);
}

// raw words of one "piece" (16 of the 32 dims of block w of one key row): dims 8p..8p+7 and 16+8p..16+8p+7
template<int T> struct c18_piece { static constexpr int N = (T == TQ8) ? 7 : (T == TQ4) ? 4 : 8; };

template<int T> static __device__ __forceinline__ void c18_piece_load(const char * row, const int w, const int p, uint32_t * raw) {
    if constexpr (T == TQ8) {
        const char * base = row + 34 * w;
        raw[6] = *reinterpret_cast<const uint16_t *>(base);
        if (w & 1) {   // qs dword aligned
            const uint32_t * a = reinterpret_cast<const uint32_t *>(base + 2 + 8 * p);
            const uint32_t * b = reinterpret_cast<const uint32_t *>(base + 18 + 8 * p);
            raw[0] = a[0]; raw[1] = a[1]; raw[3] = b[0]; raw[4] = b[1]; raw[2] = 0; raw[5] = 0;
        } else {       // qs 2 bytes past a dword
            const uint32_t * a = reinterpret_cast<const uint32_t *>(base + 8 * p);
            const uint32_t * b = reinterpret_cast<const uint32_t *>(base + 16 + 8 * p);
            raw[0] = a[0]; raw[1] = a[1]; raw[2] = a[2]; raw[3] = b[0]; raw[4] = b[1]; raw[5] = b[2];
        }
    } else if constexpr (T == TQ4) {
        const char * base = row + 18 * w;
        raw[3] = *reinterpret_cast<const uint16_t *>(base);
        if (w & 1) {
            const uint32_t * a = reinterpret_cast<const uint32_t *>(base + 2 + 8 * p);
            raw[0] = a[0]; raw[1] = a[1]; raw[2] = 0;
        } else {
            const uint32_t * a = reinterpret_cast<const uint32_t *>(base + 8 * p);
            raw[0] = a[0]; raw[1] = a[1]; raw[2] = a[2];
        }
    } else {
        const uint4 a = *reinterpret_cast<const uint4 *>(row + 64 * w + 16 * p);
        const uint4 b = *reinterpret_cast<const uint4 *>(row + 64 * w + 32 + 16 * p);
        raw[0] = a.x; raw[1] = a.y; raw[2] = a.z; raw[3] = a.w; raw[4] = b.x; raw[5] = b.y; raw[6] = b.z; raw[7] = b.w;
    }
}
// out[0..3] = halves of dims 8p..8p+7, out[4..7] = dims 16+8p..16+8p+7 (f16, or bf16 bits for TBW)
template<int T> static __device__ __forceinline__ void c18_piece_deq(const uint32_t * raw, const int w, uint32_t * out) {
    if constexpr (T == TQ8) {
        const _Float16 d = __builtin_bit_cast(_Float16, (uint16_t) raw[6]);
        const qsa3_half2 d2 = qsa3_half2{d, d};
        uint32_t qa[2], qb[2];
        if (w & 1) { qa[0] = raw[0]; qa[1] = raw[1]; qb[0] = raw[3]; qb[1] = raw[4]; }
        else {
            qa[0] = __builtin_amdgcn_alignbyte(raw[1], raw[0], 2); qa[1] = __builtin_amdgcn_alignbyte(raw[2], raw[1], 2);
            qb[0] = __builtin_amdgcn_alignbyte(raw[4], raw[3], 2); qb[1] = __builtin_amdgcn_alignbyte(raw[5], raw[4], 2);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            uint32_t e = qsa3_q8x2_to_h2(qa[i], d2), o = qsa3_q8x2_to_h2(qa[i] >> 8, d2);
            out[2*i] = __builtin_amdgcn_perm(o, e, 0x05040100u); out[2*i+1] = __builtin_amdgcn_perm(o, e, 0x07060302u);
            e = qsa3_q8x2_to_h2(qb[i], d2); o = qsa3_q8x2_to_h2(qb[i] >> 8, d2);
            out[4+2*i] = __builtin_amdgcn_perm(o, e, 0x05040100u); out[4+2*i+1] = __builtin_amdgcn_perm(o, e, 0x07060302u);
        }
    } else if constexpr (T == TQ4) {
        const _Float16 d = __builtin_bit_cast(_Float16, (uint16_t) raw[3]);
        const qsa3_half2 d2 = qsa3_half2{d, d};
        uint32_t qq[2];
        if (w & 1) { qq[0] = raw[0]; qq[1] = raw[1]; }
        else { qq[0] = __builtin_amdgcn_alignbyte(raw[1], raw[0], 2); qq[1] = __builtin_amdgcn_alignbyte(raw[2], raw[1], 2); }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t L = qq[i], H = qq[i] >> 4;
            uint32_t e = c18_q4x2_to_h2(L, d2), o = c18_q4x2_to_h2(L >> 8, d2);
            out[2*i] = __builtin_amdgcn_perm(o, e, 0x05040100u); out[2*i+1] = __builtin_amdgcn_perm(o, e, 0x07060302u);
            e = c18_q4x2_to_h2(H, d2); o = c18_q4x2_to_h2(H >> 8, d2);
            out[4+2*i] = __builtin_amdgcn_perm(o, e, 0x05040100u); out[4+2*i+1] = __builtin_amdgcn_perm(o, e, 0x07060302u);
        }
    } else if constexpr (T == TBF) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { out[i] = c18_bf2_to_h2(raw[i]); }
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) { out[i] = raw[i]; }
    }
}
// one V element (key row, dim dd of block w) as ONE raw dword for the gather loaders: f16/bf16 bits in the low half,
// or qs byte (q4_0: the nibble's byte, pre-shifted) in the low half and the block scale d in the high half
template<int T> struct c18_gat { static constexpr int N = 1; };
template<int T> static __device__ __forceinline__ void c18_gat_load(const char * row, const int w, const int dd, uint32_t * g) {
    if constexpr (T == TQ8) {
        const block_q8_0 * blk = reinterpret_cast<const block_q8_0 *>(row + 34 * w);
        g[0] = (uint32_t) (uint8_t) blk->qs[dd] | ((uint32_t) blk->d << 16);
    } else if constexpr (T == TQ4) {
        const block_q4_0 * blk = reinterpret_cast<const block_q4_0 *>(row + 18 * w);
        g[0] = ((uint32_t) (blk->qs[dd & 15]) >> (dd & 16 ? 4 : 0)) | ((uint32_t) blk->d << 16);
    } else {
        g[0] = *reinterpret_cast<const uint16_t *>(row + 2 * (32 * w + dd));
    }
}
// two gathered elements (keys k, k+1) to one dword of halves
template<int T> static __device__ __forceinline__ uint32_t c18_gat_pair(const uint32_t * g0, const uint32_t * g1) {
    if constexpr (T == TQ8) {
        return qsa3_q8x2_to_h2((g0[0] & 0xffu) | ((g1[0] & 0xffu) << 16), __builtin_bit_cast(qsa3_half2, __builtin_amdgcn_perm(g1[0], g0[0], 0x07060302u)));
    } else if constexpr (T == TQ4) {
        return c18_q4x2_to_h2((g0[0] & 15u) | ((g1[0] & 15u) << 16), __builtin_bit_cast(qsa3_half2, __builtin_amdgcn_perm(g1[0], g0[0], 0x07060302u)));
    } else if constexpr (T == TBF) {
        return (uint32_t) c18_bf1_to_h(g0[0] & 0xffffu) | ((uint32_t) c18_bf1_to_h(g1[0] & 0xffffu) << 16);
    } else {
        return (g0[0] & 0xffffu) | (g1[0] << 16);
    }
}

#ifndef C18_HELPERS_ONLY
template<int KT, int KL, int VT, int VL> __global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8)))
void c18_attn_kernel(
        const float * __restrict__ q, const char * __restrict__ kc, const char * __restrict__ vc,
        const uint32_t * __restrict__ ublk, const uint16_t * __restrict__ umask,
        const int * __restrict__ ucount, float * __restrict__ out, const qsa3_layout s) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3)
    NO_DEVICE_CODE;
#else
    constexpr bool KBW = KT == TBW, VBW = VT == TBW;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, r = lane & 15, hi = lane >> 4;
    const int g = blockIdx.x, kvh = blockIdx.y;
    __shared__ float4 part[3][8][64];
    __shared__ uint4  ptile[3][32];
    __shared__ float  alpha_s[48];
    __shared__ float  l_s[48];

    const size_t nblk = (size_t) s.nk / 4;
    const char * kbase = kc + (size_t) kvh * s.knb2;
    const char * vbase = VL == LP ? nullptr : vc + (size_t) kvh * s.vnb2;
    const uint16_t * pvg = VL == LP ? reinterpret_cast<const uint16_t *>(vc) + (size_t) kvh * nblk * 1024 : nullptr;
    const uint32_t * gblk = ublk + (size_t) g * s.cap;
    const uint16_t * gmsk = umask + (size_t) g * s.cap;
    const int nchunks = ucount[g] >> 2;
    // wave-private staging slots (see header): 1 KB each
    char * ktile = reinterpret_cast<char *>(&part[0][w][0]);
    char * vtile = reinterpret_cast<char *>(&part[1][w][0]);

    v16s qf[3][2];
    {
        uint16_t * stage = reinterpret_cast<uint16_t *>(&part[0][0][0]) + w * (48 * 32);
        const _Float16 hs = (_Float16) s.scale;
#pragma unroll
        for (int c = 0; c < 12; ++c) {
            const int row = 4*c + (lane >> 3), dq = 4 * (lane & 7);
            const int qi = row / 12, h = row - 12*qi;
            const int query = QSA3_G * g + qi;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (query < s.n_q) {
                v = *reinterpret_cast<const float4 *>(reinterpret_cast<const char *>(q) + (size_t) query * s.q1 + (size_t) (kvh * s.gqa + h) * s.q2 + (size_t) (32*w + dq) * 4);
            }
            uint16_t h0, h1, h2, h3;
            if constexpr (KBW) {
                h0 = c18_f2bf(v.x * s.scale); h1 = c18_f2bf(v.y * s.scale); h2 = c18_f2bf(v.z * s.scale); h3 = c18_f2bf(v.w * s.scale);
            } else {
                h0 = __builtin_bit_cast(uint16_t, (_Float16) v.x * hs); h1 = __builtin_bit_cast(uint16_t, (_Float16) v.y * hs);
                h2 = __builtin_bit_cast(uint16_t, (_Float16) v.z * hs); h3 = __builtin_bit_cast(uint16_t, (_Float16) v.w * hs);
            }
            *reinterpret_cast<uint2 *>(stage + row * 32 + dq) = make_uint2((uint32_t) h0 | ((uint32_t) h1 << 16), (uint32_t) h2 | ((uint32_t) h3 << 16));
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 3; ++i)
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const uint16_t * src = stage + (16*i + r) * 32 + 16*t;
                qf[i][t] = __builtin_bit_cast(v16s, (uint4[2]){*reinterpret_cast<const uint4 *>(src), *reinterpret_cast<const uint4 *>(src + 8)});
            }
        __syncthreads();
    }
    v8f O[3][2];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) { O[i][t][e] = 0.f; }
    float m = -INFINITY, l = 0.f;
    const int own_row = 16*w + r, own_qi = own_row / 12;

    auto load_desc = [&](const int c, uint32_t * b, uint32_t & m01, uint32_t & m23) {
        const uint4 bb = *reinterpret_cast<const uint4 *>(gblk + 4*c);
        const uint2 mm = *reinterpret_cast<const uint2 *>(gmsk + 4*c);
        b[0] = bb.x; b[1] = bb.y; b[2] = bb.z; b[3] = bb.w; m01 = mm.x; m23 = mm.y;
    };
    auto blk_of = [&](const uint32_t * b, const int j) -> int {
        return b[j] == QSA3_BLK_SENT ? 0 : (int) b[j];
    };
    auto cell_on = [&](const uint32_t m01, const uint32_t m23, const int kk) -> bool {   // kk = 4*j + cell in the chunk
        const int j = kk >> 2;
        const uint32_t mk = ((j < 2 ? m01 : m23) >> (16 * (j & 1))) & 0xffffu;
        return ((mk >> (kk & 3)) & 0x1111u) != 0;
    };

    // ---------------- K loaders ----------------
    // LD (immediate). f16/bf16-native: PR code; q8_0: PR code; q4_0/bf16->f16: two pieces per lane
    auto load_k_direct = [&](const uint32_t * b, const uint32_t m01, const uint32_t m23, v16s * kf) {
        const int j = r >> 2;
        if (!cell_on(m01, m23, r)) { kf[0] = v16s{}; kf[1] = v16s{}; return; }
        const int kb = blk_of(b, j);
        const int key = 4*kb + (r & 3);
        const char * row = kbase + (size_t) key * s.knb1;
        if constexpr (KT == TQ8) {
            const uint32_t * a = reinterpret_cast<const uint32_t *>(row + (size_t)(34*w & ~3));
            uint32_t dw[9], qd[8];
#pragma unroll
            for (int i = 0; i < 9; ++i) { dw[i] = a[i]; }
            uint16_t dh;
            if (w & 1) {
                dh = (uint16_t) (dw[0] >> 16);
#pragma unroll
                for (int i = 0; i < 8; ++i) { qd[i] = dw[i+1]; }
            } else {
                dh = (uint16_t) dw[0];
#pragma unroll
                for (int i = 0; i < 8; ++i) { qd[i] = __builtin_amdgcn_alignbyte(dw[i+1], dw[i], 2); }
            }
            const _Float16 d = __builtin_bit_cast(_Float16, dh);
            const qsa3_half2 d2 = qsa3_half2{d, d};
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                uint32_t h[8];
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const uint32_t x = qd[4*t + jj];
                    const uint32_t e = qsa3_q8x2_to_h2(x, d2), o = qsa3_q8x2_to_h2(x >> 8, d2);
                    h[2*jj]   = __builtin_amdgcn_perm(o, e, 0x05040100u);
                    h[2*jj+1] = __builtin_amdgcn_perm(o, e, 0x07060302u);
                }
                kf[t] = __builtin_bit_cast(v16s, h);
            }
        } else if constexpr (KT == TF16 || KT == TBW) {
            const char * p = row + (size_t)((2*w) * 16) * 2;
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const char * qq = p + (size_t)t * 32;
                kf[t] = __builtin_bit_cast(v16s, (uint4[2]){*reinterpret_cast<const uint4 *>(qq), *reinterpret_cast<const uint4 *>(qq + 16)});
            }
        } else {
            uint32_t r0[c18_piece<KT>::N], r1[c18_piece<KT>::N], o0[8], o1[8];
            c18_piece_load<KT>(row, w, 0, r0); c18_piece_load<KT>(row, w, 1, r1);
            c18_piece_deq<KT>(r0, w, o0); c18_piece_deq<KT>(r1, w, o1);
            const uint32_t k0[8] = {o0[0], o0[1], o0[2], o0[3], o1[0], o1[1], o1[2], o1[3]};
            const uint32_t k1[8] = {o0[4], o0[5], o0[6], o0[7], o1[4], o1[5], o1[6], o1[7]};
            kf[0] = __builtin_bit_cast(v16s, k0); kf[1] = __builtin_bit_cast(v16s, k1);
        }
    };
    // deferred loaders: raw words per lane
    constexpr int KRN = (KL == LDD) ? 2 * c18_piece<KT>::N : c18_piece<KT>::N;
    auto k_raw = [&](const uint32_t * b, const uint32_t m01, const uint32_t m23, uint32_t * kr, bool & kon) {
        int kk, p;
        if constexpr (KL == LS) { kk = lane >> 1; p = lane & 1; } else { kk = r; p = hi; }
        kon = cell_on(m01, m23, kk);
        if (!kon) { return; }
        const int key = 4*blk_of(b, kk >> 2) + (kk & 3);
        const char * row = kbase + (size_t) key * s.knb1;
        if constexpr (KL == LDD) { c18_piece_load<KT>(row, w, 0, kr); c18_piece_load<KT>(row, w, 1, kr + c18_piece<KT>::N); }
        else { c18_piece_load<KT>(row, w, p, kr); }
    };
    auto k_fin = [&](const uint32_t * kr, const bool kon, v16s * kf) {
        if constexpr (KL == LDD) {
            uint32_t o0[8], o1[8];
            if (kon) { c18_piece_deq<KT>(kr, w, o0); c18_piece_deq<KT>(kr + c18_piece<KT>::N, w, o1); }
            else {
#pragma unroll
                for (int i = 0; i < 8; ++i) { o0[i] = 0; o1[i] = 0; }
            }
            const uint32_t k0[8] = {o0[0], o0[1], o0[2], o0[3], o1[0], o1[1], o1[2], o1[3]};
            const uint32_t k1[8] = {o0[4], o0[5], o0[6], o0[7], o1[4], o1[5], o1[6], o1[7]};
            kf[0] = __builtin_bit_cast(v16s, k0); kf[1] = __builtin_bit_cast(v16s, k1);
        } else {
            uint32_t o[8];
            if (kon) { c18_piece_deq<KT>(kr, w, o); }
            else {
#pragma unroll
                for (int i = 0; i < 8; ++i) { o[i] = 0; }
            }
            if constexpr (KL == LX) {
                uint32_t x[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) { x[i] = c18_xchg(o[i]); }
                // p = hi holds dims 8p.. of each t; lo needs x for dims 8..15 / 24..31
                uint32_t k0[8], k1[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    k0[i] = hi ? x[i] : o[i];     k0[4+i] = hi ? o[i] : x[i];
                    k1[i] = hi ? x[4+i] : o[4+i]; k1[4+i] = hi ? o[4+i] : x[4+i];
                }
                kf[0] = __builtin_bit_cast(v16s, k0); kf[1] = __builtin_bit_cast(v16s, k1);
            } else {   // LS: tile [16 keys][4 cols of 16 B], col c of row kk at ((c ^ ((kk >> 1) & 3)) << 4), row pitch 64 B
                const int kk = lane >> 1, p = lane & 1, sw = (kk >> 1) & 3;
                *reinterpret_cast<uint4 *>(ktile + kk * 64 + ((p ^ sw) << 4))       = make_uint4(o[0], o[1], o[2], o[3]);
                *reinterpret_cast<uint4 *>(ktile + kk * 64 + (((2 + p) ^ sw) << 4)) = make_uint4(o[4], o[5], o[6], o[7]);
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
                const int rs = (r >> 1) & 3;
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    const uint4 a = *reinterpret_cast<const uint4 *>(ktile + r * 64 + (((2*t) ^ rs) << 4));
                    const uint4 bq = *reinterpret_cast<const uint4 *>(ktile + r * 64 + (((2*t + 1) ^ rs) << 4));
                    kf[t] = __builtin_bit_cast(v16s, (uint4[2]){a, bq});
                }
                __builtin_amdgcn_wave_barrier();
            }
        }
    };

    // ---------------- V loaders ----------------
    auto mask_values = [&](uint2 v, uint32_t mk) {
        uint32_t lo = 0, hi2 = 0;
        if (mk & 0x1111u) lo |= 0x0000ffffu;
        if (mk & 0x2222u) lo |= 0xffff0000u;
        if (mk & 0x4444u) hi2 |= 0x0000ffffu;
        if (mk & 0x8888u) hi2 |= 0xffff0000u;
        return make_uint2(v.x & lo, v.y & hi2);
    };
    // LD (immediate) for all types; f16 and q8_0 are the PR DIRECT_V code
    auto load_v_direct = [&](const uint32_t * b, const uint32_t m01, const uint32_t m23, v16s * vf) {
        const int kb0 = blk_of(b, 0), kb1 = blk_of(b, 1), kb2 = blk_of(b, 2), kb3 = blk_of(b, 3);
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int d = 32*w + 16*t + r;
            if constexpr (VT == TQ8) {
                auto kv4 = [&](const int kb) -> uint2 {
                    const char * row = vbase + (size_t)(4*kb) * s.vnb1 + (size_t) w * sizeof(block_q8_0);
                    uint32_t qv[4];
                    _Float16 dv[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const auto * blk = reinterpret_cast<const block_q8_0 *>(row + (size_t) j * s.vnb1);
                        dv[j] = __builtin_bit_cast(_Float16, blk->d);
                        qv[j] = (uint8_t) blk->qs[16*t + r];
                    }
                    return make_uint2(qsa3_q8x2_to_h2(qv[0] | (qv[1] << 16), qsa3_half2{dv[0], dv[1]}),
                                      qsa3_q8x2_to_h2(qv[2] | (qv[3] << 16), qsa3_half2{dv[2], dv[3]}));
                };
                vf[t] = __builtin_bit_cast(v16s, (uint2[4]){
                    mask_values(kv4(kb0), m01 & 0xffffu), mask_values(kv4(kb1), m01 >> 16),
                    mask_values(kv4(kb2), m23 & 0xffffu), mask_values(kv4(kb3), m23 >> 16)});
            } else if constexpr (VT == TF16 || VT == TBW) {
                const char * col = vbase + (size_t)d * 2;
                auto kv4 = [&](const int kb) -> uint2 {
                    const char * bb = col + (size_t)(4*kb) * s.vnb1;
                    const uint32_t a0 = *reinterpret_cast<const uint16_t *>(bb);
                    const uint32_t a1 = *reinterpret_cast<const uint16_t *>(bb + s.vnb1);
                    const uint32_t a2 = *reinterpret_cast<const uint16_t *>(bb + 2*s.vnb1);
                    const uint32_t a3 = *reinterpret_cast<const uint16_t *>(bb + 3*s.vnb1);
                    return make_uint2(a0 | (a1 << 16), a2 | (a3 << 16));
                };
                vf[t] = __builtin_bit_cast(v16s, (uint2[4]){
                    mask_values(kv4(kb0), m01 & 0xffffu), mask_values(kv4(kb1), m01 >> 16),
                    mask_values(kv4(kb2), m23 & 0xffffu), mask_values(kv4(kb3), m23 >> 16)});
            } else {
                auto kv4 = [&](const int kb) -> uint2 {
                    uint32_t gg[4][c18_gat<VT>::N];
#pragma unroll
                    for (int j = 0; j < 4; ++j) { c18_gat_load<VT>(vbase + (size_t)(4*kb + j) * s.vnb1, w, 16*t + r, gg[j]); }
                    return make_uint2(c18_gat_pair<VT>(gg[0], gg[1]), c18_gat_pair<VT>(gg[2], gg[3]));
                };
                vf[t] = __builtin_bit_cast(v16s, (uint2[4]){
                    mask_values(kv4(kb0), m01 & 0xffffu), mask_values(kv4(kb1), m01 >> 16),
                    mask_values(kv4(kb2), m23 & 0xffffu), mask_values(kv4(kb3), m23 >> 16)});
            }
        }
    };
    constexpr int VRN = (VL == LX) ? 16 * c18_gat<VT>::N : c18_piece<VT>::N;
    auto v_raw = [&](const uint32_t * b, const uint32_t m01, const uint32_t m23, uint32_t * vr, bool & von) {
        if constexpr (VL == LX) {
            von = true;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int kb = blk_of(b, j);
#pragma unroll
                for (int c = 0; c < 4; ++c) { c18_gat_load<VT>(vbase + (size_t)(4*kb + c) * s.vnb1, w, 16*hi + r, vr + (4*j + c) * c18_gat<VT>::N); }
            }
        } else {   // LS
            const int kk = lane >> 1, p = lane & 1;
            von = cell_on(m01, m23, kk);
            if (!von) { return; }
            const int key = 4*blk_of(b, kk >> 2) + (kk & 3);
            c18_piece_load<VT>(vbase + (size_t) key * s.vnb1, w, p, vr);
        }
    };
    auto v_fin = [&](const uint32_t * vr, const bool von, const uint32_t m01, const uint32_t m23, v16s * vf) {
        if constexpr (VL == LX) {
            uint32_t o[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint2 v = make_uint2(c18_gat_pair<VT>(vr + (4*j) * c18_gat<VT>::N, vr + (4*j + 1) * c18_gat<VT>::N),
                                           c18_gat_pair<VT>(vr + (4*j + 2) * c18_gat<VT>::N, vr + (4*j + 3) * c18_gat<VT>::N));
                const uint2 mv = mask_values(v, j == 0 ? (m01 & 0xffffu) : j == 1 ? (m01 >> 16) : j == 2 ? (m23 & 0xffffu) : (m23 >> 16));
                o[2*j] = mv.x; o[2*j+1] = mv.y;
            }
            uint32_t x[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) { x[i] = c18_xchg(o[i]); }
            uint32_t v0[8], v1[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) { v0[i] = hi ? x[i] : o[i]; v1[i] = hi ? o[i] : x[i]; }
            vf[0] = __builtin_bit_cast(v16s, v0); vf[1] = __builtin_bit_cast(v16s, v1);
        } else {   // LS: transposed tile [32 dims][16 keys], row pitch 32 B, key half h at ((h ^ sw(dim)) << 4),
                   // sw(dim) = ((dim >> 2) ^ (dim >> 3)) & 1 (the two lanes of a key, dims i and 8 + i, land on different banks)
            const int kk = lane >> 1, p = lane & 1;
            uint32_t o[8];
            if (von) { c18_piece_deq<VT>(vr, w, o); }
            else {
#pragma unroll
                for (int i = 0; i < 8; ++i) { o[i] = 0; }
            }
            const int kh = (kk >> 3) ^ p, koff = (kk & 7) * 2;
            char * b0 = vtile + 256 * p + (kh << 4) + koff;
            char * b1 = vtile + 256 * p + ((kh ^ 1) << 4) + koff;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                char * bb = ((i >> 2) & 1) ? b1 : b0;
                const uint32_t wa = o[i >> 1], wb = o[4 + (i >> 1)];
                *reinterpret_cast<uint16_t *>(bb + 32 * i)       = (i & 1) ? (uint16_t) (wa >> 16) : (uint16_t) (wa & 0xffffu);   // dim 8p + i
                *reinterpret_cast<uint16_t *>(bb + 512 + 32 * i) = (i & 1) ? (uint16_t) (wb >> 16) : (uint16_t) (wb & 0xffffu);   // dim 16 + 8p + i
            }
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int dim = 16*t + r, sw = ((dim >> 2) ^ (dim >> 3)) & 1;
                const uint4 a = *reinterpret_cast<const uint4 *>(vtile + dim * 32 + ((0 ^ sw) << 4));
                const uint4 bq = *reinterpret_cast<const uint4 *>(vtile + dim * 32 + ((1 ^ sw) << 4));
                vf[t] = __builtin_bit_cast(v16s, (uint4[2]){a, bq});
            }
            __builtin_amdgcn_wave_barrier();
        }
    };

    uint32_t b[4] = { 0, 0, 0, 0 };
    uint32_t m01 = 0, m23 = 0;
    v16s kf[2];
    uint32_t kr[KRN > 0 ? KRN : 1];
    bool kon = false;
    if (nchunks > 0) {
        load_desc(0, b, m01, m23);
        if constexpr (KL == LD) { load_k_direct(b, m01, m23, kf); }
        else { k_raw(b, m01, m23, kr, kon); k_fin(kr, kon, kf); }
    }

    for (int c = 0; c < nchunks; ++c) {
        v8f sc[3];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int e = 0; e < 8; ++e) { sc[i][e] = 0.f; }
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                if constexpr (KBW) { sc[i] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(kf[t], qf[i][t], sc[i]); }
                else { sc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kf[t], qf[i][t], sc[i]); }
            }
        }
        v16s vf[2];
        uint32_t vr[VRN > 0 ? VRN : 1];
        bool von = false;
        if constexpr (VL == LP) {
            const int kb0 = blk_of(b, 0), kb1 = blk_of(b, 1), kb2 = blk_of(b, 2), kb3 = blk_of(b, 3);
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int d = 32*w + 16*t + r;
                const uint2 v0 = mask_values(*reinterpret_cast<const uint2 *>(pvg + (size_t) kb0 * 1024 + d * 4), m01 & 0xffffu);
                const uint2 v1 = mask_values(*reinterpret_cast<const uint2 *>(pvg + (size_t) kb1 * 1024 + d * 4), m01 >> 16);
                const uint2 v2 = mask_values(*reinterpret_cast<const uint2 *>(pvg + (size_t) kb2 * 1024 + d * 4), m23 & 0xffffu);
                const uint2 v3 = mask_values(*reinterpret_cast<const uint2 *>(pvg + (size_t) kb3 * 1024 + d * 4), m23 >> 16);
                vf[t] = __builtin_bit_cast(v16s, (uint2[4]){v0, v1, v2, v3});
            }
        } else if constexpr (VL == LD) {
            load_v_direct(b, m01, m23, vf);
        } else {
            v_raw(b, m01, m23, vr, von);
        }
        uint32_t nb[4] = { 0, 0, 0, 0 };
        uint32_t nm01 = 0, nm23 = 0;
        v16s kfn[2];
        uint32_t nkr[KRN > 0 ? KRN : 1];
        bool nkon = false;
        if constexpr (KL == LD) {
            if (c + 1 < nchunks) { load_desc(c + 1, nb, nm01, nm23); load_k_direct(nb, nm01, nm23, kfn); }
            else { kfn[0] = kf[0]; kfn[1] = kf[1]; }
        } else {
            if (c + 1 < nchunks) { load_desc(c + 1, nb, nm01, nm23); k_raw(nb, nm01, nm23, nkr, nkon); }
        }
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            if (i != w) {
                part[i][w][lane*2+0] = make_float4(sc[i][0], sc[i][1], sc[i][2], sc[i][3]);
                part[i][w][lane*2+1] = make_float4(sc[i][4], sc[i][5], sc[i][6], sc[i][7]);
            }
        }
        __syncthreads();
        if (w < 3) {
            float scf[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) { scf[e] = 0.f; }
#pragma unroll
            for (int w2 = 0; w2 < 8; ++w2) {
                if (w2 == w) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) { scf[e] += sc[w][e]; }
                } else {
                    const float4 a = part[w][w2][lane*2+0], bq = part[w][w2][lane*2+1];
                    scf[0] += a.x; scf[1] += a.y; scf[2] += a.z; scf[3] += a.w; scf[4] += bq.x; scf[5] += bq.y; scf[6] += bq.z; scf[7] += bq.w;
                }
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int j = e >> 1, cc = 2*(e & 1) + hi;
                const uint32_t mk = ((j < 2 ? m01 : m23) >> (16 * (j & 1))) & 0xFFFFu;
                const bool kv = (mk >> (own_qi * 4 + cc)) & 1u;
                if (!kv) { scf[e] = -INFINITY; }
            }
            float mloc = scf[0];
#pragma unroll
            for (int e = 1; e < 8; ++e) { mloc = fmaxf(mloc, scf[e]); }
            mloc = fmaxf(mloc, __shfl_xor(mloc, 16));
            const float mnew  = fmaxf(m, mloc);
            const float alpha = mnew == -INFINITY ? 1.f : exp2f((m - mnew) * QSA3_L2E);
            float p[8], ls = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) { p[e] = mnew == -INFINITY ? 0.f : exp2f((scf[e] - mnew) * QSA3_L2E); ls += p[e]; }
            ls += __shfl_xor(ls, 16);
            l = l * alpha + ls;
            m = mnew;
            uint32_t pp[4], po[4];
#pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) {
                if constexpr (VBW) { pp[e2] = (uint32_t) c18_f2bf(p[2*e2]) | ((uint32_t) c18_f2bf(p[2*e2+1]) << 16); }
                else { pp[e2] = (uint32_t) qsa3_f2h(p[2*e2]) | ((uint32_t) qsa3_f2h(p[2*e2+1]) << 16); }
                po[e2] = (uint32_t) __shfl_xor((int) pp[e2], 16);
            }
            uint16_t ph[16];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint16_t mine  = (e & 1) ? (uint16_t) (pp[e>>1] >> 16) : (uint16_t) (pp[e>>1] & 0xffffu);
                const uint16_t other = (e & 1) ? (uint16_t) (po[e>>1] >> 16) : (uint16_t) (po[e>>1] & 0xffffu);
                ph[2*e]   = hi ? other : mine;
                ph[2*e+1] = hi ? mine  : other;
            }
            if (hi == 0) {
                ptile[w][r*2+0] = make_uint4((uint32_t) ph[0] | ((uint32_t) ph[1] << 16), (uint32_t) ph[2] | ((uint32_t) ph[3] << 16),
                                             (uint32_t) ph[4] | ((uint32_t) ph[5] << 16), (uint32_t) ph[6] | ((uint32_t) ph[7] << 16));
                ptile[w][r*2+1] = make_uint4((uint32_t) ph[8] | ((uint32_t) ph[9] << 16), (uint32_t) ph[10] | ((uint32_t) ph[11] << 16),
                                             (uint32_t) ph[12] | ((uint32_t) ph[13] << 16), (uint32_t) ph[14] | ((uint32_t) ph[15] << 16));
                alpha_s[own_row] = alpha;
            }
        }
        __syncthreads();
        float al[3];
#pragma unroll
        for (int i = 0; i < 3; ++i) { al[i] = alpha_s[16*i + r]; }
        if (__ballot(al[0] != 1.f || al[1] != 1.f || al[2] != 1.f)) {
#pragma unroll
            for (int i = 0; i < 3; ++i)
#pragma unroll
                for (int t = 0; t < 2; ++t)
#pragma unroll
                    for (int e = 0; e < 8; ++e) { O[i][t][e] *= al[i]; }
        }
        if constexpr (VL == LX || VL == LS) { v_fin(vr, von, m01, m23, vf); }
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const v16s pf = __builtin_bit_cast(v16s, (uint4[2]){ptile[i][r*2+0], ptile[i][r*2+1]});
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                if constexpr (VBW) { O[i][t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(vf[t], pf, O[i][t]); }
                else { O[i][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vf[t], pf, O[i][t]); }
            }
        }
        if constexpr (KL == LD) { kf[0] = kfn[0]; kf[1] = kfn[1]; }
        else { k_fin(nkr, nkon, kf); }   // unconditional: nkon is false past the last chunk (zeros, unused); keeps kf dead across the body
        b[0] = nb[0]; b[1] = nb[1]; b[2] = nb[2]; b[3] = nb[3]; m01 = nm01; m23 = nm23;
    }

    if (w < 3 && hi == 0) { l_s[own_row] = l; }
    __syncthreads();
    float * ostage = reinterpret_cast<float *>(&part[0][0][0]) + w * (16 * 32);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const float li = l_s[16*i + r];
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) { ostage[r * 32 + 16*t + 2*e + hi] = li > 0.f ? O[i][t][e] / li : 0.f; }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int rr = 4*c + (lane >> 3), dq = 4 * (lane & 7);
            const int row = 16*i + rr, qi = row / 12, h = row - 12*qi;
            const int query = QSA3_G * g + qi;
            if (query < s.n_q) {
                const int hq = kvh * s.gqa + h;
                const float4 v = *reinterpret_cast<const float4 *>(ostage + rr * 32 + dq);
                float * dst = reinterpret_cast<float *>(reinterpret_cast<char *>(out) + (size_t) query * s.o2 + (size_t) hq * s.o1) + 32*w + dq;
                *reinterpret_cast<float4 *>(dst) = v;
            }
        }
        __syncthreads();
    }
#endif
}

// pack V of type T into the PR packed layout [kvh][quad][dim][4 keys] as f16 (bf16 bits for TBW), PR qsa_pack_v indexing
template<int T> static __global__ void c18_pack_v(const char * src, uint16_t * dst, size_t nb1, size_t nb2, int nk, uint32_t count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t d = i % 256, group = i / 256, head = group / (nk / 4), key = (group % (nk / 4)) * 4;
    const char * in = src + size_t(head) * nb2 + size_t(key) * nb1;
    uint16_t h[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if constexpr (T == TQ8) {
            const auto * blk = reinterpret_cast<const block_q8_0 *>(in + j * nb1) + d / 32;
            h[j] = __builtin_bit_cast(uint16_t, (_Float16) ((_Float16) blk->qs[d % 32] * __builtin_bit_cast(_Float16, blk->d)));
        } else if constexpr (T == TQ4) {
            const auto * blk = reinterpret_cast<const block_q4_0 *>(in + j * nb1) + d / 32;
            const int qv = (d % 32) < 16 ? (blk->qs[d % 16] & 15) : (blk->qs[d % 16] >> 4);
            h[j] = __builtin_bit_cast(uint16_t, (_Float16) ((_Float16) (qv - 8) * __builtin_bit_cast(_Float16, blk->d)));
        } else if constexpr (T == TBF) {
            h[j] = c18_bf1_to_h(*reinterpret_cast<const uint16_t *>(in + j * nb1 + d * 2));
        } else {
            h[j] = *reinterpret_cast<const uint16_t *>(in + j * nb1 + d * 2);
        }
    }
    *reinterpret_cast<ushort4 *>(dst + size_t(group) * 1024 + d * 4) = make_ushort4(h[0], h[1], h[2], h[3]);
}
#endif // C18_HELPERS_ONLY
