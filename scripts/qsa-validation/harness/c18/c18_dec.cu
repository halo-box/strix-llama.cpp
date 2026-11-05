// C18 (decode route): loader microbench for the PR's qsa_decode_wmma_partial (the n_q <= 512 route, i.e. tg and ub <= 512).
// hipcc --offload-arch=gfx1151 -O3 -std=c++17 c18_dec.cu -o c18_dec
// usage: c18_dec <rows_file> <qkv_file|SYN> n_q reps layers variants
//   rows: C5 capture at ub4096; the first n_q queries of each layer are used (they are the queries of a ub=n_q ubatch at the
//   same pos0; same causal context). Launch: the PR's (qsa-decode.cu:136-158): unique ids, step 64, chunk/splits rule.
// Kernels: the PR's qsa_decode_wmma_partial<q8> verbatim (pr_dec.inc) for f16 / q8_0 as references, and
// c18_dec_kernel<T, KL, VL> = the same body with the K and V loaders as policies (c18_kernel.inc piece/gather helpers):
//   K: LD = PR (per lane, per t: f16 2 x b128; q8_0 8 x u16 qs + d), LX = split + v_permlanex16, LS = wave-private LDS tile
//   V: LD = PR (per lane 16 keys x 2 dims; q8_0 pairs dims r and 16+r), LX = split gather + permlanex16, LS = LDS transposed
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <set>
#include <map>
#include <random>

#define RDNA3
#define NO_DEVICE_CODE
#define QK8_0 32
typedef struct { uint16_t d; int8_t qs[QK8_0]; } block_q8_0;
static __device__ __forceinline__ uint16_t ggml_cuda_f32_to_act16(const float f, const bool) { return __builtin_bit_cast(uint16_t, (_Float16) f); }
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f  __attribute__((ext_vector_type(8)));
typedef _Float16 qsa3_half2 __attribute__((ext_vector_type(2)));
#define QSA3_BLK_SENT 0xFFFFFFFFu
#define QSA3_G 4
struct qsa3_layout { size_t q1, q2, o1, o2, m1; size_t knb1, vnb1, knb2, vnb2; int nk, n_q, gqa, cap; float scale; int hq, gs; };
static __device__ __forceinline__ uint16_t qsa3_f2h(const float f) { return __builtin_bit_cast(uint16_t, (_Float16) f); }
static __device__ __forceinline__ uint32_t qsa3_q8x2_to_h2(const uint32_t x, const qsa3_half2 d2) {
    const qsa3_half2 v = __builtin_bit_cast(qsa3_half2, ((x ^ 0x00800080u) & 0x00FF00FFu) | 0x64006400u);
    return __builtin_bit_cast(uint32_t, (v - qsa3_half2{(_Float16) 1152, (_Float16) 1152}) * d2);
}
#define C18_HELPERS_ONLY
#include "c18_kernel.inc"
#include "pr_dec.inc"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template <int T, int KL, int VL>
static __global__ __launch_bounds__(256) void c18_dec_kernel(
        const char * q, const char * k, const char * v, const char * ids,
        size_t q1, size_t q2, size_t k1, size_t k2, size_t v1, size_t v2, size_t i1,
        int nk, int ns, int nh, int splits, int chunk, float scale, float * partial) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3)
    NO_DEVICE_CODE;
#else
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, r = lane & 15, hi = lane >> 4;
    const int kvh = blockIdx.x, query = blockIdx.y, split = blockIdx.z;
    const int * ir = (const int *) (ids + query*i1);
    __shared__ union {
        uint16_t query_stage[8][16][32];
        float4 scores[8][64];
    } shared;
    __shared__ __align__(16) uint16_t probabilities[16][16];
    __shared__ float alpha_s[16];
    char * tile = reinterpret_cast<char *>(&shared.scores[w][0]);   // wave-private 1 KB slot (free outside the score exchange)
    const char * kb_ = k + kvh*k2;
    const char * vb_ = v + kvh*v2;
    v16s qf[2];
    const _Float16 hs = (_Float16) scale;
    for (int j = lane; j < 16*32; j += 32) {
        const int head = j/32, d = j%32;
        const float value = head < 12 ? *(const float *) (q + query*q1 + (kvh*12+head)*q2 + (w*32+d)*4) : 0.0f;
        shared.query_stage[w][head][d] = __builtin_bit_cast(uint16_t, (_Float16)value*hs);
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        const uint16_t * src = &shared.query_stage[w][r][t*16];
        qf[t] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)src, *(const uint4 *)(src+8)});
    }
    __syncthreads();
    v8f output[2] = {};
    float maximum = -INFINITY, normalizer = 0.0f;
    const int begin = split*chunk, end = min(ns, begin+chunk);
    for (int start = begin; start < end; start += 16) {
        int keys[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            int key = start+j < end ? ir[start+j] : -1;
            if (key < 0 || key >= nk) { key = -1; }
            keys[j] = key;
        }
        v8f scores = {};
        v16s kf[2];
        if constexpr (KL == LD) {
            const int key = keys[r];
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                kf[t] = v16s{};
                if (key >= 0) {
                    if constexpr (T == TQ8) {
                        const auto * blk = (const block_q8_0 *) (kb_ + key*k1) + w;
                        const _Float16 d = __builtin_bit_cast(_Float16, blk->d);
                        const qsa3_half2 d2 = qsa3_half2{d, d};
                        const uint16_t * qs = (const uint16_t *) blk->qs + 8*t;
                        uint32_t h[8];
#pragma unroll
                        for (int j = 0; j < 8; ++j) { const uint32_t x = qs[j]; h[j] = qsa3_q8x2_to_h2((x & 0xFFu) | ((x & 0xFF00u) << 8), d2); }
                        kf[t] = __builtin_bit_cast(v16s, h);
                    } else if constexpr (T == TF16) {
                        const uint16_t * src = (const uint16_t *) (kb_ + key*k1) + 32*w+16*t;
                        kf[t] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)src, *(const uint4 *)(src+8)});
                    } else {   // q4_0 / bf16: the two pieces of this t (dims 16t + 0..7 = piece 0 half t, 16t + 8..15 = piece 1 half t)
                        uint32_t r0[c18_piece<T>::N], r1[c18_piece<T>::N], o0[8], o1[8];
                        c18_piece_load<T>(kb_ + key*k1, w, 0, r0); c18_piece_load<T>(kb_ + key*k1, w, 1, r1);
                        c18_piece_deq<T>(r0, w, o0); c18_piece_deq<T>(r1, w, o1);
                        const uint32_t hh[8] = {o0[4*t], o0[4*t+1], o0[4*t+2], o0[4*t+3], o1[4*t], o1[4*t+1], o1[4*t+2], o1[4*t+3]};
                        kf[t] = __builtin_bit_cast(v16s, hh);
                    }
                }
            }
        } else {
            int kk, p;
            if constexpr (KL == LS) { kk = lane >> 1; p = lane & 1; } else { kk = r; p = hi; }
            const int key = keys[kk];
            uint32_t o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (key >= 0) { uint32_t raw[c18_piece<T>::N]; c18_piece_load<T>(kb_ + key*k1, w, p, raw); c18_piece_deq<T>(raw, w, o); }
            if constexpr (KL == LX) {
                uint32_t x[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) { x[i] = c18_xchg(o[i]); }
                uint32_t k0[8], k1v[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    k0[i] = hi ? x[i] : o[i];     k0[4+i] = hi ? o[i] : x[i];
                    k1v[i] = hi ? x[4+i] : o[4+i]; k1v[4+i] = hi ? o[4+i] : x[4+i];
                }
                kf[0] = __builtin_bit_cast(v16s, k0); kf[1] = __builtin_bit_cast(v16s, k1v);
            } else {
                const int sw = (kk >> 1) & 3;
                *reinterpret_cast<uint4 *>(tile + kk * 64 + ((p ^ sw) << 4))       = make_uint4(o[0], o[1], o[2], o[3]);
                *reinterpret_cast<uint4 *>(tile + kk * 64 + (((2 + p) ^ sw) << 4)) = make_uint4(o[4], o[5], o[6], o[7]);
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
                const int rs = (r >> 1) & 3;
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    const uint4 a = *reinterpret_cast<const uint4 *>(tile + r * 64 + (((2*t) ^ rs) << 4));
                    const uint4 bq = *reinterpret_cast<const uint4 *>(tile + r * 64 + (((2*t + 1) ^ rs) << 4));
                    kf[t] = __builtin_bit_cast(v16s, (uint4[2]){a, bq});
                }
                __builtin_amdgcn_wave_barrier();
            }
        }
#pragma unroll
        for (int t = 0; t < 2; ++t) { scores = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kf[t], qf[t], scores); }
        shared.scores[w][lane*2] = make_float4(scores[0],scores[1],scores[2],scores[3]);
        shared.scores[w][lane*2+1] = make_float4(scores[4],scores[5],scores[6],scores[7]);
        __syncthreads();
        if (w == 0) {
            float score[8] = {};
#pragma unroll
            for (int sw = 0; sw < 8; ++sw) {
                const float4 lo = shared.scores[sw][lane*2], high = shared.scores[sw][lane*2+1];
                score[0]+=lo.x;score[1]+=lo.y;score[2]+=lo.z;score[3]+=lo.w;
                score[4]+=high.x;score[5]+=high.y;score[6]+=high.z;score[7]+=high.w;
            }
            float local = -INFINITY;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int key = keys[2*e+hi];
                if (key < 0 || key >= nk) { score[e] = -INFINITY; }
                local = fmaxf(local,score[e]);
            }
            local = fmaxf(local,__shfl_xor(local,16));
            const float next = fmaxf(maximum,local);
            const float alpha = next == -INFINITY ? 1.0f : exp2f((maximum-next)*1.4426950408889634f);
            float sum = 0.0f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float p = next == -INFINITY ? 0.0f : exp2f((score[e]-next)*1.4426950408889634f);
                sum += p;
                probabilities[r][2*e+hi] = __builtin_bit_cast(uint16_t,(_Float16)p);
            }
            sum += __shfl_xor(sum,16);
            normalizer = normalizer*alpha+sum;
            maximum = next;
            if (hi == 0) { alpha_s[r] = alpha; }
        }
        __syncthreads();
        const float alpha = alpha_s[r];
        const v16s pf = __builtin_bit_cast(v16s, (uint4[2]){
            *(const uint4 *)&probabilities[r][0], *(const uint4 *)&probabilities[r][8]});
        v16s vf[2];
        if constexpr (VL == LD) {
            if constexpr (T == TQ8) {   // PR: dims r and 16 + r of the same block convert together
                uint32_t vpair[16] = {};
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const int key = keys[j];
                    if (key >= 0) {
                        const auto * blk = (const block_q8_0 *) (vb_ + key*v1) + w;
                        const _Float16 d = __builtin_bit_cast(_Float16, blk->d);
                        vpair[j] = qsa3_q8x2_to_h2((uint32_t) (uint8_t) blk->qs[r] | ((uint32_t) (uint8_t) blk->qs[16+r] << 16), qsa3_half2{d, d});
                    }
                }
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    uint16_t values[16];
#pragma unroll
                    for (int j = 0; j < 16; ++j) { values[j] = (uint16_t) (vpair[j] >> (16*t)); }
                    vf[t] = __builtin_bit_cast(v16s, values);
                }
            } else {
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    uint32_t g[16], hh[8];
#pragma unroll
                    for (int j = 0; j < 16; ++j) { g[j] = 0; if (keys[j] >= 0) { c18_gat_load<T>(vb_ + keys[j]*v1, w, 16*t + r, &g[j]); } }
#pragma unroll
                    for (int j = 0; j < 8; ++j) { hh[j] = c18_gat_pair<T>(&g[2*j], &g[2*j+1]); if (keys[2*j] < 0) hh[j] &= 0xffff0000u; if (keys[2*j+1] < 0) hh[j] &= 0x0000ffffu; }
                    vf[t] = __builtin_bit_cast(v16s, hh);
                }
            }
        } else if constexpr (VL == LX) {
            uint32_t g[16], o[8];
#pragma unroll
            for (int j = 0; j < 16; ++j) { g[j] = 0; if (keys[j] >= 0) { c18_gat_load<T>(vb_ + keys[j]*v1, w, 16*hi + r, &g[j]); } }
#pragma unroll
            for (int j = 0; j < 8; ++j) { o[j] = c18_gat_pair<T>(&g[2*j], &g[2*j+1]); if (keys[2*j] < 0) o[j] &= 0xffff0000u; if (keys[2*j+1] < 0) o[j] &= 0x0000ffffu; }
            uint32_t x[8], a0[8], a1[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) { x[i] = c18_xchg(o[i]); a0[i] = hi ? x[i] : o[i]; a1[i] = hi ? o[i] : x[i]; }
            vf[0] = __builtin_bit_cast(v16s, a0); vf[1] = __builtin_bit_cast(v16s, a1);
        } else {   // LS: as the prefill V tile (see c18_kernel.inc)
            const int kk = lane >> 1, p = lane & 1;
            const int key = keys[kk];
            uint32_t o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (key >= 0) { uint32_t raw[c18_piece<T>::N]; c18_piece_load<T>(vb_ + key*v1, w, p, raw); c18_piece_deq<T>(raw, w, o); }
            const int kh = (kk >> 3) ^ p, koff = (kk & 7) * 2;
            char * b0 = tile + 256 * p + (kh << 4) + koff;
            char * b1 = tile + 256 * p + ((kh ^ 1) << 4) + koff;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                char * bb = ((i >> 2) & 1) ? b1 : b0;
                const uint32_t wa = o[i >> 1], wb = o[4 + (i >> 1)];
                *reinterpret_cast<uint16_t *>(bb + 32 * i)       = (i & 1) ? (uint16_t) (wa >> 16) : (uint16_t) (wa & 0xffffu);
                *reinterpret_cast<uint16_t *>(bb + 512 + 32 * i) = (i & 1) ? (uint16_t) (wb >> 16) : (uint16_t) (wb & 0xffffu);
            }
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int dim = 16*t + r, sw = ((dim >> 2) ^ (dim >> 3)) & 1;
                const uint4 a = *reinterpret_cast<const uint4 *>(tile + dim * 32 + ((0 ^ sw) << 4));
                const uint4 bq = *reinterpret_cast<const uint4 *>(tile + dim * 32 + ((1 ^ sw) << 4));
                vf[t] = __builtin_bit_cast(v16s, (uint4[2]){a, bq});
            }
            __builtin_amdgcn_wave_barrier();
        }
#pragma unroll
        for (int t = 0; t < 2; ++t) {
#pragma unroll
            for (int e = 0; e < 8; ++e) { output[t][e] *= alpha; }
            output[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vf[t],pf,output[t]);
        }
        __syncthreads();
    }
    if (r < 12) {
        float * dst = partial + ((size_t(query)*nh+kvh*12+r)*splits+split)*258;
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) { dst[32*w+16*t+2*e+hi] = output[t][e]; }
        if (w == 0 && hi == 0) { dst[256] = maximum; dst[257] = normalizer; }
    }
#endif
}

struct hdr { char name[64]; int32_t n_q, ns, nk, has_mask, pos0, kt; };
struct qhdr { char name[64]; int32_t which, type; int64_t ne[4], nb[4]; uint64_t nbytes; };
static float h2f(uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }
static uint16_t f2h(float f) { return __builtin_bit_cast(uint16_t, (_Float16) f); }
static uint16_t f2bf(float f) { uint32_t u = __builtin_bit_cast(uint32_t, f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t) (u >> 16); }
static float bf2f(uint16_t b) { return __builtin_bit_cast(float, (uint32_t) b << 16); }
static const int HB[4] = {512, 272, 144, 512};
static const char * TN[4] = {"f16", "q8_0", "q4_0", "bf16"};
static const char * LN[4] = {"D", "Dd", "X", "S"};
static void build_cache(int T, const std::vector<uint16_t> & f, int nk, std::vector<uint8_t> & out) {
    const size_t rowb = 2 * (size_t) HB[T];
    out.assign((size_t) nk * rowb, 0);
    for (size_t kh = 0; kh < (size_t) nk * 2; ++kh) {
        const uint16_t * x = f.data() + kh * 256; uint8_t * o = out.data() + kh * HB[T];
        if (T == TF16) memcpy(o, x, 512);
        else if (T == TBF) { uint16_t * ob = (uint16_t *) o; for (int d = 0; d < 256; ++d) ob[d] = f2bf(h2f(x[d])); }
        else if (T == TQ8) { block_q8_0 * bq = (block_q8_0 *) o;
            for (int b = 0; b < 8; ++b) { float am = 0; for (int i = 0; i < 32; ++i) am = std::max(am, std::fabs(h2f(x[32*b+i])));
                const float d = am / 127.f, id = d ? 1.f / d : 0.f; bq[b].d = f2h(d); for (int i = 0; i < 32; ++i) bq[b].qs[i] = (int8_t) roundf(h2f(x[32*b+i]) * id); } }
        else { block_q4_0 * bq = (block_q4_0 *) o;
            for (int b = 0; b < 8; ++b) { float am = 0, mx = 0; for (int i = 0; i < 32; ++i) { const float v = h2f(x[32*b+i]); if (am < std::fabs(v)) { am = std::fabs(v); mx = v; } }
                const float d = mx / -8.f, id = d ? 1.f / d : 0.f; bq[b].d = f2h(d);
                for (int j = 0; j < 16; ++j) { const uint8_t a = (uint8_t) std::min(15, (int) (int8_t) (h2f(x[32*b+j]) * id + 8.5f)); const uint8_t c = (uint8_t) std::min(15, (int) (int8_t) (h2f(x[32*b+16+j]) * id + 8.5f)); bq[b].qs[j] = a | (c << 4); } } }
    }
}
static float cache_val(int T, const std::vector<uint8_t> & B, int key, int kvh, int d) {
    const uint8_t * o = B.data() + ((size_t) key * 2 + kvh) * HB[T];
    if (T == TF16) return h2f(((const uint16_t *) o)[d]);
    if (T == TBF) { float v = bf2f(((const uint16_t *) o)[d]); v = std::min(65504.f, std::max(-65504.f, v)); return (float) (_Float16) v; }
    if (T == TQ8) { const block_q8_0 * bq = (const block_q8_0 *) o + d / 32; return (float) (_Float16) ((float) bq->qs[d % 32] * h2f(bq->d)); }
    const block_q4_0 * bq = (const block_q4_0 *) o + d / 32; const int j = d % 32;
    const int qv = j < 16 ? (bq->qs[j] & 15) : (bq->qs[j - 16] >> 4); return (float) (_Float16) ((float) (qv - 8) * h2f(bq->d));
}
typedef void (*dfn_t)(const char *, const char *, const char *, const char *, size_t, size_t, size_t, size_t, size_t, size_t, size_t, int, int, int, int, int, float, float *);
struct variant { std::string name, group; int KT, KL, VT, VL; dfn_t f; };

int main(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage: rows qkv|SYN n_q reps layers variants\n"); return 1; }
    const std::string qkvf = argv[2]; const int NQ = atoi(argv[3]), reps = atoi(argv[4]);
    std::set<int> layers; std::set<std::string> vsel;
    { const std::string a = argv[5]; if (a != "all") { std::string x; for (char c : a + ",") { if (c == ',') { layers.insert(atoi(x.c_str())); x.clear(); } else x += c; } } }
    { const std::string a = argv[6]; std::string x; for (char c : a + ",") { if (c == ',') { vsel.insert(x); x.clear(); } else x += c; } }
    std::vector<variant> V;
    // mixed K/V types: the kernel template takes one type for both; K sweep and V sweep use same-type pairs with the other side's
    // loader held at the PR's LD, so K-loader and V-loader costs are separated by the loader switch (see MEASUREMENTS C18)
#define DV(T, KL, VL, g) V.push_back({std::string("K") + TN[T] + LN[KL] + "-V" + TN[T] + LN[VL], g, T, KL, T, VL, c18_dec_kernel<T, KL, VL>})
    DV(TF16, LD, LD, "base"); DV(TF16, LX, LD, "k"); DV(TF16, LS, LD, "k"); DV(TF16, LD, LX, "v"); DV(TF16, LD, LS, "v"); DV(TF16, LX, LX, "p"); DV(TF16, LX, LS, "p"); DV(TF16, LS, LS, "p");
    DV(TQ8, LD, LD, "base");  DV(TQ8, LX, LD, "k");  DV(TQ8, LS, LD, "k");  DV(TQ8, LD, LX, "v");  DV(TQ8, LD, LS, "v");  DV(TQ8, LX, LX, "p");  DV(TQ8, LX, LS, "p");  DV(TQ8, LS, LS, "p");
    DV(TQ4, LD, LD, "base");  DV(TQ4, LX, LD, "k");  DV(TQ4, LS, LD, "k");  DV(TQ4, LD, LX, "v");  DV(TQ4, LD, LS, "v");  DV(TQ4, LX, LX, "p");  DV(TQ4, LX, LS, "p");  DV(TQ4, LS, LS, "p");
    DV(TBF, LD, LD, "base");  DV(TBF, LX, LD, "k");  DV(TBF, LS, LD, "k");  DV(TBF, LD, LX, "v");  DV(TBF, LD, LS, "v");  DV(TBF, LX, LX, "p");  DV(TBF, LX, LS, "p");  DV(TBF, LS, LS, "p");
    V.push_back({"PR-dec-f16", "pr", TF16, LD, TF16, LD, nullptr}); V.push_back({"PR-dec-q8_0", "pr", TQ8, LD, TQ8, LD, nullptr});
    FILE * fr = fopen(argv[1], "rb"); if (!fr) { perror("rows"); return 1; }
    FILE * fq = qkvf == "SYN" ? nullptr : fopen(qkvf.c_str(), "rb");
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    printf("# c18_dec rows %s data %s n_q %d reps %d variants %zu\n", argv[1], qkvf.c_str(), NQ, reps, V.size());
    std::map<std::string, double> sum_med; int n_layers = 0; int L = -1;
    hdr h;
    while (fread(&h, sizeof(h), 1, fr) == 1) {
        ++L;
        std::vector<int32_t> rows((size_t) h.n_q * h.ns);
        if (fread(rows.data(), 4, rows.size(), fr) != rows.size()) { fprintf(stderr, "short rows\n"); return 1; }
        std::vector<float> Qh; std::vector<uint16_t> K16, V16; const int nk = h.nk, ns = h.ns;
        if (fq) {
            for (int i = 0; i < 3; ++i) {
                qhdr qh; if (fread(&qh, sizeof(qh), 1, fq) != 1) { fprintf(stderr, "short qkv\n"); return 1; }
                std::vector<uint8_t> B(qh.nbytes); if (fread(B.data(), 1, qh.nbytes, fq) != qh.nbytes) { fprintf(stderr, "short qkv data\n"); return 1; }
                if (i == 0) { Qh.resize((size_t) NQ * 24 * 256); memcpy(Qh.data(), B.data(), Qh.size() * 4); }
                else { std::vector<uint16_t> & D = i == 1 ? K16 : V16; D.resize((size_t) nk * 512); memcpy(D.data(), B.data(), D.size() * 2); }
            }
        }
        if (!layers.empty() && !layers.count(L)) continue;
        if (!fq) {
            std::mt19937_64 rg(0xC18ULL + (uint64_t) L * 7919ULL + (uint64_t) nk);
            auto u = [&]() { return (float) ((rg() >> 40) * (2.0 / 16777216.0) - 1.0) * 1.7f; };
            Qh.resize((size_t) NQ * 24 * 256); for (auto & x : Qh) x = u();
            K16.resize((size_t) nk * 512); V16.resize((size_t) nk * 512); for (auto & x : K16) x = f2h(u()); for (auto & x : V16) x = f2h(u());
        }
        std::vector<uint8_t> KC[4], VC[4]; char * dK[4] = {}, * dV[4] = {};
        for (int T = 0; T < 4; ++T) { build_cache(T, K16, nk, KC[T]); build_cache(T, V16, nk, VC[T]);
            CK(hipMalloc(&dK[T], KC[T].size())); CK(hipMemcpy(dK[T], KC[T].data(), KC[T].size(), hipMemcpyHostToDevice));
            CK(hipMalloc(&dV[T], VC[T].size())); CK(hipMemcpy(dV[T], VC[T].data(), VC[T].size(), hipMemcpyHostToDevice)); }
        // PR launch (qsa-decode.cu:136-158)
        int32_t * d_ids, * d_uids; float * d_q, * d_part;
        CK(hipMalloc(&d_ids, (size_t) NQ * ns * 4)); CK(hipMemcpy(d_ids, rows.data(), (size_t) NQ * ns * 4, hipMemcpyHostToDevice));
        CK(hipMalloc(&d_uids, (size_t) NQ * ns * 4));
        hipLaunchKernelGGL(qsa_decode_unique_ids, dim3(NQ), dim3(256), 0, 0, (const char *) d_ids, (size_t) ns * 4, d_uids, ns, nk);
        CK(hipGetLastError()); CK(hipDeviceSynchronize());
        std::vector<int32_t> uids((size_t) NQ * ns); CK(hipMemcpy(uids.data(), d_uids, uids.size() * 4, hipMemcpyDeviceToHost));
        const int step = 64, nh = 24;
        int chunk = step;
        if (NQ > 8) { const int target = std::max(1, (8 * ((ns + step - 1) / step) + NQ - 1) / NQ); chunk = ((ns + target - 1) / target + 15) / 16 * 16; }
        const int splits = (ns + chunk - 1) / chunk;
        CK(hipMalloc(&d_q, Qh.size() * 4)); CK(hipMemcpy(d_q, Qh.data(), Qh.size() * 4, hipMemcpyHostToDevice));
        const size_t pn = (size_t) NQ * nh * splits * 258;
        CK(hipMalloc(&d_part, pn * 4));
        const size_t q1 = 24 * 1024, q2 = 1024;
        auto launch = [&](int i) {
            const variant & v = V[i]; const int T = v.KT; const size_t k1 = 2 * (size_t) HB[T], k2 = HB[T];
            const dim3 gr(2, NQ, splits), bl(256);
            if (!v.f) {
                if (T == TQ8) hipLaunchKernelGGL(qsa_decode_wmma_partial<true>, gr, bl, 0, 0, (const char *) d_q, dK[T], dV[T], (const char *) nullptr, (const char *) d_uids, q1, q2, k1, k2, k1, k2, (size_t) 0, (size_t) ns * 4, nk, ns, nh, splits, chunk, 1.0f / 16.0f, d_part);
                else          hipLaunchKernelGGL(qsa_decode_wmma_partial<false>, gr, bl, 0, 0, (const char *) d_q, dK[T], dV[T], (const char *) nullptr, (const char *) d_uids, q1, q2, k1, k2, k1, k2, (size_t) 0, (size_t) ns * 4, nk, ns, nh, splits, chunk, 1.0f / 16.0f, d_part);
            } else hipLaunchKernelGGL(v.f, gr, bl, 0, 0, (const char *) d_q, dK[T], dV[T], (const char *) d_uids, q1, q2, k1, k2, k1, k2, (size_t) ns * 4, nk, ns, nh, splits, chunk, 1.0f / 16.0f, d_part);
            CK(hipGetLastError());
        };
        std::vector<int> vl; for (int i = 0; i < (int) V.size(); ++i) if (vsel.count("all") || vsel.count(V[i].name) || V[i].group == "base" || V[i].group == "pr") vl.push_back(i);
        // correctness: merged output (partials -> softmax combine on the host) of the first variant per type vs CPU f64 (first layer,
        // 3 queries x 24 heads); every other variant bitwise on the raw partial buffer
        std::vector<float> o(pn); std::map<int, std::vector<float>> first;
        for (int i : vl) {
            CK(hipMemset(d_part, 0xff, pn * 4)); launch(i); CK(hipDeviceSynchronize()); CK(hipMemcpy(o.data(), d_part, pn * 4, hipMemcpyDeviceToHost));
            const variant & v = V[i]; const int T = v.KT;
            if (!first.count(T)) {
                first[T] = o; double maxrel = -1;
                if (n_layers == 0) {
                    maxrel = 0; const int qs_s[3] = {0, NQ / 2, NQ - 1};
                    for (int qq : qs_s) for (int hq = 0; hq < 24; ++hq) {
                        const int kvh = hq / 12; const float * qv = Qh.data() + ((size_t) qq * 24 + hq) * 256;
                        std::set<int> cells; for (int k = 0; k < ns; ++k) { const int c = rows[(size_t) qq * ns + k]; if (c >= 0 && c < nk) cells.insert(c); }
                        std::vector<double> sc; double mx = -1e300;
                        for (int c : cells) { double s = 0; for (int d = 0; d < 256; ++d) s += (double) (float) ((_Float16) qv[d] * (_Float16) (1.0f / 16.0f)) * cache_val(T, KC[T], c, kvh, d); sc.push_back(s); mx = std::max(mx, s); }
                        double den = 0; for (double & s : sc) { s = std::exp(s - mx); den += s; }
                        std::vector<double> acc(256, 0); int ci = 0; for (int c : cells) { for (int d = 0; d < 256; ++d) acc[d] += sc[ci] * cache_val(T, VC[T], c, kvh, d); ++ci; }
                        // host merge of the splits (qsa_decode_merge)
                        const float * src = o.data() + ((size_t) qq * nh + hq) * splits * 258;
                        float M = -INFINITY; for (int s = 0; s < splits; ++s) M = std::max(M, src[s * 258 + 256]);
                        double amax = 0, emax = 0;
                        for (int d = 0; d < 256; ++d) { double val = 0, sum = 0; for (int s = 0; s < splits; ++s) { const float cnt = src[s * 258 + 257]; const double f = cnt > 0 ? std::exp(src[s * 258 + 256] - M) : 0; val += src[s * 258 + d] * f; sum += cnt * f; }
                            const double gv = sum > 0 ? val / sum : 0, rv = acc[d] / den; amax = std::max(amax, std::fabs(rv)); emax = std::max(emax, std::fabs(rv - gv)); }
                        maxrel = std::max(maxrel, emax / std::max(amax, 1e-30));
                    }
                }
                printf("L%d CHK %-18s type %s cpu_f64_maxrel %.3e (type reference)\n", L, v.name.c_str(), TN[T], maxrel);
            } else {
                size_t neq = 0; for (size_t k = 0; k < pn; ++k) neq += memcmp(&o[k], &first[T][k], 4) != 0;
                printf("L%d CHK %-18s type %s words_differ_vs_type_ref %zu\n", L, v.name.c_str(), TN[T], neq);
            }
        }
        std::vector<std::vector<float>> t(V.size());
        for (int it = 0; it < 3 + reps; ++it) for (int i : vl) {
            CK(hipEventRecord(e0, 0)); launch(i); CK(hipEventRecord(e1, 0)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it >= 3) t[i].push_back(ms);
        }
        long valid = 0; for (int32_t x : uids) valid += x >= 0;
        for (int i : vl) {
            std::vector<float> x = t[i]; std::sort(x.begin(), x.end());
            double mean = 0, sd = 0; for (float y : x) mean += y; mean /= x.size(); for (float y : x) sd += (y - mean) * (y - mean); sd = std::sqrt(sd / std::max<size_t>(1, x.size() - 1));
            const double med = x.size() % 2 ? x[x.size() / 2] : 0.5 * (x[x.size() / 2 - 1] + x[x.size() / 2]);
            printf("L%d RES %-18s nk %d n_q %d splits %d chunk %d unique/q %.1f med %.4f mean %.4f sd %.4f min %.4f n %zu ns/key-head %.3f\n", L, V[i].name.c_str(), nk, NQ, splits, chunk, (double) valid / NQ, med, mean, sd, x[0], x.size(), med * 1e6 / ((double) valid * 2));
            sum_med[V[i].name] += med;
        }
        fflush(stdout); ++n_layers;
        for (int T = 0; T < 4; ++T) { hipFree(dK[T]); hipFree(dV[T]); }
        hipFree(d_ids); hipFree(d_uids); hipFree(d_q); hipFree(d_part);
    }
    printf("SUMMARY layers %d (mean of per-layer medians, ms)\n", n_layers);
    for (auto & kv : sum_med) printf("SUM %-18s %.4f\n", kv.first.c_str(), kv.second / n_layers);
    return 0;
}
