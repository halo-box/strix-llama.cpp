// R2-M-LAB-BANDS-OP: R1-M-Lab A's bands_dec.cu + UNI/TBO fixtures (bands_ref.h), BANDS_DEFECTS=1 planted defects, TBO bitwise check; n_q 0 = all.
// R1-M-Lab A: C18 decode harness (c18_dec.cu lines 1-279 verbatim: kernels, PR decode kernel include) + an extended check main.
// hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I../c18 bands_dec.cu -o bands_dec
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

#include "bands_ref.h"
static const char * TN[4] = {"f16", "q8_0", "q4_0", "bf16"};
static const char * LN[4] = {"D", "Dd", "X", "S"};

// R2 (op): planted defects on the contract (R1-Q8 definitions, bands_ref.h defect_row; d3 = V x 1.001 before the f16 rounding),
// each measured against the clean contract on the same rows: DEFECT <tag> L<l> pair <k>/<v> d<i> cells_dropped <n> <errstat>
static void run_defects(const std::string & tag, const bands::fixture & F, const std::vector<int> & sel, const bands::deq & KD, const bands::deq & VD,
                        const char * kt, const char * vt, const std::vector<double> & con, int nthr) {
    for (int def = 1; def <= 3; ++def) {
        std::vector<int32_t> rows = F.rows; size_t dropped = 0;
        if (def < 3) for (int q : sel) { int32_t * r = rows.data() + (size_t) q * F.ns; std::vector<int> c0, c1; bands::row_cells(r, F.ns, F.nk, c0); bands::defect_row(def, r, F.ns); bands::row_cells(r, F.ns, F.nk, c1); dropped += c0.size() - c1.size(); }
        std::vector<double> d; bands::oracle_rows(bands::O_CON, F, sel, rows, KD, VD, 1.0f / 16.0f, def == 3 ? 1.001f : 1.0f, d, nthr);
        bands::errstat es; for (size_t r = 0; r < sel.size() * 24; ++r) es.add_row(con.data() + r * 256, d.data() + r * 256);
        printf("DEFECT %s L%d pair %s/%s d%d cells_dropped %zu %s\n", tag.c_str(), F.L, kt, vt, def, dropped, bands::fmt_stat(es).c_str());
    }
}
// R2 (op): harness output vs the dumped test-backend-ops ROCm0 output (whole tensor), and test-backend-ops' CPU output vs the contract
static void tbo_compare(const std::string & tag, const bands::fixture & F, const char * name, const float * o, size_t n) {
    if (F.out_gpu.empty()) return;
    size_t neq = 0; double se = 0, sr = 0, mx = 0;
    for (size_t i = 0; i < n; ++i) { neq += memcmp(&o[i], &F.out_gpu[i], 4) != 0; const double e = (double) o[i] - F.out_gpu[i]; se += e * e; sr += (double) F.out_gpu[i] * F.out_gpu[i]; mx = std::max(mx, std::fabs(e)); }
    printf("TBO_BITWISE %s %s words %zu words_differ_vs_tbo_rocm0 %zu nmse %.4e maxabs %.4e\n", tag.c_str(), name, n, neq, sr > 0 ? se / sr : 0.0, mx);
}

// the PR's qsa_decode_merge, verbatim (ggml/src/ggml-cuda/qsa-decode.cu:65-79 at 94e086847): the device merge test-backend-ops runs
static __global__ __launch_bounds__(256) void pr_qsa_decode_merge(
        const float * partial, float * out, int splits) {
    const int row = blockIdx.x, d = threadIdx.x;
    const float * src = partial + size_t(row)*splits*258;
    float maximum = -INFINITY;
    for (int s = 0; s < splits; ++s) { maximum = fmaxf(maximum, src[s*258 + 256]); }
    float sum = 0.0f, value = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float count = src[s*258 + 257];
        const float factor = count > 0.0f ? expf(src[s*258 + 256] - maximum) : 0.0f;
        value += src[s*258 + d]*factor;
        sum += count*factor;
    }
    out[size_t(row)*256 + d] = sum > 0.0f ? value/sum : 0.0f;
}
typedef void (*dfn_t)(const char *, const char *, const char *, const char *, size_t, size_t, size_t, size_t, size_t, size_t, size_t, int, int, int, int, int, float, float *);
struct variant { std::string name, group; int T, KL, VL; dfn_t f; };

// usage: bands_dec <rows|SYN:U:nk:seed> <qkv|SYN> n_q <variants: ship|name,...> <tag> [nthr]
//   the first n_q queries of every layer (all of them, all 24 heads); PR launch (qsa-decode.cu:136-158) as c18_dec.cu;
//   partials merged on the host with the PR's qsa_decode_merge arithmetic (qsa-decode.cu:65-79: f32, expf, same order).
int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: rows qkv n_q variants tag [nthr]\n"); return 1; }
    const std::string rowsf = argv[1], qkvf = argv[2], vs = argv[4], tag = argv[5];
    const int NQarg = atoi(argv[3]), nthr = argc > 6 ? atoi(argv[6]) : 32; const bool defects = getenv("BANDS_DEFECTS") && atoi(getenv("BANDS_DEFECTS"));
    std::vector<variant> V;
#define DV(T, KL, VL, g) V.push_back({std::string("K") + TN[T] + LN[KL] + "-V" + TN[T] + LN[VL], g, T, KL, VL, c18_dec_kernel<T, KL, VL>})
    // per type: the PR's D+D loaders first (pair reference for the bitwise check), then the shipped loader (C18 decision 2)
    DV(TF16, LD, LD, "base"); DV(TF16, LD, LS, "ship");
    DV(TQ8,  LD, LD, "base"); DV(TQ8,  LD, LS, "ship");
    DV(TQ4,  LD, LD, "base"); DV(TQ4,  LX, LS, "ship");
    DV(TBF,  LD, LD, "base"); DV(TBF,  LX, LS, "ship");
    V.push_back({"PR-dec-f16", "pr", TF16, LD, LD, nullptr}); V.push_back({"PR-dec-q8_0", "pr", TQ8, LD, LD, nullptr});
    std::vector<int> vl;
    { std::set<std::string> sel; std::string x; for (char c : vs + ",") { if (c == ',') { sel.insert(x); x.clear(); } else x += c; }
      for (int T = 0; T < 4; ++T) for (int i = 0; i < (int) V.size(); ++i) if (V[i].T == T && (sel.count(V[i].name) || sel.count("ship"))) vl.push_back(i); }
    bands::fixture_reader R; if (!R.open(rowsf, qkvf)) return 1;
    printf("# bands_dec tag %s rows %s data %s n_q %d variants %zu defects %d\n", tag.c_str(), rowsf.c_str(), qkvf.c_str(), NQarg, vl.size(), (int) defects);
    std::map<std::string, bands::errstat> pooled; std::map<std::string, double> nmse_lmax;
    bands::fixture F;
    while (R.next(F, [](int) { return true; })) {
        const int L = F.L, nk = F.nk, ns = F.ns, NQ = NQarg > 0 ? std::min(NQarg, F.n_q) : F.n_q;
        printf("FIXTURE %s L%d %s n_q_used %d\n", tag.c_str(), L, F.label.c_str(), NQ);
        {   // R2 (op): input validation: FNV-1a of the rows; moments and a 10-bin histogram of Q and of the f16 K/V values
            printf("ROWS_HASH %s %016llx n %zu\n", tag.c_str(), (unsigned long long) bands::fnv1a(F.rows.data(), F.rows.size() * 4), F.rows.size());
            auto st = [&](const char * nm, size_t n, const std::function<float(size_t)> & x) {
                if (!n) return; double s1 = 0, s2 = 0, mn = 1e30, mx = -1e30; size_t h[10] = {0};
                for (size_t i = 0; i < n; ++i) { const double v = x(i); s1 += v; s2 += v * v; mn = std::min(mn, v); mx = std::max(mx, v); int b = (int) std::floor((v + 1.0) * 5.0); b = std::max(0, std::min(9, b)); ++h[b]; }
                printf("VALSTAT %s %s n %zu mean %.5f var %.5f min %.6f max %.6f hist", tag.c_str(), nm, n, s1 / n, s2 / n - (s1 / n) * (s1 / n), mn, mx);
                for (int b = 0; b < 10; ++b) printf(" %.4f", (double) h[b] / n); printf("\n");
            };
            st("Q", F.Q.size(), [&](size_t i) { return F.Q[i]; });
            st("K16", F.K16.size(), [&](size_t i) { return bands::h2f(F.K16[i]); });
            st("V16", F.V16.size(), [&](size_t i) { return bands::h2f(F.V16[i]); });
            if (getenv("BANDS_ROWS_ONLY")) continue;
        }
        std::vector<int> sel; for (int q = 0; q < NQ; ++q) sel.push_back(q);
        std::vector<uint8_t> KC[4], VC[4]; bands::deq KD[4], VD[4]; char * dK[4] = {}, * dV[4] = {};
        std::set<int> need; for (int i : vl) need.insert(V[i].T);
        { std::set<int> ok; for (int T : need) if ((F.rawKT < 0 || F.rawKT == T) && (F.rawVT < 0 || F.rawVT == T)) ok.insert(T); need = ok; }
        for (int T : need) { bands::build_cache_fx(T, F, true, KC[T]); bands::build_cache_fx(T, F, false, VC[T]); bands::build_deq(T, KC[T], nk, KD[T]); bands::build_deq(T, VC[T], nk, VD[T]);
            CK(hipMalloc(&dK[T], KC[T].size())); CK(hipMemcpy(dK[T], KC[T].data(), KC[T].size(), hipMemcpyHostToDevice));
            CK(hipMalloc(&dV[T], VC[T].size())); CK(hipMemcpy(dV[T], VC[T].data(), VC[T].size(), hipMemcpyHostToDevice)); }
        for (int T : need) printf("CACHE_HASH %s L%d KV %s %016llx %016llx\n", tag.c_str(), L, TN[T], (unsigned long long) bands::fnv1a(KC[T].data(), KC[T].size()), (unsigned long long) bands::fnv1a(VC[T].data(), VC[T].size()));
        int32_t * d_ids, * d_uids; float * d_q, * d_part;
        CK(hipMalloc(&d_ids, (size_t) NQ * ns * 4)); CK(hipMemcpy(d_ids, F.rows.data(), (size_t) NQ * ns * 4, hipMemcpyHostToDevice));
        CK(hipMalloc(&d_uids, (size_t) NQ * ns * 4));
        hipLaunchKernelGGL(qsa_decode_unique_ids, dim3(NQ), dim3(256), 0, 0, (const char *) d_ids, (size_t) ns * 4, d_uids, ns, nk);
        CK(hipGetLastError()); CK(hipDeviceSynchronize());
        {   // R2 (op): which occurrence of a duplicated key the PR's atomicCAS dedup kept (UIDS_HASH varies when the race does);
            // BANDS_DEDUP=first|last replaces the kernel's result by a host dedup keeping the first / last occurrence
            std::vector<int32_t> u((size_t) NQ * ns); CK(hipMemcpy(u.data(), d_uids, u.size() * 4, hipMemcpyDeviceToHost));
            size_t ndup = 0, nlast = 0;
            for (int q = 0; q < NQ; ++q) { const int32_t * r = F.rows.data() + (size_t) q * ns; for (int j = 0; j < ns; ++j) { if (r[j] < 0 || r[j] >= nk) continue; for (int j2 = j + 1; j2 < ns; ++j2) if (r[j2] == r[j]) { ++ndup; nlast += u[(size_t) q * ns + j2] == r[j]; } } }
            printf("UIDS %s L%d hash %016llx duplicate_pairs %zu kept_later_occurrence %zu\n", tag.c_str(), L, (unsigned long long) bands::fnv1a(u.data(), u.size() * 4), ndup, nlast);
            if (const char * dm = getenv("BANDS_DEDUP")) {
                const bool last = strcmp(dm, "last") == 0;
                for (int q = 0; q < NQ; ++q) { const int32_t * r = F.rows.data() + (size_t) q * ns; int32_t * o = u.data() + (size_t) q * ns; std::set<int> seen;
                    if (!last) { for (int j = 0; j < ns; ++j) { o[j] = -1; if (r[j] >= 0 && r[j] < nk && seen.insert(r[j]).second) o[j] = r[j]; } }
                    else { for (int j = ns - 1; j >= 0; --j) { o[j] = -1; if (r[j] >= 0 && r[j] < nk && seen.insert(r[j]).second) o[j] = r[j]; } } }
                CK(hipMemcpy(d_uids, u.data(), u.size() * 4, hipMemcpyHostToDevice));
                printf("UIDS %s L%d host_dedup %s hash %016llx\n", tag.c_str(), L, dm, (unsigned long long) bands::fnv1a(u.data(), u.size() * 4));
            }
        }
        const int step = 64, nh = 24;
        int chunk = step;
        if (NQ > 8) { const int target = std::max(1, (8 * ((ns + step - 1) / step) + NQ - 1) / NQ); chunk = ((ns + target - 1) / target + 15) / 16 * 16; }
        const int splits = (ns + chunk - 1) / chunk;
        CK(hipMalloc(&d_q, (size_t) NQ * 24 * 256 * 4)); CK(hipMemcpy(d_q, F.Q.data(), (size_t) NQ * 24 * 256 * 4, hipMemcpyHostToDevice));
        const size_t pn = (size_t) NQ * nh * splits * 258;
        CK(hipMalloc(&d_part, pn * 4));
        float * d_merged; CK(hipMalloc(&d_merged, (size_t) NQ * nh * 256 * 4));
        const size_t q1 = 24 * 1024, q2 = 1024;
        auto launch = [&](int i) {
            const variant & v = V[i]; const int T = v.T; const size_t k1 = 2 * (size_t) bands::hbytes(T), k2 = bands::hbytes(T);
            const dim3 gr(2, NQ, splits), bl(256);
            if (!v.f) {
                if (T == TQ8) hipLaunchKernelGGL(qsa_decode_wmma_partial<true>, gr, bl, 0, 0, (const char *) d_q, dK[T], dV[T], (const char *) nullptr, (const char *) d_uids, q1, q2, k1, k2, k1, k2, (size_t) 0, (size_t) ns * 4, nk, ns, nh, splits, chunk, 1.0f / 16.0f, d_part);
                else          hipLaunchKernelGGL(qsa_decode_wmma_partial<false>, gr, bl, 0, 0, (const char *) d_q, dK[T], dV[T], (const char *) nullptr, (const char *) d_uids, q1, q2, k1, k2, k1, k2, (size_t) 0, (size_t) ns * 4, nk, ns, nh, splits, chunk, 1.0f / 16.0f, d_part);
            } else hipLaunchKernelGGL(v.f, gr, bl, 0, 0, (const char *) d_q, dK[T], dV[T], (const char *) d_uids, q1, q2, k1, k2, k1, k2, (size_t) ns * 4, nk, ns, nh, splits, chunk, 1.0f / 16.0f, d_part);
            CK(hipGetLastError());
        };
        std::vector<float> o(pn), first, merged((size_t) NQ * nh * 256);
        int curT = -1; std::vector<double> ref[3];
        printf("CASE %s n_q %d ns %d nk %d blocks %d splits %d chunk %d\n", tag.c_str(), NQ, ns, nk, F.blocks, splits, chunk);
        for (int i : vl) {
            const variant & v = V[i]; const int T = v.T;
            if (!need.count(T)) continue;
            if (T != curT) {
                curT = T; first.clear();
                for (int m = 0; m < 3; ++m) bands::oracle_rows(m, F, sel, F.rows, KD[T], VD[T], 1.0f / 16.0f, 1.0f, ref[m], nthr);
                if (defects) run_defects(tag, F, sel, KD[T], VD[T], TN[T], TN[T], ref[bands::O_CON], nthr);
                if (!F.out_cpu.empty()) { bands::errstat es; for (int r = 0; r < NQ * 24; ++r) es.add_row(ref[bands::O_CON].data() + (size_t) r * 256, F.out_cpu.data() + (size_t) r * 256);
                    printf("TBO_CPU %s L%d pair %s/%s tbo_cpu_vs_contract %s\n", tag.c_str(), L, TN[T], TN[T], bands::fmt_stat(es).c_str()); }
            }
            CK(hipMemset(d_part, 0xff, pn * 4)); launch(i); CK(hipDeviceSynchronize()); CK(hipMemcpy(o.data(), d_part, pn * 4, hipMemcpyDeviceToHost));
            if (first.empty()) { first = o; printf("BITWISE %s L%d %-14s type %s pair_reference splits %d chunk %d\n", tag.c_str(), L, v.name.c_str(), TN[T], splits, chunk); }
            else { size_t neq = 0; for (size_t k = 0; k < pn; ++k) neq += memcmp(&o[k], &first[k], 4) != 0;
                   printf("BITWISE %s L%d %-14s type %s words_differ_vs_type_ref %zu\n", tag.c_str(), L, v.name.c_str(), TN[T], neq); }
            for (int row = 0; row < NQ * nh; ++row) {   // qsa_decode_merge (qsa-decode.cu:65-79)
                const float * src = o.data() + (size_t) row * splits * 258;
                float maximum = -INFINITY; for (int s = 0; s < splits; ++s) maximum = fmaxf(maximum, src[s*258 + 256]);
                for (int d = 0; d < 256; ++d) {
                    float sum = 0.0f, value = 0.0f;
                    for (int s = 0; s < splits; ++s) { const float count = src[s*258 + 257]; const float factor = count > 0.0f ? expf(src[s*258 + 256] - maximum) : 0.0f; value += src[s*258 + d]*factor; sum += count*factor; }
                    merged[(size_t) row * 256 + d] = sum > 0.0f ? value/sum : 0.0f;
                }
            }
            if (const char * pf = getenv("BANDS_PARTIAL_CMP")) {   // R2 (op): partials vs test-backend-ops' partial dump
                FILE * f = fopen(pf, "rb"); std::vector<float> tp(pn); const size_t got = f ? fread(tp.data(), 4, pn, f) : 0; if (f) fclose(f);
                size_t neq = 0, neq_stat = 0; for (size_t k = 0; k < pn && got == pn; ++k) { const bool d = memcmp(&o[k], &tp[k], 4) != 0; neq += d; neq_stat += d && (k % 258) >= 256; }
                printf("PARTIAL_CMP %s %s read %zu of %zu words_differ %zu of_which_max_or_count %zu\n", tag.c_str(), v.name.c_str(), got, pn, neq, neq_stat);
                if (getenv("BANDS_MERGE_TBO_PARTIAL") && got == pn) { CK(hipMemcpy(d_part, tp.data(), pn * 4, hipMemcpyHostToDevice)); printf("MERGE_INPUT %s tbo_partials\n", tag.c_str()); }
            }
            {   // R2 (op): device merge (the PR kernel); the BAND lines below use it. Host merge kept for the R1-A comparison.
                std::vector<float> hm = merged;
                hipLaunchKernelGGL(pr_qsa_decode_merge, dim3(NQ * nh), dim3(256), 0, 0, (const float *) d_part, d_merged, splits); CK(hipGetLastError()); CK(hipDeviceSynchronize());
                CK(hipMemcpy(merged.data(), d_merged, merged.size() * 4, hipMemcpyDeviceToHost));
                size_t neq = 0; double se = 0, sr = 0; for (size_t k = 0; k < merged.size(); ++k) { neq += memcmp(&hm[k], &merged[k], 4) != 0; const double e = (double) hm[k] - merged[k]; se += e * e; sr += (double) merged[k] * merged[k]; }
                printf("HOSTMERGE %s L%d %-14s words_differ_host_vs_device %zu nmse %.4e\n", tag.c_str(), L, v.name.c_str(), neq, sr > 0 ? se / sr : 0.0);
            }
            tbo_compare(tag, F, v.name.c_str(), merged.data(), merged.size());
            for (int m = 0; m < 3; ++m) {
                bands::errstat es;
                for (int r = 0; r < NQ * nh; ++r) es.add_row(ref[m].data() + (size_t) r * 256, merged.data() + (size_t) r * 256);
                printf("BAND %s L%d %-14s pair %s/%s gpu_vs_%s %s\n", tag.c_str(), L, v.name.c_str(), TN[T], TN[T], bands::oname(m), bands::fmt_stat(es).c_str());
                const std::string pk = v.name + " " + TN[T] + "/" + TN[T] + " gpu_vs_" + bands::oname(m);
                pooled[pk].merge(es); nmse_lmax[pk] = std::max(nmse_lmax[pk], es.nmse());
            }
            fflush(stdout);
        }
        for (int T = 0; T < 4; ++T) { if (dK[T]) hipFree(dK[T]); if (dV[T]) hipFree(dV[T]); }
        hipFree(d_ids); hipFree(d_uids); hipFree(d_q); hipFree(d_part); hipFree(d_merged);
    }
    for (auto & kv : pooled) printf("BANDSUM %s %s nmse_layer_max %.4e %s\n", tag.c_str(), kv.first.c_str(), nmse_lmax[kv.first], bands::fmt_stat(kv.second).c_str());
    return 0;
}
