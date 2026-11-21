// x86 CPU kernels of the fork types: the fork's additions to ggml/src/ggml-cpu/arch/x86/quants.c.
#if (defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)) && !defined(GGML_CPU_GENERIC)
#define GGML_COMMON_IMPL_C
#include "../ggml-quants-ext.h"
#include "quants-ext.h"
#include "simd-mappings.h"

#include <immintrin.h>
#include <math.h>
#include <string.h>

// copied from ggml/src/ggml-cpu/arch/x86/quants.c (upstream 9048bad8d): local there
#define UNUSED GGML_UNUSED
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
#endif
// from ggml/src/ggml-cpu/arch/x86/quants.c after upstream line 4190 (upstream 9048bad8d, fork 1fe05fa45)

#ifdef GGML_USE_TRELLIS
// TQ2_T / TQK6 / TQK7 x Q8_0.
//
// Every trellis step t (0..31) of a 128-weight block yields weights 4t..4t+3, which
// ggml_tq_state_i16 (see ggml_cpu_tq_init) holds per 15-bit state as four int16, so a
// block is 32 table lookups (8 bytes each) plus an int16 x int8 dot. The 256 KiB table
// lives in L2; the lookups are 64-bit gathers, which bound the kernel (gathers from the
// 8 KiB pair codebook need twice as many elements and measured ~35% slower; an int8
// state table is no faster and loses ~1% per weight).
//
// AVX-512: the 32 states of a block come from one masked load of the stream bytes and
// two VPERMB that place, per dword lane t, the bytes holding state t's 16-bit window
// (constant per type: TQK's window for step t starts at bit (31-t)*K, circularly;
// TQ2_T's state t is qs[t] | qs[t-1] << 8), then a per-lane variable shift and mask.
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VBMI__)
#define GGML_TQ_AVX512 1
#endif

#if defined(GGML_TQ_AVX512)
static const uint8_t tq_perm_tq2t[128] = {
    0,31,0,0, 1,0,1,1, 2,1,2,2, 3,2,3,3, 4,3,4,4, 5,4,5,5, 6,5,6,6, 7,6,7,7,
    8,7,8,8, 9,8,9,9, 10,9,10,10, 11,10,11,11, 12,11,12,12, 13,12,13,13, 14,13,14,14, 15,14,15,15,
    16,15,16,16, 17,16,17,17, 18,17,18,18, 19,18,19,19, 20,19,20,20, 21,20,21,21, 22,21,22,22, 23,22,23,23,
    24,23,24,24, 25,24,25,25, 26,25,26,26, 27,26,27,27, 28,27,28,28, 29,28,29,29, 30,29,30,30, 31,30,31,31,
};
static const uint32_t tq_shift_tq2t[32] = {0};
static const uint8_t tq_perm_tqk6[128] = {
    23,0,1,2, 22,23,0,1, 21,22,23,0, 21,22,23,0, 20,21,22,23, 19,20,21,22, 18,19,20,21, 18,19,20,21,
    17,18,19,20, 16,17,18,19, 15,16,17,18, 15,16,17,18, 14,15,16,17, 13,14,15,16, 12,13,14,15, 12,13,14,15,
    11,12,13,14, 10,11,12,13, 9,10,11,12, 9,10,11,12, 8,9,10,11, 7,8,9,10, 6,7,8,9, 6,7,8,9,
    5,6,7,8, 4,5,6,7, 3,4,5,6, 3,4,5,6, 2,3,4,5, 1,2,3,4, 0,1,2,3, 0,1,2,3,
};
static const uint32_t tq_shift_tqk6[32] = {
    2,4,6,0, 2,4,6,0, 2,4,6,0, 2,4,6,0, 2,4,6,0, 2,4,6,0, 2,4,6,0, 2,4,6,0,
};
static const uint8_t tq_perm_tqk7[128] = {
    27,0,1,2, 26,27,0,1, 25,26,27,0, 24,25,26,27, 23,24,25,26, 22,23,24,25, 21,22,23,24, 21,22,23,24,
    20,21,22,23, 19,20,21,22, 18,19,20,21, 17,18,19,20, 16,17,18,19, 15,16,17,18, 14,15,16,17, 14,15,16,17,
    13,14,15,16, 12,13,14,15, 11,12,13,14, 10,11,12,13, 9,10,11,12, 8,9,10,11, 7,8,9,10, 7,8,9,10,
    6,7,8,9, 5,6,7,8, 4,5,6,7, 3,4,5,6, 2,3,4,5, 1,2,3,4, 0,1,2,3, 0,1,2,3,
};
static const uint32_t tq_shift_tqk7[32] = {
    1,2,3,4,5,6,7,0, 1,2,3,4,5,6,7,0, 1,2,3,4,5,6,7,0, 1,2,3,4,5,6,7,0,
};

static inline __m512i tq_dot_i16_x32(__m512i acc, __m512i w, __m512i a) {
#if defined(__AVX512VNNI__)
    return _mm512_dpwssd_epi32(acc, w, a);
#else
    return _mm512_add_epi32(acc, _mm512_madd_epi16(w, a));
#endif
}

static void ggml_vec_dot_tq_q8_0_avx512(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
                                        int kb, const uint8_t * permt, const uint32_t * shiftt) {
    const int nb = n / QK_TQK;
    const int nbytes = 4*kb;
    const size_t bsize = sizeof(ggml_half) + nbytes;
    const __mmask64 lmask = (__mmask64) ((1ull << nbytes) - 1);

    const __m512i perm0  = _mm512_loadu_si512(permt);
    const __m512i perm1  = _mm512_loadu_si512(permt + 64);
    const __m512i shift0 = _mm512_loadu_si512(shiftt);
    const __m512i shift1 = _mm512_loadu_si512(shiftt + 16);
    const __m512i smask  = _mm512_set1_epi32(TQ_STATE_MASK);
    const long long * tab = (const long long *) ggml_tq_state_i16;

    const uint8_t    * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    __m512 acc = _mm512_setzero_ps();

    for (int i = 0; i < nb; ++i, x += bsize, y += QK_TQK/QK8_0) {
        const __m512i raw = _mm512_maskz_loadu_epi8(lmask, x + sizeof(ggml_half));
        const __m512i s0 = _mm512_and_si512(_mm512_srlv_epi32(_mm512_permutexvar_epi8(perm0, raw), shift0), smask);
        const __m512i s1 = _mm512_and_si512(_mm512_srlv_epi32(_mm512_permutexvar_epi8(perm1, raw), shift1), smask);

        // 8 states -> 32 int16 weights = one Q8_0 block
        const __m512i w0 = _mm512_i32gather_epi64(_mm512_castsi512_si256(s0),       tab, 8);
        const __m512i w1 = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(s0, 1), tab, 8);
        const __m512i w2 = _mm512_i32gather_epi64(_mm512_castsi512_si256(s1),       tab, 8);
        const __m512i w3 = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(s1, 1), tab, 8);

        const __m512i a0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *) y[0].qs));
        const __m512i a1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *) y[1].qs));
        const __m512i a2 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *) y[2].qs));
        const __m512i a3 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *) y[3].qs));

        const __m512i p0 = tq_dot_i16_x32(_mm512_setzero_si512(), w0, a0);
        const __m512i p1 = tq_dot_i16_x32(_mm512_setzero_si512(), w1, a1);
        const __m512i p2 = tq_dot_i16_x32(_mm512_setzero_si512(), w2, a2);
        const __m512i p3 = tq_dot_i16_x32(_mm512_setzero_si512(), w3, a3);

        const float dx = GGML_CPU_FP16_TO_FP32(*(const ggml_half *) x);
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p0), _mm512_set1_ps(dx * GGML_CPU_FP16_TO_FP32(y[0].d)), acc);
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p1), _mm512_set1_ps(dx * GGML_CPU_FP16_TO_FP32(y[1].d)), acc);
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p2), _mm512_set1_ps(dx * GGML_CPU_FP16_TO_FP32(y[2].d)), acc);
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p3), _mm512_set1_ps(dx * GGML_CPU_FP16_TO_FP32(y[3].d)), acc);
    }

    *s = _mm512_reduce_add_ps(acc) * (1.0f / GGML_TQ_I16_SCALE);
}
#elif defined(__AVX2__)
// AVX2: states in scalar code (the stream is copied with its first 4 bytes repeated
// after it, so every 16-bit window is one unaligned 32-bit load), table rows loaded
// as scalars into 4 x 64-bit vectors, then VPMADDWD.
static void ggml_vec_dot_tq_q8_0_avx2(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int kb) {
    const int nb = n / QK_TQK;
    const int nbytes = 4*kb;
    const size_t bsize = sizeof(ggml_half) + nbytes;
    const long long * tab = (const long long *) ggml_tq_state_i16;

    const uint8_t    * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    uint8_t buf[32 + 4];
    int32_t st[32];

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i, x += bsize, y += QK_TQK/QK8_0) {
        const uint8_t * qs = x + sizeof(ggml_half);
        if (kb == 8) {
            for (int t = 0; t < 32; ++t) {
                st[t] = (int32_t) ((((uint32_t) qs[(t + 31) & 31] << 8) | qs[t]) & TQ_STATE_MASK);
            }
        } else {
            memcpy(buf, qs, nbytes);
            memcpy(buf + nbytes, qs, 4);
            for (int t = 0; t < 32; ++t) {
                const uint32_t off = (uint32_t) (31 - t) * (uint32_t) kb;
                uint32_t w;
                memcpy(&w, buf + (off >> 3), 4);
                st[t] = (int32_t) ((w >> (off & 7)) & TQ_STATE_MASK);
            }
        }

        const float dx = GGML_CPU_FP16_TO_FP32(*(const ggml_half *) x);
        for (int k = 0; k < QK_TQK/QK8_0; ++k) {
            // scalar 64-bit loads beat VPGATHERQQ here (Zen 5: 70-76 vs 83-87 ns/block)
            const int32_t * sk = st + 8*k;
            const __m256i w0 = _mm256_set_epi64x(tab[sk[3]], tab[sk[2]], tab[sk[1]], tab[sk[0]]);
            const __m256i w1 = _mm256_set_epi64x(tab[sk[7]], tab[sk[6]], tab[sk[5]], tab[sk[4]]);
            const __m256i a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *) y[k].qs));
            const __m256i a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *) (y[k].qs + 16)));
            const __m256i p  = _mm256_add_epi32(_mm256_madd_epi16(w0, a0), _mm256_madd_epi16(w1, a1));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(dx * GGML_CPU_FP16_TO_FP32(y[k].d)), acc);
        }
    }

    *s = hsum_float_8(acc) * (1.0f / GGML_TQ_I16_SCALE);
}
#endif

void ggml_vec_dot_tq2_t_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_TQ2_T == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
#if defined(GGML_TQ_AVX512)
    ggml_vec_dot_tq_q8_0_avx512(n, s, vx, vy, 8, tq_perm_tq2t, tq_shift_tq2t);
#elif defined(__AVX2__)
    ggml_vec_dot_tq_q8_0_avx2(n, s, vx, vy, 8);
#else
    ggml_vec_dot_tq2_t_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_tqk6_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_TQK == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
#if defined(GGML_TQ_AVX512)
    ggml_vec_dot_tq_q8_0_avx512(n, s, vx, vy, 6, tq_perm_tqk6, tq_shift_tqk6);
#elif defined(__AVX2__)
    ggml_vec_dot_tq_q8_0_avx2(n, s, vx, vy, 6);
#else
    ggml_vec_dot_tqk6_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_tqk7_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_TQK == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
#if defined(GGML_TQ_AVX512)
    ggml_vec_dot_tq_q8_0_avx512(n, s, vx, vy, 7, tq_perm_tqk7, tq_shift_tqk7);
#elif defined(__AVX2__)
    ggml_vec_dot_tq_q8_0_avx2(n, s, vx, vy, 7);
#else
    ggml_vec_dot_tqk7_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

// TQP5 / TQP6 / TQP7 x Q8_0.
//
// A 128-weight block is 64 steps of one weight pair each; Q8_0 block k covers steps 16k..16k+15.
// The 8 states of lane l (steps 8l..8l+7, 16 weights) all lie in one 64-bit window of the
// circular stream starting at byte (7-l)*K (only lane 0 wraps), at constant bit offsets
// (7-j)*K, so a state is a shift and a mask of a GPR. Each state is hashed to its codebook
// pair index (tq2t_hyb_index(2*s) = (s * 0x9e3779b1) >> 21) and the pair (two int16) is read
// from the 8 KiB ggml_tq_pair_i16, which stays in L1, by a broadcast load blended into a YMM
// (no shuffle port), then VPMADDWD against the sign-extended Q8_0 activations.
// Zen 5, one thread: ~15 vs ~93 ns per block for the generic kernel. With 16 threads streaming
// weights the L1 codebook beats lookups in the 256 KiB state table (2.0 vs 2.5-2.7 ns per
// block) and VPGATHERDD on either table (2.9-3.1); AVX-512 masked broadcasts are faster on one
// thread only, so AVX-512 builds run this kernel too.
#if defined(__AVX2__)
#if defined(_MSC_VER)
#define TQP_INLINE __forceinline
#else
#define TQP_INLINE inline __attribute__((always_inline))
#endif

static TQP_INLINE uint64_t tqp_ld64(const uint8_t * p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

// stream bits from byte (7-l)*kb, circular modulo 8*kb bytes (kb <= 7, so only lane 0 wraps)
static TQP_INLINE uint64_t tqp_window(const uint8_t * qs, int kb, int l) {
    if (l == 0) {
        return (tqp_ld64(qs + 8*kb - 8) >> (8*(8 - kb))) | (tqp_ld64(qs) << (8*kb));
    }
    return tqp_ld64(qs + (7 - l)*kb);
}

// the two weights of step 8l+j as an int32 (int16 pair)
// the deepest state, j = 0 at kb = 7, ends at bit 7*7 + TQ_STATE_BITS of the 64-bit window
static_assert(TQ_STATE_BITS + 6*7 + 7 <= 64, "TQP state window needs TQ_STATE_BITS <= 15");
static TQP_INLINE int32_t tqp_pair(const int32_t * cb, uint64_t w, int kb, int j) {
    const uint32_t s = (uint32_t) (w >> ((7 - j)*kb)) & TQ_STATE_MASK;
    return cb[(s * 0x9e3779b1u) >> 21];
}

// the 16 weights of lane l as 16 int16
static TQP_INLINE __m256i tqp_lane(const int32_t * cb, const uint8_t * qs, int kb, int l) {
    const uint64_t w = tqp_window(qs, kb, l);
    __m256i v = _mm256_castsi128_si256(_mm_cvtsi32_si128(tqp_pair(cb, w, kb, 0)));
    v = _mm256_blend_epi32(v, _mm256_set1_epi32(tqp_pair(cb, w, kb, 1)), 0x02);
    v = _mm256_blend_epi32(v, _mm256_set1_epi32(tqp_pair(cb, w, kb, 2)), 0x04);
    v = _mm256_blend_epi32(v, _mm256_set1_epi32(tqp_pair(cb, w, kb, 3)), 0x08);
    v = _mm256_blend_epi32(v, _mm256_set1_epi32(tqp_pair(cb, w, kb, 4)), 0x10);
    v = _mm256_blend_epi32(v, _mm256_set1_epi32(tqp_pair(cb, w, kb, 5)), 0x20);
    v = _mm256_blend_epi32(v, _mm256_set1_epi32(tqp_pair(cb, w, kb, 6)), 0x40);
    v = _mm256_blend_epi32(v, _mm256_set1_epi32(tqp_pair(cb, w, kb, 7)), 0x80);
    return v;
}

static TQP_INLINE void ggml_vec_dot_tqp_q8_0_avx2(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int kb) {
    const int nb = n / QK_TQP;
    const size_t bsize = sizeof(ggml_half) + 8*kb;
    const int32_t * cb = (const int32_t *) ggml_tq_pair_i16;

    const uint8_t    * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i, x += bsize, y += QK_TQP/QK8_0) {
        const uint8_t * qs = x + sizeof(ggml_half);
        const float dx = GGML_CPU_FP16_TO_FP32(*(const ggml_half *) x);
        for (int k = 0; k < QK_TQP/QK8_0; ++k) {
            const __m256i w0 = tqp_lane(cb, qs, kb, 2*k);
            const __m256i w1 = tqp_lane(cb, qs, kb, 2*k + 1);
            const __m256i a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *) y[k].qs));
            const __m256i a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *) (y[k].qs + 16)));
            const __m256i p  = _mm256_add_epi32(_mm256_madd_epi16(w0, a0), _mm256_madd_epi16(w1, a1));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(dx * GGML_CPU_FP16_TO_FP32(y[k].d)), acc);
        }
    }

    *s = hsum_float_8(acc) * (1.0f / GGML_TQ_I16_SCALE);
}
#endif

#define GGML_VEC_DOT_TQP(K)                                                                                    \
void ggml_vec_dot_tqp##K##_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx,      \
                                size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {                \
    assert(n % QK_TQP == 0);                                                                                   \
    assert(nrc == 1);                                                                                          \
    UNUSED(nrc);                                                                                               \
    UNUSED(bx);                                                                                                \
    UNUSED(by);                                                                                                \
    UNUSED(bs);                                                                                                \
    GGML_VEC_DOT_TQP_BODY(K)                                                                                   \
}

#if defined(__AVX2__)
#define GGML_VEC_DOT_TQP_BODY(K) ggml_vec_dot_tqp_q8_0_avx2(n, s, vx, vy, K);
#else
#define GGML_VEC_DOT_TQP_BODY(K) ggml_vec_dot_tqp##K##_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif

GGML_VEC_DOT_TQP(5)
GGML_VEC_DOT_TQP(6)
GGML_VEC_DOT_TQP(7)
#endif // GGML_USE_TRELLIS
#endif
