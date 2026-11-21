// Generic CPU kernels of the fork types: the fork's additions to ggml/src/ggml-cpu/quants.c. Built in each CPU variant.
#define GGML_COMMON_IMPL_C
#include "../ggml-quants-ext.h"
#include "quants-ext.h"
#include "simd-mappings.h"

#include <math.h>
#include <string.h>

#define UNUSED GGML_UNUSED // as in ggml/src/ggml-cpu/quants.c
// from ggml/src/ggml-cpu/quants.c after upstream line 33 (upstream 9048bad8d, fork 1fe05fa45)
#ifdef GGML_USE_TRELLIS
void quantize_row_ptq1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_ptq1_0_ref(x, y, k);
}

void quantize_row_tq2_t(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_tq2_t_ref(x, y, k);
}

void quantize_row_tqk6(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_tqk6_ref(x, y, k);
}

void quantize_row_tqk7(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_tqk7_ref(x, y, k);
}

void quantize_row_tqp5(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_tqp5_ref(x, y, k);
}

void quantize_row_tqp6(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_tqp6_ref(x, y, k);
}

void quantize_row_tqp7(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_tqp7_ref(x, y, k);
}
#endif // GGML_USE_TRELLIS


// from ggml/src/ggml-cpu/quants.c after upstream line 1387 (upstream 9048bad8d, fork 1fe05fa45)

#ifdef GGML_USE_TRELLIS
// PTQ1_0 x Q8_0. The trits are stored base-3 interleaved rather than in element
// order, so decode a block into element order first using the same traversal as
// dequantize_row_ptq1_0 -- that keeps the two provably in step. Four Q8_0 blocks
// cover one 128-wide PTQ1_0 block.
void ggml_vec_dot_ptq1_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK_PTQ1_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_ptq1_0 * GGML_RESTRICT x = vx;
    const block_q8_0   * GGML_RESTRICT y = vy;

    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    static const size_t  stages[3] = {32, 16, 8};

    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        int8_t q[QK_PTQ1_0];
        int o = 0;

        size_t j = 0;
        for (size_t st = 0; st < 3; ++st) {
            const size_t c = stages[st];
            for (; j + c <= sizeof(x->qs); j += c) {
                for (size_t nn = 0; nn < 5; ++nn) {
                    for (size_t m = 0; m < c; ++m) {
                        const uint8_t v  = x[i].qs[j + m] * pow3[nn];
                        const int16_t xi = ((uint16_t) v * 3) >> 8;
                        q[o++] = (int8_t) (xi - 1);
                    }
                }
            }
        }
        for (size_t nn = 0; nn < 4; ++nn) {
            for (size_t h = 0; h < sizeof(x->qh); ++h) {
                const uint8_t v  = x[i].qh[h] * pow3[nn];
                const int16_t xi = ((uint16_t) v * 3) >> 8;
                q[o++] = (int8_t) (xi - 1);
            }
        }
        assert(o == QK_PTQ1_0);

        const float d0 = GGML_CPU_FP16_TO_FP32(x[i].d);
        float sumi = 0.0f;

        for (int k = 0; k < 4; k++) {
            const block_q8_0 * GGML_RESTRICT yb = &y[i * 4 + k];
            const float d1 = GGML_CPU_FP16_TO_FP32(yb->d);
            int sumi_block = 0;
            for (int b = 0; b < 32; ++b) {
                sumi_block += (int) q[k*32 + b] * (int) yb->qs[b];
            }
            sumi += d1 * sumi_block;
        }

        sumf += d0 * sumi;
    }

    *s = sumf;
}

// TQ2_T / TQK6 / TQK7 state table: entry s holds the four codebook weights of trellis
// state s (pair tq2t_hyb_index(2s) then pair tq2t_hyb_index(2s+1)) as int16 at scale
// GGML_TQ_I16_SCALE (2^13). |codebook| < 4, so every value fits; fp16 codebook values
// >= 2^-3 in magnitude are exact, smaller ones are off by at most 2^-14. Indexing by
// state instead of by codebook pair halves the lookups per weight and removes the
// per-state hash multiply. Filled once by ggml_cpu_tq_init (from ggml_cpu_init).
// read as 64-bit (state) and 32-bit (pair) entries by the x86 kernels, so align past both
#if defined(_MSC_VER)
#define GGML_TQ_TABLE_ALIGN __declspec(align(64))
#else
#define GGML_TQ_TABLE_ALIGN __attribute__((aligned(64)))
#endif
GGML_TQ_TABLE_ALIGN int16_t ggml_tq_state_i16[(TQ_STATE_MASK + 1)*4];

// The 2048 codebook pairs, rounded the same way: ggml_tq_state_i16[4*s + j] ==
// ggml_tq_pair_i16[2*tq2t_hyb_index(2*s) + j]. 8 KiB, so it stays in L1; the TQP SIMD
// kernels hash the state (one multiply) instead of reading the 256 KiB state table,
// which is faster once many threads stream weights through L2 (TQP uses one pair per
// state, so the state table saves no lookups there).
GGML_TQ_TABLE_ALIGN int16_t ggml_tq_pair_i16[2048*2];

void ggml_cpu_tq_init(void) {
    for (uint32_t i = 0; i < 2048*2; ++i) {
        const float v = GGML_CPU_FP16_TO_FP32(tq2t_lut_f16[i]) * GGML_TQ_I16_SCALE;
        ggml_tq_pair_i16[i] = (int16_t) (v < 0.0f ? v - 0.5f : v + 0.5f);
    }
    for (uint32_t st = 0; st <= TQ_STATE_MASK; ++st) {
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t idx = tq2t_hyb_index(2*st + h);
            for (uint32_t j = 0; j < 2; ++j) {
                const float v = GGML_CPU_FP16_TO_FP32(tq2t_lut_f16[2*idx + j]) * GGML_TQ_I16_SCALE;
                ggml_tq_state_i16[4*st + 2*h + j] = (int16_t) (v < 0.0f ? v - 0.5f : v + 0.5f);
            }
        }
    }
}

// Shared TQ x Q8_0 kernel. Every trellis step t (0..31) of a 128-weight block yields
// weights 4t..4t+3 = ggml_tq_state_i16[4*state(t) ...], dotted in int32 against the
// Q8_0 activations (Q8_0 block k covers steps 8k..8k+7). kb = 8 is TQ2_T's byte stream.
// States: the stream is copied with its first 4 bytes repeated after it, so every
// (circular) 16-bit window is one 32-bit little-endian read; same values as tqk_state.
static inline void tq_states(const uint8_t * GGML_RESTRICT qs, int kb, uint32_t * GGML_RESTRICT st) {
    if (kb == 8) {
        for (int t = 0; t < 32; ++t) {
            st[t] = (((uint32_t) qs[(t + 31) & 31] << 8) | qs[t]) & TQ_STATE_MASK;
        }
        return;
    }
    const int nbytes = 4*kb;
    uint8_t buf[4*8 + 4];
    memcpy(buf, qs, nbytes);
    memcpy(buf + nbytes, qs, 4);
    for (int t = 0; t < 32; ++t) {
        const uint32_t off = (uint32_t) (31 - t) * (uint32_t) kb;
        const uint8_t * p = buf + (off >> 3);
        const uint32_t w = (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16);
        st[t] = (w >> (off & 7)) & TQ_STATE_MASK;
    }
}

static void ggml_vec_dot_tq_q8_0_impl(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int kb) {
    const int qk = QK_TQK;
    const int nb = n / qk;
    const size_t bsize = sizeof(ggml_half) + 4*kb;

    assert(n % qk == 0);
    static_assert(QK_TQK == QK_TQ2_T, "TQ2_T and TQK share the kernel");

    const uint8_t    * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    float sumf = 0.0f;

    for (int i = 0; i < nb; i++, x += bsize) {
        uint32_t st[32];
        tq_states(x + sizeof(ggml_half), kb, st);
        float sumb = 0.0f;
        for (int k = 0; k < QK_TQK/QK8_0; k++) {
            const block_q8_0 * GGML_RESTRICT yb = &y[i*(QK_TQK/QK8_0) + k];
            int32_t acc = 0;
            for (int t = 8*k; t < 8*k + 8; ++t) {
                const int16_t * w = ggml_tq_state_i16 + 4*st[t];
                const int8_t  * q = yb->qs + 4*(t - 8*k);
                acc += w[0]*q[0] + w[1]*q[1] + w[2]*q[2] + w[3]*q[3];
            }
            sumb += GGML_CPU_FP16_TO_FP32(yb->d) * (float) acc;
        }
        sumf += GGML_CPU_FP16_TO_FP32(*(const ggml_half *) x) * sumb;
    }

    *s = sumf * (1.0f / GGML_TQ_I16_SCALE);
}

void ggml_vec_dot_tq2_t_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    ggml_vec_dot_tq_q8_0_impl(n, s, vx, vy, 8);
}

void ggml_vec_dot_tqk6_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    ggml_vec_dot_tq_q8_0_impl(n, s, vx, vy, 6);
}

void ggml_vec_dot_tqk7_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    ggml_vec_dot_tq_q8_0_impl(n, s, vx, vy, 7);
}

// TQP x Q8_0. Step t (0..63) yields weights 2t, 2t+1 = the first pair of
// ggml_tq_state_i16[4*state(t)]; Q8_0 block k covers steps 16k..16k+15.
static void ggml_vec_dot_tqp_q8_0_impl(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int kb) {
    const int qk = QK_TQP;
    const int nb = n / qk;
    const int nbytes = 8*kb;
    const size_t bsize = sizeof(ggml_half) + nbytes;

    assert(n % qk == 0);

    const uint8_t    * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    float sumf = 0.0f;

    for (int i = 0; i < nb; i++, x += bsize) {
        // stream copy with its first 4 bytes repeated, so each window is one plain read
        uint8_t buf[8*7 + 4];
        memcpy(buf, x + sizeof(ggml_half), nbytes);
        memcpy(buf + nbytes, x + sizeof(ggml_half), 4);
        float sumb = 0.0f;
        for (int k = 0; k < QK_TQP/QK8_0; k++) {
            const block_q8_0 * GGML_RESTRICT yb = &y[i*(QK_TQP/QK8_0) + k];
            int32_t acc = 0;
            for (int t = 16*k; t < 16*k + 16; ++t) {
                const uint32_t off = (uint32_t) (63 - t) * (uint32_t) kb;
                const uint8_t * p = buf + (off >> 3);
                const uint32_t w = (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16);
                const int16_t * wv = ggml_tq_state_i16 + 4*((w >> (off & 7)) & TQ_STATE_MASK);
                const int8_t  * q = yb->qs + 2*(t - 16*k);
                acc += wv[0]*q[0] + wv[1]*q[1];
            }
            sumb += GGML_CPU_FP16_TO_FP32(yb->d) * (float) acc;
        }
        sumf += GGML_CPU_FP16_TO_FP32(*(const ggml_half *) x) * sumb;
    }

    *s = sumf * (1.0f / GGML_TQ_I16_SCALE);
}

void ggml_vec_dot_tqp5_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    ggml_vec_dot_tqp_q8_0_impl(n, s, vx, vy, 5);
}

void ggml_vec_dot_tqp6_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    ggml_vec_dot_tqp_q8_0_impl(n, s, vx, vy, 6);
}

void ggml_vec_dot_tqp7_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    ggml_vec_dot_tqp_q8_0_impl(n, s, vx, vy, 7);
}
#endif // GGML_USE_TRELLIS
