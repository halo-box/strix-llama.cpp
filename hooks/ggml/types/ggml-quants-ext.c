// Reference codecs of the fork types: the fork's additions to ggml/src/ggml-quants.c. Built in ggml-base.
#define GGML_COMMON_IMPL_C
#include "ggml-quants-ext.h"
#include "ggml-impl.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define GGML_USE_TRELLIS
// from ggml/src/ggml-quants.c after upstream line 4 (upstream 9048bad8d, fork 1fe05fa45)
#include "rocmfp4/rocmfp4.h"
#include "rocmfpx/rocmfpx.h"

// from ggml/src/ggml-quants.c after upstream line 616 (upstream 9048bad8d, fork 1fe05fa45)
void dequantize_row_q4_0_rocmfp4(const block_rocmfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    rocmfp4_dequantize_row_q4_0(x, y, k);
}

void dequantize_row_q4_0_rocmfp4_fast(const block_rocmfp4_fast * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    rocmfp4_dequantize_row_q4_0_fast(x, y, k);
}

void dequantize_row_q2_0_rocmfpx(const block_rocmfp2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    rocmfpx_dequantize_row_fp2(x, y, k);
}

void dequantize_row_q3_0_rocmfpx(const block_rocmfp3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    rocmfpx_dequantize_row_fp3(x, y, k);
}

void dequantize_row_q6_0_rocmfpx(const block_rocmfp6 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    rocmfpx_dequantize_row_fp6(x, y, k);
}

void dequantize_row_q8_0_rocmfpx(const block_rocmfp8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    rocmfpx_dequantize_row_fp8(x, y, k);
}


// from ggml/src/ggml-quants.c after upstream line 2137 (upstream 9048bad8d, fork 1fe05fa45)
// ====================== PTQ1_0 (Prism ternary, group 128) ======================
// Base-3 trit packing identical to upstream TQ1_0, but at block 128 so one fp16
// scale covers 128 weights. qs is 24 bytes, which TQ1_0's fixed 32-then-16 byte
// staging cannot cover, so the stages are generalised to 32/16/8; at TQ1_0's
// 48-byte qs this reduces to exactly its original 32-then-16 behaviour.
static const size_t ptq1_0_stages[3] = {32, 16, 8};

void quantize_row_ptq1_0_ref(const float * GGML_RESTRICT x, block_ptq1_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PTQ1_0 == 0);
    const int64_t nb = k / QK_PTQ1_0;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK_PTQ1_0; j++) {
            amax = MAX(amax, fabsf(x[j]));
        }

        const float d  = amax;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        size_t j = 0;
        for (size_t s = 0; s < 3; ++s) {
            const size_t c = ptq1_0_stages[s];
            for (; j + c <= sizeof(y->qs); j += c) {
                for (size_t m = 0; m < c; ++m) {
                    uint8_t q = 0;
                    for (size_t n = 0; n < 5; ++n) {
                        int xi = lroundf(x[m + n*c] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                        q *= 3;
                        q += xi;
                    }
                    // ceiling division (243 == pow(3, 5))
                    q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                    y[i].qs[j + m] = q;
                }
                x += 5*c;
            }
        }
        // 4 elements per byte
        for (size_t h = 0; h < sizeof(y->qh); ++h) {
            uint8_t q = 0;
            for (size_t m = 0; m < 4; ++m) {
                int xi = lroundf(x[h + m*sizeof(y->qh)] * id) + 1;
                q *= 3;
                q += xi;
            }
            // shift the first value to the most significant trit
            q *= 3;
            q = ((uint16_t)q * 256 + (243 - 1)) / 243;
            y[i].qh[h] = q;
        }
        x += 4*sizeof(y->qh);
    }
}

void dequantize_row_ptq1_0(const block_ptq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PTQ1_0 == 0);
    const int64_t nb = k / QK_PTQ1_0;

    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    for (int64_t i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        size_t j = 0;
        for (size_t s = 0; s < 3; ++s) {
            const size_t c = ptq1_0_stages[s];
            for (; j + c <= sizeof(x->qs); j += c) {
                for (size_t n = 0; n < 5; ++n) {
                    for (size_t m = 0; m < c; ++m) {
                        uint8_t q = x[i].qs[j + m] * pow3[n];
                        int16_t xi = ((uint16_t) q * 3) >> 8;
                        *y++ = (float) (xi - 1) * d;
                    }
                }
            }
        }
        for (size_t n = 0; n < 4; ++n) {
            for (size_t h = 0; h < sizeof(x->qh); ++h) {
                uint8_t q = x[i].qh[h] * pow3[n];
                int16_t xi = ((uint16_t) q * 3) >> 8;
                *y++ = (float) (xi - 1) * d;
            }
        }
    }
}

size_t quantize_ptq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights; // ternary codes come from the weights themselves; an imatrix has no role
    const size_t row_size = ggml_row_size(GGML_TYPE_PTQ1_0, n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_ptq1_0_ref(src, (block_ptq1_0 *)qrow, n_per_row);
        src  += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

// ====================== TQ2_T (trellis, 2.125 bpw, group 128) ======================
// Decoder is the format definition (bit-exact with the encoder's reference decoder).
// Encoder: PLACEHOLDER. Real TQ2_T files come from an offline encoder; this is a
// greedy coordinate descent over the 32 path bytes (four sweeps, each byte chosen to
// minimise the error of the two steps that read it), followed by a least-squares
// refit of d. It is correct -- any byte string decodes -- but far from optimal. It
// exists so ggml's from_float (test-backend-ops, CPU reference paths) has something.

static inline float tq2t_lut_val(uint32_t i) {
    return GGML_FP16_TO_FP32(tq2t_lut_f16[i]);
}

// Squared error of step t (weights 4t..4t+3 of y) under state (a << 8) | b.
static float tq2t_step_err(const float * GGML_RESTRICT y, int t, uint32_t a, uint32_t b) {
    const uint32_t s = (a << 8) | b;
    float err = 0.0f;
    for (int h = 0; h < 2; ++h) {
        const uint32_t idx = tq2t_hyb_index(2*s + h);
        const float dx = y[4*t + 2*h + 0] - tq2t_lut_val(2*idx + 0);
        const float dy = y[4*t + 2*h + 1] - tq2t_lut_val(2*idx + 1);
        err += dx*dx + dy*dy;
    }
    return err;
}

void quantize_row_tq2_t_ref(const float * GGML_RESTRICT x, block_tq2_t * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ2_T == 0);
    const int64_t nb = k / QK_TQ2_T;
    const int nsteps = QK_TQ2_T/4;

    for (int64_t i = 0; i < nb; i++, x += QK_TQ2_T) {
        float sumsq = 0.0f;
        for (int j = 0; j < QK_TQ2_T; j++) {
            sumsq += x[j]*x[j];
        }
        memset(y[i].qs, 0, sizeof(y[i].qs));
        if (sumsq == 0.0f) {
            y[i].d = GGML_FP32_TO_FP16(0.0f);
            continue;
        }
        const float rms = sqrtf(sumsq / QK_TQ2_T);
        float yn[QK_TQ2_T];
        for (int j = 0; j < QK_TQ2_T; j++) {
            yn[j] = x[j] / rms;
        }

        uint8_t * qs = y[i].qs;
        for (int pass = 0; pass < 4; ++pass) {
            for (int t = 0; t < nsteps; ++t) {
                const uint32_t prev = qs[(t + nsteps - 1) % nsteps];
                const uint32_t next = qs[(t + 1) % nsteps];
                float best = INFINITY;
                int best_c = 0;
                for (int c = 0; c < 256; ++c) {
                    const float e = tq2t_step_err(yn, t, prev, c) + tq2t_step_err(yn, (t + 1) % nsteps, c, next);
                    if (e < best) {
                        best = e;
                        best_c = c;
                    }
                }
                qs[t] = (uint8_t) best_c;
            }
        }

        // least-squares scale for the chosen path
        float sxc = 0.0f, scc = 0.0f;
        for (int t = 0; t < nsteps; ++t) {
            const uint32_t s = ((uint32_t) qs[(t + nsteps - 1) % nsteps] << 8) | qs[t];
            for (int j = 0; j < 4; ++j) {
                const float c = tq2t_lut_val(2*tq2t_hyb_index(2*s + (j >> 1)) + (j & 1));
                sxc += x[4*t + j] * c;
                scc += c * c;
            }
        }
        y[i].d = GGML_FP32_TO_FP16(scc > 0.0f ? sxc / scc : rms);
    }
}

void dequantize_row_tq2_t(const block_tq2_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ2_T == 0);
    const int64_t nb = k / QK_TQ2_T;
    const int nsteps = QK_TQ2_T/4;

    for (int64_t i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        for (int t = 0; t < nsteps; ++t) {
            const uint32_t s = ((uint32_t) qs[(t + nsteps - 1) % nsteps] << 8) | qs[t];
            for (int h = 0; h < 2; ++h) {
                const uint32_t idx = tq2t_hyb_index(2*s + h);
                *y++ = d * tq2t_lut_val(2*idx + 0);
                *y++ = d * tq2t_lut_val(2*idx + 1);
            }
        }
    }
}

size_t quantize_tq2_t(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights; // placeholder encoder ignores importance weights
    const size_t row_size = ggml_row_size(GGML_TYPE_TQ2_T, n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_tq2_t_ref(src, (block_tq2_t *)qrow, n_per_row);
        src  += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

// ====================== TQK6 / TQK7 (bit-packed trellis, group 128) ======================
// Same codebook and value formula as TQ2_T; the state of step t is the 16-bit circular
// window at stream bit (31 - t)*K (tqk_state in ggml-common.h). Decoder is the format
// definition (bit-exact with the encoder's reference decoder). Encoder: PLACEHOLDER, a greedy
// coordinate descent over the 32 K-bit chunks (chunk t = step t's new bits, at bit
// (31 - t)*K; it is read by the windows of steps t, t+1, t+2 mod 32 since K > 5) at
// three target scales, each followed by a least-squares refit of d; the best is kept. Real model files come from an offline encoder.

// Write step t's K new bits (bits (31 - t)*K .. +K-1, never wrapping).
static void tqk_set_chunk(uint8_t * GGML_RESTRICT qs, int k, int t, uint32_t v) {
    const int off = (31 - t) * k;
    for (int b = 0; b < k; ++b) {
        const int i = off + b;
        if ((v >> b) & 1) {
            qs[i >> 3] |=  (uint8_t) (1u << (i & 7));
        } else {
            qs[i >> 3] &= (uint8_t) ~(1u << (i & 7));
        }
    }
}

static float tqk_step_err(const float * GGML_RESTRICT y, int t, uint32_t s) {
    float err = 0.0f;
    for (int h = 0; h < 2; ++h) {
        const uint32_t idx = tq2t_hyb_index(2*s + h);
        const float dx = y[4*t + 2*h + 0] - tq2t_lut_val(2*idx + 0);
        const float dy = y[4*t + 2*h + 1] - tq2t_lut_val(2*idx + 1);
        err += dx*dx + dy*dy;
    }
    return err;
}

static void quantize_row_tqk_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k, int kb) {
    assert(k % QK_TQK == 0);
    assert(kb == 6 || kb == 7);
    const int64_t nb = k / QK_TQK;
    const int nsteps = QK_TQK/4;
    const size_t qs_bytes = 4*kb;
    const size_t bsize = sizeof(ggml_half) + qs_bytes;
    uint8_t * y = (uint8_t *) vy;

    for (int64_t i = 0; i < nb; i++, x += QK_TQK, y += bsize) {
        ggml_half * dp = (ggml_half *) y;
        uint8_t * qs = y + sizeof(ggml_half);
        memset(qs, 0, qs_bytes);

        float sumsq = 0.0f;
        for (int j = 0; j < QK_TQK; j++) {
            sumsq += x[j]*x[j];
        }
        if (sumsq == 0.0f) {
            *dp = GGML_FP32_TO_FP16(0.0f);
            continue;
        }
        const float rms = sqrtf(sumsq / QK_TQK);

        // A few target scales (the codebook is unit-variance Gaussian, the data may not
        // be); keep the path whose least-squares-rescaled error is lowest.
        static const float scales[3] = { 0.8f, 1.0f, 1.25f };
        uint8_t cand[4*7];
        float best_err = INFINITY;
        float best_d = rms;
        for (int si = 0; si < 3; ++si) {
            float yn[QK_TQK];
            for (int j = 0; j < QK_TQK; j++) {
                yn[j] = x[j] * scales[si] / rms;
            }
            memset(cand, 0, qs_bytes);

            for (int pass = 0; pass < 6; ++pass) {
                for (int t = 0; t < nsteps; ++t) {
                    float best = INFINITY;
                    uint32_t best_v = 0;
                    for (uint32_t v = 0; v < (1u << kb); ++v) {
                        tqk_set_chunk(cand, kb, t, v);
                        float e = 0.0f;
                        for (int m = 0; m < 3; ++m) {
                            const int tt = (t + m) % nsteps;
                            e += tqk_step_err(yn, tt, tqk_state(cand, kb, tt));
                        }
                        if (e < best) {
                            best = e;
                            best_v = v;
                        }
                    }
                    tqk_set_chunk(cand, kb, t, best_v);
                }
            }

            // least-squares scale for this path, and its error at that scale
            float sxc = 0.0f, scc = 0.0f;
            for (int t = 0; t < nsteps; ++t) {
                const uint32_t s = tqk_state(cand, kb, t);
                for (int j = 0; j < 4; ++j) {
                    const float c = tq2t_lut_val(2*tq2t_hyb_index(2*s + (j >> 1)) + (j & 1));
                    sxc += x[4*t + j] * c;
                    scc += c * c;
                }
            }
            const float d = scc > 0.0f ? sxc / scc : rms;
            const float err = sumsq - 2.0f*d*sxc + d*d*scc;
            if (err < best_err) {
                best_err = err;
                best_d = d;
                memcpy(qs, cand, qs_bytes);
            }
        }
        *dp = GGML_FP32_TO_FP16(best_d);
    }
}

static void dequantize_row_tqk_impl(const void * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k, int kb) {
    assert(k % QK_TQK == 0);
    const int64_t nb = k / QK_TQK;
    const int nsteps = QK_TQK/4;
    const size_t bsize = sizeof(ggml_half) + 4*kb;
    const uint8_t * x = (const uint8_t *) vx;

    for (int64_t i = 0; i < nb; ++i, x += bsize) {
        const float d = GGML_FP16_TO_FP32(*(const ggml_half *) x);
        const uint8_t * qs = x + sizeof(ggml_half);
        for (int t = 0; t < nsteps; ++t) {
            const uint32_t s = tqk_state(qs, kb, t);
            for (int h = 0; h < 2; ++h) {
                const uint32_t idx = tq2t_hyb_index(2*s + h);
                *y++ = d * tq2t_lut_val(2*idx + 0);
                *y++ = d * tq2t_lut_val(2*idx + 1);
            }
        }
    }
}

void quantize_row_tqk6_ref(const float * GGML_RESTRICT x, block_tqk6 * GGML_RESTRICT y, int64_t k) {
    quantize_row_tqk_impl(x, y, k, 6);
}

void quantize_row_tqk7_ref(const float * GGML_RESTRICT x, block_tqk7 * GGML_RESTRICT y, int64_t k) {
    quantize_row_tqk_impl(x, y, k, 7);
}

void dequantize_row_tqk6(const block_tqk6 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    dequantize_row_tqk_impl(x, y, k, 6);
}

void dequantize_row_tqk7(const block_tqk7 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    dequantize_row_tqk_impl(x, y, k, 7);
}

size_t quantize_tqk6(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights; // placeholder encoder ignores importance weights
    const size_t row_size = ggml_row_size(GGML_TYPE_TQK6, n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_tqk6_ref(src, (block_tqk6 *)qrow, n_per_row);
        src  += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

size_t quantize_tqk7(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights; // placeholder encoder ignores importance weights
    const size_t row_size = ggml_row_size(GGML_TYPE_TQK7, n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_tqk7_ref(src, (block_tqk7 *)qrow, n_per_row);
        src  += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

// ====================== TQP5 / TQP6 / TQP7 (pair trellis, group 128) ======================
// 64 steps of 2 weights; the state of step t is the 16-bit circular window at stream bit
// (63 - t)*K (tqp_state in ggml-common.h), and its pair is TQ2_T's first pair of that state.
// Decoder is the format definition.
// Encoder: PLACEHOLDER, a greedy coordinate descent over the 64 K-bit chunks (chunk t is read
// by the 15-bit windows of steps t, t+1, t+2 mod 64) at three target scales, each followed by a
// least-squares refit of d; the best is kept. Real model files come from an offline encoder.

static void tqp_set_chunk(uint8_t * GGML_RESTRICT qs, int k, int t, uint32_t v) {
    const int off = (63 - t) * k;
    for (int b = 0; b < k; ++b) {
        const int i = off + b;
        if ((v >> b) & 1) {
            qs[i >> 3] |=  (uint8_t) (1u << (i & 7));
        } else {
            qs[i >> 3] &= (uint8_t) ~(1u << (i & 7));
        }
    }
}

static inline float tqp_step_err(const float * GGML_RESTRICT y, int t, uint32_t s) {
    const uint32_t idx = tq2t_hyb_index(2*s);
    const float dx = y[2*t + 0] - tq2t_lut_val(2*idx + 0);
    const float dy = y[2*t + 1] - tq2t_lut_val(2*idx + 1);
    return dx*dx + dy*dy;
}

static void quantize_row_tqp_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k, int kb) {
    assert(k % QK_TQP == 0);
    assert(kb >= 5 && kb <= 7);
    const int64_t nb = k / QK_TQP;
    const int nsteps = QK_TQP/2;
    const size_t qs_bytes = 8*kb;
    const size_t bsize = sizeof(ggml_half) + qs_bytes;
    uint8_t * y = (uint8_t *) vy;

    for (int64_t i = 0; i < nb; i++, x += QK_TQP, y += bsize) {
        ggml_half * dp = (ggml_half *) y;
        uint8_t * qs = y + sizeof(ggml_half);
        memset(qs, 0, qs_bytes);

        float sumsq = 0.0f;
        for (int j = 0; j < QK_TQP; j++) {
            sumsq += x[j]*x[j];
        }
        if (sumsq == 0.0f) {
            *dp = GGML_FP32_TO_FP16(0.0f);
            continue;
        }
        const float rms = sqrtf(sumsq / QK_TQP);

        static const float scales[3] = { 0.8f, 1.0f, 1.25f };
        uint8_t cand[8*7];
        float best_err = INFINITY;
        float best_d = rms;
        for (int si = 0; si < 3; ++si) {
            float yn[QK_TQP];
            for (int j = 0; j < QK_TQP; j++) {
                yn[j] = x[j] * scales[si] / rms;
            }
            memset(cand, 0, qs_bytes);

            for (int pass = 0; pass < 6; ++pass) {
                for (int t = 0; t < nsteps; ++t) {
                    float best = INFINITY;
                    uint32_t best_v = 0;
                    for (uint32_t v = 0; v < (1u << kb); ++v) {
                        tqp_set_chunk(cand, kb, t, v);
                        float e = 0.0f;
                        for (int m = 0; m < 3; ++m) {
                            const int tt = (t + m) % nsteps;
                            e += tqp_step_err(yn, tt, tqp_state(cand, kb, tt));
                        }
                        if (e < best) {
                            best = e;
                            best_v = v;
                        }
                    }
                    tqp_set_chunk(cand, kb, t, best_v);
                }
            }

            float sxc = 0.0f, scc = 0.0f;
            for (int t = 0; t < nsteps; ++t) {
                const uint32_t idx = tq2t_hyb_index(2*tqp_state(cand, kb, t));
                for (int j = 0; j < 2; ++j) {
                    const float c = tq2t_lut_val(2*idx + j);
                    sxc += x[2*t + j] * c;
                    scc += c * c;
                }
            }
            const float d = scc > 0.0f ? sxc / scc : rms;
            const float err = sumsq - 2.0f*d*sxc + d*d*scc;
            if (err < best_err) {
                best_err = err;
                best_d = d;
                memcpy(qs, cand, qs_bytes);
            }
        }
        *dp = GGML_FP32_TO_FP16(best_d);
    }
}

static void dequantize_row_tqp_impl(const void * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k, int kb) {
    assert(k % QK_TQP == 0);
    const int64_t nb = k / QK_TQP;
    const int nsteps = QK_TQP/2;
    const size_t bsize = sizeof(ggml_half) + 8*kb;
    const uint8_t * x = (const uint8_t *) vx;

    for (int64_t i = 0; i < nb; ++i, x += bsize) {
        const float d = GGML_FP16_TO_FP32(*(const ggml_half *) x);
        const uint8_t * qs = x + sizeof(ggml_half);
        for (int t = 0; t < nsteps; ++t) {
            const uint32_t idx = tq2t_hyb_index(2*tqp_state(qs, kb, t));
            *y++ = d * tq2t_lut_val(2*idx + 0);
            *y++ = d * tq2t_lut_val(2*idx + 1);
        }
    }
}

#define TQP_DEFINE(K)                                                                                  \
void quantize_row_tqp##K##_ref(const float * GGML_RESTRICT x, block_tqp##K * GGML_RESTRICT y, int64_t k) { \
    quantize_row_tqp_impl(x, y, k, K);                                                                 \
}                                                                                                      \
void dequantize_row_tqp##K(const block_tqp##K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {  \
    dequantize_row_tqp_impl(x, y, k, K);                                                               \
}                                                                                                      \
size_t quantize_tqp##K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) { \
    (void)quant_weights;                                                                               \
    const size_t row_size = ggml_row_size(GGML_TYPE_TQP##K, n_per_row);                                \
    char * qrow = (char *)dst;                                                                         \
    for (int64_t row = 0; row < nrow; ++row) {                                                         \
        quantize_row_tqp##K##_ref(src, (block_tqp##K *)qrow, n_per_row);                               \
        src  += n_per_row;                                                                             \
        qrow += row_size;                                                                              \
    }                                                                                                  \
    return nrow * row_size;                                                                            \
}

TQP_DEFINE(5)
TQP_DEFINE(6)
TQP_DEFINE(7)

#undef TQP_DEFINE


