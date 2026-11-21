// CPU kernels of the ROCmFPx types: the fork's additions to ggml/src/ggml-cpu/ggml-cpu.c. Built in each CPU variant.
#define GGML_COMMON_IMPL_C
#include "../ggml-quants-ext.h"
#include "quants-ext.h"
#include "simd-mappings.h"

#include <math.h>
#include <string.h>
// from ggml/src/ggml-cpu/ggml-cpu.c after upstream line 217 (upstream 9048bad8d, fork 1fe05fa45)
static inline uint32_t ggml_rocmfpx_get_bits_cpu(const uint8_t * src, int bit_pos, int nbits) {
    const int byte_pos = bit_pos >> 3;
    const int shift    = bit_pos & 7;
    uint32_t v = src[byte_pos];

    v |= (uint32_t) src[byte_pos + 1] << 8;
    v |= (uint32_t) src[byte_pos + 2] << 16;

    return (v >> shift) & ((1u << nbits) - 1u);
}

static inline int ggml_rocmfpx_decode_fp3_cpu(uint32_t code) {
    static const int8_t table[8] = { 0, 1, 2, 4, 0, -1, -2, -4 };
    return table[code & 7u];
}

static inline int ggml_rocmfpx_decode_fp2_cpu(uint32_t code) {
    static const int8_t table[4] = { -4, -1, 1, 4 };
    return table[code & 3u];
}

static void ggml_vec_dot_rocmfpx_fp2_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    assert(n % QK_ROCMFP2 == 0);
    assert(QK_ROCMFP2 == QK8_0);

    const block_rocmfp2 * GGML_RESTRICT x = (const block_rocmfp2 *) vx;
    const block_q8_0    * GGML_RESTRICT y = (const block_q8_0 *) vy;
    const int nb = n / QK_ROCMFP2;
    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ++ib) {
        const float dy = GGML_CPU_FP16_TO_FP32(y[ib].d);
        int sumi0 = 0;
        int sumi1 = 0;
        for (int j = 0; j < QK_ROCMFP2/2; ++j) {
            const uint8_t c0 = (x[ib].qs[j/4] >> (2*(j % 4))) & 3u;
            const uint8_t c1 = (x[ib].qs[4 + j/4] >> (2*(j % 4))) & 3u;
            sumi0 += ggml_rocmfpx_decode_fp2_cpu(c0) * (int) y[ib].qs[j];
            sumi1 += ggml_rocmfpx_decode_fp2_cpu(c1) * (int) y[ib].qs[j + QK_ROCMFP2/2];
        }
        sumf += dy * (
            rocmfpx_ue4m3_to_fp32(x[ib].e[0]) * (float) sumi0 +
            rocmfpx_ue4m3_to_fp32(x[ib].e[1]) * (float) sumi1);
    }
    *s = sumf;
}

static inline int ggml_rocmfpx_decode_fp6_cpu(uint32_t code) {
    // asymmetric signed range [-32, 31]: sign|0 encodes -32, matching
    // rocmfpx_decode_fp6_code() in ggml/rocmfpx/rocmfpx.c
    const int mag = (int) (code & 31u);
    return (code & 32u) ? -(mag == 0 ? 32 : mag) : mag;
}

static void ggml_vec_dot_rocmfpx_fp3_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    assert(n % QK_ROCMFP3 == 0);
    assert(QK_ROCMFP3 == QK8_0);

    const block_rocmfp3 * GGML_RESTRICT x = (const block_rocmfp3 *) vx;
    const block_q8_0    * GGML_RESTRICT y = (const block_q8_0 *) vy;

    const int nb = n / QK_ROCMFP3;
    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ++ib) {
        const float dy = GGML_CPU_FP16_TO_FP32(y[ib].d);

        int sumi0 = 0;
        int sumi1 = 0;
        for (int j = 0; j < QK_ROCMFP3/2; ++j) {
            const int q0 = ggml_rocmfpx_decode_fp3_cpu(ggml_rocmfpx_get_bits_cpu(x[ib].qs, j*3, 3));
            const int q1 = ggml_rocmfpx_decode_fp3_cpu(ggml_rocmfpx_get_bits_cpu(x[ib].qs, (j + QK_ROCMFP3/2)*3, 3));
            sumi0 += q0 * (int) y[ib].qs[j];
            sumi1 += q1 * (int) y[ib].qs[j + QK_ROCMFP3/2];
        }

        sumf += dy * (
            rocmfpx_ue4m3_to_fp32(x[ib].e[0]) * (float) sumi0 +
            rocmfpx_ue4m3_to_fp32(x[ib].e[1]) * (float) sumi1);
    }

    *s = sumf;
}

static void ggml_vec_dot_rocmfpx_fp6_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    assert(n % QK_ROCMFP6 == 0);
    assert(QK_ROCMFP6 == QK8_0);

    const block_rocmfp6 * GGML_RESTRICT x = (const block_rocmfp6 *) vx;
    const block_q8_0    * GGML_RESTRICT y = (const block_q8_0 *) vy;

    const int nb = n / QK_ROCMFP6;
    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ++ib) {
        const float dy = GGML_CPU_FP16_TO_FP32(y[ib].d);

        int sumi0 = 0;
        int sumi1 = 0;
        for (int j = 0; j < QK_ROCMFP6/2; ++j) {
            const int q0 = ggml_rocmfpx_decode_fp6_cpu(ggml_rocmfpx_get_bits_cpu(x[ib].qs, j*6, 6));
            const int q1 = ggml_rocmfpx_decode_fp6_cpu(ggml_rocmfpx_get_bits_cpu(x[ib].qs, (j + QK_ROCMFP6/2)*6, 6));
            sumi0 += q0 * (int) y[ib].qs[j];
            sumi1 += q1 * (int) y[ib].qs[j + QK_ROCMFP6/2];
        }

        sumf += dy * (
            rocmfpx_ue4m3_to_fp32(x[ib].e[0]) * (float) sumi0 +
            rocmfpx_ue4m3_to_fp32(x[ib].e[1]) * (float) sumi1);
    }

    *s = sumf;
}

static void ggml_vec_dot_rocmfpx_fp8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    assert(n % QK_ROCMFP8 == 0);
    assert(QK_ROCMFP8 == QK8_0);

    const block_rocmfp8 * GGML_RESTRICT x = (const block_rocmfp8 *) vx;
    const block_q8_0    * GGML_RESTRICT y = (const block_q8_0 *) vy;

    const int nb = n / QK_ROCMFP8;
    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ++ib) {
        const float dx = rocmfpx_ue4m3_to_fp32(x[ib].e);
        const float dy = GGML_CPU_FP16_TO_FP32(y[ib].d);
        const float d  = dx * dy;

        int sumi = 0;
        for (int j = 0; j < QK_ROCMFP8; ++j) {
            sumi += (int) x[ib].qs[j] * (int) y[ib].qs[j];
        }

        sumf += d * (float) sumi;
    }

    *s = sumf;
}


// wrappers of the CPU type traits (HOOK(type-traits-cpu))
#define GGML_HOOK_CPU_ROCMFPX(TNAME, BITS) \
    void ggml_hook_type_##TNAME##_cpu_from_float(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) { \
        rocmfpx_quantize_row_fp##BITS(x, y, k); \
    } \
    void ggml_hook_type_##TNAME##_cpu_vec_dot(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x, \
                                              size_t bx, const void * GGML_RESTRICT y, size_t by, int nrc) { \
        ggml_vec_dot_rocmfpx_fp##BITS##_q8_0(n, s, bs, x, bx, y, by, nrc); \
    }

GGML_HOOK_CPU_ROCMFPX(q2_0_rocmfpx, 2)
GGML_HOOK_CPU_ROCMFPX(q3_0_rocmfpx, 3)
GGML_HOOK_CPU_ROCMFPX(q6_0_rocmfpx, 6)
GGML_HOOK_CPU_ROCMFPX(q8_0_rocmfpx, 8)
