// EXL3 (exllamav3) linear on the CPU, see ggml/exl3/README.md
//
// y = H(H(x*suh) W_hat) * svh/128 with H the unnormalized 128-point Hadamard: the two 1/sqrt(128) are one exact 1/128.
// W_hat is decoded tile by tile; every decoded value is exactly ggml_exl3_codebook_value() of its trellis state.

#include "exl3-ops.h"

#include "ggml-cpu-impl.h"
#include "ops.h"
#include "simd-mappings.h"
#include "../../exl3/exl3.h"

#include <cstring>
#include <mutex>

namespace {

constexpr int64_t HAD   = GGML_EXL3_HAD_SIZE;
constexpr int64_t TILE  = GGML_EXL3_TILE;
constexpr int64_t TILES = HAD/TILE; // tiles per Hadamard block

// ring position i holds tile element idx[i] = r*16 + c (tensor-core fragment order, as in ggml_exl3_decode_tile)
struct tile_perm {
    uint8_t idx[256] = {};

    constexpr tile_perm() {
        for (int i = 0; i < 256; i++) {
            const int t = i / 8;
            const int j = i % 8;
            const int r = (t % 4)*2 + (j & 1) + 8*((j >> 1) & 1);
            const int c = t / 4 + 8*((j >> 2) & 1);
            idx[i] = (uint8_t) (r*16 + c);
        }
    }
};

constexpr tile_perm perm;

float          lut[3][65536];
std::once_flag lut_once[3];

const float * get_lut(ggml_exl3_codebook cb) {
    std::call_once(lut_once[cb], [cb] {
        for (uint32_t s = 0; s < 65536; s++) {
            lut[cb][s] = ggml_fp16_to_fp32(ggml_exl3_codebook_value((uint16_t) s, cb));
        }
    });
    return lut[cb];
}

// one tile into 16 rows of dst with row stride ld; same ring walk as ggml_exl3_decode_tile
inline void decode_tile(const int16_t * tile, int tile_words, const float * cb_lut, float * dst, int64_t ld) {
    uint32_t ring[65];
    const int n_ring = tile_words / 2;
    for (int j = 0; j < n_ring; j++) {
        ring[j] = (uint32_t) (uint16_t) tile[2*j] | ((uint32_t) (uint16_t) tile[2*j + 1] << 16);
    }
    ring[n_ring] = ring[0]; // a state window can wrap past the last word

    const int n_bits = 16*tile_words;
    const int ka     = tile_words / 16;
    const int half   = tile_words % 16 != 0;

    int end = 0;
    for (int i = 0; i < 256; i++) {
        end += ka + (half ? (0xAAAA >> (i & 15)) & 1 : 0);
        int b0 = end - 16;
        if (b0 < 0) {
            b0 += n_bits;
        }
        const uint64_t pair  = ((uint64_t) ring[b0 >> 5] << 32) | ring[(b0 >> 5) + 1];
        const uint16_t state = (uint16_t) (pair >> (48 - (b0 & 31)));
        const int      e     = perm.idx[i];
        dst[(e >> 4)*ld + (e & 15)] = cb_lut[state];
    }
}

inline void fwht128(float * x) {
    for (int64_t h = 1; h < HAD; h <<= 1) {
        for (int64_t i = 0; i < HAD; i += 2*h) {
            for (int64_t j = i; j < i + h; j++) {
                const float a = x[j];
                const float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
}

// acc[q*16 + c] += sum_i x[q*ldx + i] * panel[i*16 + c] over one 128 block, for 4 rows q
// the block sum is kept apart from acc, so the rounding error grows with sqrt(128) + sqrt(k/128), not sqrt(k)
inline void gemm_rows4(const float * x, int64_t ldx, const float * panel, float * acc) {
    float s0[TILE] = {};
    float s1[TILE] = {};
    float s2[TILE] = {};
    float s3[TILE] = {};
    const float * x0 = x;
    const float * x1 = x + ldx;
    const float * x2 = x + 2*ldx;
    const float * x3 = x + 3*ldx;
    for (int64_t i = 0; i < HAD; i++) {
        const float * p = panel + i*TILE;
        const float a0 = x0[i];
        const float a1 = x1[i];
        const float a2 = x2[i];
        const float a3 = x3[i];
        for (int64_t c = 0; c < TILE; c++) {
            s0[c] += a0*p[c];
            s1[c] += a1*p[c];
            s2[c] += a2*p[c];
            s3[c] += a3*p[c];
        }
    }
    for (int64_t c = 0; c < TILE; c++) {
        acc[c]          += s0[c];
        acc[TILE + c]   += s1[c];
        acc[2*TILE + c] += s2[c];
        acc[3*TILE + c] += s3[c];
    }
}

inline void gemm_rows1(const float * x, const float * panel, float * acc) {
    float s[TILE] = {};
    for (int64_t i = 0; i < HAD; i++) {
        const float * p = panel + i*TILE;
        const float a = x[i];
        for (int64_t c = 0; c < TILE; c++) {
            s[c] += a*p[c];
        }
    }
    for (int64_t c = 0; c < TILE; c++) {
        acc[c] += s[c];
    }
}

// work buffer: [xh, nr x k][per thread: panel 128 x 16, acc nr x 16], each part aligned to a cache line
size_t xh_size(int64_t nr, int64_t k) {
    return GGML_PAD(nr*k*sizeof(float), CACHE_LINE_SIZE);
}

size_t thread_size(int64_t nr) {
    return GGML_PAD((HAD*TILE + nr*TILE)*sizeof(float), CACHE_LINE_SIZE);
}

} // namespace

bool ggml_cpu_exl3_supports_op(const ggml_tensor * op) {
    const ggml_tensor * trellis = op->src[0];
    const ggml_tensor * x       = op->src[1];
    const int           cb      = ggml_get_op_params_i32(op, 0);
    return op->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) &&
           ggml_exl3_rate_supported(trellis->ne[0], (ggml_exl3_codebook) cb);
}

size_t ggml_cpu_exl3_work_size(const ggml_tensor * op, int n_tasks) {
    const ggml_tensor * x = op->src[1];
    return CACHE_LINE_SIZE + xh_size(ggml_nrows(x), x->ne[0]) + n_tasks*thread_size(ggml_nrows(x));
}

void ggml_compute_forward_exl3_mul_mat(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * trellis = dst->src[0];
    const ggml_tensor * x       = dst->src[1];
    const ggml_tensor * suh     = dst->src[2];
    const ggml_tensor * svh     = dst->src[3];

    const ggml_exl3_codebook cb = (ggml_exl3_codebook) ggml_get_op_params_i32(dst, 0);

    const int     tile_words = (int) trellis->ne[0];
    const int64_t tn         = trellis->ne[1];
    const int64_t n          = tn*TILE;
    const int64_t k          = trellis->ne[2]*TILE;
    const int64_t nr         = ggml_nrows(x);

    GGML_ASSERT(x->nb[0] == sizeof(float) && ggml_is_contiguous(dst));
    GGML_ASSERT(params->wsize >= ggml_cpu_exl3_work_size(dst, params->nth));

    const int ith = params->ith;
    const int nth = params->nth;

    char  * wdata = (char *) GGML_PAD((uintptr_t) params->wdata, CACHE_LINE_SIZE);
    float * xh    = (float *) wdata;
    float * panel = (float *) (wdata + xh_size(nr, k) + ith*thread_size(nr));
    float * acc   = panel + HAD*TILE;
    float * y     = (float *) dst->data;

    const float * cb_lut = get_lut(cb);

    // xh = H(x*suh)
    const ggml_fp16_t * su = (const ggml_fp16_t *) suh->data;
    for (int64_t r = ith; r < nr; r += nth) {
        const int64_t i11 = r % x->ne[1];
        const int64_t i12 = (r / x->ne[1]) % x->ne[2];
        const int64_t i13 = r / (x->ne[1]*x->ne[2]);
        const float * xr  = (const float *) ((const char *) x->data + i11*x->nb[1] + i12*x->nb[2] + i13*x->nb[3]);
        float       * dr  = xh + r*k;
        for (int64_t i = 0; i < k; i++) {
            dr[i] = xr[i]*GGML_CPU_FP16_TO_FP32(su[i]);
        }
        for (int64_t i = 0; i < k; i += HAD) {
            fwht128(dr + i);
        }
    }

    if (ith == 0) {
        ggml_threadpool_chunk_set(params->threadpool, nth);
    }
    ggml_barrier(params->threadpool);

    // y = xh W_hat, one column tile per chunk; each 128 x 16 panel of W_hat is decoded once for all rows
    const int16_t * tq = (const int16_t *) trellis->data;
    for (int64_t nt = ith; nt < tn; nt = ggml_threadpool_chunk_add(params->threadpool, 1)) {
        memset(acc, 0, nr*TILE*sizeof(float));
        for (int64_t kb = 0; kb < k/HAD; kb++) {
            for (int64_t t = 0; t < TILES; t++) {
                decode_tile(tq + ((kb*TILES + t)*tn + nt)*tile_words, tile_words, cb_lut, panel + t*TILE*TILE, TILE);
            }
            const float * xb = xh + kb*HAD;
            int64_t r = 0;
            for (; r + 4 <= nr; r += 4) {
                gemm_rows4(xb + r*k, k, panel, acc + r*TILE);
            }
            for (; r < nr; r++) {
                gemm_rows1(xb + r*k, panel, acc + r*TILE);
            }
        }
        for (int64_t r = 0; r < nr; r++) {
            memcpy(y + r*n + nt*TILE, acc + r*TILE, TILE*sizeof(float));
        }
    }

    ggml_barrier(params->threadpool);

    // y = H(y) * svh/128
    const ggml_fp16_t * sv = (const ggml_fp16_t *) svh->data;
    const int64_t       nb = n/HAD;
    for (int64_t j = ith; j < nr*nb; j += nth) {
        const int64_t b  = j % nb;
        float       * yb = y + (j / nb)*n + b*HAD;
        fwht128(yb);
        for (int64_t i = 0; i < HAD; i++) {
            yb[i] *= GGML_CPU_FP16_TO_FP32(sv[b*HAD + i])*(1.0f/HAD);
        }
    }
}

bool ggml_cpu_exl3_get_rows_supports_op(const ggml_tensor * op) {
    const ggml_tensor * table = op->src[0];
    const ggml_tensor * signs = op->src[2];
    return op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) &&
           ggml_exl3_embd_k(table->ne[0], signs->ne[1]) > 0;
}

// each row is decoded by ggml_exl3_decode_embd_row, the reference decoder, so the values are its values
void ggml_compute_forward_exl3_get_rows(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * table = dst->src[0];
    const ggml_tensor * ids   = dst->src[1];
    const ggml_tensor * signs = dst->src[2];

    const int64_t n_out  = ggml_get_op_params_i32(dst, 0);
    const int64_t groups = signs->ne[1];
    const int     K      = ggml_exl3_embd_k(table->ne[0], groups);
    GGML_ASSERT(K > 0 && dst->ne[0] == n_out && ggml_is_contiguous(dst));

    for (int64_t i = params->ith; i < ids->ne[0]; i += params->nth) {
        const int32_t id = *(const int32_t *) ((const char *) ids->data + i*ids->nb[0]);
        GGML_ASSERT(id >= 0 && id < table->ne[1]);
        ggml_exl3_decode_embd_row((const int16_t *) ((const char *) table->data + id*table->nb[1]), groups, K,
                (const ggml_fp16_t *) signs->data, n_out, (float *) ((char *) dst->data + i*dst->nb[1]));
    }
}
