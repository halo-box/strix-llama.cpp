// EXL3 trellis decode, CPU reference.
//
// Format: turboderp-org/exllamav3 (MIT), read at a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a.
// Checked against its independent decoder tests/testlib/trellis.py. See README.md.

#include "exl3.h"

#include <stdlib.h>
#include <string.h>

int ggml_exl3_codebook_from_name(const char * name) {
    if (strcmp(name, "3inst") == 0) {
        return GGML_EXL3_CODEBOOK_3INST;
    }
    if (strcmp(name, "mcg") == 0) {
        return GGML_EXL3_CODEBOOK_MCG;
    }
    if (strcmp(name, "mul1") == 0) {
        return GGML_EXL3_CODEBOOK_MUL1;
    }
    return -1;
}

const char * ggml_exl3_codebook_name(enum ggml_exl3_codebook cb) {
    switch (cb) {
        case GGML_EXL3_CODEBOOK_3INST: return "3inst";
        case GGML_EXL3_CODEBOOK_MCG:   return "mcg";
        case GGML_EXL3_CODEBOOK_MUL1:  return "mul1";
    }
    return "unknown";
}

bool ggml_exl3_rate_supported(int64_t tile_words, enum ggml_exl3_codebook cb) {
    if (cb != GGML_EXL3_CODEBOOK_3INST && cb != GGML_EXL3_CODEBOOK_MCG && cb != GGML_EXL3_CODEBOOK_MUL1) {
        return false;
    }
    if (tile_words % 16 == 0) {
        return tile_words >= 16 && tile_words <= 128;
    }
    // half rates: 16*K = 24, 40, 56
    return cb == GGML_EXL3_CODEBOOK_MUL1 && (tile_words == 24 || tile_words == 40 || tile_words == 56);
}

static float exl3_half_to_float(uint16_t bits) {
    ggml_fp16_t h;
    memcpy(&h, &bits, sizeof(h));
    return ggml_fp16_to_fp32(h);
}

ggml_fp16_t ggml_exl3_codebook_value(uint16_t state, enum ggml_exl3_codebook cb) {
    // uint32 arithmetic wraps; each result is one fp16 rounding of an exact float sum (exact in fp32 here)
    uint32_t x = state;
    if (cb == GGML_EXL3_CODEBOOK_MUL1) {
        x *= GGML_EXL3_MUL1_MULT;
        const uint32_t bsum = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24);
        const float h = (float) (1024 + bsum); // fp16 bits 0x6400 + bsum
        return ggml_fp32_to_fp16(h * exl3_half_to_float(0x1eee) + exl3_half_to_float(0xc931));
    }
    if (cb == GGML_EXL3_CODEBOOK_MCG) {
        x *= GGML_EXL3_MCG_MULT;
    } else {
        x = x * 89226354u + 64248484u;
    }
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    return ggml_fp32_to_fp16(exl3_half_to_float((uint16_t) (x & 0xffff)) + exl3_half_to_float((uint16_t) (x >> 16)));
}

void ggml_exl3_decode_tile(const int16_t * tile, int64_t tile_words, enum ggml_exl3_codebook cb, ggml_fp16_t * dst) {
    GGML_ASSERT(ggml_exl3_rate_supported(tile_words, cb));

    // the ring: 32-bit words from little-endian uint16 pairs, ring bit p is bit 31 - p%32 of word p/32
    uint32_t ring[64];
    const int n_ring = (int) tile_words / 2;
    for (int j = 0; j < n_ring; j++) {
        ring[j] = (uint32_t) (uint16_t) tile[2*j] | ((uint32_t) (uint16_t) tile[2*j + 1] << 16);
    }
    const int n_bits = 16 * (int) tile_words;
    const int ka     = (int) tile_words / 16;
    const int half   = tile_words % 16 != 0;

    int end = 0;
    for (int i = 0; i < 256; i++) {
        // position i adds D(i) bits; half rates give odd positions one more bit (mask 0xAAAA)
        end += ka + (half ? (0xAAAA >> (i & 15)) & 1 : 0);

        // the state is ring bits [end - 16, end), MSB first, wrapping (tail-biting)
        int b0 = end - 16;
        if (b0 < 0) {
            b0 += n_bits;
        }
        const int      wi    = b0 >> 5;
        const uint64_t pair  = ((uint64_t) ring[wi] << 32) | ring[(wi + 1) % n_ring];
        const uint16_t state = (uint16_t) (pair >> (48 - (b0 & 31)));

        // ring position i = 8*t + j holds tile element (r, c), the tensor-core fragment order
        const int t = i / 8;
        const int j = i % 8;
        const int r = (t % 4)*2 + (j & 1) + 8*((j >> 1) & 1);
        const int c = t / 4 + 8*((j >> 2) & 1);
        dst[r*16 + c] = ggml_exl3_codebook_value(state, cb);
    }
}

void ggml_exl3_decode_rotated(
        const int16_t * trellis, int64_t k, int64_t n, int64_t tile_words, enum ggml_exl3_codebook cb, ggml_fp16_t * dst) {
    GGML_ASSERT(k % GGML_EXL3_TILE == 0 && n % GGML_EXL3_TILE == 0);

    const int64_t tk = k / GGML_EXL3_TILE;
    const int64_t tn = n / GGML_EXL3_TILE;
    ggml_fp16_t tile[256];
    for (int64_t kt = 0; kt < tk; kt++) {
        for (int64_t nt = 0; nt < tn; nt++) {
            ggml_exl3_decode_tile(trellis + (kt*tn + nt)*tile_words, tile_words, cb, tile);
            for (int r = 0; r < 16; r++) {
                memcpy(dst + (kt*16 + r)*n + nt*16, tile + r*16, 16*sizeof(ggml_fp16_t));
            }
        }
    }
}

// unnormalized natural-order Walsh-Hadamard transform of 128 values with stride s
static void exl3_fwht128(double * x, int s) {
    for (int h = 1; h < GGML_EXL3_HAD_SIZE; h <<= 1) {
        for (int i = 0; i < GGML_EXL3_HAD_SIZE; i += 2*h) {
            for (int j = i; j < i + h; j++) {
                const double a = x[j*s];
                const double b = x[(j + h)*s];
                x[j*s]       = a + b;
                x[(j + h)*s] = a - b;
            }
        }
    }
}

void ggml_exl3_dequantize(
        const int16_t * trellis, const ggml_fp16_t * suh, const ggml_fp16_t * svh,
        int64_t k, int64_t n, int64_t tile_words, enum ggml_exl3_codebook cb, float * dst) {
    const int B = GGML_EXL3_HAD_SIZE;
    GGML_ASSERT(k % B == 0 && n % B == 0);

    // H is block diagonal: each 128 x 128 block of W needs only its own 8 x 8 tiles
    // order as in exllamav3: H on the left, suh, H on the right, svh. W_hat values are multiples of 2^-24 below 4,
    // so in double the butterflies and suh are exact, svh rounds once, and the two 1/sqrt(128) are one exact 1/128
    const int64_t tn = n / GGML_EXL3_TILE;
    static const double inv_b = 1.0 / 128.0;

    double * blk = (double *) malloc(sizeof(double)*B*B);
    GGML_ASSERT(blk != NULL);
    ggml_fp16_t tile[256];

    for (int64_t bk = 0; bk < k / B; bk++) {
        for (int64_t bn = 0; bn < n / B; bn++) {
            for (int ti = 0; ti < B / GGML_EXL3_TILE; ti++) {
                for (int tj = 0; tj < B / GGML_EXL3_TILE; tj++) {
                    const int64_t kt = bk*(B/GGML_EXL3_TILE) + ti;
                    const int64_t nt = bn*(B/GGML_EXL3_TILE) + tj;
                    ggml_exl3_decode_tile(trellis + (kt*tn + nt)*tile_words, tile_words, cb, tile);
                    for (int r = 0; r < 16; r++) {
                        for (int c = 0; c < 16; c++) {
                            blk[(ti*16 + r)*B + tj*16 + c] = ggml_fp16_to_fp32(tile[r*16 + c]);
                        }
                    }
                }
            }
            for (int c = 0; c < B; c++) {
                exl3_fwht128(blk + c, B);
            }
            for (int r = 0; r < B; r++) {
                const double su = ggml_fp16_to_fp32(suh[bk*B + r]);
                double * row = blk + r*B;
                for (int c = 0; c < B; c++) {
                    row[c] *= su;
                }
                exl3_fwht128(row, 1);
                float * out = dst + (bk*B + r)*n + bn*B;
                for (int c = 0; c < B; c++) {
                    out[c] = (float) (row[c]*(double) ggml_fp16_to_fp32(svh[bn*B + c])*inv_b);
                }
            }
        }
    }
    free(blk);
}

bool ggml_exl3_check_linear(
        const struct ggml_tensor * trellis, const struct ggml_tensor * suh, const struct ggml_tensor * svh,
        enum ggml_exl3_codebook cb) {
    if (trellis->type != GGML_TYPE_I16 || suh->type != GGML_TYPE_F16 || svh->type != GGML_TYPE_F16) {
        return false;
    }
    if (!ggml_exl3_rate_supported(trellis->ne[0], cb) || trellis->ne[3] != 1) {
        return false;
    }
    const int64_t k = trellis->ne[2]*GGML_EXL3_TILE;
    const int64_t n = trellis->ne[1]*GGML_EXL3_TILE;
    if (k % GGML_EXL3_HAD_SIZE != 0 || n % GGML_EXL3_HAD_SIZE != 0) {
        return false;
    }
    return ggml_n_dims(suh) == 1 && suh->ne[0] == k && ggml_n_dims(svh) == 1 && svh->ne[0] == n;
}

int64_t ggml_exl3_ring_words(int64_t dim, int K) {
    return 1 + dim*K/16;
}

int ggml_exl3_embd_k(int64_t row_words, int64_t groups) {
    if (groups <= 0 || row_words % groups != 0) {
        return -1;
    }
    // exllamav3 Embedding: K = (row_words / groups - 1) / 16
    const int64_t w = row_words/groups - 1;
    if (w <= 0 || w % 16 != 0 || w/16 > 8) {
        return -1;
    }
    return (int) (w/16);
}

void ggml_exl3_decode_ring(const int16_t * ring, int64_t dim, int K, float * dst) {
    GGML_ASSERT(K >= 1 && K <= 8 && dim > 0 && dim <= GGML_EXL3_GROUP_DIM && (dim*K) % 16 == 0);

    // as unpack_rows of ngram_codec.py: the symbol of position i through a two-word window, wrapping to word 0
    const int64_t     words  = dim*K/16;
    const uint16_t  * stream = (const uint16_t *) (ring + 1);
    const float       scale  = exl3_half_to_float((uint16_t) ring[0]);
    uint32_t sym[GGML_EXL3_GROUP_DIM];
    for (int64_t i = 0; i < dim; i++) {
        const int64_t  bit    = i*K;
        const uint32_t window = (uint32_t) stream[bit >> 4] | ((uint32_t) stream[((bit >> 4) + 1) % words] << 16);
        sym[i] = (window >> (bit & 15)) & ((1u << K) - 1);
    }
    // the state of position i stacks the symbols of positions i, i - 1, ... from bit 0 up
    const int nj = (15 + K)/K;
    for (int64_t i = 0; i < dim; i++) {
        uint32_t s = 0;
        for (int j = 0; j < nj; j++) {
            s |= sym[(i - j + dim) % dim] << (j*K);
        }
        // fp16 codebook value times fp16 scale: exact in f32
        dst[i] = ggml_fp16_to_fp32(ggml_exl3_codebook_value((uint16_t) s, GGML_EXL3_CODEBOOK_MUL1))*scale;
    }
}

// orthonormal Walsh-Hadamard transform of 256 values in f32, the butterfly order of ngram_codec.hadamard_rotate
static void exl3_hadamard_rotate_256(float * x) {
    for (int h = 1; h < GGML_EXL3_GROUP_DIM; h <<= 1) {
        for (int i = 0; i < GGML_EXL3_GROUP_DIM; i += 2*h) {
            for (int j = i; j < i + h; j++) {
                const float a = x[j];
                const float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < GGML_EXL3_GROUP_DIM; i++) {
        x[i] *= 0.0625f; // 256 ** -0.5
    }
}

void ggml_exl3_decode_embd_row(
        const int16_t * row, int64_t groups, int K, const ggml_fp16_t * signs, int64_t n_out, float * dst) {
    GGML_ASSERT(n_out <= groups*GGML_EXL3_GROUP_DIM);
    const int64_t rw = ggml_exl3_ring_words(GGML_EXL3_GROUP_DIM, K);
    float x[GGML_EXL3_GROUP_DIM];
    for (int64_t g = 0; g*GGML_EXL3_GROUP_DIM < n_out; g++) {
        ggml_exl3_decode_ring(row + g*rw, GGML_EXL3_GROUP_DIM, K, x);
        exl3_hadamard_rotate_256(x);
        const int64_t n = n_out - g*GGML_EXL3_GROUP_DIM < GGML_EXL3_GROUP_DIM ? n_out - g*GGML_EXL3_GROUP_DIM : GGML_EXL3_GROUP_DIM;
        for (int64_t i = 0; i < n; i++) {
            dst[g*GGML_EXL3_GROUP_DIM + i] = x[i]*ggml_fp16_to_fp32(signs[g*GGML_EXL3_GROUP_DIM + i]);
        }
    }
}

void ggml_exl3_decode_ngram_row(const int16_t * row, int K, const ggml_fp16_t * bias, float * dst) {
    ggml_exl3_decode_ring(row, GGML_EXL3_ROW_DIM, K, dst);
    for (int i = 0; i < GGML_EXL3_ROW_DIM; i++) {
        dst[i] = dst[i] + ggml_fp16_to_fp32(bias[i]);
    }
}

int64_t ggml_exl3_ngram_head(const int64_t * head_offsets, int64_t n_heads, int64_t idx) {
    // searchsorted(head_offsets, idx, right = True) - 1, clamped as NGramEmbedding.fetch_rows does
    int64_t h = 0;
    while (h < n_heads && head_offsets[h] <= idx) {
        h++;
    }
    h -= 1;
    return h < 0 ? 0 : h >= n_heads ? n_heads - 1 : h;
}
