// EXL3 trellis decode, CPU reference.
//
// Format: turboderp-org/exllamav3 (MIT), read at a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a.
// Checked against its independent decoder tests/testlib/trellis.py. See README.md.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// codebook ids follow exllamav3 (cb 0, 1, 2)
enum ggml_exl3_codebook {
    GGML_EXL3_CODEBOOK_3INST = 0,
    GGML_EXL3_CODEBOOK_MCG   = 1,
    GGML_EXL3_CODEBOOK_MUL1  = 2,
};

#define GGML_EXL3_TILE     16
#define GGML_EXL3_HAD_SIZE 128

// value of the .mcg / .mul1 marker scalar, read as uint32
#define GGML_EXL3_MCG_MULT  0xCBAC1FEDu
#define GGML_EXL3_MUL1_MULT 0x83DCD12Du

// "3inst", "mcg" or "mul1"; returns -1 for other names
GGML_API int          ggml_exl3_codebook_from_name(const char * name);
GGML_API const char * ggml_exl3_codebook_name(enum ggml_exl3_codebook cb);

// tile_words is the number of int16 words of one 16x16 tile (16*K, the last trellis dim)
// legal: K = 1..8 for every codebook, K = 1.5, 2.5, 3.5 with mul1 only
GGML_API bool ggml_exl3_rate_supported(int64_t tile_words, enum ggml_exl3_codebook cb);

// fp16 value of one 16-bit trellis state
GGML_API ggml_fp16_t ggml_exl3_codebook_value(uint16_t state, enum ggml_exl3_codebook cb);

// one tile: tile_words int16 -> 256 fp16, row-major 16x16
GGML_API void ggml_exl3_decode_tile(const int16_t * tile, int64_t tile_words, enum ggml_exl3_codebook cb, ggml_fp16_t * dst);

// trellis [k/16][n/16][tile_words] -> rotated weight W_hat, k x n fp16 row-major (no Hadamard, no scales)
GGML_API void ggml_exl3_decode_rotated(
        const int16_t * trellis, int64_t k, int64_t n, int64_t tile_words, enum ggml_exl3_codebook cb, ggml_fp16_t * dst);

// W = diag(suh) * H * W_hat * H * diag(svh), k x n float row-major (y = x W)
// H is the normalized 128-point Sylvester Hadamard, applied per 128 block on both sides
GGML_API void ggml_exl3_dequantize(
        const int16_t * trellis, const ggml_fp16_t * suh, const ggml_fp16_t * svh,
        int64_t k, int64_t n, int64_t tile_words, enum ggml_exl3_codebook cb, float * dst);

// checks the types and shapes of one EXL3 linear:
// trellis I16 [tile_words, n/16, k/16], suh F16 [k], svh F16 [n], k and n multiples of 128
GGML_API bool ggml_exl3_check_linear(
        const struct ggml_tensor * trellis, const struct ggml_tensor * suh, const struct ggml_tensor * svh,
        enum ggml_exl3_codebook cb);

// Row rings (exllamav3 ngram_codec.py): token embedding tables and hashed n-gram (PLE) tables. A ring of dim positions
// is 1 + dim*K/16 int16 words: word 0 holds the fp16 scale, then stream bits [i*K, (i+1)*K) (LSB first in little-endian
// uint16 words) are the low K bits of the 16-bit state of position i, whose higher bits are the symbols of positions
// i-1, i-2, ... (mod dim). Always the mul1 codebook, K = 1..8.
#define GGML_EXL3_ROW_DIM   160 // n-gram row: one ring, plus the bias of its hash head
#define GGML_EXL3_GROUP_DIM 256 // token embedding row: rings of 256, each rotated by H256 and signs

// int16 words of one ring
GGML_API int64_t ggml_exl3_ring_words(int64_t dim, int K);

// K of a token embedding row of row_words int16 for `groups` rings of 256, or -1 if the width does not fit
GGML_API int ggml_exl3_embd_k(int64_t row_words, int64_t groups);

// one ring: dst[i] = mul1(state_i) * scale, an exact f32 product
GGML_API void ggml_exl3_decode_ring(const int16_t * ring, int64_t dim, int K, float * dst);

// one token embedding row: group g = signs[g] * H256(ring g) for the rings back to back, H256 the orthonormal
// Walsh-Hadamard transform in f32, in the butterfly order of ngram_codec.hadamard_rotate; the first n_out values
// signs: F16 [groups][256]
GGML_API void ggml_exl3_decode_embd_row(
        const int16_t * row, int64_t groups, int K, const ggml_fp16_t * signs, int64_t n_out, float * dst);

// one n-gram row of 160: the ring plus its head's bias (F16 [160]), one f32 rounding
GGML_API void ggml_exl3_decode_ngram_row(const int16_t * row, int K, const ggml_fp16_t * bias, float * dst);

// head of n-gram row idx: the last h with head_offsets[h] <= idx, clamped to [0, n_heads)
GGML_API int64_t ggml_exl3_ngram_head(const int64_t * head_offsets, int64_t n_heads, int64_t idx);

#ifdef __cplusplus
}
#endif
