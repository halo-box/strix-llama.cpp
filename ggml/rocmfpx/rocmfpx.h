#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QK_ROCMFPX 32

#define QK_ROCMFP2 QK_ROCMFPX
#define QK_ROCMFP3 QK_ROCMFPX
#define QK_ROCMFP6 QK_ROCMFPX
#define QK_ROCMFP8 QK_ROCMFPX

#define QS_ROCMFP2 ((QK_ROCMFP2 * 2) / 8)
#define QR_ROCMFP2 1
#define QI_ROCMFP2 (QK_ROCMFP2 / (4 * QR_ROCMFP2))

// Q7-DOT32 is deliberately a 256-value macroblock rather than a conventional
// 30-byte block. Eight independently-scaled 32-value groups are laid out as
// 8 x 28 packed-code bytes followed by 8 x FP16 scales. This keeps every
// packed group 4-byte aligned and every macroblock 16-byte aligned in a GGUF
// tensor while retaining the 7.5 bpw wire size.
#define QK_ROCMFP7       256
#define QG_ROCMFP7       32
#define NG_ROCMFP7       (QK_ROCMFP7 / QG_ROCMFP7)
#define QS_ROCMFP7_GROUP ((QG_ROCMFP7 * 7) / 8)
#define QS_ROCMFP7       (NG_ROCMFP7 * QS_ROCMFP7_GROUP)

#define QS_ROCMFP3 ((QK_ROCMFP3 * 3) / 8)
#define QS_ROCMFP6 ((QK_ROCMFP6 * 6) / 8)
#define QS_ROCMFP8 QK_ROCMFP8

#define QR_ROCMFP3 1
#define QI_ROCMFP3 (QK_ROCMFP3 / (4 * QR_ROCMFP3))

#define QR_ROCMFP6 1
#define QI_ROCMFP6 (QK_ROCMFP6 / (4 * QR_ROCMFP6))

#define QR_ROCMFP8 1
#define QI_ROCMFP8 (QK_ROCMFP8 / (4 * QR_ROCMFP8))

#define QR_ROCMFP7 1
#define QI_ROCMFP7 (QK_ROCMFP7 / (4 * QR_ROCMFP7))

// AMD-native experimental family layouts. The GGUF types are registered, but
// the layouts stay isolated from the promoted ROCmFP4 formats while evaluated.
typedef struct {
    uint8_t qs[QS_ROCMFP2];
    uint8_t e[2];
} block_rocmfp2;

typedef struct {
    uint8_t qs[QS_ROCMFP3];
    uint8_t e[2];
} block_rocmfp3;

typedef struct {
    uint8_t qs[QS_ROCMFP6];
    uint8_t e[2];
} block_rocmfp6;

typedef struct {
    int8_t  qs[QS_ROCMFP8];
    uint8_t e;
} block_rocmfp8;

typedef struct {
    uint8_t     qs[QS_ROCMFP7];
    ggml_fp16_t d[NG_ROCMFP7];
} block_rocmfp7;

#if defined(__cplusplus)
static_assert(sizeof(block_rocmfp2) == QS_ROCMFP2 + 2*sizeof(uint8_t), "wrong rocmfp2 block size/padding");
static_assert(sizeof(block_rocmfp3) == QS_ROCMFP3 + 2*sizeof(uint8_t), "wrong rocmfp3 block size/padding");
static_assert(sizeof(block_rocmfp6) == QS_ROCMFP6 + 2*sizeof(uint8_t), "wrong rocmfp6 block size/padding");
static_assert(sizeof(block_rocmfp8) == QS_ROCMFP8 + sizeof(uint8_t), "wrong rocmfp8 block size/padding");
static_assert(sizeof(block_rocmfp7) == 240, "wrong rocmfp7 macroblock size/padding");
#else
_Static_assert(sizeof(block_rocmfp2) == QS_ROCMFP2 + 2*sizeof(uint8_t), "wrong rocmfp2 block size/padding");
_Static_assert(sizeof(block_rocmfp3) == QS_ROCMFP3 + 2*sizeof(uint8_t), "wrong rocmfp3 block size/padding");
_Static_assert(sizeof(block_rocmfp6) == QS_ROCMFP6 + 2*sizeof(uint8_t), "wrong rocmfp6 block size/padding");
_Static_assert(sizeof(block_rocmfp8) == QS_ROCMFP8 + sizeof(uint8_t), "wrong rocmfp8 block size/padding");
_Static_assert(sizeof(block_rocmfp7) == 240, "wrong rocmfp7 macroblock size/padding");
#endif

GGML_API float  rocmfpx_ue4m3_to_fp32(uint8_t e);
GGML_API bool   rocmfpx_scale_is_valid(uint8_t e);
GGML_API size_t rocmfpx_row_size_fp2(int64_t k);
GGML_API size_t rocmfpx_row_size_fp3(int64_t k);
GGML_API size_t rocmfpx_row_size_fp6(int64_t k);
GGML_API size_t rocmfpx_row_size_fp8(int64_t k);
GGML_API size_t rocmfpx_row_size_fp7(int64_t k);

GGML_API void   rocmfpx_quantize_row_fp2_ref(const float * GGML_RESTRICT x, block_rocmfp2 * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_fp2(const block_rocmfp2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_fp2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_fp2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void   rocmfpx_quantize_row_fp3_ref(const float * GGML_RESTRICT x, block_rocmfp3 * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_fp3(const block_rocmfp3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_fp3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_fp3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void   rocmfpx_quantize_row_fp6_ref(const float * GGML_RESTRICT x, block_rocmfp6 * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_fp6(const block_rocmfp6 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_fp6(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_fp6(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void   rocmfpx_quantize_row_fp8_ref(const float * GGML_RESTRICT x, block_rocmfp8 * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_fp8(const block_rocmfp8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_fp8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_fp8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void   rocmfpx_quantize_row_fp7_ref(const float * GGML_RESTRICT x, block_rocmfp7 * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_fp7(const block_rocmfp7 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_fp7(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_fp7(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API bool rocmfpx_validate_row_data_fp2(const void * data, size_t nbytes);
GGML_API bool rocmfpx_validate_row_data_fp3(const void * data, size_t nbytes);
GGML_API bool rocmfpx_validate_row_data_fp6(const void * data, size_t nbytes);
GGML_API bool rocmfpx_validate_row_data_fp8(const void * data, size_t nbytes);
GGML_API bool rocmfpx_validate_row_data_fp7(const void * data, size_t nbytes);

#ifdef __cplusplus
}
#endif
