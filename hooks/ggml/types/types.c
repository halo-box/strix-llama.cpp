// The fork types in ggml-base: codec wrappers of the type traits (HOOK(type-traits)), quantize (HOOK(type-quantize))
// and validate (HOOK(type-validate)). The codecs are in ggml-quants-ext.c, rocmfp4/ and rocmfpx/.

#define GGML_COMMON_DECL_C
#include "ggml-quants-ext.h"
#include "ggml-impl.h"

#include <stdio.h>


static_assert(sizeof(block_rocmfp4)      == 18 && QK_ROCMFP4 == 32,  "q4_0_rocmfp4 size in GGML_HOOK_TYPES_LIST");
static_assert(sizeof(block_rocmfp4_fast) == 17,                      "q4_0_rocmfp4_fast size in GGML_HOOK_TYPES_LIST");
static_assert(sizeof(block_rocmfp6)      == 26 && QK_ROCMFP6 == 32,  "q6_0_rocmfpx size in GGML_HOOK_TYPES_LIST");
static_assert(sizeof(block_rocmfp8)      == 33 && QK_ROCMFP8 == 32,  "q8_0_rocmfpx size in GGML_HOOK_TYPES_LIST");
static_assert(sizeof(block_rocmfp3)      == 14 && QK_ROCMFP3 == 32,  "q3_0_rocmfpx size in GGML_HOOK_TYPES_LIST");
static_assert(sizeof(block_rocmfp2)      == 10 && QK_ROCMFP2 == 32,  "q2_0_rocmfpx size in GGML_HOOK_TYPES_LIST");
static_assert(sizeof(block_ptq1_0)       == 28 && QK_PTQ1_0 == 128,  "ptq1_0 size in GGML_HOOK_TYPES_LIST");
static_assert(sizeof(block_tq2_t)        == 34 && QK_TQ2_T == 128,   "tq2_t size in GGML_HOOK_TYPES_LIST");
static_assert(sizeof(block_tqk6)         == 26 && QK_TQK == 128,     "tqk6 size in GGML_HOOK_TYPES_LIST");
static_assert(sizeof(block_tqk7)         == 30,                      "tqk7 size in GGML_HOOK_TYPES_LIST");
static_assert(sizeof(block_tqp5)         == 42 && QK_TQP == 128,     "tqp5 size in GGML_HOOK_TYPES_LIST");
static_assert(sizeof(block_tqp6)         == 50,                      "tqp6 size in GGML_HOOK_TYPES_LIST");
static_assert(sizeof(block_tqp7)         == 58,                      "tqp7 size in GGML_HOOK_TYPES_LIST");
static_assert(sizeof(block_q8_0)         == 34 && QK8_0 == 32,       "q8_0_hook_example size in GGML_HOOK_TYPES_LIST");

#define GGML_HOOK_CODECS(TNAME, TO_FLOAT, FROM_REF)                                                                   \
    void ggml_hook_type_##TNAME##_to_float(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {       \
        TO_FLOAT(x, y, k);                                                                                            \
    }                                                                                                                 \
    void ggml_hook_type_##TNAME##_from_float_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) { \
        FROM_REF(x, y, k);                                                                                            \
    }

GGML_HOOK_CODECS(q4_0_rocmfp4,      rocmfp4_dequantize_row_q4_0,      rocmfp4_quantize_row_q4_0_ref)
GGML_HOOK_CODECS(q4_0_rocmfp4_fast, rocmfp4_dequantize_row_q4_0_fast, rocmfp4_quantize_row_q4_0_fast_ref)
GGML_HOOK_CODECS(q6_0_rocmfpx,      rocmfpx_dequantize_row_fp6,       rocmfpx_quantize_row_fp6_ref)
GGML_HOOK_CODECS(q8_0_rocmfpx,      rocmfpx_dequantize_row_fp8,       rocmfpx_quantize_row_fp8_ref)
GGML_HOOK_CODECS(q3_0_rocmfpx,      rocmfpx_dequantize_row_fp3,       rocmfpx_quantize_row_fp3_ref)
GGML_HOOK_CODECS(q2_0_rocmfpx,      rocmfpx_dequantize_row_fp2,       rocmfpx_quantize_row_fp2_ref)
GGML_HOOK_CODECS(ptq1_0,            dequantize_row_ptq1_0,            quantize_row_ptq1_0_ref)
GGML_HOOK_CODECS(tq2_t,             dequantize_row_tq2_t,             quantize_row_tq2_t_ref)
GGML_HOOK_CODECS(tqk6,              dequantize_row_tqk6,              quantize_row_tqk6_ref)
GGML_HOOK_CODECS(tqk7,              dequantize_row_tqk7,              quantize_row_tqk7_ref)
GGML_HOOK_CODECS(tqp5,              dequantize_row_tqp5,              quantize_row_tqp5_ref)
GGML_HOOK_CODECS(tqp6,              dequantize_row_tqp6,              quantize_row_tqp6_ref)
GGML_HOOK_CODECS(tqp7,              dequantize_row_tqp7,              quantize_row_tqp7_ref)
GGML_HOOK_CODECS(q8_0_hook_example, dequantize_row_q8_0,              quantize_row_q8_0_ref)

// the fork's cases of ggml_quantize_chunk (ggml/src/ggml.c, upstream 9048bad8d, fork 1fe05fa45). src and dst are at the first row.
bool ggml_hook_type_quantize(enum ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix, size_t * result) {
    switch ((int) type) {
        case GGML_TYPE_Q4_0_ROCMFP4:      *result = rocmfp4_quantize_q4_0     (src, dst, nrows, n_per_row, imatrix); return true;
        case GGML_TYPE_Q4_0_ROCMFP4_FAST: *result = rocmfp4_quantize_q4_0_fast(src, dst, nrows, n_per_row, imatrix); return true;
        case GGML_TYPE_Q3_0_ROCMFPX:      *result = rocmfpx_quantize_fp3      (src, dst, nrows, n_per_row, imatrix); return true;
        case GGML_TYPE_Q2_0_ROCMFPX:      *result = rocmfpx_quantize_fp2      (src, dst, nrows, n_per_row, imatrix); return true;
        case GGML_TYPE_Q6_0_ROCMFPX:      *result = rocmfpx_quantize_fp6      (src, dst, nrows, n_per_row, imatrix); return true;
        case GGML_TYPE_Q8_0_ROCMFPX:      *result = rocmfpx_quantize_fp8      (src, dst, nrows, n_per_row, imatrix); return true;
        case GGML_TYPE_PTQ1_0:            *result = quantize_ptq1_0           (src, dst, nrows, n_per_row, imatrix); return true;
        case GGML_TYPE_TQ2_T:             *result = quantize_tq2_t            (src, dst, nrows, n_per_row, imatrix); return true;
        case GGML_TYPE_TQK6:              *result = quantize_tqk6             (src, dst, nrows, n_per_row, imatrix); return true;
        case GGML_TYPE_TQK7:              *result = quantize_tqk7             (src, dst, nrows, n_per_row, imatrix); return true;
        case GGML_TYPE_TQP5:              *result = quantize_tqp5             (src, dst, nrows, n_per_row, imatrix); return true;
        case GGML_TYPE_TQP6:              *result = quantize_tqp6             (src, dst, nrows, n_per_row, imatrix); return true;
        case GGML_TYPE_TQP7:              *result = quantize_tqp7             (src, dst, nrows, n_per_row, imatrix); return true;
        case GGML_TYPE_Q8_0_HOOK_EXAMPLE: *result = quantize_q8_0             (src, dst, nrows, n_per_row, imatrix); return true;
        default:                          return false;
    }
}

// copied from ggml/src/ggml-quants.c (upstream 9048bad8d): static there
static bool isinf_fp16(ggml_fp16_t f) {
    return (f & 0x7c00) == 0x7c00 && (f & 0x03ff) == 0;
}

static bool isnan_fp16(ggml_fp16_t f) {
    return (f & 0x7c00) == 0x7c00 && (f & 0x03ff) != 0;
}

static bool validate_fp16(ggml_fp16_t f, size_t i) {
    if (isinf_fp16(f)) {
        fprintf(stderr, "ggml_validate_row_data: found inf value at block %zu\n", i);
        return false;
    }

    if (isnan_fp16(f)) {
        fprintf(stderr, "ggml_validate_row_data: found nan value at block %zu\n", i);
        return false;
    }

    return true;
}

#define VALIDATE_ROW_DATA_D_F16_IMPL(type, data, nb) \
    const type * q = (const type *) (data); \
    for (size_t i = 0; i < (nb); ++i) { \
        if (!validate_fp16(q[i].d, i)) { \
            return false; \
        } \
    }

static bool validate(enum ggml_type type, const void * data, size_t nbytes) {
    const size_t nb = nbytes / ggml_type_size(type);
    switch ((int) type) {
        // the fork's cases of ggml_validate_row_data (ggml/src/ggml-quants.c, upstream 9048bad8d, fork 1fe05fa45)
        case GGML_TYPE_Q4_0_ROCMFP4:
            {
                if (!rocmfp4_validate_row_data(data, nbytes)) {
                    fprintf(stderr, "%s: invalid ROCmFP4 row data\n", __func__);
                    return false;
                }
            } break;
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
            {
                if (!rocmfp4_validate_row_data_fast(data, nbytes)) {
                    fprintf(stderr, "%s: invalid ROCmFP4 fast row data\n", __func__);
                    return false;
                }
            } break;
        case GGML_TYPE_Q3_0_ROCMFPX:
            {
                if (!rocmfpx_validate_row_data_fp3(data, nbytes)) {
                    fprintf(stderr, "%s: invalid ROCmFPx FP3 row data\n", __func__);
                    return false;
                }
            } break;
        case GGML_TYPE_Q2_0_ROCMFPX:
            {
                if (!rocmfpx_validate_row_data_fp2(data, nbytes)) {
                    fprintf(stderr, "%s: invalid ROCmFPx FP2 row data\n", __func__);
                    return false;
                }
            } break;
        case GGML_TYPE_Q6_0_ROCMFPX:
            {
                if (!rocmfpx_validate_row_data_fp6(data, nbytes)) {
                    fprintf(stderr, "%s: invalid ROCmFPx FP6 row data\n", __func__);
                    return false;
                }
            } break;
        case GGML_TYPE_Q8_0_ROCMFPX:
            {
                if (!rocmfpx_validate_row_data_fp8(data, nbytes)) {
                    fprintf(stderr, "%s: invalid ROCmFPx FP8 row data\n", __func__);
                    return false;
                }
            } break;
        case GGML_TYPE_PTQ1_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_ptq1_0, data, nb);
            } break;
        case GGML_TYPE_TQ2_T:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tq2_t, data, nb);
            } break;
        case GGML_TYPE_TQK6:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tqk6, data, nb);
            } break;
        case GGML_TYPE_TQK7:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tqk7, data, nb);
            } break;
        case GGML_TYPE_TQP5:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tqp5, data, nb);
            } break;
        case GGML_TYPE_TQP6:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tqp6, data, nb);
            } break;
        case GGML_TYPE_TQP7:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tqp7, data, nb);
            } break;
        case GGML_TYPE_Q8_0_HOOK_EXAMPLE:
            return ggml_validate_row_data(GGML_TYPE_Q8_0, data, nbytes);
        default:
            break;
    }
    return true;
}

bool ggml_hook_type_validate(enum ggml_type type, const void * data, size_t nbytes, bool * ok) {
    if (!ggml_hook_type_is_hook(type)) {
        return false;
    }
    *ok = validate(type, data, nbytes);
    return true;
}

// the fork cases of ggml_ftype_to_ggml_type (ggml/src/ggml.c after upstream line 1442, fork 1fe05fa45). Upstream
// asserts on an ftype it does not know: the wrapper answers for the fork ftypes and calls upstream for the others.
enum ggml_type __real_ggml_ftype_to_ggml_type(enum ggml_ftype ftype);
GGML_API enum ggml_type __wrap_ggml_ftype_to_ggml_type(enum ggml_ftype ftype);
enum ggml_type __wrap_ggml_ftype_to_ggml_type(enum ggml_ftype ftype) {
    enum ggml_type wtype = GGML_TYPE_COUNT;
    switch ((int) ftype) {
        case GGML_FTYPE_MOSTLY_Q4_0_ROCMFP4:
        case GGML_FTYPE_MOSTLY_Q4_0_ROCMFP4_LEAN:
        case GGML_FTYPE_MOSTLY_Q4_0_ROCMFP4_COHERENT:
            wtype = GGML_TYPE_Q4_0_ROCMFP4; break;
        case GGML_FTYPE_MOSTLY_Q4_0_ROCMFP4_FAST:
        case GGML_FTYPE_MOSTLY_Q4_0_ROCMFP4_FAST_COHERENT:
        case GGML_FTYPE_MOSTLY_Q4_0_ROCMFP4_STRIX:
        case GGML_FTYPE_MOSTLY_Q4_0_ROCMFP4_STRIX_LEAN:
            wtype = GGML_TYPE_Q4_0_ROCMFP4_FAST; break;
        case GGML_FTYPE_MOSTLY_Q3_0_ROCMFPX:  wtype = GGML_TYPE_Q3_0_ROCMFPX; break;
        case GGML_FTYPE_MOSTLY_Q2_0_ROCMFPX:  wtype = GGML_TYPE_Q2_0_ROCMFPX; break;
        case GGML_FTYPE_MOSTLY_Q6_0_ROCMFPX:  wtype = GGML_TYPE_Q6_0_ROCMFPX; break;
        case GGML_FTYPE_MOSTLY_Q8_0_ROCMFPX:  wtype = GGML_TYPE_Q8_0_ROCMFPX; break;
        case GGML_FTYPE_MOSTLY_PTQ1_0:        wtype = GGML_TYPE_PTQ1_0; break;
        case GGML_FTYPE_MOSTLY_TQ2_T:         wtype = GGML_TYPE_TQ2_T; break;
        case GGML_FTYPE_MOSTLY_TQK6:         wtype = GGML_TYPE_TQK6; break;
        case GGML_FTYPE_MOSTLY_TQK7:         wtype = GGML_TYPE_TQK7; break;
        case GGML_FTYPE_MOSTLY_TQP5:         wtype = GGML_TYPE_TQP5; break;
        case GGML_FTYPE_MOSTLY_TQP6:         wtype = GGML_TYPE_TQP6; break;
        case GGML_FTYPE_MOSTLY_TQP7:         wtype = GGML_TYPE_TQP7; break;
        default: return __real_ggml_ftype_to_ggml_type(ftype);
    }
    return wtype;
}
