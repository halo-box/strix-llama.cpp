#pragma once

// Reference codecs of the fork types: the fork's additions to ggml/src/ggml-quants.h.

#define GGML_COMMON_DECL_C
#include "ggml-common-ext.h"
#include "ggml-quants.h"

#ifdef __cplusplus
extern "C" {
#endif
// from ggml/src/ggml-quants.h after upstream line 7 (upstream 9048bad8d, fork 1fe05fa45)
// ROCmFPx block layouts, for the dequantize_row_* wrappers below
#include "rocmfp4/rocmfp4.h"
#include "rocmfpx/rocmfpx.h"


// from ggml/src/ggml-quants.h after upstream line 37 (upstream 9048bad8d, fork 1fe05fa45)
GGML_API void quantize_row_ptq1_0_ref(const float * GGML_RESTRICT x, block_ptq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tq2_t_ref(const float * GGML_RESTRICT x, block_tq2_t * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tqk6_ref(const float * GGML_RESTRICT x, block_tqk6 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tqk7_ref(const float * GGML_RESTRICT x, block_tqk7 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tqp5_ref(const float * GGML_RESTRICT x, block_tqp5 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tqp6_ref(const float * GGML_RESTRICT x, block_tqp6 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tqp7_ref(const float * GGML_RESTRICT x, block_tqp7 * GGML_RESTRICT y, int64_t k);

// from ggml/src/ggml-quants.h after upstream line 59 (upstream 9048bad8d, fork 1fe05fa45)
// ROCmFPx types under the ggml naming convention (dequantize_row_<type_name>),
// so tooling that resolves dequantizers by type name (gguf-py/tests/test_quants.py)
// reaches them. The codecs themselves live in ggml/rocmfp4 and ggml/rocmfpx.
GGML_API void dequantize_row_q4_0_rocmfp4     (const block_rocmfp4      * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_0_rocmfp4_fast(const block_rocmfp4_fast * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q2_0_rocmfpx     (const block_rocmfp2      * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q3_0_rocmfpx     (const block_rocmfp3      * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q6_0_rocmfpx     (const block_rocmfp6      * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_0_rocmfpx     (const block_rocmfp8      * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);


// from ggml/src/ggml-quants.h after upstream line 67 (upstream 9048bad8d, fork 1fe05fa45)
GGML_API void dequantize_row_ptq1_0(const block_ptq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tq2_t(const block_tq2_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tqk6(const block_tqk6 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tqk7(const block_tqk7 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tqp5(const block_tqp5 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tqp6(const block_tqp6 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tqp7(const block_tqp7 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// from ggml/src/ggml-quants.h after upstream line 91 (upstream 9048bad8d, fork 1fe05fa45)
GGML_API size_t quantize_ptq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tq2_t(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tqk6(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tqk7(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tqp5(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tqp6(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tqp7(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

#ifdef __cplusplus
}
#endif
