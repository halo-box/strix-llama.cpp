#pragma once

#include "common.cuh"
#include "q7-view-mode.cuh"
#include "../../rocmfpx/q7-q8-view.h"

#include <cstddef>
#include <cstdint>

static_assert(
    QK_ROCMFP7 == NG_ROCMFP7 * QK8_0,
    "Q7 macro must map to an integral number of Q8_0 blocks");
static_assert(
    ROCMFPX_Q7_Q8_VIEW_MACRO_BYTES ==
        NG_ROCMFP7 * sizeof(block_q8_0),
    "Q7 Q8-view macro must contain eight ordinary Q8_0 blocks");

static inline bool ggml_cuda_q7_q8_view_enabled() {
    const ggml_cuda_q7_view_mode mode =
        ggml_cuda_q7_view_mode_get();
    return
        mode == GGML_CUDA_Q7_VIEW_Q8 ||
        mode == GGML_CUDA_Q7_VIEW_Q8_NO_OUTPUT;
}

static inline bool ggml_cuda_q7_q8_view_includes_output() {
    return ggml_cuda_q7_view_mode_get() == GGML_CUDA_Q7_VIEW_Q8;
}

static inline bool ggml_cuda_q7_q8_view_nbytes_checked(
        const int64_t nrows,
        const int64_t ncols,
        size_t * view_size) {
    if (ncols <= 0 || ncols % QK_ROCMFP7 != 0) {
        return false;
    }
    return rocmfpx_q7_q8_view_size(
        nrows,
        ncols / QK_ROCMFP7,
        view_size);
}

static inline size_t ggml_cuda_q7_q8_view_nbytes(
        const int64_t nrows,
        const int64_t ncols) {
    size_t view_size = 0;
    return ggml_cuda_q7_q8_view_nbytes_checked(
        nrows,
        ncols,
        &view_size) ?
        view_size :
        0;
}

static inline bool ggml_cuda_q7_q8_view_eligible(
        const int cc,
        const ggml_type type,
        const int64_t nrows,
        const int64_t ncols,
        const int64_t stride_row_blocks) {
    size_t view_size = 0;
    return
        ggml_cuda_q7_q8_view_enabled() &&
        cc == GGML_CUDA_CC_OFFSET_AMD + 0x1151 &&
        type == GGML_TYPE_Q7_0_ROCMFPX &&
        // The runtime caller additionally requires a normal contiguous 2D
        // model weight and excludes token embedding lookup storage.
        nrows > 0 &&
        ncols > 0 &&
        ncols % QK_ROCMFP7 == 0 &&
        stride_row_blocks == ncols / QK_ROCMFP7 &&
        ggml_cuda_q7_q8_view_nbytes_checked(
            nrows,
            ncols,
            &view_size) &&
        view_size > 0 &&
        rocmfpx_q7_q8_view_mmq_addressable(
            nrows,
            ncols / QK_ROCMFP7);
}
