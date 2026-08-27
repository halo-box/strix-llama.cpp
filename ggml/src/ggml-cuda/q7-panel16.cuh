#pragma once

#include "common.cuh"
#include "q7-view-mode.cuh"

#include <cstddef>
#include <cstdint>

// gfx1151-only device-cache layout for canonical ROCmFPX Q7 weights.
// The GGUF and ordinary tensor allocation remain canonical block_rocmfp7.
// A cache contains one panel for each [16 output rows, 256 K values] tile:
//
//   q[group8][physical_quartet8][16 rows * 28 bits == 56 bytes]
//   d[group8][row16] as raw FP16 bits
//
// Physical quartet order pairs the logical quartets consumed by the mirrored
// wave32 halves: 0,4,1,5,2,6,3,7.
static constexpr int GGML_CUDA_Q7_PANEL16_ROWS = 16;
static constexpr int GGML_CUDA_Q7_PANEL16_GROUPS = 8;
static constexpr int GGML_CUDA_Q7_PANEL16_QUARTETS = 8;
static constexpr int GGML_CUDA_Q7_PANEL16_QUARTET_BITS = 28;
static constexpr int GGML_CUDA_Q7_PANEL16_PLANE_BYTES =
    GGML_CUDA_Q7_PANEL16_ROWS * GGML_CUDA_Q7_PANEL16_QUARTET_BITS / 8;

struct alignas(128) block_rocmfp7_panel16 {
    uint8_t q
        [GGML_CUDA_Q7_PANEL16_GROUPS]
        [GGML_CUDA_Q7_PANEL16_QUARTETS]
        [GGML_CUDA_Q7_PANEL16_PLANE_BYTES];
    uint16_t d
        [GGML_CUDA_Q7_PANEL16_GROUPS]
        [GGML_CUDA_Q7_PANEL16_ROWS];
};

static_assert(GGML_CUDA_Q7_PANEL16_GROUPS == NG_ROCMFP7,
              "Q7 panel group count must match canonical Q7");
static_assert(sizeof(block_rocmfp7_panel16) ==
                  GGML_CUDA_Q7_PANEL16_ROWS * sizeof(block_rocmfp7),
              "Q7 panel must preserve the canonical 7.5 bits/weight");
static_assert(sizeof(block_rocmfp7_panel16) == 3840,
              "Q7 panel16 must occupy exactly 3840 bytes");
static_assert(offsetof(block_rocmfp7_panel16, d) == 3584,
              "Q7 panel scale offset must follow the packed planes");

static constexpr __host__ __device__ int
ggml_cuda_q7_panel16_quartet_slot(const int logical_quartet) {
    return 2 * (logical_quartet & 3) + (logical_quartet >> 2);
}

static constexpr size_t ggml_cuda_q7_panel16_nbytes(
        const int64_t nrows,
        const int64_t ncols) {
    return
        static_cast<size_t>(nrows / GGML_CUDA_Q7_PANEL16_ROWS) *
        static_cast<size_t>(ncols / QK_ROCMFP7) *
        sizeof(block_rocmfp7_panel16);
}

// The cache path is opt-in only. There is deliberately no "auto" mode while
// allocation/repack and end-to-end model gates are under development.
static inline bool ggml_cuda_q7_panel16_cache_enabled() {
    return ggml_cuda_q7_view_mode_get() ==
        GGML_CUDA_Q7_VIEW_PANEL16;
}

// Host-visible allocation/repack eligibility shared with the CUDA buffer
// lifecycle. Runtime launch code adds its narrower plain-MUL_MAT constraints.
static inline bool ggml_cuda_q7_panel16_cache_eligible(
        const int cc,
        const ggml_type type,
        const int64_t nrows,
        const int64_t ncols,
        const int64_t stride_row_blocks) {
    return
        ggml_cuda_q7_panel16_cache_enabled() &&
        cc == GGML_CUDA_CC_OFFSET_AMD + 0x1151 &&
        type == GGML_TYPE_Q7_0_ROCMFPX &&
        nrows > 0 &&
        nrows % GGML_CUDA_Q7_PANEL16_ROWS == 0 &&
        ncols > 0 &&
        ncols % QK_ROCMFP7 == 0 &&
        stride_row_blocks == ncols / QK_ROCMFP7;
}
