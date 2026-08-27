#pragma once

#include "rocmfpx.h"

#include <limits.h>
#include <string.h>

// Host-only, lossless expansion used by the gfx1151 Q7 -> standard-Q8
// compute view. Each Q7 256-value macro becomes eight ordinary Q8_0 blocks:
//
//   [2 raw FP16 scale bytes][32 sign-extended int8 codes]
//
// This helper intentionally describes the output as bytes so the ROCmFPX
// wire-format layer does not depend on ggml-common.h's block_q8_0 type.
#define ROCMFPX_Q7_Q8_VIEW_BLOCK_BYTES (2 + QG_ROCMFP7)
#define ROCMFPX_Q7_Q8_VIEW_MACRO_BYTES \
    (NG_ROCMFP7 * ROCMFPX_Q7_Q8_VIEW_BLOCK_BYTES)

static inline bool rocmfpx_q7_q8_view_size(
        const int64_t nrows,
        const int64_t macros_per_row,
        size_t * view_size) {
    if (view_size == NULL ||
        nrows <= 0 ||
        macros_per_row <= 0 ||
        nrows > INT64_MAX / macros_per_row) {
        return false;
    }

    const int64_t macro_count = nrows * macros_per_row;
    if ((uint64_t) macro_count >
        SIZE_MAX / ROCMFPX_Q7_Q8_VIEW_MACRO_BYTES) {
        return false;
    }

    *view_size =
        (size_t) macro_count * ROCMFPX_Q7_Q8_VIEW_MACRO_BYTES;
    return true;
}

static inline bool rocmfpx_q7_q8_view_mmq_addressable(
        const int64_t nrows,
        const int64_t macros_per_row) {
    if (nrows <= 0 ||
        nrows > INT_MAX ||
        macros_per_row <= 0 ||
        macros_per_row > INT_MAX / QK_ROCMFP7 ||
        nrows > INT64_MAX / macros_per_row) {
        return false;
    }

    const int64_t macro_count = nrows * macros_per_row;
    return macro_count <= INT_MAX / NG_ROCMFP7;
}

static inline void rocmfpx_q7_expand_macro_to_q8_view(
        const block_rocmfp7 * source,
        void * destination) {
    uint8_t * output = (uint8_t *) destination;
    for (int group = 0; group < NG_ROCMFP7; ++group) {
        uint8_t * q8_block =
            output + group * ROCMFPX_Q7_Q8_VIEW_BLOCK_BYTES;
        memcpy(q8_block, &source->d[group], sizeof(source->d[group]));

        int8_t * q8_codes = (int8_t *) (q8_block + sizeof(source->d[group]));
        const uint8_t * q7_codes =
            source->qs + group * QS_ROCMFP7_GROUP;
        for (int pack = 0; pack < QG_ROCMFP7 / 8; ++pack) {
            uint64_t bits = 0;
            for (int byte = 0; byte < 7; ++byte) {
                bits |=
                    (uint64_t) q7_codes[pack * 7 + byte] <<
                    (8 * byte);
            }
            for (int index = 0; index < 8; ++index) {
                const uint8_t code =
                    (uint8_t) ((bits >> (7 * index)) & 0x7fu);
                q8_codes[pack * 8 + index] =
                    (int8_t) ((code & 0x40u) ?
                        (int) code - 128 :
                        (int) code);
            }
        }
    }
}

static inline bool rocmfpx_q7_expand_rows_to_q8_view(
        const block_rocmfp7 * source,
        void * destination,
        const int64_t nrows,
        const int64_t macros_per_row) {
    size_t view_size;
    if (source == NULL ||
        destination == NULL ||
        !rocmfpx_q7_q8_view_size(
            nrows,
            macros_per_row,
            &view_size)) {
        return false;
    }

    uint8_t * output = (uint8_t *) destination;
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t macro_index = 0;
             macro_index < macros_per_row;
             ++macro_index) {
            const int64_t linear_index =
                row * macros_per_row + macro_index;
            rocmfpx_q7_expand_macro_to_q8_view(
                &source[linear_index],
                output +
                    (size_t) linear_index *
                        ROCMFPX_Q7_Q8_VIEW_MACRO_BYTES);
        }
    }
    (void) view_size;
    return true;
}
