// Quantize and validate dispatch for the hook types (ggml-hook-points.h). Built in ggml-base.

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

bool ggml_hook_type_quantize(enum ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix, size_t * result) {
    if (type == GGML_TYPE_Q8_0_HOOK_EXAMPLE) {
        *result = quantize_q8_0(src, dst, nrows, n_per_row, imatrix);
        return true;
    }
    return false;
}

bool ggml_hook_type_validate(enum ggml_type type, const void * data, size_t nbytes, bool * ok) {
    if (type == GGML_TYPE_Q8_0_HOOK_EXAMPLE) {
        *ok = ggml_validate_row_data(GGML_TYPE_Q8_0, data, nbytes);
        return true;
    }
    return false;
}
