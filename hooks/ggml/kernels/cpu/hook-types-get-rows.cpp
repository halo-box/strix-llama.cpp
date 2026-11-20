// GET_ROWS of a hook type on the CPU (token embeddings), through the to_float of the type. Upstream aborts on unknown types.

#include "ggml-hooks-cpu.h"

#include "ggml-cpu-impl.h"
#include "ggml-impl.h"

#include <algorithm>

static bool hook_type_get_rows_supports(const ggml_hook_dev *, const ggml_tensor * t) {
    return ggml_hook_type_is_hook(t->src[0]->type) && ggml_get_type_traits(t->src[0]->type)->to_float && t->type == GGML_TYPE_F32 &&
           t->src[1]->type == GGML_TYPE_I32;
}

static void hook_type_get_rows_compute(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_to_float_t to_float = ggml_get_type_traits(src0->type)->to_float;

    const int64_t nc = src0->ne[0];
    const int64_t nr = ggml_nelements(src1);
    const int64_t dr = (nr + params->nth - 1) / params->nth;
    const int64_t i0 = dr * params->ith;
    const int64_t i1 = std::min(i0 + dr, nr);

    for (int64_t i = i0; i < i1; ++i) {
        const int64_t i12 = i / (src1->ne[0] * src1->ne[1]);
        const int64_t i11 = (i - i12 * src1->ne[0] * src1->ne[1]) / src1->ne[0];
        const int64_t i10 = i - i12 * src1->ne[0] * src1->ne[1] - i11 * src1->ne[0];
        const int64_t i01 = *(const int32_t *) ((const char *) src1->data + i10 * src1->nb[0] + i11 * src1->nb[1] + i12 * src1->nb[2]);
        GGML_ASSERT(i01 >= 0 && i01 < src0->ne[1]);
        to_float((const char *) src0->data + i01 * src0->nb[1] + i11 * src0->nb[2] + i12 * src0->nb[3],
                 (float *) ((char *) dst->data + i10 * dst->nb[1] + i11 * dst->nb[2] + i12 * dst->nb[3]), nc);
    }
}

GGML_HOOK_REGISTER_KERNEL(ggml_hook_op_kernel("hook_types_cpu_get_rows", GGML_HOOK_BACKEND_CPU, GGML_OP_GET_ROWS, 0, hook_type_get_rows_supports,
                                              ggml_hook_fn(hook_type_get_rows_compute)));
