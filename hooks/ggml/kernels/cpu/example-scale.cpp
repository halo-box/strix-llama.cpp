// example: SCALE f32 on CPU. Off by default, used by test-ggml-hooks.

#include "ggml-hooks-cpu.h"

#include "ggml-cpu-impl.h"
#include "ggml-impl.h"

#include <algorithm>

static bool example_scale_supports(const ggml_hook_dev *, const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && t->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_is_contiguous(t->src[0]);
}

static void example_scale_compute(const ggml_compute_params * params, ggml_tensor * dst) {
    const float   s  = ggml_get_op_params_f32(dst, 0);
    const float   b  = ggml_get_op_params_f32(dst, 1);
    const int64_t n  = ggml_nelements(dst);
    const int64_t dn = (n + params->nth - 1) / params->nth;
    const int64_t i0 = dn * params->ith;
    const int64_t i1 = std::min(n, i0 + dn);

    const float * a = (const float *) dst->src[0]->data;
    float *       d = (float *) dst->data;
    for (int64_t i = i0; i < i1; ++i) {
        d[i] = a[i] * s + b;
    }
}

GGML_HOOK_REGISTER_KERNEL([] {
    ggml_hook_kernel k = ggml_hook_op_kernel("example_cpu_scale_f32", GGML_HOOK_BACKEND_CPU, GGML_OP_SCALE, 0, example_scale_supports, ggml_hook_fn(example_scale_compute));
    k.enabled_by_default = false;
    return k;
}());
