// example: SCALE f32 op kernel and SCALE + ADD fusion on CUDA/HIP. Off by default, used by test-ggml-hooks.

#include "common.cuh"
#include "ggml-impl.h"

#include "ggml-hooks-cuda.h"
#include "ggml-hooks.h"

static __global__ void example_scale_add_kernel(const float * a, const float * b, float * d, int64_t n, float scale, float bias) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    d[i] = a[i] * scale + bias + (b ? b[i] : 0.0f);
}

static void example_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b, ggml_tensor * d, float scale, float bias) {
    const int64_t n = ggml_nelements(d);
    const int64_t blocks = (n + 255) / 256;
    example_scale_add_kernel<<<blocks, 256, 0, ctx.stream()>>>((const float *) a->data, b ? (const float *) b->data : nullptr, (float *) d->data, n, scale, bias);
    CUDA_CHECK(cudaGetLastError());
}

static bool example_f32_contiguous(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

static bool example_scale_supports(const ggml_hook_dev *, const ggml_tensor * t) {
    return example_f32_contiguous(t) && example_f32_contiguous(t->src[0]);
}

static bool example_scale_compute(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    example_launch(ctx, dst->src[0], nullptr, dst, ggml_get_op_params_f32(dst, 0), ggml_get_op_params_f32(dst, 1));
    return true;
}

static const ggml_tensor * example_add_other(const ggml_tensor * scale, const ggml_tensor * add) {
    if (add->src[0] == scale) {
        return add->src[1];
    }
    if (add->src[1] == scale) {
        return add->src[0];
    }
    return nullptr;
}

static int example_scale_add_match(const ggml_hook_dev *, const ggml_cgraph * cgraph, int i) {
    if (!ggml_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_ADD })) {
        return 0;
    }
    const ggml_tensor * scale = cgraph->nodes[i];
    const ggml_tensor * add   = cgraph->nodes[i + 1];
    const ggml_tensor * b     = example_add_other(scale, add);
    if (!b || !example_scale_supports(nullptr, scale) || !example_f32_contiguous(add) || !example_f32_contiguous(b) ||
        !ggml_are_same_shape(scale, b) || !ggml_are_same_shape(scale, add)) {
        return 0;
    }
    return 2;
}

static void example_scale_add_compute(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int) {
    const ggml_tensor * scale = cgraph->nodes[i];
    ggml_tensor *       add   = cgraph->nodes[i + 1];
    example_launch(ctx, scale->src[0], example_add_other(scale, add), add, ggml_get_op_params_f32(scale, 0), ggml_get_op_params_f32(scale, 1));
}

GGML_HOOK_REGISTER_KERNEL([] {
    ggml_hook_kernel k = ggml_hook_op_kernel("example_cuda_scale_f32", GGML_HOOK_BACKEND_CUDA, GGML_OP_SCALE, 0, example_scale_supports, ggml_hook_fn(example_scale_compute));
    k.enabled_by_default = false;
    return k;
}());

GGML_HOOK_REGISTER_KERNEL([] {
    ggml_hook_kernel k = ggml_hook_fusion_kernel("example_cuda_scale_add_f32", GGML_HOOK_BACKEND_CUDA, GGML_OP_SCALE, 0, example_scale_add_match, ggml_hook_fn(example_scale_add_compute));
    k.enabled_by_default = false;
    return k;
}());
