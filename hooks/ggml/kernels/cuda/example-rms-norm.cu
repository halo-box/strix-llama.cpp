// example: RMS_NORM f32 on CUDA/HIP, one block per row. Off by default.
// Used to measure the glue: it runs twice per layer of a transformer, so it cuts the graph often.

#include "common.cuh"
#include "ggml-impl.h"

#include "ggml-hooks-cuda.h"
#include "ggml-hooks.h"

static constexpr int EXAMPLE_RMS_BLOCK = 256;

static __global__ void example_rms_norm_kernel(const float * x, float * dst, const int ncols, const float eps) {
    const int64_t row = blockIdx.x;
    x   += row * ncols;
    dst += row * ncols;

    float sum = 0.0f;
    for (int c = threadIdx.x; c < ncols; c += EXAMPLE_RMS_BLOCK) {
        const float v = x[c];
        sum += v * v;
    }
    sum = warp_reduce_sum(sum);

    __shared__ float partial[EXAMPLE_RMS_BLOCK / WARP_SIZE];
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) {
        partial[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        sum = lane < EXAMPLE_RMS_BLOCK / WARP_SIZE ? partial[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            partial[0] = sum;
        }
    }
    __syncthreads();

    const float scale = rsqrtf(partial[0] / ncols + eps);
    for (int c = threadIdx.x; c < ncols; c += EXAMPLE_RMS_BLOCK) {
        dst[c] = scale * x[c];
    }
}

static bool example_rms_norm_supports(const ggml_hook_dev *, const ggml_tensor * t) {
    const ggml_tensor * src = t->src[0];
    return t->type == GGML_TYPE_F32 && src->type == GGML_TYPE_F32 && ggml_is_contiguous(src) && ggml_is_contiguous(t) &&
           src->ne[0] <= INT32_MAX && ggml_nrows(src) <= INT32_MAX;
}

static bool example_rms_norm_compute(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float eps = ggml_get_op_params_f32(dst, 0);
    example_rms_norm_kernel<<<(int) ggml_nrows(src), EXAMPLE_RMS_BLOCK, 0, ctx.stream()>>>(
        (const float *) src->data, (float *) dst->data, (int) src->ne[0], eps);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

GGML_HOOK_REGISTER_KERNEL([] {
    ggml_hook_kernel k = ggml_hook_op_kernel("example_cuda_rms_norm_f32", GGML_HOOK_BACKEND_CUDA, GGML_OP_RMS_NORM, 0,
                                             example_rms_norm_supports, ggml_hook_fn(example_rms_norm_compute));
    k.enabled_by_default = false;
    return k;
}());
