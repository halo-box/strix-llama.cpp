// hooks: ggml/src/ggml-cuda/fill.cu of the fork (1fe05fa45, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#include "fill.cuh"
#include "convert.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/fork-cuda/README.md

#define CUDA_FILL_BLOCK_SIZE 256

template <typename T>
static __global__ void fill_kernel(T * dst, const int64_t k, const T value) {
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = value;
}

void ggml_cuda_op_fill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(dst));

    float value;
    memcpy(&value, dst->op_params, sizeof(float));

    const int64_t k = ggml_nelements(dst);
    const int64_t num_blocks = (k + CUDA_FILL_BLOCK_SIZE - 1) / CUDA_FILL_BLOCK_SIZE;

    switch (dst->type) {
        case GGML_TYPE_F32:
            fill_kernel<<<num_blocks, CUDA_FILL_BLOCK_SIZE, 0, stream>>>((float *)dst_d, k, value);
            break;
        case GGML_TYPE_F16:
            fill_kernel<<<num_blocks, CUDA_FILL_BLOCK_SIZE, 0, stream>>>((half *)dst_d, k, ggml_cuda_cast<half>(value));
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}
} // namespace ggml_fork
