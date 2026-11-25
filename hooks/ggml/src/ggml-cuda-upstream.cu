// The upstream ggml/src/ggml-cuda/ggml-cuda.cu, compiled through this file (ggml-hooks.cmake marks the original as a
// header): the hooks reach static upstream functions here. The upstream public functions that the hooks replace are
// renamed *__upstream (hooks_rename).

#include "../../../ggml/src/ggml-cuda/ggml-cuda.cu"

// the device answer of upstream, before the backend wrapper
bool ggml_hook_cuda_upstream_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return ggml_backend_cuda_device_supports_op(dev, op);
}

cudaStream_t ggml_hook_cuda_ctx_stream(ggml_backend_cuda_context & ctx) {
    return ctx.stream();
}

int ggml_hook_cuda_ctx_device(ggml_backend_cuda_context & ctx) {
    return ctx.device;
}
