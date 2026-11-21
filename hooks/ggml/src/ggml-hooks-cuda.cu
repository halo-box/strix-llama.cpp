// CUDA/HIP/MUSA driver for the backend wrapper (see ggml-hooks-impl.h). Built in the backend target.

#include "ggml-hooks-cuda.h"
#include "ggml-hooks-impl.h"

#include "common.cuh"
#include "ggml-cuda.h"

namespace {

void cuda_dev_info(ggml_backend_dev_t dev, ggml_hook_dev * info) {
    // ggml_backend_cuda_device_context starts with `int device`
    const int device = *(const int *) dev->context;
    const auto & d = ggml_cuda_info().devices[device];
    info->device     = device;
    info->cc         = d.cc;
    info->name       = GGML_CUDA_NAME;
    info->integrated = d.integrated;
}

ggml_backend_cuda_context & cuda_ctx(ggml_backend_t backend) {
    return *(ggml_backend_cuda_context *) backend->context;
}

void cuda_begin(ggml_backend_t backend) {
    ggml_cuda_set_device(cuda_ctx(backend).device);
}

bool cuda_run_op(ggml_backend_t backend, const ggml_hook_kernel * k, ggml_cgraph * cgraph, int i) {
    return k->compute.cuda_op(cuda_ctx(backend), cgraph->nodes[i]);
}

void cuda_run_fusion(ggml_backend_t backend, const ggml_hook_kernel * k, ggml_cgraph * cgraph, int i, int n) {
    k->compute.cuda_fusion(cuda_ctx(backend), cgraph, i, n);
}

void cuda_end(ggml_backend_t, bool) {
}

constexpr ggml_hook_driver driver = {
    /* .backend       = */ GGML_HOOK_BACKEND_CUDA,
    /* .reg_name      = */ GGML_CUDA_NAME,
    /* .reg           = */ ggml_backend_cuda_reg,
    /* .dev_info      = */ cuda_dev_info,
    /* .cut_ops       = */ true,
    /* .begin         = */ cuda_begin,
    /* .run_op        = */ cuda_run_op,
    /* .run_fusion    = */ cuda_run_fusion,
    /* .end           = */ cuda_end,
    /* .installed     = */ nullptr,
    /* .private_bufts = */ nullptr,
};

GGML_HOOK_REGISTER_DRIVER(driver);

} // namespace

// Backends created without the registry (ggml_backend_{name}_init): the upstream function is renamed in its file
// (hooks_rename, ggml-hooks.cmake). This one installs the wrapper of the registry and wraps the new backend.
extern "C" { GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init__upstream(int device); }
ggml_backend_t ggml_backend_cuda_init(int device) {
    ggml_backend_t backend = ggml_backend_cuda_init__upstream(device);
    ggml_hook_install(ggml_backend_cuda_reg());
    ggml_hook_wrap_backend(backend);
    return backend;
}
