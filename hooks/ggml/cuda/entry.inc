// The entry points of the fork ggml-cuda.cu for the hooks bridge (hooks/ggml/kernels/cuda/cuda-bridge.cu). Included at the
// end of the fork ggml-cuda.cu, where its static functions are visible.

namespace ggml_fork {

bool ggml_hook_fork_compute_forward(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    return ggml_cuda_compute_forward(ctx, dst);
}

bool ggml_hook_fork_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return ggml_backend_cuda_device_supports_op(dev, op);
}

bool ggml_hook_fork_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return ggml_backend_cuda_device_offload_op(dev, op);
}

// backend->context is a fork context
enum ggml_status ggml_hook_fork_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    return ggml_backend_cuda_graph_compute(backend, cgraph);
}

void ggml_hook_fork_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_graph_optimize(backend, cgraph, params);
}

// the MMB opt-in of the fork registry procs (ggml_backend_cuda_set_mmb_*), on the fork backend shim
void ggml_hook_fork_set_mmb_enabled(ggml_backend_t backend, bool enable) {
    ggml_backend_cuda_set_mmb_enabled(backend, enable);
}

void ggml_hook_fork_set_mmb_small_batch(ggml_backend_t backend, bool enable) {
    ggml_backend_cuda_set_mmb_small_batch(backend, enable);
}

} // namespace ggml_fork
