// hooks: ggml/src/ggml-cuda/dsv4-hc.cuh of the fork (1fe05fa45, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#include "common.cuh"
#include "ggml.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// gate_scale_in/gate_scale_out: the SCALE nodes of a fused SCALE -> SIGMOID -> SCALE chain feeding src[2]
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in = nullptr, const ggml_tensor * gate_scale_out = nullptr);
} // namespace ggml_fork
