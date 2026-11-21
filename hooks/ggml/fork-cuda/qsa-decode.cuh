// hooks: ggml/src/ggml-cuda/qsa-decode.cuh of the fork (1fe05fa45, upstream 9048bad8d, new in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/fork-cuda/README.md
// the per-query decode kernels take 1..512 queries; larger ubatches go to qsa_prefill. Up to 512 queries the
// per-query gathers cost less than qsa_prefill's pack of the whole K/V cache, which grows with the depth
#define QSA_DECODE_MAX_QUERIES 512
bool ggml_cuda_flash_attn_ext_qsa_decode_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
bool ggml_cuda_flash_attn_ext_qsa_decode_supported_dev(int device, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_qsa_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
} // namespace ggml_fork
