#pragma once
#include "common.cuh"
// the per-query decode kernels take 1..512 queries; larger ubatches go to qsa_prefill. Up to 512 queries the
// per-query gathers cost less than qsa_prefill's pack of the whole K/V cache, which grows with the depth
#define QSA_DECODE_MAX_QUERIES 512
bool ggml_cuda_flash_attn_ext_qsa_decode_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_qsa_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
