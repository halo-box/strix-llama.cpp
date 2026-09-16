#pragma once
#include "common.cuh"
bool ggml_cuda_flash_attn_ext_qsa_prefill_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_qsa_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
