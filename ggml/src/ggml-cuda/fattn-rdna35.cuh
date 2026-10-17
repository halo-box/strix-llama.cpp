#pragma once

#include "common.cuh"

// Flash attention for RDNA3.5 (gfx115x) WMMA, head size 64, 128, 256 or 512: prefill (f16 K/V, other K/V types converted to
// f16 first) and, at depth, small batches with f16 or q8_0 K/V (decode, speculative verification).
bool ggml_cuda_flash_attn_ext_rdna35_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
bool ggml_cuda_flash_attn_ext_rdna35_supported(int device, const ggml_tensor * dst);
// scratch behind dst: f16 copies of a K/V cache of another type (none when the kernel reads q8_0 itself)
size_t ggml_cuda_flash_attn_ext_rdna35_alloc_size(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_rdna35(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
