#pragma once
#include "common.cuh"
// Quantized-weight BF16 WMMA GEMM on gfx1151, from 512 tokens up.
// Off unless the backend context opts in (ggml_backend_cuda_set_mmb_enabled).
bool ggml_cuda_mmb_supported_mm  (ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
bool ggml_cuda_mmb_supported_mmid(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst);
void ggml_cuda_mul_mat_mmb   (ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cuda_mul_mat_id_mmb(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
void ggml_cuda_mmb_begin_graph(ggml_backend_cuda_context & ctx);
// producers that can emit a BF16 copy of an F32 output register it here; returns the BF16 buffer to fill (n elements)
uint16_t * ggml_cuda_mmb_cache_produce(ggml_backend_cuda_context & ctx, const ggml_tensor * t, size_t n);
uint16_t * ggml_cuda_mmb_cache_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * t, size_t n);
// BF16 copy of tensor t if one is cached for the current graph (consumers may read it instead of the F32 data)
const uint16_t * ggml_cuda_mmb_cache_lookup(ggml_backend_cuda_context & ctx, const ggml_tensor * t);
// producer slots (pinned until the next producer of the same kind): 0 = HC normalized stream xn, 1 = HC gate
uint16_t * ggml_cuda_mmb_slot_reserve(ggml_backend_cuda_context & ctx, int slot, const ggml_tensor * t, size_t n);
void ggml_cuda_mmb_marks_clear(ggml_backend_cuda_context & ctx);
size_t ggml_cuda_mmb_marks_count(ggml_backend_cuda_context & ctx);
void ggml_cuda_mmb_mark_bf16_only(ggml_backend_cuda_context & ctx, const ggml_tensor * t);
bool ggml_cuda_mmb_is_bf16_only(ggml_backend_cuda_context & ctx, const ggml_tensor * t);
bool ggml_cuda_mmb_gatemix();
// two F32-weight GEMMs (M1 + M2 <= 128 rows) on the same F32 activations in one pass (qwen4exp ssm_beta + ssm_alpha)
bool ggml_cuda_mmb_f32_dual(ggml_backend_cuda_context & ctx, const ggml_tensor * w1, const ggml_tensor * w2, const ggml_tensor * x, ggml_tensor * d1, ggml_tensor * d2);
bool ggml_cuda_mmb_down16();
bool ggml_cuda_mmb_res16();
bool ggml_cuda_mmb_blk16();
bool ggml_cuda_hc_gate_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * lo, const ggml_tensor * xn, ggml_tensor * dst, int hc, float scale, float bias);
bool ggml_cuda_mmb_supported_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * glu);
void ggml_cuda_mul_mat_id_mmb_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * glu);
void ggml_cuda_mmb_shadow_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w);
void ggml_cuda_mmb_release_all(ggml_backend_cuda_context & ctx);
