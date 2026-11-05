#pragma once
#include "common.cuh"
// Quantized-weight BF16 WMMA GEMM on gfx1151, from 512 GEMM rows up, or 32 with the small-batch opt-in (see mmb_min_t).
// Off unless the backend context opts in (ggml_backend_cuda_set_mmb_enabled).
bool ggml_cuda_mmb_supported_mm  (ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
bool ggml_cuda_mmb_supported_mmid(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst);
void ggml_cuda_mul_mat_mmb   (ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
// qwen4exp QSA indexer scorer: F32 GEMM + relu + 4-head sum (+ compact visibility) in one kernel, dst [M, T/4]
bool ggml_cuda_mmb_idx_score_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * mm, int heads);
void ggml_cuda_mmb_idx_score(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int32_t * tails, const int32_t * starts);
void ggml_cuda_mul_mat_id_mmb(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
void ggml_cuda_mmb_begin_graph(ggml_backend_cuda_context & ctx);
// producers that can emit a BF16 copy of an F32 output register it here; returns the BF16 buffer to fill (n elements)
// (f16: an F16 copy, saturating RNE, for the dense F16 readers; BF16 lookups never return it)
uint16_t * ggml_cuda_mmb_cache_produce(ggml_backend_cuda_context & ctx, const ggml_tensor * t, size_t n, bool f16 = false);
uint16_t * ggml_cuda_mmb_cache_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * t, size_t n);
// BF16 copy of tensor t if one is cached for the current graph (consumers may read it instead of the F32 data)
const uint16_t * ggml_cuda_mmb_cache_lookup(ggml_backend_cuda_context & ctx, const ggml_tensor * t);
void ggml_cuda_mmb_cache_unproduce(ggml_backend_cuda_context & ctx, const uint16_t * p);
// producer slots (pinned until the next producer of the same kind): 0 = HC normalized stream xn, 1 = HC gate
// 2 = routed GLU output, 3 = gate-mix output, 4 = F16 copy of the gate-mix output (dense F16)
uint16_t * ggml_cuda_mmb_slot_reserve(ggml_backend_cuda_context & ctx, int slot, const ggml_tensor * t, size_t n, bool f16 = false);
// int8 copy of the normalized streams for the W8A8 HC down: the producing combine allocates/keys it, the GEMM
// looks it up by its src1 root (a mismatch means it does not hold this xn and the GEMM quantizes itself)
uint8_t * ggml_cuda_mmb_xnq8_buf(ggml_backend_cuda_context & ctx, const ggml_tensor * xn_root, size_t bytes, size_t scale_bytes);
float * ggml_cuda_mmb_xnq8_scale(ggml_backend_cuda_context & ctx);
const uint8_t * ggml_cuda_mmb_xnq8_get(ggml_backend_cuda_context & ctx, const ggml_tensor * xn_root, const float ** scale_out, int * gs_out = nullptr);
void ggml_cuda_mmb_xnq8_set_layout(ggml_backend_cuda_context & ctx, int gs);
void ggml_cuda_mmb_marks_clear(ggml_backend_cuda_context & ctx);
size_t ggml_cuda_mmb_marks_count(ggml_backend_cuda_context & ctx);
void ggml_cuda_mmb_mark_bf16_only(ggml_backend_cuda_context & ctx, const ggml_tensor * t);
bool ggml_cuda_mmb_is_bf16_only(ggml_backend_cuda_context & ctx, const ggml_tensor * t);
// dense F16 marks (graph_optimize): WANT = some reader takes an F16 copy (dense-r2 GEMM), ONLY = every
// reader is a dense-r2 GEMM, so a producer that writes the F16 copy may skip the BF16 one. Keyed by the root tensor.
#define MMB_F16_WANT 1
#define MMB_F16_ONLY 2
void ggml_cuda_mmb_mark_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * t, int bits);
int  ggml_cuda_mmb_f16_marks(ggml_backend_cuda_context & ctx, const ggml_tensor * t);
// reader kind of MUL_MAT node n for the dense F16 copies: 1 = dense-r2 GEMM, 0 = none
int  ggml_cuda_mmb_f16_reader(ggml_backend_cuda_context & ctx, const ggml_tensor * n);
bool ggml_cuda_mmb_gatemix();
// two F32-weight GEMMs (M1 + M2 <= 128 rows) on the same F32 activations in one pass (qwen4exp ssm_beta + ssm_alpha)
bool ggml_cuda_mmb_f32_dual(ggml_backend_cuda_context & ctx, const ggml_tensor * w1, const ggml_tensor * w2, const ggml_tensor * x, ggml_tensor * d1, ggml_tensor * d2);
bool ggml_cuda_mmb_down16();
bool ggml_cuda_mmb_res16();
bool ggml_cuda_mmb_blk16();
// shape and type checks of ggml_cuda_hc_gate_mix (which also needs the BF16 xn copy)
bool ggml_cuda_hc_gate_mix_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * lo, const ggml_tensor * xn, const ggml_tensor * dst, int hc);
bool ggml_cuda_hc_gate_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * lo, const ggml_tensor * xn, ggml_tensor * dst, int hc, float scale, float bias);
bool ggml_cuda_mmb_supported_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * glu);
void ggml_cuda_mul_mat_id_mmb_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * glu);
void ggml_cuda_mmb_shadow_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w);
void ggml_cuda_mmb_down_r2_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w);
// down R2: routed Q5_1 down weights repacked in place (R2 layout); every consumer of a packed tensor must go
// through ggml_cuda_mul_mat_id_mmb (R2 WMMA kernel) or ggml_cuda_mul_mat_id_r2_gemv.
bool ggml_cuda_mmb_r2_packed(const ggml_tensor * w);
void ggml_cuda_mmb_r2_forget(const void * base, size_t size);
// the in-place layout ("dense R2", "down R2", "GLU R2") of a repacked weight overlapping the range, nullptr if none
const char * ggml_cuda_mmb_r2_layout(const void * base, size_t size);
void ggml_cuda_mul_mat_id_r2_gemv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
// GLU R2: routed Q4_K gate/up pairs repacked in place (G2 layout). ggml_cuda_mmb_r2_packed() is true for them too;
// their consumers are ggml_cuda_mul_mat_id_mmb_glu (G2 WMMA), the fused G2 GLU GEMV and ggml_cuda_mul_mat_id_g2_gemv.
bool ggml_cuda_mmb_g2_packed(const ggml_tensor * w);
void ggml_cuda_mmb_glu_r2_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw);
void ggml_cuda_mul_mat_id_g2_gemv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
bool ggml_cuda_mmb_g2_glu_gemv_ok(const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * glu);
void ggml_cuda_mul_mat_id_g2_gemv_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * glu);
// dense R2: dense Q8_0 weights repacked in place to the k7/k8 tile layout; every MUL_MAT on a packed tensor must go
// through ggml_cuda_mul_mat_mmb (MMB batches), ggml_cuda_mul_mat_dq (any other T > 1) or the RDNA3.5 fused-quantize
// matvecs (T == 1, packed-aware), never MMVQ/MMQ/cuBLAS. Buffer writes/frees drop the record (ggml_cuda_mmb_r2_forget).
bool ggml_cuda_mmb_dq_packed(const ggml_tensor * w);
void ggml_cuda_mmb_dense_r2_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w);
void ggml_cuda_mul_mat_dq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cuda_mmb_release_all(ggml_backend_cuda_context & ctx);
// GGML_CUDA_OVL (bitmask, default 15 = all on; 0 = off): side streams of the intra-graph concurrency on RDNA3.5.
//   1: GDN z region, 2: FFN shared expert region, 4: attention K/V region, 8: small-expert tier of the routed GLU
int ggml_cuda_ovl_mask();
