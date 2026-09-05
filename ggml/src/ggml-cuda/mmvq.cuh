#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// RDNA3.5 fused-quantize matvec with an activation prologue: y' = op(y_scale*y + y_bias) (op 1 = silu, 2 = sigmoid)
// applied while the activations are quantized in-kernel. `y` replaces dst->src[1] (same shape, F32, contiguous).
bool ggml_cuda_mul_mat_vec_q_fq_prologue_ok(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, const ggml_tensor * y);
void ggml_cuda_mul_mat_vec_q_fq_prologue(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * y,
        float y_scale, float y_bias, int y_op);

// RDNA3.5 grouped matvec: independent single-column matvecs sharing the same F32 activation vector `y`
// (Q8_0 rows via the fused-quantize path, optionally with a Q8_0 gate + GLU epilogue) in one launch.
#define GGML_CUDA_MMV_GROUP_MAX 4
struct ggml_cuda_mmv_group_seg {
    const ggml_tensor * w    = nullptr;   // [ncols, nrows] Q8_0 or F32
    const ggml_tensor * gate = nullptr;   // optional Q8_0 gate weights of the same shape (GLU fusion)
    ggml_tensor *       dst  = nullptr;   // F32 [nrows]; the GLU output when gate != nullptr
    ggml_glu_op         glu_op = GGML_GLU_OP_SWIGLU;
    float               glu_limit = 0.0f;
};
bool ggml_cuda_mmv_group_seg_ok(const ggml_tensor * w, const ggml_tensor * y);
void ggml_cuda_mmv_group(ggml_backend_cuda_context & ctx, const ggml_tensor * y, const ggml_cuda_mmv_group_seg * segs, int nseg);

// RDNA3.5 qwen4exp GDN output projection with the gated per-head norm in the prologue:
//   y' = sigmoid(z) * (rms_norm_128(attn) * norm_w), then the fused-quantize matvec of dst->src[0] with y'
bool ggml_cuda_mul_mat_vec_q_fq_gdn_gate_ok(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, const ggml_tensor * z);
void ggml_cuda_mul_mat_vec_q_fq_gdn_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const float * attn,
        const ggml_tensor * z, const ggml_tensor * norm_w, float eps);

// RDNA3.5 one-token IQ4_NL MoE down projection with the selected-expert weighted sum in the kernel epilogue.
bool ggml_cuda_mul_mat_id_weighted_rdna3_5_ok(const ggml_tensor * experts, const ggml_tensor * weights, const ggml_tensor * dst);
void ggml_cuda_mul_mat_id_weighted_rdna3_5(
        ggml_backend_cuda_context & ctx, const ggml_tensor * experts, const ggml_tensor * weights, ggml_tensor * dst);
