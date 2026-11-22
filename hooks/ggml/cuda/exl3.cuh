// hooks: ggml/src/ggml-cuda/exl3.cuh of the fork (3b76dee48, upstream 9048bad8d, new in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

// GGML_OP_EXL3_MUL_MAT, GGML_OP_EXL3_MUL_MAT_ID and GGML_OP_EXL3_GET_ROWS on RDNA3 / RDNA4 (HIP builds, GGML_EXL3=ON)
bool ggml_cuda_exl3_supports_op(int device, const ggml_tensor * op);

void ggml_cuda_op_exl3_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_exl3_mul_mat_id_supports_op(int device, const ggml_tensor * op);

void ggml_cuda_op_exl3_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// dst = EXL3_MUL_MAT_ID(glu) where glu = swiglu(gate, up) (F32, split, contiguous, not read by any other node) and the op
// takes its GEMM path: the input rotation computes the SwiGLU, so the glu node is not computed
bool ggml_cuda_exl3_mul_mat_id_swiglu_fusable(int device, const ggml_tensor * glu, const ggml_tensor * dst);

void ggml_cuda_op_exl3_mul_mat_id_swiglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * glu);

bool ggml_cuda_exl3_get_rows_supports_op(int device, const ggml_tensor * op);

void ggml_cuda_op_exl3_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
} // namespace ggml_fork
