#include "common.cuh"

// GGML_OP_EXL3_MUL_MAT, GGML_OP_EXL3_MUL_MAT_ID and GGML_OP_EXL3_GET_ROWS on RDNA3 / RDNA4 (HIP builds with GGML_EXL3=ON),
// see exl3.cu
bool ggml_cuda_exl3_supports_op(int device, const ggml_tensor * op);

void ggml_cuda_op_exl3_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_exl3_mul_mat_id_supports_op(int device, const ggml_tensor * op);

void ggml_cuda_op_exl3_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_exl3_get_rows_supports_op(int device, const ggml_tensor * op);

void ggml_cuda_op_exl3_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
