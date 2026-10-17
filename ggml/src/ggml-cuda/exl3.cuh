#include "common.cuh"

// GGML_OP_EXL3_MUL_MAT on RDNA3 / RDNA4 (HIP builds with GGML_EXL3=ON), see exl3.cu
bool ggml_cuda_exl3_supports_op(int device, const ggml_tensor * op);

void ggml_cuda_op_exl3_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
