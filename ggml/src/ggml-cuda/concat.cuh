#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// CONCAT(state, TRANSPOSE(MUL(b, x))) with the product read only by the concat: writes the product in place of the
// standalone multiply. false when the shapes/layouts do not fit (the caller then computes the nodes normally).
bool ggml_cuda_op_concat_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * mul);
