// hooks: ggml/src/ggml-cuda/vendors/cuda.h of the fork (403dfcb03, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

} // namespace ggml_fork
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

typedef uint32_t ggml_cuda_thread_mask_t;

#ifdef GGML_USE_NCCL
} // namespace ggml_fork
#include <nccl.h>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md
#endif // GGML_USE_NCCL

#if CUDART_VERSION >= 11080
} // namespace ggml_fork
#include <cuda_fp8.h>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md
#define FP8_AVAILABLE
#endif // CUDART_VERSION >= 11080

#if CUDART_VERSION >= 12080
} // namespace ggml_fork
#include <cuda_fp4.h>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md
#endif // CUDART_VERSION >= 12080

#if CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020
} // namespace ggml_fork
