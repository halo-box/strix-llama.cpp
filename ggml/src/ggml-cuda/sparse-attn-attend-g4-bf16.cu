// The 96 sa_attend_g4 instances (48 census indices x ROT false / true) and the 8 sa_attend_fixup instances with K type bf16 (GGML_CUDA_SA_G4_KT_ and
// GGML_CUDA_SA_FIXUP_KT_ of sparse-attn-census.h), one TU per K type so the G4 instances build in parallel. Dispatched by
// sparse-attn-attend-g4.cu through ggml_cuda_sa_g4_kernels_bf16 and ggml_cuda_sa_fixup_kernels_bf16.

#include "sparse-attn-attend-g4.cuh"
#include "sparse-attn-census.h"

#if defined(GGML_USE_HIP)

#define SA_G4_FN(kt, vt, vl, epi, gqa) sa_attend_g4<SA_TID_##kt, SA_TID_##vt, SA_VL_##vl, SA_EPI_##epi, gqa, false>, \
                                       sa_attend_g4<SA_TID_##kt, SA_TID_##vt, SA_VL_##vl, SA_EPI_##epi, gqa, true>,
extern const sa_g4_kernel_t ggml_cuda_sa_g4_kernels_bf16[2 * GGML_CUDA_SA_G4_PER_KT];
const sa_g4_kernel_t ggml_cuda_sa_g4_kernels_bf16[2 * GGML_CUDA_SA_G4_PER_KT] = { GGML_CUDA_SA_G4_KT_(SA_G4_FN, bf16) };
#undef SA_G4_FN

#define SA_FIXUP_FN(kt, vt, epi) sa_attend_fixup<SA_TID_##kt, SA_TID_##vt, SA_EPI_##epi>,
extern const sa_fixup_kernel_t ggml_cuda_sa_fixup_kernels_bf16[GGML_CUDA_SA_FIXUP_PER_KT];
const sa_fixup_kernel_t ggml_cuda_sa_fixup_kernels_bf16[GGML_CUDA_SA_FIXUP_PER_KT] = { GGML_CUDA_SA_FIXUP_KT_(SA_FIXUP_FN, bf16) };
#undef SA_FIXUP_FN

#endif // defined(GGML_USE_HIP)
