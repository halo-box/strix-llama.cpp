// The sa_attend_g4 and sa_attend_fixup instances with K type q4_0, one TU per K type so they build in parallel.

#include "sparse-attn-attend-g4.cuh"
#include "sparse-attn-census.h"

#if defined(GGML_USE_HIP)

#define SA_G4_FN(kt, vt, vl, epi, gqa) sa_attend_g4<SA_TID_##kt, SA_TID_##vt, SA_VL_##vl, SA_EPI_##epi, gqa, false>, \
                                       sa_attend_g4<SA_TID_##kt, SA_TID_##vt, SA_VL_##vl, SA_EPI_##epi, gqa, true>,
extern const sa_g4_kernel_t ggml_cuda_sa_g4_kernels_q4_0[2 * GGML_CUDA_SA_G4_PER_KT];
const sa_g4_kernel_t ggml_cuda_sa_g4_kernels_q4_0[2 * GGML_CUDA_SA_G4_PER_KT] = { GGML_CUDA_SA_G4_KT_(SA_G4_FN, q4_0) };
#undef SA_G4_FN

#define SA_FIXUP_FN(kt, vt, epi) sa_attend_fixup<SA_TID_##kt, SA_TID_##vt, SA_EPI_##epi>,
extern const sa_fixup_kernel_t ggml_cuda_sa_fixup_kernels_q4_0[GGML_CUDA_SA_FIXUP_PER_KT];
const sa_fixup_kernel_t ggml_cuda_sa_fixup_kernels_q4_0[GGML_CUDA_SA_FIXUP_PER_KT] = { GGML_CUDA_SA_FIXUP_KT_(SA_FIXUP_FN, q4_0) };
#undef SA_FIXUP_FN

#endif // defined(GGML_USE_HIP)
