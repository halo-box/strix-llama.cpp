// Generic tier: the G4 and G1 WMMA attends with K = V = SA_T_GEN, whose loaders convert any type of S through a runtime switch on the ggml type. Runs every K / V pair outside S_fast, and every pair under GGML_CUDA_SA_FORCE=generic.
// Workspace layouts are the S_fast ones, so get_alloc_size is unchanged. No gated instance: the MUL runs unfused.

#include "sparse-attn.cuh"
#include "sparse-attn-census.h"

#if defined(GGML_USE_HIP)

#include "sparse-attn-attend-g4.cuh"
#include "sparse-attn-attend-g1.cuh"

#define SA_GEN_G4_FN(vl, gqa) sa_attend_g4<SA_T_GEN, SA_T_GEN, SA_VL_##vl, SA_EPI_PLAIN, gqa, false>, \
                              sa_attend_g4<SA_T_GEN, SA_T_GEN, SA_VL_##vl, SA_EPI_PLAIN, gqa, true>,
extern const sa_g4_kernel_t ggml_cuda_sa_gen_g4_kernels[2 * GGML_CUDA_SA_GEN_G4_N];
const sa_g4_kernel_t ggml_cuda_sa_gen_g4_kernels[2 * GGML_CUDA_SA_GEN_G4_N] = { GGML_CUDA_SA_GEN_G4_INSTANCES(SA_GEN_G4_FN) };
#undef SA_GEN_G4_FN

extern const sa_fixup_kernel_t ggml_cuda_sa_gen_fixup_kernel;
const sa_fixup_kernel_t ggml_cuda_sa_gen_fixup_kernel = sa_attend_fixup<SA_T_GEN, SA_T_GEN, SA_EPI_PLAIN>;

#define SA_GEN_G1_FN(gqa) sa_attend_g1<SA_T_GEN, SA_T_GEN, gqa, false>, sa_attend_g1<SA_T_GEN, SA_T_GEN, gqa, true>,
extern const sa_g1_kernel_t ggml_cuda_sa_gen_g1_kernels[2 * GGML_CUDA_SA_GEN_G1_N];
const sa_g1_kernel_t ggml_cuda_sa_gen_g1_kernels[2 * GGML_CUDA_SA_GEN_G1_N] = { GGML_CUDA_SA_GEN_G1_INSTANCES(SA_GEN_G1_FN) };
#undef SA_GEN_G1_FN

// one generic G1 launch, same grid and arguments as ggml_cuda_sa_g1_launch with the ggml types p.kt_g / p.vt_g
void ggml_cuda_sa_gen_g1_launch(cudaStream_t stream, const ggml_cuda_sa_g1_launch_params & p) {
    GGML_ASSERT(p.gqa_index >= 0 && p.gqa_index < GGML_CUDA_SA_GEN_G1_N);
    GGML_ASSERT(p.splits >= 1 && p.splits <= 65535);
    sa_g1_args a = sa_g1_args_from(p);
    a.kt_g      = p.kt_g;
    a.vt_g      = p.vt_g;
    const dim3 grid((unsigned) ((int64_t) p.n_q * p.n_stream * p.n_head_kv), (unsigned) p.splits, 1);
    const int rot = a.rot_q != 0 || a.rot_o != 0 ? 1 : 0;
    ggml_cuda_sa_gen_g1_kernels[2 * p.gqa_index + rot]<<<grid, 256, 0, stream>>>(a);
    const int cls = sa_g1_split_class(p.splits);
    ggml_cuda_sa_census_add((ggml_cuda_sa_census_id) (GGML_CUDA_SA_CENSUS_GEN_G1_FIRST + p.gqa_index * SA_G1_SPLIT_CLASSES + cls));
}

#endif // defined(GGML_USE_HIP)
