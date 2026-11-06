// The generic tier of GGML_OP_SPARSE_ATTN on HIP (PLAN-v3 §6.1 S_generic, §6.2 "sparse-attn-generic.cu"): the G4 and G1
// WMMA attends of sparse-attn-attend-g4.cuh / sparse-attn-attend-g1.cuh instantiated with K = V = SA_T_GEN, whose loaders
// convert every element of any type of S (f16, bf16, q8_0, q4_0, f32, q4_1, q5_0, q5_1, iq4_nl) through the runtime switch
// sa_any_piece / sa_any_gather16 / sa_any_elem (sparse-attn-loaders.cuh), the ggml types being kernel arguments.
//
// It runs every K / V pair with a member outside S_fast (S_generic), and every pair under GGML_CUDA_SA_FORCE=generic.
// Selection, workspace and dispatch: sparse-attn-attend-g4.cu (sa_select). The workspace layouts are the S_fast ones
// (packed V f16 by sa_pack_v<SA_T_GEN> of sparse-attn-pack.cu, direct-V wave flags, G1 split partials), so get_alloc_size
// is unchanged. The gate fusion has no generic instance (ggml_cuda_sparse_attn_gate_ok is false: the MUL runs unfused).
//
// Instances (census keys of sparse-attn-census.h):
//   sa_attend_g4<SA_T_GEN, SA_T_GEN, VL, SA_EPI_PLAIN, GQA>  VL packed | direct, GQA 4 | 8 | 12   sa_generic/G4/<VL>/gqa<n>
//   sa_attend_fixup<SA_T_GEN, SA_T_GEN, SA_EPI_PLAIN>                                             sa_generic/fixup
//   sa_attend_g1<SA_T_GEN, SA_T_GEN, GQA>                     GQA 4 | 8 | 12                      sa_generic/G1/splits<c>/gqa<n>
// Speed against S_fast is UNPROVEN (U8) and recorded by the W5 harness (scripts/qsa-validation/harness/w5).

#include "sparse-attn.cuh"
#include "sparse-attn-census.h"

#if defined(GGML_USE_HIP)

#include "sparse-attn-attend-g4.cuh"
#include "sparse-attn-attend-g1.cuh"

#define SA_GEN_G4_FN(vl, gqa) sa_attend_g4<SA_T_GEN, SA_T_GEN, SA_VL_##vl, SA_EPI_PLAIN, gqa>,
extern const sa_g4_kernel_t ggml_cuda_sa_gen_g4_kernels[GGML_CUDA_SA_GEN_G4_N];
const sa_g4_kernel_t ggml_cuda_sa_gen_g4_kernels[GGML_CUDA_SA_GEN_G4_N] = { GGML_CUDA_SA_GEN_G4_INSTANCES(SA_GEN_G4_FN) };
#undef SA_GEN_G4_FN

extern const sa_fixup_kernel_t ggml_cuda_sa_gen_fixup_kernel;
const sa_fixup_kernel_t ggml_cuda_sa_gen_fixup_kernel = sa_attend_fixup<SA_T_GEN, SA_T_GEN, SA_EPI_PLAIN>;

#define SA_GEN_G1_FN(gqa) sa_attend_g1<SA_T_GEN, SA_T_GEN, gqa>,
extern const sa_g1_kernel_t ggml_cuda_sa_gen_g1_kernels[GGML_CUDA_SA_GEN_G1_N];
const sa_g1_kernel_t ggml_cuda_sa_gen_g1_kernels[GGML_CUDA_SA_GEN_G1_N] = { GGML_CUDA_SA_GEN_G1_INSTANCES(SA_GEN_G1_FN) };
#undef SA_GEN_G1_FN

// one generic G1 launch (and its census count); the caller runs ggml_cuda_sa_merge_splits when splits > 1. Same grid and
// arguments as ggml_cuda_sa_g1_launch (sparse-attn-attend-g1.cu), with the ggml types p.kt_g / p.vt_g
void ggml_cuda_sa_gen_g1_launch(cudaStream_t stream, const ggml_cuda_sa_g1_launch_params & p) {
    GGML_ASSERT(p.gqa_index >= 0 && p.gqa_index < GGML_CUDA_SA_GEN_G1_N);
    GGML_ASSERT(p.splits >= 1 && p.splits <= 65535);
    sa_g1_args a = {};
    a.q = p.q; a.q_nb1 = p.q_nb[1]; a.q_nb2 = p.q_nb[2]; a.q_nb3 = p.q_nb[3];
    a.k = p.k; a.k_nb1 = p.k_nb[1]; a.k_nb2 = p.k_nb[2]; a.k_nb3 = p.k_nb[3];
    a.v = p.v; a.v_nb1 = p.v_nb[1]; a.v_nb2 = p.v_nb[2]; a.v_nb3 = p.v_nb[3];
    a.counts    = p.counts;
    a.quads     = p.quads;
    a.masks     = p.masks;
    a.n_groups  = p.n_groups;
    a.cap       = p.cap;
    a.n_q       = p.n_q;
    a.n_head    = p.n_head;
    a.n_head_kv = p.n_head_kv;
    a.n_stream  = p.n_stream;
    a.splits    = p.splits;
    a.scale     = p.scale;
    a.out       = p.out;
    a.partial   = p.partial;
    a.kt_g      = p.kt_g;
    a.vt_g      = p.vt_g;
    const dim3 grid((unsigned) ((int64_t) p.n_q * p.n_stream * p.n_head_kv), (unsigned) p.splits, 1);
    ggml_cuda_sa_gen_g1_kernels[p.gqa_index]<<<grid, 256, 0, stream>>>(a);
    const int cls = sa_g1_split_class(p.splits);
    ggml_cuda_sa_census_add((ggml_cuda_sa_census_id) (GGML_CUDA_SA_CENSUS_GEN_G1_FIRST + p.gqa_index * SA_G1_SPLIT_CLASSES + cls));
}

#endif // defined(GGML_USE_HIP)
