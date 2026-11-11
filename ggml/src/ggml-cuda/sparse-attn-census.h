#pragma once

// Dispatch census of the HIP sparse-attention kernels (a test hook, not a stable API). The tables below drive both the template instantiations and the host dispatch, so the reported keys cannot drift from the compiled kernels.
//
// Keys (one count per launch):
//   sa_plan/G<g>                                GGML_OP_SPARSE_ATTN_PLAN with group g
//   sa_attend/G4/<KT>/<VT>/<VL>/<EPI>/gqa<n>    sa_attend_g4: VL packed | direct, EPI plain | gated, GQA 4 | 8 | 12
//   sa_fixup/G4/<KT>/<VT>/<EPI>                 sa_attend_fixup, after every G4 launch (redoes blocks with a non-finite V)
//   sa_pack/<VT>                                sa_pack_v
//   sa_attend/G1/<KT>/<VT>/splits<c>/gqa<n>     sa_attend_g1, c the split-count class (1, 2-5, 6-16, 17-33, 34+;
//                                               atoi of the segment gives the class's lower bound)
//   sa_merge                                    sa_merge_splits, after every G1 launch with more than one split
//   sa_generic/G4/<VL>/gqa<n>                   generic tier G4 attend (any K / V pair of S)
//   sa_generic/G1/splits<c>/gqa<n>              generic tier G1 attend
//   sa_generic/fixup, sa_generic/pack           generic tier fixup and V pack
//   sa_portable/G<g>                            portable tier on a plan with group g (every HIP arch but RDNA3.5)
//   qsa_expand                                  one fused qwen4exp complete-block selection (expand.cu)
// A key also counts every KEY/... below it. Procs (ggml_backend_reg_get_proc_address of the HIP backend):
//   ggml_backend_debug_dispatch_census_reset / _count(key) / _list() (NULL-terminated) / _skip(backend, label) (NULL
//   when label is a compiled key or prefix, else why HIP cannot emit it).
// With HIP graphs, ids dispatched during capture are stored in the graph (as uint16_t) and counted at every replay.

// groups G of the plan kernels; also the G range of the HIP group policy for get_alloc_size
#define GGML_CUDA_SA_PLAN_GROUPS(X) \
    X(1)                            \
    X(4)

#define GGML_CUDA_SA_CENSUS_PLAN_KEY(g) "sa_plan/G" #g

// G4 attend instances, K type outermost; the tokens map to SA_TID_<kt>, SA_VL_<vl>, SA_EPI_<epi>
#define GGML_CUDA_SA_G4_INSTANCES(X)                   \
    GGML_CUDA_SA_G4_KT_(X, f16)                        \
    GGML_CUDA_SA_G4_KT_(X, bf16)                       \
    GGML_CUDA_SA_G4_KT_(X, q8_0)                       \
    GGML_CUDA_SA_G4_KT_(X, q4_0)
// the 48 instances of one K type (one sparse-attn-attend-g4-<kt>.cu each)
#define GGML_CUDA_SA_G4_KT_(X, kt)                     \
    GGML_CUDA_SA_G4_VT_(X, kt, f16)                    \
    GGML_CUDA_SA_G4_VT_(X, kt, bf16)                   \
    GGML_CUDA_SA_G4_VT_(X, kt, q8_0)                   \
    GGML_CUDA_SA_G4_VT_(X, kt, q4_0)
#define GGML_CUDA_SA_G4_VT_(X, kt, vt)                 \
    GGML_CUDA_SA_G4_VL_(X, kt, vt, packed)             \
    GGML_CUDA_SA_G4_VL_(X, kt, vt, direct)
#define GGML_CUDA_SA_G4_VL_(X, kt, vt, vl)             \
    GGML_CUDA_SA_G4_EPI_(X, kt, vt, vl, plain)         \
    GGML_CUDA_SA_G4_EPI_(X, kt, vt, vl, gated)
#define GGML_CUDA_SA_G4_EPI_(X, kt, vt, vl, epi)       \
    X(kt, vt, vl, epi, 4)                              \
    X(kt, vt, vl, epi, 8)                              \
    X(kt, vt, vl, epi, 12)

#define GGML_CUDA_SA_G4_PER_KT 48   // instances of one K type: VT 4 x VL 2 x EPI 2 x GQA 3

#define GGML_CUDA_SA_CENSUS_G4_KEY(kt, vt, vl, epi, gqa) "sa_attend/G4/" #kt "/" #vt "/" #vl "/" #epi "/gqa" #gqa

// fixup instances (sa_attend_fixup<KT, VT, EPI>; GQA is a runtime argument), K type outermost
#define GGML_CUDA_SA_FIXUP_INSTANCES(X)                \
    GGML_CUDA_SA_FIXUP_KT_(X, f16)                     \
    GGML_CUDA_SA_FIXUP_KT_(X, bf16)                    \
    GGML_CUDA_SA_FIXUP_KT_(X, q8_0)                    \
    GGML_CUDA_SA_FIXUP_KT_(X, q4_0)
#define GGML_CUDA_SA_FIXUP_KT_(X, kt)                  \
    GGML_CUDA_SA_FIXUP_VT_(X, kt, f16)                 \
    GGML_CUDA_SA_FIXUP_VT_(X, kt, bf16)                \
    GGML_CUDA_SA_FIXUP_VT_(X, kt, q8_0)                \
    GGML_CUDA_SA_FIXUP_VT_(X, kt, q4_0)
#define GGML_CUDA_SA_FIXUP_VT_(X, kt, vt)              \
    X(kt, vt, plain)                                   \
    X(kt, vt, gated)

#define GGML_CUDA_SA_FIXUP_PER_KT 8   // VT 4 x EPI 2

#define GGML_CUDA_SA_CENSUS_FIXUP_KEY(kt, vt, epi) "sa_fixup/G4/" #kt "/" #vt "/" #epi

// V types with a pack kernel (sa_pack_v<T>)
#define GGML_CUDA_SA_PACK_TYPES(X) \
    X(f16)                         \
    X(bf16)                        \
    X(q8_0)                        \
    X(q4_0)

// G1 attend instances, dispatched at index (kt * 4 + vt) * 3 + gqa_index
#define GGML_CUDA_SA_G1_INSTANCES(X)                   \
    GGML_CUDA_SA_G1_KT_(X, f16)                        \
    GGML_CUDA_SA_G1_KT_(X, bf16)                       \
    GGML_CUDA_SA_G1_KT_(X, q8_0)                       \
    GGML_CUDA_SA_G1_KT_(X, q4_0)
#define GGML_CUDA_SA_G1_KT_(X, kt)                     \
    GGML_CUDA_SA_G1_VT_(X, kt, f16)                    \
    GGML_CUDA_SA_G1_VT_(X, kt, bf16)                   \
    GGML_CUDA_SA_G1_VT_(X, kt, q8_0)                   \
    GGML_CUDA_SA_G1_VT_(X, kt, q4_0)
#define GGML_CUDA_SA_G1_VT_(X, kt, vt)                 \
    X(kt, vt, 4)                                       \
    X(kt, vt, 8)                                       \
    X(kt, vt, 12)

#define GGML_CUDA_SA_G1_N 48   // KT 4 x VT 4 x GQA 3

// G1 census keys: every instance x 5 split classes; id GGML_CUDA_SA_CENSUS_G1_FIRST + instance * 5 + class
#define GGML_CUDA_SA_G1_CENSUS_(X, kt, vt, gqa)        \
    X(kt, vt, gqa, 0, "1")                             \
    X(kt, vt, gqa, 1, "2-5")                           \
    X(kt, vt, gqa, 2, "6-16")                          \
    X(kt, vt, gqa, 3, "17-33")                         \
    X(kt, vt, gqa, 4, "34+")

#define GGML_CUDA_SA_CENSUS_G1_KEY(kt, vt, gqa, cls) "sa_attend/G1/" #kt "/" #vt "/splits" cls "/gqa" #gqa

// generic tier G4 instances, index vl * 3 + gqa_index (vl: SA_V_PACKED 0, SA_V_SPLITGATHER 1)
#define GGML_CUDA_SA_GEN_G4_INSTANCES(X) \
    X(packed, 4)                         \
    X(packed, 8)                         \
    X(packed, 12)                        \
    X(direct, 4)                         \
    X(direct, 8)                         \
    X(direct, 12)

#define GGML_CUDA_SA_GEN_G4_N 6

#define GGML_CUDA_SA_CENSUS_GEN_G4_KEY(vl, gqa) "sa_generic/G4/" #vl "/gqa" #gqa

// generic tier G1 instances, index gqa_index; census id GGML_CUDA_SA_CENSUS_GEN_G1_FIRST + gqa_index * 5 + class
#define GGML_CUDA_SA_GEN_G1_INSTANCES(X) \
    X(4)                                 \
    X(8)                                 \
    X(12)

#define GGML_CUDA_SA_GEN_G1_N 3

#define GGML_CUDA_SA_CENSUS_GEN_G1_KEY(gqa, cls) "sa_generic/G1/splits" cls "/gqa" #gqa

// portable tier: one kernel, one key per group of GGML_CUDA_SA_PLAN_GROUPS
#define GGML_CUDA_SA_CENSUS_PORTABLE_KEY(g) "sa_portable/G" #g

enum ggml_cuda_sa_census_id {
#define GGML_CUDA_SA_CENSUS_PLAN_ID(g) GGML_CUDA_SA_CENSUS_PLAN_G##g,
    GGML_CUDA_SA_PLAN_GROUPS(GGML_CUDA_SA_CENSUS_PLAN_ID)
#undef GGML_CUDA_SA_CENSUS_PLAN_ID
    GGML_CUDA_SA_CENSUS_QSA_EXPAND,
#define GGML_CUDA_SA_CENSUS_G4_ID(kt, vt, vl, epi, gqa) GGML_CUDA_SA_CENSUS_G4_##kt##_##vt##_##vl##_##epi##_##gqa,
    GGML_CUDA_SA_G4_INSTANCES(GGML_CUDA_SA_CENSUS_G4_ID)
#undef GGML_CUDA_SA_CENSUS_G4_ID
#define GGML_CUDA_SA_CENSUS_FIXUP_ID(kt, vt, epi) GGML_CUDA_SA_CENSUS_FIXUP_##kt##_##vt##_##epi,
    GGML_CUDA_SA_FIXUP_INSTANCES(GGML_CUDA_SA_CENSUS_FIXUP_ID)
#undef GGML_CUDA_SA_CENSUS_FIXUP_ID
#define GGML_CUDA_SA_CENSUS_PACK_ID(vt) GGML_CUDA_SA_CENSUS_PACK_##vt,
    GGML_CUDA_SA_PACK_TYPES(GGML_CUDA_SA_CENSUS_PACK_ID)
#undef GGML_CUDA_SA_CENSUS_PACK_ID
#define GGML_CUDA_SA_CENSUS_G1_ID(kt, vt, gqa, c, cls) GGML_CUDA_SA_CENSUS_G1_##kt##_##vt##_##gqa##_c##c,
#define GGML_CUDA_SA_CENSUS_G1_INST_(kt, vt, gqa) GGML_CUDA_SA_G1_CENSUS_(GGML_CUDA_SA_CENSUS_G1_ID, kt, vt, gqa)
    GGML_CUDA_SA_G1_INSTANCES(GGML_CUDA_SA_CENSUS_G1_INST_)
#undef GGML_CUDA_SA_CENSUS_G1_INST_
#undef GGML_CUDA_SA_CENSUS_G1_ID
    GGML_CUDA_SA_CENSUS_MERGE,
#define GGML_CUDA_SA_CENSUS_GEN_G4_ID(vl, gqa) GGML_CUDA_SA_CENSUS_GEN_G4_##vl##_##gqa,
    GGML_CUDA_SA_GEN_G4_INSTANCES(GGML_CUDA_SA_CENSUS_GEN_G4_ID)
#undef GGML_CUDA_SA_CENSUS_GEN_G4_ID
#define GGML_CUDA_SA_CENSUS_GEN_G1_ID(gqa, c, cls) GGML_CUDA_SA_CENSUS_GEN_G1_##gqa##_c##c,
#define GGML_CUDA_SA_CENSUS_GEN_G1_INST_(gqa) GGML_CUDA_SA_G1_CENSUS_(GGML_CUDA_SA_CENSUS_GEN_G1_ID_, generic, generic, gqa)
#define GGML_CUDA_SA_CENSUS_GEN_G1_ID_(kt, vt, gqa, c, cls) GGML_CUDA_SA_CENSUS_GEN_G1_ID(gqa, c, cls)
    GGML_CUDA_SA_GEN_G1_INSTANCES(GGML_CUDA_SA_CENSUS_GEN_G1_INST_)
#undef GGML_CUDA_SA_CENSUS_GEN_G1_ID_
#undef GGML_CUDA_SA_CENSUS_GEN_G1_INST_
#undef GGML_CUDA_SA_CENSUS_GEN_G1_ID
    GGML_CUDA_SA_CENSUS_GEN_FIXUP,
    GGML_CUDA_SA_CENSUS_GEN_PACK,
#define GGML_CUDA_SA_CENSUS_PORTABLE_ID(g) GGML_CUDA_SA_CENSUS_PORTABLE_G##g,
    GGML_CUDA_SA_PLAN_GROUPS(GGML_CUDA_SA_CENSUS_PORTABLE_ID)
#undef GGML_CUDA_SA_CENSUS_PORTABLE_ID
    GGML_CUDA_SA_CENSUS_N,
};

#define GGML_CUDA_SA_CENSUS_GEN_G4_FIRST GGML_CUDA_SA_CENSUS_GEN_G4_packed_4
// instance i, split class c: GGML_CUDA_SA_CENSUS_GEN_G1_FIRST + 5 * i + c
#define GGML_CUDA_SA_CENSUS_GEN_G1_FIRST GGML_CUDA_SA_CENSUS_GEN_G1_4_c0
static_assert(GGML_CUDA_SA_CENSUS_GEN_G1_FIRST - GGML_CUDA_SA_CENSUS_GEN_G4_FIRST == GGML_CUDA_SA_GEN_G4_N,
              "GGML_CUDA_SA_GEN_G4_INSTANCES must list GGML_CUDA_SA_GEN_G4_N instances");
static_assert(GGML_CUDA_SA_CENSUS_GEN_FIXUP - GGML_CUDA_SA_CENSUS_GEN_G1_FIRST == 5 * GGML_CUDA_SA_GEN_G1_N,
              "GGML_CUDA_SA_GEN_G1_INSTANCES must list GGML_CUDA_SA_GEN_G1_N instances of 5 split classes each");

#define GGML_CUDA_SA_CENSUS_G4_FIRST GGML_CUDA_SA_CENSUS_G4_f16_f16_packed_plain_4
static_assert(GGML_CUDA_SA_CENSUS_FIXUP_f16_f16_plain - GGML_CUDA_SA_CENSUS_G4_FIRST == 4 * GGML_CUDA_SA_G4_PER_KT,
              "GGML_CUDA_SA_G4_INSTANCES must list 4 K types of GGML_CUDA_SA_G4_PER_KT instances each");
#define GGML_CUDA_SA_CENSUS_FIXUP_FIRST GGML_CUDA_SA_CENSUS_FIXUP_f16_f16_plain
static_assert(GGML_CUDA_SA_CENSUS_PACK_f16 - GGML_CUDA_SA_CENSUS_FIXUP_FIRST == 4 * GGML_CUDA_SA_FIXUP_PER_KT,
              "GGML_CUDA_SA_FIXUP_INSTANCES must list 4 K types of GGML_CUDA_SA_FIXUP_PER_KT instances each");

// instance i, split class c: GGML_CUDA_SA_CENSUS_G1_FIRST + 5 * i + c
#define GGML_CUDA_SA_CENSUS_G1_FIRST GGML_CUDA_SA_CENSUS_G1_f16_f16_4_c0
static_assert(GGML_CUDA_SA_CENSUS_MERGE - GGML_CUDA_SA_CENSUS_G1_FIRST == 5 * GGML_CUDA_SA_G1_N,
              "GGML_CUDA_SA_G1_INSTANCES must list GGML_CUDA_SA_G1_N instances of 5 split classes each");

// key strings in ggml_cuda_sa_census_id order, NULL-terminated
static const char * const ggml_cuda_sa_census_keys[GGML_CUDA_SA_CENSUS_N + 1] = {
#define GGML_CUDA_SA_CENSUS_PLAN_STR(g) GGML_CUDA_SA_CENSUS_PLAN_KEY(g),
    GGML_CUDA_SA_PLAN_GROUPS(GGML_CUDA_SA_CENSUS_PLAN_STR)
#undef GGML_CUDA_SA_CENSUS_PLAN_STR
    "qsa_expand",
#define GGML_CUDA_SA_CENSUS_G4_STR(kt, vt, vl, epi, gqa) GGML_CUDA_SA_CENSUS_G4_KEY(kt, vt, vl, epi, gqa),
    GGML_CUDA_SA_G4_INSTANCES(GGML_CUDA_SA_CENSUS_G4_STR)
#undef GGML_CUDA_SA_CENSUS_G4_STR
#define GGML_CUDA_SA_CENSUS_FIXUP_STR(kt, vt, epi) GGML_CUDA_SA_CENSUS_FIXUP_KEY(kt, vt, epi),
    GGML_CUDA_SA_FIXUP_INSTANCES(GGML_CUDA_SA_CENSUS_FIXUP_STR)
#undef GGML_CUDA_SA_CENSUS_FIXUP_STR
#define GGML_CUDA_SA_CENSUS_PACK_STR(vt) "sa_pack/" #vt,
    GGML_CUDA_SA_PACK_TYPES(GGML_CUDA_SA_CENSUS_PACK_STR)
#undef GGML_CUDA_SA_CENSUS_PACK_STR
#define GGML_CUDA_SA_CENSUS_G1_STR(kt, vt, gqa, c, cls) GGML_CUDA_SA_CENSUS_G1_KEY(kt, vt, gqa, cls),
#define GGML_CUDA_SA_CENSUS_G1_INST_(kt, vt, gqa) GGML_CUDA_SA_G1_CENSUS_(GGML_CUDA_SA_CENSUS_G1_STR, kt, vt, gqa)
    GGML_CUDA_SA_G1_INSTANCES(GGML_CUDA_SA_CENSUS_G1_INST_)
#undef GGML_CUDA_SA_CENSUS_G1_INST_
#undef GGML_CUDA_SA_CENSUS_G1_STR
    "sa_merge",
#define GGML_CUDA_SA_CENSUS_GEN_G4_STR(vl, gqa) GGML_CUDA_SA_CENSUS_GEN_G4_KEY(vl, gqa),
    GGML_CUDA_SA_GEN_G4_INSTANCES(GGML_CUDA_SA_CENSUS_GEN_G4_STR)
#undef GGML_CUDA_SA_CENSUS_GEN_G4_STR
#define GGML_CUDA_SA_CENSUS_GEN_G1_STR_(kt, vt, gqa, c, cls) GGML_CUDA_SA_CENSUS_GEN_G1_KEY(gqa, cls),
#define GGML_CUDA_SA_CENSUS_GEN_G1_INST_(gqa) GGML_CUDA_SA_G1_CENSUS_(GGML_CUDA_SA_CENSUS_GEN_G1_STR_, generic, generic, gqa)
    GGML_CUDA_SA_GEN_G1_INSTANCES(GGML_CUDA_SA_CENSUS_GEN_G1_INST_)
#undef GGML_CUDA_SA_CENSUS_GEN_G1_INST_
#undef GGML_CUDA_SA_CENSUS_GEN_G1_STR_
    "sa_generic/fixup",
    "sa_generic/pack",
#define GGML_CUDA_SA_CENSUS_PORTABLE_STR(g) GGML_CUDA_SA_CENSUS_PORTABLE_KEY(g),
    GGML_CUDA_SA_PLAN_GROUPS(GGML_CUDA_SA_CENSUS_PORTABLE_STR)
#undef GGML_CUDA_SA_CENSUS_PORTABLE_STR
    nullptr,
};

// counts one dispatch while the census is on (one relaxed load when off); defined in ggml-cuda.cu
void ggml_cuda_sa_census_add(ggml_cuda_sa_census_id id);
