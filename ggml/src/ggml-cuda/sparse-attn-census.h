#pragma once

// Dispatch census of the HIP sparse-attention kernels (a test hook, not a stable API; PLAN-v3 §6.2).
//
// Every key is generated from an instantiation table below. The same table drives the template instantiations and the
// host dispatch of the kernel family it describes (sparse-attn-plan.cu instantiates and dispatches exactly the groups
// of GGML_CUDA_SA_PLAN_GROUPS, the sparse-attn-attend-g4-<kt>.cu files instantiate and the dispatch of
// sparse-attn-attend-g4.cu indexes exactly the instances of GGML_CUDA_SA_G4_INSTANCES and GGML_CUDA_SA_FIXUP_INSTANCES,
// sparse-attn-pack.cu dispatches
// the types of GGML_CUDA_SA_PACK_TYPES), so the list of keys the census reports cannot drift from the compiled kernels.
//
// Keys:
//   sa_plan/G<g>                                one GGML_OP_SPARSE_ATTN_PLAN dispatch with group g (init + rows + merge
//                                               kernels of sparse-attn-plan.cu)
//   sa_attend/G4/<KT>/<VT>/<VL>/<EPI>/gqa<n>    one launch of sa_attend_g4 (sparse-attn-attend-g4.cuh): K type, V type,
//                                               V loader (packed | direct), epilogue (plain | gated), GQA (4 | 8 | 12).
//                                               The key of PLAN-v3 §6.2, sa_attend/G4/<KT>/<VT>/<VL>/<EPI>, is the prefix
//                                               of its three gqa<n> instances: count("sa_attend/G4/f16/q8_0/packed/plain")
//                                               sums them
//   sa_fixup/G4/<KT>/<VT>/<EPI>                 one launch of sa_attend_fixup (sparse-attn-attend-g4.cuh), which follows
//                                               every sa_attend_g4 launch and recomputes only the blocks that flagged a
//                                               non-finite V half (it returns at once for the others)
//   sa_pack/<VT>                                one launch of sa_pack_v (sparse-attn-pack.cu)
//   sa_attend/G1/<KT>/<VT>/splits<c>/gqa<n>     one launch of sa_attend_g1 (sparse-attn-attend-g1.cuh): K type, V type, the
//                                               split-count class c of the launch (1, 2-5, 6-16, 17-33, 34+: the classes of
//                                               the PLAN-v3 §14.2 band table; the split count itself is the shape-only
//                                               sa_g1_splits) and GQA. The key of PLAN-v3 §6.2, sa_attend/G1/<KT>/<VT>/
//                                               splits<n>, with n the first count of the class (atoi of the segment gives
//                                               the class's lower bound), extended by the GQA level as the G4 keys are
//   sa_merge                                    one launch of sa_merge_splits (sparse-attn-merge.cu), which follows every
//                                               sa_attend_g1 launch with more than one split
//   qsa_expand                                  one fused qwen4exp complete-block selection (expand.cu, matched in ggml-cuda.cu)
//
// The procs are returned by ggml_backend_reg_get_proc_address of the HIP backend (ggml-cuda.cu):
//   void                 ggml_backend_debug_dispatch_census_reset(void)          clear the counts and start counting
//   uint64_t             ggml_backend_debug_dispatch_census_count(const char *)  exactly KEY plus every KEY/... below it
//   size_t               ggml_backend_debug_dispatch_census_get(char *, size_t)  "KEY=count;..." (non-zero keys, key order)
//   const char * const * ggml_backend_debug_dispatch_census_list(void)           every compiled key, NULL-terminated
//   const char *         ggml_backend_debug_dispatch_census_skip(ggml_backend_t, const char * label)
//                        NULL when label is a compiled key (or a prefix of one), else why HIP cannot emit it
// Ids are recorded into HIP graphs as uint16_t (ggml_cuda_graph::census_ids).
// Counts are launches: a dispatch counts when the host issues it, and with HIP graphs on the ids dispatched while a graph
// is captured are recorded into the graph (ggml_cuda_graph::census_ids) and counted again at every replay (ggml-cuda.cu).
// GGML_CUDA_DISPATCH_CENSUS_LOG=1 counts from backend registration and prints the census to stderr at exit.

// groups G of the plan kernels (sa_plan_merge<G>); also the G range of the HIP group policy for get_alloc_size (§2.1)
#define GGML_CUDA_SA_PLAN_GROUPS(X) \
    X(1)                            \
    X(4)

#define GGML_CUDA_SA_CENSUS_PLAN_KEY(g) "sa_plan/G" #g

// G4 attend instances (PLAN-v3 §6.2: KT 4 x (VT, VL) 8 x EPI 2, times GQA {4, 8, 12}), K type outermost. The type
// tokens are the ggml type names; sparse-attn-loaders.cuh maps them to the kernel's ids (SA_TID_<kt>, SA_VL_<vl>, SA_EPI_<epi>).
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

// G1 attend instances (sa_attend_g1<KT, VT, GQA>: KT 4 x VT 4 x GQA 3), K type outermost; one kernel each, dispatched
// by sparse-attn-attend-g1.cu at index (kt * 4 + vt) * 3 + gqa_index
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

// census keys of the G1 instances: every instance x the 5 split classes (sa_g1_split_class, sparse-attn-attend-g1.cuh);
// key id GGML_CUDA_SA_CENSUS_G1_FIRST + instance * 5 + class
#define GGML_CUDA_SA_G1_CENSUS_(X, kt, vt, gqa)        \
    X(kt, vt, gqa, 0, "1")                             \
    X(kt, vt, gqa, 1, "2-5")                           \
    X(kt, vt, gqa, 2, "6-16")                          \
    X(kt, vt, gqa, 3, "17-33")                         \
    X(kt, vt, gqa, 4, "34+")

#define GGML_CUDA_SA_CENSUS_G1_KEY(kt, vt, gqa, cls) "sa_attend/G1/" #kt "/" #vt "/splits" cls "/gqa" #gqa

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
    GGML_CUDA_SA_CENSUS_N,
};

// first G4 attend id; instance i of GGML_CUDA_SA_G4_INSTANCES has id GGML_CUDA_SA_CENSUS_G4_FIRST + i
#define GGML_CUDA_SA_CENSUS_G4_FIRST GGML_CUDA_SA_CENSUS_G4_f16_f16_packed_plain_4
static_assert(GGML_CUDA_SA_CENSUS_FIXUP_f16_f16_plain - GGML_CUDA_SA_CENSUS_G4_FIRST == 4 * GGML_CUDA_SA_G4_PER_KT,
              "GGML_CUDA_SA_G4_INSTANCES must list 4 K types of GGML_CUDA_SA_G4_PER_KT instances each");
// first fixup id; instance i of GGML_CUDA_SA_FIXUP_INSTANCES has id GGML_CUDA_SA_CENSUS_FIXUP_FIRST + i
#define GGML_CUDA_SA_CENSUS_FIXUP_FIRST GGML_CUDA_SA_CENSUS_FIXUP_f16_f16_plain
static_assert(GGML_CUDA_SA_CENSUS_PACK_f16 - GGML_CUDA_SA_CENSUS_FIXUP_FIRST == 4 * GGML_CUDA_SA_FIXUP_PER_KT,
              "GGML_CUDA_SA_FIXUP_INSTANCES must list 4 K types of GGML_CUDA_SA_FIXUP_PER_KT instances each");

// first G1 id; instance i of GGML_CUDA_SA_G1_INSTANCES, split class c has id GGML_CUDA_SA_CENSUS_G1_FIRST + 5 * i + c
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
    nullptr,
};

// counts one dispatch of key id while the census is on (one relaxed load when it is off); defined in ggml-cuda.cu
void ggml_cuda_sa_census_add(ggml_cuda_sa_census_id id);
