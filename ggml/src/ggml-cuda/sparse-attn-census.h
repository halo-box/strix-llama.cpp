#pragma once

// Dispatch census of the HIP sparse-attention kernels (a test hook, not a stable API; PLAN-v3 §6.2).
//
// Every key is generated from an instantiation table below. The same table drives the template instantiations and the
// host dispatch of the kernel family it describes (sparse-attn-plan.cu instantiates and dispatches exactly the groups
// of GGML_CUDA_SA_PLAN_GROUPS), so the list of keys the census reports cannot drift from the compiled kernels.
//
// Keys:
//   sa_plan/G<g>   one GGML_OP_SPARSE_ATTN_PLAN dispatch with group g (init + rows + merge kernels of sparse-attn-plan.cu)
//   qsa_expand     one fused qwen4exp complete-block selection (expand.cu, matched in ggml-cuda.cu)
//
// The procs are returned by ggml_backend_reg_get_proc_address of the HIP backend (ggml-cuda.cu):
//   void                 ggml_backend_debug_dispatch_census_reset(void)          clear the counts and start counting
//   uint64_t             ggml_backend_debug_dispatch_census_count(const char *)  exactly KEY plus every KEY/... below it
//   size_t               ggml_backend_debug_dispatch_census_get(char *, size_t)  "KEY=count;..." (non-zero keys, key order)
//   const char * const * ggml_backend_debug_dispatch_census_list(void)           every compiled key, NULL-terminated
//   const char *         ggml_backend_debug_dispatch_census_skip(ggml_backend_t, const char * label)
//                        NULL when label is a compiled key (or a prefix of one), else why HIP cannot emit it
// Counts are launches: a dispatch counts when the host issues it, and with HIP graphs on the ids dispatched while a graph
// is captured are recorded into the graph (ggml_cuda_graph::census_ids) and counted again at every replay (ggml-cuda.cu).
// GGML_CUDA_DISPATCH_CENSUS_LOG=1 counts from backend registration and prints the census to stderr at exit.

// groups G of the plan kernels (sa_plan_merge<G>); also the G range of the HIP group policy for get_alloc_size (§2.1)
#define GGML_CUDA_SA_PLAN_GROUPS(X) \
    X(1)                            \
    X(4)

#define GGML_CUDA_SA_CENSUS_PLAN_KEY(g) "sa_plan/G" #g

enum ggml_cuda_sa_census_id {
#define GGML_CUDA_SA_CENSUS_PLAN_ID(g) GGML_CUDA_SA_CENSUS_PLAN_G##g,
    GGML_CUDA_SA_PLAN_GROUPS(GGML_CUDA_SA_CENSUS_PLAN_ID)
#undef GGML_CUDA_SA_CENSUS_PLAN_ID
    GGML_CUDA_SA_CENSUS_QSA_EXPAND,
    GGML_CUDA_SA_CENSUS_N,
};

// key strings in ggml_cuda_sa_census_id order, NULL-terminated
static const char * const ggml_cuda_sa_census_keys[GGML_CUDA_SA_CENSUS_N + 1] = {
#define GGML_CUDA_SA_CENSUS_PLAN_STR(g) GGML_CUDA_SA_CENSUS_PLAN_KEY(g),
    GGML_CUDA_SA_PLAN_GROUPS(GGML_CUDA_SA_CENSUS_PLAN_STR)
#undef GGML_CUDA_SA_CENSUS_PLAN_STR
    "qsa_expand",
    nullptr,
};

// counts one dispatch of key id while the census is on (one relaxed load when it is off); defined in ggml-cuda.cu
void ggml_cuda_sa_census_add(ggml_cuda_sa_census_id id);
