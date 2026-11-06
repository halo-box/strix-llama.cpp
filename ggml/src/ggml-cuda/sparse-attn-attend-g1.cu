// GGML_OP_SPARSE_ATTN with a G = 1 plan on HIP (PLAN-v3 §6.2, D3, D9): the 48 sa_attend_g1 instances
// (sparse-attn-attend-g1.cuh, GGML_CUDA_SA_G1_INSTANCES of sparse-attn-census.h), their launcher, the shape-only split
// count, and the HIP work-split and shortcut policies (the procs ggml_backend_sparse_attn_group and
// ggml_backend_sparse_attn_prefer_dense, ggml-backend.h; PLAN-v3 §5 "G and shortcut policy").
//
// Kernel selection, workspace and dispatch of the whole op live in sparse-attn-attend-g4.cu (sa_select), which calls
// ggml_cuda_sa_g1_launch below for G = 1 plans and sa_merge_splits (sparse-attn-merge.cu) when the split count is > 1.

#include "sparse-attn.cuh"
#include "sparse-attn-attend-g1.cuh"
#include "sparse-attn-census.h"

#include <cstdlib>
#include <cstring>
#include <string>

// ---- policies (every HIP build; the kernels below are HIP only) ----

// S_fast type index of the policy tables (f16, bf16, q8_0, q4_0), -1 otherwise
static int sa_policy_type(const ggml_type t) {
    switch (t) {
        case GGML_TYPE_F16:  return 0;
        case GGML_TYPE_BF16: return 1;
        case GGML_TYPE_Q8_0: return 2;
        case GGML_TYPE_Q4_0: return 3;
        default:             return -1;
    }
}

// GGML_CUDA_SA_FORCE (comma-separated) g1 | g4: the group policy returns that G (a test switch, PLAN-v3 §6.2; it changes
// the G llama builds its plans with, never placement; get_alloc_size covers both G whatever the switch says, §5). Read
// at every call, so a process can reserve with the policy's G and run with a forced one (AT-C5).
static int sa_force_group(void) {
    const char * e = getenv("GGML_CUDA_SA_FORCE");
    if (e == nullptr) {
        return 0;
    }
    const std::string s(e);
    int g = 0;
    size_t p = 0;
    while (p <= s.size()) {
        const size_t q = std::min(s.find(',', p), s.size());
        const std::string tok = s.substr(p, q - p);
        if (tok == "g1") {
            g = 1;
        } else if (tok == "g4") {
            g = 4;
        }
        p = q + 1;
    }
    return g;
}

// AT-A crossover (PLAN-v3 §16.5), fitted on the W4 kernels (scripts/qsa-validation/summaries/w4_ata.md): G1 where
// t_G1 = plan(G1) + G1 attend (+ merge) beats t_G4 = plan(G4) + min(G4 packed + pack, G4 direct) by more than 3 x the
// p90 run-to-run noise (p90 0.33 % over 1188 samples: margin 1 %) at every tested n_q up to the threshold, on every
// fixture of the n_kv class (C18 wiki / rand rows at n_kv 8192 and 135168 with SYN data, SYN:1.8 plans at 32768 and
// 65536; n_q 2 .. 512; C18 protocol, two processes). Keyed by the V type (D3: (n_q, n_kv, T_v)); n_kv is mapped to the
// nearest measured class on a log scale (boundaries at the geometric midpoints 16384, 46341, 94118); n_kv > 135168 uses
// the 135168 class and n_q between two tested values takes the lower one's verdict (both UNPROVEN extrapolations).
// Types outside S_fast (the generic tier, W5) keep the default n_q < 128 (C1 d3, C13 d1-d2).
bool ggml_cuda_sa_g1_beats_g4(int64_t n_q_per_stream, int64_t n_kv, ggml_type type_k, ggml_type type_v) {
    GGML_UNUSED(type_k);
    // largest n_q with G1: [V f16, bf16, q8_0, q4_0][n_kv class 8192, 32768, 65536, 135168]
    static const int64_t nq_max[4][4] = {
        { 32, 16, 16,  64 },
        { 32, 32, 16,  64 },
        { 32, 32, 32,  96 },
        { 32, 64, 64, 128 },
    };
    const int tv = sa_policy_type(type_v);
    if (tv < 0) {
        return n_q_per_stream < 128;
    }
    const int c = n_kv < 16384 ? 0 : n_kv < 46341 ? 1 : n_kv < 94118 ? 2 : 3;
    return n_q_per_stream <= nq_max[tv][c];
}

// The work split G of a QSA layer's plan (D3). Inputs: queries per stream, n_kv, the K / V types, and whether the layer
// is in the dense shortcut (all-visible rows, n_kv <= top_k + r - 1), which has its own decision (P2-9).
//
// Past the shortcut: the two-way rule fitted by AT-A (PLAN-v3 §16.5) on the W4 kernels (ggml_cuda_sa_g1_beats_g4).
// In the shortcut: the default n_q < 128 ? 1 : 4 (C1 d3, C13) until AT-W7PERF (W7) measures the all-visible rows.
int ggml_cuda_sparse_attn_group(int device, int64_t n_q_per_stream, int64_t n_kv, ggml_type type_k, ggml_type type_v, bool shortcut) {
    GGML_UNUSED(device);
    if (const int f = sa_force_group()) {
        return f;
    }
    if (shortcut) {
        return n_q_per_stream < 128 ? 1 : 4;
    }
    return ggml_cuda_sa_g1_beats_g4(n_q_per_stream, n_kv, type_k, type_v) ? 1 : 4;
}

// Dense shortcut preference (D4): f16 / f16 keeps the dense causal FLASH_ATTN_EXT, every other pair runs the sparse op
// with all-visible rows, until AT-W7PERF (W7) flips a (pair, n_q band) whose sparse cell is below 0.98 x dense. The n_q
// bands of D4: {1, 2-8, 9-32, 33-127, 128-511, 512-1023, >= 1024}; the table has one entry per band so a flip is a
// one-line change.
bool ggml_cuda_sparse_attn_prefer_dense(int device, ggml_type type_k, ggml_type type_v, int64_t n_q, int64_t n_kv) {
    GGML_UNUSED(device);
    GGML_UNUSED(n_kv);
    const int band = n_q <= 1 ? 0 : n_q <= 8 ? 1 : n_q <= 32 ? 2 : n_q <= 127 ? 3 : n_q <= 511 ? 4 : n_q <= 1023 ? 5 : 6;
    const int tk = sa_policy_type(type_k), tv = sa_policy_type(type_v);
    if (tk < 0 || tv < 0) {
        return false;   // S_generic pairs: sparse (no measured dense cell)
    }
    // [K f16, bf16, q8_0, q4_0][V f16, bf16, q8_0, q4_0][band]: 1 = dense. Default (D4): f16 / f16 only.
    static const bool dense[4][4][7] = {
        { { 1, 1, 1, 1, 1, 1, 1 }, { 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0 } },
        { { 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0 } },
        { { 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0 } },
        { { 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0 } },
    };
    return dense[tk][tv][band];
}

#if defined(GGML_USE_HIP)

// the 48 G1 instances, in GGML_CUDA_SA_G1_INSTANCES order: index (kt * 4 + vt) * 3 + gqa_index
#define SA_G1_FN(kt, vt, gqa) sa_attend_g1<SA_TID_##kt, SA_TID_##vt, gqa>,
static const sa_g1_kernel_t sa_g1_kernels[GGML_CUDA_SA_G1_N] = { GGML_CUDA_SA_G1_INSTANCES(SA_G1_FN) };
#undef SA_G1_FN

int64_t ggml_cuda_sa_g1_splits(int64_t n_q, int64_t n_sel, int64_t n_kv) {
    return sa_g1_splits(n_q, n_sel, n_kv);
}

size_t ggml_cuda_sa_g1_partial_bytes(int64_t n_q, int64_t n_head, int64_t n_stream, int64_t splits) {
    return sa_g1_partial_bytes(n_q, n_head, n_stream, splits);
}

size_t ggml_cuda_sa_g1_partial_bytes_max(int64_t n_q, int64_t n_sel, int64_t n_kv, int64_t n_head, int64_t n_stream) {
    return sa_g1_partial_bytes_max(n_q, n_sel, n_kv, n_head, n_stream);
}

void ggml_cuda_sa_g1_launch(cudaStream_t stream, const ggml_cuda_sa_g1_launch_params & p) {
    GGML_ASSERT(p.kt >= 0 && p.kt < SA_T_COUNT && p.vt >= 0 && p.vt < SA_T_COUNT && p.gqa_index >= 0 && p.gqa_index < 3);
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
    const int idx = (p.kt * SA_T_COUNT + p.vt) * 3 + p.gqa_index;
    // grid: x = (stream, query, kv head) with the kv head fastest (the PR's blockIdx.x), y = split
    const dim3 grid((unsigned) ((int64_t) p.n_q * p.n_stream * p.n_head_kv), (unsigned) p.splits, 1);
    sa_g1_kernels[idx]<<<grid, 256, 0, stream>>>(a);
    const int cls = sa_g1_split_class(p.splits);
    ggml_cuda_sa_census_add((ggml_cuda_sa_census_id) (GGML_CUDA_SA_CENSUS_G1_FIRST + idx * SA_G1_SPLIT_CLASSES + cls));
}

#endif // defined(GGML_USE_HIP)
