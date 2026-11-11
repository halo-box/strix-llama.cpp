// GGML_OP_SPARSE_ATTN with a G = 1 plan on HIP: the sa_attend_g1 instances, their launcher, the shape-only split count, and the HIP group and dense-shortcut policies. Selection and dispatch of the whole op are in sparse-attn-attend-g4.cu.

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

// G1 vs G4 crossover past the shortcut, measured per V type: G1 when n_q is at most the largest n_q where plan(G1) + G1 attend (+ merge) beat plan(G4) + the best G4 attend by more than the noise margin on every fixture of the n_kv class (classes 8192, 32768, 65536, 135168; boundaries at their geometric midpoints). Types outside S_fast: n_q < 128.
bool ggml_cuda_sa_g1_beats_g4(int64_t n_q_per_stream, int64_t n_kv, ggml_type type_k, ggml_type type_v) {
    GGML_UNUSED(type_k);
    // largest n_q with G1: [V f16, bf16, q8_0, q4_0][n_kv class 8192, 32768, 65536, 135168]
    static const int64_t nq_max[4][4] = {
        { 32, 16, 16,  64 },
        { 32, 32, 16,  64 },
        { 32, 32, 32,  64 },
        { 32, 64, 64, 128 },
    };
    const int tv = sa_policy_type(type_v);
    if (tv < 0) {
        return n_q_per_stream < 128;
    }
    const int c = n_kv < 16384 ? 0 : n_kv < 46341 ? 1 : n_kv < 94118 ? 2 : 3;
    return n_q_per_stream <= nq_max[tv][c];
}

// The work split G of a QSA layer's plan. In the dense shortcut (all-visible rows, n_kv <= top_k + r - 1): G1 up to n_q 32, G4 above, the same crossover for every S_fast pair. Past it: ggml_cuda_sa_g1_beats_g4.
int ggml_cuda_sparse_attn_group(int device, int64_t n_q_per_stream, int64_t n_kv, ggml_type type_k, ggml_type type_v, bool shortcut) {
    GGML_UNUSED(device);
    // test switch GGML_CUDA_SA_FORCE=g1 | g4, read at every call (get_alloc_size covers both G)
    if (ggml_cuda_sa_force_has("g1") || ggml_cuda_sa_force_has("g4")) {
        return ggml_cuda_sa_force_has("g4") ? 4 : 1;
    }
    if (shortcut) {
        return n_q_per_stream <= 32 ? 1 : 4;
    }
    return ggml_cuda_sa_g1_beats_g4(n_q_per_stream, n_kv, type_k, type_v) ? 1 : 4;
}

// Dense shortcut preference: f16 / f16 keeps the dense causal FLASH_ATTN_EXT; another pair is dense in the n_q bands {1, 2-8, 9-32, 33-127, 128-511, 512-1023, >= 1024} where sparse (plan + attend) was not below 0.98 x dense.
bool ggml_cuda_sparse_attn_prefer_dense(int device, ggml_type type_k, ggml_type type_v, int64_t n_q, int64_t n_kv) {
    GGML_UNUSED(device);
    GGML_UNUSED(n_kv);
    const int band = n_q <= 1 ? 0 : n_q <= 8 ? 1 : n_q <= 32 ? 2 : n_q <= 127 ? 3 : n_q <= 511 ? 4 : n_q <= 1023 ? 5 : 6;
    const int tk = sa_policy_type(type_k), tv = sa_policy_type(type_v);
    if (tk < 0 || tv < 0) {
        return false;   // S_generic pairs: sparse (no measured dense cell)
    }
    // [K f16, bf16, q8_0, q4_0][V f16, bf16, q8_0, q4_0][band]: 1 = dense
    static const bool dense[4][4][7] = {
        { { 1, 1, 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 0, 0, 0 }, { 1, 1, 1, 1, 0, 0, 0 }, { 1, 1, 1, 1, 0, 0, 0 } },   // K f16
        { { 1, 1, 1, 1, 0, 0, 0 }, { 1, 1, 1, 0, 0, 0, 0 }, { 0, 1, 1, 0, 0, 0, 0 }, { 0, 1, 1, 1, 0, 0, 0 } },   // K bf16
        { { 1, 1, 1, 1, 0, 0, 0 }, { 0, 1, 1, 1, 0, 0, 0 }, { 1, 1, 1, 1, 0, 0, 0 }, { 1, 1, 1, 1, 0, 0, 0 } },   // K q8_0
        { { 1, 1, 1, 1, 0, 0, 0 }, { 0, 1, 1, 1, 0, 0, 0 }, { 0, 1, 1, 1, 0, 0, 0 }, { 1, 1, 1, 1, 0, 0, 0 } },   // K q4_0
    };
    return dense[tk][tv][band];
}

#if defined(GGML_USE_HIP)

// two instances per census index i = (kt * 4 + vt) * 3 + gqa_index: 2 * i is ROT = false (no rotation code), 2 * i + 1 true
#define SA_G1_FN(kt, vt, gqa) sa_attend_g1<SA_TID_##kt, SA_TID_##vt, gqa, false>, sa_attend_g1<SA_TID_##kt, SA_TID_##vt, gqa, true>,
static const sa_g1_kernel_t sa_g1_kernels[2 * GGML_CUDA_SA_G1_N] = { GGML_CUDA_SA_G1_INSTANCES(SA_G1_FN) };
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
    a.rot_q       = p.rot_q;
    a.rot_o       = p.splits == 1 ? p.rot_o : 0;   // with more splits sa_merge_splits applies H_o
    a.rot_q_scale = ggml_cuda_sa_rot_scale(p.rot_q);
    a.rot_o_scale = ggml_cuda_sa_rot_scale(p.rot_o);
    a.rot_dump    = p.rot_dump;
    const int idx = (p.kt * SA_T_COUNT + p.vt) * 3 + p.gqa_index;
    // grid: x = (stream, query, kv head) with the kv head fastest, y = split
    const dim3 grid((unsigned) ((int64_t) p.n_q * p.n_stream * p.n_head_kv), (unsigned) p.splits, 1);
    const int rot = a.rot_q != 0 || a.rot_o != 0 ? 1 : 0;
    sa_g1_kernels[2 * idx + rot]<<<grid, 256, 0, stream>>>(a);
    const int cls = sa_g1_split_class(p.splits);
    ggml_cuda_sa_census_add((ggml_cuda_sa_census_id) (GGML_CUDA_SA_CENSUS_G1_FIRST + idx * SA_G1_SPLIT_CLASSES + cls));
}

#endif // defined(GGML_USE_HIP)
