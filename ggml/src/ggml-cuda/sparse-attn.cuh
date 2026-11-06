#pragma once

// HIP sparse attention (GGML_OP_SPARSE_ATTN_PLAN, GGML_OP_SPARSE_ATTN; contracts in ggml.h, PLAN-v3 §2, §6.2).
// Host code of every sparse-attn-*.cu file is HIP only: on CUDA builds supports_op returns false for both ops.

#include "common.cuh"

// GGML_OP_SPARSE_ATTN_PLAN (sparse-attn-plan.cu)
bool   ggml_cuda_sparse_attn_plan_supported(int device, const ggml_tensor * op);
// ggml_nbytes(op) for the plan's own G, rounded up to the largest dst over every G of GGML_CUDA_SA_PLAN_GROUPS, plus the
// sort/dedup scratch (ggml_sparse_attn_plan_scratch_nbytes), which the kernels place right after ggml_nbytes(op)
size_t ggml_cuda_sparse_attn_plan_get_alloc_size(const ggml_tensor * op);
void   ggml_cuda_sparse_attn_plan(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GGML_OP_SPARSE_ATTN (sparse-attn-attend-g4.cu; G4 instances in sparse-attn-attend-g4-<kt>.cu, V pack in
// sparse-attn-pack.cu, G1 instances in sparse-attn-attend-g1.cu, split merge in sparse-attn-merge.cu, the generic tier in
// sparse-attn-generic.cu, the portable tier in sparse-attn-portable.cu)
bool   ggml_cuda_sparse_attn_supported(int device, const ggml_tensor * op);
// round_up(ggml_nbytes(op), 256) + the largest workspace any G of the policy range and any n_q' <= n_q (and n_kv' <= n_kv)
// can use (PLAN-v3 §5 "workspace size"); ggml_nbytes(op) when no shape needs a workspace
size_t ggml_cuda_sparse_attn_get_alloc_size(int device, const ggml_tensor * op);
void   ggml_cuda_sparse_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// gate fusion (ggml-cuda.cu): SPARSE_ATTN -> [views] -> CONT(gate) -> SIGMOID -> MUL computed by the EPI = gated instance.
// gate: F32, row R = (s * n_q + i) * n_head + h at gate->data + R * gate->nb[1]; out32 (the MUL output) and dst16 (its
// 16-bit copy for an MMB GEMM reader) may each be nullptr, not both
bool   ggml_cuda_sparse_attn_gate_ok(int device, const ggml_tensor * op);
void   ggml_cuda_sparse_attn_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * gate,
                                   float * out32, uint16_t * dst16, bool dst16_f16);
// why this process never dispatches census key `key` (or a key under the prefix `key`) on HIP device `device` (-1: any
// device): a forcing switch excludes it, or the device's arch runs another tier; nullptr when it can be dispatched
const char * ggml_cuda_sparse_attn_census_skip(int device, const char * key);
// the workspace this node uses: [dst->data + *offset, + *bytes) (the fusion overlap check)
void   ggml_cuda_sparse_attn_ws_range(int device, const ggml_tensor * op, size_t * offset, size_t * bytes);

// V pack (sparse-attn-pack.cu): f16 [n_stream][n_head_kv][n_quads][256][4] of cells 0 .. 4 * n_quads - 1 of v into dst;
// non-finite halves written as 0, their quad flagged in qflags [n_stream][n_head_kv][n_quads] and *any set (cleared first)
// generic: the generic tier's pack (sa_pack_v<SA_T_GEN>, any type of S), else the S_fast instance of v's type
void   ggml_cuda_sparse_attn_pack_v(cudaStream_t stream, const ggml_tensor * v, int64_t n_quads, uint16_t * dst,
                                    uint8_t * qflags, uint32_t * any, bool generic);

// G1 attend (sparse-attn-attend-g1.cu): the shape-only split count (sa_g1_splits of sparse-attn-attend-g1.cuh), the
// partial bytes of a split count, and their max over every runtime n_q' <= n_q, n_kv' <= n_kv (PLAN-v3 §5)
int64_t ggml_cuda_sa_g1_splits(int64_t n_q, int64_t n_sel, int64_t n_kv);
size_t  ggml_cuda_sa_g1_partial_bytes(int64_t n_q, int64_t n_head, int64_t n_stream, int64_t splits);
size_t  ggml_cuda_sa_g1_partial_bytes_max(int64_t n_q, int64_t n_sel, int64_t n_kv, int64_t n_head, int64_t n_stream);

struct ggml_cuda_sa_g1_launch_params {
    const char *     q;
    size_t           q_nb[4];
    const char *     k;
    size_t           k_nb[4];
    const char *     v;
    size_t           v_nb[4];
    const uint32_t * counts;
    const uint32_t * quads;
    const uint16_t * masks;
    int              n_groups, cap, n_q, n_head, n_head_kv, n_stream, splits;
    int              kt, vt, gqa_index;   // sa_type_id of K and V, GQA 4 / 8 / 12 -> 0 / 1 / 2
    int              kt_g, vt_g;          // the ggml types of K and V (the generic tier)
    float            scale;
    float *          out;                 // splits == 1
    float *          partial;             // splits > 1
};
// one sa_attend_g1 launch (and its census count); the caller runs ggml_cuda_sa_merge_splits when splits > 1
void ggml_cuda_sa_g1_launch(cudaStream_t stream, const ggml_cuda_sa_g1_launch_params & p);
// the same for the generic tier (sa_attend_g1<SA_T_GEN, SA_T_GEN, GQA>, sparse-attn-generic.cu): any K / V type of S
// (p.kt_g, p.vt_g); p.kt / p.vt are not read
void ggml_cuda_sa_gen_g1_launch(cudaStream_t stream, const ggml_cuda_sa_g1_launch_params & p);

// the portable tier (sparse-attn-portable.cu): one scalar launch for any plan G, any K / V type of S and any HIP arch
struct ggml_cuda_sa_portable_launch_params {
    const char *     q;
    size_t           q_nb[4];
    const char *     k;
    size_t           k_nb[4];
    const char *     v;
    size_t           v_nb[4];
    const uint32_t * counts;
    const uint32_t * quads;
    const uint16_t * masks;
    int              n_groups, cap, G, n_q, n_head, n_head_kv, n_stream, gqa;
    int              kt_g, vt_g;          // the ggml types of K and V
    float            scale;
    float *          out;                 // dst
};
void ggml_cuda_sa_portable_launch(cudaStream_t stream, const ggml_cuda_sa_portable_launch_params & p);
// sa_merge_splits (sparse-attn-merge.cu): n_rows output rows of 256 floats from [n_rows][splits][258] partials
void ggml_cuda_sa_merge_splits(cudaStream_t stream, const float * partial, float * out, int64_t n_rows, int splits);

// HIP policies (sparse-attn-attend-g1.cu), exposed as the procs ggml_backend_sparse_attn_group and
// ggml_backend_sparse_attn_prefer_dense (ggml-backend.h, PLAN-v3 §5 "G and shortcut policy")
int  ggml_cuda_sparse_attn_group(int device, int64_t n_q_per_stream, int64_t n_kv, ggml_type type_k, ggml_type type_v, bool shortcut);
bool ggml_cuda_sparse_attn_prefer_dense(int device, ggml_type type_k, ggml_type type_v, int64_t n_q, int64_t n_kv);
// the AT-A crossover rule (PLAN-v3 §16.5): true where the G1 attend beats the G4 attend past the shortcut
bool ggml_cuda_sa_g1_beats_g4(int64_t n_q_per_stream, int64_t n_kv, ggml_type type_k, ggml_type type_v);
