#pragma once

// HIP sparse attention (GGML_OP_SPARSE_ATTN_PLAN, GGML_OP_SPARSE_ATTN; contracts in ggml.h). HIP only: on CUDA
// supports_op returns false for both ops.

#include "common.cuh"

bool   ggml_cuda_sparse_attn_plan_supported(int device, const ggml_tensor * op);
// ggml_nbytes(op) rounded up to the largest dst over every G of GGML_CUDA_SA_PLAN_GROUPS, plus the sort/dedup scratch
// that the kernels place right after it
size_t ggml_cuda_sparse_attn_plan_get_alloc_size(const ggml_tensor * op);
void   ggml_cuda_sparse_attn_plan(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GGML_OP_SPARSE_ATTN (sparse-attn-attend-*.cu, -pack, -merge, -generic, -portable)
bool   ggml_cuda_sparse_attn_supported(int device, const ggml_tensor * op);
// round_up(ggml_nbytes(op), 256) + the largest workspace any G of the policy range and any n_q' <= n_q, n_kv' <= n_kv
// can use; ggml_nbytes(op) when no shape needs one
size_t ggml_cuda_sparse_attn_get_alloc_size(int device, const ggml_tensor * op);
void   ggml_cuda_sparse_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// gate fusion: SPARSE_ATTN -> [views] -> CONT(gate) -> SIGMOID -> MUL in one gated instance. gate: F32, row
// R = (s * n_q + i) * n_head + h. out32 (MUL output) and dst16 (16-bit copy for a GEMM reader) may be nullptr, not both
bool   ggml_cuda_sparse_attn_gate_ok(int device, const ggml_tensor * op);
void   ggml_cuda_sparse_attn_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * gate,
                                   float * out32, uint16_t * dst16, bool dst16_f16);
// why census key `key` (or prefix) is never dispatched on device `device` (-1: any): a forcing switch or another tier
// for the arch; nullptr when it can be dispatched
const char * ggml_cuda_sparse_attn_census_skip(int device, const char * key);

// why _supported / _plan_supported is false for op, NULL when true
const char * ggml_cuda_sparse_attn_unsupported_reason(int device, const ggml_tensor * op);
// GGML_CUDA_SA_FORCE holds `token` (tokens listed in sparse-attn-attend-g4.cu)
bool ggml_cuda_sa_force_has(const char * token);
// the workspace this node uses: [dst->data + *offset, + *bytes)
void   ggml_cuda_sparse_attn_ws_range(int device, const ggml_tensor * op, size_t * offset, size_t * bytes);

// V pack: f16 [n_stream][n_head_kv][n_quads][256][4] of cells 0 .. 4 * n_quads - 1; non-finite halves written as 0 and
// their quad flagged in qflags [n_stream][n_head_kv][n_quads] and *any (cleared first). generic: any type of S
void   ggml_cuda_sparse_attn_pack_v(cudaStream_t stream, const ggml_tensor * v, int64_t n_quads, uint16_t * dst,
                                    uint8_t * qflags, uint32_t * any, bool generic);

// G1 attend: the shape-only split count, the partial bytes of a split count, and their max over n_q' <= n_q, n_kv' <= n_kv
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
    int              rot_q, rot_o;        // op_params[1], op_params[2]; rot_o applied by the merge when splits > 1
    int              rot_dump;            // test switch: H_q q into out, nothing else
    float *          out;                 // splits == 1
    float *          partial;             // splits > 1
};
// one G1 launch (and its census count); the caller runs ggml_cuda_sa_merge_splits when splits > 1
void ggml_cuda_sa_g1_launch(cudaStream_t stream, const ggml_cuda_sa_g1_launch_params & p);
// the same for the generic tier: any K / V type of S (p.kt_g, p.vt_g)
void ggml_cuda_sa_gen_g1_launch(cudaStream_t stream, const ggml_cuda_sa_g1_launch_params & p);

// the portable tier: one scalar launch for any plan G, any K / V type of S and any HIP arch
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
    int              rot_q, rot_o;        // op_params[1], op_params[2]
    int              rot_dump;            // test switch: H_q q into out, nothing else
    float *          out;                 // dst
};
void ggml_cuda_sa_portable_launch(cudaStream_t stream, const ggml_cuda_sa_portable_launch_params & p);
// n_rows output rows of 256 floats from [n_rows][splits][258] partials, then H_o over blocks of rot_o dims (0: none)
void ggml_cuda_sa_merge_splits(cudaStream_t stream, const float * partial, float * out, int64_t n_rows, int splits, int rot_o);

// 1 / sqrtf(rot), the scale fwht.cu applies (1 for rot 0)
static inline float ggml_cuda_sa_rot_scale(const int rot) {
    return rot != 0 ? 1 / sqrtf(rot) : 1.0f;
}

// HIP policies, exposed as the procs ggml_backend_sparse_attn_group and ggml_backend_sparse_attn_prefer_dense
int  ggml_cuda_sparse_attn_group(int device, int64_t n_q_per_stream, int64_t n_kv, ggml_type type_k, ggml_type type_v, bool shortcut);
bool ggml_cuda_sparse_attn_prefer_dense(int device, ggml_type type_k, ggml_type type_v, int64_t n_q, int64_t n_kv);
// true where the G1 attend beats the G4 attend past the shortcut
bool ggml_cuda_sa_g1_beats_g4(int64_t n_q_per_stream, int64_t n_kv, ggml_type type_k, ggml_type type_v);
