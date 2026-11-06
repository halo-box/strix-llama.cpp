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

// GGML_OP_SPARSE_ATTN (sparse-attn-attend-g4.cu; instances in sparse-attn-attend-g4-<kt>.cu, V pack in sparse-attn-pack.cu)
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
// why this process never dispatches census key `key` (a forcing switch excludes it), or nullptr
const char * ggml_cuda_sparse_attn_census_skip(const char * key);
// the workspace this node uses: [dst->data + *offset, + *bytes) (the fusion overlap check)
void   ggml_cuda_sparse_attn_ws_range(int device, const ggml_tensor * op, size_t * offset, size_t * bytes);

// V pack (sparse-attn-pack.cu): f16 [n_stream][n_head_kv][n_quads][256][4] of cells 0 .. 4 * n_quads - 1 of v into dst;
// non-finite halves written as 0, their quad flagged in qflags [n_stream][n_head_kv][n_quads] and *any set (cleared first)
void   ggml_cuda_sparse_attn_pack_v(cudaStream_t stream, const ggml_tensor * v, int64_t n_quads, uint16_t * dst,
                                    uint8_t * qflags, uint32_t * any);
