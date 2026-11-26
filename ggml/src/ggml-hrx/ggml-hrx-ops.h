#pragma once

#include "ggml-hrx-common.h"

// supports_X must return true only for nodes that op_X can run. op_X returns false on failure.
// ops/binary.cpp is the reference for this pattern.

bool ggml_hrx_supports_binary(const ggml_hrx_device_context * dev, const ggml_tensor * op); // ADD SUB MUL DIV
bool ggml_hrx_op_binary(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_unary(const ggml_hrx_device_context * dev, const ggml_tensor * op); // UNARY SQR SQRT LOG CLAMP FILL REPEAT, strided SCALE
bool ggml_hrx_op_unary(ggml_backend_hrx_context * ctx, const ggml_tensor * node);
bool ggml_hrx_supports_arange(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_arange(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

// Drops the size-1 dims of ne and merges each dim pair that is contiguous in all n_t tensors. Tensor t has the shape
// tne[t] (in each dim ne or 1, or a divisor of ne for a repeat) and the element strides ts[t] (0 for a size-1 dim).
// Unused dims get ne 1 and stride 0. Returns the number of dims that are left, at least 1.
int ggml_hrx_collapse_dims(int64_t ne[GGML_MAX_DIMS], int n_t, int64_t (*tne)[GGML_MAX_DIMS], int64_t (*ts)[GGML_MAX_DIMS]);

bool ggml_hrx_supports_scale(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_scale(ggml_backend_hrx_context * ctx, const ggml_tensor * node);
bool ggml_hrx_supports_glu(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_glu(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_cpy(const ggml_hrx_device_context * dev, const ggml_tensor * op); // CPY DUP CONT
bool ggml_hrx_op_cpy(ggml_backend_hrx_context * ctx, const ggml_tensor * node);
bool ggml_hrx_supports_concat(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_concat(ggml_backend_hrx_context * ctx, const ggml_tensor * node);
// A CONCAT that the tiled kernel runs: along dim 0, b is a long transposed matrix. Auto-fusion leaves it alone.
bool ggml_hrx_concat_is_transposed(const ggml_tensor * op);
bool ggml_hrx_supports_pad(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_pad(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_get_rows(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_get_rows(ggml_backend_hrx_context * ctx, const ggml_tensor * node);
bool ggml_hrx_supports_set_rows(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_set_rows(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_rms_norm(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_rms_norm(ggml_backend_hrx_context * ctx, const ggml_tensor * node);
// CONCAT rows + RMS_NORM + SOFT_MAX(MUL_MAT) weights + MUL_MAT: a softmax-weighted mix of RMS-normalized rows
int  ggml_hrx_fuse_rms_norm_mix(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, std::vector<int> * absorbed);

bool ggml_hrx_supports_rope(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_rope(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_soft_max(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_soft_max(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_sum_rows(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_sum_rows(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_argsort(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_argsort(ggml_backend_hrx_context * ctx, const ggml_tensor * node);
bool ggml_hrx_supports_top_k(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_top_k(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_argmax(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_argmax(ggml_backend_hrx_context * ctx, const ggml_tensor * node);
bool ggml_hrx_supports_cumsum(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_cumsum(ggml_backend_hrx_context * ctx, const ggml_tensor * node);
bool ggml_hrx_supports_sum(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_sum(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_ssm_conv(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_ssm_conv(ggml_backend_hrx_context * ctx, const ggml_tensor * node);
// SSM_CONV of CONCAT(past, transposed cur), optionally with SCALE(ADD(cur, .)): the CONCAT writes only the columns of
// x that are read later, the SSM_CONV reads cur instead of x.
int  ggml_hrx_fuse_ssm_conv_tail(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, std::vector<int> * absorbed);
int  ggml_hrx_fuse_ssm_conv_concat(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, std::vector<int> * absorbed);

bool ggml_hrx_supports_mul_mat(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_mul_mat(ggml_backend_hrx_context * ctx, const ggml_tensor * node);
int  ggml_hrx_fuse_mul_mat_vec_group(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, std::vector<int> * absorbed);
// graph_optimize: moves the MUL_MATs that ggml_hrx_fuse_mul_mat_vec_group can run together next to each other.
void ggml_hrx_mul_mat_group_order(const ggml_hrx_device_context * dev, ggml_cgraph * cgraph);
void ggml_hrx_flash_attn_alloc_deps(ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params);
bool ggml_hrx_supports_mul_mat_id(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_mul_mat_id(ggml_backend_hrx_context * ctx, const ggml_tensor * node);
// ops/mul-mat-id.cpp: merged gate/up MUL_MAT_ID, expert scale, SITU activation and down MUL_MAT_ID.
int  ggml_hrx_fuse_moe_glu(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, std::vector<int> * absorbed);

// ops/npu.cpp: a part of large Q8_0 MUL_MATs runs as an int8 GEMM on an XDNA NPU in the same device group.
// GGML_HRX_NPU=1 turns it on. The NPU result reaches the node output before the first dispatch that uses it.
struct ggml_hrx_npu * ggml_hrx_npu_create(ggml_hrx_device_context * dev, hrx_device_t npu_device);
void ggml_hrx_npu_free(struct ggml_hrx_npu * npu);
// Drops the NPU weights derived from tensors in this buffer. Call it before the buffer is freed.
void ggml_hrx_npu_forget(struct ggml_hrx_npu * npu, hrx_buffer_t buffer);
// graph_optimize: moves the MUL_MATs that can overlap with later nodes on the NPU next to their inputs.
void ggml_hrx_npu_graph_order(const ggml_hrx_device_context * dev, ggml_cgraph * cgraph);
// Plans the NPU work of cgraph. Returns true if a node uses the NPU.
bool ggml_hrx_npu_plan(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph);
// Runs node i if the plan gives it to the NPU. Returns false if the node is not planned for the NPU.
bool ggml_hrx_npu_run(ggml_backend_hrx_context * ctx, int i, bool * ok);
// Before a dispatch with these bindings: completes the NPU results that they overlap.
bool ggml_hrx_npu_gate(ggml_backend_hrx_context * ctx, const ggml_cgraph * graph, int i, bool * ok);
bool ggml_hrx_npu_sync(ggml_backend_hrx_context * ctx, const std::vector<hrx_buffer_ref_t> & refs);
// Completes all NPU results in flight.
bool ggml_hrx_npu_finish(ggml_backend_hrx_context * ctx);
void ggml_hrx_npu_state_free(struct ggml_hrx_npu_state * st);
// While a graph is recorded: the NPU work goes into a list of HRX graphs and NPU actions. Launch it before the last graph.
void                         ggml_hrx_npu_record_begin(ggml_backend_hrx_context * ctx);
struct ggml_hrx_npu_replay * ggml_hrx_npu_record_end(ggml_backend_hrx_context * ctx);
bool                         ggml_hrx_npu_replay_launch(ggml_backend_hrx_context * ctx, struct ggml_hrx_npu_replay * r);
void                         ggml_hrx_npu_replay_free(struct ggml_hrx_npu_replay * r);

bool ggml_hrx_supports_flash_attn_ext(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_flash_attn_ext(ggml_backend_hrx_context * ctx, const ggml_tensor * node);
int  ggml_hrx_fuse_flash_attn_bias_sub(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, std::vector<int> * absorbed);
int  ggml_hrx_fuse_flash_attn_bias_mm(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, std::vector<int> * absorbed);
// ops/gated-update.cpp: x - tanh(Wf z) * x + tanh(Wg z) * m with z = Wd [rms_norm(x); rms_norm(m)].
int  ggml_hrx_fuse_gated_update(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, std::vector<int> * absorbed);

// Fusions. Each tries the nodes that start at index i. On a match it returns the index of the last node of the range.
// With absorbed != nullptr it only matches and adds the earlier nodes that it computes itself (they have no other
// readers and do not run). Else it runs the range and sets ctx->dispatch_writes to all tensors that it writes.
// A fusion without absorbed nodes can return -1 in the run (its nodes then run one by one). A failed dispatch after a
// match is fatal (GGML_ABORT). ops/fuse.cpp holds the list; GGML_HRX_DISABLE_FUSION=1 turns
// all of them off.
typedef int (*ggml_hrx_fuse_fn)(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, std::vector<int> * absorbed);

// Matches the fusions of cgraph. Call it before ggml_hrx_autofuse_plan.
void ggml_hrx_fuse_plan(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph);
// If a fusion starts at node i, runs it and returns its last node. name is the label for GGML_HRX_PROFILE_OPS.
int  ggml_hrx_fuse(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, const char ** name);
// True if a later fusion computes node i. Then node i does not run.
bool ggml_hrx_fuse_absorbed(const ggml_backend_hrx_context * ctx, int i);
// The first node from i on that a fusion runs or absorbs.
int  ggml_hrx_fuse_next(const ggml_backend_hrx_context * ctx, int i);
void ggml_hrx_fuse_free(struct ggml_hrx_fusion_plan * plan);

// Helpers for the fusions. A use count of -1 means unknown.
int32_t ggml_hrx_use_count(const ggml_cgraph * g, const ggml_tensor * t);
bool    ggml_hrx_only_used_by(const ggml_cgraph * g, const ggml_tensor * t, int n_uses); // n_uses reads and no output flag
int     ggml_hrx_next_node(const ggml_cgraph * g, int i);                                // next node that runs, or -1
int     ggml_hrx_node_index(const ggml_cgraph * g, const ggml_tensor * t, int last, int max_dist); // index of t in (last - max_dist, last], or -1
bool    ggml_hrx_overlap(const ggml_tensor * a, const ggml_tensor * b);
