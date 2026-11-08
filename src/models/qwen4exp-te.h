#pragma once

// The rows of a non-compact QSA ubatch (PLAN-v3 §4.1 row 4, W8): PR219's TE set, top_k(expanded, width), with every cell
// the visibility hides set to -1, so the rows carry the visibility and GGML_OP_SPARSE_ATTN needs no mask.
//
// expanded is the per-cell score: the block score with its per-block bias, gathered to the cells by cell_blk, plus the
// strip's KQ mask (0 visible, -inf hidden), or plus the per-cell bias when no mask is used (that bias then carries the
// visibility itself, llama_memory_hybrid_idx::set_input_qsa_scan). PR219's masked route attended
// top_k(expanded) ∩ {mask == 0}. expanded is -inf exactly where the mask is -inf, because the scores and the per-block bias
// are finite on every visible cell: a cell's block has the cell's own sequence set (the scan groups cells by exact
// sequence set, or holds one sequence), so its bias is 0 or 1e9 there, never -inf. Gathering expanded at the picked
// indices therefore gives that set without a second read of the mask. tests/test-qsa-te-rows.cpp checks it against
// PR219's masked-route ops on the r1_oracle layouts.
//
// Shared by llama_model_qwen4exp::graph::build_qsa_top_k and tests/test-qsa-te-rows.cpp, so the test runs the
// production ops.

#include "ggml.h"

#include <cfloat>

// score: F32 [n_blocks, n_query, n_stream], the per-block bias already added; cell_blk: I32 [n_kv, n_stream];
// add: F32 [n_kv, n_query, n_stream], the strip's KQ mask or the per-cell bias. Returns F32 [n_kv, n_query, n_stream].
// Every token of a block gets the block score; the budget is whole blocks, so top-k cuts on a block boundary.
static inline ggml_tensor * qwen4exp_te_expand(ggml_context * ctx, ggml_tensor * score, ggml_tensor * cell_blk,
        ggml_tensor * add) {
    ggml_tensor * expanded = ggml_get_rows(ctx, ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3)), cell_blk);
    expanded = ggml_cont(ctx, ggml_permute(ctx, expanded, 1, 0, 2, 3));
    return ggml_add(ctx, expanded, add);
}

struct qwen4exp_te_sel {
    ggml_tensor * top_k = nullptr;   // I32 [width, n_query, 1, n_stream]: the indices top_k(expanded, width) picks
    ggml_tensor * rows  = nullptr;   // I32 [width, n_query, 1, n_stream]: the same index where expanded is finite, else -1
};

// expanded: F32 [n_kv, n_query, n_stream], contiguous. width = min(n_kv, indexer_top_k + ratio - 1).
static inline qwen4exp_te_sel qwen4exp_te_select(ggml_context * ctx, ggml_tensor * expanded, int64_t width) {
    const int64_t n_kv = expanded->ne[0], n_query = expanded->ne[1], n_stream = expanded->ne[2];
    GGML_ASSERT(expanded->type == GGML_TYPE_F32 && ggml_is_contiguous(expanded) && expanded->ne[3] == 1);
    GGML_ASSERT(width >= 1 && width <= n_kv);
    // the f32 index below is exact up to 2^24 cells, the QSA ceiling (PLAN-v3 §4.4)
    GGML_ASSERT(n_kv <= ((int64_t) 1 << 24));

    qwen4exp_te_sel sel;
    sel.top_k = ggml_cont(ctx, ggml_top_k(ctx, expanded, width));
    sel.top_k = ggml_reshape_4d(ctx, sel.top_k, width, n_query, 1, n_stream);

    // one value per (picked cell, query, stream): expanded viewed as rows of one element, the indices as one row of
    // width per (query, stream)
    ggml_tensor * values = ggml_get_rows(ctx, ggml_reshape_4d(ctx, expanded, 1, n_kv, n_query, n_stream),
            ggml_reshape_3d(ctx, sel.top_k, width, n_query, n_stream));
    values = ggml_reshape_4d(ctx, values, width, n_query, 1, n_stream);
    // 1 for every finite value (x + FLT_MAX > 0 for any finite x > -FLT_MAX), 0 for -inf (and NaN)
    ggml_tensor * visible = ggml_step(ctx, ggml_scale_bias(ctx, values, 1.0f, FLT_MAX));
    ggml_tensor * idx1 = ggml_scale_bias(ctx, ggml_cast(ctx, sel.top_k, GGML_TYPE_F32), 1.0f, 1.0f);   // index + 1
    ggml_tensor * rows = ggml_scale_bias(ctx, ggml_mul(ctx, idx1, visible), 1.0f, -1.0f);             // index or -1
    sel.rows = ggml_cast(ctx, rows, GGML_TYPE_I32);
    return sel;
}
