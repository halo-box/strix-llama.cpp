#pragma once

// The rows of a non-compact QSA ubatch: top_k(expanded, width) with every cell the visibility hides set to -1, so
// GGML_OP_SPARSE_ATTN needs no mask. Shared by the qwen4exp graph and the tests.
//
// Visibility is read from the mask source (vis), not from expanded: expanded must keep the TOP_K as its only consumer,
// or the Vulkan TOPK_QSA fusion does not fire.

#include "ggml.h"

#include <cfloat>

// score: F32 [n_blocks, n_query, n_stream], per-block bias added; cell_blk: I32 [n_kv, n_stream]; add: F32
// [n_kv, n_query, n_stream], the KQ mask or the per-cell bias. Returns F32 [n_kv, n_query, n_stream].
static inline ggml_tensor * qwen4exp_te_expand(ggml_context * ctx, ggml_tensor * score, ggml_tensor * cell_blk,
        ggml_tensor * add) {
    ggml_tensor * expanded = ggml_get_rows(ctx, ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3)), cell_blk);
    expanded = ggml_cont(ctx, ggml_permute(ctx, expanded, 1, 0, 2, 3));
    return ggml_add(ctx, expanded, add);
}

struct qwen4exp_te_sel {
    ggml_tensor * top_k = nullptr;   // I32 [width, n_query, 1, n_stream]: the indices top_k(expanded, width) picks
    ggml_tensor * rows  = nullptr;   // I32 [width, n_query, 1, n_stream]: the same index where vis is finite, else -1
};

// expanded: F32 [n_kv, n_query, n_stream], contiguous. width = min(n_kv, indexer_top_k + ratio - 1).
// vis: the source of expanded's ADD before any cast: KQ mask (F16 / F32 [n_kv, n_query, 1, n_stream]) or per-cell bias
// (F32 [n_kv, n_query, n_stream]), elements contiguous. A cell is visible iff vis is finite there.
static inline qwen4exp_te_sel qwen4exp_te_select(ggml_context * ctx, ggml_tensor * expanded, ggml_tensor * vis, int64_t width) {
    const int64_t n_kv = expanded->ne[0], n_query = expanded->ne[1], n_stream = expanded->ne[2];
    GGML_ASSERT(expanded->type == GGML_TYPE_F32 && ggml_is_contiguous(expanded) && expanded->ne[3] == 1);
    GGML_ASSERT(width >= 1 && width <= n_kv);
    // the f32 index below is exact up to 2^24 cells
    GGML_ASSERT(n_kv <= ((int64_t) 1 << 24));
    GGML_ASSERT(vis != nullptr && (vis->type == GGML_TYPE_F32 || vis->type == GGML_TYPE_F16));
    GGML_ASSERT(vis->ne[0] == n_kv && vis->ne[1] == n_query && vis->ne[2] * vis->ne[3] == n_stream);
    GGML_ASSERT(vis->nb[0] == ggml_type_size(vis->type));
    // the stream stride: ne[3] of the 4-D mask, ne[2] of the 3-D bias
    const size_t nb_stream = vis->ne[3] == n_stream ? vis->nb[3] : vis->nb[2];

    qwen4exp_te_sel sel;
    sel.top_k = ggml_cont(ctx, ggml_top_k(ctx, expanded, width));
    sel.top_k = ggml_reshape_4d(ctx, sel.top_k, width, n_query, 1, n_stream);

    // vis as rows of one element, gathered at the picked indices; an F16 source gives F32
    ggml_tensor * vis_cells = ggml_view_4d(ctx, vis, 1, n_kv, n_query, n_stream, vis->nb[0], vis->nb[1], nb_stream, 0);
    ggml_tensor * values = ggml_get_rows(ctx, vis_cells, ggml_reshape_3d(ctx, sel.top_k, width, n_query, n_stream));
    values = ggml_reshape_4d(ctx, values, width, n_query, 1, n_stream);
    // 1 for every finite value (x + FLT_MAX > 0 for any finite x > -FLT_MAX), 0 for -inf (and NaN)
    ggml_tensor * visible = ggml_step(ctx, ggml_scale_bias(ctx, values, 1.0f, FLT_MAX));
    ggml_tensor * idx1 = ggml_scale_bias(ctx, ggml_cast(ctx, sel.top_k, GGML_TYPE_F32), 1.0f, 1.0f);   // index + 1
    ggml_tensor * rows = ggml_scale_bias(ctx, ggml_mul(ctx, idx1, visible), 1.0f, -1.0f);             // index or -1
    sel.rows = ggml_cast(ctx, rows, GGML_TYPE_I32);
    return sel;
}
