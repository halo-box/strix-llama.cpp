#pragma once
#include "common.cuh"

// Readers of a conv-history concat that a tail-only materialization can serve: the rollback-slot views of the
// last H columns, their CONTs, and copies of either. Anything else reading the concat needs the whole tensor.
static inline int64_t ggml_cuda_conv_tail_view_col(const ggml_tensor * t, const ggml_tensor * cc, int64_t H) {
    if (t->op != GGML_OP_VIEW || t->view_src != cc || t->type != GGML_TYPE_F32 ||
        t->ne[0] != H || t->ne[1] != cc->ne[1] || t->ne[2] != cc->ne[2] || t->ne[3] != 1 ||
        t->nb[1] != cc->nb[1] || t->nb[2] != cc->nb[2] || t->view_offs % sizeof(float) != 0) {
        return -1;
    }
    const int64_t col = (int64_t) (t->view_offs / sizeof(float));
    return (col >= 0 && col + H <= cc->ne[0]) ? col : -1;
}

static inline bool ggml_cuda_conv_tail_reader(const ggml_tensor * t, const ggml_tensor * cc, int64_t H, int64_t & tail_from) {
    if (t->op == GGML_OP_VIEW) {
        const int64_t col = ggml_cuda_conv_tail_view_col(t, cc, H);
        if (col < 0) {
            return false;
        }
        tail_from = std::min(tail_from, col);
        return true;
    }
    if (t->op == GGML_OP_CONT) {
        return t->src[0] && ggml_cuda_conv_tail_view_col(t->src[0], cc, H) >= 0;
    }
    if (t->op == GGML_OP_CPY) {
        const ggml_tensor * s = t->src[0];
        if (!s) {
            return false;
        }
        if (s->op == GGML_OP_CONT) {
            s = s->src[0];
        }
        return s && ggml_cuda_conv_tail_view_col(s, cc, H) >= 0;
    }
    return false;
}

static inline bool ggml_cuda_reads(const ggml_tensor * t, const ggml_tensor * root) {
    for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) {
        if (t->src[s] == root || t->src[s]->view_src == root) {
            return true;
        }
    }
    return false;
}

// the concat match is dispatched as "this node plus the next": the next node must be a no-op the loop skips anyway
static inline bool ggml_cuda_conv_next_is_noop(const ggml_cgraph * cgraph, int i) {
    if (i + 1 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * t = cgraph->nodes[i + 1];
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// GGML_CONV_FUSE_DEBUG=1 prints why a concat did not match, with the node index the decision was made at
static inline bool ggml_cuda_conv_reject(const char * which, const char * why, int node) {
    static const bool dbg = getenv("GGML_CONV_FUSE_DEBUG") != nullptr;
    if (dbg) {
        fprintf(stderr, "%s: no fusion, %s (node %d)\n", which, why, node);
    }
    return false;
}

// nodes after `from` (inclusive) that read t, other than t itself and `except`: a scan, so it works on graph
// copies and views whose use-count tables are not populated
static inline int ggml_cuda_conv_readers(const ggml_cgraph * cgraph, int from, const ggml_tensor * t, const ggml_tensor * except) {
    int n = 0;
    for (int k = from; k < cgraph->n_nodes; ++k) {
        const ggml_tensor * r = cgraph->nodes[k];
        if (r == t || r == except || r->op == GGML_OP_RESHAPE || r->op == GGML_OP_VIEW || r->op == GGML_OP_TRANSPOSE || r->op == GGML_OP_PERMUTE) {
            continue; // a view of t is not a consumer: whoever reads the view is counted through view_src
        }
        if (ggml_cuda_reads(r, t)) {
            ++n;
        }
    }
    return n;
}

// the whole concat, either directly or through a view that covers all of it
static inline bool ggml_cuda_conv_whole(const ggml_tensor * s, const ggml_tensor * cc) {
    return s == cc || (s && s->view_src == cc && s->view_offs == 0 && ggml_nelements(s) == ggml_nelements(cc) && ggml_is_contiguous(s));
}

// concat src[1] is TRANSPOSE(view of x): x must be the whole contiguous storage of a [C, T] activation, whatever
// shape its producer gave it (qwen4exp's grouped norm leaves it [n_embd, hc, T]); C and T come from the transpose
static inline const ggml_tensor * ggml_cuda_conv_activation(const ggml_tensor * tr, int64_t & C, int64_t & T) {
    if (!tr || tr->op != GGML_OP_TRANSPOSE || tr->type != GGML_TYPE_F32 || !tr->view_src || tr->view_offs != 0 ||
        tr->ne[2] != 1 || tr->ne[3] != 1) {
        return nullptr;
    }
    T = tr->ne[0];
    C = tr->ne[1];
    const ggml_tensor * x = tr->view_src;
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || ggml_nelements(x) != C * T ||
        tr->nb[1] != sizeof(float) || tr->nb[0] != (size_t) C * sizeof(float)) {
        return nullptr;
    }
    return x;
}
