#include "ggml-hrx-ops.h"

#include <climits>
#include <cstdlib>

struct ggml_hrx_fusion {
    const char *     name;
    ggml_hrx_fuse_fn fn;
};

// Tried in order. Put a fusion that covers a longer chain before one that covers a part of it.
static const ggml_hrx_fusion ggml_hrx_fusions[] = {
    { "rms_norm_mix",      ggml_hrx_fuse_rms_norm_mix },
    { "ssm_conv_tail",     ggml_hrx_fuse_ssm_conv_tail },
    { "ssm_conv_concat",   ggml_hrx_fuse_ssm_conv_concat },
    { "fa_bias_mm",        ggml_hrx_fuse_flash_attn_bias_mm },
    { "fa_bias_sub",       ggml_hrx_fuse_flash_attn_bias_sub },
    { "gated_update",      ggml_hrx_fuse_gated_update },
    { "mul_mat_vec_group", ggml_hrx_fuse_mul_mat_vec_group },
    { "moe_glu",           ggml_hrx_fuse_moe_glu },
};

static constexpr int     GGML_HRX_N_FUSIONS     = (int) (sizeof(ggml_hrx_fusions) / sizeof(ggml_hrx_fusions[0]));
static constexpr uint8_t GGML_HRX_FUSE_RANGE    = 1; // the node is in the node range of a fusion
static constexpr uint8_t GGML_HRX_FUSE_ABSORBED = 2; // a later fusion computes what the node is used for. The node does not run.

struct ggml_hrx_fusion_plan {
    std::vector<int16_t> fusion; // fusion that starts at node i, or -1
    std::vector<int>     last;   // last node of that fusion
    std::vector<uint8_t> absorbs; // that fusion absorbs earlier nodes
    std::vector<uint8_t> state;
    std::vector<int>     next;   // first node from i on that is in a fusion or absorbed, or n_nodes
};

void ggml_hrx_fuse_plan(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph) {
    static const bool disabled = getenv("GGML_HRX_DISABLE_FUSION") != nullptr;
    if (ctx->fuse == nullptr) {
        ctx->fuse = new ggml_hrx_fusion_plan;
    }
    ggml_hrx_fusion_plan & p = *ctx->fuse;
    const int n = cgraph->n_nodes;
    p.fusion.assign(n, -1);
    p.last.assign(n, -1);
    p.absorbs.assign(n, 0);
    p.state.assign(n, 0);
    p.next.assign(n + 1, n);
    if (disabled) {
        return;
    }
    std::vector<int> absorbed;
    for (int i = 0; i < n; ++i) {
        for (int f = 0; f < GGML_HRX_N_FUSIONS; ++f) {
            absorbed.clear();
            const int last = ggml_hrx_fusions[f].fn(ctx, cgraph, i, &absorbed);
            if (last < i) {
                continue;
            }
            // A node belongs to one fusion at most.
            bool free = true;
            for (int k : absorbed) {
                free = free && k < i && p.state[k] == 0;
            }
            if (!free) {
                continue;
            }
            for (int k : absorbed) {
                p.state[k] = GGML_HRX_FUSE_ABSORBED;
            }
            for (int k = i; k <= last; ++k) {
                p.state[k] = GGML_HRX_FUSE_RANGE;
            }
            p.fusion[i]  = (int16_t) f;
            p.last[i]    = last;
            p.absorbs[i] = !absorbed.empty();
            i           = last;
            break;
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        p.next[i] = p.state[i] != 0 ? i : p.next[i + 1];
    }
}

bool ggml_hrx_fuse_absorbed(const ggml_backend_hrx_context * ctx, int i) {
    return ctx->fuse != nullptr && i < (int) ctx->fuse->state.size() && ctx->fuse->state[i] == GGML_HRX_FUSE_ABSORBED;
}

int ggml_hrx_fuse_next(const ggml_backend_hrx_context * ctx, int i) {
    if (ctx->fuse == nullptr || i >= (int) ctx->fuse->state.size()) {
        return INT_MAX;
    }
    return ctx->fuse->next[i];
}

int ggml_hrx_fuse(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, const char ** name) {
    if (ctx->fuse == nullptr || i >= (int) ctx->fuse->fusion.size() || ctx->fuse->fusion[i] < 0) {
        return -1;
    }
    const ggml_hrx_fusion & f = ggml_hrx_fusions[ctx->fuse->fusion[i]];
    const int last = f.fn(ctx, cgraph, i, nullptr);
    // A fusion can still decline (for example its kernel does not compile). Then the nodes run one by one.
    if (last < 0 && !ctx->fuse->absorbs[i]) {
        return -1;
    }
    if (last != ctx->fuse->last[i]) {
        GGML_ABORT("%s: fusion %s at node %d does not match its plan", __func__, f.name, i);
    }
    *name = f.name;
    return last;
}

void ggml_hrx_fuse_free(ggml_hrx_fusion_plan * plan) {
    delete plan;
}

int32_t ggml_hrx_use_count(const ggml_cgraph * g, const ggml_tensor * t) {
    if (g->use_counts == nullptr || g->visited_hash_set.size == 0) {
        return -1;
    }
    const size_t pos = ggml_hash_find(&g->visited_hash_set, t);
    if (pos == GGML_HASHSET_FULL || !ggml_bitset_get(g->visited_hash_set.used, pos)) {
        return -1;
    }
    return g->use_counts[pos];
}

bool ggml_hrx_only_used_by(const ggml_cgraph * g, const ggml_tensor * t, int n_uses) {
    return (t->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && ggml_hrx_use_count(g, t) == n_uses;
}

int ggml_hrx_next_node(const ggml_cgraph * g, int i) {
    for (int k = i + 1; k < g->n_nodes; ++k) {
        const ggml_tensor * t = g->nodes[k];
        if (!ggml_is_empty(t) && !ggml_op_is_empty(t->op) && (t->flags & GGML_TENSOR_FLAG_COMPUTE) != 0) {
            return k;
        }
    }
    return -1;
}

int ggml_hrx_node_index(const ggml_cgraph * g, const ggml_tensor * t, int last, int max_dist) {
    for (int k = last; k >= 0 && k > last - max_dist; --k) {
        if (g->nodes[k] == t) {
            return k;
        }
    }
    return -1;
}

bool ggml_hrx_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const ggml_backend_buffer_t ba = a->view_src != nullptr ? a->view_src->buffer : a->buffer;
    const ggml_backend_buffer_t bb = b->view_src != nullptr ? b->view_src->buffer : b->buffer;
    if (ba != bb) {
        return false;
    }
    const uint8_t * a0 = (const uint8_t *) a->data;
    const uint8_t * b0 = (const uint8_t *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}
