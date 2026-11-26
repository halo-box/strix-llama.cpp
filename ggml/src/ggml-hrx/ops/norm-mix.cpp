#include "ggml-hrx-ops.h"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <string>

// A softmax-weighted mix of RMS-normalized rows, as one dispatch:
//   stack  = CONCAT(r_0, CONCAT(r_1, ... CONCAT(r_{R-2}, r_{R-1}), dim 1), dim 1)   rows r_k of [ne0, 1, T]
//   norm   = RMS_NORM(stack)                                                          [ne0, R, T]
//   probs  = SOFT_MAX(MUL_MAT(norm, REPEAT(query, [ne0, nq, T])), scale)             [R, nq, T]
//   out    = MUL_MAT(CONT(TRANSPOSE(norm)), probs)                                    [ne0, nq, T]
// With R = 1 there is no CONCAT. The intermediate results must have no other readers.

// Row and query counts of the kernel (kernels/norm/rms_norm_mix_f32.loom).
static constexpr int     GGML_HRX_NORM_MIX_MAX_ROWS = 9;
static constexpr int64_t GGML_HRX_NORM_MIX_MAX_NQ   = 4;
// Each lane keeps up to 4 vec4 of each row. ne0 = 2304 uses 192 lanes with 3 vec4: 7.4 us for each token at decode
// and 55 ms for 120 mixes of 2048 tokens; 576 lanes with 1 vec4 took 8.3 us and 57 ms (gfx1151, device timestamps).
static constexpr int64_t GGML_HRX_NORM_MIX_LANE_CHUNKS = 256;

struct ggml_hrx_norm_mix {
    const ggml_tensor * rows[GGML_HRX_NORM_MIX_MAX_ROWS] = {}; // in stack order
    int                 n_rows   = 0;
    const ggml_tensor * norm     = nullptr;
    const ggml_tensor * query    = nullptr;
    const ggml_tensor * soft_max = nullptr;
    const ggml_tensor * out      = nullptr;
    int                 last     = -1;
};

static int32_t ggml_hrx_norm_mix_uses(const ggml_cgraph * g, const ggml_tensor * t) {
    if (g->use_counts == nullptr || g->visited_hash_set.size == 0) {
        return -1;
    }
    const size_t pos = ggml_hash_find(&g->visited_hash_set, t);
    if (pos == GGML_HASHSET_FULL || !ggml_bitset_get(g->visited_hash_set.used, pos)) {
        return -1;
    }
    return g->use_counts[pos];
}

// An intermediate result: n readers in the pattern and none outside.
static bool ggml_hrx_norm_mix_inner(const ggml_cgraph * g, const ggml_tensor * t, int32_t n) {
    return (t->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && ggml_hrx_norm_mix_uses(g, t) == n;
}

static int ggml_hrx_norm_mix_next(const ggml_cgraph * g, int j) {
    for (; j < g->n_nodes; ++j) {
        const ggml_tensor * t = g->nodes[j];
        if (!ggml_is_empty(t) && !ggml_op_is_empty(t->op) && (t->flags & GGML_TENSOR_FLAG_COMPUTE) != 0) {
            return j;
        }
    }
    return -1;
}

static bool ggml_hrx_norm_mix_row_ok(const ggml_tensor * r, int64_t ne0, int64_t T) {
    return r->type == GGML_TYPE_F32 && r->ne[0] == ne0 && r->ne[1] == 1 && r->ne[2] == T && r->ne[3] == 1 && ggml_is_contiguous(r) &&
           ggml_hrx_tensor_on_device(r);
}

static bool ggml_hrx_norm_mix_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const ggml_backend_buffer_t ba = a->view_src != nullptr ? a->view_src->buffer : a->buffer;
    const ggml_backend_buffer_t bb = b->view_src != nullptr ? b->view_src->buffer : b->buffer;
    if (ba != bb) {
        return false;
    }
    const uint8_t * a0 = (const uint8_t *) a->data;
    const uint8_t * b0 = (const uint8_t *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

static bool ggml_hrx_norm_mix_match(const ggml_cgraph * g, int i, ggml_hrx_norm_mix & m) {
    const ggml_tensor * t = g->nodes[i];
    if (t->op != GGML_OP_CONCAT && t->op != GGML_OP_RMS_NORM) {
        return false;
    }
    const ggml_tensor * first = t->op == GGML_OP_CONCAT ? t->src[1] : t->src[0];
    const int64_t       ne0   = first->ne[0];
    const int64_t       T     = first->ne[2];
    if (!ggml_hrx_norm_mix_row_ok(first, ne0, T)) {
        return false;
    }

    // The CONCAT chain puts each new row in front of the stack. rows gets the inner row first.
    const ggml_tensor * rows[GGML_HRX_NORM_MIX_MAX_ROWS];
    int                 n_rows = 0;
    rows[n_rows++]             = first;
    const ggml_tensor * stack  = first;
    int                 j      = i;
    while (t->op == GGML_OP_CONCAT) {
        if (t->type != GGML_TYPE_F32 || ggml_get_op_params_i32(t, 0) != 1 || t->src[1] != stack || n_rows >= GGML_HRX_NORM_MIX_MAX_ROWS ||
            !ggml_hrx_norm_mix_row_ok(t->src[0], ne0, T) || (stack != first && !ggml_hrx_norm_mix_inner(g, stack, 1))) {
            return false;
        }
        rows[n_rows++] = t->src[0];
        stack          = t;
        j              = ggml_hrx_norm_mix_next(g, j + 1);
        if (j < 0) {
            return false;
        }
        t = g->nodes[j];
    }
    const ggml_tensor * norm = t;
    if (norm->op != GGML_OP_RMS_NORM || norm->type != GGML_TYPE_F32 || norm->src[0] != stack || norm->ne[1] != n_rows ||
        (stack != first && !ggml_hrx_norm_mix_inner(g, stack, 1)) || !ggml_hrx_norm_mix_inner(g, norm, 2)) {
        return false;
    }
    float eps;
    memcpy(&eps, norm->op_params, sizeof(float));
    if (!(eps >= 0.0f)) {
        return false;
    }

    // The other nodes, in either order of the two MUL_MAT operands. The second MUL_MAT is the last node.
    const ggml_tensor * cont = nullptr;
    const ggml_tensor * rep  = nullptr;
    const ggml_tensor * mm_s = nullptr;
    const ggml_tensor * sm   = nullptr;
    for (int k = 0; k < 4; ++k) {
        j = ggml_hrx_norm_mix_next(g, j + 1);
        if (j < 0) {
            return false;
        }
        const ggml_tensor * n = g->nodes[j];
        const ggml_tensor ** slot = n->op == GGML_OP_CONT ? &cont : n->op == GGML_OP_REPEAT ? &rep : n->op == GGML_OP_MUL_MAT ? &mm_s :
                                    n->op == GGML_OP_SOFT_MAX ? &sm : nullptr;
        if (slot == nullptr || *slot != nullptr) {
            return false;
        }
        *slot = n;
    }
    j = ggml_hrx_norm_mix_next(g, j + 1);
    if (j < 0) {
        return false;
    }
    const ggml_tensor * out = g->nodes[j];
    const ggml_tensor * tr  = cont->src[0];
    const ggml_tensor * q   = rep->src[0];
    const int64_t       nq  = q->ne[1];
    if (tr->op != GGML_OP_TRANSPOSE || tr->src[0] != norm || !ggml_hrx_norm_mix_inner(g, tr, 1) || cont->type != GGML_TYPE_F32 ||
        !ggml_hrx_norm_mix_inner(g, cont, 1)) {
        return false;
    }
    if (q->type != GGML_TYPE_F32 || q->ne[0] != ne0 || nq > GGML_HRX_NORM_MIX_MAX_NQ || q->ne[2] != 1 || q->ne[3] != 1 || q->nb[0] != 4 ||
        q->nb[1] % 4 != 0 || !ggml_hrx_tensor_on_device(q) || rep->type != GGML_TYPE_F32 || rep->ne[0] != ne0 || rep->ne[1] != nq ||
        rep->ne[2] != T || rep->ne[3] != 1 || !ggml_hrx_norm_mix_inner(g, rep, 1)) {
        return false;
    }
    if (mm_s->src[0] != norm || mm_s->src[1] != rep || mm_s->type != GGML_TYPE_F32 || !ggml_hrx_norm_mix_inner(g, mm_s, 1)) {
        return false;
    }
    float max_bias;
    memcpy(&max_bias, (const float *) sm->op_params + 1, sizeof(float));
    if (sm->src[0] != mm_s || sm->src[1] != nullptr || sm->src[2] != nullptr || max_bias != 0.0f || sm->type != GGML_TYPE_F32 ||
        !ggml_hrx_norm_mix_inner(g, sm, 1)) {
        return false;
    }
    if (out->op != GGML_OP_MUL_MAT || out->src[0] != cont || out->src[1] != sm || out->type != GGML_TYPE_F32 || out->ne[0] != ne0 ||
        out->ne[1] != nq || out->ne[2] != T || out->ne[3] != 1 || !ggml_is_contiguous(out) || !ggml_hrx_tensor_on_device(out)) {
        return false;
    }
    if (ne0 < 4 || ne0 % 4 != 0 || ne0 / 4 > 4 * 1024 || T < 1 || ggml_nbytes(out) / 4 >= INT_MAX - 4 || ggml_nbytes(q) / 4 >= INT_MAX - 4 ||
        ne0 * T >= INT_MAX - 4) {
        return false;
    }
    // The kernel writes out while it reads the rows and the query.
    if (ggml_hrx_norm_mix_overlap(out, q)) {
        return false;
    }
    for (int k = 0; k < n_rows; ++k) {
        if (ggml_hrx_norm_mix_overlap(out, rows[k])) {
            return false;
        }
        m.rows[k] = rows[n_rows - 1 - k];
    }
    m.n_rows   = n_rows;
    m.norm     = norm;
    m.query    = q;
    m.soft_max = sm;
    m.out      = out;
    m.last     = j;
    return true;
}

int ggml_hrx_fuse_rms_norm_mix(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, std::vector<int> * absorbed) {
    ggml_hrx_norm_mix m;
    if (!ggml_hrx_norm_mix_match(cgraph, i, m)) {
        return -1;
    }
    if (absorbed != nullptr) {
        return m.last;
    }
    const int64_t ne0    = m.out->ne[0];
    const int64_t nq     = m.out->ne[1];
    const int64_t T      = m.out->ne[2];
    const int64_t chunks = ne0 / 4;
    const int64_t vecs   = std::min<int64_t>(4, (chunks + GGML_HRX_NORM_MIX_LANE_CHUNKS - 1) / GGML_HRX_NORM_MIX_LANE_CHUNKS);
    const int64_t lanes  = ((chunks + vecs - 1) / vecs + 63) / 64 * 64;

    uint32_t eps_bits;
    uint32_t scale_bits;
    memcpy(&eps_bits, m.norm->op_params, sizeof(uint32_t));
    memcpy(&scale_bits, m.soft_max->op_params, sizeof(uint32_t));

    ggml_hrx_kernel_desc desc;
    desc.file   = "norm/rms_norm_mix_f32.loom";
    desc.symbol = "ggml_rms_norm_mix_f32";
    desc.config = {
        { "ggml.norm_mix.rows", std::to_string(m.n_rows) },
        { "ggml.norm_mix.nq", std::to_string(nq) },
        { "ggml.norm_mix.lanes", std::to_string(lanes) },
        { "ggml.norm_mix.vecs", std::to_string(vecs) },
    };
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
    if (kernel == nullptr) {
        return -1;
    }

    std::vector<const ggml_tensor *> buffers(m.rows, m.rows + GGML_HRX_NORM_MIX_MAX_ROWS);
    for (int k = m.n_rows; k < GGML_HRX_NORM_MIX_MAX_ROWS; ++k) {
        buffers[k] = m.rows[0];
    }
    buffers.push_back(m.query);
    buffers.push_back(m.out);
    const std::vector<int64_t> scalars = {
        ne0, ne0 * T, (int64_t) (m.query->nb[1] / 4), (int64_t) (ggml_nbytes(m.query) / 4), (int64_t) (ggml_nbytes(m.out) / 4),
        (int64_t) eps_bits, (int64_t) scale_bits,
    };

    // The dispatch writes only out. The rows and the query do not overlap it.
    const std::vector<const ggml_tensor *> writes = { m.out };
    ctx->dispatch_writes = &writes;
    const bool ok = ggml_hrx_dispatch(ctx, kernel, { T }, scalars, buffers);
    ctx->dispatch_writes = nullptr;
    if (!ok) {
        GGML_ABORT("%s: failed to run nodes %d..%d (%s)", __func__, i, m.last, m.out->name);
    }
    return m.last;
}
