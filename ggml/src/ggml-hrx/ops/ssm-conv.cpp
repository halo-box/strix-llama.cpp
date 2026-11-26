#include "ggml-hrx-ops.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <string>

static constexpr int64_t GGML_HRX_SSM_CONV_DIRECT_MAX_TOKENS = 16;

bool ggml_hrx_supports_ssm_conv(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * x = op->src[0];
    const ggml_tensor * w = op->src[1];
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
        return false;
    }
    // The tiled kernel keeps nc - 1 + 32 values of each x row in workgroup memory.
    if ((w->ne[0] > 33 && op->ne[1] > GGML_HRX_SSM_CONV_DIRECT_MAX_TOKENS) || x->ne[3] != 1 || w->ne[2] != 1 || w->ne[3] != 1 || x->ne[1] != w->ne[1] || op->ne[0] != x->ne[1] ||
        op->ne[1] != x->ne[0] - w->ne[0] + 1 || op->ne[2] != x->ne[2]) {
        return false;
    }
    int64_t xs[GGML_MAX_DIMS];
    int64_t ws[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(x, xs) || !ggml_hrx_elem_strides(w, ws)) {
        return false;
    }
    // Sizes, counts and spans are 32-bit launch scalars.
    return ggml_nelements(op) < INT_MAX && ggml_nbytes(x) / 4 < INT_MAX && ggml_nbytes(w) / 4 < INT_MAX;
}

bool ggml_hrx_op_ssm_conv(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * x = node->src[0];
    const ggml_tensor * w = node->src[1];
    const int64_t n = ggml_nelements(node);
    if (n == 0) {
        return true;
    }

    int64_t xs[GGML_MAX_DIMS];
    int64_t ws[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(x, xs);
    ggml_hrx_elem_strides(w, ws);

    const int64_t channels = node->ne[0];
    const int64_t n_t      = node->ne[1];

    // With few tokens, the x rows are short and the direct kernel reads them in few cache lines.
    // gfx1151, 2304 channels, nc 4 (test-backend-ops perf): n_t 1: direct 1.4 us, tiled 3.4 us. n_t 16: 4.1 vs 5.4 us.
    // n_t 32: 10.6 vs 7.5 us. n_t 512: 157 vs 54 us.
    if (n_t <= GGML_HRX_SSM_CONV_DIRECT_MAX_TOKENS) {
        ggml_hrx_kernel_desc desc;
        desc.file   = "ssm/ssm_conv_f32.loom";
        desc.symbol = "ggml_ssm_conv_f32_direct";
        desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
        ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
        return ggml_hrx_dispatch(ctx, kernel, { n },
            {
                n, channels, n_t, w->ne[0],
                xs[0], xs[1], xs[2], (int64_t) (ggml_nbytes(x) / 4),
                ws[0], ws[1], (int64_t) (ggml_nbytes(w) / 4),
            },
            { x, w, node });
    }

    // Tiles of 64 channels and 32 tokens. Must match ssm/ssm_conv_f32.loom.
    const int64_t n_cb     = (channels + 63) / 64;
    const int64_t n_tb     = (n_t + 31) / 32;

    ggml_hrx_kernel_desc desc;
    desc.file   = "ssm/ssm_conv_f32.loom";
    desc.symbol = "ggml_ssm_conv_f32";
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    return ggml_hrx_dispatch(ctx, kernel, { n_cb * n_tb * node->ne[2] },
        {
            channels, n_t, n_cb, n_tb, w->ne[0], x->ne[0],
            xs[0], xs[1], xs[2], (int64_t) (ggml_nbytes(x) / 4),
            ws[0], ws[1], (int64_t) (ggml_nbytes(w) / 4), n,
        },
        { x, w, node });
}

// SSM_CONV of x = CONCAT(past, b) along dim 0, b the transpose of cur (stride 1 along the channels):
//   out = SSM_CONV(x, w)                                            [C, T, S]
// and if out is read only by y = SCALE(ADD(cur, RESHAPE(out))) right after it, y instead, or ADD(r, y) if that comes right
// after y and is its only reader. With per-token taps, y = SCALE(ADD(cur, out + sum over l of window_l * tap_l)), window l a
// view of x from column nc - 1 - l: its range starts after the last other node in between, the chain nodes before it
// (the SSM_CONV, a CONT) are absorbed. Two fusions share one match. At the CONCAT only the columns of x that are read
// later are written: past, and the last columns (the conv state store). At the range, a halo kernel copies the first
// inputs of each tile of tokens to scratch, then the conv kernel reads that and cur. x and cur stay allocated until the
// range: x has readers in it, cur is read by the ADD. Thus dst can be cur itself.

// Fewer tokens stay in auto-fusion (AF_SSM_CONV_MAX_TOKENS in ops/autofuse.cpp).
static constexpr int64_t GGML_HRX_SCONV_MIN_TOKENS = 17;
static constexpr int64_t GGML_HRX_SCONV_MAX_NC     = 8;
// The SSM_CONV and the views of x are at most this many nodes after the CONCAT.
static constexpr int     GGML_HRX_SCONV_MAX_DIST   = 512;
// The tail kernel writes the columns of x that other nodes read one after the other.
static constexpr int64_t GGML_HRX_SCONV_MAX_TAIL   = 64;

struct ggml_hrx_sconv_match {
    const ggml_tensor * x       = nullptr; // CONCAT
    const ggml_tensor * conv    = nullptr;
    const ggml_tensor * scale   = nullptr; // or nullptr
    const ggml_tensor * add2    = nullptr; // or nullptr
    const ggml_tensor * resid   = nullptr; // the other source of add2
    const ggml_tensor * dyn     = nullptr; // the tensor that the taps view, or nullptr
    int64_t             d_base  = 0;       // tap 0 of head 0, token 0 in dyn, and the strides of the taps, in elements
    int64_t             d_div   = 4;       // channels of a head
    int64_t             d_s1    = 0;
    int64_t             d_s2    = 0;
    int                 k_conv  = -1;      // node indices
    int                 k_first = -1;      // first node of the range
    int                 k_last  = -1;
    int64_t             x_first = 0;       // first column of x that other nodes read
    std::vector<int>    absorbed;
};

// Columns [j0, j1] of x (rows of length W, contiguous) that view v reads.
static bool ggml_hrx_sconv_view_cols(const ggml_tensor * v, int64_t W, int64_t & j0, int64_t & j1) {
    const size_t row = (size_t) W * sizeof(float);
    if (v->type != GGML_TYPE_F32 || v->view_offs % sizeof(float) != 0) {
        return false;
    }
    j0 = (int64_t) (v->view_offs % row) / (int64_t) sizeof(float);
    j1 = j0;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (v->ne[d] <= 1) {
            continue;
        }
        if (v->nb[d] < row && v->nb[d] % sizeof(float) == 0) {
            j1 += (v->ne[d] - 1) * (int64_t) (v->nb[d] / sizeof(float));
        } else if (v->nb[d] % row != 0) {
            return false;
        }
    }
    return j1 < W;
}

// b is a view of the data of t, from its first element on.
static bool ggml_hrx_sconv_same_data(const ggml_tensor * t, const ggml_tensor * b) {
    const ggml_tensor * root = t->view_src != nullptr ? t->view_src : t;
    const size_t        offs = t->view_src != nullptr ? t->view_offs : 0;
    return b->view_src == root && b->view_offs == offs;
}

static bool ggml_hrx_sconv_f32(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && t->nb[0] % sizeof(float) == 0 && t->nb[1] % sizeof(float) == 0 && t->nb[2] % sizeof(float) == 0 &&
           t->nb[3] % sizeof(float) == 0;
}

// The first node after k that is op and reads src (ordered: only the next node that runs).
static int ggml_hrx_sconv_user(const ggml_cgraph * g, int k, int k_max, ggml_op op, const ggml_tensor * src) {
    for (int j = k + 1; j < g->n_nodes && j <= k_max; ++j) {
        const ggml_tensor * n = g->nodes[j];
        if (n->op == op && (n->src[0] == src || n->src[1] == src)) {
            return j;
        }
    }
    return -1;
}

// cur of y = SCALE(ADD(cur, .)): b is its transpose.
static bool ggml_hrx_sconv_cur_ok(const ggml_tensor * cur, const ggml_tensor * b) {
    const ggml_tensor * b0 = b->src[0];
    const int64_t       T  = b->ne[0];
    const int64_t       C  = b->ne[1];
    const int64_t       S  = b->ne[2];
    return cur != nullptr && (cur == b0 || (b0 != nullptr && cur == b0->src[0])) && ggml_hrx_sconv_same_data(cur, b) &&
           cur->type == GGML_TYPE_F32 && cur->ne[0] == C && cur->ne[1] == T * S && cur->ne[2] == 1 && cur->ne[3] == 1 &&
           cur->nb[0] == sizeof(float) && cur->nb[1] == b->nb[0] && (S == 1 || b->nb[2] == (size_t) T * b->nb[0]);
}

// The per-token taps after the SSM_CONV: for l < nc, acc = ADD(acc, RESHAPE(MUL(RESHAPE(CONT(TRANSPOSE(VIEW(x)))), tap_l))),
// from acc = RESHAPE(conv), then SCALE(ADD(cur, acc)). views gets the views of x that the windows read.
static bool ggml_hrx_sconv_match_dyn(const ggml_cgraph * g, int i, ggml_hrx_sconv_match & m, std::vector<const ggml_tensor *> & views) {
    const ggml_tensor * x     = m.x;
    const ggml_tensor * b     = x->src[1];
    const int64_t       P     = x->src[0]->ne[0];
    const int64_t       T     = b->ne[0];
    const int64_t       C     = b->ne[1];
    const int64_t       S     = b->ne[2];
    const int           k_max = i + GGML_HRX_SCONV_MAX_DIST;
    const int           k_rs  = ggml_hrx_sconv_user(g, m.k_conv, k_max, GGML_OP_RESHAPE, m.conv);
    if (k_rs < 0 || !ggml_hrx_only_used_by(g, m.conv, 1)) {
        return false;
    }
    std::vector<int>    chain = { m.k_conv };
    const ggml_tensor * acc   = g->nodes[k_rs];
    const ggml_tensor * tap0  = nullptr;
    int                 k     = k_rs;
    if (acc->ne[0] != C || acc->ne[1] != T * S || acc->ne[2] != 1) {
        return false;
    }
    for (int64_t l = 0; l <= P; ++l) {
        const int k_a = ggml_hrx_sconv_user(g, k, k_max, GGML_OP_ADD, acc);
        if (k_a < 0 || !ggml_hrx_only_used_by(g, acc, 1)) {
            return false;
        }
        const ggml_tensor * a    = g->nodes[k_a];
        const ggml_tensor * term = a->src[0] == acc ? a->src[1] : a->src[0];
        const ggml_tensor * mul  = term->src[0];
        if (a->type != GGML_TYPE_F32 || !ggml_is_contiguous(a) || !ggml_are_same_shape(a, acc) || term->op != GGML_OP_RESHAPE ||
            mul == nullptr || mul->op != GGML_OP_MUL || !ggml_hrx_only_used_by(g, term, 1) || !ggml_hrx_only_used_by(g, mul, 1)) {
            return false;
        }
        const ggml_tensor * w3   = mul->src[0];
        const ggml_tensor * tap  = mul->src[1];
        const ggml_tensor * cont = w3->src[0];
        const ggml_tensor * tr   = cont != nullptr ? cont->src[0] : nullptr;
        const ggml_tensor * wv   = tr != nullptr ? tr->src[0] : nullptr;
        if (w3->op != GGML_OP_RESHAPE || cont == nullptr || cont->op != GGML_OP_CONT || tr == nullptr || tr->op != GGML_OP_TRANSPOSE ||
            wv == nullptr || wv->op != GGML_OP_VIEW || wv->src[0] != x || !ggml_hrx_only_used_by(g, w3, 1) ||
            !ggml_hrx_only_used_by(g, cont, 1) || !ggml_hrx_only_used_by(g, tr, 1) || !ggml_hrx_only_used_by(g, wv, 1) ||
            cont->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            return false;
        }
        // The window reads input column t + P - l. The tap of head h is one value for each token.
        if (wv->ne[0] != T || wv->ne[1] != C || wv->ne[2] != S || wv->ne[3] != 1 || wv->nb[1] != x->nb[1] || wv->nb[2] != x->nb[2] ||
            wv->view_offs != (size_t) (P - l) * sizeof(float) || w3->ne[1] < 1 || w3->ne[0] * w3->ne[1] != C || w3->ne[2] != T * S ||
            w3->ne[0] % 4 != 0 || tap->op != GGML_OP_VIEW || tap->type != GGML_TYPE_F32 || tap->view_src == nullptr ||
            tap->ne[0] != 1 || tap->ne[1] != w3->ne[1] || tap->ne[2] != T * S || tap->ne[3] != 1 || tap->view_offs % sizeof(float) != 0 ||
            tap->nb[1] % sizeof(float) != 0 || tap->nb[2] % sizeof(float) != 0) {
            return false;
        }
        if (tap0 == nullptr) {
            tap0     = tap;
            m.d_div  = w3->ne[0];
            m.d_s1   = (int64_t) (tap->nb[1] / sizeof(float));
            m.d_s2   = (int64_t) (tap->nb[2] / sizeof(float));
        } else if (w3->ne[0] != m.d_div || tap->view_src != tap0->view_src || tap->nb[1] != tap0->nb[1] || tap->nb[2] != tap0->nb[2] ||
                   tap->view_offs != tap0->view_offs + (size_t) l * sizeof(float)) {
            return false;
        }
        const int k_c = ggml_hrx_node_index(g, cont, k_a - 1, GGML_HRX_SCONV_MAX_DIST);
        const int k_m = ggml_hrx_node_index(g, mul, k_a - 1, GGML_HRX_SCONV_MAX_DIST);
        if (k_c < 0 || k_m < 0) {
            return false;
        }
        chain.insert(chain.end(), { k_c, k_m, k_a });
        views.push_back(wv);
        views.push_back(tr);
        acc = a;
        k   = k_a;
    }
    const int           k_add = ggml_hrx_sconv_user(g, k, k_max, GGML_OP_ADD, acc);
    const ggml_tensor * add   = k_add >= 0 ? g->nodes[k_add] : nullptr;
    const ggml_tensor * cur   = add == nullptr ? nullptr : add->src[0] == acc ? add->src[1] : add->src[0];
    const int           k_sc  = add != nullptr ? ggml_hrx_sconv_user(g, k_add, k_max, GGML_OP_SCALE, add) : -1;
    const ggml_tensor * scale = k_sc >= 0 ? g->nodes[k_sc] : nullptr;
    if (!ggml_hrx_only_used_by(g, acc, 1) || !ggml_hrx_sconv_cur_ok(cur, b) || add->type != GGML_TYPE_F32 || !ggml_is_contiguous(add) ||
        !ggml_are_same_shape(add, acc) || !ggml_hrx_only_used_by(g, add, 1) || scale == nullptr || scale->src[0] != add ||
        scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || !ggml_are_same_shape(scale, add) ||
        ggml_nbytes(tap0->view_src) / sizeof(float) >= INT_MAX - 4) {
        return false;
    }
    chain.insert(chain.end(), { k_add, k_sc });
    // The range: from the last node that runs and is not in the chain on. The chain nodes before it are absorbed.
    int k_first = k_sc;
    for (int j = k_sc - 1; j >= m.k_conv; --j) {
        const ggml_tensor * n = g->nodes[j];
        if (ggml_op_is_empty(n->op) || ggml_is_empty(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (std::find(chain.begin(), chain.end(), j) == chain.end()) {
            break;
        }
        k_first = j;
    }
    for (int j : chain) {
        if (j > k_sc) {
            return false;
        }
        if (j < k_first) {
            m.absorbed.push_back(j);
        }
    }
    m.scale   = scale;
    m.k_first = k_first;
    m.k_last  = k_sc;
    m.dyn     = tap0->view_src;
    m.d_base  = (int64_t) (tap0->view_offs / sizeof(float));
    return true;
}

static bool ggml_hrx_sconv_match_at(const ggml_cgraph * g, int i, ggml_hrx_sconv_match & m) {
    const ggml_tensor * x = g->nodes[i];
    if (x->op != GGML_OP_CONCAT || ggml_get_op_params_i32(x, 0) != 0 || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) ||
        (x->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }
    const ggml_tensor * past = x->src[0];
    const ggml_tensor * b    = x->src[1];
    const int64_t       P    = past->ne[0];
    const int64_t       T    = b->ne[0];
    const int64_t       C    = b->ne[1];
    const int64_t       S    = b->ne[2];
    if (!ggml_hrx_sconv_f32(past) || !ggml_hrx_sconv_f32(b) || P < 1 || P + 1 > GGML_HRX_SCONV_MAX_NC || T < GGML_HRX_SCONV_MIN_TOKENS ||
        C % 4 != 0 || past->ne[1] != C || past->ne[2] != S || past->ne[3] != 1 || b->ne[3] != 1 || b->nb[1] != sizeof(float) ||
        b->nb[0] % (4 * sizeof(float)) != 0 || b->nb[2] % (4 * sizeof(float)) != 0 || ggml_nelements(x) >= INT_MAX - 4 ||
        ggml_nbytes(b) / sizeof(float) >= INT_MAX - 4 || ggml_nbytes(past) / sizeof(float) >= INT_MAX - 4) {
        return false;
    }

    // The SSM_CONV, the windows of the taps, and the other readers of x: views of its last columns.
    const int32_t n_uses = ggml_hrx_use_count(g, x);
    if (n_uses < 1) {
        return false;
    }
    for (int k = i + 1; k < g->n_nodes && k <= i + GGML_HRX_SCONV_MAX_DIST; ++k) {
        const ggml_tensor * n = g->nodes[k];
        if (n->op == GGML_OP_SSM_CONV && n->src[0] == x) {
            if (m.conv != nullptr) {
                return false;
            }
            m.conv   = n;
            m.k_conv = k;
        }
    }
    const ggml_tensor * conv = m.conv;
    if (conv == nullptr) {
        return false;
    }
    const ggml_tensor * w = conv->src[1];
    if (conv->src[1] == x || conv->type != GGML_TYPE_F32 || !ggml_is_contiguous(conv) || conv->ne[0] != C || conv->ne[1] != T ||
        conv->ne[2] != S || conv->ne[3] != 1 || !ggml_hrx_sconv_f32(w) || w->ne[0] != P + 1 || w->ne[1] != C || w->ne[2] != 1 ||
        w->ne[3] != 1 || ggml_nbytes(w) / sizeof(float) >= INT_MAX - 4) {
        return false;
    }
    m.x       = x;
    m.k_first = m.k_conv;
    std::vector<const ggml_tensor *> windows;
    const bool dyn = ggml_hrx_sconv_match_dyn(g, i, m, windows);
    if (!dyn) {
        windows.clear();
        m.absorbed.clear();
        m.k_first = m.k_conv;
        m.k_last  = -1;
        m.scale   = nullptr;
        m.dyn     = nullptr;
    }
    int32_t n_found = 0;
    int64_t x_first = x->ne[0];
    for (int k = i + 1; k < g->n_nodes && k <= i + GGML_HRX_SCONV_MAX_DIST; ++k) {
        const ggml_tensor * n = g->nodes[k];
        int32_t refs = 0;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            refs += n->src[j] == x ? 1 : 0;
        }
        n_found += refs;
        if (n == conv || std::find(windows.begin(), windows.end(), n) != windows.end()) {
            continue;
        }
        if (n->view_src == x) {
            int64_t j0;
            int64_t j1;
            if (!ggml_hrx_sconv_view_cols(n, x->ne[0], j0, j1)) {
                return false;
            }
            x_first = std::min(x_first, j0);
        } else if (refs > 0) {
            return false;
        }
    }
    if (n_found != n_uses || x->ne[0] - x_first > GGML_HRX_SCONV_MAX_TAIL) {
        return false;
    }
    m.x_first = x_first;

    // Optional y = SCALE(ADD(cur, RESHAPE(conv))) right after the SSM_CONV, cur the tensor that b is the transpose of.
    const int           k_add = dyn ? -1 : ggml_hrx_next_node(g, m.k_conv);
    const ggml_tensor * add   = k_add >= 0 ? g->nodes[k_add] : nullptr;
    const ggml_tensor * rs    = add == nullptr || add->op != GGML_OP_ADD ? nullptr :
                                add->src[0]->op == GGML_OP_RESHAPE && add->src[0]->src[0] == conv ? add->src[0] : add->src[1];
    const ggml_tensor * cur   = rs == nullptr ? nullptr : add->src[0] == rs ? add->src[1] : add->src[0];
    const int           k_sc  = add != nullptr ? ggml_hrx_next_node(g, k_add) : -1;
    const ggml_tensor * scale = k_sc >= 0 ? g->nodes[k_sc] : nullptr;
    if (rs != nullptr && rs->op == GGML_OP_RESHAPE && rs->src[0] == conv && rs->ne[0] == C && rs->ne[1] == T * S && rs->ne[2] == 1 &&
        ggml_hrx_only_used_by(g, conv, 1) && ggml_hrx_only_used_by(g, rs, 1) && cur != rs && ggml_hrx_sconv_cur_ok(cur, b) &&
        add->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(add) && ggml_are_same_shape(add, rs) && ggml_hrx_only_used_by(g, add, 1) && scale != nullptr &&
        scale->op == GGML_OP_SCALE && scale->src[0] == add && scale->type == GGML_TYPE_F32 && ggml_is_contiguous(scale) &&
        ggml_are_same_shape(scale, add)) {
        m.scale  = scale;
        m.k_last = k_sc;
        // Optional ADD(r, y): r has stride 1 along the channels and rows 16-byte aligned.
        const int           k_a2  = ggml_hrx_next_node(g, k_sc);
        const ggml_tensor * add2  = k_a2 >= 0 ? g->nodes[k_a2] : nullptr;
        const ggml_tensor * resid = add2 == nullptr || add2->op != GGML_OP_ADD ? nullptr : add2->src[0] == scale ? add2->src[1] : add2->src[0];
        if (resid != nullptr && resid != scale && (add2->src[0] == scale || add2->src[1] == scale) && ggml_hrx_only_used_by(g, scale, 1) &&
            add2->type == GGML_TYPE_F32 && ggml_is_contiguous(add2) && ggml_are_same_shape(add2, scale) && resid->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(resid, scale) && resid->nb[0] == sizeof(float) && resid->nb[1] % (4 * sizeof(float)) == 0 &&
            ggml_nbytes(resid) / sizeof(float) < INT_MAX - 4) {
            m.add2   = add2;
            m.resid  = resid;
            m.k_last = k_a2;
        }
    }

    const ggml_tensor * dst = m.add2 != nullptr ? m.add2 : m.scale != nullptr ? m.scale : conv;
    for (const ggml_tensor * t : { past, b, w, dst, x, m.resid != nullptr ? m.resid : dst, m.dyn != nullptr ? m.dyn : dst }) {
        if (!ggml_hrx_tensor_on_device(t)) {
            return false;
        }
    }
    // The tail kernel reads past and b and writes x. The halo kernel reads x and b, then the conv kernel reads the halo
    // and b and writes dst: dst can be b itself, at the same elements.
    const ggml_backend_buffer_t dst_buf = dst->view_src != nullptr ? dst->view_src->buffer : dst->buffer;
    const ggml_backend_buffer_t b_buf   = b->view_src != nullptr ? b->view_src->buffer : b->buffer;
    const bool dst_is_b = dst->data == b->data && dst_buf == b_buf && b->nb[0] == (size_t) C * sizeof(float) &&
                          (S == 1 || b->nb[2] == (size_t) T * C * sizeof(float));
    if (ggml_hrx_overlap(x, past) || ggml_hrx_overlap(x, b) || ggml_hrx_overlap(dst, w) || (ggml_hrx_overlap(dst, b) && !dst_is_b)) {
        return false;
    }
    // The same for r: the conv kernel reads it at the elements that it writes.
    if (m.resid != nullptr && ggml_hrx_overlap(dst, m.resid) && (dst->data != m.resid->data || !ggml_is_contiguous(m.resid))) {
        return false;
    }
    if (m.dyn != nullptr && ggml_hrx_overlap(dst, m.dyn)) {
        return false;
    }
    // x, b and the taps must not change between the CONCAT (the taps: their node) and the range. The nodes of the range
    // are the fusion itself.
    const int k_dyn = m.dyn != nullptr ? ggml_hrx_node_index(g, m.dyn, m.k_first - 1, GGML_HRX_SCONV_MAX_DIST) : -1;
    for (int k = i + 1; k < m.k_first; ++k) {
        const ggml_tensor * n = g->nodes[k];
        if (!ggml_op_is_empty(n->op) && !ggml_is_empty(n) && (n->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 &&
            (ggml_hrx_overlap(n, x) || ggml_hrx_overlap(n, b) || (m.dyn != nullptr && k > k_dyn && ggml_hrx_overlap(n, m.dyn)))) {
            return false;
        }
    }
    return true;
}

// Tokens of one workgroup of 64 lanes: gfx1151, C 2304 and 4096 tokens, 0.29 ms for the halo and conv kernels (about
// 260 GB/s for cur and dst). Fewer tokens get smaller tiles for more workgroups.
static int64_t ggml_hrx_sconv_tokens(int64_t n_t) {
    return n_t >= 1024 ? 32 : 8;
}

static ggml_hrx_kernel * ggml_hrx_sconv_kernel(ggml_backend_hrx_context * ctx, const char * symbol, const ggml_hrx_sconv_match & m) {
    ggml_hrx_kernel_desc desc;
    desc.file   = "ssm/ssm_conv_concat_f32.loom";
    desc.symbol = symbol;
    desc.config = {
        { "ggml.sconv.nc", std::to_string(m.x->src[0]->ne[0] + 1) },
        { "ggml.sconv.lanes", "64" },
        { "ggml.sconv.tokens", std::to_string(ggml_hrx_sconv_tokens(m.x->src[1]->ne[0])) },
        { "ggml.sconv.res", m.add2 != nullptr ? "2" : m.scale != nullptr ? "1" : "0" },
        { "ggml.sconv.dyn", m.dyn != nullptr ? "1" : "0" },
    };
    return ggml_hrx_get_kernel(ctx->device, desc);
}

// Both fusions run only if all three kernels compile: the SSM_CONV range does not read x.
static bool ggml_hrx_sconv_kernels_ok(ggml_backend_hrx_context * ctx, const ggml_hrx_sconv_match & m) {
    for (const char * symbol : { "ggml_ssm_conv_concat_tail_f32", "ggml_ssm_conv_concat_halo_f32", "ggml_ssm_conv_concat_f32" }) {
        if (ggml_hrx_sconv_kernel(ctx, symbol, m) == nullptr) {
            return false;
        }
    }
    return true;
}

// The three kernels have the same arguments. tail: at the CONCAT. halo and conv: at the SSM_CONV.
static bool ggml_hrx_sconv_run(ggml_backend_hrx_context * ctx, const ggml_hrx_sconv_match & m, bool tail) {
    const ggml_tensor * x    = m.x;
    const ggml_tensor * past = x->src[0];
    const ggml_tensor * b    = x->src[1];
    const ggml_tensor * w    = m.conv->src[1];
    const ggml_tensor * dst  = m.add2 != nullptr ? m.add2 : m.scale != nullptr ? m.scale : m.conv;
    const ggml_tensor * res  = m.resid != nullptr ? m.resid : dst;
    const ggml_tensor * dyn  = m.dyn != nullptr ? m.dyn : dst;
    const int64_t       P    = past->ne[0];
    const int64_t       T    = b->ne[0];
    const int64_t       C    = b->ne[1];
    const int64_t       S    = b->ne[2];
    const int64_t       tok  = ggml_hrx_sconv_tokens(T);
    const int64_t       n_cg = (C / 4 + 63) / 64;
    const int64_t       n_tt = (T + tok - 1) / tok;
    const int64_t       n_h  = S * n_tt * P * C;
    if (n_h >= INT_MAX - 4) {
        return false;
    }
    const ggml_tensor * halo = tail ? x : ggml_hrx_scratch(ctx, (size_t) n_h * sizeof(float));
    if (halo == nullptr) {
        return false;
    }
    int64_t ps[GGML_MAX_DIMS];
    int64_t bs[GGML_MAX_DIMS];
    int64_t ws[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(past, ps);
    ggml_hrx_elem_strides(b, bs);
    ggml_hrx_elem_strides(w, ws);
    uint32_t sb[2] = { 0x3f800000u, 0 };
    if (m.scale != nullptr) {
        memcpy(sb, m.scale->op_params, sizeof(sb));
    }
    const std::vector<int64_t> scalars = {
        C, T, n_cg, n_tt,
        ps[0], ps[1], ps[2], (int64_t) (ggml_nbytes(past) / 4),
        bs[0], bs[2], (int64_t) (ggml_nbytes(b) / 4),
        ws[0], ws[1], (int64_t) (ggml_nbytes(w) / 4),
        ggml_nelements(dst), x->ne[0], m.x_first, ggml_nelements(x), n_h, (int64_t) (res->nb[1] / 4), (int64_t) (ggml_nbytes(res) / 4),
        m.d_base, m.d_div, m.d_s1, m.d_s2, (int64_t) (ggml_nbytes(dyn) / 4), (int64_t) sb[0], (int64_t) sb[1],
    };
    const std::vector<const ggml_tensor *> buffers = { past, b, w, dst, x, halo, res, dyn };
    if (tail) {
        const std::vector<const ggml_tensor *> writes = { x };
        ctx->dispatch_writes = &writes;
        const bool ok = ggml_hrx_dispatch(ctx, ggml_hrx_sconv_kernel(ctx, "ggml_ssm_conv_concat_tail_f32", m), { n_cg * S }, scalars, buffers);
        ctx->dispatch_writes = nullptr;
        return ok;
    }
    const std::vector<const ggml_tensor *> writes = { dst };
    ctx->dispatch_writes = &writes;
    const bool ok = ggml_hrx_dispatch(ctx, ggml_hrx_sconv_kernel(ctx, "ggml_ssm_conv_concat_halo_f32", m), { n_cg * n_tt * S }, scalars, buffers) &&
                    ggml_hrx_dispatch(ctx, ggml_hrx_sconv_kernel(ctx, "ggml_ssm_conv_concat_f32", m), { n_cg * n_tt * S }, scalars, buffers);
    ctx->dispatch_writes = nullptr;
    return ok;
}

int ggml_hrx_fuse_ssm_conv_tail(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, std::vector<int> * absorbed) {
    ggml_hrx_sconv_match m;
    if (!ggml_hrx_sconv_match_at(cgraph, i, m)) {
        return -1;
    }
    if (absorbed != nullptr) {
        return ggml_hrx_sconv_kernels_ok(ctx, m) ? i : -1;
    }
    if (!ggml_hrx_sconv_run(ctx, m, true)) {
        GGML_ABORT("%s: failed to run node %d (%s)", __func__, i, m.x->name);
    }
    return i;
}

// The CONCAT that node n reads: SSM_CONV(x), CONT(TRANSPOSE(VIEW(x))) or MUL(RESHAPE(that), tap).
static const ggml_tensor * ggml_hrx_sconv_x_of(const ggml_tensor * n) {
    if (n->op == GGML_OP_SSM_CONV) {
        return n->src[0];
    }
    if (n->op == GGML_OP_MUL && n->src[0]->op == GGML_OP_RESHAPE) {
        n = n->src[0]->src[0];
    }
    if (n != nullptr && n->op == GGML_OP_CONT && n->src[0]->op == GGML_OP_TRANSPOSE) {
        return n->src[0]->view_src;
    }
    return nullptr;
}

int ggml_hrx_fuse_ssm_conv_concat(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, std::vector<int> * absorbed) {
    const ggml_tensor * x = ggml_hrx_sconv_x_of(cgraph->nodes[i]);
    if (x == nullptr || x->op != GGML_OP_CONCAT) {
        return -1;
    }
    // The same match as at the CONCAT: the CONCAT did not write all of x.
    const int k_x = ggml_hrx_node_index(cgraph, x, i - 1, GGML_HRX_SCONV_MAX_DIST);
    ggml_hrx_sconv_match m;
    if (k_x < 0 || !ggml_hrx_sconv_match_at(cgraph, k_x, m) || m.k_first != i) {
        return -1;
    }
    const int last = m.k_last >= 0 ? m.k_last : m.k_conv;
    if (absorbed != nullptr) {
        if (!ggml_hrx_sconv_kernels_ok(ctx, m)) {
            return -1;
        }
        absorbed->insert(absorbed->end(), m.absorbed.begin(), m.absorbed.end());
        return last;
    }
    if (!ggml_hrx_sconv_run(ctx, m, false)) {
        GGML_ABORT("%s: failed to run nodes %d..%d (%s)", __func__, i, last, cgraph->nodes[i]->name);
    }
    return last;
}
