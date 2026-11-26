#include "ggml-hrx-ops.h"

#include <climits>
#include <string>

// Must match the op codes in elementwise/binary_f32.loom.
static int ggml_hrx_binary_code(ggml_op op) {
    switch (op) {
        case GGML_OP_ADD: return 0;
        case GGML_OP_SUB: return 1;
        case GGML_OP_MUL: return 2;
        case GGML_OP_DIV: return 3;
        default:          return -1;
    }
}

// Row r of dst uses row (r % rows) of x, at element (r % rows) * rs.
static bool ggml_hrx_binary_rows_form(const ggml_tensor * x, const ggml_tensor * dst, int64_t * rows, int64_t * rs) {
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(x, s)) {
        return false;
    }
    *rows = 1;
    *rs   = 0;
    int64_t next  = 0;
    bool    bcast = false;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (dst->ne[i] == 1) {
            continue;
        }
        if (x->ne[i] == 1) {
            bcast = true;
            continue;
        }
        if (bcast || x->ne[i] != dst->ne[i] || (*rows > 1 && s[i] != next)) {
            return false;
        }
        if (*rows == 1) {
            *rs = s[i];
        }
        *rows *= x->ne[i];
        next = s[i] * x->ne[i];
    }
    return *rs < INT_MAX;
}

// The rows kernel does 4 elements for each work item. It needs contiguous rows that are multiples of 4.
static bool ggml_hrx_binary_rows_ok(const ggml_tensor * op) {
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (!ggml_is_contiguous(op) || !ggml_are_same_shape(a, op) || !ggml_can_repeat(b, op) || op->ne[0] % 4 != 0 ||
        a->nb[0] != sizeof(float)) {
        return false;
    }
    if (b->ne[0] != 1 && (b->ne[0] != op->ne[0] || b->nb[0] != sizeof(float))) {
        return false;
    }
    // The last workgroup can go 4092 elements past n (1024 lanes). All offsets must stay below 2^31 - 8.
    const int64_t max_elems = INT_MAX - 4096;
    if (ggml_nelements(op) > max_elems || (int64_t) (ggml_nbytes(a) / 4) > max_elems || (int64_t) (ggml_nbytes(b) / 4) > max_elems) {
        return false;
    }
    int64_t rows;
    int64_t rs;
    return ggml_hrx_binary_rows_form(a, op, &rows, &rs) && ggml_hrx_binary_rows_form(b, op, &rows, &rs);
}

// Element strides for the bcast kernel. A size-1 dim gets stride 0. Thus a broadcast dim of b needs no modulo.
static bool ggml_hrx_binary_strides(const ggml_tensor * t, int64_t s[GGML_MAX_DIMS]) {
    if (!ggml_hrx_elem_strides(t, s) || ggml_nbytes(t) / ggml_type_size(t->type) >= INT_MAX) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] == 1) {
            s[i] = 0;
        }
        if (s[i] >= INT_MAX) {
            return false;
        }
    }
    return true;
}

bool ggml_hrx_supports_binary(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (ggml_hrx_binary_code(op->op) < 0 || a == nullptr || b == nullptr) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        return false;
    }
    if (ggml_hrx_binary_rows_ok(op)) {
        return true;
    }
    // The bcast kernel.
    int64_t s[GGML_MAX_DIMS];
    return ggml_are_same_shape(a, op) && ggml_can_repeat(b, op) && ggml_nelements(op) < INT_MAX &&
           ggml_hrx_binary_strides(a, s) && ggml_hrx_binary_strides(b, s) && ggml_hrx_binary_strides(op, s);
}

static bool ggml_hrx_op_binary_bcast(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];
    int64_t ne[GGML_MAX_DIMS];
    int64_t tne[3][GGML_MAX_DIMS];
    int64_t ts[3][GGML_MAX_DIMS];
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        ne[i]     = node->ne[i];
        tne[0][i] = node->ne[i];
        tne[1][i] = b->ne[i];
        tne[2][i] = node->ne[i];
    }
    ggml_hrx_binary_strides(a, ts[0]);
    ggml_hrx_binary_strides(b, ts[1]);
    ggml_hrx_binary_strides(node, ts[2]);
    const int dims = ggml_hrx_collapse_dims(ne, 3, tne, ts);
    // The modulo is only needed if b is repeated more than one time along a dim where b ne > 1.
    bool b_mod = false;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        b_mod = b_mod || (tne[1][i] != 1 && tne[1][i] != ne[i]);
    }

    ggml_hrx_kernel_desc desc;
    desc.file   = "elementwise/binary_f32.loom";
    desc.symbol = "ggml_binary_f32_bcast";
    desc.config = {
        { "ggml.binary.op", std::to_string(ggml_hrx_binary_code(node->op)) },
        { "ggml.binary.b_mod", b_mod ? "1" : "0" },
        { "ggml.binary.dims", dims == 1 ? "1" : dims == 2 ? "2" : "4" },
    };
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    const int64_t n = ggml_nelements(node);
    return ggml_hrx_dispatch(ctx, kernel, { n },
        {
            n, ne[0], ne[1], ne[2],
            ts[0][0], ts[0][1], ts[0][2], ts[0][3], (int64_t) (ggml_nbytes(a) / 4),
            tne[1][0], tne[1][1], tne[1][2], tne[1][3],
            ts[1][0], ts[1][1], ts[1][2], ts[1][3], (int64_t) (ggml_nbytes(b) / 4),
            ts[2][0], ts[2][1], ts[2][2], ts[2][3], (int64_t) (ggml_nbytes(node) / 4),
        },
        { a, b, node });
}

bool ggml_hrx_op_binary(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];
    if (ggml_nelements(node) == 0) {
        return true;
    }
    if (!ggml_hrx_binary_rows_ok(node)) {
        return ggml_hrx_op_binary_bcast(ctx, node);
    }
    int64_t a_rows;
    int64_t a_rs;
    int64_t b_rows;
    int64_t b_rs;
    ggml_hrx_binary_rows_form(a, node, &a_rows, &a_rs);
    ggml_hrx_binary_rows_form(b, node, &b_rows, &b_rs);

    ggml_hrx_kernel_desc desc;
    desc.file   = "elementwise/binary_f32.loom";
    desc.symbol = "ggml_binary_f32_rows";
    desc.config = {
        { "ggml.binary.op", std::to_string(ggml_hrx_binary_code(node->op)) },
        { "ggml.binary.b_bcast0", b->ne[0] == 1 ? "1" : "0" },
    };
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    const int64_t n = ggml_nelements(node);
    return ggml_hrx_dispatch(ctx, kernel, { n },
        { n, node->ne[0], a_rs, (int64_t) (ggml_nbytes(a) / 4), b_rows, b_rs, (int64_t) (ggml_nbytes(b) / 4) },
        { a, b, node });
}
