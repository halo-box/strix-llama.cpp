#include "ggml-hrx-ops.h"

#include <climits>
#include <cstring>
#include <string>

// Must match the op codes in elementwise/unary_f32.loom.
enum ggml_hrx_unary_code {
    GGML_HRX_UNARY_COPY    = 0,
    GGML_HRX_UNARY_SQR     = 1,
    GGML_HRX_UNARY_SQRT    = 2,
    GGML_HRX_UNARY_LOG     = 3,
    GGML_HRX_UNARY_SIGMOID = 4,
    GGML_HRX_UNARY_TANH    = 5,
    GGML_HRX_UNARY_CLAMP   = 6,
    GGML_HRX_UNARY_FILL    = 7,
    GGML_HRX_UNARY_EXP     = 8,
    GGML_HRX_UNARY_NEG     = 9,
    GGML_HRX_UNARY_ABS     = 10,
    GGML_HRX_UNARY_SILU    = 11,
    GGML_HRX_UNARY_RELU    = 12,
    GGML_HRX_UNARY_GELU    = 13,
    GGML_HRX_UNARY_SGN     = 14,
    GGML_HRX_UNARY_STEP    = 15,
    GGML_HRX_UNARY_SCALE   = 16,
};

static int ggml_hrx_unary_code(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_REPEAT: return GGML_HRX_UNARY_COPY;
        case GGML_OP_SQR:    return GGML_HRX_UNARY_SQR;
        case GGML_OP_SQRT:   return GGML_HRX_UNARY_SQRT;
        case GGML_OP_LOG:    return GGML_HRX_UNARY_LOG;
        case GGML_OP_CLAMP:  return GGML_HRX_UNARY_CLAMP;
        case GGML_OP_FILL:   return GGML_HRX_UNARY_FILL;
        case GGML_OP_SCALE:  return GGML_HRX_UNARY_SCALE;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SIGMOID: return GGML_HRX_UNARY_SIGMOID;
                case GGML_UNARY_OP_TANH:    return GGML_HRX_UNARY_TANH;
                case GGML_UNARY_OP_EXP:     return GGML_HRX_UNARY_EXP;
                case GGML_UNARY_OP_NEG:     return GGML_HRX_UNARY_NEG;
                case GGML_UNARY_OP_ABS:     return GGML_HRX_UNARY_ABS;
                case GGML_UNARY_OP_SILU:    return GGML_HRX_UNARY_SILU;
                case GGML_UNARY_OP_RELU:    return GGML_HRX_UNARY_RELU;
                case GGML_UNARY_OP_GELU:    return GGML_HRX_UNARY_GELU;
                case GGML_UNARY_OP_SGN:     return GGML_HRX_UNARY_SGN;
                case GGML_UNARY_OP_STEP:    return GGML_HRX_UNARY_STEP;
                default:                    return -1;
            }
        default: return -1;
    }
}

// Element strides. A size-1 dim gets stride 0.
static bool ggml_hrx_unary_strides(const ggml_tensor * t, int64_t s[GGML_MAX_DIMS]) {
    if (!ggml_hrx_elem_strides(t, s) || ggml_nbytes(t) / ggml_type_size(t->type) > (size_t) INT_MAX - 4096) {
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

int ggml_hrx_collapse_dims(int64_t ne[GGML_MAX_DIMS], int n_t, int64_t (*tne)[GGML_MAX_DIMS], int64_t (*ts)[GGML_MAX_DIMS]) {
    int n = 0;
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        if (ne[k] == 1) {
            continue;
        }
        const int j     = n - 1;
        bool      merge = n > 0;
        for (int t = 0; t < n_t && merge; ++t) {
            const bool bcast = tne[t][j] == 1 && tne[t][k] == 1;
            const bool full  = tne[t][j] == ne[j] && tne[t][k] == ne[k] && ts[t][k] == ts[t][j] * ne[j];
            merge            = bcast || full;
        }
        if (merge) {
            ne[j] *= ne[k];
            for (int t = 0; t < n_t; ++t) {
                tne[t][j] *= tne[t][k];
            }
            continue;
        }
        ne[n] = ne[k];
        for (int t = 0; t < n_t; ++t) {
            tne[t][n] = tne[t][k];
            ts[t][n]  = ts[t][k];
        }
        ++n;
    }
    for (int k = n; k < GGML_MAX_DIMS; ++k) {
        ne[k] = 1;
        for (int t = 0; t < n_t; ++t) {
            tne[t][k] = 1;
            ts[t][k]  = 0;
        }
    }
    return std::max(n, 1);
}

static int64_t ggml_hrx_f32_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits;
}

bool ggml_hrx_supports_unary(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    if (ggml_hrx_unary_code(op) < 0 || src == nullptr || src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    if (op->op == GGML_OP_REPEAT ? !ggml_can_repeat(src, op) : !ggml_are_same_shape(src, op)) {
        return false;
    }
    // The last workgroup of the vec4 kernel can go 4092 elements past n. All offsets must stay below 2^31 - 8.
    int64_t s[GGML_MAX_DIMS];
    return ggml_nelements(op) <= INT_MAX - 8192 && ggml_hrx_unary_strides(src, s) && ggml_hrx_unary_strides(op, s);
}

bool ggml_hrx_op_unary(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];
    const int64_t       n   = ggml_nelements(node);
    if (n == 0) {
        return true;
    }
    const int code = ggml_hrx_unary_code(node);
    float     p0   = 0.0f;
    float     p1   = 0.0f;
    if (code == GGML_HRX_UNARY_CLAMP || code == GGML_HRX_UNARY_SCALE) {
        p0 = ggml_get_op_params_f32(node, 0);
        p1 = ggml_get_op_params_f32(node, 1);
    } else if (code == GGML_HRX_UNARY_FILL) {
        p0 = ggml_get_op_params_f32(node, 0);
    }

    ggml_hrx_kernel_desc desc;
    desc.file = "elementwise/unary_f32.loom";
    if (node->op != GGML_OP_REPEAT && ggml_is_contiguous(src) && ggml_is_contiguous(node)) {
        desc.symbol = "ggml_unary_f32_vec4";
        desc.config = { { "ggml.unary.op", std::to_string(code) } };
        desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
        ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
        return ggml_hrx_dispatch(ctx, kernel, { n }, { n, ggml_hrx_f32_bits(p0), ggml_hrx_f32_bits(p1) }, { src, node });
    }

    int64_t ne[GGML_MAX_DIMS];
    int64_t tne[2][GGML_MAX_DIMS];
    int64_t ts[2][GGML_MAX_DIMS];
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        ne[i]     = node->ne[i];
        tne[0][i] = src->ne[i];
        tne[1][i] = node->ne[i];
    }
    ggml_hrx_unary_strides(src, ts[0]);
    ggml_hrx_unary_strides(node, ts[1]);
    const int dims = ggml_hrx_collapse_dims(ne, 2, tne, ts);
    // The modulo is only needed if src is repeated more than one time along a dim where src ne > 1.
    bool src_mod = false;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        src_mod = src_mod || (tne[0][i] != 1 && tne[0][i] != ne[i]);
    }

    desc.symbol = "ggml_unary_f32";
    desc.config = {
        { "ggml.unary.op", std::to_string(code) },
        { "ggml.unary.src_mod", src_mod ? "1" : "0" },
        { "ggml.unary.dims", dims == 1 ? "1" : dims == 2 ? "2" : "4" },
    };
    desc.config.push_back(ggml_hrx_elementwise_wg_config(ctx->device));
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
    return ggml_hrx_dispatch(ctx, kernel, { n },
        {
            n, ne[0], ne[1], ne[2],
            tne[0][0], tne[0][1], tne[0][2], tne[0][3],
            ts[0][0], ts[0][1], ts[0][2], ts[0][3], (int64_t) (ggml_nbytes(src) / 4),
            ts[1][0], ts[1][1], ts[1][2], ts[1][3], (int64_t) (ggml_nbytes(node) / 4),
            ggml_hrx_f32_bits(p0), ggml_hrx_f32_bits(p1),
        },
        { src, node });
}

bool ggml_hrx_supports_arange(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    return op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && ggml_nelements(op) < INT_MAX;
}

bool ggml_hrx_op_arange(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const int64_t n = ggml_nelements(node);
    if (n == 0) {
        return true;
    }
    ggml_hrx_kernel_desc desc;
    desc.file   = "elementwise/unary_f32.loom";
    desc.symbol = "ggml_arange_f32";
    desc.config = { ggml_hrx_elementwise_wg_config(ctx->device) };
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
    return ggml_hrx_dispatch(ctx, kernel, { n },
        { n, ggml_hrx_f32_bits(ggml_get_op_params_f32(node, 0)), ggml_hrx_f32_bits(ggml_get_op_params_f32(node, 2)) },
        { node });
}
