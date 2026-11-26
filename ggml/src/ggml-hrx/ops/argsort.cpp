#include "ggml-hrx-ops.h"

#include <climits>
#include <string>

// The kernel keeps one row of keys in workgroup memory (16 KB at this length). Each value reads all keys of the row.
static constexpr int64_t GGML_HRX_ARGSORT_MAX_NE0 = 4096;
// Longer rows: a bitonic sort of blocks of this length in workgroup memory, then merge passes.
static constexpr int64_t GGML_HRX_ARGSORT_BLOCK = 2048;

bool ggml_hrx_supports_argsort(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_I32 || !ggml_are_same_shape(src, op)) {
        return false;
    }
    const int32_t order = ggml_get_op_params_i32(op, 0);
    if (order != GGML_SORT_ORDER_ASC && order != GGML_SORT_ORDER_DESC) {
        return false;
    }
    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(src, ss) || !ggml_hrx_elem_strides(op, ds) || ss[0] != 1 || ds[0] != 1) {
        return false;
    }
    // The long-row path keeps 2 x 2 i32 for each value in scratch memory.
    const bool long_ok = ggml_nelements(src) < INT_MAX / 4 - 1024;
    return (src->ne[0] <= GGML_HRX_ARGSORT_MAX_NE0 || long_ok) && ggml_nrows(src) < INT_MAX && ggml_nbytes(src) / 4 < INT_MAX - 1024 &&
           ggml_nbytes(op) / 4 < INT_MAX - 1024;
}

static bool ggml_hrx_argsort_long(ggml_backend_hrx_context * ctx, const ggml_tensor * node, const int64_t ss[GGML_MAX_DIMS],
                                  const int64_t ds[GGML_MAX_DIMS]) {
    const ggml_tensor * src    = node->src[0];
    const int64_t       ne0    = src->ne[0];
    const int64_t       n      = ggml_nelements(src);
    const int64_t       blocks = (ne0 + GGML_HRX_ARGSORT_BLOCK - 1) / GGML_HRX_ARGSORT_BLOCK;
    const char *        order  = ggml_get_op_params_i32(node, 0) == GGML_SORT_ORDER_DESC ? "1" : "0";

    // Two areas of (key, index) pairs. The passes go from one to the other.
    const int64_t       area = 2 * n;
    const ggml_tensor * work = ggml_hrx_scratch(ctx, (size_t) (2 * area) * sizeof(int32_t));
    if (work == nullptr) {
        return false;
    }

    ggml_hrx_kernel_desc bdesc;
    bdesc.file   = "sort/argsort_long_f32.loom";
    bdesc.symbol = "ggml_argsort_block_f32";
    bdesc.config = {
        { "ggml.argsort.order", order },
        { "ggml.argsort.block", std::to_string(GGML_HRX_ARGSORT_BLOCK) },
        { "ggml.argsort.wg_size", std::to_string(GGML_HRX_ARGSORT_BLOCK / 2) },
    };
    if (!ggml_hrx_dispatch(ctx, ggml_hrx_get_kernel(ctx->device, bdesc), { ggml_nrows(src) * blocks },
            { ne0, blocks, src->ne[1], src->ne[2], ss[1], ss[2], ss[3], (int64_t) (ggml_nbytes(src) / 4), 0, 2 * area },
            { src, work })) {
        return false;
    }

    const int64_t wg = ctx->device->tune->norm_wg_size;
    int64_t       in = 0;
    for (int64_t run = GGML_HRX_ARGSORT_BLOCK; run < ne0; run *= 2) {
        const bool final = 2 * run >= ne0;
        int64_t    steps = 1;
        while ((int64_t(1) << steps) <= run) {
            steps++;
        }
        ggml_hrx_kernel_desc mdesc;
        mdesc.file   = "sort/argsort_long_f32.loom";
        mdesc.symbol = "ggml_argsort_merge_f32";
        mdesc.config = {
            { "ggml.argsort.final", final ? "1" : "0" },
            { "ggml.argsort.wg_size", std::to_string(wg) },
        };
        if (!ggml_hrx_dispatch(ctx, ggml_hrx_get_kernel(ctx->device, mdesc), { (n + wg - 1) / wg },
                {
                    n, ne0, src->ne[1], src->ne[2], run, steps, in, area - in, 2 * area,
                    ds[1], ds[2], ds[3], (int64_t) (ggml_nbytes(node) / 4),
                },
                { work, node })) {
            return false;
        }
        in = area - in;
    }
    return true;
}

bool ggml_hrx_op_argsort(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];
    if (ggml_nelements(node) == 0) {
        return true;
    }

    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(src, ss);
    ggml_hrx_elem_strides(node, ds);

    if (src->ne[0] > GGML_HRX_ARGSORT_MAX_NE0) {
        return ggml_hrx_argsort_long(ctx, node, ss, ds);
    }

    // Powers of two limit the number of kernel variants.
    int64_t cols = 64;
    while (cols < src->ne[0]) {
        cols *= 2;
    }
    const int64_t wg_size = std::min<int64_t>(cols, ctx->device->tune->norm_wg_size);

    ggml_hrx_kernel_desc desc;
    desc.file   = "sort/argsort_f32.loom";
    desc.symbol = "ggml_argsort_f32";
    desc.config = {
        { "ggml.argsort.order", ggml_get_op_params_i32(node, 0) == GGML_SORT_ORDER_DESC ? "1" : "0" },
        { "ggml.argsort.cols", std::to_string(cols) },
        { "ggml.argsort.wg_size", std::to_string(wg_size) },
    };
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    return ggml_hrx_dispatch(ctx, kernel, { ggml_nrows(src) },
        {
            src->ne[0], src->ne[1], src->ne[2],
            ss[1], ss[2], ss[3], (int64_t) (ggml_nbytes(src) / 4),
            ds[1], ds[2], ds[3], (int64_t) (ggml_nbytes(node) / 4),
        },
        { src, node });
}
