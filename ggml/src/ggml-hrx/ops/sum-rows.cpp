#include "ggml-hrx-ops.h"

#include <climits>
#include <string>

// Rows above this length use one workgroup for each row. Below it, 64 lanes do at most 16 loads each.
static constexpr int64_t GGML_HRX_SUM_ROWS_LANES_MAX_NE0 = 1024;

bool ggml_hrx_supports_sum_rows(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(src, ss) || !ggml_hrx_elem_strides(op, ds)) {
        return false;
    }
    // All offsets must stay below 2^31 - 2.
    const int64_t max_elems = INT_MAX - 1024;
    return ggml_nrows(src) <= max_elems && (int64_t) (ggml_nbytes(src) / 4) <= max_elems && (int64_t) (ggml_nbytes(op) / 4) <= max_elems;
}

bool ggml_hrx_op_sum_rows(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];
    if (ggml_nelements(node) == 0) {
        return true;
    }

    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(src, ss);
    ggml_hrx_elem_strides(node, ds);

    const ggml_hrx_tune * tune = ctx->device->tune;
    const int64_t rows = ggml_nrows(src);
    const int64_t ne0  = src->ne[0];
    const std::vector<int64_t> scalars = {
        rows, ne0, src->ne[1], src->ne[2],
        ss[0], ss[1], ss[2], ss[3], (int64_t) (ggml_nbytes(src) / 4),
        ds[1], ds[2], ds[3], (int64_t) (ggml_nbytes(node) / 4),
    };

    ggml_hrx_kernel_desc desc;
    desc.file   = "reduce/sum_rows_f32.loom";
    desc.libs   = { "motifs/reduce.loom" };
    desc.config = { { "ggml.norm.wg_size", std::to_string(tune->norm_wg_size) } };

    if (ne0 > GGML_HRX_SUM_ROWS_LANES_MAX_NE0) {
        desc.symbol = "ggml_sum_rows_f32_wg";
        desc.config.push_back({ "ggml.sum_rows.lanes", "64" });
        ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
        return ggml_hrx_dispatch(ctx, kernel, { rows }, scalars, { src, node });
    }

    const int64_t lanes          = ne0 <= 16 ? 16 : ne0 <= 32 ? 32 : 64;
    const int64_t rows_per_group = tune->norm_wg_size / lanes;
    desc.symbol = "ggml_sum_rows_f32_rows";
    desc.config.push_back({ "ggml.sum_rows.lanes", std::to_string(lanes) });
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
    return ggml_hrx_dispatch(ctx, kernel, { (rows + rows_per_group - 1) / rows_per_group }, scalars, { src, node });
}
