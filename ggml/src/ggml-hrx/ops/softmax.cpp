#include "ggml-hrx-ops.h"

#include <climits>
#include <cstring>
#include <string>

// Rows without a mask up to this length use 16 to 64 work items for each row. Other rows use one workgroup for each row.
static constexpr int64_t GGML_HRX_SOFT_MAX_ROWS_MAX_NE0 = 1024;
// Rows from this length use 1024 lanes. SOFT_MAX [200058, 1] took 520 us with 256 lanes and 142 us with 1024 lanes
// (gfx1151, GGML_HRX_PROFILE_OPS, llama-server decode with backend sampling).
static constexpr int64_t GGML_HRX_SOFT_MAX_WIDE_NE0 = 16384;
static constexpr int64_t GGML_HRX_SOFT_MAX_SPLIT_CHUNK = 4096;

bool ggml_hrx_supports_soft_max(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src  = op->src[0];
    const ggml_tensor * mask = op->src[1];
    if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || (mask != nullptr && mask->type != GGML_TYPE_F32)) {
        return false;
    }
    // Sinks and ALiBi (max_bias > 0) are not supported.
    float max_bias;
    memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
    if (op->src[2] != nullptr || max_bias != 0.0f) {
        return false;
    }
    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    int64_t ms[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(src, ss) || !ggml_hrx_elem_strides(op, ds) || ss[0] != 1 || ds[0] != 1) {
        return false;
    }
    if (mask != nullptr && (!ggml_hrx_elem_strides(mask, ms) || ms[0] != 1 || mask->ne[0] != src->ne[0] || mask->ne[1] < src->ne[1] ||
                            src->ne[2] % mask->ne[2] != 0 || src->ne[3] % mask->ne[3] != 0 || ggml_nbytes(mask) / 4 >= INT_MAX)) {
        return false;
    }
    // Sizes, counts and spans are 32-bit launch scalars.
    return ggml_nrows(op) < INT_MAX && ggml_nbytes(src) / 4 < INT_MAX && ggml_nbytes(op) / 4 < INT_MAX;
}

bool ggml_hrx_op_soft_max(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src  = node->src[0];
    const ggml_tensor * mask = node->src[1];
    if (ggml_nelements(node) == 0) {
        return true;
    }

    uint32_t scale_bits;
    memcpy(&scale_bits, node->op_params, sizeof(uint32_t));

    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    int64_t ms[GGML_MAX_DIMS] = { 1, 0, 0, 0 };
    ggml_hrx_elem_strides(src, ss);
    ggml_hrx_elem_strides(node, ds);
    if (mask != nullptr) {
        ggml_hrx_elem_strides(mask, ms);
    }

    const int64_t norm_wg_size = ctx->device->tune->norm_wg_size;
    const int64_t rows = ggml_nrows(node);
    const int64_t ne0  = src->ne[0];

    if (mask == nullptr && ne0 <= GGML_HRX_SOFT_MAX_ROWS_MAX_NE0) {
        const int64_t lanes          = ne0 <= 16 ? 16 : ne0 <= 32 ? 32 : 64;
        const int64_t rows_per_group = norm_wg_size / lanes;

        ggml_hrx_kernel_desc desc;
        desc.file   = "softmax/soft_max_rows_f32.loom";
        desc.symbol = "ggml_soft_max_f32_rows";
        desc.libs   = { "motifs/reduce.loom" };
        desc.config = {
            { "ggml.norm.wg_size", std::to_string(norm_wg_size) },
            { "ggml.soft_max.lanes", std::to_string(lanes) },
        };
        ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
        return ggml_hrx_dispatch(ctx, kernel, { (rows + rows_per_group - 1) / rows_per_group },
            {
                rows, ne0, src->ne[1], src->ne[2],
                ss[1], ss[2], ss[3], (int64_t) (ggml_nbytes(src) / 4),
                ds[1], ds[2], ds[3], (int64_t) (ggml_nbytes(node) / 4),
                (int64_t) scale_bits,
            },
            { src, node });
    }

    // A long row without a mask runs in parts of GGML_HRX_SOFT_MAX_SPLIT_CHUNK values, one workgroup each.
    // gfx1151, a row of 201024 (backend sampler of a verify batch, device time): 9.6 + 13.4 us, 151 us in one workgroup.
    if (mask == nullptr && ne0 >= GGML_HRX_SOFT_MAX_WIDE_NE0) {
        const int64_t n_parts   = (ne0 + GGML_HRX_SOFT_MAX_SPLIT_CHUNK - 1) / GGML_HRX_SOFT_MAX_SPLIT_CHUNK;
        const int64_t groups    = rows * n_parts;
        const int64_t part_span = 2 * groups;
        const ggml_tensor * part = ggml_hrx_scratch(ctx, (size_t) part_span * sizeof(float));
        if (part == nullptr || groups >= INT_MAX / 2) {
            return false;
        }

        ggml_hrx_kernel_desc desc;
        desc.file   = "softmax/soft_max_split_f32.loom";
        desc.symbol = "ggml_soft_max_f32_split_part";
        desc.config = { { "ggml.norm.wg_size", std::to_string(norm_wg_size) } };
        ggml_hrx_kernel * k_part = ggml_hrx_get_kernel(ctx->device, desc);
        desc.symbol = "ggml_soft_max_f32_split_norm";
        ggml_hrx_kernel * k_norm = ggml_hrx_get_kernel(ctx->device, desc);

        return ggml_hrx_dispatch(ctx, k_part, { groups },
                   {
                       groups, n_parts, GGML_HRX_SOFT_MAX_SPLIT_CHUNK, ne0, src->ne[1], src->ne[2],
                       ss[1], ss[2], ss[3], (int64_t) (ggml_nbytes(src) / 4), part_span, (int64_t) scale_bits,
                   },
                   { src, part }) &&
               ggml_hrx_dispatch(ctx, k_norm, { groups },
                   {
                       groups, n_parts, GGML_HRX_SOFT_MAX_SPLIT_CHUNK, ne0, src->ne[1], src->ne[2],
                       ss[1], ss[2], ss[3], (int64_t) (ggml_nbytes(src) / 4), part_span,
                       ds[1], ds[2], ds[3], (int64_t) (ggml_nbytes(node) / 4), (int64_t) scale_bits,
                   },
                   { src, part, node });
    }

    const int64_t wg_size = ne0 >= GGML_HRX_SOFT_MAX_WIDE_NE0 ? 1024 : norm_wg_size;

    ggml_hrx_kernel_desc desc;
    desc.file   = "softmax/soft_max_f32.loom";
    desc.symbol = "ggml_soft_max_f32";
    desc.config = {
        { "ggml.norm.wg_size", std::to_string(wg_size) },
        { "ggml.soft_max.mask", mask != nullptr ? "1" : "0" },
    };
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    // Without a mask, src takes the place of the mask buffer. The kernel does not read it.
    const ggml_tensor * mask_buf = mask != nullptr ? mask : src;
    return ggml_hrx_dispatch(ctx, kernel, { rows },
        {
            rows, ne0, src->ne[1], src->ne[2],
            ss[1], ss[2], ss[3], (int64_t) (ggml_nbytes(src) / 4),
            mask != nullptr ? mask->ne[2] : 1, mask != nullptr ? mask->ne[3] : 1, ms[1], ms[2], ms[3], (int64_t) (ggml_nbytes(mask_buf) / 4),
            ds[1], ds[2], ds[3], (int64_t) (ggml_nbytes(node) / 4),
            (int64_t) scale_bits,
        },
        { src, mask_buf, node });
}
