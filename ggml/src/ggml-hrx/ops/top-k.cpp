#include "ggml-hrx-ops.h"

#include <climits>
#include <string>

// Values of a row that the kernel keeps in workgroup memory: 32 KB of keys and indices.
static constexpr int64_t GGML_HRX_TOP_K_CAP = 4096;

bool ggml_hrx_supports_top_k(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src = op->src[0];
    if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_I32) {
        return false;
    }
    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(src, ss) || !ggml_hrx_elem_strides(op, ds) || ss[0] != 1 || ds[0] != 1) {
        return false;
    }
    // The pairs of the two-step path take 8 bytes for each value.
    const int64_t max_elems = INT_MAX / 2 - 1024;
    return ggml_nelements(src) <= max_elems && (int64_t) (ggml_nbytes(src) / 4) <= max_elems && (int64_t) (ggml_nbytes(op) / 4) <= max_elems;
}

static bool ggml_hrx_top_k_dispatch(ggml_backend_hrx_context * ctx, bool in_pairs, bool out_pairs, int64_t wg, int64_t ne0,
                                    int64_t chunk, int64_t chunks, int64_t k, int64_t rows, const int64_t ne[2], const int64_t s[3],
                                    int64_t src_span, const int64_t d[3], int64_t dst_span, const ggml_tensor * src,
                                    const ggml_tensor * dst) {
    const bool lds = chunk <= GGML_HRX_TOP_K_CAP;

    ggml_hrx_kernel_desc desc;
    desc.file   = "sort/top_k_f32.loom";
    desc.symbol = "ggml_top_k_f32";
    desc.config = {
        { "ggml.top_k.in_pairs", in_pairs ? "1" : "0" },
        { "ggml.top_k.out_pairs", out_pairs ? "1" : "0" },
        { "ggml.top_k.lds", lds ? "1" : "0" },
        { "ggml.top_k.cap", lds ? std::to_string(GGML_HRX_TOP_K_CAP) : "64" },
        { "ggml.top_k.wg_size", std::to_string(wg) },
    };
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    return ggml_hrx_dispatch(ctx, kernel, { rows * chunks },
        { ne0, chunk, chunks, k, ne[0], ne[1], s[0], s[1], s[2], src_span, d[0], d[1], d[2], dst_span }, { src, dst });
}

bool ggml_hrx_op_top_k(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];
    if (ggml_nelements(node) == 0) {
        return true;
    }

    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(src, ss);
    ggml_hrx_elem_strides(node, ds);

    const int64_t n        = src->ne[0];
    const int64_t k        = node->ne[0];
    const int64_t rows     = ggml_nrows(src);
    const int64_t ne[2]    = { src->ne[1], src->ne[2] };
    const int64_t s[3]     = { ss[1], ss[2], ss[3] };
    const int64_t d[3]     = { ds[1], ds[2], ds[3] };
    const int64_t src_span = (int64_t) (ggml_nbytes(src) / 4);
    const int64_t dst_span = (int64_t) (ggml_nbytes(node) / 4);
    const int64_t norm_wg  = std::max<int64_t>(256, ctx->device->tune->norm_wg_size);

    // A long row in few rows goes in two steps: the top k of each chunk of at most CAP values, then the top k of these.
    // Both steps keep their values in workgroup memory, and the first step uses many workgroups (limits not tuned).
    const int64_t chunks = (n + GGML_HRX_TOP_K_CAP - 1) / GGML_HRX_TOP_K_CAP;
    const int64_t chunk  = (n + chunks - 1) / chunks;
    const int64_t pairs  = chunks * k;
    if (chunks > 1 && k <= chunk && 2 * pairs <= n && rows * chunks <= 64 * (int64_t) ctx->device->n_cu) {
        const ggml_tensor * part = ggml_hrx_scratch(ctx, (size_t) (rows * pairs * 2) * sizeof(int32_t));
        if (part == nullptr) {
            return false;
        }
        const int64_t part_span = rows * pairs * 2;
        const int64_t none[3]   = { 0, 0, 0 };
        if (!ggml_hrx_top_k_dispatch(ctx, false, true, norm_wg, n, chunk, chunks, k, rows, ne, s, src_span, none, part_span, src, part)) {
            return false;
        }
        const int64_t ps[3] = { 2 * pairs, 2 * pairs * ne[0], 2 * pairs * ne[0] * ne[1] };
        const int64_t wg    = pairs > GGML_HRX_TOP_K_CAP ? 1024 : norm_wg;
        return ggml_hrx_top_k_dispatch(ctx, true, false, wg, pairs, pairs, 1, k, rows, ne, ps, part_span, d, dst_span, part, node);
    }

    const int64_t wg = n > GGML_HRX_TOP_K_CAP ? 1024 : norm_wg;
    return ggml_hrx_top_k_dispatch(ctx, false, false, wg, n, n, 1, k, rows, ne, s, src_span, d, dst_span, src, node);
}
