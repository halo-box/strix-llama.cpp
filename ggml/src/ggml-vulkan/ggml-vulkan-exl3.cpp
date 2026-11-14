#include "ggml-vulkan-common.h"

#if defined(GGML_USE_EXL3)
#include "../../exl3/exl3.h"

// EXL3 linear, see ggml/exl3/README.md: y = H(H(x*suh) W_hat) * svh/128

// The kernels are written for wave32 with f16 arithmetic and use non-private "last workgroup finishes the segment"
// arrival counters on coherent buffers without the Vulkan memory model. They were validated on RADV RDNA3 (gfx1151,
// Mesa 26.0.8 and 26.2.4) only; widen this check once another driver/architecture passes test-backend-ops -o EXL3.
bool ggml_vk_exl3_device_ok(const vk_device & device) {
    return device->driver_id == vk::DriverId::eMesaRadv && device->architecture == vk_device_architecture::AMD_RDNA3 &&
           device->fp16 && device->subgroup_shuffle && device->subgroup_arithmetic && device->subgroup_size_control &&
           device->subgroup_require_full_support && device->subgroup_min_size <= 32 && 32 <= device->subgroup_max_size &&
           device->properties.limits.maxComputeSharedMemorySize >= EXL3_ROUTE_SHMEM;
}

// the GEMM path: KHR coopmat 16x16x16 f16 x f16 -> f32 at wave32 (coopmat2 devices take it too)
bool ggml_vk_exl3_device_cm1(const vk_device & device) {
    return ggml_vk_exl3_device_ok(device) && device->coopmat_support && device->coopmat_support_16x16x16_f32acc &&
           device->properties.limits.maxComputeSharedMemorySize >= EXL3_MM_SHMEM;
}

// pipeline index of a rate: K = 1..8 -> 0..7, K = 1.5, 2.5, 3.5 -> 8..10
static int ggml_vk_exl3_rate_idx(int64_t tile_words) {
    if (tile_words % 16 == 0) {
        return (int) (tile_words / 16) - 1;
    }
    return 8 + (int) (tile_words / 16) - 1;
}

// GGML_VK_DISABLE_EXL3=1: leave the EXL3 ops to the CPU (A/B against the CPU path)
static bool ggml_vk_exl3_disabled() {
    static const bool disabled = [] { const char * e = getenv("GGML_VK_DISABLE_EXL3"); return e && atoi(e) != 0; }();
    return disabled;
}

bool ggml_vk_exl3_supports_op(const vk_device & device, const ggml_tensor * op) {
    const ggml_tensor * trellis = op->src[0];
    const ggml_tensor * x       = op->src[1];
    const ggml_tensor * suh     = op->src[2];
    const ggml_tensor * svh     = op->src[3];
    const int           cb      = ggml_get_op_params_i32(op, 0);
    if (ggml_vk_exl3_disabled() || !ggml_vk_exl3_device_ok(device) || op->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 ||
        x->nb[0] != sizeof(float) || !ggml_is_contiguous(op) || !ggml_is_contiguous(trellis) ||
        !ggml_is_contiguous(suh) || !ggml_is_contiguous(svh) ||
        !ggml_exl3_rate_supported(trellis->ne[0], (ggml_exl3_codebook) cb)) {
        return false;
    }
    for (int i = 1; i < 4; ++i) {
        if (x->nb[i] % sizeof(float) != 0) {
            return false;
        }
    }
    // 32-bit indexing in the shaders
    return ggml_nelements(op) < INT32_MAX && ggml_nelements(x) < INT32_MAX && ggml_nbytes(x) < UINT32_MAX &&
           ggml_nelements(trellis) < INT32_MAX;
}

// rows up to this take the dot kernel, more the GEMM
#define EXL3_DOT_MAX_ROWS 8u

// GEMM products: one (xh rounded to f16 once) from 512 rows, hi + lo below, as the HIP path (exl3_gemm_products)
static uint32_t ggml_vk_exl3_gemm_products(uint32_t rows) {
    return rows >= 512 ? 1u : 2u;
}

static void ggml_vk_exl3_mul_mm(ggml_backend_vk_context * ctx, vk_context & subctx, ggml_tensor * dst, int ri, int cb) {
    const ggml_tensor * trellis = dst->src[0];
    const ggml_tensor * x       = dst->src[1];
    const ggml_tensor * suh     = dst->src[2];
    const ggml_tensor * svh     = dst->src[3];

    const uint32_t n        = (uint32_t) trellis->ne[1]*16;
    const uint32_t k        = (uint32_t) trellis->ne[2]*16;
    const uint32_t rows     = (uint32_t) ggml_nrows(x);
    const uint32_t products = ggml_vk_exl3_gemm_products(ggml_is_contiguous(x) ? rows : 0);
    const uint32_t row_blks = CEIL_DIV(rows, EXL3_MM_ROWS);
    const uint32_t rows_pad = row_blks*EXL3_MM_ROWS;

    // scratch: inv [rows_pad] f32, hi [rows_pad, k] f16, lo (two products)
    const size_t inv_size = GGML_PAD((size_t) rows_pad*sizeof(float), 256);
    const size_t h_size   = GGML_PAD((size_t) rows_pad*k*sizeof(ggml_fp16_t), 256);
    const size_t total    = inv_size + h_size*products;
    if (ctx->prealloc_size_split_k < total) {
        ctx->prealloc_size_split_k = total;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    if (ctx->prealloc_split_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_pipeline rot = ctx->device->pipeline_exl3_rotate_split;
    vk_pipeline mm  = ctx->device->pipeline_exl3_mm[ri][cb][products - 1];
    GGML_ASSERT(rot && mm);
    ggml_pipeline_request_descriptor_sets(ctx, rot, 1);
    ggml_pipeline_request_descriptor_sets(ctx, mm, 1);

    vk_subbuffer x_buf   = ggml_vk_tensor_subbuffer(ctx, x, true);
    vk_subbuffer inv_buf = { ctx->prealloc_split_k, 0, inv_size };
    vk_subbuffer hi_buf  = { ctx->prealloc_split_k, inv_size, h_size };
    vk_subbuffer lo_buf  = products == 2 ? vk_subbuffer{ ctx->prealloc_split_k, inv_size + h_size, h_size } : hi_buf;

    const vk_op_exl3_rotate_split_push_constants rpc = {
        k, rows, (uint32_t) x->ne[1], (uint32_t) x->ne[2],
        (uint32_t) (x->nb[1]/sizeof(float)), (uint32_t) (x->nb[2]/sizeof(float)), (uint32_t) (x->nb[3]/sizeof(float)),
        (uint32_t) ((x_buf.size - ggml_nbytes(x))/sizeof(float)),
        0, products == 2 ? 1u : 0u, 1, 0, 0,
    };
    ggml_vk_dispatch_pipeline(ctx, subctx, rot,
        { x_buf, ggml_vk_tensor_subbuffer(ctx, suh), hi_buf, lo_buf, inv_buf }, rpc, { rows_pad, 1, 1 });
    ggml_vk_sync_buffers(ctx, subctx);

    const vk_op_exl3_mmv_push_constants mpc = { k, n, rows, 0 };
    ggml_vk_dispatch_pipeline(ctx, subctx, mm,
        { ggml_vk_tensor_subbuffer(ctx, trellis), hi_buf, lo_buf, inv_buf, ggml_vk_tensor_subbuffer(ctx, svh),
          ggml_vk_tensor_subbuffer(ctx, dst) },
        mpc, { n/128, row_blks, 1 });
    ctx->prealloc_split_k_need_sync = true;
}

// EXL3_MUL_MAT_ID: x F32 [k, ne11, n_tokens], ids I32 [n_used, n_tokens], experts stacked in dim 3 of the trellis
bool ggml_vk_exl3_mul_mat_id_supports_op(const vk_device & device, const ggml_tensor * op) {
    const ggml_tensor * trellis = op->src[0];
    const ggml_tensor * x       = op->src[1];
    const ggml_tensor * ids     = op->src[2];
    const ggml_tensor * suh     = op->src[3];
    const ggml_tensor * svh     = op->src[4];
    const int           cb      = ggml_get_op_params_i32(op, 0);
    if (ggml_vk_exl3_disabled() || !ggml_vk_exl3_device_ok(device) || op->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 ||
        x->nb[0] != sizeof(float) || x->nb[1] % sizeof(float) != 0 || x->nb[2] % sizeof(float) != 0 ||
        ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) || ids->nb[1] % sizeof(int32_t) != 0 ||
        !ggml_is_contiguous(op) || !ggml_is_contiguous(trellis) || !ggml_is_contiguous(suh) || !ggml_is_contiguous(svh) ||
        trellis->ne[3] > EXL3_MAX_EXPERTS || !ggml_exl3_rate_supported(trellis->ne[0], (ggml_exl3_codebook) cb)) {
        return false;
    }
    return ggml_nelements(op) < INT32_MAX && ggml_nbytes(x) < UINT32_MAX && ggml_nelements(trellis) < INT32_MAX &&
           (ggml_nelements(ids) + 128)*trellis->ne[2]*16 < INT32_MAX;
}

void ggml_vk_exl3_mul_mat_id(ggml_backend_vk_context * ctx, vk_context & subctx, ggml_tensor * dst) {
    const ggml_tensor * trellis = dst->src[0];
    const ggml_tensor * x       = dst->src[1];
    const ggml_tensor * ids     = dst->src[2];
    const ggml_tensor * suh     = dst->src[3];
    const ggml_tensor * svh     = dst->src[4];
    const int           cb      = ggml_get_op_params_i32(dst, 0);

    const uint32_t n        = (uint32_t) trellis->ne[1]*16;
    const uint32_t k        = (uint32_t) trellis->ne[2]*16;
    const uint32_t n_exp    = (uint32_t) trellis->ne[3];
    const uint32_t n_used   = (uint32_t) ids->ne[0];
    const uint32_t n_tokens = (uint32_t) ids->ne[1];
    const uint32_t n_pairs  = n_used*n_tokens;
    const uint32_t tq_words = (uint32_t) (trellis->ne[0]*trellis->ne[1]*trellis->ne[2]/2);
    const int      ri       = ggml_vk_exl3_rate_idx(trellis->ne[0]);

    // the GEMM (groups of up to 128 rows of one expert) from 8 rows per expert on average, else the dot kernel with
    // groups as wide as the average rows per used expert (1, 2, 4 or 8; one token: every pair is a group of one row)
    const bool     gemm     = n_pairs > EXL3_DOT_MAX_ROWS*n_exp && ctx->device->pipeline_exl3_mm_id[ri][cb][0];
    const uint32_t per_exp  = CEIL_DIV(n_pairs, std::min(n_exp, n_pairs));
    const uint32_t mi       = per_exp <= 1 ? 0 : per_exp <= 2 ? 1 : per_exp <= 4 ? 2 : 3;
    const uint32_t gsize    = gemm ? EXL3_MM_ROWS : 1u << mi;
    const uint32_t g_max    = n_pairs/gsize + std::min(n_exp, n_pairs);

    // scratch: ord [n_pairs], grp [g_max][3], counters (dot), then the rotated rows (GEMM: 128 pad rows past n_pairs)
    const uint32_t n_counters = gemm ? 0 : g_max*(n/128);
    const size_t ord_size  = GGML_PAD((size_t) n_pairs*sizeof(uint32_t), 256);
    const size_t grp_size  = GGML_PAD((size_t) g_max*3*sizeof(uint32_t), 256);
    const size_t cn_size   = GGML_PAD((size_t) std::max(n_counters, 1u)*sizeof(uint32_t), 256);
    const size_t head      = ord_size + grp_size + cn_size;
    // one product for a prompt-sized ubatch (512 tokens or more), hi + lo below; HIP decides per expert (one product
    // from 512 routed rows), so at -ub 512+ most experts take one product here and hi + lo there
    const uint32_t products = ggml_vk_exl3_gemm_products(n_tokens);
    const uint32_t rows_pad = n_pairs + EXL3_MM_ROWS;
    const size_t inv_size  = GGML_PAD((size_t) rows_pad*sizeof(float), 256);
    const size_t h_size    = GGML_PAD((size_t) rows_pad*k*sizeof(ggml_fp16_t), 256);
    const size_t xh_size   = (size_t) n_pairs*k*sizeof(float);
    const size_t total     = head + (gemm ? inv_size + h_size*products : xh_size);
    if (ctx->prealloc_size_split_k < total) {
        ctx->prealloc_size_split_k = total;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    if (ctx->prealloc_split_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_pipeline route = ctx->device->pipeline_exl3_route;
    vk_pipeline rot   = gemm ? ctx->device->pipeline_exl3_rotate_split_id : ctx->device->pipeline_exl3_rotate_id;
    vk_pipeline mm    = gemm ? ctx->device->pipeline_exl3_mm_id[ri][cb][products - 1] : ctx->device->pipeline_exl3_mmv_id[ri][cb][mi];
    GGML_ASSERT(route && rot && mm);
    ggml_pipeline_request_descriptor_sets(ctx, route, 1);
    ggml_pipeline_request_descriptor_sets(ctx, rot, 1);
    ggml_pipeline_request_descriptor_sets(ctx, mm, 1);

    vk_buffer    sb      = ctx->prealloc_split_k;
    vk_subbuffer ids_buf = ggml_vk_tensor_subbuffer(ctx, ids);
    vk_subbuffer ord_buf = { sb, 0, ord_size };
    vk_subbuffer grp_buf = { sb, ord_size, grp_size };
    vk_subbuffer cn_buf  = { sb, ord_size + grp_size, cn_size };
    vk_subbuffer x_buf   = ggml_vk_tensor_subbuffer(ctx, x, true);
    const uint32_t x_off  = (uint32_t) ((x_buf.size - ggml_nbytes(x))/sizeof(float));
    const uint32_t ids_s1 = (uint32_t) (ids->nb[1]/sizeof(int32_t));

    const vk_op_exl3_route_push_constants rpc = { n_used, n_tokens, ids_s1, n_exp, gsize, g_max, n_counters };
    ggml_vk_dispatch_pipeline(ctx, subctx, route, { ids_buf, ord_buf, grp_buf, cn_buf }, rpc, { 1, 1, 1 });
    ggml_vk_sync_buffers(ctx, subctx);

    const vk_op_exl3_mmv_push_constants mpc = { k, n, n_pairs, tq_words };
    if (gemm) {
        vk_subbuffer inv_buf = { sb, head, inv_size };
        vk_subbuffer hi_buf  = { sb, head + inv_size, h_size };
        vk_subbuffer lo_buf  = products == 2 ? vk_subbuffer{ sb, head + inv_size + h_size, h_size } : hi_buf;
        const vk_op_exl3_rotate_split_push_constants spc = {
            k, n_pairs, (uint32_t) x->ne[1], 1,
            (uint32_t) (x->nb[1]/sizeof(float)), (uint32_t) (x->nb[2]/sizeof(float)), 0, x_off,
            0, products == 2 ? 1u : 0u, n_used, ids_s1, n_exp,
        };
        ggml_vk_dispatch_pipeline(ctx, subctx, rot,
            { x_buf, ggml_vk_tensor_subbuffer(ctx, suh), hi_buf, lo_buf, inv_buf, ids_buf, ord_buf }, spc, { rows_pad, 1, 1 });
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, mm,
            { ggml_vk_tensor_subbuffer(ctx, trellis), hi_buf, lo_buf, inv_buf, ggml_vk_tensor_subbuffer(ctx, svh),
              ggml_vk_tensor_subbuffer(ctx, dst), grp_buf, ord_buf },
            mpc, { n/128, g_max, 1 });
    } else {
        vk_subbuffer xh_buf = { sb, head, xh_size };
        const vk_op_exl3_rotate_push_constants xpc = {
            k, n_pairs, (uint32_t) x->ne[1], 1,
            (uint32_t) (x->nb[1]/sizeof(float)), (uint32_t) (x->nb[2]/sizeof(float)), 0, x_off,
            0, n_used, ids_s1, n_exp,
        };
        ggml_vk_dispatch_pipeline(ctx, subctx, rot,
            { x_buf, ggml_vk_tensor_subbuffer(ctx, suh), xh_buf, cn_buf, ids_buf, ord_buf }, xpc,
            { std::max(1u, CEIL_DIV(n_pairs*(k/128), 8u)), 1, 1 });
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, mm,
            { ggml_vk_tensor_subbuffer(ctx, trellis), xh_buf, ggml_vk_tensor_subbuffer(ctx, svh),
              ggml_vk_tensor_subbuffer(ctx, dst), cn_buf, grp_buf, ord_buf },
            mpc, { n/32, g_max, 1 });
    }
    ctx->prealloc_split_k_need_sync = true;
}

void ggml_vk_exl3_mul_mat(ggml_backend_vk_context * ctx, vk_context & subctx, ggml_tensor * dst) {
    const ggml_tensor * trellis = dst->src[0];
    const ggml_tensor * x       = dst->src[1];
    const ggml_tensor * suh     = dst->src[2];
    const ggml_tensor * svh     = dst->src[3];
    const int           cb      = ggml_get_op_params_i32(dst, 0);

    const uint32_t n    = (uint32_t) trellis->ne[1]*16;
    const uint32_t k    = (uint32_t) trellis->ne[2]*16;
    const uint32_t rows = (uint32_t) ggml_nrows(x);
    const int      ri   = ggml_vk_exl3_rate_idx(trellis->ne[0]);

    // more than 8 rows: the coopmat GEMM, if the device has it
    if (rows > EXL3_DOT_MAX_ROWS && ctx->device->pipeline_exl3_mm[ri][cb][0]) {
        ggml_vk_exl3_mul_mm(ctx, subctx, dst, ri, cb);
        return;
    }

    // dot kernel: M rows per pass (1, 2, 4 or 8), more rows in groups of 8
    const uint32_t mi       = rows <= 1 ? 0 : rows <= 2 ? 1 : rows <= 4 ? 2 : 3;
    const uint32_t m        = 1u << mi;
    const uint32_t n_groups = CEIL_DIV(rows, m);

    // scratch: arrival counters, then xh (f32 [rows, k])
    const uint32_t n_counters = n_groups*(n/128);
    const size_t   xh_off     = GGML_PAD((size_t) n_counters*sizeof(uint32_t), 256);
    const size_t   xh_size    = (size_t) rows*k*sizeof(float);
    if (ctx->prealloc_size_split_k < xh_off + xh_size) {
        ctx->prealloc_size_split_k = xh_off + xh_size;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    if (ctx->prealloc_split_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_pipeline rot = ctx->device->pipeline_exl3_rotate;
    vk_pipeline mmv = ctx->device->pipeline_exl3_mmv[ri][cb][mi];
    GGML_ASSERT(rot && mmv);
    ggml_pipeline_request_descriptor_sets(ctx, rot, 1);
    ggml_pipeline_request_descriptor_sets(ctx, mmv, 1);

    vk_subbuffer x_buf  = ggml_vk_tensor_subbuffer(ctx, x, true);
    vk_subbuffer cn_buf = { ctx->prealloc_split_k, 0, xh_off };
    vk_subbuffer xh_buf = { ctx->prealloc_split_k, xh_off, xh_size };

    const vk_op_exl3_rotate_push_constants rpc = {
        k, rows, (uint32_t) x->ne[1], (uint32_t) x->ne[2],
        (uint32_t) (x->nb[1]/sizeof(float)), (uint32_t) (x->nb[2]/sizeof(float)), (uint32_t) (x->nb[3]/sizeof(float)),
        (uint32_t) ((x_buf.size - ggml_nbytes(x))/sizeof(float)),
        n_counters, 1, 0, 0,
    };
    const uint32_t rot_wgs = std::max(1u, CEIL_DIV(rows*(k/128), 8u));
    ggml_vk_dispatch_pipeline(ctx, subctx, rot,
        { x_buf, ggml_vk_tensor_subbuffer(ctx, suh), xh_buf, cn_buf }, rpc, { rot_wgs, 1, 1 });
    ggml_vk_sync_buffers(ctx, subctx);

    const vk_op_exl3_mmv_push_constants mpc = { k, n, rows, 0 };
    ggml_vk_dispatch_pipeline(ctx, subctx, mmv,
        { ggml_vk_tensor_subbuffer(ctx, trellis), xh_buf, ggml_vk_tensor_subbuffer(ctx, svh), ggml_vk_tensor_subbuffer(ctx, dst), cn_buf },
        mpc, { n/32, n_groups, 1 });
    ctx->prealloc_split_k_need_sync = true;
}
#endif
