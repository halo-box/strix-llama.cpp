#include "ggml-hrx-ops.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// The WMMA kernel gets wave64 providers for gfx11 and gfx12 from motifs/flash_attn.loom.
// Other architectures use the scalar kernel.
static bool ggml_hrx_flash_attn_ext_wmma_arch(const ggml_hrx_device_context * dev) {
    const ggml_hrx_matrix_op op = dev->arch->matrix_op;
    return (op == GGML_HRX_MATRIX_OP_WMMA_GFX11 || op == GGML_HRX_MATRIX_OP_WMMA_GFX12) && dev->arch->wave64;
}

// The WMMA kernel loads 16 x 16 K and V fragments directly from the views.
// The HRX base address keeps the 16-byte alignment of the binding offset.
static bool ggml_hrx_flash_attn_ext_use_wmma(const ggml_hrx_device_context * dev, const ggml_tensor * k, const ggml_tensor * v) {
    if (!ggml_hrx_flash_attn_ext_wmma_arch(dev)) {
        return false;
    }
    const int64_t d_k = k->ne[0];
    const int64_t d_v = v->ne[0];
    if (d_k % 16 != 0 || d_v % 16 != 0 || d_k > 256 || d_v > 256 || k->ne[1] < 16) {
        return false;
    }
    if (k->ne[2] != v->ne[2] || k->ne[3] != v->ne[3]) {
        return false;
    }
    for (const ggml_tensor * t : { k, v }) {
        if ((uintptr_t) t->data % 16 != 0 || t->nb[1] % 16 != 0 || t->nb[2] % 16 != 0 || t->nb[3] % 16 != 0) {
            return false;
        }
        // Fragment addresses in one 16-token window must fit in 32 bits.
        if (t->nb[1] / 2 > 16777216) {
            return false;
        }
    }
    return true;
}

static bool ggml_hrx_fa_wide_rows(const ggml_tensor * op) {
    const ggml_tensor * mask = op->src[3];
    const ggml_tensor * ids = ggml_flash_attn_ext_get_bias_indices(op);
    return (mask && ggml_nbytes(mask) / sizeof(ggml_fp16_t) >= INT_MAX) ||
           (ids && ggml_nbytes(ids) / sizeof(int32_t) >= INT_MAX);
}

bool ggml_hrx_supports_flash_attn_ext(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    const ggml_tensor * q    = op->src[0];
    const ggml_tensor * k    = op->src[1];
    const ggml_tensor * v    = op->src[2];
    const ggml_tensor * mask  = op->src[3];
    const ggml_tensor * sinks = op->src[4];
    if (q == nullptr || k == nullptr || v == nullptr) {
        return false;
    }
    // kv_rows are not supported.
    if (op->src[5] != nullptr) {
        return false;
    }
    if (sinks != nullptr && (sinks->type != GGML_TYPE_F32 || sinks->ne[0] != q->ne[2] || !ggml_is_contiguous(sinks))) {
        return false;
    }
    // A selected-key list (src[6]) is only a hint when there is a mask. Without a mask it holds the visibility.
    if ((op->src[6] != nullptr && mask == nullptr) || op->src[7] != nullptr) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        return false;
    }
    if (mask != nullptr && mask->type != GGML_TYPE_F16) {
        return false;
    }
    // ALiBi and logit softcapping are not supported.
    if (ggml_get_op_params_f32(op, 1) != 0.0f || ggml_get_op_params_f32(op, 2) != 0.0f) {
        return false;
    }

    const int64_t d_k  = k->ne[0];
    const int64_t d_v  = v->ne[0];
    const int64_t n_kv = k->ne[1];

    // The scalar kernel holds at most 256 channels per head.
    if (d_k < 1 || d_k > 256 || d_v < 1 || d_v > 256 || q->ne[0] != d_k || v->ne[1] != n_kv) {
        return false;
    }
    if (k->ne[2] < 1 || k->ne[3] < 1 || v->ne[2] < 1 || v->ne[3] < 1) {
        return false;
    }
    if (q->ne[2] % k->ne[2] != 0 || q->ne[2] % v->ne[2] != 0 || q->ne[3] % k->ne[3] != 0 || q->ne[3] % v->ne[3] != 0) {
        return false;
    }
    if (q->nb[0] != sizeof(float) || k->nb[0] != sizeof(ggml_fp16_t) || v->nb[0] != sizeof(ggml_fp16_t)) {
        return false;
    }
    if (op->ne[0] != d_v || op->ne[1] != q->ne[2] || op->ne[2] != q->ne[1] || op->ne[3] != q->ne[3] ||
        !ggml_is_contiguous(op)) {
        return false;
    }
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(q, s) || !ggml_hrx_elem_strides(k, s) || !ggml_hrx_elem_strides(v, s)) {
        return false;
    }
    if (mask != nullptr) {
        if (mask->nb[0] != sizeof(ggml_fp16_t) || mask->ne[0] < n_kv || mask->ne[1] < q->ne[1] ||
            mask->ne[2] < 1 || mask->ne[3] < 1 || q->ne[2] % mask->ne[2] != 0 || q->ne[3] % mask->ne[3] != 0 || !ggml_hrx_elem_strides(mask, s)) {
            return false;
        }
        if (ggml_hrx_fa_wide_rows(op) &&
            (!ggml_hrx_flash_attn_ext_use_wmma(dev, k, v) || !ggml_is_contiguous(mask) ||
             mask->ne[0] != n_kv || mask->ne[1] != q->ne[1])) {
            return false;
        }
    }
    if (ggml_flash_attn_ext_has_indexed_bias(op)) {
        const ggml_tensor * table   = ggml_flash_attn_ext_get_bias_table(op);
        const ggml_tensor * indices = ggml_flash_attn_ext_get_bias_indices(op);
        if (mask == nullptr || table == nullptr || indices == nullptr || table->type != GGML_TYPE_F32 ||
            indices->type != GGML_TYPE_I32 || !ggml_is_contiguous(table) || !ggml_is_contiguous(indices) ||
            table->ne[1] != q->ne[2] || table->ne[2] != q->ne[1] * q->ne[3] || indices->ne[1] != table->ne[2] ||
            indices->ne[0] < n_kv || ggml_nbytes(table) / 4 >= INT_MAX) {
            return false;
        }
    }
    // Element indices and spans are 32-bit launch scalars.
    return ggml_nelements(op) < INT_MAX && ggml_nbytes(q) / 4 < INT_MAX && ggml_nbytes(k) / 2 < INT_MAX &&
           ggml_nbytes(v) / 2 < INT_MAX;
}

// Launch scalars and bindings of the indexed bias. Without a bias, q fills the bindings and the kernels do not read them.
// sub: a SUB(table, s) node fused into the FLASH_ATTN_EXT. The kernel reads its sources and subtracts s itself.
struct ggml_hrx_fa_bias_args {
    bool                on      = false;
    int64_t             n_bins  = 1;
    int64_t             i_s1    = 0;
    int64_t             t_span  = 1;
    int64_t             i_span  = 1;
    const ggml_tensor * table   = nullptr;
    const ggml_tensor * indices = nullptr;
    bool                sub_on  = false;
    int64_t             s_s1    = 0;
    int64_t             s_s2    = 0;
    int64_t             s_span  = 1;
    const ggml_tensor * s       = nullptr;
    // mm: the MUL_MAT(w, r) that makes the table, fused into the node. The table binding then holds w.
    bool                mm_on   = false;
    int64_t             w_s1    = 0;
    int64_t             r_s1    = 0;
    int64_t             r_s2    = 0;
    int64_t             r_span  = 1;
    const ggml_tensor * r       = nullptr;
};

static ggml_hrx_fa_bias_args ggml_hrx_fa_bias(const ggml_tensor * node, const ggml_tensor * sub, const ggml_tensor * mm) {
    ggml_hrx_fa_bias_args b;
    b.table   = node->src[0];
    b.indices = node->src[0];
    b.s       = node->src[0];
    b.r       = node->src[0];
    if (!ggml_flash_attn_ext_has_indexed_bias(node)) {
        return b;
    }
    const ggml_tensor * table   = sub != nullptr ? sub->src[0] : ggml_flash_attn_ext_get_bias_table(node);
    const ggml_tensor * indices = ggml_flash_attn_ext_get_bias_indices(node);
    b.on      = true;
    b.n_bins  = table->ne[0];
    b.i_s1    = indices->ne[0];
    b.t_span  = (int64_t) (ggml_nbytes(table) / 4);
    b.i_span  = (int64_t) (ggml_nbytes(indices) / 4);
    b.table   = table;
    b.indices = indices;
    if (sub != nullptr) {
        b.sub_on = true;
        b.s      = sub->src[1];
        b.s_s1   = (int64_t) (b.s->nb[1] / 4);
        b.s_s2   = (int64_t) (b.s->nb[2] / 4);
        b.s_span = (int64_t) (ggml_nbytes(b.s) / 4);
    }
    if (mm != nullptr) {
        b.mm_on  = true;
        b.table  = mm->src[0];
        b.t_span = (int64_t) (ggml_nbytes(b.table) / 4);
        b.w_s1   = (int64_t) (b.table->nb[1] / 4);
        b.r      = mm->src[1];
        b.r_s1   = (int64_t) (b.r->nb[1] / 4);
        b.r_s2   = (int64_t) (b.r->nb[2] / 4);
        b.r_span = (int64_t) (ggml_nbytes(b.r) / 4);
    }
    return b;
}

// The tile kernel keeps the classes of at most 65536 keys in LDS. It skips the tiles that no row of a workgroup sees and adds the
// bias of a far tile without loads, so it beats the WMMA kernel on long KV too. Measured on gfx1151 with 4096 queries, 24/8
// heads, causal mask and indexed bias (test-backend-ops perf): 32768 keys 84 ms against 171 ms.
static constexpr int64_t GGML_HRX_FA_TILE_MAX_KV = 65536;

// Up to this many keys the tile kernel runs workgroups of 2 query tiles (6 waves for 3 heads of a KV head), else 6 query tiles
// with the prefetch (18 waves). Measured on gfx1151, 4096 queries, 24/8 heads, indexed bias (test-backend-ops perf): window
// 1024 and 5120 keys 6.7 ms against 9.4 ms, causal 4096 keys 9.4 ms against 9.7 ms, causal 8192 keys 28.2 ms against 19.9 ms.
static constexpr int64_t GGML_HRX_FA_TILE_SHORT_KV = 6144;

// The tile kernel (flash_attn_ext_f32_f16_tile.loom) takes batches of 128-channel heads on gfx11.
static bool ggml_hrx_flash_attn_ext_use_tile(const ggml_hrx_device_context * dev, const ggml_tensor * q, const ggml_tensor * k,
                                             const ggml_tensor * v) {
    return dev->arch->matrix_op == GGML_HRX_MATRIX_OP_WMMA_GFX11 && q->ne[1] >= 64 && k->ne[0] == 128 && v->ne[0] == 128 &&
           k->ne[1] <= GGML_HRX_FA_TILE_MAX_KV;
}

// The tile kernel keeps the tile classes of a workgroup in LDS: one mask head row and at most 64 words of 32 tiles (65536 keys).
// The classes take 36 values for each word of a block row (flash_attn_ext_range_f16.loom).
static bool ggml_hrx_flash_attn_ext_tile_ok(const ggml_hrx_device_context * dev, const ggml_tensor * node) {
    const ggml_tensor * mask    = node->src[3];
    const int64_t       n_words = ((node->src[1]->ne[1] + 31) / 32 + 31) / 32;
    return !ggml_hrx_fa_wide_rows(node) && node->src[0]->ne[1] > dev->tune->fa_small_batch && ggml_hrx_flash_attn_ext_use_tile(dev, node->src[0], node->src[1], node->src[2]) &&
           (mask == nullptr || (mask->ne[2] == 1 && n_words <= 64 && 36 * ((node->src[0]->ne[1] + 15) / 16) * mask->ne[3] * n_words < INT_MAX));
}

// A graph input or a leaf: the graph does not write it, so it holds the same values for all nodes.
static bool ggml_hrx_fa_cls_static(const ggml_tensor * t) {
    const ggml_tensor * b = t->view_src != nullptr ? t->view_src : t;
    return (b->flags & GGML_TENSOR_FLAG_INPUT) != 0 || b->op == GGML_OP_NONE;
}

// The slot for the key. hit is true if it holds the classes of the key for the graph that runs.
static ggml_hrx_fa_cls_slot * ggml_hrx_fa_cls_get(ggml_backend_hrx_context * ctx, const int64_t (&key)[12], bool cache, size_t bytes, bool & hit) {
    hit = false;
    auto * slot = &ctx->fa_cls_slots[0];
    for (auto & sl : ctx->fa_cls_slots) {
        if (cache && sl.tensor.buffer && memcmp(sl.key, key, sizeof(key)) == 0 && sl.tensor.ne[0] >= (int64_t) bytes) {
            hit = sl.graph == ctx->n_graphs_seen;
            slot = &sl;
            break;
        }
        if (sl.used < slot->used) {
            slot = &sl;
        }
    }
    ggml_tensor & t = slot->tensor;
    if (t.ne[0] < (int64_t) bytes) {
        bytes = std::max(bytes, (size_t) t.ne[0] * 2);
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(&ctx->device->buft, bytes);
        if (buffer == nullptr) {
            return nullptr;
        }
        // Recorded graphs can still refer to earlier allocations.
        ctx->fa_cls_buffers.push_back(buffer);
        t.type   = GGML_TYPE_I8;
        t.buffer = buffer;
        t.data   = ggml_backend_buffer_get_base(buffer);
        t.ne[0]  = (int64_t) bytes;
        t.ne[1] = t.ne[2] = t.ne[3] = 1;
        t.nb[0] = 1;
        t.nb[1] = t.nb[2] = t.nb[3] = bytes;
    }
    slot->graph = cache ? ctx->n_graphs_seen : -1;
    memcpy(slot->key, key, sizeof(key));
    slot->used = ++ctx->fa_cls_clock;
    return slot;
}

// The tile kernel loads 16 mask values and 16 indices at once if the rows are aligned and hold the keys of the last tile.
static bool ggml_hrx_flash_attn_ext_vec_mask(const ggml_tensor * node) {
    const ggml_tensor * mask = node->src[3];
    const int64_t       kv32 = (node->src[1]->ne[1] + 31) / 32 * 32;
    auto rows_ok = [&](const ggml_tensor * t) {
        const size_t a = 16 * ggml_type_size(t->type);
        return t->ne[0] >= kv32 && (uintptr_t) t->data % a == 0 && t->nb[1] % a == 0 && t->nb[2] % a == 0 && t->nb[3] % a == 0;
    };
    return mask != nullptr && rows_ok(mask) && (!ggml_flash_attn_ext_has_indexed_bias(node) || rows_ok(ggml_flash_attn_ext_get_bias_indices(node)));
}

static int64_t ggml_hrx_flash_attn_ext_largest_divisor(int64_t n, int64_t limit) {
    int64_t best = 1;
    for (int64_t d = 2; d <= std::min(n, limit); ++d) {
        if (n % d == 0) {
            best = d;
        }
    }
    return best;
}

// More heads per tile share each K/V load. Fewer heads per tile give more workgroups.
static int64_t ggml_hrx_flash_attn_ext_heads_per_tile(const ggml_hrx_tune * tune, int64_t n_q, int64_t n_head, int64_t ratio,
                                                      int64_t n_seq) {
    auto n_wg = [&](int64_t hg) {
        return (n_q * hg + 15) / 16 * (n_head / hg) * n_seq;
    };
    const int64_t target = std::min(tune->fa_target_workgroups, n_wg(1));
    int64_t       best   = 1;
    for (int64_t hg = 2; hg <= std::min<int64_t>(ratio, 16); ++hg) {
        if (ratio % hg == 0 && n_wg(hg) >= target) {
            best = hg;
        }
    }
    return best;
}

// The waves of a tile split its KV. Few tiles get more waves.
static int64_t ggml_hrx_flash_attn_ext_waves(const ggml_hrx_tune * tune, int64_t n_tiles, int64_t n_kv) {
    const int64_t n_sub = (n_kv + 15) / 16;
    if (n_tiles >= tune->fa_many_tiles) {
        return 4;
    }
    if (n_sub >= tune->fa_long_kv_subblocks) {
        return 16;
    }
    return n_sub >= tune->fa_mid_kv_subblocks ? 8 : 4;
}

// sub: see ggml_hrx_fa_bias_args. Only the WMMA kernel takes it. s_copy: the FA output shares memory with the SUB value s,
// so some workgroups would write it while others still read s. The kernel then reads a copy of s in the scratch.
static bool ggml_hrx_flash_attn_ext_run(ggml_backend_hrx_context * ctx, const ggml_tensor * node, const ggml_tensor * sub, bool s_copy,
                                        const ggml_tensor * mm = nullptr) {
    const ggml_tensor * q    = node->src[0];
    const ggml_tensor * k    = node->src[1];
    const ggml_tensor * v    = node->src[2];
    const ggml_tensor * mask  = node->src[3];
    const ggml_tensor * sinks = node->src[4];

    int64_t qs[GGML_MAX_DIMS];
    int64_t ks[GGML_MAX_DIMS];
    int64_t vs[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(q, qs);
    ggml_hrx_elem_strides(k, ks);
    ggml_hrx_elem_strides(v, vs);

    // Without a mask or sinks, the kernel does not read that binding. Q fills it.
    int64_t ms[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    int64_t m_ne2  = 1;
    int64_t m_ne3  = 1;
    int64_t m_span = 1;
    if (mask != nullptr) {
        ggml_hrx_elem_strides(mask, ms);
        m_ne2  = mask->ne[2];
        m_ne3  = mask->ne[3];
        m_span = (int64_t) (ggml_nbytes(mask) / 2);
    }

    const float scale = ggml_get_op_params_f32(node, 0);
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &scale, sizeof(scale_bits));

    ggml_hrx_fa_bias_args bias = ggml_hrx_fa_bias(node, sub, mm);

    if (ggml_hrx_flash_attn_ext_use_wmma(ctx->device, k, v)) {
        const int64_t n_q    = q->ne[1];
        const int64_t n_head = q->ne[2];
        const int64_t n_seq  = q->ne[3];
        const int64_t n_kv   = k->ne[1];
        const int64_t ratio  = n_head / k->ne[2];
        const ggml_hrx_tune * tune = ctx->device->tune;
        const int64_t n_sub  = (n_kv + 15) / 16;
        const bool    small  = n_q <= tune->fa_small_batch;
        int64_t       hg     = ggml_hrx_flash_attn_ext_heads_per_tile(tune, n_q, n_head, ratio, n_seq);
        int64_t       n_x    = (n_q * hg + 15) / 16;
        int64_t       waves  = ggml_hrx_flash_attn_ext_waves(tune, n_x * (n_head / hg) * n_seq, n_kv);
        int64_t       chunk  = n_sub;
        if (small) {
            // Small batches split the KV across workgroups. All heads of a KV head share one tile.
            // The split depends only on n_kv, so a row has the same reduction order for each batch size.
            hg    = ggml_hrx_flash_attn_ext_largest_divisor(ratio, 16);
            n_x   = (n_q * hg + 15) / 16;
            waves = tune->fa_split_waves;
            chunk = std::max(tune->fa_min_split_subblocks, (n_sub + tune->fa_max_splits - 1) / tune->fa_max_splits);
        }
        const int64_t n_split = (n_sub + chunk - 1) / chunk;

        const ggml_tensor * part      = q;
        int64_t             part_span = 1;
        if (n_split > 1) {
            part_span = n_split * (ggml_nelements(node) + 2 * ggml_nelements(node) / v->ne[0]);
            if (part_span >= INT_MAX) {
                return false;
            }
        }
        // The classes of the 32-key tiles for each block of 16 rows (query block, sequence), see flash_attn_ext_range_f16.loom.
        const int64_t n_qb      = (n_q + 15) / 16;
        const int64_t n_rowsb   = n_qb * m_ne2 * m_ne3;
        const int64_t n_words   = ((n_kv + 31) / 32 + 31) / 32;
        const bool    use_tile  = ggml_hrx_flash_attn_ext_tile_ok(ctx->device, node);
        const bool    use_range = use_tile && mask != nullptr;
        // Tile kernel config: hg heads and qt query tiles a workgroup. Long KV: 16 or more waves, and waves 0-15 prefetch the
        // next K/V tile into registers. Short KV: small workgroups, several of them on a WGP.
        const int64_t thg = ggml_hrx_flash_attn_ext_largest_divisor(ratio, 8);
        const bool    pf  = n_kv > GGML_HRX_FA_TILE_SHORT_KV && (16 + thg - 1) / thg <= 8;
        const int64_t tqt = pf ? (16 + thg - 1) / thg : std::max<int64_t>(1, std::min<int64_t>(4, 8 / thg));
        // Toeplitz tiles get the bias from matrix products. Their LDS (1728 bytes a wave) fits 18 waves.
        const bool    toep      = use_range && bias.mm_on && thg * tqt <= 18;
        const int64_t r_span    = use_range ? 36 * n_rowsb * n_words : 2;
        const ggml_tensor * cls = q;
        bool                cls_hit = false;
        if (use_range) {
            const int64_t key[12] = { (int64_t) (intptr_t) mask->data, (int64_t) (intptr_t) bias.indices->data, n_q, n_kv, m_ne2, m_ne3,
                                      ms[1], ms[2], ms[3], bias.on ? bias.n_bins : 0, bias.i_s1, n_rowsb };
            const bool cache = ggml_hrx_fa_cls_static(mask) && (!bias.on || ggml_hrx_fa_cls_static(bias.indices));
            ggml_hrx_fa_cls_slot * slot = ggml_hrx_fa_cls_get(ctx, key, cache, (size_t) r_span * sizeof(int32_t), cls_hit);
            if (slot == nullptr) {
                return false;
            }
            cls = &slot->tensor;
        }
        // The copy of s follows the partial results in the scratch.
        const size_t s_offset   = n_split > 1 ? ((size_t) part_span * sizeof(float) + 255) / 256 * 256 : 0;
        const size_t part_bytes = s_offset + (s_copy ? ggml_nbytes(bias.s) : 0);
        if (n_split > 1 || s_copy) {
            part = ggml_hrx_scratch(ctx, part_bytes);
            if (part == nullptr) {
                return false;
            }
        }
        // The classes, unless an earlier node of the graph made them.
        if (use_range && !cls_hit) {
            ggml_hrx_kernel_desc rdesc;
            rdesc.file   = "flash_attn/flash_attn_ext_range_f16.loom";
            rdesc.symbol = "ggml_flash_attn_ext_range_f16";
            rdesc.config = { { "ggml.flash_attn.bias", bias.on ? "1" : "0" } };
            std::vector<const ggml_tensor *> cls_writes = { cls };
            const std::vector<const ggml_tensor *> * old_writes = ctx->dispatch_writes;
            ctx->dispatch_writes = &cls_writes;
            const bool ok = ggml_hrx_dispatch(ctx, ggml_hrx_get_kernel(ctx->device, rdesc), { n_rowsb, n_words },
                    { n_kv, n_q, m_ne2, ms[1], ms[2], ms[3], m_span, bias.i_s1, bias.i_span, bias.n_bins, n_rowsb, n_words, 0, r_span },
                    { mask, bias.indices, cls });
            ctx->dispatch_writes = old_writes;
            if (!ok) {
                return false;
            }
        }
        // The recorder must see the writes of the copy and of the FA, see ggml_hrx_fuse_fn.
        ggml_tensor                      s_tmp;
        std::vector<const ggml_tensor *> writes = { node };
        if (sub != nullptr) {
            ctx->dispatch_writes = &writes;
        }
        if (s_copy) {
            writes.push_back(&s_tmp);
            s_tmp        = *bias.s;
            s_tmp.op     = GGML_OP_CPY;
            s_tmp.src[0] = const_cast<ggml_tensor *>(bias.s);
            s_tmp.src[1] = nullptr;
            s_tmp.view_src  = nullptr;
            s_tmp.view_offs = 0;
            s_tmp.buffer    = part->buffer;
            s_tmp.data      = (char *) part->data + s_offset;
            s_tmp.nb[0]     = sizeof(float);
            for (int d = 1; d < GGML_MAX_DIMS; ++d) {
                s_tmp.nb[d] = s_tmp.nb[d - 1] * s_tmp.ne[d - 1];
            }
            if (!ggml_hrx_op_cpy(ctx, &s_tmp)) {
                return false;
            }
            bias.s      = &s_tmp;
            bias.s_s1   = (int64_t) (s_tmp.nb[1] / 4);
            bias.s_s2   = (int64_t) (s_tmp.nb[2] / 4);
            bias.s_span = (int64_t) (ggml_nbytes(&s_tmp) / 4);
            if (n_split == 1 && !use_range) {
                part = q;
            }
        }

        if (use_tile) {
            const bool vec_mask = ggml_hrx_flash_attn_ext_vec_mask(node);
            if (bias.mm_on && !vec_mask) {
                return false;
            }
            ggml_hrx_kernel_desc tdesc;
            tdesc.file   = "flash_attn/flash_attn_ext_f32_f16_tile.loom";
            tdesc.symbol = "ggml_flash_attn_ext_f32_f16_tile";
            // On gfx1151 the 5120-key Inkling window took 6.31 ms with adjacent head groups and 7.61-7.91 ms with adjacent query tiles.
            const bool head_first = strcmp(ctx->device->arch->name, "gfx1151") == 0 && n_q == 4096 && n_kv == 5120 &&
                                    n_head == 24 && ratio == 3 && bias.mm_on;
            tdesc.config = { { "ggml.flash_attn.hg", std::to_string(thg) }, { "ggml.flash_attn.qt", std::to_string(tqt) },
                             { "ggml.flash_attn.mask", mask != nullptr ? "1" : "0" }, { "ggml.flash_attn.bias", bias.on ? "1" : "0" },
                             { "ggml.flash_attn.bias_sub", bias.sub_on ? "1" : "0" }, { "ggml.flash_attn.vec_mask", vec_mask ? "1" : "0" },
                             { "ggml.flash_attn.bias_mm", bias.mm_on ? "1" : "0" }, { "ggml.flash_attn.prefetch", pf ? "1" : "0" },
                             { "ggml.flash_attn.toep", toep ? "1" : "0" }, { "ggml.flash_attn.head_first", head_first ? "1" : "0" } };
            return ggml_hrx_dispatch(ctx, ggml_hrx_get_kernel(ctx->device, tdesc), { (n_q + 16 * tqt - 1) / (16 * tqt), n_head / thg, n_seq },
                {
                    n_q, n_head, n_kv, ratio, n_seq / k->ne[3],
                    qs[1], qs[2], qs[3], (int64_t) (ggml_nbytes(q) / 4),
                    ks[1], ks[2], ks[3], vs[1], vs[2], vs[3],
                    m_ne2, m_ne3, ms[1], ms[2], ms[3], m_span,
                    ggml_nelements(node), (int64_t) scale_bits,
                    sinks != nullptr ? 1 : 0, sinks != nullptr ? sinks->ne[0] : 1,
                    bias.n_bins, bias.i_s1, bias.t_span, bias.i_span,
                    bias.s_s1, bias.s_s2, bias.s_span, use_range ? 1 : 0, 0, r_span, n_qb, n_rowsb, n_words,
                    bias.w_s1, bias.r_s1, bias.r_s2, bias.r_span,
                },
                { q, k, v, mask != nullptr ? mask : q, sinks != nullptr ? sinks : q, node, cls, bias.table, bias.indices, bias.s, bias.r });
        }
        if (bias.mm_on) {
            return false;
        }

        ggml_hrx_kernel_desc desc;
        desc.file   = "flash_attn/flash_attn_ext_f32_f16_wmma.loom";
        desc.symbol = "ggml_flash_attn_ext_f32_f16_wmma";
        desc.libs   = { "motifs/flash_attn.loom", "motifs/fa_bias.loom" };
        // The head groups vary fastest in the grid when the tile count fits in y. The workgroups that run at the same time
        // then read all heads of the same tokens: K/V rows of the cache, mask and bias indices. Measured on gfx1151 with
        // the KV cache layout (8 KV heads next to each other), 2048 queries, 34816 keys: 1080 ms with the tiles fastest, 105 ms.
        const bool group_major = n_x <= (int64_t) ctx->device->arch->max_workgroups_yz;
        const bool wide = ggml_hrx_fa_wide_rows(node);
        auto get_kernel = [&](int64_t count) {
            desc.config = { { "ggml.flash_attn.waves", std::to_string(count) }, { "ggml.flash_attn.bias", bias.on ? "1" : "0" },
                            { "ggml.flash_attn.wide_rows", wide ? "1" : "0" }, { "ggml.flash_attn.group_major", group_major ? "1" : "0" }, { "ggml.flash_attn.bias_sub", bias.sub_on ? "1" : "0" } };
            return ggml_hrx_get_kernel(ctx->device, desc);
        };
        ggml_hrx_kernel * kernel = get_kernel(waves);

        // On gfx1151 a 16-wave fused bias kernel can exceed the register limit. Keep 16 waves when the compiler accepts it.
        if (kernel == nullptr && waves == 16) {
            kernel = get_kernel(8);
        }
        if (!ggml_hrx_dispatch(ctx, kernel, { n_x, n_head / hg, n_seq * n_split },
                {
                    n_q, n_head, k->ne[0], v->ne[0], n_kv, hg, ratio, n_seq / k->ne[3],
                    qs[1], qs[2], qs[3], (int64_t) (ggml_nbytes(q) / 4),
                    ks[1], ks[2], ks[3], vs[1], vs[2], vs[3],
                    mask != nullptr ? 1 : 0, m_ne2, m_ne3, wide ? 0 : ms[1], wide ? 0 : ms[2], wide ? 0 : ms[3], wide ? n_kv : m_span,
                    ggml_nelements(node), (int64_t) scale_bits, n_split, chunk, part_span,
                    sinks != nullptr ? 1 : 0, sinks != nullptr ? sinks->ne[0] : 1,
                    bias.n_bins, bias.i_s1, bias.t_span, wide && bias.on ? n_kv : bias.i_span,
                    bias.s_s1, bias.s_s2, bias.s_span,
                },
                { q, k, v, mask != nullptr ? mask : q, sinks != nullptr ? sinks : q, node, part, bias.table, bias.indices, bias.s })) {
            return false;
        }
        if (n_split == 1) {
            return true;
        }

        ggml_hrx_kernel_desc rdesc;
        rdesc.file   = "flash_attn/flash_attn_ext_reduce_f32.loom";
        rdesc.symbol = "ggml_flash_attn_ext_reduce_f32";
        ggml_hrx_kernel * reduce = ggml_hrx_get_kernel(ctx->device, rdesc);
        return ggml_hrx_dispatch(ctx, reduce, { ggml_nelements(node) / v->ne[0] },
            { n_split, v->ne[0], ggml_nelements(node), part_span }, { part, node });
    }

    if (sub != nullptr) {
        return false;
    }

    ggml_hrx_kernel_desc desc;
    desc.file   = "flash_attn/flash_attn_ext_f32_f16.loom";
    desc.symbol = "ggml_flash_attn_ext_f32_f16";
    desc.libs   = { "motifs/fa_bias.loom" };
    desc.config = { { "ggml.flash_attn.bias", bias.on ? "1" : "0" } };
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    const int64_t n_q    = q->ne[1];
    const int64_t n_head = q->ne[2];
    const int64_t n_seq  = q->ne[3];
    return ggml_hrx_dispatch(ctx, kernel, { n_q, n_head, n_seq },
        {
            n_q, n_head, n_seq, k->ne[0], v->ne[0], k->ne[1],
            qs[1], qs[2], qs[3], (int64_t) (ggml_nbytes(q) / 4),
            n_head / k->ne[2], n_seq / k->ne[3], ks[1], ks[2], ks[3], (int64_t) (ggml_nbytes(k) / 2),
            n_head / v->ne[2], n_seq / v->ne[3], vs[1], vs[2], vs[3], (int64_t) (ggml_nbytes(v) / 2),
            mask != nullptr ? 1 : 0, m_ne2, m_ne3, ms[1], ms[2], ms[3], m_span,
            ggml_nelements(node), (int64_t) scale_bits,
            sinks != nullptr ? 1 : 0, sinks != nullptr ? sinks->ne[0] : 1,
            bias.n_bins, bias.i_s1, bias.t_span, bias.i_span,
        },
        { q, k, v, mask != nullptr ? mask : q, sinks != nullptr ? sinks : q, node, bias.table, bias.indices });
}

bool ggml_hrx_op_flash_attn_ext(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    return ggml_hrx_flash_attn_ext_run(ctx, node, nullptr, false);
}

// SUB(table, s) right before the FLASH_ATTN_EXT that takes it as the bias table, with s of shape [1, n_head, n_q]:
// the FA kernel subtracts s from the table values itself. This saves a read and a write of the table.
// The result is the same: the kernel does the same f32 subtraction before it adds the mask.
// The FLASH_ATTN_EXT node of the SUB fusion at node i, or -1.
static int ggml_hrx_fa_bias_sub_match(const ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, bool * s_copy) {
    const ggml_tensor * sub = cgraph->nodes[i];
    if (sub->op != GGML_OP_SUB || sub->type != GGML_TYPE_F32 || (sub->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return -1;
    }
    const int j = ggml_hrx_next_node(cgraph, i);
    if (j < 0) {
        return -1;
    }
    const ggml_tensor * fa = cgraph->nodes[j];
    if (fa->op != GGML_OP_FLASH_ATTN_EXT || !ggml_flash_attn_ext_has_indexed_bias(fa) || ggml_flash_attn_ext_get_bias_table(fa) != sub ||
        !ggml_hrx_flash_attn_ext_use_wmma(ctx->device, fa->src[1], fa->src[2])) {
        return -1;
    }
    const ggml_tensor * a = sub->src[0];
    const ggml_tensor * s = sub->src[1];
    if (a->type != GGML_TYPE_F32 || s->type != GGML_TYPE_F32 || !ggml_is_contiguous(a) || !ggml_are_same_shape(a, sub) ||
        s->ne[0] != 1 || s->ne[1] != a->ne[1] || s->ne[2] != a->ne[2] || s->ne[3] != 1 || a->ne[3] != 1 ||
        s->nb[1] % 4 != 0 || s->nb[2] % 4 != 0 || ggml_nbytes(s) / 4 >= INT_MAX) {
        return -1;
    }
    // The FA output must not take the memory of the table. If it takes the memory of s, the kernel reads a copy of s.
    if (ggml_hrx_overlap(fa, a)) {
        return -1;
    }
    *s_copy = ggml_hrx_overlap(fa, s);
    // The bias table of the FA must be the only read of the SUB result.
    int n_reads = 0;
    for (int k = i + 1; k < cgraph->n_nodes; ++k) {
        const ggml_tensor * n = cgraph->nodes[k];
        for (int m = 0; m < GGML_MAX_SRC; ++m) {
            n_reads += n->src[m] != nullptr && (n->src[m] == sub || n->src[m]->view_src == sub);
        }
    }
    if (n_reads != 1) {
        return -1;
    }
    return j;
}

int ggml_hrx_fuse_flash_attn_bias_sub(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, std::vector<int> * absorbed) {
    bool      s_copy = false;
    const int j      = ggml_hrx_fa_bias_sub_match(ctx, cgraph, i, &s_copy);
    if (j < 0 || absorbed != nullptr) {
        return j;
    }
    if (!ggml_hrx_flash_attn_ext_run(ctx, cgraph->nodes[j], cgraph->nodes[i], s_copy)) {
        GGML_ABORT("%s: failed to run node %d (%s)", __func__, j, cgraph->nodes[j]->name);
    }
    ctx->dispatch_writes = nullptr;
    return j;
}

// The MUL_MAT fusion below reads w and r of the table MUL_MAT when the FLASH_ATTN_EXT runs. Thus they stay allocated until then.
void ggml_hrx_flash_attn_alloc_deps(ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    static const bool disabled = getenv("GGML_HRX_DISABLE_FUSION") != nullptr;
    if (disabled || params == nullptr || params->add_alloc_dep == nullptr) {
        return;
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * fa = cgraph->nodes[i];
        if (fa->op != GGML_OP_FLASH_ATTN_EXT || !ggml_flash_attn_ext_has_indexed_bias(fa)) {
            continue;
        }
        const ggml_tensor * sub = ggml_flash_attn_ext_get_bias_table(fa);
        if (sub->op != GGML_OP_SUB || sub->src[0]->op != GGML_OP_MUL_MAT) {
            continue;
        }
        for (ggml_tensor * t : { sub->src[0]->src[0], sub->src[0]->src[1] }) {
            params->add_alloc_dep(params->user_data, t->view_src != nullptr ? t->view_src : t, fa);
        }
    }
}

// MUL_MAT(w, r) -> SUB(table, s) -> FLASH_ATTN_EXT with that table: the tile kernel computes the table values of the keys it reads
// from w and r. The MUL_MAT does not run. This saves the write and the read of the table, [n_bins, n_head, n_q] floats. A plain
// tile takes one value for all its keys, a Toeplitz tile 3 matrix products with f16 inputs, other tiles 16 products for each key
// with the sums of ggml_mul_mat_small_k. Measured on gfx1151, 4096 queries, 24/8 heads, 1025 bins, causal 32768 keys: 95 ms
// against 95 ms + 3 ms (FA + table).
int ggml_hrx_fuse_flash_attn_bias_mm(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, int i, std::vector<int> * absorbed) {
    const ggml_tensor * sub = cgraph->nodes[i];
    if (sub->op != GGML_OP_SUB || sub->src[0]->op != GGML_OP_MUL_MAT) {
        return -1;
    }
    bool      s_copy = false;
    const int j      = ggml_hrx_fa_bias_sub_match(ctx, cgraph, i, &s_copy);
    if (j < 0) {
        return -1;
    }
    const ggml_tensor * fa = cgraph->nodes[j];
    const ggml_tensor * mm = sub->src[0];
    const ggml_tensor * w  = mm->src[0];
    const ggml_tensor * r  = mm->src[1];
    if (!ggml_hrx_flash_attn_ext_tile_ok(ctx->device, fa) ||
        !ggml_hrx_flash_attn_ext_vec_mask(fa)) {
        return -1;
    }
    if (w->type != GGML_TYPE_F32 || r->type != GGML_TYPE_F32 || w->ne[0] != 16 || r->ne[0] != 16 || w->ne[2] != 1 || w->ne[3] != 1 ||
        r->ne[3] != 1 || w->nb[0] != 4 || r->nb[0] != 4 || w->nb[1] % 16 != 0 || (uintptr_t) w->data % 16 != 0 ||
        ggml_nbytes(w) / 4 >= INT_MAX || ggml_nbytes(r) / 4 >= INT_MAX) {
        return -1;
    }
    const int k = ggml_hrx_node_index(cgraph, mm, i - 1, i);
    if (k < 0) {
        return -1;
    }
    // The SUB is the only read of the MUL_MAT. No node up to the FA may write w or r.
    int n_reads = 0;
    for (int m = k + 1; m < cgraph->n_nodes; ++m) {
        const ggml_tensor * n = cgraph->nodes[m];
        for (int q = 0; q < GGML_MAX_SRC; ++q) {
            n_reads += n->src[q] != nullptr && (n->src[q] == mm || n->src[q]->view_src == mm);
        }
        if (m <= j && m != i && !ggml_op_is_empty(n->op) && (ggml_hrx_overlap(n, w) || ggml_hrx_overlap(n, r))) {
            return -1;
        }
    }
    if (n_reads != 1 || (mm->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return -1;
    }
    if (absorbed != nullptr) {
        absorbed->push_back(k);
        return j;
    }
    if (!ggml_hrx_flash_attn_ext_run(ctx, fa, sub, s_copy, mm)) {
        GGML_ABORT("%s: failed to run node %d (%s)", __func__, j, fa->name);
    }
    ctx->dispatch_writes = nullptr;
    return j;
}
