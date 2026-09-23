#include "llama-memory-hybrid-idx.h"
#include "prefix.h"

// A qwen4exp MTP context is built with a recurrent filter that matches nothing (the nextn layer is not recurrent) so
// it can carry an indexer cache for sparse draft attention. Skip the recurrent child there: an empty recurrent cache
// still refuses partial seq_rm, which would leave a rejected draft in the cache.
static bool hybrid_idx_no_recr(const llama_memory_recurrent * r) {
    if (!r) { return true; }
    for (ggml_tensor * t : r->r_l) { if (t) { return false; } }
    for (ggml_tensor * t : r->s_l) { if (t) { return false; } }
    return true;
}

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"


#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iterator>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    if (!mem_idx || !offload || n_swa != 0) { return; }
    qsa_prefix = qsa_prefix_state(kv_size);
    const int layers = model.hparams.n_layer_all;
    qsa_keys.resize(layers, nullptr); qsa_ready.resize(layers, 0);
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr> contexts;
    for (int il=0; il<layers; ++il) {
        if (!model.hparams.has_kv(il) || !filter_idx(il) || model.hparams.dsv4_compress_ratios[il] != 4) { continue; }
        auto * dev = model.dev_layer(il);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "ROCm") != 0) { continue; }
        incremental_qsa = true;
        auto * buft = ggml_backend_dev_buffer_type(dev);
        auto & ctx = contexts[buft];
        if (!ctx) {
            ctx.reset(ggml_init({size_t(layers)*ggml_tensor_overhead(), nullptr, true}));
            if (!ctx) { throw std::runtime_error("QSA cache context allocation failed"); }
        }
        qsa_keys[il] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, model.hparams.indexer_head_size, (kv_size+3)/4 + 1);
        ggml_format_name(qsa_keys[il], "cache_qsa_k_l%d", il);
    }
    for (auto & entry : contexts) {
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(entry.second.get(), entry.first);
        if (!buffer) { throw std::runtime_error("QSA cache buffer allocation failed"); }
        ggml_backend_buffer_clear(buffer, 0);
        qsa_buffers.emplace_back(std::move(entry.second), buffer);
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!hybrid_idx_no_recr(get_mem_recr()) && !get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    auto result = std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
    if (result->get_status() != LLAMA_MEMORY_STATUS_NO_UPDATE) { qsa_invalidate(); }
    return result;
}

void llama_memory_hybrid_idx::clear(bool data) {
    qsa_prefix.reset(); qsa_recover_pending = false; std::fill(qsa_ready.begin(), qsa_ready.end(), 0);
    if (data) { for (const auto & entry : qsa_buffers) { ggml_backend_buffer_clear(entry.second.get(), 0); } }
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!hybrid_idx_no_recr(get_mem_recr()) && !get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (incremental_qsa && !qsa_prefix.valid) { qsa_recover_pending = true; }
    if (incremental_qsa && qsa_prefix.valid && (seq_id < 0 || seq_id == qsa_prefix.sequence)) {
        if (p1 < 0 || size_t(p1) >= qsa_prefix.cells.size()) {
            qsa_prefix.truncate(std::max(0, p0));
            for (auto & ready : qsa_ready) { ready = std::min<int64_t>(ready, qsa_prefix.cells.size()/4); }
        } else { qsa_invalidate(); }
    }
    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    qsa_invalidate();
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    if (!qsa_prefix.valid || (qsa_prefix.sequence >= 0 && seq_id != qsa_prefix.sequence)) { qsa_invalidate(); }
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    qsa_invalidate();
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    qsa_invalidate();
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    for (const auto & entry : qsa_buffers) {
        mb[ggml_backend_buffer_get_type(entry.second.get())] += ggml_backend_buffer_get_size(entry.second.get());
    }
    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // note: repeats llama_memory_hybrid::state_write so the recurrent section can be skipped in step with state_read
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        get_mem_attn()->state_write(io, seq_id, flags);
    }
    if (!hybrid_idx_no_recr(get_mem_recr())) { get_mem_recr()->state_write(io, seq_id, flags); }

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) { qsa_invalidate(); }
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        if (!hybrid_idx_no_recr(get_mem_recr())) { get_mem_recr()->state_read(io, seq_id, flags); }

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    qsa_invalidate();
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    if (!hybrid_idx_no_recr(get_mem_recr())) { get_mem_recr()->seq_rm(seq_id, -1, -1); }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    if (!set_input_qsa_prefix(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias)) {
        set_input_qsa_scan(cell_blk, blk_cells, blk_pos, bias, nullptr, ubatch, ratio, blk_bias);
    }
}

void llama_memory_hybrid_idx::set_input_qsa_blocks(
        ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
        ggml_tensor * bias, ggml_tensor * tail_idxs, const llama_ubatch * ubatch, uint32_t ratio) const {
    if (tail_idxs && cell_blk->ne[1] == 1 && qsa_metadata(blk_cells, blk_pos, bias, tail_idxs, *ubatch, ratio)) {
        return;
    }
    set_input_qsa_scan(cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio, true);
}

// On the tracked prefix the compact metadata follows directly: the complete blocks are the first L/4 position
// blocks, every block start is its own limit, unfinished blocks are never visible (INT32_MAX), and a query's
// tail is the cells of its own partial block.
bool llama_memory_hybrid_idx::qsa_metadata(ggml_tensor * members, ggml_tensor * pos, ggml_tensor * bias,
        ggml_tensor * tails, const llama_ubatch & u, uint32_t ratio) const {
    if (ratio != 4 || bias->type != GGML_TYPE_I32 || !qsa_prefix_matches(u)) { return false; }
    const size_t blocks = pos->ne[0]/4, complete = qsa_prefix.cells.size()/4;
    if (blocks == 0 || complete > blocks) { return false; }
    auto * c = members ? (int32_t *) members->data : nullptr; auto * p = pos ? (int32_t *) pos->data : nullptr;
    auto * lim = (int32_t *) bias->data; auto * tail = (int32_t *) tails->data;
    if (!lim || !tail) { return false; }
    if (c) {
        std::copy_n(qsa_prefix.cells.data(), complete*4, c);
        std::fill(c+complete*4, c+blocks*4, 0);
    }
    if (p) {
        for (int a=0; a<4; ++a) {
            std::copy_n(qsa_prefix.block_positions.data(), complete, p+a*blocks);
            std::fill(p+a*blocks+complete, p+(a+1)*blocks, 0);
        }
    }
    const int64_t n_tps = u.n_tokens;
    const int64_t n_lim = bias->ne[0] - 2*n_tps;
    const int64_t n_seq = n_lim / (int64_t) blocks;
    const int64_t row   = qsa_prefix.sequence;
    if (n_lim % (int64_t) blocks != 0 || n_seq < 1 || row < 0 || row >= n_seq) { return false; }
    const int64_t base  = blocks * n_seq;
    std::fill(lim, lim + base, INT32_MAX);
    std::copy_n(qsa_prefix.block_positions.data(), complete, lim + row*blocks);
    std::fill(lim + row*blocks + complete, lim + (row+1)*blocks, INT32_MAX);
    std::fill(tail, tail+3*u.n_tokens, -1);
    for (uint32_t i=0; i<u.n_tokens; ++i) {
        int32_t end = u.pos[i]+1, start = end/4*4; lim[base+i] = start;
        lim[base + n_tps + i] = (int32_t) row;
        for (int j=0; j<end-start; ++j) { tail[3*i+j] = qsa_prefix.cells[start+j]; }
    }
    return true;
}

// On the tracked prefix (qsa_prefix), position k of the one sequence lives in cell qsa_prefix.cells[k] and
// nothing else occupies the cache, so the arrays the scan derives per cell follow directly: complete blocks
// are the first L/ratio position blocks, everything else maps to the spare block, and a block's bias is
// -inf past the complete ones, the "always visible" value from the query's incomplete tail onwards, else 0.
bool llama_memory_hybrid_idx::set_input_qsa_prefix(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    if (!blk_bias || ratio == 0 || cell_blk->ne[1] != 1 || !qsa_prefix_matches(*ubatch)) {
        return false;
    }
    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t r        = ratio;
    const int64_t n_blocks = blk_pos->ne[0]/4;
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t L        = (int64_t) qsa_prefix.cells.size();
    if (blk_cells->ne[0] != r*n_blocks || blk_cells->ne[1] != 1 || bias->ne[0] != n_blocks || bias->ne[1] != n_tokens ||
        bias->ne[2] != 1 || bias->type != GGML_TYPE_F32 || qsa_prefix.end != L || L > n_kv || n_blocks*r < n_kv) {
        return false;
    }
    if (!get_mem_idx()) {
        return false;
    }
    const auto & cells = get_mem_idx()->get_cells(ubatch->seq_id[0][0]);
    if ((int64_t) cells.get_used() != L) {
        return false;
    }
    for (int64_t k = 0; k < L; ++k) {
        if (qsa_prefix.cells[k] < 0 || qsa_prefix.cells[k] >= n_kv) {
            return false;
        }
    }

    const int64_t n_bid     = L/r;
    const bool    have_dead = n_bid < n_blocks;
    const int32_t dead_bid  = (int32_t) (have_dead ? n_bid : n_blocks - 1);

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    int32_t * dst_blk_pos   = (int32_t *) blk_pos->data;
    float   * dst_bias      = (float   *) bias->data;
    // the incremental key-cache graph leaves the member and position arrays unallocated, as the scan expects
    if (!dst_cell_blk || !dst_bias) {
        return false;
    }

    std::fill(dst_cell_blk, dst_cell_blk + n_kv, dead_bid);
    if (dst_blk_cells) {
        std::fill(dst_blk_cells, dst_blk_cells + r*n_blocks, 0);
    }
    if (dst_blk_pos) {
        std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks, 0);
    }
    for (int64_t k = 0; k < n_bid*r; ++k) {
        const int32_t cell = qsa_prefix.cells[k];
        if (dst_blk_cells) {
            dst_blk_cells[k] = cell;
        }
        dst_cell_blk[cell] = (int32_t) (k/r);
    }
    if (dst_blk_pos) {
        for (int64_t sec = 0; sec < 4; ++sec) {
            for (int64_t b = 0; b < n_bid; ++b) {
                dst_blk_pos[sec*n_blocks + b] = (int32_t) (b*r);
            }
        }
    }
    for (int64_t i = 0; i < n_tokens; ++i) {
        const int64_t tail_start = ((int64_t) ubatch->pos[i] + 1)/r*r;
        float * row = dst_bias + i*n_blocks;
        for (int64_t b = 0; b < n_blocks; ++b) {
            row[b] = b >= n_bid ? -INFINITY : (b*r >= tail_start ? 1e9f : 0.0f);
        }
        if (have_dead) {
            row[dead_bid] = 1e9f;
        }
    }
    return true;
}

void llama_memory_hybrid_idx::set_input_qsa_scan(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        ggml_tensor * tail_idxs,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    GGML_ASSERT(tail_idxs || ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = blk_pos->ne[0]/(4*n_ns);
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    // with tails the per-cell block map is not consumed (the selection gathers cells by block)
    int32_t * dst_cell_blk  = tail_idxs ? nullptr : (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    int32_t * dst_blk_pos   = (int32_t *) blk_pos->data;
    const bool compact = bias->type == GGML_TYPE_I32;
    float   * dst_bias      = compact ? nullptr : (float *) bias->data;
    int32_t * limits        = compact ? (int32_t *) bias->data : nullptr;
    if (compact) {
        GGML_ASSERT(tail_idxs && blk_bias && n_ns == 1 && (ggml_nelements(bias) - 2*n_tps) % n_blocks == 0);
    }
    const int64_t n_seq = compact ? (ggml_nelements(bias) - 2*n_tps) / n_blocks : 1;
    int32_t * dst_tail = tail_idxs ? (int32_t *) tail_idxs->data : nullptr;
    if (tail_idxs) {
        GGML_ASSERT(blk_bias && r > 1 && ggml_backend_buffer_is_host(tail_idxs->buffer));
        GGML_ASSERT(tail_idxs->ne[0] == r-1 && tail_idxs->ne[1] == n_tps && tail_idxs->ne[2] == n_ns);
        std::fill(dst_tail, dst_tail + ggml_nelements(tail_idxs), -1);
    }

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // TODO: this runs per ubatch and is O(n_kv) per stream, about 865 us at 33k context. the cost
    //       is the per-cell scan rather than these allocations, so hoisting them buys nothing
    std::vector<int32_t>  blk_of(n_kv);
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_blocks);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    if (dst_blk_pos) { std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0); }

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk ? dst_cell_blk + s*n_kv : nullptr;
        int32_t * cur_blk_cells = dst_blk_cells ? dst_blk_cells + s*(r*n_blocks) : nullptr;

        if (cur_blk_cells) { std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0); }

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            std::fill(blk_of.begin(),   blk_of.end(),   -1);
            std::fill(cell_grp.begin(), cell_grp.end(), -1);
            std::fill(grp_head.begin(), grp_head.end(), -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        group_cells();

        // mrope repeats one position across an image, so rank cells instead of using the position
        if (dup && ubatch->is_pos_2d() && one_seq) {
            order.clear();
            order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    order.push_back((int32_t) j);
                }
            }

            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            });

            rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                rank[order[k]] = (int32_t) k;
            }

            ranked = true;

            group_cells();
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        int32_t n_bid = 0;

        for (int64_t pb = 0; pb < n_blocks; ++pb) {
            for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                if (grp_slots[g] != slots_full) {
                    continue;
                }

                grp_bid[g] = n_bid++;

                bid_idx  .push_back((int32_t) (pb*r));
                bid_cell .push_back(grp_first[g]);
                bid_slot0.push_back(grp_slot0[g]);
            }
        }

        GGML_ASSERT(n_bid <= n_blocks);

        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

            if (ranked) {
                const int32_t   c = bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                if (dst_blk_pos) { dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec]; }
            }
        }

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? n_bid : n_blocks - 1;

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t g = cell_grp[j];

            blk_of[j] = g < 0 ? -1 : grp_bid[g];

            if (blk_of[j] >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                if (cur_blk_cells) { cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j; }
            }

            if (cur_cell_blk) { cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j]; }
        }

        // member cells per group, for the tails (a group's slot k holds the cell at position k within the block)
        std::vector<int32_t> group_members;
        if (dst_tail) {
            group_members.assign(grp_first.size()*r, -1);
            for (int64_t j=0;j<n_kv;++j) {
                const int32_t g=cell_grp[j];
                if (g<0) { continue; }
                const int64_t idx=ranked ? rank[j] : cells.pos_get(j);
                const int64_t slot=g*r+idx%r;
                GGML_ASSERT(group_members[slot]<0 || group_members[slot]==j);
                group_members[slot]=(int32_t)j;
            }
        }

        if (compact) {
            // one start row per sequence: block b is visible to seq s iff the group's rep cell carries s,
            // so blocks shared by several sequences (seq_cp) stay visible to each of them, as in the masked path.
            // only the rows the queries read are filled
            std::fill(limits, limits + n_blocks*n_seq, INT32_MAX);
            llama_kv_cells::seq_set_t rows;
            for (int64_t ii=0;ii<n_tps;++ii) {
                const llama_seq_id row = ubatch->seq_id[s*n_tps + ii][0];
                GGML_ASSERT(row >= 0 && row < n_seq);
                rows.set(row);
            }
            for (int64_t row=0;row<n_seq;++row) {
                if (!rows.test(row)) { continue; }
                for (int64_t b=0;b<n_bid;++b) {
                    if (cells.seq_has((uint32_t) bid_cell[b], (llama_seq_id) row)) {
                        limits[row*n_blocks + b] = bid_idx[b];
                    }
                }
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (dst_tail && q+1>tail_start) {
                const int64_t pb=tail_start/r;
                if (pb>=0 && pb<n_blocks) {
                    int32_t * tail=dst_tail+(s*n_tps+ii)*(r-1);
                    for (int32_t g=grp_head[pb];g>=0;g=grp_next[g]) {
                        if (!cells.seq_has((uint32_t)grp_first[g],seq_id)) { continue; }
                        for (int64_t slot=0;slot<q+1-tail_start;++slot) {
                            const int32_t cell=group_members[g*r+slot];
                            if (cell<0 || !cells.seq_has((uint32_t)cell,seq_id)) { continue; }
                            GGML_ASSERT(tail[slot]<0 || tail[slot]==cell);
                            tail[slot]=cell;
                        }
                    }
                }
            }

            if (compact) {
                GGML_ASSERT(tail_start >= 0 && tail_start <= 16777216);
                const int64_t base = n_blocks*n_seq;
                limits[base + ii] = (int32_t)tail_start;
                limits[base + n_tps + ii] = (int32_t)seq_id;
                continue;
            }

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b >= n_bid || !cells.seq_has((uint32_t) bid_cell[b], seq_id)) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    // with tails the own block is never scored (its cells come from the tail), so it is hidden
                    cur_blk_bias[b] = bid_idx[b] >= tail_start ? (dst_tail ? -INFINITY : 1e9f) : 0.0f;
                }

                // the spare block holds the unpooled cells, which are the incomplete tail, so
                // it gets the tail value. it must stay finite: a sequence with fewer than
                // `ratio` cells owns no full block, and a row of -inf only gives a nan.
                if (have_dead) {
                    cur_blk_bias[dead_bid] = dst_tail ? -INFINITY : 1e9f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    qsa_slots(sinfos_idx),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
        if (res && i_cur < qsa_slots.size()) {
            const_cast<llama_memory_hybrid_idx *>(mem)->qsa_apply(ctx_idx->get_ubatch(), qsa_slots[i_cur]);
        } else if (!res) { const_cast<llama_memory_hybrid_idx *>(mem)->qsa_invalidate(); }
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias);
}

void llama_memory_hybrid_idx_context::set_input_qsa_blocks(
        ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
        ggml_tensor * bias, ggml_tensor * tail_idxs, const llama_ubatch * ubatch, uint32_t ratio) const {
    GGML_ASSERT(mem != nullptr);
    mem->set_input_qsa_blocks(cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio);
}

bool llama_memory_hybrid_idx_context::qsa_position_prefix(const llama_ubatch & ubatch, uint32_t ratio) const {
    if (qsa_prefix_matches(ubatch)) { return true; }
    if (!qsa_scalar_visibility(ubatch, ratio)) { return false; }
    const llama_seq_id seq=ubatch.seq_id[0][0];
    return qsa_single_sequence_prefix(mem->get_mem_idx()->get_cells(seq),get_idx()->get_n_kv(),seq);
}

uint32_t llama_memory_hybrid_idx_context::qsa_n_kv_window() const {
    const uint32_t n_kv = get_idx() ? get_idx()->get_n_kv() : 0;
    if (!mem || !mem->get_mem_idx()) { return n_kv; }
    llama_pos pos_max = -1;
    // a unified cache maps every seq id to stream 0; otherwise seq s is stream s and seq_to_stream holds n_stream ids.
    // this is the stream count of the cache: get_n_stream() counts the streams of the ubatch, 1 for a single slot
    const uint32_t n_stream = mem->get_mem_idx()->get_n_stream();
    const llama_seq_id n_seq_scan = n_stream == 1 ? (llama_seq_id) LLAMA_MAX_SEQ : (llama_seq_id) n_stream;
    for (llama_seq_id s = 0; s < n_seq_scan; ++s) {
        pos_max = std::max(pos_max, mem->get_mem_idx()->get_cells(s).seq_pos_max(s));
    }
    if (pos_max < 0) { return n_kv; }
    const uint32_t window = ((uint32_t) pos_max + 1 + 255) / 256 * 256;
    return std::max(n_kv, window);
}

bool llama_memory_hybrid_idx_context::qsa_scalar_visibility(const llama_ubatch & ubatch, uint32_t ratio) const {
    if (qsa_prefix_matches(ubatch)) { return true; }
    if (get_n_stream()!=1 || !get_idx() || !ubatch.token || !ubatch.n_tokens || !ubatch.seq_id || !ubatch.n_seq_id ||
            ubatch.n_seq_id[0]<1 || !ubatch.seq_id[0]) { return false; }
    // one stream: every seq id of the ubatch reads the same cells
    return qsa_scalar_visibility_cells(mem->get_mem_idx()->get_cells(ubatch.seq_id[0][0]), get_idx()->get_n_kv(), ratio, ubatch);
}

void llama_memory_hybrid_idx::qsa_invalidate() {
    qsa_prefix.invalidate(); qsa_recover_pending = true; std::fill(qsa_ready.begin(), qsa_ready.end(), 0);
}

bool llama_memory_hybrid_idx::qsa_recover(llama_seq_id seq) {
    if (mem_idx->get_n_stream() != 1) { return false; }
    const auto & raw = mem_idx->get_cells(seq);
    if (raw.size() != qsa_prefix.positions.size()) { return false; }
    qsa_prefix_state rebuilt(qsa_prefix.positions.size());
    rebuilt.cells.resize(raw.get_used(), -1);
    for (uint32_t cell=raw.used_min(); cell<raw.used_max_p1(); ++cell) {
        if (raw.is_empty(cell)) { continue; }
        const int32_t pos = raw.pos_get(cell);
        const auto & ext = raw.ext_get(cell);
        if (pos < 0 || size_t(pos) >= rebuilt.cells.size() || rebuilt.cells[pos] >= 0 ||
            !raw.seq_has(cell, seq) || raw.seq_get_all(cell).count() != 1 ||
            !((ext.x == 0 && ext.y == 0) || (ext.x == pos && ext.y == pos))) { return false; }
        rebuilt.cells[pos] = cell; rebuilt.positions[cell] = pos;
    }
    if (std::find(rebuilt.cells.begin(), rebuilt.cells.end(), -1) != rebuilt.cells.end()) { return false; }
    rebuilt.sequence = seq;
    for (size_t b=0; b<rebuilt.cells.size()/4; ++b) { rebuilt.block_positions.push_back(b*4); }
    qsa_prefix = std::move(rebuilt);
    return true;
}

void llama_memory_hybrid_idx::qsa_apply(const llama_ubatch & u, const llama_kv_cache::slot_info & slots) {
    if (!incremental_qsa) { return; }
    const auto reject = [&]() { qsa_invalidate(); qsa_recover_pending = false; };
    if (!u.token || !u.pos || !u.n_tokens || slots.n_stream() != 1 ||
        slots.size() != u.n_tokens || !u.n_pos || !u.seq_id || !u.n_seq_id || u.n_seq_id[0] != 1 || !u.seq_id[0]) {
        reject(); return;
    }
    const auto seq = u.seq_id[0][0];
    for (uint32_t i=0; i<u.n_tokens; ++i) {
        if (u.n_seq_id[i] != 1 || !u.seq_id[i] || u.seq_id[i][0] != seq || int64_t(u.pos[i]) != int64_t(u.pos[0])+i) {
            reject(); return;
        }
        for (uint32_t axis=1; axis<u.n_pos; ++axis) {
            if (u.pos[i+axis*u.n_tokens] != u.pos[i]) { reject(); return; }
        }
    }
    if (!qsa_prefix.valid && (!qsa_recover_pending || !qsa_recover(seq))) { qsa_recover_pending = false; return; }
    qsa_recover_pending = false;
    if (!qsa_prefix.apply(seq, u.pos[0], slots.idxs[0])) { reject(); }
}

bool llama_memory_hybrid_idx::qsa_prefix_matches(const llama_ubatch & u) const {
    if (!incremental_qsa || !qsa_prefix.valid || !u.token || !u.pos || !u.n_tokens || !u.n_pos || !u.seq_id || !u.n_seq_id ||
        u.pos[0] != qsa_prefix.begin || int64_t(u.n_tokens) != qsa_prefix.end-qsa_prefix.begin) { return false; }
    for (uint32_t i=0; i<u.n_tokens; ++i) {
        if (u.n_seq_id[i] != 1 || !u.seq_id[i] || u.seq_id[i][0] != qsa_prefix.sequence || u.pos[i] != u.pos[0]+int32_t(i)) { return false; }
        for (uint32_t a=1; a<u.n_pos; ++a) { if (u.pos[i+a*u.n_tokens] != u.pos[i]) { return false; } }
    }
    return true;
}

bool llama_memory_hybrid_idx::qsa_fast(int il, const llama_ubatch & u) const {
    // re-pool only the (n+3)/4+2 blocks this ubatch touches instead of every block of the cache. Limited to 127 tokens:
    // with 128..512-token ubatches the results were not bit-equal to the full re-pool (perplexity at ub 512 changed on
    // gfx1151, 2026-09-23) and the cause is not known yet
    return qsa_prefix_matches(u) && u.n_tokens <= 127 && qsa_keys.at(il) &&
        qsa_ready.at(il) >= int64_t(qsa_prefix.previous_size/4) &&
        qsa_prefix.cells.size()/4 >= (u.n_tokens+3)/4+1;
}

ggml_tensor * llama_memory_hybrid_idx::qsa_cache(ggml_context * ctx, int il, int64_t blocks) const {
    if (!incremental_qsa || !qsa_keys.at(il)) { return nullptr; }
    auto * k = qsa_keys[il]; GGML_ASSERT(blocks + 1 <= k->ne[1]);
    return ggml_view_2d(ctx, k, k->ne[0], blocks + 1, k->nb[1], 0);
}

void llama_memory_hybrid_idx::qsa_fill_updates(ggml_tensor * members, ggml_tensor * pos, ggml_tensor * rows) const {
    const int count = rows->ne[0], complete = qsa_prefix.cells.size()/4;
    const int first = qsa_prefix.begin/4, last = std::min(complete, (qsa_prefix.end+3)/4);
    GGML_ASSERT(last-first < count && complete >= count - 1);
    std::vector<int32_t> ids = { complete };
    for (int b=first; b<last; ++b) { ids.push_back(b); }
    for (int b=0; int(ids.size())<count; ++b) { if (b<first || b>=last) { ids.push_back(b); } }
    auto * c = (int32_t *) members->data; auto * p = (int32_t *) pos->data; auto * w = (int64_t *) rows->data;
    for (int i=0; i<count; ++i) {
        w[i] = ids[i];
        for (int j=0; j<4; ++j) { c[4*i+j] = ids[i] == complete ? 0 : qsa_prefix.cells[ids[i]*4+j]; p[j*count+i] = ids[i] == complete ? 0 : ids[i]*4; }
    }
}

void llama_memory_hybrid_idx::qsa_commit(int il) const {
    GGML_ASSERT(qsa_prefix.valid); qsa_ready.at(il) = qsa_prefix.cells.size()/4;
}
