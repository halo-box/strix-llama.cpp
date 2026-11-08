#pragma once

#include "llama-memory-hybrid.h"
#include "qsa-prefix-state.h"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
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
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    // Defined out of line because kpool_layout is incomplete here.
    ~llama_memory_hybrid_idx();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // The model's indexer pool size.
    uint32_t get_kpool() const { return hparams_idx.indexer_kpool; }

    // Whether pools are kpool consecutive cells in sequence order (qwen4exp) instead of kpool consecutive positions.
    bool get_kpool_by_order() const { return hparams_idx.indexer_kpool_by_order; }

    // Which cells of a sequence make up which pool of kpool consecutive positions (or cells, in order mode).
    // It is kept here because it outlives the batch: pools are fixed by the positions relative to the
    // sequence's first one, so a ubatch only ever appends to it. Sequence edits drop it, see mem_idx_stale.
    struct kpool_layout;

    const kpool_layout & kpool_layout_update();
    const kpool_layout & kpool_layout_get() const;

    // The pooled keys persist in the idx cache across batches. A sequence edit can regroup the pools
    // from some position on, which stales every pooled key at or after it. POS_CLEAN means none.
    using stale_pos_t = std::array<llama_pos, LLAMA_MAX_SEQ>;

    static constexpr llama_pos POS_CLEAN = std::numeric_limits<llama_pos>::max();

    static stale_pos_t stale_pos_clean() {
        stale_pos_t res;
        res.fill(POS_CLEAN);
        return res;
    }

    const stale_pos_t & mem_idx_stale_get() const { return mem_idx_stale; }
    void mem_idx_stale_clear() { mem_idx_stale.fill(POS_CLEAN); }

    // qwen4exp QSA inputs, see llama_memory_hybrid_idx_context::set_input_qsa
    // causal_attn selects the rule: causal forces the query's own block on, non-causal lets every visible block compete on score
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias, bool causal_attn) const;
    // complete-block selection metadata: tails and, when the bias is I32, compact limits
    void set_input_qsa_blocks(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                              ggml_tensor * bias, ggml_tensor * tail_idxs,
                              const llama_ubatch * ubatch, uint32_t ratio) const;

    void qsa_apply(const llama_ubatch & ubatch, const llama_kv_cache::slot_info & slots);
    void qsa_invalidate();
    bool qsa_prefix_matches(const llama_ubatch & ubatch) const;
    bool qsa_fast(int il, const llama_ubatch & ubatch) const;
    // the tracked prefix is position-ordered (no image in it): rank == position
    bool qsa_prefix_identity() const { return qsa_prefix.identity(); }
    ggml_tensor * qsa_cache(ggml_context * ctx, int il, int64_t blocks) const;
    void qsa_fill_updates(ggml_tensor * members, ggml_tensor * positions, ggml_tensor * rows) const;
    void qsa_commit(int il) const;

    // the full-cache context with the attention and indexer views limited to n_kv cells: reserve pass B (PLAN-v3 §7.1.3),
    // the dense shortcut's graph at the largest padded n_kv it serves
    llama_memory_context_ptr init_full_n_kv(uint32_t n_kv);

    // PLAN-v3 §4.2 / §4.3: the compact predicate of a ubatch over the cells of kv (the attention cache at init, the
    // indexer cache at graph time), each of the ubatch's n_stream streams with the sequence ids of its own tokens;
    // QSA_COMPACT_OK or the reason of qsa_compact_reason (prefix.h)
    static int qsa_compact_reason(const llama_kv_cache * kv, uint32_t n_stream, uint32_t n_kv, uint32_t ratio, const llama_ubatch & ubatch);

    // the non-compact re-split (PLAN-v3 §4.3, P2-2): U = min(n_ubatch, strips x 512) tokens in total per non-compact ubatch
    static constexpr uint32_t QSA_NONCOMPACT_STRIPS = 1;
    uint32_t qsa_noncompact_cap(uint32_t n_ubatch) const { return std::min<uint32_t>(n_ubatch, qsa_nc_strips * 512); }
    // replays of init_batch since the memory was made, and in its last call (test-qsa-resplit assert (vi))
    uint64_t qsa_n_resplit() const { return n_qsa_resplit; }
    uint32_t qsa_last_replays() const { return qsa_last_n_replay; }
    // NC0 per ubatch of the last init_batch (true: init-time non-compact), and the sizes the replay used
    const std::vector<uint8_t> & qsa_last_nc0() const { return qsa_last_nc0_v; }
    // test hook: the ratios of the QSA layers, the indexer top-k and S (default: from the model and QSA_NONCOMPACT_STRIPS)
    void qsa_resplit_configure(std::vector<uint32_t> ratios, uint32_t top_k, uint32_t strips) {
        qsa_nc_ratios = std::move(ratios); qsa_nc_top_k = top_k; qsa_nc_strips = strips;
    }
    // the rank decision of the last scan, per stream of its ubatch (test-qsa-visibility's agreement assert)
    const std::vector<uint8_t> & qsa_last_scan_ranked() const { return qsa_scan_ranked; }
    uint32_t get_n_seq_max() const { return n_seq_max_qsa; }

private:
    // closed-form metadata for an incremental ubatch on the tracked prefix; false when the scan is needed
    bool set_input_qsa_prefix(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                              ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias) const;
    // the per-cell scan over the whole window; with tail_idxs it also writes each query's own partial block
    // (r-1 cells, -1 padded) and, for an I32 bias, the compact visibility limits
    // [n_seq x n_blocks starts] ++ [n_tps tails] ++ [n_tps row idx]
    void set_input_qsa_scan(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                            ggml_tensor * bias, ggml_tensor * tail_idxs, const llama_ubatch * ubatch,
                            uint32_t ratio, bool blk_bias, bool causal_attn) const;
    // closed-form block metadata on the tracked prefix for the compact I32 bias; false when the scan is needed
    bool qsa_metadata(ggml_tensor * cells, ggml_tensor * positions, ggml_tensor * bias,
                      ggml_tensor * tails, const llama_ubatch & ubatch, uint32_t ratio) const;

    bool incremental_qsa = false;
    bool qsa_recover_pending = false;
    llama_seq_id qsa_recover_failed = -1; // the sequence the last recovery attempt failed for
    bool qsa_recover(llama_seq_id seq);
    qsa_prefix_state qsa_prefix;
    mutable std::vector<int64_t> qsa_ready;
    std::vector<ggml_tensor *> qsa_keys;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> qsa_buffers;
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // unique_ptr because kpool_layout is incomplete here
    std::unique_ptr<kpool_layout> kpool_lay;

    // seq_id < 0 stales every sequence, p0 < 0 stales the sequence from its first position
    void mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0);

    // the position an edit at p0 stales the sequence from
    llama_pos mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const;

    stale_pos_t mem_idx_stale = stale_pos_clean();

    // NC0 inputs (PLAN-v3 §4.3): the compress ratios of the QSA layers and the indexer top-k, and S
    std::vector<uint32_t> qsa_nc_ratios;
    uint32_t qsa_nc_top_k  = 0;
    uint32_t qsa_nc_strips = QSA_NONCOMPACT_STRIPS;
    uint32_t n_seq_max_qsa = 1;
    uint64_t n_qsa_resplit = 0;
    uint32_t qsa_last_n_replay = 0;
    std::vector<uint8_t> qsa_last_nc0_v;
    mutable std::vector<uint8_t> qsa_scan_ranked;
    // NC0(u) on the cells prepare simulated: the selection path (n_kv > top_k + r - 1) and the compact predicate rejects u
    bool qsa_nc0(const llama_ubatch & ubatch, const llama_kv_cache::slot_info & sinfo) const;
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    class kpool_access {
    public:
        ggml_tensor * gather_key_gate(ggml_tensor * idxs) const;
        ggml_tensor * scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const;
        ggml_tensor * gather_pooled(ggml_tensor * idxs) const;

    private:
        friend class llama_memory_hybrid_idx_context;

        kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd);

        ggml_context * ctx;
        ggml_tensor  * key_gate;
        ggml_tensor  * pooled;
    };

    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // a full-cache context whose attention and indexer views hold n_kv cells (reserve pass B)
    llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem, uint32_t n_kv);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context(); // Defined out of line because kpool_state is incomplete here.

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    bool qsa_prefix_matches(const llama_ubatch & u) const { return mem && mem->qsa_prefix_matches(u); }
    bool qsa_fast(int il, const llama_ubatch & u) const { return mem && mem->qsa_fast(il, u); }
    // cells the block metadata must cover: the KV view is sized by occupied cells, but a cache whose positions
    // run ahead of its cells (an MTP draft never receives the image cells an M-RoPE image pins to one position)
    // has blocks past that view. Never below get_n_kv(), padded like it so graph reuse keeps its cadence.
    uint32_t qsa_n_kv_window() const;
    ggml_tensor * qsa_cache(ggml_context * ctx, int il) const {
        return mem ? mem->qsa_cache(ctx, il, (qsa_n_kv_window()+3)/4) : nullptr;
    }
    void qsa_fill_updates(ggml_tensor * members, ggml_tensor * pos, ggml_tensor * rows) const { mem->qsa_fill_updates(members, pos, rows); }
    void qsa_commit(int il) const { mem->qsa_commit(il); }

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias, bool causal_attn) const;
    void set_input_qsa_blocks(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                              ggml_tensor * bias, ggml_tensor * tail_idxs,
                              const llama_ubatch * ubatch, uint32_t ratio) const;
    // one stream, scalar positions in range, no 2d rope extents, no block split over sequence sets
    // (qsa_scalar_visibility_cells): the compact visibility rule applies
    bool qsa_scalar_visibility(const llama_ubatch & ubatch, uint32_t ratio) const;
    // why the compact predicate rejects ubatch on the indexer cells (QSA_COMPACT_OK when it accepts), for the decode error
    int qsa_compact_reason(const llama_ubatch & ubatch, uint32_t ratio) const;
    // the re-split cap U of this memory (PLAN-v3 §4.3) and whether it serves several sequences
    uint32_t qsa_noncompact_cap(uint32_t n_ubatch) const { return mem ? mem->qsa_noncompact_cap(n_ubatch) : n_ubatch; }
    bool qsa_multi_seq() const { return mem && mem->get_n_seq_max() > 1; }
    // scalar visibility and the cached cells form a single-sequence prefix with unique positions
    bool qsa_position_prefix(const llama_ubatch & ubatch, uint32_t ratio) const;

    // glm5-next and qwen4exp, complete pools of kpool cells per sequence, scored as whole pools.
    uint32_t get_n_kpool    () const; // Padded pool count, where the last pool is always unused.
    uint32_t get_n_kpool_new() const; // Pools to re-pool this ubatch, padded to a stable bound, never below 1.
    kpool_access get_kpool_access(ggml_context * ctx, int32_t il, int64_t n_embd) const;
    // sel_mask (F32 [n_sel, 1, 1, n_tokens], can be null): 0 for the live selection slots, -inf for the dead ones
    // new_pool_pos (I32 [4*n_new]): M-RoPE position of each new pool's first member, for pooled keys rotated at pooling time
    void set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
                         ggml_tensor * sel_mask, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
                         const llama_ubatch * ubatch, ggml_tensor * new_pool_pos = nullptr) const;

private:
    llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;
    const slot_info_vec_t qsa_slots;

    // the indexer cells of each ubatch, kept for pools in cache order (qwen4exp): token s*n + i of ubatch u
    // sits in cell idxs[s][i] of stream strm[s] of sinfos_kpool[u], and several cells can share a position
    const slot_info_vec_t sinfos_kpool;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;

    // Which pools of the layout this ubatch must re-pool. The layout itself belongs to the memory.
    struct kpool_state;
    kpool_state kpool_build_sizes() const;
    void kpool_build_state(const llama_ubatch & ubatch);
    const kpool_state & kpool_cur() const;

    // unique_ptr because kpool_state is incomplete here.
    std::unique_ptr<kpool_state> kpool_st;

    // The ubatch kpool_st was built for, guards against reads before apply.
    size_t i_kpool = SIZE_MAX;

    // Whether this context tracks k-pool states.
    bool kpool_track() const;

    // Positions each sequence must re-pool from, cleared only after the first ubatch succeeds
    llama_memory_hybrid_idx::stale_pos_t mem_idx_stale_batch = llama_memory_hybrid_idx::stale_pos_clean();
};
