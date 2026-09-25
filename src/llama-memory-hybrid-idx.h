#pragma once

#include "llama-memory-hybrid.h"
#include "qsa-prefix-state.h"

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

    ~llama_memory_hybrid_idx() = default;

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
                       bool blk_bias) const;
    // complete-block selection metadata: tails and, when the bias is I32, compact limits
    void set_input_qsa_blocks(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                              ggml_tensor * bias, ggml_tensor * tail_idxs,
                              const llama_ubatch * ubatch, uint32_t ratio) const;

    void qsa_apply(const llama_ubatch & ubatch, const llama_kv_cache::slot_info & slots);
    void qsa_invalidate();
    bool qsa_prefix_matches(const llama_ubatch & ubatch) const;
    bool qsa_fast(int il, const llama_ubatch & ubatch) const;
    ggml_tensor * qsa_cache(ggml_context * ctx, int il, int64_t blocks) const;
    void qsa_fill_updates(ggml_tensor * members, ggml_tensor * positions, ggml_tensor * rows) const;
    void qsa_commit(int il) const;

private:
    // closed-form metadata for an incremental ubatch on the tracked prefix; false when the scan is needed
    bool set_input_qsa_prefix(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                              ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias) const;
    // the per-cell scan over the whole window; with tail_idxs it also writes each query's own partial block
    // (r-1 cells, -1 padded) and, for an I32 bias, the compact visibility limits
    // [n_seq x n_blocks starts] ++ [n_tps tails] ++ [n_tps row idx]
    void set_input_qsa_scan(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                            ggml_tensor * bias, ggml_tensor * tail_idxs, const llama_ubatch * ubatch,
                            uint32_t ratio, bool blk_bias) const;
    // closed-form block metadata on the tracked prefix for the compact I32 bias; false when the scan is needed
    bool qsa_metadata(ggml_tensor * cells, ggml_tensor * positions, ggml_tensor * bias,
                      ggml_tensor * tails, const llama_ubatch & ubatch, uint32_t ratio) const;

    bool incremental_qsa = false;
    bool qsa_recover_pending = false;
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
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
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

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

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

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias) const;
    void set_input_qsa_blocks(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                              ggml_tensor * bias, ggml_tensor * tail_idxs,
                              const llama_ubatch * ubatch, uint32_t ratio) const;
    // one stream, scalar positions in range, no 2d rope extents, no block split over sequence sets
    // (qsa_scalar_visibility_cells): the compact visibility rule applies
    bool qsa_scalar_visibility(const llama_ubatch & ubatch, uint32_t ratio) const;
    // scalar visibility and the cached cells form a single-sequence prefix with unique positions
    bool qsa_position_prefix(const llama_ubatch & ubatch, uint32_t ratio) const;

private:
    const llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;
    const slot_info_vec_t qsa_slots;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;
};
