#pragma once

#include "llama-memory-hybrid.h"

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

    // QSA pooled-key cache: one row per block, holding the mean-pooled, RMS-normalised and
    // rotated indexer key that build_qsa_top_k scores against. A block that is full never
    // changes again -- its cells are written once and pooling, normalisation and rotation are
    // all position-determined -- so only the tail of the cache is recomputed per ubatch.
    // Anything that moves cells (a shift, a removal, a copy, a state load) drops the lot.
    llama_kv_cache * get_mem_pool() const;  // nullptr when the model carries no indexer

    // how many trailing blocks the next graph must recompute. n_blocks while the cache is
    // invalid, otherwise the tail this ubatch can touch: the blocks its own tokens land in,
    // plus the blocks n_kv gains when it next grows by a padding step.
    uint32_t qsa_pool_n_recomp(uint32_t ratio, uint32_t n_tokens, uint32_t n_kv, uint32_t n_pad_kv) const;

    // Blocks are cut on the position line, so what a cache-disturbing operation costs is a
    // watermark, not a flag: everything below the first position it touches is still right.
    // Speculative decoding drops its rejected tail with seq_rm on every single step, and
    // treating that as "forget everything" made the cache recompute the whole table each time.
    void qsa_pool_invalidate() const;                    // forget the lot
    void qsa_pool_invalidate_from(llama_pos p0) const;   // forget positions >= p0
    void qsa_pool_validate(uint32_t n_pos) const;        // positions < n_pos are now pooled

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    // pool_* are null when the caller wants every block recomputed inline (the pre-cache path):
    //   pool_idxs  I64 [n_recomp]        rows of the pooled cache this ubatch rewrites
    //   pool_cells I32 [ratio*n_recomp]  cells making up each rewritten block
    //   pool_pos   I32 [4*n_recomp]      mrope position rows of each rewritten block
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, ggml_tensor * pool_idxs, ggml_tensor * pool_cells,
                       ggml_tensor * pool_pos, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias) const;

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // the pooled-key cache is addressed by block, so it needs its own hparams too
    llama_hparams hparams_pool;

    const std::unique_ptr<llama_kv_cache> mem_pool;

    // pooled keys are correct for the blocks that cover positions below this
    mutable uint32_t pool_valid_pos = 0;

    // the pooled rows are addressed by block index, and set_input_qsa numbers blocks
    // bucket-major across the sequence groups of a stream. With two sequences in one
    // (unified) stream a block gained by either shifts every later index, and the shorter
    // sequence's tail blocks sit mid-table where the recompute window cannot reach them.
    // So the cache is only trusted while a single sequence is present in the stream.
    bool qsa_pool_one_seq() const;
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

    // the QSA pooled-key cache, and how many trailing blocks the graph must rewrite into it
    llama_kv_cache * get_mem_pool() const;
    uint32_t qsa_pool_n_recomp(uint32_t ratio, uint32_t n_tokens, uint32_t n_kv, uint32_t n_pad_kv) const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    // pool_* are null when the caller wants every block recomputed inline (the pre-cache path):
    //   pool_idxs  I64 [n_recomp]        rows of the pooled cache this ubatch rewrites
    //   pool_cells I32 [ratio*n_recomp]  cells making up each rewritten block
    //   pool_pos   I32 [4*n_recomp]      mrope position rows of each rewritten block
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, ggml_tensor * pool_idxs, ggml_tensor * pool_cells,
                       ggml_tensor * pool_pos, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias) const;

private:
    const llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;
};
