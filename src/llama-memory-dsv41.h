#pragma once

#include "llama-dsv41.h"
#include "llama-dsv41-engram.h"
#include "llama-memory.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
struct llama_model;

struct llama_dsv41_memory_config {
    uint32_t n_ctx = 0;
    uint32_t n_seq = 0;
    uint32_t n_ubatch = 0;
    uint32_t n_layer = LLAMA_DSV41_N_LAYER;
    uint32_t raw_window = LLAMA_DSV41_N_SWA;
    uint32_t kv_width = LLAMA_DSV41_N_HEAD_DIM;
    uint32_t index_width = LLAMA_DSV41_N_INDEX_HEAD_DIM;
    uint32_t candidate_topk_blocks = LLAMA_DSV41_CANDIDATE_TOPK_BLOCKS;
    uint32_t candidate_block_size = LLAMA_DSV41_CANDIDATE_BLOCK_SIZE;
    uint32_t candidate_source_layer = LLAMA_DSV41_CANDIDATE_SOURCE_LAYER;
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_index = GGML_TYPE_F16;
    bool no_alloc = false;
    bool attach_no_alloc_buffers = false;
    bool engram_enabled = false;
    bool expert_enabled = false;
    std::vector<uint32_t> kv_sources = { 2, 8, 14, 20 };
    std::vector<uint32_t> index_sources = { 2, 8, 14, 20, 24, 28, 32, 36 };
    std::vector<uint32_t> ratios;
    std::function<ggml_backend_buffer_type_t(int32_t)> buft_for_layer;
    std::unique_ptr<llama_dsv41_engram_runtime> engram;
};

struct llama_dsv41_raw_plan {
    std::vector<int32_t> persist_src_idxs;
    std::vector<int64_t> write_idxs;
    std::vector<int32_t> read_idxs;
    std::vector<float> mask;
    std::vector<int32_t> n_visible;
};

struct llama_dsv41_source_plan {
    uint32_t source_layer = 0;
    uint32_t ratio = 0;
    uint32_t capacity = 0;
    llama_dsv41_compression_plan compression;
    std::vector<int32_t> read_idxs;
};

struct llama_dsv41_graph_topology {
    uint32_t n_tokens = 0;
    uint32_t n_seqs = 0;
    uint32_t n_outputs = 0;
    std::vector<llama_seq_id> seq_ids;
    std::vector<llama_pos> start_positions;
    std::vector<uint32_t> visible_raw_widths;
    std::vector<uint32_t> visible_compressed_widths;
    std::vector<uint32_t> source_ratios;
    std::vector<uint32_t> source_carry_counts;
    uint32_t candidate_width = 0;
    uint64_t backend_layout = 0;
    uint64_t transaction_generation = 0;
    bool engram_enabled = false;
    bool expert_enabled = false;

    bool same_topology(const llama_dsv41_graph_topology & other) const;
};

struct llama_dsv41_memory_plan {
    uint64_t generation = 0;
    std::vector<llama_pos> positions;
    std::vector<std::vector<llama_seq_id>> token_seq_ids;
    llama_dsv41_raw_plan raw;
    std::vector<llama_dsv41_source_plan> sources;
    uint32_t candidate_width = 0;
};

class llama_memory_dsv41;

class llama_memory_dsv41_context : public llama_memory_context_i {
public:
    llama_memory_dsv41_context(llama_memory_status status);
    llama_memory_dsv41_context(llama_memory_dsv41 * memory, bool full);
    llama_memory_dsv41_context(llama_memory_dsv41 * memory, std::vector<llama_ubatch> ubatches);
    ~llama_memory_dsv41_context() override;

    bool next() override;
    bool apply() override;
    void commit() override;
    void rollback() override;

    llama_memory_status get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    const llama_dsv41_memory_plan & plan() const;
    const llama_dsv41_memory_plan & graph_plan(const llama_ubatch & ubatch) const;
    llama_dsv41_graph_topology topology(const llama_ubatch & ubatch, uint32_t n_outputs) const;
    const llama_dsv41_engram_transaction * engram_transaction() const;
    std::vector<int32_t> engram_row_ids(uint32_t engram_layer) const;
    const llama_memory_dsv41 * memory() const { return mem; }

    void stage_candidate_ids(uint32_t token, const std::vector<int32_t> & ids);

private:
    struct transaction_state;

    llama_memory_status status;
    llama_memory_dsv41 * mem = nullptr;
    bool full = false;
    size_t i_next = 0;
    std::vector<llama_ubatch> ubatches;
    mutable std::unique_ptr<transaction_state> transaction;
};

class llama_memory_dsv41 : public llama_memory_i {
public:
    explicit llama_memory_dsv41(llama_dsv41_memory_config config);
    llama_memory_dsv41(
            const llama_model & model,
            ggml_type type_k,
            bool offload,
            uint32_t n_ctx,
            uint32_t n_seq,
            uint32_t n_ubatch,
            std::unique_ptr<llama_dsv41_engram_runtime> engram);
    ~llama_memory_dsv41() override;

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;
    void clear(bool data) override;
    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    [[noreturn]] void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    [[noreturn]] void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;
    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;
    void set_graph_workspace_size(size_t size) override;
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read(llama_io_read_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    ggml_tensor * raw_k(uint32_t layer) const;
    ggml_tensor * compressed_kv(uint32_t source_layer) const;
    ggml_tensor * index_keys(uint32_t source_layer) const;
    ggml_tensor * compressor_carry_kv(uint32_t source_layer) const;
    ggml_tensor * compressor_carry_score(uint32_t source_layer) const;
    ggml_tensor * candidate_scores() const;
    ggml_tensor * candidate_ids() const;
    ggml_tensor * committed_candidate_ids() const;
    ggml_tensor * position_state() const;

    const llama_dsv41_memory_config & config() const;
    llama_dsv41_memory_accounting accounting() const;
    std::vector<int32_t> sequence_candidate_ids(llama_seq_id seq_id) const;
    size_t retained_rollback_count() const;
    bool engram_enabled() const;

private:
    friend class llama_memory_dsv41_context;
    struct impl;
    std::unique_ptr<impl> pimpl;
};
