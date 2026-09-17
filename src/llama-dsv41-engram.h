#pragma once

#include "ggml-backend.h"
#include "llama-engram.h"
#include "llama.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

struct llama_dsv41_engram_extent {
    std::string fname;
    uint64_t offset = 0;
    uint32_t rows = 0;
    int64_t columns = 0;
    int64_t row_count = 0;
    int32_t type = 0;
};

void llama_dsv41_validate_engram_extent(const llama_dsv41_engram_extent & extent);

struct llama_dsv41_engram_token {
    int32_t token = -1;
    llama_pos pos = -1;
    std::vector<llama_seq_id> seq_ids;
    uint8_t text = 1;
};

struct llama_dsv41_engram_sequence_state {
    llama_engram_history history = {};
    llama_pos pos = -1;
};

struct llama_dsv41_engram_snapshot {
    std::map<llama_seq_id, llama_dsv41_engram_sequence_state> sequences;
};

struct llama_dsv41_engram_transaction {
    llama_dsv41_engram_transaction();
    ~llama_dsv41_engram_transaction();
    llama_dsv41_engram_transaction(llama_dsv41_engram_transaction && other) noexcept;
    llama_dsv41_engram_transaction & operator=(llama_dsv41_engram_transaction && other) noexcept;

    llama_dsv41_engram_transaction(const llama_dsv41_engram_transaction &) = delete;
    llama_dsv41_engram_transaction & operator=(const llama_dsv41_engram_transaction &) = delete;

    size_t token_count() const;
    const uint32_t * row_ids(uint32_t layer) const;
    const float * rows(uint32_t layer) const;
    const uint8_t * text_mask() const;
    void upload_layer(
            uint32_t layer,
            size_t token_offset,
            size_t token_count,
            ggml_tensor * rows_input,
            ggml_tensor * text_select_input) const;

    struct impl;
    std::unique_ptr<impl> pimpl;
};

class llama_dsv41_engram_runtime {
public:
    llama_dsv41_engram_runtime(
            llama_engram_layout layout,
            const std::array<llama_dsv41_engram_extent, LLAMA_ENGRAM_LAYERS> & extents,
            size_t max_tokens);
    ~llama_dsv41_engram_runtime();

    llama_dsv41_engram_transaction prepare(const std::vector<llama_dsv41_engram_token> & tokens);
    void commit(llama_dsv41_engram_transaction & transaction);
    void rollback(llama_dsv41_engram_transaction & transaction);

    void seq_reset(llama_seq_id seq_id);
    void seq_copy(llama_seq_id seq_id_src, llama_seq_id seq_id_dst);
    void seq_remove(llama_seq_id seq_id);
    llama_dsv41_engram_snapshot checkpoint() const;
    void restore(const llama_dsv41_engram_snapshot & snapshot);
    llama_dsv41_engram_sequence_state sequence(llama_seq_id seq_id) const;

    size_t max_tokens() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// text_select row i keeps the original residual; row tokens+i selects the BF16-updated residual.
ggml_tensor * llama_dsv41_build_engram_add(
        ggml_context * ctx,
        ggml_tensor * residual,
        ggml_tensor * projected,
        ggml_tensor * q_norm,
        ggml_tensor * k_norm,
        ggml_tensor * text_select,
        float rms_eps,
        ggml_backend_sched_t sched,
        ggml_backend_t backend_cpu);

ggml_tensor * llama_dsv41_build_engram_gate(
        ggml_context * ctx,
        ggml_tensor * dot,
        ggml_backend_sched_t sched,
        ggml_backend_t backend_cpu);

ggml_tensor * llama_dsv41_build_engram(
        ggml_context * ctx,
        ggml_tensor * residual,
        ggml_tensor * rows,
        ggml_tensor * engram_kv,
        ggml_tensor * q_norm,
        ggml_tensor * k_norm,
        ggml_tensor * text_select,
        float rms_eps,
        ggml_backend_sched_t sched,
        ggml_backend_t backend_cpu);
