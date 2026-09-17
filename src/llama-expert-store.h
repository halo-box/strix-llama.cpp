#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum llama_expert_projection {
    LLAMA_EXPERT_PROJECTION_GATE = 0,
    LLAMA_EXPERT_PROJECTION_UP,
    LLAMA_EXPERT_PROJECTION_DOWN,
};

struct llama_expert_store_tensor {
    std::string             name;
    std::string             fname;
    size_t                  file_index  = 0;
    int32_t                 layer       = -1;
    llama_expert_projection projection  = LLAMA_EXPERT_PROJECTION_GATE;
    ggml_type               type        = GGML_TYPE_COUNT;
    int64_t                 ne[3]       = {};
    size_t                  nb[3]       = {};
    uint64_t                file_offset = 0;
    uint64_t                file_size   = 0;
};

struct llama_expert_store_params {
    size_t cache_bytes = 0;
    size_t cache_slots = 0;
    size_t io_alignment = 4096;
    bool   direct_io = true;
    bool   allow_buffered_io = false; // opt-in only; page-cache bytes are outside cache_bytes
};

struct llama_expert_store_request {
    int32_t                 layer      = -1;
    llama_expert_projection projection = LLAMA_EXPERT_PROJECTION_GATE;
    std::vector<int32_t>    expert_ids;
};

struct llama_expert_store_stats {
    uint64_t hits       = 0;
    uint64_t misses     = 0;
    uint64_t bytes_read = 0;
    uint64_t evictions  = 0;
};

struct llama_expert_store_aligned_read {
    uint64_t offset = 0;
    size_t   size   = 0;
    size_t   prefix = 0;
};

llama_expert_store_aligned_read llama_expert_store_align_read(
        uint64_t offset, size_t size, size_t alignment, uint64_t file_size);

void llama_expert_store_validate_tensor(const llama_expert_store_tensor & tensor);

struct llama_expert_store {
    struct payload {
        int32_t                 layer      = -1;
        llama_expert_projection projection = LLAMA_EXPERT_PROJECTION_GATE;
        int32_t                 expert_id  = -1;
        uint32_t                slot_id    = 0;
        ggml_type               type       = GGML_TYPE_COUNT;
        const uint8_t *         data       = nullptr;
        size_t                  size       = 0;
    };

    struct lease {
        lease();
        lease(lease && other) noexcept;
        lease & operator=(lease && other) noexcept;
        ~lease();

        lease(const lease &) = delete;
        lease & operator=(const lease &) = delete;

        const std::vector<std::vector<uint32_t>> & slot_ids() const;
        std::vector<payload> payloads() const;

    private:
        friend struct llama_expert_store;

        struct impl;
        std::unique_ptr<impl> pimpl;
    };

    llama_expert_store(std::vector<llama_expert_store_tensor> tensors, const llama_expert_store_params & params);
    ~llama_expert_store();

    // The lease pins every unique returned slot. Keep it until the backend upload completes.
    lease acquire(const std::vector<llama_expert_store_request> & requests);

    llama_expert_store_stats stats() const;
    size_t resident_bytes() const;
    size_t resident_entries() const;
    bool direct_io_active() const;

private:
    struct impl;
    std::shared_ptr<impl> pimpl;
};
