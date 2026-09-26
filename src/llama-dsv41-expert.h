#pragma once

#include "llama-expert-store.h"

#include "ggml-backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

struct llama_dsv41_expert_runtime_params {
    size_t cache_bytes = 0;
    size_t cache_slots = 0;
    bool direct_io = true;
    bool allow_buffered_io = false;
    bool no_alloc = false;
};

struct llama_dsv41_expert_runtime {
    using buft_selector = std::function<ggml_backend_buffer_type_t(const llama_expert_store_tensor & tensor)>;
    using upload_fn = std::function<void(ggml_tensor * tensor, size_t offset, const void * data, size_t size)>;
    using publish_fn = std::function<void()>;

    llama_dsv41_expert_runtime(
            std::vector<llama_expert_store_tensor> tensors,
            const llama_dsv41_expert_runtime_params & params,
            buft_selector select_buft,
            upload_fn upload = {},
            publish_fn before_publish = {});
    ~llama_dsv41_expert_runtime();

    llama_dsv41_expert_runtime(const llama_dsv41_expert_runtime &) = delete;
    llama_dsv41_expert_runtime & operator=(const llama_dsv41_expert_runtime &) = delete;

    std::vector<int32_t> remap(int32_t layer, const std::vector<int32_t> & expert_ids);
    void acquire_context();
    void release_context();
    void release(int32_t layer);
    void release_all();
    void release_all_after_sync(ggml_backend_sched_t sched);

    ggml_tensor * cache_tensor(int32_t layer, llama_expert_projection projection) const;
    size_t cache_slots() const;
    size_t cache_bytes() const;
    size_t staging_bytes() const;

    void set_error(const std::string & error);
    std::string consume_error();

private:
    friend ggml_tensor * llama_dsv41_build_expert_remap(
            ggml_context *, ggml_tensor *, llama_dsv41_expert_runtime &, int32_t, ggml_backend_sched_t, ggml_backend_t);
    friend ggml_tensor * llama_dsv41_build_expert_release(
            ggml_context *, ggml_tensor *, llama_dsv41_expert_runtime &, int32_t, ggml_backend_sched_t, ggml_backend_t);

    struct impl;
    std::unique_ptr<impl> pimpl;
};

std::vector<llama_expert_store_tensor> llama_dsv41_register_expert_tensors(
        const std::function<llama_expert_store_tensor(
                const std::string & name,
                int32_t layer,
                llama_expert_projection projection,
                const std::initializer_list<int64_t> & ne)> & register_tensor);

ggml_tensor * llama_dsv41_build_expert_remap(
        ggml_context * ctx,
        ggml_tensor * selected_experts,
        llama_dsv41_expert_runtime & runtime,
        int32_t layer,
        ggml_backend_sched_t sched,
        ggml_backend_t backend_cpu);

ggml_tensor * llama_dsv41_build_expert_release(
        ggml_context * ctx,
        ggml_tensor * experts,
        llama_dsv41_expert_runtime & runtime,
        int32_t layer,
        ggml_backend_sched_t sched,
        ggml_backend_t backend_cpu);
