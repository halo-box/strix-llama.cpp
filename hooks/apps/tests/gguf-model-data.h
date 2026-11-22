// hooks: tests/gguf-model-data.h of the fork (3b76dee48, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include "ggml-cpp.h"
#include "gguf.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

struct gguf_remote_tensor {
    std::string  name;
    ggml_type    type    = GGML_TYPE_F32;
    int64_t      ne[4]   = {1, 1, 1, 1};  // dimensions, unused dims = 1
    uint32_t     n_dims  = 0;
};

struct gguf_remote_model {
    // Selected KV metadata
    std::string architecture;               // general.architecture
    uint32_t    n_embd          = 0;        // <arch>.embedding_length
    uint32_t    n_ff            = 0;        // <arch>.feed_forward_length
    uint32_t    n_vocab         = 0;        // inferred from token_embd.weight ne[1]
    uint32_t    n_layer         = 0;        // <arch>.block_count
    uint32_t    n_head          = 0;        // <arch>.attention.head_count
    uint32_t    n_head_kv       = 0;        // <arch>.attention.head_count_kv
    uint32_t    n_expert        = 0;        // <arch>.expert_count (0 if absent)
    uint32_t    n_embd_head_k   = 0;        // <arch>.attention.key_length
    uint32_t    n_embd_head_v   = 0;        // <arch>.attention.value_length
    uint16_t    n_split         = 0;        // split.count (0 = not split)
    uint32_t    n_split_tensors = 0;        // split.tensors.count (0 if not split)

    std::vector<gguf_remote_tensor> tensors;
};

// Fetch model metadata from HuggingFace with local caching.
// repo: e.g., "ggml-org/Qwen3-32B-GGUF"
// quant: e.g., "Q8_0" -- auto-detects filename (including first shard of split models)
// Returns nullopt if download fails or network is unavailable.
std::optional<gguf_remote_model> gguf_fetch_model_meta(
    const std::string & repo,
    const std::string & quant = "Q8_0",
    const std::string & cache_dir = "",  // empty = default
    bool verbose = true);

gguf_context_ptr gguf_fetch_gguf_ctx(
    const std::string & repo,
    const std::string & quant = "Q8_0",
    const std::string & cache_dir = "",
    bool verbose = true);
} // namespace ggml_fork
