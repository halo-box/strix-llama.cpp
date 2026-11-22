// hooks: common/imatrix-loader.h of the fork (1fe05fa45, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include <cstdint>
#include <map>
#include <string>
#include <vector>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

inline constexpr const char * LLM_KV_IMATRIX_DATASETS    = "imatrix.datasets";
inline constexpr const char * LLM_KV_IMATRIX_CHUNK_COUNT = "imatrix.chunk_count";
inline constexpr const char * LLM_KV_IMATRIX_CHUNK_SIZE  = "imatrix.chunk_size";
inline constexpr const char * LLM_KV_IMATRIX_STATS_SCHEMA = "imatrix.stats_schema";
inline constexpr const char * LLM_KV_IMATRIX_N_LAYER_NEXTN = "imatrix.n_layer_nextn";

struct common_imatrix_entry {
    std::vector<float>   sums;
    std::vector<float>   activations;
    std::vector<int64_t> counts;
};

struct common_imatrix {
    std::map<std::string, common_imatrix_entry> entries;
    std::vector<std::string> datasets;
    int32_t chunk_count    = 0;
    int32_t chunk_size     = 0;
    int32_t n_layer_nextn  = 0;
    bool    is_legacy      = false;
    bool    has_metadata   = false;
};

bool common_imatrix_load(const std::string & fname, common_imatrix & imatrix);
} // namespace ggml_fork
