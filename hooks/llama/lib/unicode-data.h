// hooks: src/unicode-data.h of the fork (1fe05fa45, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/lib/README.md

} // namespace ggml_fork
#include <cstdint>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/lib/README.md

struct range_nfd {
    uint32_t first;
    uint32_t last;
    uint32_t nfd;
};

extern const std::unordered_map<uint32_t, uint8_t> unicode_combining_class;
extern const std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> unicode_decomposition;
extern const std::unordered_map<uint64_t, uint32_t> unicode_composition;

static const uint32_t MAX_CODEPOINTS = 0x110000;

extern const std::initializer_list<std::pair<uint32_t, uint16_t>> unicode_ranges_flags;
extern const std::unordered_set<uint32_t> unicode_set_whitespace;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_lowercase;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_uppercase;
extern const std::initializer_list<range_nfd> unicode_ranges_nfd;
} // namespace ggml_fork
