// hooks: common/unicode.h of the fork (25b1c175e, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include <cstdint>
#include <string_view>
#include <vector>
#include <string>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

// UTF-8 parsing utilities for streaming-aware unicode support

struct utf8_parse_result {
    uint32_t codepoint;      // Decoded codepoint (only valid if status == SUCCESS)
    size_t bytes_consumed;   // How many bytes this codepoint uses (1-4), or the length of the valid prefix if status != SUCCESS
    enum status { SUCCESS, INCOMPLETE, INVALID } status;

    utf8_parse_result(enum status s, uint32_t cp = 0, size_t bytes = 0)
        : codepoint(cp), bytes_consumed(bytes), status(s) {}
};

// Determine the expected length of a UTF-8 sequence from its first byte
// Returns 0 for invalid first bytes
size_t common_utf8_sequence_length(unsigned char first_byte);

// Check if a string ends with a complete UTF-8 sequence.
bool common_utf8_is_complete(const std::string & s);

// Parse a single UTF-8 codepoint from input
utf8_parse_result common_parse_utf8_codepoint(std::string_view input, size_t offset);

std::string common_unicode_cpts_to_utf8(const std::vector<uint32_t> & cps);
std::string common_unicode_cpt_to_utf8(uint32_t cpt);
} // namespace ggml_fork
