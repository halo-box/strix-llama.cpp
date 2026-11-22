// hooks: common/jinja/parser.h of the fork (403dfcb03, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include "lexer.h"
#include "runtime.h"
#include "utils.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include <string>
#include <stdexcept>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

namespace jinja {

// parse from a list of tokens into an AST (program)
// may throw parser_exception on error
program parse_from_tokens(const lexer_result & lexer_res);

struct parser_exception : public std::runtime_error {
    parser_exception(const std::string & msg, const std::string & source, size_t pos)
        : std::runtime_error(fmt_error_with_source("parser", msg, source, pos)) {}
};

} // namespace jinja
} // namespace ggml_fork
