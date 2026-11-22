// hooks: common/json-schema-to-grammar.h of the fork (25b1c175e, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include "json-schema.h"
#include "json.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include <functional>
#include <string>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

std::string json_schema_to_grammar(const common_json & schema, bool force_gbnf = false);
std::string json_schema_to_grammar(const common_chat_schema_document & schema);

struct common_grammar_builder {
    std::function<std::string(const std::string &, const std::string &)>    add_rule;
    std::function<std::string(const std::string &, const common_chat_schema &)> add_schema;
};

struct common_grammar_options {
    bool dotall = false;
};

std::string gbnf_format_literal(const std::string & literal);
std::string gbnf_format_rule_name(const std::string & name);

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options = {});
} // namespace ggml_fork
