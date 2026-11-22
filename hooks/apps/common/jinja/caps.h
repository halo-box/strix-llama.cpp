// hooks: common/jinja/caps.h of the fork (403dfcb03, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include "runtime.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include <string>
#include <map>
#include <vector>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

namespace jinja {

struct caps {
    bool supports_tools = true;
    bool supports_tool_calls = true;
    bool supports_system_role = true;
    bool supports_parallel_tool_calls = true;

    // supports preserve reasoning trace in the full history, not just the last assistant message
    bool supports_preserve_reasoning = false;

    // supports reasoning effort levels
    bool supports_reasoning_effort = false;

    // reasoning effort levels the template renders distinctly, from lowest to highest
    std::vector<std::string> reasoning_effort_levels;

    // reasoning effort levels the template raises on
    std::vector<std::string> reasoning_effort_rejected;

    // one of the 2 content capabilities must be true
    bool supports_string_content = true;
    bool supports_typed_content = false;

    bool supports_object_arguments = false;

    // for reporting on server
    std::map<std::string, bool> to_map() const;

    // for debugging
    std::string to_string() const;
};

caps caps_get(jinja::program & prog);

void caps_apply_preserve_reasoning(jinja::context & ctx, bool enabled);
void caps_apply_reasoning_effort(jinja::context & ctx, const std::string & effort);

} // namespace jinja
} // namespace ggml_fork
