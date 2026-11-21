// hooks: common/console.h of the fork (1fe05fa45, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
// Console functions

#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include "common.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include <functional>
#include <string>
#include <vector>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

enum display_type {
    DISPLAY_TYPE_RESET = 0,
    DISPLAY_TYPE_INFO,
    DISPLAY_TYPE_PROMPT,
    DISPLAY_TYPE_REASONING,
    DISPLAY_TYPE_USER_INPUT,
    DISPLAY_TYPE_ERROR
};

namespace console {
    void init(bool use_simple_io, bool use_advanced_display);
    void cleanup();
    void set_display(display_type display);
    bool readline(std::string & line, bool multiline_input);

    using completion_callback = std::function<std::vector<std::pair<std::string, size_t>>(std::string_view, size_t)>;
    void set_completion_callback(completion_callback cb);

    namespace spinner {
        void start();
        void stop();
    }

    // note: the logging API below output directly to stdout
    // it can negatively impact performance if used on inference thread
    // only use in in a dedicated CLI thread
    // for logging in inference thread, use log.h instead

    LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
    void log(const char * fmt, ...);

    LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
    void error(const char * fmt, ...);

    void flush();
}
} // namespace ggml_fork
