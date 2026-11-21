// hooks: common/parsers/parsers.cpp of the fork (1fe05fa45, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
#include "parsers.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include "log.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

void foreach_function(const json & tools, const std::function<void(size_t, const json &)> & fn) {
    for (size_t i = 0; i < tools.size(); i++) {
        const auto & tool = tools[i];
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            LOG_INF("Skipping tool without function: %s", tool.dump(2).c_str());
            continue;
        }
        fn(i, tool);
    }
}

void foreach_parameter(const json & function, const std::function<void(size_t, const common_chat_schema_property &, const common_chat_schema_document_ptr &)> & fn) {
    auto         params = common_chat_tool_parameters(function);
    auto         doc    = std::make_shared<const common_chat_schema_document>(common_chat_schema_from_json(params));
    const auto * object = dynamic_cast<const common_chat_schema_object *>(doc->root.get());
    if (!object) {
        return;
    }
    for (size_t i = 0; i < object->properties.size(); i++) {
        fn(i, object->properties[i], doc);
    }
}
} // namespace ggml_fork
