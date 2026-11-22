// hooks: src/models/mistral4.cpp of the fork (1fe05fa45, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
#include "models.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/lib/README.md

std::unique_ptr<llm_graph_context> llama_model_mistral4::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

} // namespace ggml_fork
