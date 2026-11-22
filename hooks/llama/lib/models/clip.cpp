// hooks: src/models/clip.cpp of the fork (403dfcb03, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
#include "models.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/lib/README.md

// Stub to allow llama-quantize to open mmproj GGUFs

[[noreturn]]
void llama_model_clip::load_arch_hparams(llama_model_loader &) {
    GGML_ABORT("CLIP is a quant-only stub; load_arch_hparams should not be called");
}

[[noreturn]]
void llama_model_clip::load_arch_tensors(llama_model_loader &) {
    GGML_ABORT("CLIP is a quant-only stub; load_arch_tensors should not be called");
}

[[noreturn]]
std::unique_ptr<llm_graph_context> llama_model_clip::build_arch_graph(const llm_graph_params &) const {
    GGML_ABORT("CLIP has no inference graph via llama_model dispatch; runtime lives in tools/mtmd");
}
} // namespace ggml_fork
