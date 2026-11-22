// hooks: src/llama-cparams.cpp of the fork (3b76dee48, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
#include "llama-cparams.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/lib/README.md

size_t llama_max_parallel_sequences(void) {
    return LLAMA_MAX_SEQ;
}
} // namespace ggml_fork
