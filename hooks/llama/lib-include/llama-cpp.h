// hooks: include/llama-cpp.h of the fork (1fe05fa45, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/lib/README.md

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

} // namespace ggml_fork
#include <memory>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/lib/README.md

} // namespace ggml_fork
#include "llama.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/lib/README.md

struct llama_model_deleter {
    void operator()(llama_model * model) { llama_model_free(model); }
};

struct llama_context_deleter {
    void operator()(llama_context * context) { llama_free(context); }
};

struct llama_sampler_deleter {
    void operator()(llama_sampler * sampler) { llama_sampler_free(sampler); }
};

struct llama_adapter_lora_deleter {
    void operator()(llama_adapter_lora * adapter) { llama_adapter_lora_free(adapter); }
};

struct llama_batch_ext_deleter {
    void operator()(llama_batch_ext * batch) { llama_batch_ext_free(batch); }
};

typedef std::unique_ptr<llama_model, llama_model_deleter> llama_model_ptr;
typedef std::unique_ptr<llama_context, llama_context_deleter> llama_context_ptr;
typedef std::unique_ptr<llama_sampler, llama_sampler_deleter> llama_sampler_ptr;
typedef std::unique_ptr<llama_adapter_lora, llama_adapter_lora_deleter> llama_adapter_lora_ptr;
typedef std::unique_ptr<llama_batch_ext, llama_batch_ext_deleter> llama_batch_ext_ptr;
} // namespace ggml_fork
