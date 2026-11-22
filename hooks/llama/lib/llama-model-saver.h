// hooks: src/llama-model-saver.h of the fork (1fe05fa45, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/lib/README.md

} // namespace ggml_fork
#include "gguf.h"
#include "llama.h"
#include "llama-arch.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/lib/README.md

} // namespace ggml_fork
#include <vector>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/lib/README.md

struct llama_model_saver {
    struct gguf_context * gguf_ctx = nullptr;
    const struct llama_model * model;
    const struct LLM_KV llm_kv;

    llama_model_saver(const struct llama_model * model);
    ~llama_model_saver();

    void add_kv(enum llm_kv key, uint32_t     value);
    void add_kv(enum llm_kv key, int32_t      value);
    void add_kv(enum llm_kv key, uint64_t     value);
    void add_kv(enum llm_kv key, float        value);
    void add_kv(enum llm_kv key, bool         value);
    void add_kv(enum llm_kv key, const char * value);

    [[noreturn]]
    void add_kv(enum llm_kv key, char value); // needed to make the template below compile

    template <typename Container>
    void add_kv(enum llm_kv key, const Container & value, bool per_layer = false);

    void add_kv(enum llm_kv key, const std::vector<std::string> & value);

    void add_tensor(const struct ggml_tensor * tensor);

    void add_kv_from_model();

    void add_tensors_from_model();

    void save(const std::string & path_model);
};
} // namespace ggml_fork
