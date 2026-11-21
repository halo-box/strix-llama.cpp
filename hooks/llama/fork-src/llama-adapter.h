// hooks: src/llama-adapter.h of the fork (1fe05fa45, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/fork-src/README.md

} // namespace ggml_fork
#include "llama.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/fork-src/README.md

} // namespace ggml_fork
#include "ggml-cpp.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/fork-src/README.md

} // namespace ggml_fork
#include <string>
#include <unordered_map>
#include <vector>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/fork-src/README.md

// TODO: pimpl

//
// llama_adapter_cvec
//

struct llama_adapter_cvec {
    ggml_tensor * tensor_for(int il) const;

    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const;

    bool apply(
            const llama_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

private:
    bool init(const llama_model & model);

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<ggml_tensor *> tensors; // per layer
};

using llama_adapter_cvec_ptr = std::shared_ptr<llama_adapter_cvec>;

//
// llama_adapter_lora
//

struct llama_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llama_adapter_lora_weight() = default;
    llama_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b) : a(a), b(b) {}
};

struct llama_adapter_lora {
    llama_model * model = nullptr;

    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<llama_token> alora_invocation_tokens;

    explicit llama_adapter_lora(llama_model * model) : model(model) {}
    ~llama_adapter_lora() = default;

    llama_adapter_lora_weight * get_weight(ggml_tensor * w);

    uint32_t get_n_nodes() const {
        return ab_map.size() * 6u; // a, b, scale, add, 2 x mul_mat
    }
};

using llama_adapter_loras = std::unordered_map<llama_adapter_lora *, float>;
using llama_adapter_loras_ptr = std::unique_ptr<llama_adapter_loras>;
} // namespace ggml_fork
