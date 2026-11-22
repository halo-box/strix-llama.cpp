// The llama-ext.h functions that common and the tools use, with the llama switch (see llama-dispatch.cpp).
// On: the fork functions through hooks/llama/lib/ext-bridge.cpp. Off: the upstream ones, renamed *__upstream.
// llama_arch_name and llama_arch_kv_name are tables of names, the upstream ones serve both.

#include "llama-ext.h"

#include <array>
#include <map>

bool llama_hook_fork_on(); // llama-dispatch.cpp

int32_t            ggml_fork_bridge_llama_model_n_expert(const void * model);
int32_t            ggml_fork_bridge_llama_model_n_devices(const void * model);
ggml_backend_dev_t ggml_fork_bridge_llama_model_get_device(const void * model, int i);
std::map<ggml_backend_buffer_type_t, std::array<size_t, 3>> ggml_fork_bridge_llama_get_memory_breakdown(const void * ctx);
bool               ggml_fork_bridge_llama_batch_ext_set_decision_order(void * batch, int32_t idx, int order);
const char *       ggml_fork_bridge_llama_model_decision_label(const void * model, uint32_t i);
uint32_t           ggml_fork_bridge_llama_model_get_tok_embd(const void * model, float * out);
uint32_t           ggml_fork_bridge_llama_model_get_tok_embd_rows(const void * model, const int32_t * tokens, uint32_t n_tokens, float * out);
uint32_t           ggml_fork_bridge_llama_model_n_tokens_delay(const void * model);
bool               ggml_fork_bridge_llama_model_classifier_free_guidance(const void * model);

LLAMA_API int32_t llama_model_n_expert__upstream(const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices__upstream(const struct llama_model * model);
LLAMA_API ggml_backend_dev_t llama_model_get_device__upstream(const struct llama_model * model, int i);
LLAMA_API llama_memory_breakdown llama_get_memory_breakdown__upstream(const struct llama_context * ctx);
LLAMA_API bool llama_batch_ext_set_decision_order__upstream(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);
LLAMA_API const char * llama_model_decision_label__upstream(const struct llama_model * model, uint32_t i);
LLAMA_API uint32_t llama_model_get_tok_embd__upstream(const struct llama_model * model, float * out);
LLAMA_API uint32_t llama_model_get_tok_embd_rows__upstream(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out);
LLAMA_API uint32_t llama_model_n_tokens_delay__upstream(const struct llama_model * model);
LLAMA_API bool llama_model_classifier_free_guidance__upstream(const struct llama_model * model);

int32_t llama_model_n_expert(const struct llama_model * model) {
    return llama_hook_fork_on() ? ggml_fork_bridge_llama_model_n_expert(model) : llama_model_n_expert__upstream(model);
}

int32_t llama_model_n_devices(const struct llama_model * model) {
    return llama_hook_fork_on() ? ggml_fork_bridge_llama_model_n_devices(model) : llama_model_n_devices__upstream(model);
}

ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i) {
    return llama_hook_fork_on() ? ggml_fork_bridge_llama_model_get_device(model, i) : llama_model_get_device__upstream(model, i);
}

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    if (!llama_hook_fork_on()) {
        return llama_get_memory_breakdown__upstream(ctx);
    }
    llama_memory_breakdown out;
    for (const auto & [buft, mb] : ggml_fork_bridge_llama_get_memory_breakdown(ctx)) {
        out[buft].model   = mb[0];
        out[buft].context = mb[1];
        out[buft].compute = mb[2];
    }
    return out;
}

bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order) {
    return llama_hook_fork_on() ? ggml_fork_bridge_llama_batch_ext_set_decision_order(batch, idx, (int) order)
                                : llama_batch_ext_set_decision_order__upstream(batch, idx, order);
}

const char * llama_model_decision_label(const struct llama_model * model, uint32_t i) {
    return llama_hook_fork_on() ? ggml_fork_bridge_llama_model_decision_label(model, i) : llama_model_decision_label__upstream(model, i);
}

uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out) {
    return llama_hook_fork_on() ? ggml_fork_bridge_llama_model_get_tok_embd(model, out) : llama_model_get_tok_embd__upstream(model, out);
}

uint32_t llama_model_get_tok_embd_rows(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out) {
    return llama_hook_fork_on() ? ggml_fork_bridge_llama_model_get_tok_embd_rows(model, tokens, n_tokens, out)
                                : llama_model_get_tok_embd_rows__upstream(model, tokens, n_tokens, out);
}

uint32_t llama_model_n_tokens_delay(const struct llama_model * model) {
    return llama_hook_fork_on() ? ggml_fork_bridge_llama_model_n_tokens_delay(model) : llama_model_n_tokens_delay__upstream(model);
}

bool llama_model_classifier_free_guidance(const struct llama_model * model) {
    return llama_hook_fork_on() ? ggml_fork_bridge_llama_model_classifier_free_guidance(model)
                                : llama_model_classifier_free_guidance__upstream(model);
}
