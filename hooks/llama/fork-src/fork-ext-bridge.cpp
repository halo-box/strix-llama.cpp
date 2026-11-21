// The llama-ext.h functions of the fork used by common and the tools, with types that both trees share, for the
// dispatcher (hooks/llama/src/llama-fork-ext-dispatch.cpp). Built with the fork libllama.

#include "llama-ext.h"

#include <array>
#include <map>

int32_t ggml_fork_bridge_llama_model_n_expert(const void * model) {
    return ggml_fork::llama_model_n_expert((const ggml_fork::llama_model *) model);
}

int32_t ggml_fork_bridge_llama_model_n_devices(const void * model) {
    return ggml_fork::llama_model_n_devices((const ggml_fork::llama_model *) model);
}

ggml_backend_dev_t ggml_fork_bridge_llama_model_get_device(const void * model, int i) {
    return ggml_fork::llama_model_get_device((const ggml_fork::llama_model *) model, i);
}

// model, context, compute
std::map<ggml_backend_buffer_type_t, std::array<size_t, 3>> ggml_fork_bridge_llama_get_memory_breakdown(const void * ctx) {
    std::map<ggml_backend_buffer_type_t, std::array<size_t, 3>> out;
    for (const auto & [buft, mb] : ggml_fork::llama_get_memory_breakdown((const ggml_fork::llama_context *) ctx)) {
        out[buft] = { mb.model, mb.context, mb.compute };
    }
    return out;
}

bool ggml_fork_bridge_llama_batch_ext_set_decision_order(void * batch, int32_t idx, int order) {
    return ggml_fork::llama_batch_ext_set_decision_order((ggml_fork::llama_batch_ext *) batch, idx, (ggml_fork::llama_decision_order) order);
}

const char * ggml_fork_bridge_llama_model_decision_label(const void * model, uint32_t i) {
    return ggml_fork::llama_model_decision_label((const ggml_fork::llama_model *) model, i);
}

uint32_t ggml_fork_bridge_llama_model_get_tok_embd(const void * model, float * out) {
    return ggml_fork::llama_model_get_tok_embd((const ggml_fork::llama_model *) model, out);
}

uint32_t ggml_fork_bridge_llama_model_get_tok_embd_rows(const void * model, const int32_t * tokens, uint32_t n_tokens, float * out) {
    return ggml_fork::llama_model_get_tok_embd_rows((const ggml_fork::llama_model *) model, tokens, n_tokens, out);
}

uint32_t ggml_fork_bridge_llama_model_n_tokens_delay(const void * model) {
    return ggml_fork::llama_model_n_tokens_delay((const ggml_fork::llama_model *) model);
}

bool ggml_fork_bridge_llama_model_classifier_free_guidance(const void * model) {
    return ggml_fork::llama_model_classifier_free_guidance((const ggml_fork::llama_model *) model);
}
