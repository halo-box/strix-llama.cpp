// hooks: src/models/jina-bert-v3.cpp of the fork (403dfcb03, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
#include "models.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/lib/README.md

void llama_model_jina_bert_v3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);

    switch (hparams.n_layer()) {
        case 24:
            type = LLM_TYPE_558M; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_jina_bert_v3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (n_token_types == 0) {
        throw std::runtime_error(arch_name() + " model needs to define token type count");
    }
    tok_embd     = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    type_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_TYPES, "weight"), {n_embd, n_token_types}, TENSOR_NOT_REQUIRED);

    tok_norm   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);
    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias",   0), {n_embd}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0);

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);

        layer.attn_out_norm   = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0);
        layer.attn_out_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "bias", i),   {n_embd}, 0);

        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, TENSOR_NOT_REQUIRED);
        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

        layer.layer_out_norm   = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, 0);
        layer.layer_out_norm_b = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "bias", i),   {n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_jina_bert_v3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

} // namespace ggml_fork
