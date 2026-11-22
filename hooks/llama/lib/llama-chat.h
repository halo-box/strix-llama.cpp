// hooks: src/llama-chat.h of the fork (3b76dee48, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/lib/README.md

} // namespace ggml_fork
#include <string>
#include <vector>
#include <cstdint>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/lib/README.md

enum llm_chat_template {
    LLM_CHAT_TEMPLATE_CHATML,
    LLM_CHAT_TEMPLATE_LLAMA_2,
    LLM_CHAT_TEMPLATE_LLAMA_2_SYS,
    LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS,
    LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP,
    LLM_CHAT_TEMPLATE_MISTRAL_V1,
    LLM_CHAT_TEMPLATE_MISTRAL_V3,
    LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN,
    LLM_CHAT_TEMPLATE_MISTRAL_V7,
    LLM_CHAT_TEMPLATE_MISTRAL_V7_TEKKEN,
    LLM_CHAT_TEMPLATE_PHI_3,
    LLM_CHAT_TEMPLATE_PHI_4,
    LLM_CHAT_TEMPLATE_FALCON_3,
    LLM_CHAT_TEMPLATE_ZEPHYR,
    LLM_CHAT_TEMPLATE_MONARCH,
    LLM_CHAT_TEMPLATE_GEMMA,
    LLM_CHAT_TEMPLATE_ORION,
    LLM_CHAT_TEMPLATE_OPENCHAT,
    LLM_CHAT_TEMPLATE_VICUNA,
    LLM_CHAT_TEMPLATE_VICUNA_ORCA,
    LLM_CHAT_TEMPLATE_DEEPSEEK,
    LLM_CHAT_TEMPLATE_DEEPSEEK_2,
    LLM_CHAT_TEMPLATE_DEEPSEEK_3,
    LLM_CHAT_TEMPLATE_DEEPSEEK_OCR,
    LLM_CHAT_TEMPLATE_COMMAND_R,
    LLM_CHAT_TEMPLATE_LLAMA_3,
    LLM_CHAT_TEMPLATE_CHATGLM_3,
    LLM_CHAT_TEMPLATE_CHATGLM_4,
    LLM_CHAT_TEMPLATE_GLMEDGE,
    LLM_CHAT_TEMPLATE_MINICPM,
    LLM_CHAT_TEMPLATE_EXAONE_3,
    LLM_CHAT_TEMPLATE_EXAONE_4,
    LLM_CHAT_TEMPLATE_EXAONE_MOE,
    LLM_CHAT_TEMPLATE_RWKV_WORLD,
    LLM_CHAT_TEMPLATE_GRANITE_3_X,
    LLM_CHAT_TEMPLATE_GRANITE_4_0,
    LLM_CHAT_TEMPLATE_GRANITE_4_1,
    LLM_CHAT_TEMPLATE_GIGACHAT,
    LLM_CHAT_TEMPLATE_MEGREZ,
    LLM_CHAT_TEMPLATE_YANDEX,
    LLM_CHAT_TEMPLATE_BAILING,
    LLM_CHAT_TEMPLATE_BAILING_THINK,
    LLM_CHAT_TEMPLATE_BAILING2,
    LLM_CHAT_TEMPLATE_LLAMA4,
    LLM_CHAT_TEMPLATE_SMOLVLM,
    LLM_CHAT_TEMPLATE_DOTS1,
    LLM_CHAT_TEMPLATE_HUNYUAN_MOE,
    LLM_CHAT_TEMPLATE_OPENAI_MOE,
    LLM_CHAT_TEMPLATE_HUNYUAN_DENSE,
    LLM_CHAT_TEMPLATE_HUNYUAN_VL,
    LLM_CHAT_TEMPLATE_KIMI_K2,
    LLM_CHAT_TEMPLATE_SEED_OSS,
    LLM_CHAT_TEMPLATE_GROK_2,
    LLM_CHAT_TEMPLATE_PANGU_EMBED,
    LLM_CHAT_TEMPLATE_SOLAR_OPEN,
    LLM_CHAT_TEMPLATE_UNKNOWN,
};

struct llama_chat_message;

llm_chat_template llm_chat_template_from_str(const std::string & name);

llm_chat_template llm_chat_detect_template(const std::string & tmpl);

int32_t llm_chat_apply_template(
    llm_chat_template tmpl,
    const std::vector<const llama_chat_message *> & chat,
    std::string & dest, bool add_ass);
} // namespace ggml_fork
