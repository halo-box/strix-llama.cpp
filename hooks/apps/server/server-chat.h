// hooks: tools/server/server-chat.h of the fork (3b76dee48, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
// Chat conversion functions for server (Responses API, Anthropic API, OAI streaming diffs)

#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include "chat.h"
#include "server-common.h"
#include "server-http.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include "json.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

// Convert OpenAI Responses API format to OpenAI Chat Completions API format
json server_chat_convert_responses_to_chatcmpl(const json & body);

// Convert OpenAI Responses API tools to OpenAI Chat Completions API tools, non-function tools are skipped
json server_chat_convert_responses_tools(const json & tools);

// Convert Anthropic Messages API format to OpenAI Chat Completions API format
json server_chat_convert_anthropic_to_oai(const json & body);

// convert OpenAI transcriptions API format to OpenAI Chat Completions API format
json convert_transcriptions_to_chatcmpl(
    const json & body,
    const common_chat_templates * tmpls,
    const std::map<std::string, uploaded_file> & in_files,
    std::vector<raw_buffer> & out_files);

// convert OpenAI speech API format to the /tts format
json convert_speech_to_tts(const json & body);

// prompt of the "asr" chat template, live ASR models (e.g. LFM3-Audio live ASR) transcribe with it
// empty if the model has no such template, throws on a template error
llama_tokens server_chat_get_asr_prompt(const llama_model * model);

json server_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff);
} // namespace ggml_fork
