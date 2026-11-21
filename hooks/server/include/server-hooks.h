#pragma once

// server hooks: request fields, slot lifecycle callbacks and extra HTTP routes, defined outside tools/server.
// Hook files go in hooks/server/handlers/. See hooks/README.md.
// The callbacks get plain values, not the server structs: those are private to server-context.cpp.
// on_request and register_routes wrap upstream functions (hooks/llama/llama-hooks.cmake). on_slot_start, on_token,
// on_slot_release are hook lines in server-context.cpp, and hooks_data is a field of task_params (server-task.h).

#include "llama.h"

#include "json.h"
#include "preset.h"
#include "server-task.h"

#include <functional>
#include <string>
#include <unordered_set>

struct server_http_context;
struct server_context;

#ifndef GGML_HOOKS
// hooks not configured: the hook lines of tools/server do nothing
inline void server_hooks_on_slot_start(int, int, llama_context *, const common_json &) {}
inline bool server_hooks_on_token(int, llama_token, const std::string &) { return true; }
inline void server_hooks_on_slot_release(int) {}
inline void server_hooks_on_task_launch(int, llama_context *, server_task &) {}
inline int  server_hooks_n_ctx_max(const llama_model *, const common_params &) { return 0; }
inline void server_hooks_on_update(const std::function<common_json()> &) {}
inline void server_hooks_on_usage(const server_task_result_cmpl_final &, common_json &) {}
inline void server_hooks_public_paths(std::unordered_set<std::string> &) {}
inline bool server_hooks_model_hidden(const common_preset &) { return false; }
#else

#include "ggml-hooks-link.h"

struct server_hooks {
    const char * name = nullptr;

    // a completion request was parsed: read your own fields from `body` and keep them in `hooks_data`,
    // which travels with the task (task_params::hooks_data) to the slot that runs it
    void (*on_request)(const common_json & body, common_json & hooks_data) = nullptr;

    // a slot starts a task
    void (*on_slot_start)(int id_slot, int id_task, llama_context * ctx, const common_json & hooks_data) = nullptr;

    // a token was sampled for a slot. Return false to stop the generation of this slot.
    bool (*on_token)(int id_slot, llama_token token, const std::string & piece) = nullptr;

    // a slot is released, before its state is reset
    void (*on_slot_release)(int id_slot) = nullptr;

    // add HTTP routes (server_http_context::get/post), before the server starts listening
    void (*register_routes)(server_http_context & http) = nullptr;

    // server_context::load_model: change the params before, see the loaded server after (ok: the load succeeded)
    void (*on_load_params)(common_params & params) = nullptr;
    void (*on_load)(server_context & server, bool ok) = nullptr;
    // the server_context is destroyed
    void (*on_unload)(server_context & server) = nullptr;

    // a task is about to be given to a slot: the hook can change it (prompt tokens, params)
    void (*on_task_launch)(int id_slot, llama_context * ctx, server_task & task) = nullptr;

    // context size of a slot: return > 0 to cap it there instead of the training context of the model
    int (*n_ctx_max)(const llama_model * model, const common_params & params) = nullptr;

    // start of each update_slots: slots() returns the slots as json (server_slot::to_json(true)), on demand
    void (*on_update)(const std::function<common_json()> & slots) = nullptr;

    // the OpenAI compatible usage of a final completion result
    void (*on_usage)(const server_task_result_cmpl_final & res, common_json & usage) = nullptr;

    // paths served without an API key, like the UI assets
    void (*public_paths)(std::unordered_set<std::string> & paths) = nullptr;

    // router mode: return true to hide a model of the presets from the model list
    bool (*model_hidden)(const common_preset & preset) = nullptr;
};

void server_hook_register(const server_hooks & hooks);

// called from tools/server (HOOK lines)
void server_hooks_on_slot_start(int id_slot, int id_task, llama_context * ctx, const common_json & hooks_data);
bool server_hooks_on_token(int id_slot, llama_token token, const std::string & piece);
void server_hooks_on_slot_release(int id_slot);
void server_hooks_on_task_launch(int id_slot, llama_context * ctx, server_task & task);
int  server_hooks_n_ctx_max(const llama_model * model, const common_params & params);
void server_hooks_on_update(const std::function<common_json()> & slots);
void server_hooks_on_usage(const server_task_result_cmpl_final & res, common_json & usage);
void server_hooks_public_paths(std::unordered_set<std::string> & paths);
bool server_hooks_model_hidden(const common_preset & preset);
void server_hooks_register_routes(server_http_context & http);

#define SERVER_HOOK_REGISTER(...) \
    static const bool GGML_HOOK_CONCAT(server_hooks_registered_, __LINE__) = [] { server_hooks h_ = __VA_ARGS__; server_hook_register(h_); return true; }()

#endif
