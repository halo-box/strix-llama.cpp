#include "server-hooks.h"

#include "server-context.h"
#include "server-task.h"

#include <mutex>
#include <vector>

namespace {

struct hooks_registry {
    std::mutex                mutex;
    std::vector<server_hooks> hooks;
};

hooks_registry & registry() {
    static auto * r = new hooks_registry();
    return *r;
}

// registration happens at load time, before the server threads start
const std::vector<server_hooks> & all() {
    return registry().hooks;
}

} // namespace

void server_hook_register(const server_hooks & hooks) {
    auto & r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    r.hooks.push_back(hooks);
}

void server_hooks_on_slot_start(int id_slot, int id_task, llama_context * ctx, const common_json & hooks_data) {
    for (const auto & h : all()) {
        if (h.on_slot_start) {
            h.on_slot_start(id_slot, id_task, ctx, hooks_data);
        }
    }
}

bool server_hooks_on_token(int id_slot, llama_token token, const std::string & piece) {
    bool cont = true;
    for (const auto & h : all()) {
        if (h.on_token && !h.on_token(id_slot, token, piece)) {
            cont = false;
        }
    }
    return cont;
}

void server_hooks_on_slot_release(int id_slot) {
    for (const auto & h : all()) {
        if (h.on_slot_release) {
            h.on_slot_release(id_slot);
        }
    }
}

void server_hooks_on_task_launch(int id_slot, llama_context * ctx, server_task & task) {
    for (const auto & h : all()) {
        if (h.on_task_launch) {
            h.on_task_launch(id_slot, ctx, task);
        }
    }
}

int server_hooks_n_ctx_max(const llama_model * model, const common_params & params) {
    for (const auto & h : all()) {
        if (h.n_ctx_max) {
            if (const int n = h.n_ctx_max(model, params); n > 0) {
                return n;
            }
        }
    }
    return 0;
}

void server_hooks_on_update(const std::function<common_json()> & slots) {
    for (const auto & h : all()) {
        if (h.on_update) {
            h.on_update(slots);
        }
    }
}

void server_hooks_on_usage(const server_task_result_cmpl_final & res, common_json & usage) {
    for (const auto & h : all()) {
        if (h.on_usage) {
            h.on_usage(res, usage);
        }
    }
}

void server_hooks_public_paths(std::unordered_set<std::string> & paths) {
    for (const auto & h : all()) {
        if (h.public_paths) {
            h.public_paths(paths);
        }
    }
}

bool server_hooks_model_hidden(const common_preset & preset) {
    for (const auto & h : all()) {
        if (h.model_hidden && h.model_hidden(preset)) {
            return true;
        }
    }
    return false;
}

// link-time wrappers, see hooks/llama/llama-hooks.cmake

#define SERVER_HOOK_WRAP extern "C" __attribute__((visibility("default")))

extern "C" task_params __real__ZN13server_schema22eval_llama_cmpl_schemaEPK11llama_vocabRK13common_paramsRKSt6vectorI16llama_logit_biasSaIS7_EERK11common_json(
        const llama_vocab * vocab, const common_params & params_base, const std::vector<llama_logit_bias> & logit_bias_eog, const common_json & data);
SERVER_HOOK_WRAP task_params __wrap__ZN13server_schema22eval_llama_cmpl_schemaEPK11llama_vocabRK13common_paramsRKSt6vectorI16llama_logit_biasSaIS7_EERK11common_json(
        const llama_vocab * vocab, const common_params & params_base, const std::vector<llama_logit_bias> & logit_bias_eog, const common_json & data) {
    task_params params = __real__ZN13server_schema22eval_llama_cmpl_schemaEPK11llama_vocabRK13common_paramsRKSt6vectorI16llama_logit_biasSaIS7_EERK11common_json(
        vocab, params_base, logit_bias_eog, data);
    for (const auto & h : all()) {
        if (h.on_request) {
            h.on_request(data, params.hooks_data);
        }
    }
    return params;
}

// the routes of the hooks, called by the wrapper of server_http_context::start (server-hooks-http.cpp)
void server_hooks_register_routes(server_http_context & http) {
    for (const auto & h : all()) {
        if (h.register_routes) {
            h.register_routes(http);
        }
    }
}

extern "C" bool __real__ZN14server_context10load_modelER13common_params(server_context * self, common_params & params);
SERVER_HOOK_WRAP bool __wrap__ZN14server_context10load_modelER13common_params(server_context * self, common_params & params) {
    for (const auto & h : all()) {
        if (h.on_load_params) {
            h.on_load_params(params);
        }
    }
    const bool ok = __real__ZN14server_context10load_modelER13common_params(self, params);
    for (const auto & h : all()) {
        if (h.on_load) {
            h.on_load(*self, ok);
        }
    }
    return ok;
}

extern "C" void __real__ZN14server_contextD1Ev(server_context * self);
SERVER_HOOK_WRAP void __wrap__ZN14server_contextD1Ev(server_context * self) {
    for (const auto & h : all()) {
        if (h.on_unload) {
            h.on_unload(*self);
        }
    }
    __real__ZN14server_contextD1Ev(self);
}
