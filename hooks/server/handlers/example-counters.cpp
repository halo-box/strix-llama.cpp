// example server hook: off unless LLAMA_SERVER_HOOKS_EXAMPLE=1.
// - reads the request field "hooks_example_tag" and keeps it with the task
// - counts every callback, adds "hooks_example": true to the usage of the results
// - GET /hooks/example returns the counters and the tag of the last task. It is public (no API key needed).

#include "server-hooks.h"

#include "common-hooks.h"

#include "server-http.h"

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>

namespace {

std::atomic<long> n_requests{ 0 };
std::atomic<long> n_slot_starts{ 0 };
std::atomic<long> n_tokens{ 0 };
std::atomic<long> n_releases{ 0 };
std::atomic<long> n_load_params{ 0 };
std::atomic<long> n_loads{ 0 };
std::atomic<long> n_task_launches{ 0 };
std::atomic<long> n_updates{ 0 };
std::atomic<long> n_ctx_max_calls{ 0 };
std::atomic<long> n_spec_draft{ 0 };
std::atomic<long> n_spec_accept{ 0 };
std::mutex        tag_mutex;
std::string       last_tag;

void on_request(const common_json & body, common_json & hooks_data) {
    n_requests++;
    if (body.contains("hooks_example_tag")) {
        hooks_data["example_tag"] = body.at("hooks_example_tag").get<std::string>();
    }
}

void on_slot_start(int, int, llama_context *, const common_json & hooks_data) {
    n_slot_starts++;
    if (hooks_data.contains("example_tag")) {
        std::lock_guard<std::mutex> lock(tag_mutex);
        last_tag = hooks_data.at("example_tag").get<std::string>();
    }
}

bool on_token(int, llama_token, const std::string &) {
    n_tokens++;
    return true;
}

void on_slot_release(int) {
    n_releases++;
}

void register_routes(server_http_context & http) {
    http.get("/hooks/example", [](const server_http_req &) {
        common_json res = common_json::object();
        res["requests"]    = (int64_t) n_requests.load();
        res["slot_starts"] = (int64_t) n_slot_starts.load();
        res["tokens"]      = (int64_t) n_tokens.load();
        res["releases"]    = (int64_t) n_releases.load();
        res["load_params"] = (int64_t) n_load_params.load();
        res["loads"]       = (int64_t) n_loads.load();
        res["task_launch"] = (int64_t) n_task_launches.load();
        res["updates"]     = (int64_t) n_updates.load();
        res["n_ctx_max"]   = (int64_t) n_ctx_max_calls.load();
        res["spec_draft"]  = (int64_t) n_spec_draft.load();
        res["spec_accept"] = (int64_t) n_spec_accept.load();
        {
            std::lock_guard<std::mutex> lock(tag_mutex);
            res["last_tag"] = last_tag;
        }
        auto out  = std::make_unique<server_http_res>();
        out->data = res.dump();
        return out;
    });
}

bool example_enabled() {
    const char * env = getenv("LLAMA_SERVER_HOOKS_EXAMPLE");
    return env && std::string(env) == "1";
}

} // namespace

static const bool example_registered = [] {
    if (example_enabled()) {
        server_hooks h;
        h.name            = "example_counters";
        h.on_request      = on_request;
        h.on_slot_start   = on_slot_start;
        h.on_token        = on_token;
        h.on_slot_release = on_slot_release;
        h.register_routes = register_routes;
        h.on_load_params  = [](common_params &) { n_load_params++; };
        h.on_load         = [](server_context &, bool ok) { n_loads += ok; };
        h.on_task_launch  = [](int, llama_context *, server_task &) { n_task_launches++; };
        h.n_ctx_max       = [](const llama_model *, const common_params &) { n_ctx_max_calls++; return 0; };
        h.on_update       = [](const std::function<common_json()> & slots) { n_updates += slots().is_array(); };
        h.on_usage        = [](const server_task_result_cmpl_final &, common_json & usage) { usage["hooks_example"] = true; };
        h.public_paths    = [](std::unordered_set<std::string> & paths) { paths.insert("/hooks/example"); };
        server_hook_register(h);

        common_hooks c;
        c.name        = "example_counters_spec";
        c.spec_draft  = [](common_speculative *) { n_spec_draft++; };
        c.spec_accept = [](common_speculative *, llama_seq_id, uint16_t) { n_spec_accept++; };
        common_hooks_register(c);
    }
    return true;
}();
