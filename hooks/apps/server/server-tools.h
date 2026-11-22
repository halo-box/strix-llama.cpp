// hooks: tools/server/server-tools.h of the fork (403dfcb03, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include "server-common.h"
#include "server-http.h"
#include "server-queue.h"
#include "server-mcp.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include <atomic>
#include <functional>
#include <memory>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

struct server_tool {
    std::string name;
    std::string display_name;
    bool permission_write = false;
    bool support_stream = false; // if true, output can be streamed
    bool uses_cwd = false;       // if true, the tool resolves paths and runs against the working directory

    virtual ~server_tool() = default;
    virtual json get_definition() const = 0;
    virtual std::string type() const { return "server"; }

    struct stream {
        server_response & qr;
        int id;
        std::function<bool()> alive;
        void push(const std::string & chunk);
    };
    // should_stop: true once the caller stopped waiting, e.g. the client dropped the connection; a null should_stop means "never stop"
    virtual json invoke(json params, stream * st = nullptr, const std::function<bool()> & should_stop = nullptr) const = 0;

    json to_json() const;
};

struct server_tools_runtime; // impl detail, defined in server-tools.cpp

struct server_tools {
    std::vector<std::unique_ptr<server_tool>> tools;

    // for streaming
    server_response queue_res;
    std::atomic<int> res_id{0};

    // set when --tools-runtime is configured; routes every tool call through an isolate
    std::unique_ptr<server_tools_runtime> runtime;

    void setup(const std::vector<std::string> & enabled_tools,
               server_mcp & mcp_mgr,
               const std::string & tools_runtime);

    server_http_context::handler_t handle_get;
    server_http_context::handler_t handle_post;

    server_tools();
    ~server_tools();
};
} // namespace ggml_fork
