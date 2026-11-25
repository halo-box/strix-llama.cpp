// hooks: tools/server/server-connect.h of the fork (403dfcb03, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include "common.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include <atomic>
#include <memory>
#include <string>
#include <thread>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

struct common_subproc;

// spawns llama-connect, which exposes this server to a remote browser over WebRTC
// core binary is in a separate project: https://github.com/ggml-org/llama-connect
struct server_connect {
    server_connect();
    ~server_connect();

    server_connect(const server_connect &) = delete;
    server_connect & operator=(const server_connect &) = delete;

    // path of the llama-connect binary, empty if not found
    static std::string find_binary();

    // why --connect cannot work here, empty if it can
    static std::string unavailable_reason(const common_params & params);

    static std::string resolve_code(const std::string & code);

    // port is the bound one, it differs from params.port with --port 0 or the GCP override
    bool start(const common_params & params, int port);

    // idempotent, also called by the destructor
    void stop();

private:
    std::unique_ptr<common_subproc> proc;
    std::thread log_thread;
    std::atomic<bool> stopping{false}; // tells the log thread the exit is expected
};
} // namespace ggml_fork
