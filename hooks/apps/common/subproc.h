// hooks: common/subproc.h of the fork (403dfcb03, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include <atomic>
#include <cstdio>
#include <string>
#include <vector>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

#ifdef LLAMA_SUBPROCESS
} // namespace ggml_fork
#include <sheredom/subprocess.h>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md
#else
// dummy values to allow compilation when subprocess is disabled
struct subprocess_s {};
static constexpr int subprocess_option_no_window = 0;
static constexpr int subprocess_option_combined_stdout_stderr = 0;
static constexpr int subprocess_option_inherit_environment = 0;
static constexpr int subprocess_option_search_user_path = 0;
#endif

// RAII-style wrapper around https://github.com/sheredom/subprocess.h,
// exposing method calls instead of free functions operating on subprocess_s.
struct common_subproc {
    common_subproc() = default;
    ~common_subproc();

    common_subproc(const common_subproc &) = delete;
    common_subproc & operator=(const common_subproc &) = delete;

    // spawn a child process; if env is non-empty it replaces the child's environment
    // (do not combine with subprocess_option_inherit_environment)
    bool create(
            const std::vector<std::string> & args,
            int options,
            const std::vector<std::string> & env = {},
            const char * cwd = nullptr);

    bool alive();

    // true if LLAMA_SUBPROCESS was enabled at build time; when false, create() always fails
    static bool is_supported();

    FILE * stdin_file();
    FILE * stdout_file();
    FILE * stderr_file();

    // close stdin and detach it from the process, so a later join()/destroy() won't double-close it;
    // use this after writing all input to signal EOF to the child while it's still running
    void close_stdin();

    void terminate();

    // wait for the process to exit, release the underlying handle and return its exit code
    int join();

private:
    subprocess_s proc {};
    std::atomic<bool> is_created{false};

    bool has_handle() const;
};
} // namespace ggml_fork
