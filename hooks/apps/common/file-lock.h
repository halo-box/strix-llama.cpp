// hooks: common/file-lock.h of the fork (25b1c175e, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include <functional>
#include <string>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

// Cross-process exclusive lock (flock on POSIX, LockFileEx on Windows).
// Lazy: acquire() opens the lock file, so a read-only cache works if the lock is never acquired.
class common_file_lock {
public:
    common_file_lock(const std::string & lock_path);
    ~common_file_lock();

    // take the lock, waiting while another process holds it;
    // keep_waiting is polled before each retry, false cancels the wait.
    // Also returns false when locking is unavailable.
    bool acquire(const std::function<bool()> & keep_waiting = {});

    void close();

    common_file_lock(const common_file_lock &) = delete;
    common_file_lock & operator=(const common_file_lock &) = delete;

private:
    enum class status {
        acquired,
        busy,  // another process holds the lock
        error, // file locking is unavailable: no exclusion is possible
    };

    status try_lock();

    std::string lock_path;

    int  fd     = -1;
    bool locked = false;
};
} // namespace ggml_fork
