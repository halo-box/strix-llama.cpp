// hooks: common/debug.h of the fork (25b1c175e, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include <memory>
#include <string>
#include <vector>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

// common debug functions and structs

struct common_params;

// Intended to use as callback for ggml_backend_sched_eval_callback
// prints tensors that are processed in the computation graph
// by default prints all tensors, but can be configured by creating a `common_debug_cb_user_data` instance with
// non-empty filter_patterns. See examples/debug.cpp for possible usage patterns
// `common_debug_cb_user_data` contains `abort_on_nan` flag that determines whether an error should be thrown whenever a NaN is encountered
// in a tensor (useful for stopping debug sessions on first erroneous tensor)
// The callback data will be passed as the third parameter (user_data)
bool common_debug_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);

struct common_debug_cb_user_data {
    struct impl;
    std::unique_ptr<impl> pimpl;

    common_debug_cb_user_data();
    ~common_debug_cb_user_data();

    common_debug_cb_user_data(const common_debug_cb_user_data &) = delete;
    common_debug_cb_user_data & operator=(const common_debug_cb_user_data &) = delete;

    common_debug_cb_user_data(common_params & params, const std::vector<std::string> & filter_patterns, bool abort_on_nan = false);
};
} // namespace ggml_fork
