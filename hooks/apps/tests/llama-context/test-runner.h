// hooks: tests/llama-context/test-runner.h of the fork (3b76dee48, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include "test-case.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include <string>
#include <vector>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

//
// the runner: the single place that decides how a case is executed
//

struct case_run {
    std::string name;
    test_status status = test_status::SKIP;
    std::string reason;
};

// the cases selected by the filter, all of them when it is empty, in table order
std::vector<const ctx_test *> select_cases(const std::string & filter);

// run the selected cases for one model and return one verdict per case - with report, print the
// verdict of every case as it finishes, for single model mode
std::vector<case_run> run_cases(model_run & mr, const std::vector<const ctx_test *> & cases, bool report);
} // namespace ggml_fork
