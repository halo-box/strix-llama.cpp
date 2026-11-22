// hooks: tests/llama-context/test-case.h of the fork (3b76dee48, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include "test-util.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include <memory>
#include <string>
#include <vector>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

//
// the interface of a single context state case
//

enum class test_status {
    PASS,
    FAIL,
    SKIP,
};

// the verdict as printed in the result table, colored
const char * test_status_str(test_status status);

// how many columns the verdict takes in the table - the escapes of the color are not printed
size_t test_status_width(test_status status);

// a skipped case does not fail the run, a failed case always does
test_status merge_status(test_status a, test_status b);

enum test_group {
    TEST_GROUP_STATE, // KV state save / load / copy
    TEST_GROUP_RS,    // recurrent state, also run against a dirty cache
};

struct case_result {
    test_status status = test_status::PASS;
    std::string reason; // why the case was skipped or failed, empty when it passed

    static case_result pass();
    static case_result fail(const std::string & why = {});
    static case_result skip(const std::string & why);
};

struct ctx_test {
    virtual ~ctx_test() = default;

    // the column header of the --models table, keep it short
    virtual const char * name() const = 0;

    // one line of what the case verifies, printed by --list
    virtual const char * description() const { return ""; }

    virtual test_group group() const { return TEST_GROUP_STATE; }

    // the case compares against the logits produced by the baseline case
    virtual bool needs_baseline() const { return false; }

    // the case is also run against a cache pre-filled with a known pattern, so that a restore
    // reading cells it never wrote cannot pass
    virtual bool uses_fill() const { return group() == TEST_GROUP_RS; }

    // non-empty means the case does not apply to this model, the text is the reported reason
    virtual std::string skip_reason(model_run & mr) const;

    // the scenario, the checks are written with CTX_REQUIRE
    virtual case_result run(model_run & mr) const = 0;
};

// record an expectation that did not hold and give up on the case - the runner reports the
// reason, the body does not print anything
#define CTX_REQUIRE(cond, msg) \
    do { \
        if (!(cond)) { \
            return case_result::fail(msg); \
        } \
    } while (0)

using ctx_test_ptr = std::unique_ptr<const ctx_test>;

// the cases of one source file, in table order
std::vector<ctx_test_ptr> make_kv_cases();  // cases-kv.cpp
std::vector<ctx_test_ptr> make_rs_cases();  // cases-rs.cpp

// all cases, in table order
const std::vector<ctx_test_ptr> & test_cases();
} // namespace ggml_fork
