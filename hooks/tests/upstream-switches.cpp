// Compiled into the test programs that read upstream internals (hooks/llama/llama-hooks.cmake): they check upstream
// structures or exact upstream numbers, so they run the upstream code. HOOKS_UPSTREAM_SWITCHES: the switches to turn off.

#include "ggml-hooks.h"

[[maybe_unused]] static const int upstream_switches = [] {
    for (const char * name : { HOOKS_UPSTREAM_SWITCHES }) {
        ggml_hook_kernel_set_enabled(name, false);
    }
    return 0;
}();
