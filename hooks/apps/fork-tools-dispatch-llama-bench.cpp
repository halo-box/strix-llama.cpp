// The entry of llama-bench with the fork_tools switch (GGML_HOOKS=+fork_tools, off, read once): the fork tool
// (hooks/apps/fork-llama-bench, namespace ggml_fork) or the upstream one, renamed llama_bench__upstream.

#include "ggml-hooks.h"

struct common_params;

int llama_bench__upstream(int argc, char ** argv);

namespace ggml_fork {
int llama_bench(int argc, char ** argv);
}

static bool fork_tools_on() {
    static const bool on = ggml_hook_switch("fork_tools", false);
    return on;
}

int llama_bench(int argc, char ** argv) {
    return fork_tools_on() ? ggml_fork::llama_bench(argc, argv) : llama_bench__upstream(argc, argv);
}
