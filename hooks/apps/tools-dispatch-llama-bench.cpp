// The entry of llama-bench with the tools switch (on, GGML_HOOKS=-tools: upstream, read once): the fork tool
// (hooks/apps/llama-bench, namespace ggml_fork) or the upstream one, renamed llama_bench__upstream.

#include "ggml-hooks.h"

struct common_params;

int llama_bench__upstream(int argc, char ** argv);

namespace ggml_fork {
int llama_bench(int argc, char ** argv);
}

static bool fork_tools_on() {
    static const bool on = ggml_hook_switch("tools", true);
    return on;
}

int llama_bench(int argc, char ** argv) {
    return fork_tools_on() ? ggml_fork::llama_bench(argc, argv) : llama_bench__upstream(argc, argv);
}
