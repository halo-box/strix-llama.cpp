// The entry of llama-quantize with the tools switch (on, GGML_HOOKS=-tools: upstream, read once): the fork tool
// (hooks/apps/quantize, namespace ggml_fork) or the upstream one, renamed llama_quantize__upstream.

#include "ggml-hooks.h"

struct common_params;

int llama_quantize__upstream(int argc, char ** argv);

namespace ggml_fork {
int llama_quantize(int argc, char ** argv);
}

static bool fork_tools_on() {
    static const bool on = ggml_hook_switch("tools", true);
    return on;
}

int llama_quantize(int argc, char ** argv) {
    return fork_tools_on() ? ggml_fork::llama_quantize(argc, argv) : llama_quantize__upstream(argc, argv);
}
