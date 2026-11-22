// The entry of llama-server with the tools switch (GGML_HOOKS=+tools, off, read once): the fork tool
// (hooks/apps/server, namespace ggml_fork) or the upstream one, renamed llama_server__upstream.

#include "ggml-hooks.h"

struct common_params;

int llama_server__upstream(int argc, char ** argv);
int llama_server__upstream(common_params & params, int argc, char ** argv);

namespace ggml_fork {
int llama_server(int argc, char ** argv);
}

static bool fork_tools_on() {
    static const bool on = ggml_hook_switch("tools", false);
    return on;
}

int llama_server(int argc, char ** argv) {
    return fork_tools_on() ? ggml_fork::llama_server(argc, argv) : llama_server__upstream(argc, argv);
}

// the parsed params are the upstream struct: the fork server parses the arguments again
int llama_server(common_params & params, int argc, char ** argv) {
    return fork_tools_on() ? ggml_fork::llama_server(argc, argv) : llama_server__upstream(params, argc, argv);
}
