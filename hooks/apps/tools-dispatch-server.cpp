// The entry of llama-server with the tools switch (on, GGML_HOOKS=-tools: upstream, read once): the fork tool
// (hooks/apps/server, namespace ggml_fork) or the upstream one, renamed llama_server__upstream.

#include "ggml-hooks.h"

struct common_params;

int llama_server__upstream(int argc, char ** argv);
int llama_server__upstream(common_params & params, int argc, char ** argv);

namespace ggml_fork {
int llama_server(int argc, char ** argv);
}

static bool fork_tools_on() {
    static const bool on = ggml_hook_switch("tools", true);
    return on;
}

int llama_server(int argc, char ** argv) {
    return fork_tools_on() ? ggml_fork::llama_server(argc, argv) : llama_server__upstream(argc, argv);
}

// the server inside llama-cli: its params are the upstream struct and it has no command line to parse again, so it is
// the upstream server (on the fork libllama and backends), which llama_server_terminate stops
int llama_server(common_params & params, int argc, char ** argv) {
    return llama_server__upstream(params, argc, argv);
}
