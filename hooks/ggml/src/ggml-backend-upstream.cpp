// The upstream ggml/src/ggml-backend.cpp, compiled through this file (ggml-hooks.cmake marks the original as a header):
// the upstream scheduler is private to it, and the fork_sched dispatcher (ggml-fork-sched.cpp) needs one of its fields.
// The upstream public functions that the dispatcher replaces are renamed *__upstream (hooks_rename).

#include "../../../ggml/src/ggml-backend.cpp"

// ggml_backend_sched_get_n_graph_nodes of the fork, for the upstream scheduler
extern "C" int ggml_hook_sched_n_graph_nodes__upstream(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->graph.n_nodes;
}
