// Bridge from the upstream CUDA/HIP backend to the fork ggml-cuda code (hooks/ggml/fork-cuda, namespace ggml_fork).
// Each upstream backend gets a fork context that computes on the upstream stream, so fork and upstream kernels stay
// ordered, and a backend shim whose context is the fork context (the fork backend functions read backend->context).
//
// fork_cuda_node  (on):  the nodes that upstream refuses and the fork accepts (the fork types, the EXL3 ops, ...) are
//                        computed by the fork ggml_cuda_compute_forward.
// fork_cuda_graph (off): the fork computes whole graphs: its graph_optimize after the upstream one, then its
//                        graph_compute (fusions, kernel selection, CUDA graphs). GGML_HOOKS=+fork_cuda_graph.

#include "../../fork-cuda/common.cuh"

#include "ggml-backend-impl.h"
#include "ggml-hooks.h"

#include <cstring>
#include <mutex>
#include <unordered_map>

struct ggml_backend_cuda_context; // the upstream one (an incomplete type here)

// ggml-cuda-upstream.cu
bool         ggml_hook_cuda_upstream_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op);
cudaStream_t ggml_hook_cuda_ctx_stream(::ggml_backend_cuda_context & ctx);
int          ggml_hook_cuda_ctx_device(::ggml_backend_cuda_context & ctx);

namespace ggml_fork {
// fork-entry.inc
bool ggml_hook_fork_compute_forward(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_hook_fork_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op);
enum ggml_status ggml_hook_fork_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph);
void ggml_hook_fork_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params);
void ggml_hook_fork_set_mmb_enabled(ggml_backend_t backend, bool enable);
void ggml_hook_fork_set_mmb_small_batch(ggml_backend_t backend, bool enable);

static void bridge_sync(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream)); // CUDA_CHECK of the fork code
}
} // namespace ggml_fork

namespace {

struct fork_state {
    ggml_fork::ggml_backend_cuda_context * ctx = nullptr;
    ggml_backend                           shim = {}; // the upstream backend with the fork context
};

std::mutex fork_mutex;
std::unordered_map<const void *, fork_state *> fork_states; // by upstream backend context

fork_state & state_of(::ggml_backend_cuda_context & uctx, ggml_backend_t backend = nullptr) {
    std::lock_guard<std::mutex> lock(fork_mutex);
    fork_state *& s = fork_states[&uctx];
    if (!s) {
        s = new fork_state();
    }
    if (!s->ctx) {
        const int device = ggml_hook_cuda_ctx_device(uctx);
        s->ctx = new ggml_fork::ggml_backend_cuda_context(device);
        s->ctx->streams[device][0] = ggml_hook_cuda_ctx_stream(uctx); // owned by the upstream context
    }
    if (backend && !s->shim.context) {
        s->shim         = *backend;
        s->shim.context = s->ctx;
    }
    return *s;
}

void fork_backend_init(ggml_backend_t backend) {
    state_of(*(::ggml_backend_cuda_context *) backend->context, backend);
}

void fork_backend_free(ggml_backend_t backend) {
    std::lock_guard<std::mutex> lock(fork_mutex);
    auto it = fork_states.find(backend->context);
    if (it == fork_states.end()) {
        return;
    }
    fork_state * s = it->second;
    if (s->ctx) {
        ggml_fork::bridge_sync(s->ctx->streams[s->ctx->device][0]);
        s->ctx->streams[s->ctx->device][0] = nullptr; // the upstream context destroys it
        delete s->ctx;
    }
    delete s;
    fork_states.erase(it);
}

// ggml_backend_cuda_reg() takes a lock: the devices are read once
ggml_backend_dev_t cuda_dev(int device) {
    static ggml_backend_dev_t devs[GGML_CUDA_MAX_DEVICES] = {};
    if (!devs[device]) {
        devs[device] = ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device);
    }
    return devs[device];
}

// fork_cuda_node

bool node_supports(const ggml_hook_dev * dev, const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return false; // no compute
        default:
            break;
    }
    // the answer is asked for every node of every graph: it is kept per node, with what the two supports_op read
    uint64_t sig = ((uint64_t) t->op << 32) ^ (uint64_t) t->type ^ ((uint64_t) dev->device << 48);
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        sig = sig * 1099511628211ull ^ (uint64_t) t->ne[k];
    }
    for (int k = 0; k < 4; ++k) {
        sig = sig * 1099511628211ull ^ (uint64_t) (uint32_t) t->op_params[k];
    }
    for (int j = 0; j < GGML_MAX_SRC && t->src[j]; ++j) {
        const ggml_tensor * src = t->src[j];
        sig = sig * 1099511628211ull ^ (uintptr_t) src ^ ((uint64_t) src->type << 56) ^ (uintptr_t) src->buffer;
        for (int k = 0; k < GGML_MAX_DIMS; ++k) {
            sig = sig * 1099511628211ull ^ (uint64_t) src->ne[k] ^ ((uint64_t) src->nb[k] << 20);
        }
    }
    struct entry { const ggml_tensor * t; uint64_t sig; bool res; };
    static thread_local entry cache[4096];
    entry & e = cache[((uintptr_t) t >> 4) % 4096];
    if (e.t == t && e.sig == sig) {
        return e.res;
    }
    ggml_backend_dev_t d = cuda_dev(dev->device);
    e = { t, sig, !ggml_hook_cuda_upstream_supports_op(d, t) && ggml_fork::ggml_hook_fork_supports_op(d, t) };
    return e.res;
}

bool node_compute(::ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    return ggml_fork::ggml_hook_fork_compute_forward(*state_of(ctx).ctx, dst);
}

// fork_cuda_graph

bool graph_on() {
    static const bool on = ggml_hook_switch("fork_cuda_graph", false);
    return on;
}

// the whole graph from its first node
int graph_match(const ggml_hook_dev *, const ggml_cgraph * cgraph, int i) {
    for (int j = 0; j < i; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (!ggml_is_empty(n) && !ggml_op_is_empty(n->op) && (n->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            return 0; // a computed node before i: the graph was not taken at its start
        }
    }
    return cgraph->n_nodes - i >= 2 ? cgraph->n_nodes - i : 0;
}

void graph_compute(::ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int n) {
    fork_state & s = state_of(ctx);
    GGML_ASSERT(s.shim.context && "fork_cuda_graph: the backend was not initialized through the hooks");
    ggml_cgraph v = *cgraph;
    v.nodes   = cgraph->nodes + i;
    v.n_nodes = n;
    if (i != 0 || n != cgraph->n_nodes) {
        v.uid = 0;
    }
    GGML_ASSERT(ggml_fork::ggml_hook_fork_graph_compute(&s.shim, &v) == GGML_STATUS_SUCCESS);
}

void graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    if (!graph_on()) {
        return;
    }
    fork_state & s = state_of(*(::ggml_backend_cuda_context *) backend->context, backend);
    ggml_fork::ggml_hook_fork_graph_optimize(&s.shim, cgraph, params);
}

// the registry procs of the fork that libllama asks for: on the fork context of the upstream backend
void set_mmb_enabled(ggml_backend_t backend, bool enable) {
    ggml_fork::ggml_hook_fork_set_mmb_enabled(&state_of(*(::ggml_backend_cuda_context *) backend->context, backend).shim, enable);
}

void set_mmb_small_batch(ggml_backend_t backend, bool enable) {
    ggml_fork::ggml_hook_fork_set_mmb_small_batch(&state_of(*(::ggml_backend_cuda_context *) backend->context, backend).shim, enable);
}

void * proc_address(ggml_backend_reg_t, const char * name) {
    if (strcmp(name, "ggml_backend_cuda_set_mmb_enabled") == 0) {
        return (void *) set_mmb_enabled;
    }
    if (strcmp(name, "ggml_backend_cuda_set_mmb_small_batch") == 0) {
        return (void *) set_mmb_small_batch;
    }
    return nullptr;
}

} // namespace

// every op, after the kernels of the op
GGML_HOOK_REGISTER_KERNEL(ggml_hook_op_kernel("fork_cuda_node", GGML_HOOK_BACKEND_CUDA, GGML_OP_COUNT, -1000, node_supports,
                                              ggml_hook_fn(static_cast<ggml_hook_cuda_op_fn>(node_compute))));

GGML_HOOK_REGISTER_KERNEL([] {
    auto k = ggml_hook_fusion_kernel("fork_cuda_graph", GGML_HOOK_BACKEND_CUDA, GGML_OP_COUNT, 1000, graph_match,
                                     ggml_hook_fn(static_cast<ggml_hook_cuda_fusion_fn>(graph_compute)));
    k.enabled_by_default = false;
    return k;
}());

GGML_HOOK_REGISTER_BACKEND_EXT([] {
    auto e = ggml_hook_ext("fork_cuda", GGML_HOOK_BACKEND_CUDA);
    e.backend_init   = fork_backend_init;
    e.backend_free   = fork_backend_free;
    e.graph_optimize = graph_optimize;
    e.proc_address   = proc_address;
    return e;
}());
