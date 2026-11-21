// Bridge from the upstream Vulkan backend to the fork ggml-vulkan code (hooks/ggml/fork-vulkan, namespace ggml_fork).
// The fork device is built on the upstream instance and logical device (ggml-vulkan-upstream.cpp), so the fork kernels
// read the upstream buffers and submit to the same queues. Each upstream backend gets a fork backend on its device.
//
// fork_vk_node  (on):  the nodes that upstream refuses and the fork accepts (the fork types, ...) run through the fork
//                      graph_compute, one node at a time, after a barrier.
// fork_vk_graph (off): the fork computes the graphs instead of upstream: its graph_optimize and graph_compute.
//                      GGML_HOOKS=+fork_vk_graph (a switch).
// The synchronize of the upstream backend also waits for the fork backend.

#include "../../fork-vulkan/ggml-vulkan-common.h"

#include "ggml-backend-impl.h"
#include "ggml-hooks.h"

#include <mutex>
#include <unordered_map>

// ggml-vulkan-upstream.cpp
bool               ggml_hook_vk_upstream_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op);
size_t             ggml_hook_vk_backend_index(ggml_backend_t backend);
ggml_backend_dev_t ggml_hook_vk_upstream_dev(size_t device);
// the fork headers rename the ggml-vulkan.h functions: upstream ones come from ggml-vulkan-upstream.cpp

namespace {

struct fork_state {
    ggml_backend_t fork        = nullptr; // a fork backend on the same device
    void (*synchronize)(ggml_backend_t) = nullptr; // of the upstream backend
};

std::mutex fork_mutex;
std::unordered_map<const void *, fork_state *> fork_states; // by upstream backend context

fork_state * state_of(ggml_backend_t backend) {
    std::lock_guard<std::mutex> lock(fork_mutex);
    auto it = fork_states.find(backend->context);
    return it == fork_states.end() ? nullptr : it->second;
}

void wrap_synchronize(ggml_backend_t backend) {
    fork_state * s = state_of(backend);
    if (s->fork) {
        s->fork->iface.synchronize(s->fork);
    }
    s->synchronize(backend);
}

void fork_backend_init(ggml_backend_t backend) {
    fork_state * s = new fork_state();
    s->fork        = ggml_fork::ggml_fork_backend_vk_init(ggml_hook_vk_backend_index(backend));
    s->synchronize = backend->iface.synchronize;
    backend->iface.synchronize = wrap_synchronize;
    std::lock_guard<std::mutex> lock(fork_mutex);
    fork_states[backend->context] = s;
}

void fork_backend_free(ggml_backend_t backend) {
    fork_state * s = state_of(backend);
    if (!s) {
        return;
    }
    if (s->fork) {
        s->fork->iface.synchronize(s->fork);
        ggml_backend_free(s->fork);
    }
    std::lock_guard<std::mutex> lock(fork_mutex);
    fork_states.erase(backend->context);
    delete s;
}

// a barrier on the queue: the fork work after it sees the upstream work before it
void fork_barrier(ggml_backend_t fork) {
    auto * fctx = (ggml_fork::ggml_backend_vk_context *) fork->context;
    ggml_fork::vk_context sub = ggml_fork::ggml_vk_get_compute_ctx(fctx);
    ggml_fork::ggml_vk_sync_buffers(fctx, sub);
    ggml_fork::ggml_vk_ctx_end(sub);
    fctx->compute_ctx.reset();
    ggml_fork::ggml_vk_submit(sub, {});
}

ggml_backend_dev_t fork_dev(int device) {
    static ggml_backend_dev_t devs[64] = {};
    if (!devs[device]) {
        devs[device] = ggml_backend_reg_dev_get(ggml_fork::ggml_fork_backend_vk_reg(), device);
    }
    return devs[device];
}

ggml_backend_dev_t upstream_dev(int device) {
    static ggml_backend_dev_t devs[64] = {};
    if (!devs[device]) {
        devs[device] = ggml_hook_vk_upstream_dev(device);
    }
    return devs[device];
}

// fork_vk_node

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
    e = { t, sig, !ggml_hook_vk_upstream_supports_op(upstream_dev(dev->device), t) &&
                  fork_dev(dev->device)->iface.supports_op(fork_dev(dev->device), t) };
    return e.res;
}

// the upstream backend of a context: the hook kernels get the upstream context
ggml_backend_t fork_of(::ggml_backend_vk_context * ctx) {
    std::lock_guard<std::mutex> lock(fork_mutex);
    return fork_states.at(ctx)->fork;
}

ggml_cgraph view(ggml_cgraph * g, int i, int n) {
    ggml_cgraph v = *g;
    v.nodes   = g->nodes + i;
    v.n_nodes = n;
    v.n_leafs = 0;
    v.leafs   = nullptr;
    if (i != 0 || n != g->n_nodes) {
        v.uid = 0;
    }
    return v;
}

bool node_compute(::ggml_backend_vk_context * ctx, ::vk_context &, ggml_cgraph * cgraph, int i) {
    ggml_backend_t fork = fork_of(ctx);
    fork_barrier(fork);
    ggml_cgraph v = view(cgraph, i, 1);
    GGML_ASSERT(fork->iface.graph_compute(fork, &v) == GGML_STATUS_SUCCESS);
    return true;
}

// fork_vk_graph

bool graph_on() {
    static const bool on = ggml_hook_switch("fork_vk_graph", false);
    return on;
}

bool graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_status * status) {
    if (!graph_on()) {
        return false;
    }
    ggml_backend_t fork = state_of(backend)->fork;
    fork_barrier(fork); // after the upstream transfers
    *status = fork->iface.graph_compute(fork, cgraph);
    return true;
}

bool graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    if (!graph_on()) {
        return false;
    }
    fork_state * s = state_of(backend);
    if (s->fork->iface.graph_optimize) {
        s->fork->iface.graph_optimize(s->fork, cgraph, params);
    }
    return true;
}

} // namespace

// every op, after the kernels of the op
GGML_HOOK_REGISTER_KERNEL(ggml_hook_op_kernel("fork_vk_node", GGML_HOOK_BACKEND_VULKAN, GGML_OP_COUNT, -1000, node_supports,
                                              ggml_hook_fn(static_cast<ggml_hook_vk_op_fn>(node_compute))));

GGML_HOOK_REGISTER_BACKEND_EXT([] {
    auto e = ggml_hook_ext("fork_vk", GGML_HOOK_BACKEND_VULKAN);
    e.backend_init   = fork_backend_init;
    e.backend_free   = fork_backend_free;
    e.replace_graph_compute  = graph_compute;
    e.replace_graph_optimize = graph_optimize;
    return e;
}());
