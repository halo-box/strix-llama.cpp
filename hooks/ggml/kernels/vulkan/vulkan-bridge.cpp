// Bridge from the upstream Vulkan backend to the fork ggml-vulkan code (hooks/ggml/vulkan, namespace ggml_fork).
// The fork device is built on the upstream instance and logical device (ggml-vulkan-upstream.cpp), so the fork kernels
// can read the upstream buffers and submit to the same queues.
//
// vk_node  (on):  the nodes that upstream refuses and the fork accepts (the fork types, ...) run through the fork
//                      graph_compute, one node at a time, on a fork backend of the same device.
// vk_graph (off): the upstream devices become the fork devices (GGML_HOOKS=+vk_graph, a switch). The fork
//                      device functions take the upstream device (the device contexts have the same layout), and the
//                      buffer types, backends and events they create point at the upstream device. Every buffer, backend
//                      and event is then a fork one, as in a fork build: the fork code never reads an upstream buffer,
//                      whose vk_device_struct has the upstream layout. Hook kernels do not apply in this mode.

#include "../../vulkan/ggml-vulkan-common.h"

#include "ggml-backend-impl.h"
#include "ggml-hooks.h"

#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

// ggml-vulkan-upstream.cpp
bool               ggml_hook_vk_upstream_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op);
size_t             ggml_hook_vk_backend_index(ggml_backend_t backend);
bool               ggml_hook_vk_upstream_buffer_is_vk(ggml_backend_buffer_t buffer);
ggml_backend_dev_t ggml_hook_vk_upstream_dev(size_t device);

namespace {

bool graph_on() {
    static const bool on = ggml_hook_switch("vk_graph", false);
    return on;
}

ggml_backend_dev_t fork_dev(size_t device) {
    static ggml_backend_dev_t devs[64] = {};
    if (!devs[device]) {
        devs[device] = ggml_backend_reg_dev_get(ggml_fork::ggml_fork_backend_vk_reg(), device);
    }
    return devs[device];
}

ggml_backend_dev_t upstream_dev(size_t device) {
    static ggml_backend_dev_t devs[64] = {};
    if (!devs[device]) {
        devs[device] = ggml_hook_vk_upstream_dev(device);
    }
    return devs[device];
}

size_t dev_index(ggml_backend_dev_t dev) {
    return ((const ggml_fork::ggml_backend_vk_device_context *) dev->context)->device;
}

// ---------------------------------------------------------------------------------------------------- vk_graph

// an object the fork created for a fork device: the upstream device of the same index instead
ggml_backend_dev_t to_upstream(ggml_backend_dev_t dev) {
    ggml_backend_reg_t freg = ggml_fork::ggml_fork_backend_vk_reg();
    for (size_t i = 0; i < ggml_backend_reg_dev_count(freg); ++i) {
        if (dev == fork_dev(i)) {
            return upstream_dev(i);
        }
    }
    return dev;
}

ggml_backend_buffer_type_t own_buft(ggml_backend_buffer_type_t buft) {
    if (buft) {
        buft->device = to_upstream(buft->device);
    }
    return buft;
}

ggml_backend_t graph_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_t b = fork_dev(dev_index(dev))->iface.init_backend(dev, params);
    if (b) {
        b->device = dev;
    }
    return b;
}

ggml_backend_buffer_type_t graph_get_buffer_type(ggml_backend_dev_t dev) {
    return own_buft(fork_dev(dev_index(dev))->iface.get_buffer_type(dev));
}

ggml_backend_buffer_type_t graph_get_host_buffer_type(ggml_backend_dev_t dev) {
    auto f = fork_dev(dev_index(dev))->iface.get_host_buffer_type;
    return f ? own_buft(f(dev)) : nullptr;
}

ggml_backend_buffer_t graph_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_backend_buffer_t buf = fork_dev(dev_index(dev))->iface.buffer_from_host_ptr(dev, ptr, size, max_tensor_size);
    if (buf) {
        own_buft(buf->buft);
    }
    return buf;
}

ggml_backend_event_t graph_event_new(ggml_backend_dev_t dev) {
    ggml_backend_event_t ev = fork_dev(dev_index(dev))->iface.event_new(dev);
    if (ev) {
        ev->device = dev;
    }
    return ev;
}

void graph_device_install(ggml_backend_dev_t dev, ggml_backend_device_i * iface) {
    if (!graph_on()) {
        return;
    }
    ggml_backend_dev_t f = fork_dev(dev_index(dev));
    *iface = f->iface; // the fork device functions, on the upstream device (same device context layout)
    iface->init_backend = graph_init_backend;
    iface->get_buffer_type = graph_get_buffer_type;
    if (iface->get_host_buffer_type) {
        iface->get_host_buffer_type = graph_get_host_buffer_type;
    }
    if (iface->buffer_from_host_ptr) {
        iface->buffer_from_host_ptr = graph_buffer_from_host_ptr;
    }
    if (iface->event_new) {
        iface->event_new = graph_event_new;
    }
}

void * graph_proc_address(ggml_backend_reg_t, const char * name) {
    if (!graph_on()) {
        return nullptr;
    }
    ggml_backend_reg_t freg = ggml_fork::ggml_fork_backend_vk_reg();
    return freg->iface.get_proc_address ? freg->iface.get_proc_address(freg, name) : nullptr;
}

// ---------------------------------------------------------------------------------------------------- vk_node

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

// the synchronize of the upstream backend also waits for its fork backend
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
    fctx->submit_pending = true;
}

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

bool node_offload(ggml_backend_dev_t dev, const ggml_tensor * op, bool * result) {
    const size_t i = dev_index(dev);
    if (ggml_hook_vk_upstream_supports_op(upstream_dev(i), op)) {
        return false;
    }
    ggml_backend_dev_t f = fork_dev(i);
    *result = f->iface.offload_op ? f->iface.offload_op(f, op) : false;
    return true;
}

ggml_backend_t fork_of(::ggml_backend_vk_context * ctx) {
    std::lock_guard<std::mutex> lock(fork_mutex);
    return fork_states.at(ctx)->fork;
}

// The upstream buffers that a fork node reads or writes: while the node is recorded, their vk_buffer_struct points at
// the fork device of the same index (the same VkDevice). The fork code reaches the device through the buffer
// (buffer->device->uma, ->mutex, ->transfer_queue, ...), and an upstream vk_device_struct has another layout. The
// buffer structs, contexts and device contexts have the same layout in both trees.
struct device_swap {
    std::vector<std::pair<ggml_fork::vk_device *, ggml_fork::vk_device>> saved;

    void add(const ggml_tensor * t, const ggml_fork::vk_device & fdev) {
        while (t && t->view_src) {
            t = t->view_src;
        }
        if (!t || !t->buffer || !ggml_hook_vk_upstream_buffer_is_vk(t->buffer)) {
            return;
        }
        auto * bc = (ggml_fork::ggml_backend_vk_buffer_context *) t->buffer->context;
        ggml_fork::vk_device * d = &bc->dev_buffer->device;
        for (auto & s : saved) {
            if (s.first == d) {
                return;
            }
        }
        saved.emplace_back(d, std::move(*d));
        *d = fdev;
    }

    ~device_swap() {
        for (auto it = saved.rbegin(); it != saved.rend(); ++it) {
            *it->first = std::move(it->second);
        }
    }
};

bool node_compute(::ggml_backend_vk_context * ctx, ::vk_context &, ggml_cgraph * cgraph, int i) {
    ggml_backend_t fork = fork_of(ctx);
    const ggml_fork::vk_device & fdev = ((ggml_fork::ggml_backend_vk_context *) fork->context)->device;
    device_swap swap;
    ggml_tensor * node = cgraph->nodes[i];
    swap.add(node, fdev);
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        swap.add(node->src[j], fdev);
    }
    fork_barrier(fork);
    ggml_cgraph v = *cgraph;
    v.nodes   = cgraph->nodes + i;
    v.n_nodes = 1;
    v.n_leafs = 0;
    v.leafs   = nullptr;
    v.uid     = 0;
    GGML_ASSERT(fork->iface.graph_compute(fork, &v) == GGML_STATUS_SUCCESS);
    // the recorded commands hold the VkBuffer handles only: the buffers get their upstream device back
    return true;
}

} // namespace

// the backends created without the registry (ggml-hooks-vk.cpp): with vk_graph, fork backends
ggml_backend_t ggml_hook_vk_replace_init(size_t dev_num) {
    if (!graph_on()) {
        return nullptr;
    }
    ggml_backend_dev_t dev = upstream_dev(dev_num);
    return dev->iface.init_backend(dev, nullptr);
}

// every op, after the kernels of the op
GGML_HOOK_REGISTER_KERNEL(ggml_hook_op_kernel("vk_node", GGML_HOOK_BACKEND_VULKAN, GGML_OP_COUNT, -1000, node_supports,
                                              ggml_hook_fn(static_cast<ggml_hook_vk_op_fn>(node_compute))));

GGML_HOOK_REGISTER_BACKEND_EXT([] {
    auto e = ggml_hook_ext("vk_bridge", GGML_HOOK_BACKEND_VULKAN);
    e.backend_init   = fork_backend_init;
    e.backend_free   = fork_backend_free;
    e.device_install = graph_device_install;
    e.offload_op     = node_offload;
    e.proc_address   = graph_proc_address;
    return e;
}());
