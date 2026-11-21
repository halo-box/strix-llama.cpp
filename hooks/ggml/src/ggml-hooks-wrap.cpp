// Run-time wrapping of the upstream backends: device, backend and registry interfaces. See ggml-hooks-impl.h.

#include "ggml-hooks-impl.h"

#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct ext_entry {
    ggml_hook_backend_ext         e;
    std::string                   name;
    const void *                  module;
    const ggml_hook_backend_ext * desc;
};

// immutable once published, old lists are leaked on purpose (a backend thread may still read them)
struct ext_list {
    std::vector<const ggml_hook_backend_ext *> by_backend[GGML_HOOK_BACKEND_COUNT];
};

struct driver_entry {
    ggml_hook_driver         d;
    const void *             module;
    const ggml_hook_driver * desc;
    bool                     removed = false;
};

struct dev_state {
    ggml_backend_dev_t                      dev;
    ggml_backend_device_i                   orig;
    ggml_hook_dev                           info;
    std::string                             name;
    std::once_flag                          extra_once;
    std::vector<ggml_backend_buffer_type_t> extra;      // NULL terminated, returned by the extra_bufts proc
    std::vector<ggml_backend_buffer_type_t> ext_bufts;  // from the extensions, supported by this device
};

struct reg_state {
    ggml_backend_reg_t      reg;
    ggml_hook_driver        drv;
    ggml_backend_reg_i      orig;
    std::deque<dev_state>   devs;
    // upstream backend interface, the same for every backend of a registry. Set by the first init_backend.
    std::once_flag          backend_once;
    ggml_backend_i          orig_backend;
};

struct wrap_state {
    std::mutex                   mutex;
    std::deque<ext_entry>        exts;
    std::atomic<ext_list *>      ext_snapshot{ nullptr };
    std::deque<driver_entry>     drivers;
    std::deque<reg_state>        regs;
    std::atomic<int>             n_regs{ 0 };
    reg_state *                  reg_ptrs[32] = {};
};

wrap_state & state() {
    static auto * s = new wrap_state();
    return *s;
}

const std::vector<const ggml_hook_backend_ext *> & exts_of(ggml_hook_backend backend) {
    static const std::vector<const ggml_hook_backend_ext *> none;
    const ext_list * l = state().ext_snapshot.load(std::memory_order_acquire);
    return l ? l->by_backend[backend] : none;
}

// call with the mutex held
void publish_exts(wrap_state & s) {
    auto * l = new ext_list();
    for (const auto & x : s.exts) {
        l->by_backend[x.e.backend].push_back(&x.e);
    }
    s.ext_snapshot.store(l, std::memory_order_release);
}

reg_state * find_reg(ggml_backend_reg_t reg) {
    auto & s = state();
    const int n = s.n_regs.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i) {
        if (s.reg_ptrs[i]->reg == reg) {
            return s.reg_ptrs[i];
        }
    }
    return nullptr;
}

dev_state * find_dev(ggml_backend_dev_t dev, reg_state ** rs_out = nullptr) {
    reg_state * rs = dev ? find_reg(dev->reg) : nullptr;
    if (!rs) {
        return nullptr;
    }
    if (rs_out) {
        *rs_out = rs;
    }
    for (auto & d : rs->devs) {
        if (d.dev == dev) {
            return &d;
        }
    }
    return nullptr;
}

bool is_private_buft(const reg_state & rs, ggml_backend_buffer_type_t buft) {
    if (!rs.drv.private_bufts) {
        return false;
    }
    for (ggml_backend_buffer_type_t * p = rs.drv.private_bufts(); p && *p; ++p) {
        if (*p == buft) {
            return true;
        }
    }
    return false;
}

// device

bool uses_hook_type(const ggml_tensor * op) {
    if (ggml_hook_type_is_hook(op->type)) {
        return true;
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (op->src[i] && ggml_hook_type_is_hook(op->src[i]->type)) {
            return true;
        }
    }
    return false;
}

bool wrap_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    reg_state * rs = nullptr;
    dev_state * ds = find_dev(dev, &rs);
    for (const auto * e : exts_of(rs->drv.backend)) {
        bool result = false;
        if (e->supports_op && e->supports_op(dev, op, &result)) {
            return result;
        }
    }
    if (ggml_hook_has_kernels(rs->drv.backend) && ggml_hook_find_op(&ds->info, op)) {
        return true;
    }
    // the custom ops of the hooks (EXL3): only hook kernels compute them
    if (ggml_hook_is_hook_custom(op)) {
        return false;
    }
    // upstream does not know the hook types: only the CPU matrix multiplication uses their CPU traits
    if (uses_hook_type(op)) {
        const bool cpu_mul_mat = rs->drv.backend == GGML_HOOK_BACKEND_CPU && (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) &&
                                 !ggml_hook_type_is_hook(op->type) && op->src[0] && ggml_hook_type_is_hook(op->src[0]->type) &&
                                 (!op->src[1] || !ggml_hook_type_is_hook(op->src[1]->type));
        if (!cpu_mul_mat) {
            return false;
        }
    }
    return ds->orig.supports_op(dev, op);
}

bool wrap_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    reg_state * rs = nullptr;
    dev_state * ds = find_dev(dev, &rs);
    for (const auto * e : exts_of(rs->drv.backend)) {
        bool result = false;
        if (e->supports_buft && e->supports_buft(dev, buft, &result)) {
            return result;
        }
    }
    for (auto * b : ds->ext_bufts) {
        if (b == buft) {
            return true;
        }
    }
    return ds->orig.supports_buft(dev, buft);
}

bool wrap_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    reg_state * rs = nullptr;
    dev_state * ds = find_dev(dev, &rs);
    for (const auto * e : exts_of(rs->drv.backend)) {
        bool result = false;
        if (e->offload_op && e->offload_op(dev, op, &result)) {
            return result;
        }
    }
    return ds->orig.offload_op ? ds->orig.offload_op(dev, op) : false;
}

void wrap_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    reg_state * rs = nullptr;
    dev_state * ds = find_dev(dev, &rs);
    ds->orig.get_memory(dev, free, total);
    for (const auto * e : exts_of(rs->drv.backend)) {
        if (e->memory) {
            e->memory(dev, free, total);
        }
    }
}

void wrap_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    reg_state * rs = nullptr;
    dev_state * ds = find_dev(dev, &rs);
    ds->orig.get_props(dev, props);
    // upstream fills the memory of the props without its device interface
    for (const auto * e : exts_of(rs->drv.backend)) {
        if (e->memory) {
            e->memory(dev, &props->memory_free, &props->memory_total);
        }
    }
    for (const auto * e : exts_of(rs->drv.backend)) {
        if (e->props) {
            e->props(dev, props);
        }
    }
}

// backend

reg_state * reg_of_backend(ggml_backend_t backend) {
    return backend->device ? find_reg(backend->device->reg) : nullptr;
}

void wrap_backend_free(ggml_backend_t backend) {
    reg_state * rs = reg_of_backend(backend);
    for (const auto * e : exts_of(rs->drv.backend)) {
        if (e->backend_free) {
            e->backend_free(backend);
        }
    }
    rs->orig_backend.free(backend);
}

void wrap_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    reg_state * rs = reg_of_backend(backend);
    bool replaced = false;
    for (const auto * e : exts_of(rs->drv.backend)) {
        if (!replaced && e->replace_graph_optimize) {
            replaced = e->replace_graph_optimize(backend, cgraph, params);
        }
    }
    if (!replaced && rs->orig_backend.graph_optimize) {
        rs->orig_backend.graph_optimize(backend, cgraph, params);
    }
    for (const auto * e : exts_of(rs->drv.backend)) {
        if (e->graph_optimize) {
            e->graph_optimize(backend, cgraph, params);
        }
    }
    if (!params || !rs->drv.run_fusion || !ggml_hook_has_kernels(rs->drv.backend)) {
        return;
    }
    // on the final node order: the nodes of a fusion and their srcs stay allocated until its last node
    const dev_state * ds = find_dev(backend->device);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        int n = 0;
        if (!ggml_hook_find_fusion(&ds->info, cgraph, i, &n)) {
            continue;
        }
        ggml_tensor * last = cgraph->nodes[i + n - 1];
        for (int j = i; j < i + n - 1; ++j) {
            params->add_alloc_dep(params->user_data, cgraph->nodes[j], last);
            for (int k = 0; k < GGML_MAX_SRC; ++k) {
                if (cgraph->nodes[j]->src[k]) {
                    params->add_alloc_dep(params->user_data, cgraph->nodes[j]->src[k], last);
                }
            }
        }
        i += n - 1;
    }
}

ggml_cgraph graph_view(ggml_cgraph * g, int i0, int i1) {
    ggml_cgraph v = *g;
    v.size      = 0;
    v.n_nodes   = i1 - i0;
    v.n_leafs   = 0;
    v.nodes     = g->nodes + i0;
    v.grads     = nullptr;
    v.grad_accs = nullptr;
    v.leafs     = nullptr;
    v.uid       = 0;
    return v;
}

// a graph that no kernel took, by uid: when it comes back (graph reuse), upstream computes it without the scan
struct untaken_graph {
    ggml_backend_t backend = nullptr;
    uint64_t       uid     = 0;
    const void *   state   = nullptr;
    std::string    arch;
};

const char * model_arch() {
    const char * arch = ggml_hook_model_arch();
    return arch ? arch : "";
}

untaken_graph & untaken_slot(ggml_backend_t backend) {
    static thread_local untaken_graph slots[8];
    return slots[((uintptr_t) backend >> 4) % 8];
}

ggml_status cut_compute(reg_state * rs, const dev_state * ds, ggml_backend_t backend, ggml_cgraph * cgraph) {
    const ggml_hook_driver & drv = rs->drv;
    untaken_graph & seen = untaken_slot(backend);
    const void * state = ggml_hook_kernels_state();
    if (cgraph->uid != 0 && seen.backend == backend && seen.uid == cgraph->uid && seen.state == state &&
            seen.arch == model_arch()) {
        return rs->orig_backend.graph_compute(backend, cgraph);
    }
    int  seg_start = 0;
    bool in_run    = false;
    bool in_run_any = false; // a hook kernel ran or a node was marked computed

    auto flush = [&](int end) -> ggml_status {
        if (in_run) {
            drv.end(backend, false);
            in_run = false;
        }
        if (seg_start == 0 && end == cgraph->n_nodes && !in_run_any) {
            // no hook kernel ran: the graph itself, with its uid (the CUDA graph cache of upstream checks it)
            if (cgraph->uid != 0) {
                seen = { backend, cgraph->uid, state, model_arch() };
            }
            return rs->orig_backend.graph_compute(backend, cgraph);
        }
        if (end > seg_start) {
            ggml_cgraph v = graph_view(cgraph, seg_start, end);
            return rs->orig_backend.graph_compute(backend, &v);
        }
        return GGML_STATUS_SUCCESS;
    };

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || ggml_op_is_empty(node->op) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        const bool computed = ggml_hook_take_computed(node);
        int n = 0;
        const ggml_hook_kernel * fusion = !computed && drv.run_fusion ? ggml_hook_find_fusion(&ds->info, cgraph, i, &n) : nullptr;
        const ggml_hook_kernel * op     = !computed && !fusion && drv.cut_ops ? ggml_hook_find_op(&ds->info, node) : nullptr;
        if (!computed && !fusion && !op) {
            // an upstream node: the run of hook kernels ends, the node starts the next segment
            if (in_run) {
                drv.end(backend, false);
                in_run = false;
            }
            continue;
        }

        // the upstream segment before the node. Inside a run of hook kernels, only empty nodes can be skipped.
        if (!in_run && i > seg_start) {
            ggml_cgraph v = graph_view(cgraph, seg_start, i);
            const ggml_status st = rs->orig_backend.graph_compute(backend, &v);
            if (st != GGML_STATUS_SUCCESS) {
                return st;
            }
        }

        in_run_any = true;
        if (computed) {
            seg_start = i + 1;
            continue;
        }
        if (!in_run) {
            drv.begin(backend);
            in_run = true;
        }
        if (fusion) {
            drv.run_fusion(backend, fusion, cgraph, i, n);
            ggml_hook_kernel_used(fusion, cgraph->nodes[i + n - 1]);
            i += n - 1;
        } else if (drv.run_op(backend, op, cgraph, i)) {
            ggml_hook_kernel_used(op, node);
        } else {
            // the kernel declined: upstream computes the node in the next segment
            drv.end(backend, false);
            in_run    = false;
            seg_start = i;
            continue;
        }
        seg_start = i + 1;
    }

    // when only empty nodes follow, upstream would not submit: the run of hook kernels is the last work of the graph
    bool tail_work = false;
    for (int i = seg_start; i < cgraph->n_nodes && !tail_work; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        tail_work = !ggml_is_empty(node) && !ggml_op_is_empty(node->op) && (node->flags & GGML_TENSOR_FLAG_COMPUTE);
    }
    if (!tail_work) {
        if (in_run) {
            drv.end(backend, true);
        }
        return GGML_STATUS_SUCCESS;
    }
    return flush(cgraph->n_nodes);
}

ggml_status wrap_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    reg_state * rs = reg_of_backend(backend);
    const dev_state * ds = find_dev(backend->device);
    const auto & exts = exts_of(rs->drv.backend);

    for (const auto * e : exts) {
        if (e->graph_compute_begin) {
            e->graph_compute_begin(backend, cgraph);
        }
    }

    ggml_status st = GGML_STATUS_SUCCESS;
    bool replaced = false;
    for (const auto * e : exts) {
        if (!replaced && e->replace_graph_compute) {
            replaced = e->replace_graph_compute(backend, cgraph, &st);
        }
    }
    const bool cut = ggml_hook_has_kernels(rs->drv.backend) && (rs->drv.cut_ops || rs->drv.run_fusion);
    if (replaced) {
        // computed by another implementation of the backend
    } else if (cut) {
        st = cut_compute(rs, ds, backend, cgraph);
    } else {
        st = rs->orig_backend.graph_compute(backend, cgraph);
    }
    // a mark that no node of this graph took must not skip a node of the next graph
    ggml_hook_clear_computed();

    for (const auto * e : exts) {
        if (e->graph_compute_end) {
            e->graph_compute_end(backend, cgraph, st);
        }
    }
    return st;
}

void wrap_backend(reg_state * rs, ggml_backend_t backend) {
    std::call_once(rs->backend_once, [&] { rs->orig_backend = backend->iface; });
    backend->iface.free           = wrap_backend_free;
    backend->iface.graph_compute  = wrap_graph_compute;
    backend->iface.graph_optimize = wrap_graph_optimize;
    for (const auto * e : exts_of(rs->drv.backend)) {
        if (e->backend_init) {
            e->backend_init(backend);
        }
    }
}

ggml_backend_t wrap_init_backend(ggml_backend_dev_t dev, const char * params) {
    reg_state * rs = nullptr;
    dev_state * ds = find_dev(dev, &rs);
    ggml_backend_t backend = ds->orig.init_backend(dev, params);
    if (backend) {
        wrap_backend(rs, backend);
    }
    return backend;
}

// registry

typedef ggml_backend_buffer_type_t * (*extra_bufts_fn)(ggml_backend_dev_t);

ggml_backend_buffer_type_t * wrap_dev_extra_bufts(ggml_backend_dev_t dev) {
    reg_state * rs = nullptr;
    dev_state * ds = find_dev(dev, &rs);
    std::call_once(ds->extra_once, [&] {
        for (const auto * e : exts_of(rs->drv.backend)) {
            for (ggml_backend_buffer_type_t * p = e->extra_bufts ? e->extra_bufts(dev) : nullptr; p && *p; ++p) {
                ds->extra.push_back(*p);
                ds->ext_bufts.push_back(*p);
            }
        }
        auto fn = (extra_bufts_fn) (rs->orig.get_proc_address ? rs->orig.get_proc_address(rs->reg, "ggml_backend_dev_get_extra_bufts") : nullptr);
        for (ggml_backend_buffer_type_t * p = fn ? fn(dev) : nullptr; p && *p; ++p) {
            if (!is_private_buft(*rs, *p)) {
                ds->extra.push_back(*p);
            }
        }
        ds->extra.push_back(nullptr);
    });
    return ds->extra.data();
}

void * wrap_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    reg_state * rs = find_reg(reg);
    for (const auto * e : exts_of(rs->drv.backend)) {
        if (e->proc_address) {
            if (void * p = e->proc_address(reg, name)) {
                return p;
            }
        }
    }
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        return (void *) wrap_dev_extra_bufts;
    }
    return rs->orig.get_proc_address ? rs->orig.get_proc_address(reg, name) : nullptr;
}

} // namespace

void ggml_hook_register_backend_ext(const ggml_hook_backend_ext * ext, const void * module) {
    if (ext->abi != GGML_HOOK_ABI_VERSION || ext->backend < 0 || ext->backend >= GGML_HOOK_BACKEND_COUNT) {
        GGML_LOG_ERROR("ggml-hooks: backend extension %s is not valid: skipped\n", ext->name ? ext->name : "(null)");
        return;
    }
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    for (const auto & x : s.exts) {
        if (x.desc == ext) {
            return; // static builds: every registrar sees the descriptors of the whole executable
        }
    }
    s.exts.push_back({ *ext, ext->name ? ext->name : "", module, ext });
    s.exts.back().e.name = s.exts.back().name.c_str();
    publish_exts(s);
}

void ggml_hook_register_driver(const ggml_hook_driver * driver, const void * module) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    for (const auto & d : s.drivers) {
        if (d.desc == driver && !d.removed) {
            return;
        }
    }
    s.drivers.push_back({ *driver, module, driver });
}

// called by ggml_hook_unregister_module
void ggml_hook_wrap_unregister_module(const void * module) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    for (auto & d : s.drivers) {
        if (d.module == module) {
            d.removed = true;
        }
    }
    const size_t n = s.exts.size();
    s.exts.erase(std::remove_if(s.exts.begin(), s.exts.end(), [&](const ext_entry & x) { return x.module == module; }), s.exts.end());
    if (s.exts.size() != n) {
        publish_exts(s);
    }
}

void ggml_hook_install(ggml_backend_reg_t reg) {
    if (!reg || find_reg(reg)) {
        return;
    }
    auto & s = state();
    const ggml_hook_driver * drv = nullptr;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        if (find_reg(reg) || s.n_regs.load() >= (int) (sizeof(s.reg_ptrs) / sizeof(s.reg_ptrs[0]))) {
            return;
        }
        const char * name = reg->iface.get_name(reg);
        for (const auto & d : s.drivers) {
            if (!d.removed && strcmp(d.d.reg_name, name) == 0 && d.d.reg() == reg) {
                drv = &d.d;
                break;
            }
        }
        if (!drv) {
            return;
        }

        s.regs.emplace_back();
        reg_state & rs = s.regs.back();
        rs.reg  = reg;
        rs.drv  = *drv;
        rs.orig = reg->iface;

        const size_t n_dev = reg->iface.get_device_count(reg);
        for (size_t i = 0; i < n_dev; ++i) {
            ggml_backend_dev_t dev = reg->iface.get_device(reg, i);
            rs.devs.emplace_back();
            dev_state & ds = rs.devs.back();
            ds.dev  = dev;
            ds.orig = dev->iface;
            ds.info = {};
            ds.info.backend = drv->backend;
            ds.info.device  = (int) i;
            drv->dev_info(dev, &ds.info);
            ds.name      = ds.info.name ? ds.info.name : "";
            ds.info.name = ds.name.c_str();
        }
        // publish before wrapping: the wrapped functions look the state up
        s.reg_ptrs[s.n_regs.load()] = &rs;
        s.n_regs.fetch_add(1, std::memory_order_release);

        reg->iface.get_proc_address = wrap_get_proc_address;
        for (auto & ds : rs.devs) {
            ggml_backend_dev_t dev = ds.dev;
            dev->iface.supports_op   = wrap_supports_op;
            dev->iface.supports_buft = wrap_supports_buft;
            dev->iface.offload_op    = wrap_offload_op;
            dev->iface.get_memory    = wrap_get_memory;
            dev->iface.get_props     = wrap_get_props;
            dev->iface.init_backend  = wrap_init_backend;
        }
        GGML_LOG_DEBUG("ggml-hooks: wrapped %s, %zu device(s)\n", name, n_dev);
    }
    if (drv->installed) {
        drv->installed(reg);
    }
}

void ggml_hook_wrap_backend(ggml_backend_t backend) {
    reg_state * rs = backend ? reg_of_backend(backend) : nullptr;
    if (rs && backend->iface.graph_compute != wrap_graph_compute) {
        wrap_backend(rs, backend);
    }
}

const ggml_hook_dev * ggml_hook_dev_of(ggml_backend_dev_t dev) {
    const dev_state * ds = find_dev(dev);
    return ds ? &ds->info : nullptr;
}

const ggml_hook_dev * ggml_hook_dev_of_backend(ggml_backend_t backend) {
    return backend ? ggml_hook_dev_of(backend->device) : nullptr;
}

bool ggml_hook_skip_realloc_sync(ggml_backend_t backend) {
    reg_state * rs = backend ? reg_of_backend(backend) : nullptr;
    if (!rs) {
        return false;
    }
    for (const auto * e : exts_of(rs->drv.backend)) {
        if (e->skip_realloc_sync && e->skip_realloc_sync(backend)) {
            return true;
        }
    }
    return false;
}
