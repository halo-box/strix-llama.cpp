#pragma once

// ggml hooks: kernels, fusions and backend extensions, connected to upstream ggml without changing its files.
// The backends are wrapped at run time (see ggml-hooks-impl.h). The registry lives in ggml-base.
// See hooks/README.md.
//
// Registration: a hook file declares constant descriptors (GGML_HOOK_REGISTER_*). They run no code when the library
// is loaded: a GGML_BACKEND_DL CPU variant can be loaded on a CPU that cannot run its code. The registrar of the
// library (generated, see ggml-hooks-link.h) registers them.

#ifndef __cplusplus
#error "ggml-hooks.h is a C++ header"
#endif

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#define GGML_HOOK_ABI_VERSION 5

enum ggml_hook_backend {
    GGML_HOOK_BACKEND_CPU,
    GGML_HOOK_BACKEND_CUDA, // CUDA, HIP and MUSA
    GGML_HOOK_BACKEND_VULKAN,
    GGML_HOOK_BACKEND_COUNT,
};

enum ggml_hook_kind {
    GGML_HOOK_KIND_OP,     // computes one node
    GGML_HOOK_KIND_FUSION, // computes several consecutive nodes
};

struct ggml_hook_dev {
    enum ggml_hook_backend backend;
    int                    device;
    int                    cc;        // CUDA/HIP compute capability (GGML_CUDA_CC_*), 0 on other backends
    uint32_t               vendor_id; // Vulkan vendor id, 0 on other backends
    const char *           name;
    bool                   integrated;
};

// defined in ggml-backend-impl.h
struct ggml_backend_graph_optimize_params;

// backend contexts, defined by the upstream backends
struct ggml_compute_params;
struct ggml_backend_cuda_context;
struct ggml_backend_vk_context;
struct vk_context_struct;
typedef std::shared_ptr<vk_context_struct> vk_context;

// kernel signatures, see ggml-hooks-cpu.h, ggml-hooks-cuda.h, ggml-hooks-vk.h
typedef void (*ggml_hook_cpu_op_fn)     (const ggml_compute_params * params, ggml_tensor * dst);
typedef bool (*ggml_hook_cuda_op_fn)    (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
typedef void (*ggml_hook_cuda_fusion_fn)(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int n_nodes);
typedef bool (*ggml_hook_vk_op_fn)      (ggml_backend_vk_context * ctx, vk_context & subctx, ggml_cgraph * cgraph, int node_idx);
typedef void (*ggml_hook_vk_fusion_fn)  (ggml_backend_vk_context * ctx, vk_context & subctx, ggml_cgraph * cgraph, int node_idx, int n_nodes);

// the compute function of a kernel: the field of its backend and kind is set
struct ggml_hook_compute {
    ggml_hook_cpu_op_fn      cpu_op      = nullptr;
    ggml_hook_cuda_op_fn     cuda_op     = nullptr;
    ggml_hook_cuda_fusion_fn cuda_fusion = nullptr;
    ggml_hook_vk_op_fn       vk_op       = nullptr;
    ggml_hook_vk_fusion_fn   vk_fusion   = nullptr;
};

typedef bool (*ggml_hook_supports_fn)(const ggml_hook_dev * dev, const ggml_tensor * node);
// return the number of nodes the fusion computes, starting at node i (>= 2), or 0 if no match
typedef int  (*ggml_hook_match_fn)(const ggml_hook_dev * dev, const ggml_cgraph * cgraph, int i);

struct ggml_hook_kernel {
    int                   abi      = GGML_HOOK_ABI_VERSION;
    const char *          name     = nullptr;
    ggml_hook_backend     backend  = GGML_HOOK_BACKEND_CPU;
    ggml_hook_kind        kind     = GGML_HOOK_KIND_OP;
    ggml_op               op       = GGML_OP_NONE; // op of the node, or of the first node of a fusion; GGML_OP_COUNT: every op
    int                   priority = 0;            // higher wins
    bool                  enabled_by_default = true;
    ggml_hook_supports_fn supports = nullptr; // GGML_HOOK_KIND_OP
    ggml_hook_match_fn    match    = nullptr; // GGML_HOOK_KIND_FUSION
    ggml_hook_compute     compute;
    const char *          models   = nullptr; // nullptr: every model, else comma separated GGUF archs ("qwen35moe,llama")
    // CPU: the work buffer the kernel needs (bytes for the whole node), 0 or nullptr: the size upstream computes for the op
    size_t (*cpu_work_size)(int n_threads, const ggml_tensor * op) = nullptr;
    void *                internal = nullptr; // set by the registry in its copy
};

// backend extensions: callbacks on the wrapped device, backend and registry interfaces. Every field is optional.
struct ggml_hook_backend_ext {
    int               abi     = GGML_HOOK_ABI_VERSION;
    const char *      name    = nullptr;
    ggml_hook_backend backend = GGML_HOOK_BACKEND_CPU;

    // after upstream: change the reported memory or properties
    void (*memory)(ggml_backend_dev_t dev, size_t * free, size_t * total) = nullptr;
    void (*props)(ggml_backend_dev_t dev, ggml_backend_dev_props * props) = nullptr;
    // return true to answer instead of upstream
    bool (*supports_op)(ggml_backend_dev_t dev, const ggml_tensor * op, bool * result) = nullptr;
    bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft, bool * result) = nullptr;
    bool (*offload_op)(ggml_backend_dev_t dev, const ggml_tensor * op, bool * result) = nullptr;
    // extra buffer types of the device, put before the upstream ones. nullptr terminated, the array must live as long as the device.
    ggml_backend_buffer_type_t * (*extra_bufts)(ggml_backend_dev_t dev) = nullptr;
    // registry procs: return non nullptr to answer instead of upstream
    void * (*proc_address)(ggml_backend_reg_t reg, const char * name) = nullptr;
    // backend life time
    void (*backend_init)(ggml_backend_t backend) = nullptr;
    void (*backend_free)(ggml_backend_t backend) = nullptr;
    // after the upstream graph_optimize: reorder the graph, add allocation dependencies (params may be nullptr)
    void (*graph_optimize)(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) = nullptr;
    // around the compute of a graph
    void (*graph_compute_begin)(ggml_backend_t backend, ggml_cgraph * cgraph) = nullptr;
    void (*graph_compute_end)(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_status status) = nullptr;
    // the scheduler re-reserves its buffers: return true to skip the sync of this backend (HOOK(sched-realloc-sync))
    bool (*skip_realloc_sync)(ggml_backend_t backend) = nullptr;
    // another implementation of the backend: return true to compute the graph instead of upstream and the hook kernels
    bool (*replace_graph_compute)(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_status * status) = nullptr;
    // return true to optimize the graph instead of upstream (the graph_optimize callbacks still run after)
    bool (*replace_graph_optimize)(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) = nullptr;
};

// parameters: a hook point or a kernel asks for a value, a provider can override the default.
typedef bool (*ggml_hook_param_fn)(const ggml_hook_dev * dev, const ggml_tensor * node, int64_t * value);

struct ggml_hook_param {
    const char *       name;
    ggml_hook_param_fn fn;
};

extern "C" {

// run-time registration (tests, plugins). module: the library the entry belongs to, its entries are removed when it is unloaded.
GGML_API void ggml_hook_register_kernel(const ggml_hook_kernel * kernel, const void * module);
GGML_API void ggml_hook_register_backend_ext(const ggml_hook_backend_ext * ext, const void * module);
GGML_API void ggml_hook_register_param(const ggml_hook_param * param, const void * module);
GGML_API void ggml_hook_unregister_module(const void * module);

// model tag: set by the libllama hooks while a context builds, allocates or computes a graph.
// Kernels and rules can depend on it. Empty strings when no model is set (ggml used alone).
GGML_API void         ggml_hook_set_model(const char * arch, const char * name);
GGML_API const char * ggml_hook_model_arch(void);
GGML_API const char * ggml_hook_model_name(void);

// a kernel that computes a node later in the graph marks it: the backend wrapper then skips it (thread local, reset per graph)
GGML_API void ggml_hook_mark_computed(const ggml_tensor * node);
GGML_API bool ggml_hook_take_computed(const ggml_tensor * node);
GGML_API void ggml_hook_clear_computed(void);

// providers are tried in registration order
GGML_API int64_t ggml_hook_param_value(const char * name, const ggml_hook_dev * dev, const ggml_tensor * node, int64_t default_value);

// lookups, used by the backend wrappers
GGML_API bool                     ggml_hook_has_kernels(ggml_hook_backend backend);
GGML_API const ggml_hook_kernel * ggml_hook_find_op    (const ggml_hook_dev * dev, const ggml_tensor * node);
GGML_API const ggml_hook_kernel * ggml_hook_find_fusion(const ggml_hook_dev * dev, const ggml_cgraph * cgraph, int i, int * n_nodes);
GGML_API void                     ggml_hook_kernel_used(const ggml_hook_kernel * kernel, const ggml_tensor * node);

// a switch that is not a kernel (a replaced upstream subsystem): the GGML_HOOKS rules without an arch over its name
GGML_API bool ggml_hook_switch(const char * name, bool default_on);

// tests and A/B runs
GGML_API int      ggml_hook_kernel_set_enabled(const char * pattern, bool enabled); // returns the number of kernels changed
GGML_API uint64_t ggml_hook_kernel_hits(const char * name);
GGML_API void     ggml_hook_kernel_reset_hits(void);
GGML_API void     ggml_hook_kernel_print(void);

// the hook device of a wrapped device or backend, nullptr if not wrapped
GGML_API const ggml_hook_dev * ggml_hook_dev_of(ggml_backend_dev_t dev);
GGML_API const ggml_hook_dev * ggml_hook_dev_of_backend(ggml_backend_t backend);

}

// constexpr helpers for the descriptors

constexpr ggml_hook_compute ggml_hook_fn(ggml_hook_cpu_op_fn f)      { ggml_hook_compute c; c.cpu_op      = f; return c; }
constexpr ggml_hook_compute ggml_hook_fn(ggml_hook_cuda_op_fn f)     { ggml_hook_compute c; c.cuda_op     = f; return c; }
constexpr ggml_hook_compute ggml_hook_fn(ggml_hook_cuda_fusion_fn f) { ggml_hook_compute c; c.cuda_fusion = f; return c; }
constexpr ggml_hook_compute ggml_hook_fn(ggml_hook_vk_op_fn f)       { ggml_hook_compute c; c.vk_op       = f; return c; }
constexpr ggml_hook_compute ggml_hook_fn(ggml_hook_vk_fusion_fn f)   { ggml_hook_compute c; c.vk_fusion   = f; return c; }

constexpr ggml_hook_kernel ggml_hook_op_kernel(const char * name, ggml_hook_backend backend, ggml_op op, int priority, ggml_hook_supports_fn supports, ggml_hook_compute compute) {
    ggml_hook_kernel k;
    k.name     = name;
    k.backend  = backend;
    k.kind     = GGML_HOOK_KIND_OP;
    k.op       = op;
    k.priority = priority;
    k.supports = supports;
    k.compute  = compute;
    return k;
}

constexpr ggml_hook_kernel ggml_hook_fusion_kernel(const char * name, ggml_hook_backend backend, ggml_op first_op, int priority, ggml_hook_match_fn match, ggml_hook_compute compute) {
    ggml_hook_kernel k;
    k.name     = name;
    k.backend  = backend;
    k.kind     = GGML_HOOK_KIND_FUSION;
    k.op       = first_op;
    k.priority = priority;
    k.match    = match;
    k.compute  = compute;
    return k;
}

constexpr ggml_hook_backend_ext ggml_hook_ext(const char * name, ggml_hook_backend backend) {
    ggml_hook_backend_ext e;
    e.name    = name;
    e.backend = backend;
    return e;
}

#include "ggml-hooks-link.h"

// constant descriptors, registered by the registrar of the library. The arguments must be constant expressions,
// for example a constexpr lambda: GGML_HOOK_REGISTER_KERNEL([] { auto k = ggml_hook_op_kernel(...); k.priority = 1; return k; }());
#define GGML_HOOK_REGISTER_KERNEL(...) \
    static constexpr ggml_hook_kernel GGML_HOOK_CONCAT(ggml_hook_kernel_, __LINE__) = __VA_ARGS__; \
    GGML_HOOK_DESCRIPTOR(ggml_hook_kernels, GGML_HOOK_CONCAT(ggml_hook_kernel_, __LINE__))

#define GGML_HOOK_REGISTER_BACKEND_EXT(...) \
    static constexpr ggml_hook_backend_ext GGML_HOOK_CONCAT(ggml_hook_ext_, __LINE__) = __VA_ARGS__; \
    GGML_HOOK_DESCRIPTOR(ggml_hook_exts, GGML_HOOK_CONCAT(ggml_hook_ext_, __LINE__))

#define GGML_HOOK_REGISTER_PARAM(name, fn) \
    static constexpr ggml_hook_param GGML_HOOK_CONCAT(ggml_hook_param_, __LINE__) = { name, fn }; \
    GGML_HOOK_DESCRIPTOR(ggml_hook_params, GGML_HOOK_CONCAT(ggml_hook_param_, __LINE__))
