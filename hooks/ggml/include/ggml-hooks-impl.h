#pragma once

// Backend glue: how the hooks wrap an upstream backend. Used by ggml-hooks-{cpu,cuda,vk}.cpp, not by kernel files.
//
// A driver describes one upstream backend. ggml_hook_install() wraps every device of a matching registry:
// - device: supports_op, supports_buft, offload_op, get_memory, get_props, init_backend
// - backend (from init_backend): graph_compute, graph_optimize, free
// - registry: get_proc_address (extra buffer types)
// The wrapped graph_compute cuts the graph: upstream computes its segments with its own graph_compute,
// the driver runs the hook kernels between the segments.

#include "ggml-hooks.h"

#include "ggml-backend-impl.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_hook_driver {
    enum ggml_hook_backend backend;
    const char *           reg_name;          // ggml_backend_reg_name() of the upstream registry
    ggml_backend_reg_t     (*reg)(void);      // the upstream registry, called only when reg_name matches

    void (*dev_info)(ggml_backend_dev_t dev, struct ggml_hook_dev * info);
    // CPU: op kernels run from the CPU extra buffer type, not from a cut
    bool cut_ops;

    // a run of hook kernels between two upstream segments: begin, kernels, end. last: nothing follows in this graph.
    void (*begin)(ggml_backend_t backend);
    bool (*run_op)    (ggml_backend_t backend, const struct ggml_hook_kernel * k, struct ggml_cgraph * cgraph, int i); // false: upstream computes it
    void (*run_fusion)(ggml_backend_t backend, const struct ggml_hook_kernel * k, struct ggml_cgraph * cgraph, int i, int n);
    void (*end)(ggml_backend_t backend, bool last);

    // after the registry is wrapped
    void (*installed)(ggml_backend_reg_t reg);
    // buffer types of the hooks that must not be shown as extra buffer types of a device (NULL terminated)
    ggml_backend_buffer_type_t * (*private_bufts)(void);
};

GGML_API void ggml_hook_register_driver(const struct ggml_hook_driver * driver, const void * module);

// called by the generated registrar of each library (hooks/cmake/hooks-link.cmake)
// removes the drivers and extensions of a module (ggml_hook_unregister_module)
void ggml_hook_wrap_unregister_module(const void * module);

GGML_API void ggml_hook_register_section(int kind, const void * const * begin, const void * const * end, const void * module);

// wrap reg if a driver matches it. Safe to call many times, from any thread.
GGML_API void ggml_hook_install(ggml_backend_reg_t reg);

// wrap a backend created without the device interface (ggml_backend_cpu_init, ...). Its registry must be installed.
GGML_API void ggml_hook_wrap_backend(ggml_backend_t backend);

#ifdef __cplusplus
}
#endif

// a constant driver of the library: static constexpr ggml_hook_driver d = { ... }; GGML_HOOK_REGISTER_DRIVER(d);
#define GGML_HOOK_REGISTER_DRIVER(var) GGML_HOOK_DESCRIPTOR(ggml_hook_drivers, var)
