// Activation: link-time wrapping of the public ggml-backend-reg.cpp functions (hooks/ggml/ggml-hooks.cmake).
// Every user gets its devices and backends through them, so the upstream registries are wrapped before their first use.
// This works the same in static, shared and GGML_BACKEND_DL builds.

#include "ggml-hooks-impl.h"

#define GGML_HOOK_WRAPPED(ret, name, ...) \
    extern "C" ret __real_##name(__VA_ARGS__); \
    extern "C" __attribute__((visibility("default"))) ret __wrap_##name(__VA_ARGS__)

extern "C" size_t             __real_ggml_backend_reg_count(void);
extern "C" ggml_backend_reg_t __real_ggml_backend_reg_get(size_t index);

static void install_all() {
    const size_t n = __real_ggml_backend_reg_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_hook_install(__real_ggml_backend_reg_get(i));
    }
}

extern "C" __attribute__((visibility("default"))) size_t __wrap_ggml_backend_reg_count(void) {
    install_all();
    return __real_ggml_backend_reg_count();
}

extern "C" __attribute__((visibility("default"))) ggml_backend_reg_t __wrap_ggml_backend_reg_get(size_t index) {
    install_all();
    return __real_ggml_backend_reg_get(index);
}

GGML_HOOK_WRAPPED(ggml_backend_reg_t, ggml_backend_reg_by_name, const char * name) {
    install_all();
    return __real_ggml_backend_reg_by_name(name);
}

GGML_HOOK_WRAPPED(size_t, ggml_backend_dev_count, void) {
    install_all();
    return __real_ggml_backend_dev_count();
}

GGML_HOOK_WRAPPED(ggml_backend_dev_t, ggml_backend_dev_get, size_t index) {
    install_all();
    return __real_ggml_backend_dev_get(index);
}

GGML_HOOK_WRAPPED(ggml_backend_dev_t, ggml_backend_dev_by_name, const char * name) {
    install_all();
    return __real_ggml_backend_dev_by_name(name);
}

GGML_HOOK_WRAPPED(ggml_backend_dev_t, ggml_backend_dev_by_type, enum ggml_backend_dev_type type) {
    install_all();
    return __real_ggml_backend_dev_by_type(type);
}

GGML_HOOK_WRAPPED(ggml_backend_t, ggml_backend_init_by_name, const char * name, const char * params) {
    install_all();
    return __real_ggml_backend_init_by_name(name, params);
}

GGML_HOOK_WRAPPED(ggml_backend_t, ggml_backend_init_by_type, enum ggml_backend_dev_type type, const char * params) {
    install_all();
    return __real_ggml_backend_init_by_type(type, params);
}

GGML_HOOK_WRAPPED(ggml_backend_t, ggml_backend_init_best, void) {
    install_all();
    return __real_ggml_backend_init_best();
}

GGML_HOOK_WRAPPED(ggml_backend_reg_t, ggml_backend_load, const char * path) {
    ggml_backend_reg_t reg = __real_ggml_backend_load(path);
    install_all();
    return reg;
}

GGML_HOOK_WRAPPED(void, ggml_backend_load_all, void) {
    __real_ggml_backend_load_all();
    install_all();
}

GGML_HOOK_WRAPPED(void, ggml_backend_load_all_from_path, const char * dir_path) {
    __real_ggml_backend_load_all_from_path(dir_path);
    install_all();
}
