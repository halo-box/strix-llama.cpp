// Activation: the public ggml-backend-reg.cpp functions, renamed in that file (hooks_rename, hooks/ggml/ggml-hooks.cmake).
// Every user gets its devices and backends through them, so the upstream registries are wrapped before their first use.
// This works the same in static, shared and GGML_BACKEND_DL builds.

#include "ggml-hooks-impl.h"

// name is declared GGML_API by ggml-backend.h: the definition is exported
#define GGML_HOOK_WRAPPED(ret, name, ...) \
    extern "C" { GGML_API ret name##__upstream(__VA_ARGS__); } \
    ret name(__VA_ARGS__)

extern "C" { GGML_API size_t ggml_backend_reg_count__upstream(void); }
extern "C" { GGML_API ggml_backend_reg_t ggml_backend_reg_get__upstream(size_t index); }

static void install_all() {
    const size_t n = ggml_backend_reg_count__upstream();
    for (size_t i = 0; i < n; ++i) {
        ggml_hook_install(ggml_backend_reg_get__upstream(i));
    }
}

size_t ggml_backend_reg_count(void) {
    install_all();
    return ggml_backend_reg_count__upstream();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    install_all();
    return ggml_backend_reg_get__upstream(index);
}

GGML_HOOK_WRAPPED(ggml_backend_reg_t, ggml_backend_reg_by_name, const char * name) {
    install_all();
    return ggml_backend_reg_by_name__upstream(name);
}

GGML_HOOK_WRAPPED(size_t, ggml_backend_dev_count, void) {
    install_all();
    return ggml_backend_dev_count__upstream();
}

GGML_HOOK_WRAPPED(ggml_backend_dev_t, ggml_backend_dev_get, size_t index) {
    install_all();
    return ggml_backend_dev_get__upstream(index);
}

GGML_HOOK_WRAPPED(ggml_backend_dev_t, ggml_backend_dev_by_name, const char * name) {
    install_all();
    return ggml_backend_dev_by_name__upstream(name);
}

GGML_HOOK_WRAPPED(ggml_backend_dev_t, ggml_backend_dev_by_type, enum ggml_backend_dev_type type) {
    install_all();
    return ggml_backend_dev_by_type__upstream(type);
}

GGML_HOOK_WRAPPED(ggml_backend_t, ggml_backend_init_by_name, const char * name, const char * params) {
    install_all();
    return ggml_backend_init_by_name__upstream(name, params);
}

GGML_HOOK_WRAPPED(ggml_backend_t, ggml_backend_init_by_type, enum ggml_backend_dev_type type, const char * params) {
    install_all();
    return ggml_backend_init_by_type__upstream(type, params);
}

GGML_HOOK_WRAPPED(ggml_backend_t, ggml_backend_init_best, void) {
    install_all();
    return ggml_backend_init_best__upstream();
}

GGML_HOOK_WRAPPED(ggml_backend_reg_t, ggml_backend_load, const char * path) {
    ggml_backend_reg_t reg = ggml_backend_load__upstream(path);
    install_all();
    return reg;
}

GGML_HOOK_WRAPPED(void, ggml_backend_load_all, void) {
    ggml_backend_load_all__upstream();
    install_all();
}

GGML_HOOK_WRAPPED(void, ggml_backend_load_all_from_path, const char * dir_path) {
    ggml_backend_load_all_from_path__upstream(dir_path);
    install_all();
}
