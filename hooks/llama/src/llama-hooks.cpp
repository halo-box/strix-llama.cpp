// libllama hooks: registry and the link-time wrappers of the upstream functions (hooks/llama/llama-hooks.cmake).

#include "llama-hooks.h"
#include "llama-hooks-access.h"

#include "ggml-hooks.h"
#include "llama-adapter.h"
#include "llama-arch.h"
#include "llama-context.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model-loader.h"
#include "llama-model.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <unordered_map>

LLAMA_HOOK_ACCESS(llama_hooks_ctx_cvec, llama_context, cvec, llama_adapter_cvec_ptr);

namespace {

struct hook_entry {
    llama_hook        h;
    std::string       name;
    std::string       models; // empty: every model
    bool              enabled = true;
    std::atomic<long> hits{ 0 };
};

struct hook_rule {
    std::string pattern;
    bool        enable;
};

struct hook_registry {
    std::mutex             mutex;
    std::deque<hook_entry> entries; // stable addresses
    std::vector<hook_rule> rules;
    bool                   parsed = false;

    std::mutex                                       ctx_mutex;
    std::unordered_map<const void *, llama_context *> contexts; // context key -> context
};

hook_registry & registry() {
    static auto * r = new hook_registry();
    return *r;
}

bool glob_match(const char * p, const char * s) {
    if (*p == '\0') {
        return *s == '\0';
    }
    if (*p == '*') {
        return glob_match(p + 1, s) || (*s != '\0' && glob_match(p, s + 1));
    }
    return *s != '\0' && *p == *s && glob_match(p + 1, s + 1);
}

void apply_rules(const hook_registry & r, hook_entry & e) {
    e.enabled = true;
    for (const auto & rule : r.rules) {
        if (glob_match(rule.pattern.c_str(), e.name.c_str())) {
            e.enabled = rule.enable;
        }
    }
}

// call with the mutex held
void parse_env(hook_registry & r) {
    if (r.parsed) {
        return;
    }
    r.parsed = true;
    const char * env = getenv("LLAMA_HOOKS");
    if (!env) {
        return;
    }
    std::string s(env);
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t end = s.find(',', pos);
        if (end == std::string::npos) {
            end = s.size();
        }
        std::string tok = s.substr(pos, end - pos);
        pos = end + 1;
        if (tok.empty()) {
            continue;
        }
        bool enable = true;
        if (tok[0] == '+' || tok[0] == '-') {
            enable = tok[0] == '+';
            tok = tok.substr(1);
        }
        if (tok == "none") {
            tok = "*";
            enable = false;
        } else if (tok == "all") {
            tok = "*";
        }
        r.rules.push_back({ tok, enable });
    }
}

bool model_matches(const hook_entry & e, const std::string & arch) {
    if (e.models.empty()) {
        return true;
    }
    size_t pos = 0;
    while (pos <= e.models.size()) {
        size_t end = e.models.find(',', pos);
        if (end == std::string::npos) {
            end = e.models.size();
        }
        if (e.models.compare(pos, end - pos, arch) == 0) {
            return true;
        }
        pos = end + 1;
    }
    return false;
}

// enabled entries with the callback `field` set, highest priority first
template <typename F>
std::vector<hook_entry *> entries_with(F llama_hook::*field) {
    auto & r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    parse_env(r);
    std::vector<hook_entry *> out;
    for (auto & e : r.entries) {
        if (e.enabled && e.h.*field) {
            out.push_back(&e);
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const hook_entry * a, const hook_entry * b) { return a->h.priority > b->h.priority; });
    return out;
}

std::string arch_of(const llama_model & model) {
    return llm_arch_name(model.arch);
}

thread_local llama_context * g_current_ctx = nullptr;

// model tag (ggml hooks) and current context for the life of a scope
struct process_scope {
    std::string     prev_arch;
    std::string     prev_name;
    llama_context * prev_ctx;

    process_scope(const llama_model & model, llama_context * ctx) :
        prev_arch(ggml_hook_model_arch()), prev_name(ggml_hook_model_name()), prev_ctx(g_current_ctx) {
        ggml_hook_set_model(llm_arch_name(model.arch), model.name.c_str());
        g_current_ctx = ctx;
    }

    ~process_scope() {
        ggml_hook_set_model(prev_arch.c_str(), prev_name.c_str());
        g_current_ctx = prev_ctx;
    }
};

// a model load: the tag is set by the create_model wrapper, once the arch is known, and restored at the end of the load
struct load_scope {
    std::string prev_arch;
    std::string prev_name;

    load_scope() : prev_arch(ggml_hook_model_arch()), prev_name(ggml_hook_model_name()) {}
    ~load_scope() { ggml_hook_set_model(prev_arch.c_str(), prev_name.c_str()); }
};

std::vector<std::string> paths_of(const char * const * paths, size_t n) {
    std::vector<std::string> out;
    for (size_t i = 0; i < n; ++i) {
        out.emplace_back(paths[i]);
    }
    return out;
}

llama_model * load_model_hooks(const std::vector<std::string> & paths, llama_model_params & params) {
    for (hook_entry * e : entries_with(&llama_hook::model_params)) {
        e->h.model_params(paths, params);
        e->hits++;
    }
    for (hook_entry * e : entries_with(&llama_hook::load_model)) {
        if (llama_model * model = e->h.load_model(paths, params)) {
            e->hits++;
            LLAMA_LOG_INFO("llama-hooks: model loaded by %s\n", e->name.c_str());
            return model;
        }
    }
    return nullptr;
}

void memory_op(llama_memory_t mem, const llama_hook_memory_op & op) {
    for (hook_entry * e : entries_with(&llama_hook::memory_op)) {
        e->h.memory_op(mem, op);
        e->hits++;
    }
}

llama_hook_memory_op make_op(llama_hook_memory_op_type type, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    llama_hook_memory_op op = {};
    op.type   = type;
    op.seq_id = seq_id;
    op.p0     = p0;
    op.p1     = p1;
    return op;
}

} // namespace

void llama_hook_register(const llama_hook & hook) {
    if (!hook.name) {
        LLAMA_LOG_ERROR("llama-hooks: hook without a name: skipped\n");
        return;
    }
    auto & r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    parse_env(r);
    for (const auto & e : r.entries) {
        if (e.name == hook.name) {
            LLAMA_LOG_ERROR("llama-hooks: %s registered twice: skipped\n", hook.name);
            return;
        }
    }
    r.entries.emplace_back();
    hook_entry & e = r.entries.back();
    e.h      = hook;
    e.name   = hook.name;
    e.models = hook.models ? hook.models : "";
    e.h.name   = e.name.c_str();
    e.h.models = hook.models ? e.models.c_str() : nullptr;
    apply_rules(r, e);
}

const void * llama_hooks_context_key(const llm_graph_params & params) {
    return params.cvec;
}

const void * llama_hooks_context_key(const llama_context & ctx) {
    return llama_hook_get(ctx, llama_hooks_ctx_cvec{}).get();
}

llama_context * llama_hooks_context_of(const llm_graph_params & params) {
    auto & r = registry();
    std::lock_guard<std::mutex> lock(r.ctx_mutex);
    auto it = r.contexts.find(params.cvec);
    return it == r.contexts.end() ? nullptr : it->second;
}

llama_context * llama_hooks_current_context() {
    return g_current_ctx;
}

void llama_hook_graph_replace(ggml_cgraph * gf, ggml_tensor * from, ggml_tensor * to) {
    const int n = ggml_graph_n_nodes(gf);
    std::vector<ggml_tensor *> nodes(n);
    for (int i = 0; i < n; ++i) {
        nodes[i] = ggml_graph_node(gf, i);
    }

    for (ggml_tensor * node : nodes) {
        if (node == to) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (node->src[s] == from) {
                node->src[s] = to;
            }
        }
        if (node->view_src == from) {
            node->view_src = to;
        }
    }
    if (from->flags & GGML_TENSOR_FLAG_OUTPUT) {
        ggml_set_output(to);
    }

    // `to` takes the place of `from`; `from` is kept only if a node still reads it (for example `to` is computed from it)
    ggml_graph_clear(gf);
    for (ggml_tensor * node : nodes) {
        ggml_build_forward_expand(gf, node == from ? to : node);
    }
}

int llama_hooks_set_enabled(const char * pattern, bool enabled) {
    auto & r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    parse_env(r);
    r.rules.push_back({ pattern, enabled });
    int changed = 0;
    for (auto & e : r.entries) {
        const bool before = e.enabled;
        apply_rules(r, e);
        changed += before != e.enabled;
    }
    return changed;
}

long llama_hooks_hits(const char * name) {
    auto & r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    for (const auto & e : r.entries) {
        if (e.name == name) {
            return e.hits.load();
        }
    }
    return 0;
}

// link-time wrappers. Each wrapped symbol is listed in hooks/llama/llama-hooks.cmake.

#define LLAMA_HOOK_WRAP extern "C" __attribute__((visibility("default")))

// internal C++ functions, with their Itanium names (this is the first argument of a member function)

extern "C" llama_model * __real__Z18llama_model_createR18llama_model_loaderRK18llama_model_params(llama_model_loader & ml, const llama_model_params & params);
LLAMA_HOOK_WRAP llama_model * __wrap__Z18llama_model_createR18llama_model_loaderRK18llama_model_params(llama_model_loader & ml, const llama_model_params & params) {
    const std::string arch = ml.get_arch_name();
    // the weights are placed with supports_op: kernels and model rules see the model from here on
    std::string name;
    ml.get_key(LLM_KV_GENERAL_NAME, name, false);
    ggml_hook_set_model(arch.c_str(), name.c_str());
    for (hook_entry * e : entries_with(&llama_hook::create_model)) {
        if (!model_matches(*e, arch)) {
            continue;
        }
        if (llama_model * model = e->h.create_model(ml, params)) {
            e->hits++;
            LLAMA_LOG_INFO("llama-hooks: model created by %s\n", e->name.c_str());
            return model;
        }
    }
    return __real__Z18llama_model_createR18llama_model_loaderRK18llama_model_params(ml, params);
}

extern "C" ggml_cgraph * __real__ZNK11llama_model11build_graphERK16llm_graph_params(const llama_model * self, const llm_graph_params & params);
LLAMA_HOOK_WRAP ggml_cgraph * __wrap__ZNK11llama_model11build_graphERK16llm_graph_params(const llama_model * self, const llm_graph_params & params) {
    ggml_cgraph * gf = __real__ZNK11llama_model11build_graphERK16llm_graph_params(self, params);
    if (!gf || !params.res) {
        return gf;
    }
    const std::string arch = arch_of(*self);
    for (hook_entry * e : entries_with(&llama_hook::graph_pass)) {
        if (model_matches(*e, arch)) {
            e->h.graph_pass(*self, params, *params.res);
            e->hits++;
        }
    }
    return params.res->get_gf();
}

extern "C" llama_memory_i * __real__ZNK11llama_model13create_memoryERK19llama_memory_paramsRK13llama_cparams(const llama_model * self, const llama_memory_params & params, const llama_cparams & cparams);
LLAMA_HOOK_WRAP llama_memory_i * __wrap__ZNK11llama_model13create_memoryERK19llama_memory_paramsRK13llama_cparams(const llama_model * self, const llama_memory_params & params, const llama_cparams & cparams) {
    const std::string arch = arch_of(*self);
    for (hook_entry * e : entries_with(&llama_hook::create_memory)) {
        if (!model_matches(*e, arch)) {
            continue;
        }
        if (llama_memory_i * mem = e->h.create_memory(*self, params, cparams)) {
            e->hits++;
            return mem;
        }
    }
    return __real__ZNK11llama_model13create_memoryERK19llama_memory_paramsRK13llama_cparams(self, params, cparams);
}

#define LLAMA_HOOK_CREATE_TENSOR _ZN18llama_model_loader13create_tensorERK13llama_hparamsPKSt6vectorISt4pairIP19ggml_backend_deviceP24ggml_backend_buffer_typeESaIS9_EESD_SD_SD_RK11LLM_TN_IMPLRKSt16initializer_listIlEi
#define LLAMA_HOOK_REAL(name) GGML_HOOK_CONCAT(__real_, name)
#define LLAMA_HOOK_WRAPPED(name) GGML_HOOK_CONCAT(__wrap_, name)

extern "C" ggml_tensor * LLAMA_HOOK_REAL(LLAMA_HOOK_CREATE_TENSOR)(llama_model_loader * self, const llama_hparams & hparams, const buft_list_t * buft_list_cpu,
    const buft_list_t * buft_list_input, const buft_list_t * buft_list_output, const buft_list_t * buft_list_layer, const LLM_TN_IMPL & tn,
    const std::initializer_list<int64_t> & ne, int flags);
LLAMA_HOOK_WRAP ggml_tensor * LLAMA_HOOK_WRAPPED(LLAMA_HOOK_CREATE_TENSOR)(llama_model_loader * self, const llama_hparams & hparams, const buft_list_t * buft_list_cpu,
    const buft_list_t * buft_list_input, const buft_list_t * buft_list_output, const buft_list_t * buft_list_layer, const LLM_TN_IMPL & tn,
    const std::initializer_list<int64_t> & ne, int flags) {
    ggml_tensor * t = LLAMA_HOOK_REAL(LLAMA_HOOK_CREATE_TENSOR)(self, hparams, buft_list_cpu, buft_list_input, buft_list_output, buft_list_layer, tn, ne, flags);
    if (t) {
        for (hook_entry * e : entries_with(&llama_hook::tensor_created)) {
            e->h.tensor_created(*self, tn, t, flags);
            e->hits++;
        }
    }
    return t;
}

// public C API

extern "C" llama_model * __real_llama_model_load_from_file(const char * path, llama_model_params params);
LLAMA_HOOK_WRAP llama_model * __wrap_llama_model_load_from_file(const char * path, llama_model_params params) {
    load_scope scope;
    if (llama_model * model = load_model_hooks({ path }, params)) {
        return model;
    }
    return __real_llama_model_load_from_file(path, params);
}

extern "C" llama_model * __real_llama_model_load_from_splits(const char ** paths, size_t n_paths, llama_model_params params);
LLAMA_HOOK_WRAP llama_model * __wrap_llama_model_load_from_splits(const char ** paths, size_t n_paths, llama_model_params params) {
    load_scope scope;
    if (llama_model * model = load_model_hooks(paths_of(paths, n_paths), params)) {
        return model;
    }
    return __real_llama_model_load_from_splits(paths, n_paths, params);
}

extern "C" llama_model * __real_llama_model_load_from_file_ptr(FILE * file, llama_model_params params);
LLAMA_HOOK_WRAP llama_model * __wrap_llama_model_load_from_file_ptr(FILE * file, llama_model_params params) {
    load_scope scope;
    if (llama_model * model = load_model_hooks({}, params)) {
        return model;
    }
    return __real_llama_model_load_from_file_ptr(file, params);
}

extern "C" llama_model * __real_llama_model_init_from_user(gguf_context * metadata, llama_model_set_tensor_data_t set_tensor_data, void * ud, llama_model_params params);
LLAMA_HOOK_WRAP llama_model * __wrap_llama_model_init_from_user(gguf_context * metadata, llama_model_set_tensor_data_t set_tensor_data, void * ud, llama_model_params params) {
    load_scope scope;
    for (hook_entry * e : entries_with(&llama_hook::model_params)) {
        e->h.model_params({}, params);
        e->hits++;
    }
    return __real_llama_model_init_from_user(metadata, set_tensor_data, ud, params);
}

extern "C" uint32_t __real_llama_model_quantize(const char * fname_inp, const char * fname_out, const llama_model_quantize_params * params);
LLAMA_HOOK_WRAP uint32_t __wrap_llama_model_quantize(const char * fname_inp, const char * fname_out, const llama_model_quantize_params * params) {
    // the quantizer creates the model through llama_model_create, which sets the model tag
    load_scope scope;
    return __real_llama_model_quantize(fname_inp, fname_out, params);
}

extern "C" void __real_llama_model_free(llama_model * model);
LLAMA_HOOK_WRAP void __wrap_llama_model_free(llama_model * model) {
    if (model) {
        for (hook_entry * e : entries_with(&llama_hook::model_free)) {
            e->h.model_free(*model);
            e->hits++;
        }
    }
    __real_llama_model_free(model);
}

extern "C" llama_context * __real_llama_init_from_model(llama_model * model, llama_context_params params);
LLAMA_HOOK_WRAP llama_context * __wrap_llama_init_from_model(llama_model * model, llama_context_params params) {
    if (!model) {
        return __real_llama_init_from_model(model, params);
    }
    const std::string arch = arch_of(*model);
    for (hook_entry * e : entries_with(&llama_hook::context_params)) {
        if (model_matches(*e, arch)) {
            e->h.context_params(*model, params);
            e->hits++;
        }
    }
    llama_context * ctx = nullptr;
    {
        // the constructor reserves the graphs: the kernels see the model
        process_scope scope(*model, nullptr);
        ctx = __real_llama_init_from_model(model, params);
    }
    if (!ctx) {
        return ctx;
    }
    {
        auto & r = registry();
        std::lock_guard<std::mutex> lock(r.ctx_mutex);
        r.contexts[llama_hooks_context_key(*ctx)] = ctx;
    }
    for (hook_entry * e : entries_with(&llama_hook::context_init)) {
        if (model_matches(*e, arch)) {
            e->h.context_init(*ctx);
            e->hits++;
        }
    }
    return ctx;
}

extern "C" void __real_llama_free(llama_context * ctx);
LLAMA_HOOK_WRAP void __wrap_llama_free(llama_context * ctx) {
    if (ctx) {
        const std::string arch = arch_of(ctx->get_model());
        for (hook_entry * e : entries_with(&llama_hook::context_free)) {
            if (model_matches(*e, arch)) {
                e->h.context_free(*ctx);
                e->hits++;
            }
        }
        auto & r = registry();
        std::lock_guard<std::mutex> lock(r.ctx_mutex);
        r.contexts.erase(llama_hooks_context_key(*ctx));
    }
    __real_llama_free(ctx);
}

template <typename F>
static int32_t run_process(llama_context * ctx, int type, F && real) {
    if (!ctx) {
        return real();
    }
    const llama_model & model = ctx->get_model();
    const std::string arch = arch_of(model);
    process_scope scope(model, ctx);
    for (hook_entry * e : entries_with(&llama_hook::process_begin)) {
        if (model_matches(*e, arch)) {
            e->h.process_begin(*ctx, type);
        }
    }
    const int32_t ret = real();
    for (hook_entry * e : entries_with(&llama_hook::process_end)) {
        if (model_matches(*e, arch)) {
            e->h.process_end(*ctx, type, ret);
            e->hits++;
        }
    }
    return ret;
}

extern "C" int32_t __real_llama_process(llama_context * ctx, enum llama_process_type type, llama_batch_ext * batch);
LLAMA_HOOK_WRAP int32_t __wrap_llama_process(llama_context * ctx, enum llama_process_type type, llama_batch_ext * batch) {
    return run_process(ctx, type, [&] { return __real_llama_process(ctx, type, batch); });
}

extern "C" int32_t __real_llama_decode(llama_context * ctx, llama_batch batch);
LLAMA_HOOK_WRAP int32_t __wrap_llama_decode(llama_context * ctx, llama_batch batch) {
    return run_process(ctx, LLAMA_PROCESS_TYPE_DECODE, [&] { return __real_llama_decode(ctx, batch); });
}

extern "C" int32_t __real_llama_encode(llama_context * ctx, llama_batch batch);
LLAMA_HOOK_WRAP int32_t __wrap_llama_encode(llama_context * ctx, llama_batch batch) {
    return run_process(ctx, LLAMA_PROCESS_TYPE_ENCODE, [&] { return __real_llama_encode(ctx, batch); });
}

extern "C" void __real_llama_memory_clear(llama_memory_t mem, bool data);
LLAMA_HOOK_WRAP void __wrap_llama_memory_clear(llama_memory_t mem, bool data) {
    __real_llama_memory_clear(mem, data);
    llama_hook_memory_op op = make_op(LLAMA_HOOK_MEMORY_CLEAR, -1, -1, -1);
    op.data = data;
    memory_op(mem, op);
}

extern "C" bool __real_llama_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1);
LLAMA_HOOK_WRAP bool __wrap_llama_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    const bool res = __real_llama_memory_seq_rm(mem, seq_id, p0, p1);
    llama_hook_memory_op op = make_op(LLAMA_HOOK_MEMORY_SEQ_RM, seq_id, p0, p1);
    op.result = res;
    memory_op(mem, op);
    return res;
}

extern "C" void __real_llama_memory_seq_cp(llama_memory_t mem, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
LLAMA_HOOK_WRAP void __wrap_llama_memory_seq_cp(llama_memory_t mem, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    __real_llama_memory_seq_cp(mem, seq_id_src, seq_id_dst, p0, p1);
    llama_hook_memory_op op = make_op(LLAMA_HOOK_MEMORY_SEQ_CP, seq_id_src, p0, p1);
    op.seq_id_dst = seq_id_dst;
    memory_op(mem, op);
}

extern "C" void __real_llama_memory_seq_keep(llama_memory_t mem, llama_seq_id seq_id);
LLAMA_HOOK_WRAP void __wrap_llama_memory_seq_keep(llama_memory_t mem, llama_seq_id seq_id) {
    __real_llama_memory_seq_keep(mem, seq_id);
    memory_op(mem, make_op(LLAMA_HOOK_MEMORY_SEQ_KEEP, seq_id, -1, -1));
}

extern "C" void __real_llama_memory_seq_add(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta);
LLAMA_HOOK_WRAP void __wrap_llama_memory_seq_add(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    __real_llama_memory_seq_add(mem, seq_id, p0, p1, delta);
    llama_hook_memory_op op = make_op(LLAMA_HOOK_MEMORY_SEQ_ADD, seq_id, p0, p1);
    op.shift = delta;
    memory_op(mem, op);
}

extern "C" void __real_llama_memory_seq_div(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d);
LLAMA_HOOK_WRAP void __wrap_llama_memory_seq_div(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    __real_llama_memory_seq_div(mem, seq_id, p0, p1, d);
    llama_hook_memory_op op = make_op(LLAMA_HOOK_MEMORY_SEQ_DIV, seq_id, p0, p1);
    op.d = d;
    memory_op(mem, op);
}

// hook points (llama-hook-points.h)

void llama_hook_kv_event(const llama_kv_cache * kv, const llama_hook_kv_ev & ev) {
    for (hook_entry * e : entries_with(&llama_hook::kv_event)) {
        e->h.kv_event(*kv, ev);
        e->hits++;
    }
}

void llama_hook_sched_reserve(llama_context & ctx, llama_memory_context_i * mctx, uint32_t n_seqs, uint32_t n_tokens) {
    const std::string arch = arch_of(ctx.get_model());
    for (hook_entry * e : entries_with(&llama_hook::sched_reserve)) {
        if (model_matches(*e, arch)) {
            e->h.sched_reserve(ctx, mctx, n_seqs, n_tokens);
            e->hits++;
        }
    }
}

void llama_hook_resolve_fused_ops(llama_context & ctx, const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const std::string arch = arch_of(ctx.get_model());
    for (hook_entry * e : entries_with(&llama_hook::resolve_fused_ops)) {
        if (model_matches(*e, arch)) {
            e->h.resolve_fused_ops(ctx, mctx, n_seqs);
            e->hits++;
        }
    }
}

void llama_hook_ubatch_done(llama_context & ctx, const llama_ubatch & ubatch, ggml_status status) {
    const std::string arch = arch_of(ctx.get_model());
    for (hook_entry * e : entries_with(&llama_hook::ubatch_done)) {
        if (model_matches(*e, arch)) {
            e->h.ubatch_done(ctx, ubatch, status);
            e->hits++;
        }
    }
}

uint32_t llama_hook_graph_extra_nodes(const llama_context & ctx) {
    const std::string arch = arch_of(ctx.get_model());
    uint32_t n = 0;
    for (hook_entry * e : entries_with(&llama_hook::graph_extra_nodes)) {
        if (model_matches(*e, arch)) {
            n += e->h.graph_extra_nodes(ctx);
            e->hits++;
        }
    }
    return n;
}

bool llama_hook_tensor_str(const LLM_TN_IMPL & tn, std::string & name) {
    for (hook_entry * e : entries_with(&llama_hook::tensor_name)) {
        if (e->h.tensor_name(tn, name)) {
            e->hits++;
            return true;
        }
    }
    return false;
}

const llm_tensor_info * llama_hook_tensor_info(llm_tensor tensor) {
    for (hook_entry * e : entries_with(&llama_hook::tensor_info)) {
        if (const llm_tensor_info * info = e->h.tensor_info(tensor)) {
            e->hits++;
            return info;
        }
    }
    return nullptr;
}

ggml_type llama_hook_kv_type(const llama_model & model, const char * role, ggml_type type) {
    const std::string arch = arch_of(model);
    for (hook_entry * e : entries_with(&llama_hook::kv_type)) {
        if (model_matches(*e, arch)) {
            type = e->h.kv_type(model, role, type);
            e->hits++;
        }
    }
    return type;
}

const llama_mmap::ranges & llama_hook_mmap_ranges(const llama_model_loader & ml, uint32_t idx, const llama_mmap::ranges & ranges) {
    const auto hooks = entries_with(&llama_hook::mmap_ranges);
    if (hooks.empty()) {
        return ranges;
    }
    // used by the llama_mmap constructor right after the call
    static thread_local llama_mmap::ranges out;
    out = ranges;
    for (hook_entry * e : hooks) {
        e->h.mmap_ranges(ml, idx, out);
        e->hits++;
    }
    return out;
}

ggml_type llama_hook_quant_type(const llama_model_quantize_params * params, const ggml_tensor * tensor, ggml_type type) {
    for (hook_entry * e : entries_with(&llama_hook::quant_type)) {
        type = e->h.quant_type(params, tensor, type);
        e->hits++;
    }
    return type;
}
