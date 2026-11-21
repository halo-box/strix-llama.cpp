#include "ggml-hooks-impl.h"

#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

namespace {

struct ggml_hook_entry {
    ggml_hook_kernel          k;
    std::string           name;   // owned copies: the module that registered may be unloaded
    std::string           models;
    const void *          module  = nullptr;
    const void *          desc    = nullptr; // the descriptor of the library, to skip duplicates
    bool                  removed = false;
    bool                  enabled = false;
    std::atomic<uint64_t> hits{ 0 };
    std::atomic<bool>     logged{ false };
};

struct ggml_hook_rule {
    std::string arch; // empty: every model
    std::string pattern;
    bool        enable;
};

// immutable once published: lookups read it without a lock
struct ggml_hook_snapshot {
    std::vector<ggml_hook_entry *> ops     [GGML_HOOK_BACKEND_COUNT][GGML_OP_COUNT];
    std::vector<ggml_hook_entry *> fusions [GGML_HOOK_BACKEND_COUNT][GGML_OP_COUNT];
    bool                       any     [GGML_HOOK_BACKEND_COUNT] = {};
    std::vector<ggml_hook_rule>    model_rules; // rules with an arch, applied at lookup time
};

struct ggml_hook_param_provider {
    std::string             name;
    ggml_hook_param_fn      fn;
    const void *            module;
    const ggml_hook_param * desc;
};

struct ggml_hook_params {
    std::vector<ggml_hook_param_provider> providers;
};

struct ggml_hook_registry {
    std::mutex                  mutex;
    std::deque<ggml_hook_entry>     entries;
    std::vector<ggml_hook_rule>     rules;
    std::atomic<ggml_hook_snapshot *> snapshot{ nullptr };
    std::atomic<ggml_hook_params *>   params{ nullptr };
    bool                        log = false;
};

bool glob_match(const char * p, const char * s);

struct ggml_hook_thread_state {
    bool                                         has_model = false;
    std::string                                  arch;
    std::string                                  name;
    std::vector<const ggml_tensor *>             computed;
};

ggml_hook_thread_state & thread_state() {
    static thread_local ggml_hook_thread_state s;
    return s;
}

// The thread that sets the model tag is the one that builds and dispatches the graph. Backend worker threads
// (the CPU threadpool) never set it: they read the tag of the process, last set by a dispatching thread, so all the
// threads of one graph take the same decisions. Strings are interned and never freed.
struct ggml_hook_model_tag {
    const char * arch;
    const char * name;
};

std::atomic<const ggml_hook_model_tag *> g_model_tag{ nullptr };

const ggml_hook_model_tag * intern_tag(const std::string & arch, const std::string & name) {
    static std::mutex mutex;
    static std::deque<std::string> strings;
    static std::deque<ggml_hook_model_tag> tags;
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto & t : tags) {
        if (arch == t.arch && name == t.name) {
            return &t;
        }
    }
    strings.push_back(arch);
    const char * a = strings.back().c_str();
    strings.push_back(name);
    const char * n = strings.back().c_str();
    tags.push_back({ a, n });
    return &tags.back();
}

const char * current_arch() {
    const auto & ts = thread_state();
    if (ts.has_model) {
        return ts.arch.c_str();
    }
    const auto * tag = g_model_tag.load(std::memory_order_acquire);
    return tag ? tag->arch : "";
}

const char * current_name() {
    const auto & ts = thread_state();
    if (ts.has_model) {
        return ts.name.c_str();
    }
    const auto * tag = g_model_tag.load(std::memory_order_acquire);
    return tag ? tag->name : "";
}

// is arch in the comma separated list
bool arch_in_list(const char * list, const std::string & arch) {
    const size_t n = arch.size();
    for (const char * p = list; *p;) {
        const char * end = strchr(p, ',');
        const size_t len = end ? (size_t) (end - p) : strlen(p);
        if (len == n && strncmp(p, arch.c_str(), n) == 0) {
            return true;
        }
        if (!end) {
            break;
        }
        p = end + 1;
    }
    return false;
}

// global state plus the model filter and the model rules for the model of this thread
bool entry_active(const ggml_hook_snapshot * snap, const ggml_hook_entry * e) {
    if (!e->k.models && snap->model_rules.empty()) {
        return e->enabled;
    }
    const std::string arch = current_arch();
    if (e->k.models && !arch_in_list(e->k.models, arch)) {
        return false;
    }
    bool enabled = e->enabled;
    for (const auto & rule : snap->model_rules) {
        if (rule.arch == arch && glob_match(rule.pattern.c_str(), e->k.name)) {
            enabled = rule.enable;
        }
    }
    return enabled;
}

// never destroyed: modules unregister from static destructors, in any order
ggml_hook_registry & registry() {
    static auto * r = new ggml_hook_registry();
    return *r;
}

// '*' matches any run of chars
bool glob_match(const char * p, const char * s) {
    if (*p == '\0') {
        return *s == '\0';
    }
    if (*p == '*') {
        return glob_match(p + 1, s) || (*s != '\0' && glob_match(p, s + 1));
    }
    return *s != '\0' && *p == *s && glob_match(p + 1, s + 1);
}

const char * backend_name(ggml_hook_backend b) {
    switch (b) {
        case GGML_HOOK_BACKEND_CPU:    return "CPU";
        case GGML_HOOK_BACKEND_CUDA:   return "CUDA";
        case GGML_HOOK_BACKEND_VULKAN: return "Vulkan";
        default:                       return "?";
    }
}

// GGML_HOOKS="none,+vk_*,-vk_slow,qwen35moe:+moe_*": rules applied in order over the defaults.
// "arch:rule" applies only while that model runs (GGUF general.architecture).
void parse_env(ggml_hook_registry & r) {
    const char * log = getenv("GGML_HOOKS_LOG");
    r.log = log && *log && strcmp(log, "0") != 0;

    const char * env = getenv("GGML_HOOKS");
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
        std::string arch;
        const size_t colon = tok.find(':');
        if (colon != std::string::npos) {
            arch = tok.substr(0, colon);
            tok  = tok.substr(colon + 1);
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
        r.rules.push_back({ arch, tok, enable });
    }
}

void apply_rules(const ggml_hook_registry & r, ggml_hook_entry & e) {
    e.enabled = e.k.enabled_by_default;
    for (const auto & rule : r.rules) {
        if (rule.arch.empty() && glob_match(rule.pattern.c_str(), e.k.name)) {
            e.enabled = rule.enable;
        }
    }
}

// a model rule can turn on a kernel that is off by default
bool maybe_enabled(const ggml_hook_registry & r, const ggml_hook_entry & e) {
    if (e.enabled) {
        return true;
    }
    for (const auto & rule : r.rules) {
        if (!rule.arch.empty() && rule.enable && glob_match(rule.pattern.c_str(), e.k.name)) {
            return true;
        }
    }
    return false;
}

// call with the mutex held
void publish(ggml_hook_registry & r) {
    auto * snap = new ggml_hook_snapshot();
    for (const auto & rule : r.rules) {
        if (!rule.arch.empty()) {
            snap->model_rules.push_back(rule);
        }
    }
    for (auto & e : r.entries) {
        if (e.removed || !maybe_enabled(r, e)) {
            continue;
        }
        auto & list = e.k.kind == GGML_HOOK_KIND_OP ? snap->ops[e.k.backend][e.k.op] : snap->fusions[e.k.backend][e.k.op];
        list.push_back(&e);
        snap->any[e.k.backend] = true;
    }
    for (int b = 0; b < GGML_HOOK_BACKEND_COUNT; ++b) {
        for (int op = 0; op < GGML_OP_COUNT; ++op) {
            for (auto * list : { &snap->ops[b][op], &snap->fusions[b][op] }) {
                std::stable_sort(list->begin(), list->end(), [](const ggml_hook_entry * a, const ggml_hook_entry * b) {
                    return a->k.priority > b->k.priority;
                });
            }
        }
    }
    // old snapshots are leaked on purpose: a backend thread may still read them
    r.snapshot.store(snap, std::memory_order_release);
}

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        auto & r = registry();
        std::lock_guard<std::mutex> lock(r.mutex);
        parse_env(r);
        for (auto & e : r.entries) {
            apply_rules(r, e);
        }
        publish(r);
    });
}

void load_plugins_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        const char * env = getenv("GGML_HOOKS_PLUGINS");
        if (!env) {
            return;
        }
#ifdef _WIN32
        const char sep = ';';
#else
        const char sep = ':';
#endif
        std::string s(env);
        size_t pos = 0;
        while (pos <= s.size()) {
            size_t end = s.find(sep, pos);
            if (end == std::string::npos) {
                end = s.size();
            }
            std::string path = s.substr(pos, end - pos);
            pos = end + 1;
            if (path.empty()) {
                continue;
            }
            // the plugin registers its kernels from its static constructors
#ifdef _WIN32
            void * handle = (void *) LoadLibraryA(path.c_str());
#else
            void * handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
            if (!handle) {
                GGML_LOG_ERROR("ggml-hooks: failed to load plugin %s\n", path.c_str());
            } else {
                GGML_LOG_INFO("ggml-hooks: loaded plugin %s\n", path.c_str());
            }
        }
    });
}

const ggml_hook_snapshot * snapshot() {
    init_once();
    load_plugins_once();
    return registry().snapshot.load(std::memory_order_acquire);
}

} // namespace

static bool kernel_valid(const ggml_hook_kernel * k) {
    if (!k->name || k->backend < 0 || k->backend >= GGML_HOOK_BACKEND_COUNT || k->op < 0 || k->op >= GGML_OP_COUNT) {
        return false;
    }
    if ((k->kind == GGML_HOOK_KIND_OP && !k->supports) || (k->kind == GGML_HOOK_KIND_FUSION && !k->match)) {
        return false;
    }
    const ggml_hook_compute & c = k->compute;
    const bool op = k->kind == GGML_HOOK_KIND_OP;
    switch (k->backend) {
        case GGML_HOOK_BACKEND_CPU:    return op && c.cpu_op;
        case GGML_HOOK_BACKEND_CUDA:   return op ? c.cuda_op != nullptr : c.cuda_fusion != nullptr;
        case GGML_HOOK_BACKEND_VULKAN: return op ? c.vk_op   != nullptr : c.vk_fusion   != nullptr;
        default:                       return false;
    }
}

void ggml_hook_register_kernel(const ggml_hook_kernel * kernel, const void * module) {
    if (kernel->abi != GGML_HOOK_ABI_VERSION) {
        GGML_LOG_ERROR("ggml-hooks: kernel %s has ABI %d, expected %d: skipped\n", kernel->name, kernel->abi, GGML_HOOK_ABI_VERSION);
        return;
    }
    if (!kernel_valid(kernel)) {
        GGML_LOG_ERROR("ggml-hooks: kernel %s is not valid: skipped\n", kernel->name ? kernel->name : "(null)");
        return;
    }

    auto & r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    for (const auto & e : r.entries) {
        if (e.removed) {
            continue;
        }
        // static builds: every registrar sees the descriptors of the whole executable
        if (e.desc == kernel) {
            return;
        }
        // the same name may come from several modules (GGML_BACKEND_DL CPU variants): only the loaded one stays
        if (e.module == module && e.name == kernel->name) {
            GGML_LOG_ERROR("ggml-hooks: kernel %s registered twice: skipped\n", kernel->name);
            return;
        }
    }
    r.entries.emplace_back();
    ggml_hook_entry & e = r.entries.back();
    e.k      = *kernel;
    e.name   = kernel->name;
    e.models = kernel->models ? kernel->models : "";
    e.module = module;
    e.desc   = kernel;
    e.k.name     = e.name.c_str();
    e.k.models   = kernel->models ? e.models.c_str() : nullptr;
    e.k.internal = &e;
    apply_rules(r, e);
    // before init_once, the rules are not parsed yet and init_once publishes
    if (r.snapshot.load(std::memory_order_relaxed)) {
        publish(r);
    }
}


void ggml_hook_unregister_module(const void * module) {
    if (!module) {
        return;
    }
    auto & r = registry();
    {
        std::lock_guard<std::mutex> lock(r.mutex);
        bool changed = false;
        for (auto & e : r.entries) {
            if (e.module == module && !e.removed) {
                e.removed = true;
                changed   = true;
            }
        }
        if (changed && r.snapshot.load(std::memory_order_relaxed)) {
            publish(r);
        }
        auto * old = r.params.load(std::memory_order_relaxed);
        if (old) {
            auto * p = new ggml_hook_params(*old);
            p->providers.erase(std::remove_if(p->providers.begin(), p->providers.end(), [&](const ggml_hook_param_provider & pp) { return pp.module == module; }),
                               p->providers.end());
            r.params.store(p, std::memory_order_release);
        }
    }
    ggml_hook_wrap_unregister_module(module);
}

bool ggml_hook_has_kernels(ggml_hook_backend backend) {
    return snapshot()->any[backend];
}

const ggml_hook_kernel * ggml_hook_find_op(const ggml_hook_dev * dev, const ggml_tensor * node) {
    const auto * snap = snapshot();
    const auto & list = snap->ops[dev->backend][node->op];
    for (const ggml_hook_entry * e : list) {
        if (entry_active(snap, e) && e->k.supports(dev, node)) {
            return &e->k;
        }
    }
    return nullptr;
}

const ggml_hook_kernel * ggml_hook_find_fusion(const ggml_hook_dev * dev, const ggml_cgraph * cgraph, int i, int * n_nodes) {
    const auto * snap = snapshot();
    const auto & list = snap->fusions[dev->backend][cgraph->nodes[i]->op];
    for (const ggml_hook_entry * e : list) {
        if (!entry_active(snap, e)) {
            continue;
        }
        const int n = e->k.match(dev, cgraph, i);
        if (n >= 2 && i + n <= cgraph->n_nodes) {
            *n_nodes = n;
            return &e->k;
        }
    }
    return nullptr;
}

void ggml_hook_kernel_used(const ggml_hook_kernel * kernel, const ggml_tensor * node) {
    auto * e = (ggml_hook_entry *) kernel->internal;
    e->hits.fetch_add(1, std::memory_order_relaxed);
    if (registry().log && !e->logged.exchange(true)) {
        GGML_LOG_INFO("ggml-hooks: %s kernel %s used for %s (%s)\n", backend_name(kernel->backend), kernel->name, ggml_op_desc(node), node->name);
    }
}

void ggml_hook_set_model(const char * arch, const char * name) {
    auto & ts = thread_state();
    ts.arch      = arch ? arch : "";
    ts.name      = name ? name : "";
    ts.has_model = !ts.arch.empty() || !ts.name.empty();
    g_model_tag.store(ts.has_model ? intern_tag(ts.arch, ts.name) : nullptr, std::memory_order_release);
}

const char * ggml_hook_model_arch(void) {
    return current_arch();
}

const char * ggml_hook_model_name(void) {
    return current_name();
}

void ggml_hook_mark_computed(const ggml_tensor * node) {
    thread_state().computed.push_back(node);
}

bool ggml_hook_take_computed(const ggml_tensor * node) {
    auto & c = thread_state().computed;
    if (c.empty()) {
        return false;
    }
    auto it = std::find(c.begin(), c.end(), node);
    if (it == c.end()) {
        return false;
    }
    c.erase(it);
    return true;
}

void ggml_hook_clear_computed(void) {
    thread_state().computed.clear();
}

void ggml_hook_register_param(const ggml_hook_param * param, const void * module) {
    auto & r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    auto * old = r.params.load(std::memory_order_relaxed);
    if (old) {
        for (const auto & pp : old->providers) {
            if (pp.desc == param) {
                return;
            }
        }
    }
    auto * p = old ? new ggml_hook_params(*old) : new ggml_hook_params();
    p->providers.push_back({ param->name, param->fn, module, param });
    // the old list is leaked on purpose, like the snapshots
    r.params.store(p, std::memory_order_release);
}

int64_t ggml_hook_param_value(const char * name, const ggml_hook_dev * dev, const ggml_tensor * node, int64_t default_value) {
    const auto * p = registry().params.load(std::memory_order_acquire);
    if (!p) {
        return default_value;
    }
    for (const auto & pp : p->providers) {
        int64_t value = default_value;
        if (pp.name == name && pp.fn(dev, node, &value)) {
            return value;
        }
    }
    return default_value;
}

// called by the generated registrar of each library, in the order of hooks/cmake/hooks-link.cmake
void ggml_hook_register_section(int kind, const void * const * begin, const void * const * end, const void * module) {
    for (const void * const * p = begin; p < end; ++p) {
        switch (kind) {
            case 0: ggml_hook_register_driver((const ggml_hook_driver *) *p, module); break;
            case 1: ggml_hook_register_backend_ext((const ggml_hook_backend_ext *) *p, module); break;
            case 2: ggml_hook_register_param((const ggml_hook_param *) *p, module); break;
            case 3: ggml_hook_register_kernel((const ggml_hook_kernel *) *p, module); break;
            default: GGML_LOG_ERROR("ggml-hooks: unknown section kind %d\n", kind); return;
        }
    }
}

int ggml_hook_kernel_set_enabled(const char * pattern, bool enabled) {
    init_once();
    auto & r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    r.rules.push_back({ "", pattern, enabled });
    int n = 0;
    for (auto & e : r.entries) {
        const bool before = e.enabled;
        apply_rules(r, e);
        n += before != e.enabled;
    }
    publish(r);
    return n;
}

uint64_t ggml_hook_kernel_hits(const char * name) {
    auto & r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    uint64_t hits = 0;
    for (const auto & e : r.entries) {
        if (!e.removed && e.name == name) {
            hits += e.hits.load(std::memory_order_relaxed);
        }
    }
    return hits;
}

void ggml_hook_kernel_reset_hits(void) {
    auto & r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    for (auto & e : r.entries) {
        e.hits.store(0, std::memory_order_relaxed);
    }
}

void ggml_hook_kernel_print(void) {
    init_once();
    auto & r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    GGML_LOG_INFO("ggml-hooks: kernels\n");
    for (const auto & e : r.entries) {
        if (e.removed) {
            continue;
        }
        GGML_LOG_INFO("ggml-hooks:   %-8s %-6s %-12s %-40s prio %4d %s hits %llu\n", backend_name(e.k.backend), e.k.kind == GGML_HOOK_KIND_OP ? "op" : "fusion",
            ggml_op_name(e.k.op), e.k.name, e.k.priority, e.enabled ? "on " : "off", (unsigned long long) e.hits.load());
    }
}
