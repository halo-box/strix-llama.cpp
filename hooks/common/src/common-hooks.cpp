#include "common-hooks.h"

#include <cstring>
#include <mutex>
#include <string>

namespace {

struct hooks_registry {
    std::mutex                mutex;
    std::vector<common_hooks> hooks;
};

hooks_registry & registry() {
    static auto * r = new hooks_registry();
    return *r;
}

std::vector<common_hooks> all() {
    auto & r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    return r.hooks;
}

} // namespace

void common_hooks_register(const common_hooks & hooks) {
    auto & r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    r.hooks.push_back(hooks);
}

common_arg * common_hooks_find_arg(common_params_context & ctx, const char * name) {
    for (auto & opt : ctx.options) {
        for (const auto & arg : opt.args) {
            if (strcmp(arg, name) == 0) {
                return &opt;
            }
        }
    }
    return nullptr;
}

void common_hooks_args(common_params_context & ctx, const std::function<void(common_arg)> & add) {
    for (const auto & h : all()) {
        if (h.args) {
            h.args(ctx, add);
        }
    }
}

void common_hooks_quant_ftypes(const common_hooks_add_ftype_fn & add) {
    for (const auto & h : all()) {
        if (h.quant_ftypes) {
            h.quant_ftypes(add);
        }
    }
}

// link-time wrappers, see hooks/llama/llama-hooks.cmake

extern "C" void __real__Z24common_speculative_draftP18common_speculative(common_speculative * spec);
extern "C" __attribute__((visibility("default"))) void __wrap__Z24common_speculative_draftP18common_speculative(common_speculative * spec) {
    for (const auto & h : all()) {
        if (h.spec_draft) {
            h.spec_draft(spec);
        }
    }
    __real__Z24common_speculative_draftP18common_speculative(spec);
}

extern "C" void __real__Z25common_speculative_acceptP18common_speculativeit(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted);
extern "C" __attribute__((visibility("default"))) void __wrap__Z25common_speculative_acceptP18common_speculativeit(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    __real__Z25common_speculative_acceptP18common_speculativeit(spec, seq_id, n_accepted);
    for (const auto & h : all()) {
        if (h.spec_accept) {
            h.spec_accept(spec, seq_id, n_accepted);
        }
    }
}
extern "C" common_sampler * __real__Z19common_sampler_initPK11llama_modelR22common_params_sampling(const llama_model * model, common_params_sampling & params);
extern "C" __attribute__((visibility("default"))) common_sampler * __wrap__Z19common_sampler_initPK11llama_modelR22common_params_sampling(
        const llama_model * model, common_params_sampling & params) {
    common_sampler * smpl = __real__Z19common_sampler_initPK11llama_modelR22common_params_sampling(model, params);
    if (smpl) {
        for (const auto & h : all()) {
            if (h.sampler_init) {
                h.sampler_init(model, params, smpl);
            }
        }
    }
    return smpl;
}
