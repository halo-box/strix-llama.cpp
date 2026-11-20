#pragma once

// common hooks: CLI arguments and sampler creation, defined outside common/.
// Hook files go in hooks/common/args/. See hooks/README.md.

#include "arg.h"
#include "sampling.h"

#include <functional>
#include <vector>

// quantize: adds a quantization type to the list of llama-quantize (name, ftype, description)
typedef std::function<void(const char * name, llama_ftype ftype, const char * desc)> common_hooks_add_ftype_fn;

#ifndef GGML_HOOKS
// hooks not configured: the hook lines of common and tools do nothing
inline void common_hooks_args(common_params_context &, const std::function<void(common_arg)> &) {}
inline void common_hooks_quant_ftypes(const common_hooks_add_ftype_fn &) {}
#else

struct common_speculative;

#include "ggml-hooks-link.h"

struct common_hooks {
    const char * name = nullptr;

    // CLI arguments (HOOK(common-args), at the end of common_params_parser_init).
    // add: adds an argument, filtered by example like the upstream ones (common_arg::set_examples).
    // ctx.options: the upstream arguments, a hook can change one (common_hooks_find_arg) instead of adding a duplicate.
    // common_params has no room for new fields: handlers store their values in the hook file's own state.
    void (*args)(common_params_context & ctx, const std::function<void(common_arg)> & add) = nullptr;

    // a sampler was created (wraps common_sampler_init): add samplers to its chain, keep state for it
    void (*sampler_init)(const llama_model * model, const common_params_sampling & params, common_sampler * smpl) = nullptr;

    // before common_speculative_draft: change the draft params of each sequence (common_speculative_get_draft_params)
    void (*spec_draft)(common_speculative * spec) = nullptr;
    // after common_speculative_accept
    void (*spec_accept)(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) = nullptr;

    // llama-quantize: add quantization types (HOOK(quantize-ftypes))
    void (*quant_ftypes)(const common_hooks_add_ftype_fn & add) = nullptr;
};

void common_hooks_register(const common_hooks & hooks);

// the upstream argument with this name, nullptr if there is none
common_arg * common_hooks_find_arg(common_params_context & ctx, const char * name);

// called from the HOOK lines of arg.cpp and llama-quantize
void common_hooks_args(common_params_context & ctx, const std::function<void(common_arg)> & add);
void common_hooks_quant_ftypes(const common_hooks_add_ftype_fn & add);

#define COMMON_HOOK_REGISTER(...) \
    static const bool GGML_HOOK_CONCAT(common_hooks_registered_, __LINE__) = [] { common_hooks_register(__VA_ARGS__); return true; }()

#endif
