#pragma once

// libllama hooks: models, graph passes, memory and life time callbacks that live outside the upstream sources.
// Compiled into libllama: hooks use the internal C++ API (llama_model, llm_graph_result, ...).
// Every callback is connected by link-time wrapping of an upstream function (hooks/llama/llama-hooks.cmake),
// no upstream file has a hook line. See hooks/README.md.

#include "ggml.h"
#include "llama.h"
#include "ggml-hooks-link.h"
#include "llama-hook-points.h"

#include <cstdint>
#include <string>
#include <vector>

struct llama_model;
struct llama_model_loader;
struct llama_context;
struct llama_memory_params;
struct llama_cparams;
struct llama_memory_i;
struct llm_graph_params;
class  llm_graph_result;
struct LLM_TN_IMPL;

enum llama_hook_memory_op_type {
    LLAMA_HOOK_MEMORY_CLEAR,
    LLAMA_HOOK_MEMORY_SEQ_RM,
    LLAMA_HOOK_MEMORY_SEQ_CP,
    LLAMA_HOOK_MEMORY_SEQ_KEEP,
    LLAMA_HOOK_MEMORY_SEQ_ADD,
    LLAMA_HOOK_MEMORY_SEQ_DIV,
};

// a call of the public llama_memory_* API, after upstream ran it
struct llama_hook_memory_op {
    llama_hook_memory_op_type type;
    llama_seq_id              seq_id;
    llama_seq_id              seq_id_dst; // SEQ_CP
    llama_pos                 p0;
    llama_pos                 p1;
    llama_pos                 shift;  // SEQ_ADD
    int                       d;      // SEQ_DIV
    bool                      data;   // CLEAR
    bool                      result; // SEQ_RM
};

// every field is optional
struct llama_hook {
    const char * name     = nullptr;
    const char * models   = nullptr; // nullptr: every model, else comma separated GGUF archs ("qwen35moe,llama")
    int          priority = 0;       // higher runs first

    // before upstream loads a model or creates a context: change the params. paths is empty for
    // llama_model_load_from_file_ptr and llama_model_init_from_user (load_model is not called for the latter)
    void (*model_params)(const std::vector<std::string> & paths, llama_model_params & params) = nullptr;
    void (*context_params)(const llama_model & model, llama_context_params & params) = nullptr;
    // load a model instead of upstream (another file format). nullptr: upstream loads it
    llama_model * (*load_model)(const std::vector<std::string> & paths, const llama_model_params & params) = nullptr;
    // the loader opened a GGUF: return a model (it can subclass an upstream model, it must set model->arch), nullptr for upstream
    llama_model * (*create_model)(llama_model_loader & ml, const llama_model_params & params) = nullptr;
    // the loader created a weight tensor (metadata only, no data yet)
    void (*tensor_created)(llama_model_loader & ml, const LLM_TN_IMPL & tn, ggml_tensor * t, int flags) = nullptr;
    // memory of a context: return a memory, nullptr for upstream
    llama_memory_i * (*create_memory)(const llama_model & model, const llama_memory_params & params, const llama_cparams & cparams) = nullptr;
    // graph pass: after upstream built a graph and set its outputs. Tensors go in res.get_ctx().
    void (*graph_pass)(const llama_model & model, const llm_graph_params & params, llm_graph_result & res) = nullptr;
    // life time: after a context is created, before it is freed, before a model is freed
    void (*context_init)(llama_context & ctx) = nullptr;
    void (*context_free)(llama_context & ctx) = nullptr;
    void (*model_free)(llama_model & model) = nullptr;
    // around llama_process, llama_decode and llama_encode. type: llama_process_type
    void (*process_begin)(llama_context & ctx, int type) = nullptr;
    void (*process_end)(llama_context & ctx, int type, int32_t ret) = nullptr;
    // the users changed the memory with the public API
    void (*memory_op)(llama_memory_t mem, const llama_hook_memory_op & op) = nullptr;

    // hook points in src/ (llama-hook-points.h, hooks/HOOKS.txt)
    void (*kv_event)(const llama_kv_cache & kv, const llama_hook_kv_ev & ev) = nullptr;
    void (*sched_reserve)(llama_context & ctx, llama_memory_context_i * mctx, uint32_t n_seqs, uint32_t n_tokens) = nullptr;
    void (*resolve_fused_ops)(llama_context & ctx, const llama_memory_context_i * mctx, uint32_t n_seqs) = nullptr;
    void (*ubatch_done)(llama_context & ctx, const llama_ubatch & ubatch, ggml_status status) = nullptr;
    uint32_t (*graph_extra_nodes)(const llama_context & ctx) = nullptr;
    // a name for a tensor id that upstream does not know: return true and set name (with tn.bid, tn.xid, tn.suffix)
    bool (*tensor_name)(const LLM_TN_IMPL & tn, std::string & name) = nullptr;
    const llm_tensor_info * (*tensor_info)(llm_tensor tensor) = nullptr;
    ggml_type (*kv_type)(const llama_model & model, const char * role, ggml_type type) = nullptr;
    void (*mmap_ranges)(const llama_model_loader & ml, uint32_t idx, llama_mmap::ranges & ranges) = nullptr;
    ggml_type (*quant_type)(const llama_model_quantize_params * params, const ggml_tensor * tensor, ggml_type type) = nullptr;
};

void llama_hook_register(const llama_hook & hook);

// The context of a graph: graph passes and memories get llm_graph_params, which has no context pointer.
// The key is stable for the life of the context, also while the context constructor reserves its graphs.
// llama_hooks_context_of returns nullptr until the context is created (during the reserve in its constructor).
const void *    llama_hooks_context_key(const llm_graph_params & params);
const void *    llama_hooks_context_key(const llama_context & ctx);
llama_context * llama_hooks_context_of(const llm_graph_params & params);

// the context whose llama_process/decode/encode runs on this thread, nullptr outside
llama_context * llama_hooks_current_context();

// Replace every use of `from` by `to` in the nodes of gf (srcs and view sources), then rebuild the node list in
// the same order. Tensors created by the pass are added where first used. `from` stays only if still used.
// `to` gets the output flag of `from`. llm_graph_result keeps its own pointers (t_logits, t_embd, ...): a pass that
// replaces one of them updates it too.
void llama_hook_graph_replace(ggml_cgraph * gf, ggml_tensor * from, ggml_tensor * to);

// enabled or not with LLAMA_HOOKS="none,+name,-name" (same syntax as GGML_HOOKS, without the arch rules)
int  llama_hooks_set_enabled(const char * pattern, bool enabled);
long llama_hooks_hits(const char * name);

// register at load time
#define LLAMA_HOOK_REGISTER(...) \
    static const bool GGML_HOOK_CONCAT(llama_hook_registered_, __LINE__) = [] { llama_hook_register(__VA_ARGS__); return true; }()
