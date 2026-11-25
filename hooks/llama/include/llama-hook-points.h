#pragma once

// libllama hook points: called from the HOOK(...) lines in src/ (listed in hooks/HOOKS.txt).
// The src/ files include this header by its path in the repo. Without GGML_HOOKS (hooks not configured), every hook
// point is an inline no-op and the build is pure upstream.
// The callbacks are fields of llama_hook (llama-hooks.h).

#include "../../../src/llama-arch.h"
#include "../../../src/llama-mmap.h"

#include "ggml.h"

#include <cstdint>
#include <string>

struct llama_context;
struct llama_kv_cache;
struct llama_memory_context_i;
struct llama_model;
struct llama_model_loader;
struct llama_model_quantize_params;
struct llama_ubatch;
struct LLM_TN_IMPL;

enum llama_hook_kv_event_type {
    LLAMA_HOOK_KV_CLEAR,
    LLAMA_HOOK_KV_SEQ_RM,
    LLAMA_HOOK_KV_SEQ_CP,
    LLAMA_HOOK_KV_SEQ_KEEP,
    LLAMA_HOOK_KV_SEQ_ADD,
    LLAMA_HOOK_KV_SEQ_DIV,
    LLAMA_HOOK_KV_APPLY_UBATCH, // after the cells of a ubatch are written
    LLAMA_HOOK_KV_STREAM_COPY,  // update() copies streams (stream_copy_info)
};

// a KV cache operation, before it runs (APPLY_UBATCH: after). Every llama_kv_cache gets them, the inner caches of
// iSWA, hybrid and indexer memories too.
struct llama_hook_kv_ev {
    llama_hook_kv_event_type type;
    int32_t                  seq_id     = -1;
    int32_t                  seq_id_dst = -1;
    int32_t                  p0         = -1;
    int32_t                  p1         = -1;
    int32_t                  shift      = 0;
    int                      d          = 1;
    bool                     data       = false;
    const void *             sinfo      = nullptr; // APPLY_UBATCH: const llama_kv_cache::slot_info *
    const llama_ubatch *     ubatch     = nullptr; // APPLY_UBATCH
    const void *             sc_info    = nullptr; // STREAM_COPY: const llama_kv_cache::stream_copy_info *
};

#ifndef GGML_HOOKS

inline void        llama_hook_kv_event(const llama_kv_cache *, const llama_hook_kv_ev &) {}
inline void        llama_hook_sched_reserve(llama_context &, llama_memory_context_i *, uint32_t, uint32_t) {}
inline void        llama_hook_resolve_fused_ops(llama_context &, const llama_memory_context_i *, uint32_t) {}
inline void        llama_hook_ubatch_done(llama_context &, const llama_ubatch &, ggml_status) {}
inline uint32_t    llama_hook_graph_extra_nodes(const llama_context &) { return 0; }
inline bool        llama_hook_tensor_str(const LLM_TN_IMPL &, std::string &) { return false; }
inline const llm_tensor_info * llama_hook_tensor_info(llm_tensor) { return nullptr; }
inline ggml_type   llama_hook_kv_type(const llama_model &, const char *, ggml_type type) { return type; }
inline const llama_mmap::ranges & llama_hook_mmap_ranges(const llama_model_loader &, uint32_t, const llama_mmap::ranges & ranges) { return ranges; }
inline ggml_type   llama_hook_quant_type(const llama_model_quantize_params *, const ggml_tensor *, ggml_type type) { return type; }

#else

// HOOK(kv-*) in llama-kv-cache.cpp
void llama_hook_kv_event(const llama_kv_cache * kv, const llama_hook_kv_ev & ev);
// HOOK(sched-reserve): sched_reserve, before the worst-case graphs are reserved
void llama_hook_sched_reserve(llama_context & ctx, llama_memory_context_i * mctx, uint32_t n_seqs, uint32_t n_tokens);
// HOOK(fused-ops): end of resolve_fused_ops, to resolve the fused ops of the hooks the same way
void llama_hook_resolve_fused_ops(llama_context & ctx, const llama_memory_context_i * mctx, uint32_t n_seqs);
// HOOK(ubatch-done): process_ubatch, after the graph of a ubatch is computed
void llama_hook_ubatch_done(llama_context & ctx, const llama_ubatch & ubatch, ggml_status status);
// HOOK(graph-max-nodes): nodes the graph passes add, on top of the upstream estimate
uint32_t llama_hook_graph_extra_nodes(const llama_context & ctx);
// HOOK(tensor-name), HOOK(tensor-info): tensor ids that upstream does not know (new tensors of hook models)
bool llama_hook_tensor_str(const LLM_TN_IMPL & tn, std::string & name);
const llm_tensor_info * llama_hook_tensor_info(llm_tensor tensor);
// HOOK(kv-type-*): type of the keys of the indexer caches. role: "dsa-lid", "dsv4-lid"
ggml_type llama_hook_kv_type(const llama_model & model, const char * role, ggml_type type);
// HOOK(mmap-ranges): file ranges that the mmap does not prefetch (upstream: the lazy ranges)
const llama_mmap::ranges & llama_hook_mmap_ranges(const llama_model_loader & ml, uint32_t idx, const llama_mmap::ranges & ranges);
// HOOK(quant-type): end of llama_tensor_get_type, the type of a tensor when quantizing
ggml_type llama_hook_quant_type(const llama_model_quantize_params * params, const ggml_tensor * tensor, ggml_type type);

#endif
