#pragma once

// ggml hook points: the hook types and the scheduler hook. Included at the end of ggml.h (HOOK(types)), C and C++.
//
// A type has an id >= GGML_TYPE_HOOKS_FIRST and an entry in GGML_HOOK_TYPES_LIST below. The hook points use it:
// - ggml.c type_traits (HOOK(type-traits)): name, block, size, to_float, from_float_ref
// - ggml-cpu.c type_traits_cpu (HOOK(type-traits-cpu)): from_float, vec_dot, vec_dot_type, nrows
// - ggml_quantize_chunk (HOOK(type-quantize)) and ggml_validate_row_data (HOOK(type-validate))
// GGML_TYPE_COUNT becomes GGML_TYPE_HOOKS_END, so every [GGML_TYPE_COUNT] table of upstream grows.
// Devices without a hook kernel for a node with a hook type do not support it (backend wrapper).
//
// Without GGML_HOOKS (hooks not configured), the hook points expand to nothing.

#ifdef GGML_HOOKS

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define GGML_TYPE_HOOKS_FIRST 100
#define GGML_TYPE_HOOKS_END   160

// the upstream count, before GGML_TYPE_COUNT is redefined
enum { GGML_TYPE_COUNT_UPSTREAM = GGML_TYPE_COUNT };
#define GGML_TYPE_COUNT ((enum ggml_type) GGML_TYPE_HOOKS_END)

#ifdef __cplusplus
static_assert(GGML_TYPE_COUNT_UPSTREAM <= GGML_TYPE_HOOKS_FIRST, "upstream ggml types reach the hook type ids");
#else
_Static_assert(GGML_TYPE_COUNT_UPSTREAM <= GGML_TYPE_HOOKS_FIRST, "upstream ggml types reach the hook type ids");
#endif

// X(NAME, ID, type_name, blck_size, type_size, vec_dot_type, cpu_nrows)
// The block layouts are in hooks/ggml/types/ggml-common-ext.h. Sizes are literals here, because ggml.h cannot see
// the block structs; hooks/ggml/types/types.c checks them against sizeof.
// The codecs of each type are called through wrappers named ggml_hook_type_<type_name>_<function> (types.c, cpu/types-cpu.c).
#define GGML_HOOK_TYPES_LIST(X) \
    X(Q4_0_ROCMFP4,      100, q4_0_rocmfp4,       32, 18, GGML_TYPE_Q8_0, 1) \
    X(Q4_0_ROCMFP4_FAST, 101, q4_0_rocmfp4_fast,  32, 17, GGML_TYPE_Q8_0, 1) \
    X(Q6_0_ROCMFPX,      102, q6_0_rocmfpx,       32, 26, GGML_TYPE_Q8_0, 1) \
    X(Q8_0_ROCMFPX,      103, q8_0_rocmfpx,       32, 33, GGML_TYPE_Q8_0, 1) \
    X(Q3_0_ROCMFPX,      104, q3_0_rocmfpx,       32, 14, GGML_TYPE_Q8_0, 1) \
    X(Q2_0_ROCMFPX,      107, q2_0_rocmfpx,       32, 10, GGML_TYPE_Q8_0, 1) \
    X(PTQ1_0,            143, ptq1_0,            128, 28, GGML_TYPE_Q8_0, 1) \
    X(TQ2_T,             144, tq2_t,             128, 34, GGML_TYPE_Q8_0, 1) \
    X(TQK6,              145, tqk6,              128, 26, GGML_TYPE_Q8_0, 1) \
    X(TQK7,              146, tqk7,              128, 30, GGML_TYPE_Q8_0, 1) \
    X(TQP5,              147, tqp5,              128, 42, GGML_TYPE_Q8_0, 1) \
    X(TQP6,              148, tqp6,              128, 50, GGML_TYPE_Q8_0, 1) \
    X(TQP7,              149, tqp7,              128, 58, GGML_TYPE_Q8_0, 1) \
    X(Q8_0_HOOK_EXAMPLE, 159, q8_0_hook_example,  32, 34, GGML_TYPE_Q8_0, 1)

// The type and ftype ids of the fork: its enum lines of ggml/include/ggml.h (upstream 9048bad8d, fork 1fe05fa45),
// as defines. GGUF files store these values. GGML_HOOK_TYPES_LIST checks each type id.
// ROCmFPx experimental AMD-native formats. These are kept in a high ID
// range so upstream type IDs stay free for future ggml types.
#define GGML_TYPE_Q4_0_ROCMFP4                       ((enum ggml_type) 100) // ROCmFP4 experimental UE4M3 scales + packed AMD FP4 blocks
#define GGML_TYPE_Q4_0_ROCMFP4_FAST                  ((enum ggml_type) 101) // ROCmFP4 single-scale speed layout
#define GGML_TYPE_Q6_0_ROCMFPX                       ((enum ggml_type) 102) // ROCmFPx experimental 6-bit UE4M3-scale reference layout
#define GGML_TYPE_Q8_0_ROCMFPX                       ((enum ggml_type) 103) // ROCmFPx experimental 8-bit UE4M3-scale reference layout
#define GGML_TYPE_Q3_0_ROCMFPX                       ((enum ggml_type) 104) // ROCmFPx experimental 3-bit UE4M3-scale reference layout
// 105, 106 are TURBO3_0 / TURBO4_0 and 108 is Q4_0_ROCMI4 in charlie12345/ROCmFPX;
// left unassigned here so a GGUF from that tree is rejected rather than misread.
// Note ciru-ai/ROCmFPX diverges: it puts Q7_0_ROCMFPX at 107 and Q2_0_ROCMFPX at 108.
#define GGML_TYPE_Q2_0_ROCMFPX                       ((enum ggml_type) 107) // ROCmFPx experimental 2-bit S40 codebook + dual UE4M3 scales
// Prism-private ternary at group size 128, ported from PrismML-Eng/llama.cpp
// (Ternary-Bonsai GGUFs). High id so it slots above upstream types; type_traits
// is sized to COUNT with 44..99 and 108..142 unused (100..107 are ROCmFPx). PQ2_0 (142) is deliberately not ported:
// it has no Vulkan kernels upstream of us either.
#define GGML_TYPE_PTQ1_0                             ((enum ggml_type) 143)
// Trellis-coded 2.125 bpw (QTIP-style bitshift trellis, L=16 K=8 V=4, hashed
// 2-D "Hyb" codebook), group 128.
#define GGML_TYPE_TQ2_T                              ((enum ggml_type) 144)
// Bit-packed trellis siblings of TQ2_T (K = 6 / 7 new bits per 4-weight step,
// same codebook): 1.625 / 1.875 bpw, group 128.
#define GGML_TYPE_TQK6                               ((enum ggml_type) 145)
#define GGML_TYPE_TQK7                               ((enum ggml_type) 146)
// Pair trellis (V=2): 64 steps of 2 weights per 128-weight block, K = 5 / 6 / 7 new bits
// per step, TQ2_T codebook (first pair of each state): 2.625 / 3.125 / 3.625 bpw.
#define GGML_TYPE_TQP5                               ((enum ggml_type) 147)
#define GGML_TYPE_TQP6                               ((enum ggml_type) 148)
#define GGML_TYPE_TQP7                               ((enum ggml_type) 149)
#define GGML_TYPE_Q8_0_HOOK_EXAMPLE                  ((enum ggml_type) 159) // test type of the hooks (a copy of q8_0)

#define GGML_FTYPE_MOSTLY_Q4_0_ROCMFP4               ((enum ggml_ftype) 100) // except 1d tensors
#define GGML_FTYPE_MOSTLY_Q4_0_ROCMFP4_LEAN          ((enum ggml_ftype) 101) // ROCmFP4 with Q5_K token embeddings
#define GGML_FTYPE_MOSTLY_Q4_0_ROCMFP4_COHERENT      ((enum ggml_ftype) 102) // ROCmFP4 with Q6_K token embeddings
#define GGML_FTYPE_MOSTLY_Q4_0_ROCMFP4_FAST          ((enum ggml_ftype) 103) // ROCmFP4 single-scale speed layout
#define GGML_FTYPE_MOSTLY_Q4_0_ROCMFP4_FAST_COHERENT ((enum ggml_ftype) 104) // ROCmFP4 fast with Q6_K token embeddings
#define GGML_FTYPE_MOSTLY_Q4_0_ROCMFP4_STRIX         ((enum ggml_ftype) 105) // ROCmFP4 Strix Halo quality/speed recipe
#define GGML_FTYPE_MOSTLY_Q4_0_ROCMFP4_STRIX_LEAN    ((enum ggml_ftype) 106) // ROCmFP4 Strix Halo size-biased K/V recipe
#define GGML_FTYPE_MOSTLY_Q6_0_ROCMFPX               ((enum ggml_ftype) 110) // ROCmFPx experimental 6-bit reference layout
#define GGML_FTYPE_MOSTLY_Q8_0_ROCMFPX               ((enum ggml_ftype) 111) // ROCmFPx experimental 8-bit reference layout
#define GGML_FTYPE_MOSTLY_Q3_0_ROCMFPX               ((enum ggml_ftype) 112) // ROCmFPx experimental 3-bit reference layout
#define GGML_FTYPE_MOSTLY_Q2_0_ROCMFPX               ((enum ggml_ftype) 113) // ROCmFPx experimental 2-bit S40 codebook layout
#define GGML_FTYPE_MOSTLY_PTQ1_0                     ((enum ggml_ftype) 129) // except 1d tensors (Prism-private group-128 ternary)
#define GGML_FTYPE_MOSTLY_TQ2_T                      ((enum ggml_ftype) 130) // except 1d tensors (trellis-coded, group 128)
#define GGML_FTYPE_MOSTLY_TQK6                       ((enum ggml_ftype) 131) // except 1d tensors (trellis-coded K=6, group 128)
#define GGML_FTYPE_MOSTLY_TQK7                       ((enum ggml_ftype) 132) // except 1d tensors (trellis-coded K=7, group 128)
#define GGML_FTYPE_MOSTLY_TQP5                       ((enum ggml_ftype) 133) // except 1d tensors (pair trellis K=5, group 128)
#define GGML_FTYPE_MOSTLY_TQP6                       ((enum ggml_ftype) 134) // except 1d tensors (pair trellis K=6, group 128)
#define GGML_FTYPE_MOSTLY_TQP7                       ((enum ggml_ftype) 135) // except 1d tensors (pair trellis K=7, group 128)

#define GGML_HOOK_TYPE_ID(NAME, ID, ...) _Static_assert((int) GGML_TYPE_##NAME == ID, "id of " #NAME);
#ifndef __cplusplus
GGML_HOOK_TYPES_LIST(GGML_HOOK_TYPE_ID)
#endif
#undef GGML_HOOK_TYPE_ID

#ifdef __cplusplus
extern "C" {
#endif
// wrappers of the codecs: ggml-base (types.c) and each CPU variant (cpu/types-cpu.c)
#define GGML_HOOK_TYPE_FNS(NAME, ID, TNAME, ...) \
    GGML_API void ggml_hook_type_##TNAME##_to_float(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k); \
    GGML_API void ggml_hook_type_##TNAME##_from_float_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k); \
    void ggml_hook_type_##TNAME##_cpu_from_float(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k); \
    void ggml_hook_type_##TNAME##_cpu_vec_dot(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x, size_t bx, \
                                              const void * GGML_RESTRICT y, size_t by, int nrc);
GGML_HOOK_TYPES_LIST(GGML_HOOK_TYPE_FNS)
#undef GGML_HOOK_TYPE_FNS
#ifdef __cplusplus
}
#endif

static inline bool ggml_hook_type_is_hook(enum ggml_type type) {
    return (int) type >= GGML_TYPE_HOOKS_FIRST;
}

#ifdef __cplusplus
extern "C" {
#endif
// implemented in hooks/ggml/types/types.c (ggml-base). Return false if the type is not a hook type.
GGML_API bool ggml_hook_type_quantize(enum ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix, size_t * result);
GGML_API bool ggml_hook_type_validate(enum ggml_type type, const void * data, size_t nbytes, bool * ok);
#ifdef __cplusplus
}
#endif

#define GGML_HOOK_TYPE_TRAITS_ENTRY(NAME, ID, TNAME, BLK, SIZE, ...) \
    [ID] = { .type_name = #TNAME, .blck_size = BLK, .type_size = SIZE, .is_quantized = true, \
             .to_float = ggml_hook_type_##TNAME##_to_float, .from_float_ref = ggml_hook_type_##TNAME##_from_float_ref },
// unused ids stay empty, like the removed upstream types (Q4_2, ...)
#define GGML_HOOK_TYPE_TRAITS GGML_HOOK_TYPES_LIST(GGML_HOOK_TYPE_TRAITS_ENTRY)

#define GGML_HOOK_TYPE_TRAITS_CPU_ENTRY(NAME, ID, TNAME, BLK, SIZE, DOT_TYPE, NROWS) \
    [ID] = { .from_float = ggml_hook_type_##TNAME##_cpu_from_float, .vec_dot = ggml_hook_type_##TNAME##_cpu_vec_dot, \
             .vec_dot_type = DOT_TYPE, .nrows = NROWS },
#define GGML_HOOK_TYPE_TRAITS_CPU GGML_HOOK_TYPES_LIST(GGML_HOOK_TYPE_TRAITS_CPU_ENTRY)

#define GGML_HOOK_TYPE_QUANTIZE(type, src, dst, nrows, n_per_row, imatrix, result) ggml_hook_type_quantize(type, src, dst, nrows, n_per_row, imatrix, result)
#define GGML_HOOK_TYPE_VALIDATE(type, data, nbytes, ok) ggml_hook_type_validate(type, data, nbytes, ok)

// HOOK(sched-realloc-sync) in ggml-backend.cpp: the scheduler re-reserves its buffers and synchronizes each backend
// first. A backend extension can skip the sync of its backend (ggml_hook_backend_ext::skip_realloc_sync).
#ifdef __cplusplus
extern "C" {
#endif
struct ggml_backend;
GGML_API bool ggml_hook_skip_realloc_sync(struct ggml_backend * backend);
#ifdef __cplusplus
}
#endif
#define GGML_HOOK_SKIP_REALLOC_SYNC(backend) ggml_hook_skip_realloc_sync(backend)

// gguf: a type id inside [0, GGML_TYPE_COUNT) that no type uses (the hooks raise GGML_TYPE_COUNT). Logs and returns true.
#ifdef __cplusplus
extern "C" {
#endif
GGML_API bool ggml_hook_gguf_type_unassigned(const char * name, enum ggml_type type);
#ifdef __cplusplus
}
#endif
#define GGML_HOOK_GGUF_TYPE_UNASSIGNED(name, type) ggml_hook_gguf_type_unassigned(name, type)

// The scheduler and graph allocator functions of the fork (sched switch, hooks/ggml/src/ggml-hooks-sched.cpp):
// the declarations of ggml/include/ggml-alloc.h and ggml-backend.h of the fork (1fe05fa45), verbatim. Without the
// switch they keep the upstream behavior (see ggml-hooks-sched.cpp).
// The fork replaces the usage comment of the scheduler in ggml-backend.h by a pointer to its document:
    // see docs/development/backend-scheduler.md
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ggml_gallocr * ggml_gallocr_t;
typedef struct ggml_backend_sched * ggml_backend_sched_t;
// called right before a reserve frees an existing buffer to replace it (e.g. to wait for work that still uses it)
GGML_API void ggml_gallocr_set_free_callback(ggml_gallocr_t galloc, void (*cb)(void * user_data), void * user_data);
GGML_API bool ggml_gallocr_alloc_graph_n(
    ggml_gallocr_t galloc,
    struct ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids);
GGML_API void ggml_gallocr_pin_tensor(ggml_gallocr_t galloc, struct ggml_tensor * t);
GGML_API void ggml_gallocr_clear_pins(ggml_gallocr_t galloc);

GGML_API void ggml_backend_tensor_set_direct(struct ggml_tensor * tensor, size_t offset, size_t size);

GGML_API int                  ggml_backend_sched_get_n_graph_nodes(ggml_backend_sched_t sched);
GGML_API void                 ggml_backend_sched_prepare_inputs(ggml_backend_sched_t sched);
// Skip the backend synchronization before a graph re-allocation that keeps every buffer (a buffer that is replaced
// still waits for the backends). The caller then has to synchronize before it writes the graph inputs.
GGML_API void                 ggml_backend_sched_set_lazy_realloc_sync(ggml_backend_sched_t sched, bool lazy);

#ifdef __cplusplus
}
#endif

// FA hints and the f16 DSV4 hc_pre of the fork: the declarations of ggml/include/ggml.h (after upstream lines
// 2281 and 2486, fork 1fe05fa45), verbatim. The builders are in hooks/ggml/src/ggml-hooks-ops.c.
#define GGML_HINT_EXACT_BATCH ((enum ggml_op_hint) 2) // enum ggml_op_hint of the fork
#ifdef __cplusplus
extern "C" {
#endif
    // hint: the mask is the plain causal one over the key index (n_kv == n_q per stream, key j visible to query i
    // iff j <= i), so a backend may skip reading it. The mask must still be supplied and correct.
    GGML_API void ggml_flash_attn_ext_set_causal_prefix(
            struct ggml_tensor * a,
            bool                 causal_prefix);

    GGML_API bool ggml_flash_attn_ext_get_causal_prefix(
            const struct ggml_tensor * a);

    // sparse attention hint: attend only to the first n_kv_raw keys (dense prefix) plus the
    // keys selected by top_k. top_k is I32 [n_top_k, n_tokens, 1, n_streams]; each index i
    // selects absolute key n_kv_raw + i. Negative or out-of-range indices are ignored.
    // Backends may ignore the hint: the kq_mask must still encode the same selection, so a
    // dense fallback computes the identical result.
    GGML_API void ggml_flash_attn_ext_add_top_k(
            struct ggml_tensor * a,
            struct ggml_tensor * top_k,
                     int64_t     n_kv_raw);

    // the gated form with an f32 or f16 output (out_type): the mixed stream as an f16 matmul B
    GGML_API struct ggml_tensor * ggml_dsv4_hc_pre_gated_ext(
            struct ggml_context * ctx,
            struct ggml_tensor  * x,
            struct ggml_tensor  * gate,
            float                 scale,
            enum ggml_type        out_type);

#ifdef __cplusplus
}
#endif

// EXL3 ops of the fork: the declarations of ggml/include/ggml.h (after upstream line 2500, fork 1fe05fa45), verbatim.
// The nodes are GGML_OP_CUSTOM nodes (hooks/ggml/exl3/ggml-exl3-ops.c): the fork op ids GGML_OP_EXL3_* are not
// ggml ops here. ggml_hook_exl3_op() tells which EXL3 op a node is; only hook kernels compute them.
#ifdef __cplusplus
extern "C" {
#endif
    // EXL3 (exllamav3) linear, see ggml/exl3/README.md: y = x W with W = diag(suh) H W_hat H diag(svh) (k x n)
    //   trellis I16 [16*K, n/16, k/16], x F32 [k, ...], suh F16 [k], svh F16 [n] -> F32 [n, ...]
    //   codebook: 0 = 3inst, 1 = mcg, 2 = mul1; computed only by builds with GGML_EXL3
    GGML_API struct ggml_tensor * ggml_exl3_mul_mat(
            struct ggml_context * ctx,
            struct ggml_tensor  * trellis,
            struct ggml_tensor  * x,
            struct ggml_tensor  * suh,
            struct ggml_tensor  * svh,
            int                   codebook);

    // EXL3 experts (MoE), the EXL3 counterpart of ggml_mul_mat_id: expert e of trellis [16*K, n/16, k/16, E] with
    //   suh [k, E] and svh [n, E]; x F32 [k, ne11, n_tokens], ids I32 [n_used, n_tokens], ne11 divides n_used ->
    //   F32 [n, n_used, n_tokens]; computed only by builds with GGML_EXL3
    GGML_API struct ggml_tensor * ggml_exl3_mul_mat_id(
            struct ggml_context * ctx,
            struct ggml_tensor  * trellis,
            struct ggml_tensor  * x,
            struct ggml_tensor  * ids,
            struct ggml_tensor  * suh,
            struct ggml_tensor  * svh,
            int                   codebook);

    // EXL3 row-ring table (quantized token embedding, see ggml/exl3/README.md): the decoded rows `ids` of `table`
    //   table I16 [groups*(1 + 16*K), n_rows], ids I32 [n], signs F16 [256, groups] -> F32 [n_out, n], n_out <= groups*256
    //   computed only by builds with GGML_EXL3
    GGML_API struct ggml_tensor * ggml_exl3_get_rows(
            struct ggml_context * ctx,
            struct ggml_tensor  * table,
            struct ggml_tensor  * ids,
            struct ggml_tensor  * signs,
            int64_t               n_out);

enum ggml_hook_exl3_op_kind {
    GGML_HOOK_EXL3_NONE,
    GGML_HOOK_EXL3_MUL_MAT,     // GGML_OP_EXL3_MUL_MAT of the fork
    GGML_HOOK_EXL3_GET_ROWS,    // GGML_OP_EXL3_GET_ROWS of the fork
    GGML_HOOK_EXL3_MUL_MAT_ID,  // GGML_OP_EXL3_MUL_MAT_ID of the fork
};
GGML_API enum ggml_hook_exl3_op_kind ggml_hook_exl3_op(const struct ggml_tensor * t);
// the i32 parameter of the fork op (op_params[0] there: the codebook, or n_out for GET_ROWS)
GGML_API int32_t ggml_hook_exl3_param(const struct ggml_tensor * t);
// any custom op of the hooks: no backend computes it unless a hook kernel takes it
GGML_API bool ggml_hook_is_hook_custom(const struct ggml_tensor * t);
#ifdef __cplusplus
}
#endif

#else

#define GGML_HOOK_TYPE_TRAITS
#define GGML_HOOK_TYPE_TRAITS_CPU
#define GGML_HOOK_TYPE_QUANTIZE(type, src, dst, nrows, n_per_row, imatrix, result) false
#define GGML_HOOK_TYPE_VALIDATE(type, data, nbytes, ok) false
#define GGML_HOOK_SKIP_REALLOC_SYNC(backend) false
#define GGML_HOOK_GGUF_TYPE_UNASSIGNED(name, type) false

#endif
