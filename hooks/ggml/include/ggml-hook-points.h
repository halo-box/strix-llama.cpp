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

// X(NAME, ID, type_name, blck_size, type_size, is_quantized, to_float, from_float_ref,
//   cpu_from_float, cpu_vec_dot, cpu_vec_dot_type, cpu_nrows)
// The functions are named here and declared by the files that expand the hook points (ggml.c, ggml-cpu.c), or below.
// Quantize and validate functions: ggml_hook_type_<type_name>_quantize / _validate, see ggml-hooks-types.c.
#define GGML_HOOK_TYPES_LIST(X) \
    X(Q8_0_HOOK_EXAMPLE, 100, "q8_0_hook_example", QK8_0, sizeof(block_q8_0), true, dequantize_row_q8_0, quantize_row_q8_0_ref, \
      quantize_row_q8_0, ggml_vec_dot_q8_0_q8_0, GGML_TYPE_Q8_0, 1)

#define GGML_HOOK_TYPE_ID(NAME, ID, ...) static const enum ggml_type GGML_TYPE_##NAME = (enum ggml_type) (ID);
GGML_HOOK_TYPES_LIST(GGML_HOOK_TYPE_ID)
#undef GGML_HOOK_TYPE_ID

static inline bool ggml_hook_type_is_hook(enum ggml_type type) {
    return (int) type >= GGML_TYPE_HOOKS_FIRST;
}

#ifdef __cplusplus
extern "C" {
#endif
// implemented in hooks/ggml/src/ggml-hooks-types.c (ggml-base). Return false if the type is not a hook type.
GGML_API bool ggml_hook_type_quantize(enum ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix, size_t * result);
GGML_API bool ggml_hook_type_validate(enum ggml_type type, const void * data, size_t nbytes, bool * ok);
#ifdef __cplusplus
}
#endif

// ids between the upstream types and the hook types get a name, for the loops over every type
#define GGML_HOOK_TYPE_TRAITS_ENTRY(NAME, ID, TNAME, BLK, SIZE, QUANT, TO_F, FROM_REF, ...) \
    [ID] = { .type_name = TNAME, .blck_size = BLK, .type_size = SIZE, .is_quantized = QUANT, \
             .to_float = (ggml_to_float_t) TO_F, .from_float_ref = (ggml_from_float_t) FROM_REF },
#define GGML_HOOK_TYPE_TRAITS \
    [GGML_TYPE_COUNT_UPSTREAM ... GGML_TYPE_HOOKS_FIRST - 1] = { .type_name = "unused" }, \
    GGML_HOOK_TYPES_LIST(GGML_HOOK_TYPE_TRAITS_ENTRY)

#define GGML_HOOK_TYPE_TRAITS_CPU_ENTRY(NAME, ID, TNAME, BLK, SIZE, QUANT, TO_F, FROM_REF, CPU_FROM, CPU_DOT, CPU_DOT_TYPE, CPU_NROWS) \
    [ID] = { .from_float = (ggml_from_float_t) CPU_FROM, .vec_dot = (ggml_vec_dot_t) CPU_DOT, .vec_dot_type = CPU_DOT_TYPE, .nrows = CPU_NROWS },
#define GGML_HOOK_TYPE_TRAITS_CPU GGML_HOOK_TYPES_LIST(GGML_HOOK_TYPE_TRAITS_CPU_ENTRY)

#define GGML_HOOK_TYPE_QUANTIZE(type, src, dst, nrows, n_per_row, imatrix, result) ggml_hook_type_quantize(type, src, dst, nrows, n_per_row, imatrix, result)
#define GGML_HOOK_TYPE_VALIDATE(type, data, nbytes, ok) ggml_hook_type_validate(type, data, nbytes, ok)

// HOOK(sched-realloc-sync) in ggml-backend.cpp: the scheduler re-reserves its buffers and synchronizes each backend
// first. A backend extension can skip the sync of its backend (ggml_hook_backend_ext::skip_realloc_sync).
#ifdef __cplusplus
extern "C" {
#endif
GGML_API bool ggml_hook_skip_realloc_sync(struct ggml_backend * backend);
#ifdef __cplusplus
}
#endif
#define GGML_HOOK_SKIP_REALLOC_SYNC(backend) ggml_hook_skip_realloc_sync(backend)

#else

#define GGML_HOOK_TYPE_TRAITS
#define GGML_HOOK_TYPE_TRAITS_CPU
#define GGML_HOOK_TYPE_QUANTIZE(type, src, dst, nrows, n_per_row, imatrix, result) false
#define GGML_HOOK_TYPE_VALIDATE(type, data, nbytes, ok) false
#define GGML_HOOK_SKIP_REALLOC_SYNC(backend) false

#endif
