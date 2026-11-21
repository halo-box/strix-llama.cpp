// The fork types in each CPU variant: wrappers of the CPU type traits (HOOK(type-traits-cpu)).
// The kernels are in quants-ext.c, x86-quants-ext.c, rocmfpx-ext.c and rocmfp4/ (ggml-base).

#define GGML_COMMON_DECL_C
#include "../ggml-quants-ext.h"
#include "quants-ext.h"
#include "quants.h"

#include <stdatomic.h>

// ggml_cpu_init of the fork called ggml_cpu_tq_init after the FP16 table. Upstream does not call it:
// it runs before the first trellis kernel instead, when ggml_cpu_init has run.
static void tq_init_once(void) {
    static atomic_int state = 0; // 0: not done, 1: running, 2: done
    if (atomic_load_explicit(&state, memory_order_acquire) == 2) {
        return;
    }
    int expected = 0;
    if (atomic_compare_exchange_strong(&state, &expected, 1)) {
        ggml_cpu_tq_init();
        atomic_store_explicit(&state, 2, memory_order_release);
        return;
    }
    while (atomic_load_explicit(&state, memory_order_acquire) != 2) {
    }
}

#define GGML_HOOK_CPU(TNAME, FROM_FLOAT, VEC_DOT, INIT)                                                                     \
    void ggml_hook_type_##TNAME##_cpu_from_float(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {       \
        FROM_FLOAT(x, y, k);                                                                                                \
    }                                                                                                                       \
    void ggml_hook_type_##TNAME##_cpu_vec_dot(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x,     \
                                              size_t bx, const void * GGML_RESTRICT y, size_t by, int nrc) {               \
        INIT;                                                                                                               \
        VEC_DOT(n, s, bs, x, bx, y, by, nrc);                                                                               \
    }

GGML_HOOK_CPU(q4_0_rocmfp4,      rocmfp4_quantize_row_q4_0,      rocmfp4_vec_dot_q4_0_q8_0,      (void) 0)
GGML_HOOK_CPU(q4_0_rocmfp4_fast, rocmfp4_quantize_row_q4_0_fast, rocmfp4_vec_dot_q4_0_fast_q8_0, (void) 0)
GGML_HOOK_CPU(ptq1_0,            quantize_row_ptq1_0,            ggml_vec_dot_ptq1_0_q8_0,       tq_init_once())
GGML_HOOK_CPU(tq2_t,             quantize_row_tq2_t,             ggml_vec_dot_tq2_t_q8_0,        tq_init_once())
GGML_HOOK_CPU(tqk6,              quantize_row_tqk6,              ggml_vec_dot_tqk6_q8_0,         tq_init_once())
GGML_HOOK_CPU(tqk7,              quantize_row_tqk7,              ggml_vec_dot_tqk7_q8_0,         tq_init_once())
GGML_HOOK_CPU(tqp5,              quantize_row_tqp5,              ggml_vec_dot_tqp5_q8_0,         tq_init_once())
GGML_HOOK_CPU(tqp6,              quantize_row_tqp6,              ggml_vec_dot_tqp6_q8_0,         tq_init_once())
GGML_HOOK_CPU(tqp7,              quantize_row_tqp7,              ggml_vec_dot_tqp7_q8_0,         tq_init_once())
GGML_HOOK_CPU(q8_0_hook_example, quantize_row_q8_0,              ggml_vec_dot_q8_0_q8_0,         (void) 0)
// the ROCmFPx wrappers are in rocmfpx-ext.c: their kernels are static there
