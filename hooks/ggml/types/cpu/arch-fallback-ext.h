#pragma once

// the fork additions to ggml/src/ggml-cpu/arch-fallback.h (upstream 9048bad8d, fork 1fe05fa45), verbatim,
// in the #if chain of the upstream file. Included at the end of quants-ext.h.
#if defined(GGML_CPU_GENERIC)
// PTQ1_0 has no arch-specific kernel on any target; always use the generic one.
#define ggml_vec_dot_ptq1_0_q8_0_generic ggml_vec_dot_ptq1_0_q8_0
#define ggml_vec_dot_tqp5_q8_0_generic ggml_vec_dot_tqp5_q8_0
#define ggml_vec_dot_tqp6_q8_0_generic ggml_vec_dot_tqp6_q8_0
#define ggml_vec_dot_tqp7_q8_0_generic ggml_vec_dot_tqp7_q8_0
#define ggml_vec_dot_tq2_t_q8_0_generic ggml_vec_dot_tq2_t_q8_0
#define ggml_vec_dot_tqk6_q8_0_generic ggml_vec_dot_tqk6_q8_0
#define ggml_vec_dot_tqk7_q8_0_generic ggml_vec_dot_tqk7_q8_0
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(_M_ARM64) || defined(_M_ARM64EC)
// PTQ1_0 has no arch-specific kernel on any target; always use the generic one.
#define ggml_vec_dot_ptq1_0_q8_0_generic ggml_vec_dot_ptq1_0_q8_0
#define ggml_vec_dot_tqp5_q8_0_generic ggml_vec_dot_tqp5_q8_0
#define ggml_vec_dot_tqp6_q8_0_generic ggml_vec_dot_tqp6_q8_0
#define ggml_vec_dot_tqp7_q8_0_generic ggml_vec_dot_tqp7_q8_0
#define ggml_vec_dot_tq2_t_q8_0_generic ggml_vec_dot_tq2_t_q8_0
#define ggml_vec_dot_tqk6_q8_0_generic ggml_vec_dot_tqk6_q8_0
#define ggml_vec_dot_tqk7_q8_0_generic ggml_vec_dot_tqk7_q8_0
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
// PTQ1_0 has no arch-specific kernel on any target; always use the generic one.
#define ggml_vec_dot_ptq1_0_q8_0_generic ggml_vec_dot_ptq1_0_q8_0
// TQ2_T / TQK6 / TQK7: arch/x86/quants.c (AVX-512 VBMI or AVX2; generic otherwise)
// TQP5 / TQP6 / TQP7: arch/x86/quants.c (AVX2, also in AVX-512 builds; generic otherwise)
#elif defined(__POWERPC__) || defined(__powerpc__)
// PTQ1_0 has no arch-specific kernel on any target; always use the generic one.
#define ggml_vec_dot_ptq1_0_q8_0_generic ggml_vec_dot_ptq1_0_q8_0
#define ggml_vec_dot_tqp5_q8_0_generic ggml_vec_dot_tqp5_q8_0
#define ggml_vec_dot_tqp6_q8_0_generic ggml_vec_dot_tqp6_q8_0
#define ggml_vec_dot_tqp7_q8_0_generic ggml_vec_dot_tqp7_q8_0
#define ggml_vec_dot_tq2_t_q8_0_generic ggml_vec_dot_tq2_t_q8_0
#define ggml_vec_dot_tqk6_q8_0_generic ggml_vec_dot_tqk6_q8_0
#define ggml_vec_dot_tqk7_q8_0_generic ggml_vec_dot_tqk7_q8_0
#elif defined(__loongarch64)
// PTQ1_0 has no arch-specific kernel on any target; always use the generic one.
#define ggml_vec_dot_ptq1_0_q8_0_generic ggml_vec_dot_ptq1_0_q8_0
#define ggml_vec_dot_tqp5_q8_0_generic ggml_vec_dot_tqp5_q8_0
#define ggml_vec_dot_tqp6_q8_0_generic ggml_vec_dot_tqp6_q8_0
#define ggml_vec_dot_tqp7_q8_0_generic ggml_vec_dot_tqp7_q8_0
#define ggml_vec_dot_tq2_t_q8_0_generic ggml_vec_dot_tq2_t_q8_0
#define ggml_vec_dot_tqk6_q8_0_generic ggml_vec_dot_tqk6_q8_0
#define ggml_vec_dot_tqk7_q8_0_generic ggml_vec_dot_tqk7_q8_0
#elif defined(__riscv)
// PTQ1_0 has no arch-specific kernel on any target; always use the generic one.
#define ggml_vec_dot_ptq1_0_q8_0_generic ggml_vec_dot_ptq1_0_q8_0
#define ggml_vec_dot_tqp5_q8_0_generic ggml_vec_dot_tqp5_q8_0
#define ggml_vec_dot_tqp6_q8_0_generic ggml_vec_dot_tqp6_q8_0
#define ggml_vec_dot_tqp7_q8_0_generic ggml_vec_dot_tqp7_q8_0
#define ggml_vec_dot_tq2_t_q8_0_generic ggml_vec_dot_tq2_t_q8_0
#define ggml_vec_dot_tqk6_q8_0_generic ggml_vec_dot_tqk6_q8_0
#define ggml_vec_dot_tqk7_q8_0_generic ggml_vec_dot_tqk7_q8_0
#elif defined(__s390x__)
// PTQ1_0 has no arch-specific kernel on any target; always use the generic one.
#define ggml_vec_dot_ptq1_0_q8_0_generic ggml_vec_dot_ptq1_0_q8_0
#define ggml_vec_dot_tqp5_q8_0_generic ggml_vec_dot_tqp5_q8_0
#define ggml_vec_dot_tqp6_q8_0_generic ggml_vec_dot_tqp6_q8_0
#define ggml_vec_dot_tqp7_q8_0_generic ggml_vec_dot_tqp7_q8_0
#define ggml_vec_dot_tq2_t_q8_0_generic ggml_vec_dot_tq2_t_q8_0
#define ggml_vec_dot_tqk6_q8_0_generic ggml_vec_dot_tqk6_q8_0
#define ggml_vec_dot_tqk7_q8_0_generic ggml_vec_dot_tqk7_q8_0
#elif defined(__wasm__)
// PTQ1_0 has no arch-specific kernel on any target; always use the generic one.
#define ggml_vec_dot_ptq1_0_q8_0_generic ggml_vec_dot_ptq1_0_q8_0
#define ggml_vec_dot_tqp5_q8_0_generic ggml_vec_dot_tqp5_q8_0
#define ggml_vec_dot_tqp6_q8_0_generic ggml_vec_dot_tqp6_q8_0
#define ggml_vec_dot_tqp7_q8_0_generic ggml_vec_dot_tqp7_q8_0
#define ggml_vec_dot_tq2_t_q8_0_generic ggml_vec_dot_tq2_t_q8_0
#define ggml_vec_dot_tqk6_q8_0_generic ggml_vec_dot_tqk6_q8_0
#define ggml_vec_dot_tqk7_q8_0_generic ggml_vec_dot_tqk7_q8_0
#endif
