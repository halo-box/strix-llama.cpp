#pragma once

// Exact K/V element -> f16 conversions of the HIP sparse-attention kernels (PLAN-v3 §2.2 contract item 1, §6.2).
//
// Every helper returns the f16 value of the element's to_float rounded to f16 (RNE, +-inf on overflow), the value the
// CPU op (ggml-cpu/sparse-attn.cpp, sa_load_row) and ggml_get_to_fp16_cuda produce:
//   q8_0  sa_q8x2_to_h2: PR #219's qsa3_q8x2_to_h2 / qd_q8x2_to_h2 (qsa-prefill.cu and qsa-decode-wmma.cuh of 94e086847),
//         one copy instead of three. 0x6400 | (q + 128) is 1024 + q + 128 in half, exact, so (v - 1152) is q exactly
//         and the single half product q * d is RNE of the exact f32 product, as dequantize_block_q8_0_f16 computes it
//         (C18: 4905 prefill and 4320 decode bitwise checks).
//   q4_0  sa_q4x2_to_h2: 0x6400 | q is 1024 + q in half, exact; minus 1032 gives q - 8 exactly, then one rounding of
//         (q - 8) * d (C18 c18_q4x2_to_h2).
//   bf16  sa_bf2_to_h2 / sa_bf1_to_h: the bf16 value widened to f32 (exact) and rounded to f16 with one RNE conversion,
//         +-inf past the f16 range. C18's c18_bf2_to_h2 saturated to +-65504 first; the op contract (and to_fp16)
//         overflows to +-inf, so this one does not saturate.
// The conversions are bit-exact by construction; tests/test-backend-ops.cpp (SPARSE_ATTN, overflow and poison cases)
// checks the contract end to end against the CPU op.

#if defined(GGML_USE_HIP)

#include <cstdint>

typedef _Float16 sa_half2 __attribute__((ext_vector_type(2)));
typedef short    sa_v16s  __attribute__((ext_vector_type(16)));   // one WMMA f16 fragment (16 halves)
typedef float    sa_v8f   __attribute__((ext_vector_type(8)));    // one WMMA f32 accumulator fragment

// two q8_0 values in byte 0 and byte 2 of x, times the packed scales d2, to two packed halves (PR #219 helper, verbatim
// up to the names)
static __device__ __forceinline__ uint32_t sa_q8x2_to_h2(const uint32_t x, const sa_half2 d2) {
    const sa_half2 v = __builtin_bit_cast(sa_half2, ((x ^ 0x00800080u) & 0x00FF00FFu) | 0x64006400u);
    return __builtin_bit_cast(uint32_t, (v - sa_half2{(_Float16) 1152, (_Float16) 1152}) * d2);
}

// two q4_0 nibbles in the low nibble of byte 0 and of byte 2 of x, times d2, to two packed halves
static __device__ __forceinline__ uint32_t sa_q4x2_to_h2(const uint32_t x, const sa_half2 d2) {
    const sa_half2 v = __builtin_bit_cast(sa_half2, (x & 0x000F000Fu) | 0x64006400u);
    return __builtin_bit_cast(uint32_t, (v - sa_half2{(_Float16) 1032, (_Float16) 1032}) * d2);
}

// one bf16 (low 16 bits of x16) to f16: exact widening, one RNE rounding, +-inf on overflow
static __device__ __forceinline__ uint16_t sa_bf1_to_h(const uint32_t x16) {
    return __builtin_bit_cast(uint16_t, (_Float16) __builtin_bit_cast(float, x16 << 16));
}

// two packed bf16 to two packed f16 (two scalar RNE conversions: the packed v_cvt_pkrtz_f16_f32 truncates)
static __device__ __forceinline__ uint32_t sa_bf2_to_h2(const uint32_t x) {
    const _Float16 a = (_Float16) __builtin_bit_cast(float, x << 16);
    const _Float16 b = (_Float16) __builtin_bit_cast(float, x & 0xffff0000u);
    return (uint32_t) __builtin_bit_cast(uint16_t, a) | ((uint32_t) __builtin_bit_cast(uint16_t, b) << 16);
}

// f32 -> f16 bits, RNE
static __device__ __forceinline__ uint16_t sa_f2h(const float f) {
    return __builtin_bit_cast(uint16_t, (_Float16) f);
}

#endif // defined(GGML_USE_HIP)
