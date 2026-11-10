// K/V element conversion of the 64-lane SPARSE_ATTN tiers (sparse_attn_cm.comp, sparse_attn.comp, sparse_attn_pack.comp):
// flash_attn_dequant.glsl through the FaTypeK / FaTypeV spec constants, FLOAT_TYPE = float, then one rounding to f16 by
// the f32 -> f16 conversion (RTE where the device has it, the float-controls patch of ggml_vk_create_pipeline_func).
// PLAN-v3 §2.2: within the §14 bands, not bitwise, for this path.
//
// The offsets and strides are in the unit of the type (ggml_vk_sa_unit_bytes): one uvec4 (16 bytes) for f16 and f32,
// one element (2 bytes) for bf16, one ggml block for the quantized types.

#define BINDING_IDX_K 0
#define BINDING_IDX_V 1

#include "flash_attn_dequant.glsl"

// l > 0 on the bits, for the attend epilogues (sparse_attn_cm.comp, sparse_attn.comp; sparse_attn_reduce.comp has its own
// copy): true for a positive l (+inf included), false for 0, a negative l and NaN. The modules carry no
// SignedZeroInfNanPreserve execution mode, so a float compare may be compiled as if l were never NaN; an integer compare
// cannot be.
bool sa_l_pos(const float l) {
    return floatBitsToUint(l) - 1u < 0x7f800000u;
}

// 4 dims starting at dim d (d % 4 == 0) of the row at unit offset base
vec4 sa_dq4_k(const uint base, const uint d) {
    if (FaTypeK == GGML_TYPE_BF16) {
        return dequantize4(d, 0u, base, BINDING_IDX_K);
    }
    if (FaTypeK == GGML_TYPE_F32) {
        return dequantize4(d / 4u, 0u, base, BINDING_IDX_K);
    }
    return dequantize4(d / 32u, d % 32u, base, BINDING_IDX_K);
}

vec4 sa_dq4_v(const uint base, const uint d) {
    if (FaTypeV == GGML_TYPE_BF16) {
        return dequantize4(d, 0u, base, BINDING_IDX_V);
    }
    if (FaTypeV == GGML_TYPE_F32) {
        return dequantize4(d / 4u, 0u, base, BINDING_IDX_V);
    }
    return dequantize4(d / 32u, d % 32u, base, BINDING_IDX_V);
}

uint sa_pack2(const vec2 x) {
    return packFloat2x16(f16vec2(x));
}

// ---- the fast loaders of the WMMA kernel, S_fast types (sparse_attn_cm.comp) ----
// The bit trick of the HIP loaders (ggml-cuda/sparse-attn-q8.cuh, from PR #219's qsa3_q8x2_to_h2): 0x6400 | (q + 128) is 1024 + q + 128 in half,
// exact, so v - 1152 is q exactly, and one f16 product with the block's scale is RNE of the exact f32 product, the value the generic path computes
// with three conversions per element. q4_0 the same with 0x6400 | q and 1032. bf16: an exact f32 widening and one RNE conversion.
// The number of buffer loads is what these loaders are about (each is 2 uvec4 per K chunk, as f16, or 2 small loads per V piece): K reads whole
// 16-byte lines of the row (rows are 16-byte aligned, PLAN-v3 §2.2), V a 3-dword window that ends inside its piece, so neither leaves the row.

uint sa_q8x2(const uint x, const f16vec2 d2) {
    const f16vec2 v = unpackFloat2x16(((x ^ 0x00800080u) & 0x00FF00FFu) | 0x64006400u);
    return packFloat2x16((v - f16vec2(float16_t(1152.0))) * d2);
}

uint sa_q4x2(const uint x, const f16vec2 d2) {
    const f16vec2 v = unpackFloat2x16((x & 0x000F000Fu) | 0x64006400u);
    return packFloat2x16((v - f16vec2(float16_t(1032.0))) * d2);
}

uint sa_bf2(const uint x) {
    return packFloat2x16(f16vec2(uintBitsToFloat(x << 16), uintBitsToFloat(x & 0xffff0000u)));
}

// two packed f16 of the even dims and two of the odd dims -> the two dwords of 4 contiguous dims
uvec2 sa_zip(const uint e, const uint o) {
    return uvec2((e & 0xffffu) | (o << 16), (e >> 16) | (o & 0xffff0000u));
}

// the 16 packed scales of a block from its f16 scale (low half of x)
f16vec2 sa_d2(const uint x) {
    return unpackFloat2x16((x & 0xffffu) | (x << 16));
}

// 4 q8_0 elements (the 4 bytes of w) -> 4 contiguous f16 in two dwords
uvec2 sa_q8x4(const uint w, const f16vec2 d2) {
    return sa_zip(sa_q8x2(w, d2), sa_q8x2(w >> 8, d2));
}

// 4 q4_0 elements (one nibble per byte: shift 0 = low nibbles, dims < 16 of the block; 4 = high nibbles, dims >= 16)
uvec2 sa_q4x4(const uint w, const uint shift, const f16vec2 d2) {
    return sa_zip(sa_q4x2(w >> shift, d2), sa_q4x2(w >> (shift + 8u), d2));
}

