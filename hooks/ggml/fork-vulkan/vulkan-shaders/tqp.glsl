#ifndef TQP_GLSL
#define TQP_GLSL

// TQP5 / TQP6 / TQP7 pair-trellis helpers (included from types.glsl under DATA_A_TQP5/6/7,
// which define TQP_K = 5u / 6u / 7u, after tq2_t_lut.glsl).
//
// Format (bit-exact with tqp_state in ggml-common.h and dequantize_row_tqp* in ggml-quants.c):
//   qs is a circular stream of 64*K bits (8*K bytes = 4*K u16 words), LSB-first.
//   Step t (0..63, weights 2t, 2t+1) has the 16-bit state
//     s = sum_{b=0..15} bit(((63 - t)*K + b) mod 64*K) << b
//   and its two weights are d * (first codebook pair of TQ2_T state s).
// 64*K is a multiple of 16, so a bit wrap is also a byte and a word wrap.

#define TQP_QS_BYTES (8u*TQP_K)
#define TQP_QS_WORDS (4u*TQP_K)

// Stream bit at which the window of step t starts.
uint tqp_off(uint t) {
    return (63u - t) * TQP_K;
}

// i + 1 .. i + 4 modulo the stream length, for i < TQP_QS_BYTES (TQP_QS_WORDS).
uint tqp_byte_wrap(uint i) {
    return i >= TQP_QS_BYTES ? i - TQP_QS_BYTES : i;
}
uint tqp_word_wrap(uint i) {
    return i >= TQP_QS_WORDS ? i - TQP_QS_WORDS : i;
}

// Byte view, line for line with tqp_state(): b0, b1, b2 = qs[i0], qs[i0+1], qs[i0+2] (wrapped).
uint tqp_state_bytes(uint b0, uint b1, uint b2, uint off) {
    const uint w = b0 | (b1 << 8u) | (b2 << 16u);
    return (w >> (off & 7u)) & 0xFFFFu;
}

// lo, hi = stream bits [S, S + 32) and [S + 32, S + 64) from the 32-bit pairs
// p0 = w0 | w1 << 16, p1 = w2 | w3 << 16, p2 = w4 of the (wrapped) u16 words from word S >> 4,
// with r = S & 15. The double shift keeps r = 0 defined (no shift by 32).
void tqp_window64(uint p0, uint p1, uint p2, uint r, out uint lo, out uint hi) {
    lo = (p0 >> r) | ((p1 << 1u) << (31u - r));
    hi = (p1 >> r) | ((p2 << 1u) << (31u - r));
}

// The 15-bit state window at bit c of the 64-bit run (lo, hi), c + 15 <= 64 (TQ_STATE_BITS:
// the hash ignores bit 15). With constant c the branches fold away.
uint tqp_bits15(uint lo, uint hi, uint c) {
    if (c + 15u <= 32u) {
        return (lo >> c) & TQ_STATE_MASK;
    } else if (c >= 32u) {
        return (hi >> (c - 32u)) & TQ_STATE_MASK;
    }
    return ((lo >> c) | (hi << (32u - c))) & TQ_STATE_MASK;
}

// The two (unscaled) weights of state s: TQ2_T pair 0, one multiply per state.
vec2 tqp_pair(uint s) {
    const uint x = (s & TQ_STATE_MASK) * 0x9e3779b1u;
    return unpackHalf2x16(tq2t_lut[x >> 21u]);
}

#endif // TQP_GLSL
