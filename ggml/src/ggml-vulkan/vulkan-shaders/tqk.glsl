#ifndef TQK_GLSL
#define TQK_GLSL

// TQK6 / TQK7 state extraction (buffer-independent helpers; included from types.glsl
// under DATA_A_TQK6 / DATA_A_TQK7, which define TQK_K = 6u / 7u).
//
// Format (bit-exact with tqk_state in ggml-common.h and agention-infer's dequant_tqk):
//   qs is a circular stream of 32*K bits (4*K bytes = 2*K u16 words),
//     stream bit i = (qs[i >> 3] >> (i & 7)) & 1                      (LSB-first)
//   step t (0..31, weights 4t..4t+3) has the 16-bit state
//     s = sum_{b=0..15} bit(((31 - t)*K + b) mod 32*K) << b
//   and its 4 weights are d * tq2_t_step(s).
// Only steps 0 and 1 wrap ((31 - t)*K + 16 > 32*K iff t*K < 16 - K for K = 6, 7).
// Since 32*K is a multiple of 16, a bit wrap is a byte wrap and a word wrap: read the
// bytes (words) covering bits off..off+15 with indices taken modulo 4*K (2*K) and shift.

#define TQK_QS_BYTES (4u*TQK_K)
#define TQK_QS_WORDS (2u*TQK_K)

// Stream bit at which step t's 16-bit window starts. C: off = (31 - t)*k
uint tqk_off(uint t) {
    return (31u - t) * TQK_K;
}

// i + 1 or i + 2 modulo the stream length, for i < TQK_QS_BYTES (TQK_QS_WORDS).
uint tqk_byte_wrap(uint i) {
    return i >= TQK_QS_BYTES ? i - TQK_QS_BYTES : i;
}
uint tqk_word_wrap(uint i) {
    return i >= TQK_QS_WORDS ? i - TQK_QS_WORDS : i;
}

// Byte-view extraction, line for line with tqk_state():
//   i0 = off >> 3; i1 = (i0 + 1) % nbytes; i2 = (i0 + 2) % nbytes;
//   w = qs[i0] | qs[i1] << 8 | qs[i2] << 16;  s = (w >> (off & 7)) & 0xffff
// b0, b1, b2 are qs[i0], qs[i1], qs[i2] with i0 = off >> 3 and the wrapped i1, i2.
uint tqk_state_bytes(uint b0, uint b1, uint b2, uint off) {
    const uint w = b0 | (b1 << 8u) | (b2 << 16u);
    return (w >> (off & 7u)) & 0xFFFFu;
}

// Word-view extraction for a run of windows. lo, mid, hi are the 32-bit pairs
// (w0 | w1 << 16), (w1 | w2 << 16), (w2 | w3 << 16) of four consecutive (wrapped) u16
// words starting at word wb, and rel = off - 16*wb in [0, 48). The window lies in the
// pair starting at word rel >> 4, shifted by rel & 15 (<= 15, so 16 bits always fit).
uint tqk_state_words(uint lo, uint mid, uint hi, uint rel) {
    const uint wi = rel >> 4u;
    const uint pair = wi == 0u ? lo : (wi == 1u ? mid : hi);
    return (pair >> (rel & 15u)) & 0xFFFFu;
}

#endif // TQK_GLSL
