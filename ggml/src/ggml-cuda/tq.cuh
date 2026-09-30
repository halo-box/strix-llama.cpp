#pragma once

// Trellis quant types TQ2_T, TQK6, TQK7 (see ggml-common.h for the format).
//
// A 128-weight block is 32 trellis steps of 4 weights. Step t has a 16-bit state s (the
// codebook index uses its low TQ_STATE_BITS bits) and its weights are
//   d * lut[x >> 21].{lo,hi}, d * lut[(x >> 10) & 2047].{lo,hi},  x = (s & TQ_STATE_MASK) * 0x9e3779b1
// where lut is the 2048-point fp16 codebook tq2t_lut_f16 read as u32 pairs (lo | hi << 16).
//
// All kernels split a block into 8 lanes of 4 consecutive steps (16 weights), like the
// Vulkan shaders: lane l owns steps 4l..4l+3 = weights 16l..16l+15.

#include "common.cuh"

#define TQ_LUT_POINTS   2048
#define TQ_LANES        8   // lanes per 128-weight block
#define TQ_STEPS_LANE   4   // trellis steps per lane
#define VDR_TQ_Q8_1_MMVQ TQ_STEPS_LANE

static constexpr __host__ __device__ bool ggml_cuda_type_is_tq(ggml_type type) {
    return type == GGML_TYPE_TQ2_T || type == GGML_TYPE_TQK6 || type == GGML_TYPE_TQK7;
}

// Global (L1/L2-cached) view of the codebook as u32 pairs; the table is 16-byte aligned.
static __device__ __forceinline__ const uint32_t * tq_lut_global() {
    return (const uint32_t *) tq2t_lut_f16;
}

// Copy the codebook into shared memory (nthreads threads with linear id tid) and sync.
static __device__ __forceinline__ void tq_lut_load_shared(uint32_t * lut_s, const int tid, const int nthreads) {
    const uint32_t * lut_g = tq_lut_global();
    for (int i = tid; i < TQ_LUT_POINTS; i += nthreads) {
        lut_s[i] = lut_g[i];
    }
    __syncthreads();
}

static __device__ __forceinline__ float2 tq_unpack_pair(const uint32_t v) {
    return make_float2(__half2float(__ushort_as_half((unsigned short) (v & 0xFFFFu))),
                       __half2float(__ushort_as_half((unsigned short) (v >> 16))));
}

// The 4 unscaled weights of the step with state s: (a.x, a.y, b.x, b.y).
static __device__ __forceinline__ void tq_step(const uint32_t * lut, const uint32_t s, float2 & a, float2 & b) {
    const uint32_t x = (s & TQ_STATE_MASK) * 0x9e3779b1u;
    a = tq_unpack_pair(lut[x >> 21]);
    b = tq_unpack_pair(lut[(x >> 10) & 2047u]);
}

template <ggml_type type> struct tq_block_info;
template <> struct tq_block_info<GGML_TYPE_TQ2_T> { typedef block_tq2_t block_t; static constexpr int K = 8; };
template <> struct tq_block_info<GGML_TYPE_TQK6>  { typedef block_tqk6  block_t; static constexpr int K = 6; };
template <> struct tq_block_info<GGML_TYPE_TQK7>  { typedef block_tqk7  block_t; static constexpr int K = 7; };

// States of steps 4l..4l+3 of one block. qs is the block's qs array, 2-byte aligned
// (every trellis block has an even size and qs sits right after the fp16 scale).
template <ggml_type type>
static __device__ __forceinline__ void tq_states4(const uint8_t * qs, const int l, uint32_t s[TQ_STEPS_LANE]) {
    const uint16_t * qs16 = (const uint16_t *) qs;
    if constexpr (type == GGML_TYPE_TQ2_T) {
        // s_t = qs[(t + 31) % 32] << 8 | qs[t]
        const uint32_t w01  = qs16[2*l + 0];
        const uint32_t w23  = qs16[2*l + 1];
        const uint32_t prev = qs[(4*l + 31) & 31];
        const uint32_t b0 = w01 & 0xFF, b1 = w01 >> 8, b2 = w23 & 0xFF, b3 = w23 >> 8;
        s[0] = (prev << 8) | b0;
        s[1] = (b0   << 8) | b1;
        s[2] = (b1   << 8) | b2;
        s[3] = (b2   << 8) | b3;
    } else {
        // Step t's state is the 16 stream bits from bit (31 - t)*K (circular over 32*K bits,
        // i.e. 2*K u16 words). Steps 4l..4l+3 all lie in the four words from
        // wb = off3 >> 4, off3 = (28 - 4l)*K, since (off3 & 15) + 3*K + 16 <= 64.
        constexpr int K      = tq_block_info<type>::K;
        constexpr int nwords = 2*K;
        const int off3 = (28 - 4*l)*K;
        const int wb   = off3 >> 4;
        const int r    = off3 & 15;
        const int wi1  = wb + 1 >= nwords ? wb + 1 - nwords : wb + 1;
        const int wi2  = wb + 2 >= nwords ? wb + 2 - nwords : wb + 2;
        const int wi3  = wb + 3 >= nwords ? wb + 3 - nwords : wb + 3;
        const uint64_t v = (uint64_t) qs16[wb] | ((uint64_t) qs16[wi1] << 16) |
                           ((uint64_t) qs16[wi2] << 32) | ((uint64_t) qs16[wi3] << 48);
#pragma unroll
        for (int i = 0; i < TQ_STEPS_LANE; ++i) {
            s[i] = (uint32_t) (v >> (r + (3 - i)*K)) & 0xFFFFu;
        }
    }
}

// Dequantize lane l (16 weights) of block x into out[0..15].
template <ggml_type type>
static __device__ __forceinline__ void tq_dequant_lane(const void * x, const int l, const uint32_t * lut, float out[16]) {
    typedef typename tq_block_info<type>::block_t block_t;
    const block_t * b = (const block_t *) x;
    const float d = __half2float(b->d);
    uint32_t s[TQ_STEPS_LANE];
    tq_states4<type>(b->qs, l, s);
#pragma unroll
    for (int i = 0; i < TQ_STEPS_LANE; ++i) {
        float2 p0, p1;
        tq_step(lut, s[i], p0, p1);
        out[4*i + 0] = d*p0.x;
        out[4*i + 1] = d*p0.y;
        out[4*i + 2] = d*p1.x;
        out[4*i + 3] = d*p1.y;
    }
}

// mmvq dot product of one lane of block kbx against q8_1 values. bq8_1 points at the
// q8_1 block aligned with the start of block kbx (4 q8_1 blocks per trellis block);
// iqs = VDR_TQ_Q8_1_MMVQ * lane.
template <ggml_type type>
static __device__ __forceinline__ float vec_dot_tq_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    typedef typename tq_block_info<type>::block_t block_t;
    const block_t * b = (const block_t *) vbq + kbx;
    const int l = iqs / VDR_TQ_Q8_1_MMVQ;

    uint32_t s[TQ_STEPS_LANE];
    tq_states4<type>(b->qs, l, s);

    const block_q8_1 * b8 = bq8_1 + l/2;
    const int * q8 = (const int *) b8->qs + 4*(l & 1);
    const uint32_t * lut = tq_lut_global();

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < TQ_STEPS_LANE; ++i) {
        float2 p0, p1;
        tq_step(lut, s[i], p0, p1);
        const int v = q8[i];
        sum += p0.x * (float) (int8_t) (v      ) + p0.y * (float) (int8_t) (v >>  8)
             + p1.x * (float) (int8_t) (v >> 16) + p1.y * (float) (int8_t) (v >> 24);
    }
    return __half2float(b->d) * __low2float(b8->ds) * sum;
}
