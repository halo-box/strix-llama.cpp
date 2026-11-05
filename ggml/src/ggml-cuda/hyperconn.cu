#include <vector>
#include <algorithm>
#include "hyperconn.cuh"
#include "mmb.cuh"
#include "ggml-backend-impl.h"

// Explicitly rounded mul/add: keeps the fused kernels from being contracted into FMAs so that
// the results match the separate MUL and ADD kernels of the unfused graph bit for bit.
#if defined(__HIP_PLATFORM_AMD__)
static __device__ __forceinline__ float hc_mul_rn(const float a, const float b) {
    float result;
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}

static __device__ __forceinline__ float hc_add_rn(const float a, const float b) {
    float result;
    asm("v_add_f32_e32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}
#else
static __device__ __forceinline__ float hc_mul_rn(const float a, const float b) {
    return __fmul_rn(a, b);
}

static __device__ __forceinline__ float hc_add_rn(const float a, const float b) {
    return __fadd_rn(a, b);
}
#endif

// same expression as op_sigmoid in unary.cu
static __device__ __forceinline__ float hc_sigmoid(const float x) {
    return 1.0f / (1.0f + expf(-x));
}

// mixed[e,t] = scale * ( ((xn*sig)[0] + (xn*sig)[1]) + ... ) + bias, streams summed in order 0..hc-1
static __global__ void hc_mix_reduce_f32(
        const float * __restrict__ xn, const float * __restrict__ gate, float * __restrict__ dst,
        const int64_t n_embd, const int64_t n_tokens, const int hc, const float scale, const float bias) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n_embd * n_tokens) {
        return;
    }

    const int64_t t = index / n_embd;
    const int64_t e = index - t * n_embd;
    const int64_t hc_dim = n_embd * hc;
    const int64_t base = t * hc_dim + e;

    float acc = hc_mul_rn(xn[base], hc_sigmoid(gate[base]));
    for (int c = 1; c < hc; ++c) {
        const int64_t i = base + int64_t(c) * n_embd;
        acc = hc_add_rn(acc, hc_mul_rn(xn[i], hc_sigmoid(gate[i])));
    }

    // identical expression to scale_f32 in scale.cu
    dst[index] = scale * acc + bias;
}

// out[e,c,t] = residual[e,c,t] + block_out[e,t] * ( s2 * sigmoid(s1 * inject[c,t] + b1) + b2 )
static __global__ void hc_combine_f32(
        const float * __restrict__ residual, const float * __restrict__ block_out, const float * __restrict__ inject,
        float * __restrict__ dst, const int64_t n_embd, const int hc, const int64_t n_tokens,
        const float s1, const float b1, const float s2, const float b2) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t hc_dim = n_embd * hc;
    if (index >= hc_dim * n_tokens) {
        return;
    }

    const int64_t t   = index / hc_dim;
    const int64_t rem = index - t * hc_dim;
    const int64_t c   = rem / n_embd;
    const int64_t e   = rem - c * n_embd;

    // identical expressions to scale_f32 (scale.cu) and op_sigmoid (unary.cu)
    const float x1 = s1 * inject[t * hc + c] + b1;
    const float x2 = hc_sigmoid(x1);
    const float w  = s2 * x2 + b2;

    const float m = hc_mul_rn(block_out[t * n_embd + e], w);
    dst[index] = hc_add_rn(residual[index], m);
}


static __global__ void hc_mix_reduce_f32_hc4_parallel(
        const float * __restrict__ xn, const float * __restrict__ gate, float * __restrict__ dst,
        const int64_t n_embd, const int64_t n_tokens, const float scale, const float bias) {
    const int lane = threadIdx.x % 32;
    const int stream = threadIdx.x / 32;
    const int64_t index = int64_t(blockIdx.x) * 32 + lane;
    const int64_t count = n_embd * n_tokens;
    __shared__ float products[4][32];
    float value = 0.0f;
    if (index < count) {
        const int64_t token = index / n_embd;
        const int64_t embd = index - token * n_embd;
        const int64_t offset = token * (4 * n_embd) + stream * n_embd + embd;
        value = hc_mul_rn(xn[offset], hc_sigmoid(gate[offset]));
    }
    products[stream][lane] = value;
    __syncthreads();
    if (stream == 0 && index < count) {
        float sum = products[0][lane];
        sum = hc_add_rn(sum, products[1][lane]);
        sum = hc_add_rn(sum, products[2][lane]);
        sum = hc_add_rn(sum, products[3][lane]);
        dst[index] = scale * sum + bias;
    }
}

__device__ __forceinline__ float hc_bf2f32(const uint16_t h) { return __uint_as_float(((uint32_t) h) << 16); }
__device__ __forceinline__ uint16_t hc_f2bf32(const float f) { uint32_t u = __float_as_uint(f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }

// same reduction as hc_mix_reduce_f32, reading BF16 copies of xn and gate (16-bit HC intermediates)
static __global__ void hc_mix_reduce_bf16(
        const uint16_t * __restrict__ xn, const uint16_t * __restrict__ gate, float * __restrict__ dst,
        const int64_t n_embd, const int64_t n_tokens, const int hc, const float scale, const float bias) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n_embd * n_tokens) return;
    const int64_t t = index / n_embd, e = index - t * n_embd, base = t * (int64_t) n_embd * hc + e;
    float acc = hc_mul_rn(hc_bf2f32(xn[base]), hc_sigmoid(hc_bf2f32(gate[base])));
    for (int c = 1; c < hc; ++c) { const int64_t i = base + int64_t(c) * n_embd; acc = hc_add_rn(acc, hc_mul_rn(hc_bf2f32(xn[i]), hc_sigmoid(hc_bf2f32(gate[i])))); }
    dst[index] = scale * acc + bias;
}

static bool hc_ranges_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * a1 = a0 + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
    const char * b0 = (const char *) b->data;
    const char * b1 = b0 + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);
    return a0 < b1 && b0 < a1;
}

void ggml_cuda_op_hc_mix_reduce(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_mix_args & args) {
    const ggml_tensor * xn   = args.xn;
    const ggml_tensor * gate = args.gate;
    ggml_tensor *       dst  = args.dst;

    GGML_ASSERT(xn->type == GGML_TYPE_F32 && gate->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(xn) && ggml_is_contiguous(gate) && ggml_is_contiguous(dst));
    GGML_ASSERT(args.hc > 0);

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_tokens = ggml_nrows(dst);
    GGML_ASSERT(xn->ne[0] == n_embd * args.hc && ggml_nrows(xn) == n_tokens);
    GGML_ASSERT(ggml_are_same_shape(xn, gate));

    // The graph allocator may reuse the buffer of the (elided) sigmoid/mul intermediates, which can
    // coincide with the gate matmul output. Every thread reads hc strided elements, so an aliased
    // output has to be staged -- unless (single token) the output lands on a whole stream of the
    // aliased input: element e of the output then overwrites element e of that stream, which only
    // the same thread reads (after it has finished reading).
    auto stream_aligned_alias = [&](const ggml_tensor * in) {
        if (!hc_ranges_overlap(dst, in)) {
            return true;
        }
        if (n_tokens != 1) {
            return false;
        }
        const ptrdiff_t off = (const char *) dst->data - (const char *) in->data;
        return off >= 0 && off % (n_embd * (ptrdiff_t) sizeof(float)) == 0 &&
            off + (ptrdiff_t) ggml_nbytes(dst) <= (ptrdiff_t) ggml_nbytes(in);
    };
    const bool stage = !stream_aligned_alias(xn) || !stream_aligned_alias(gate);

    ggml_cuda_pool_alloc<float> staged(ctx.pool());
    float * out = stage ? staged.alloc(n_embd * n_tokens) : (float *) dst->data;

    const int64_t n_items = n_embd * n_tokens;
    const uint16_t * xn16 = ggml_cuda_mmb_cache_lookup(ctx, xn);
    const uint16_t * g16  = ggml_cuda_mmb_cache_lookup(ctx, gate);
    if (xn16 && g16) {
        const int threads = 256; const int blocks = (int) ((n_items + threads - 1) / threads);
        const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());
        ggml_cuda_kernel_launch(hc_mix_reduce_bf16, launch_params, xn16, g16, out, n_embd, n_tokens, args.hc, args.scale, args.bias);
    } else if (args.hc == 4 && n_tokens <= 32) {
        // one wave per stream, the same in-order sum: bit-identical to the serial kernel with more blocks in flight for decode
        const int blocks = (int) ((n_items + 31) / 32);
        const ggml_cuda_kernel_launch_params launch_params(blocks, 128, 0, ctx.stream());
        ggml_cuda_kernel_launch(hc_mix_reduce_f32_hc4_parallel, launch_params,
            (const float *) xn->data, (const float *) gate->data, out, n_embd, n_tokens, args.scale, args.bias);
    } else {
        const int threads = 256;
        const int blocks = (int) ((n_items + threads - 1) / threads);
        const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());
        ggml_cuda_kernel_launch(hc_mix_reduce_f32, launch_params,
            (const float *) xn->data, (const float *) gate->data, out, n_embd, n_tokens, args.hc, args.scale, args.bias);
    }

    if (stage) {
        CUDA_CHECK(cudaMemcpyAsync(dst->data, out, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, ctx.stream()));
    }
}

void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_combine_args & args) {
    const ggml_tensor * residual  = args.residual;
    const ggml_tensor * block_out = args.block_out;
    const ggml_tensor * inject    = args.inject;
    ggml_tensor *       dst       = args.dst;

    GGML_ASSERT(residual->type == GGML_TYPE_F32 && block_out->type == GGML_TYPE_F32 &&
                inject->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(residual) && ggml_is_contiguous(block_out) &&
                ggml_is_contiguous(inject) && ggml_is_contiguous(dst));

    const int64_t n_embd   = dst->ne[0];
    const int64_t hc       = dst->ne[1];
    const int64_t n_tokens = dst->ne[2] * dst->ne[3];
    GGML_ASSERT(ggml_are_same_shape(residual, dst));
    GGML_ASSERT(ggml_nelements(block_out) == n_embd * n_tokens);
    GGML_ASSERT(ggml_nelements(inject) == hc * n_tokens);

    // residual may be the in-place source of the add (same address, same layout): each thread then
    // reads and writes its own element only. Any other overlap needs staging.
    const bool residual_inplace = residual->data == dst->data && ggml_are_same_layout(residual, dst);
    const bool stage = (!residual_inplace && hc_ranges_overlap(dst, residual)) ||
                       hc_ranges_overlap(dst, block_out) || hc_ranges_overlap(dst, inject);

    ggml_cuda_pool_alloc<float> staged(ctx.pool());
    float * out = stage ? staged.alloc(ggml_nelements(dst)) : (float *) dst->data;

    const int64_t n_items = ggml_nelements(dst);
    const int     threads = 256;
    const int     blocks  = (int) ((n_items + threads - 1) / threads);
    const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());
    ggml_cuda_kernel_launch(hc_combine_f32, launch_params,
        (const float *) residual->data, (const float *) block_out->data, (const float *) inject->data, out,
        n_embd, (int) hc, n_tokens, args.s1, args.b1, args.s2, args.b2);

    if (stage) {
        CUDA_CHECK(cudaMemcpyAsync(dst->data, out, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, ctx.stream()));
    }
}

// ---------------------------------------------------------------------------------------------------------
// combine + grouped rms norm, one token: block per stream, 1024 threads (the rms_norm_f32<1024> layout)
//   w        = s2 * sigmoid(s1 * inject[c] + b1) + b2
//   res[e,c] = residual[e,c] + block_out[e] * w
//   xn[e,c]  = rms_norm(res[:,c]) * gamma[e,c]

#define HC_CN_BLOCK   1024
#define HC_CN_MAX_EMB 3072

static __global__ void __launch_bounds__(HC_CN_BLOCK, 1) hc_combine_norm_f32(
        const float * __restrict__ inject, const float * __restrict__ residual,
        const float * __restrict__ block_out, const float * __restrict__ gamma,
        float * __restrict__ out_res, float * __restrict__ out_xn,
        const int n_embd, const float s1, const float b1, const float s2, const float b2, const float eps) {
    __shared__ float s_sum[32];

    const int c   = blockIdx.x;   // stream
    const int t   = blockIdx.y;   // token
    const int hc  = gridDim.x;
    const int tid = threadIdx.x;

    // identical expressions to scale_f32 / op_sigmoid / scale_f32
    const float x1 = s1 * inject[(int64_t) t * hc + c] + b1;
    const float x2 = hc_sigmoid(x1);
    const float w  = s2 * x2 + b2;

    const int64_t row = (int64_t) t * hc + c;
    const float * res = residual + row * n_embd;
    float *       dst = out_res  + row * n_embd;
    block_out += (int64_t) t * n_embd;
    float xs[3];
    float tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int col = tid + k * HC_CN_BLOCK;
        xs[k] = 0.0f;
        if (col < n_embd) {
            const float m  = hc_mul_rn(block_out[col], w);
            const float xi = hc_add_rn(res[col], m);
            dst[col] = xi;
            xs[k]    = xi;
            tmp += xi * xi;
        }
    }

    tmp = block_reduce<block_reduce_method::SUM, HC_CN_BLOCK>(tmp, s_sum);

    const float mean  = tmp / n_embd;
    const float scale = rsqrtf(mean + eps);

    const float * g  = gamma  + (int64_t) c * n_embd;
    float *       xn = out_xn + row * n_embd;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int col = tid + k * HC_CN_BLOCK;
        if (col < n_embd) {
            xn[col] = scale * xs[k] * g[col];
        }
    }
}

// Read all residual, block and injection values before writing aliased outputs.
// Gamma must remain disjoint from both outputs.
#define HC_CN_SINGLE_MAX_HC 4

static __global__ void __launch_bounds__(HC_CN_BLOCK, 1) hc_combine_norm_single_f32(
        const float * inject, const float * residual, const float * block_out, const float * __restrict__ gamma,
        float * out_res, float * out_xn,
        const int n_embd, const int hc, const float s1, const float b1, const float s2, const float b2, const float eps) {
    __shared__ float s_sum[HC_CN_SINGLE_MAX_HC][32];

    const int t       = blockIdx.x;   // token
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // 1. every input element of the token into registers
    float w[HC_CN_SINGLE_MAX_HC];
#pragma unroll
    for (int c = 0; c < HC_CN_SINGLE_MAX_HC; ++c) {
        w[c] = 0.0f;
        if (c < hc) {
            // identical expressions to scale_f32 / op_sigmoid / scale_f32
            const float x1 = s1 * inject[(int64_t) t * hc + c] + b1;
            const float x2 = hc_sigmoid(x1);
            w[c] = s2 * x2 + b2;
        }
    }

    block_out += (int64_t) t * n_embd;
    float xs[HC_CN_SINGLE_MAX_HC][3];
    float tmp[HC_CN_SINGLE_MAX_HC];
#pragma unroll
    for (int c = 0; c < HC_CN_SINGLE_MAX_HC; ++c) {
        tmp[c] = 0.0f;
    }
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int col = tid + k * HC_CN_BLOCK;
        const float bo = col < n_embd ? block_out[col] : 0.0f;
#pragma unroll
        for (int c = 0; c < HC_CN_SINGLE_MAX_HC; ++c) {
            xs[c][k] = 0.0f;
            if (c < hc && col < n_embd) {
                const float m  = hc_mul_rn(bo, w[c]);
                const float xi = hc_add_rn(residual[((int64_t) t * hc + c) * n_embd + col], m);
                xs[c][k] = xi;
                tmp[c] += xi * xi;
            }
        }
    }
    __syncthreads(); // all loads of the block have completed before the first store

    // 2. per-stream block sums, same reduction tree as block_reduce<SUM, 1024> for each stream
#pragma unroll
    for (int c = 0; c < HC_CN_SINGLE_MAX_HC; ++c) {
        tmp[c] = warp_reduce_sum(tmp[c]);
    }
    if (lane_id == 0) {
#pragma unroll
        for (int c = 0; c < HC_CN_SINGLE_MAX_HC; ++c) {
            s_sum[c][warp_id] = tmp[c];
        }
    }
    __syncthreads();
#pragma unroll
    for (int c = 0; c < HC_CN_SINGLE_MAX_HC; ++c) {
        tmp[c] = lane_id < HC_CN_BLOCK / WARP_SIZE ? s_sum[c][lane_id] : 0.0f;
        tmp[c] = warp_reduce_sum(tmp[c]);
    }

    // Finish residual writes before an overlapping normalized output replaces them.
#pragma unroll
    for (int c = 0; c < HC_CN_SINGLE_MAX_HC; ++c) {
        if (c < hc) {
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                const int col = tid + k * HC_CN_BLOCK;
                if (col < n_embd) {
                    out_res[((int64_t) t * hc + c) * n_embd + col] = xs[c][k];
                }
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int c = 0; c < HC_CN_SINGLE_MAX_HC; ++c) {
        if (c < hc) {
            const float mean  = tmp[c] / n_embd;
            const float scale = rsqrtf(mean + eps);
            const int64_t row = (int64_t) t * hc + c;
            float *       xn  = out_xn  + row * n_embd;
            const float * g   = gamma   + (int64_t) c * n_embd;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                const int col = tid + k * HC_CN_BLOCK;
                if (col < n_embd) {
                    xn[col]  = scale * xs[c][k] * g[col];
                }
            }
        }
    }
}

#define HC_CN_BLOCK2 256
#define HC_INJ_HC    4   // streams (and inject outputs) the fused combine + inject kernel handles
__device__ __forceinline__ float hc_lo(const uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hc_hi(const uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ uint32_t hc_pack2(const float a, const float b) {
    return (uint32_t) hc_f2bf32(a) | ((uint32_t) hc_f2bf32(b) << 16);
}

// One (token, stream) row of the combine + norm; with INJECT it also accumulates the row's share of the next mix's inject projection.
template <bool INJECT>
static __device__ __forceinline__ void hc_combine_norm_row_b256(
        const int c, const int t, const int hc, float * s_sum,
        const float * inject, const float * residual,
        const float * block_out, const float * gamma,
        float * out_res, float * out_xn, uint16_t * out_xn_bf16, const bool store_xn_f32,
        const uint16_t * res_in_bf16, uint16_t * res_out_bf16, const uint16_t * blk_in_bf16,
        const int n_embd, const float s1, const float b1, const float s2, const float b2, const float eps,
        const float * w_inj, float * inj,
        uint8_t * out_xn_q8, float * out_xn_scale, const int q8_blocks, const int n_tokens) {
    const int tid = threadIdx.x;
    const float x1 = s1 * inject[(int64_t) t * hc + c] + b1;
    const float x2 = hc_sigmoid(x1);
    const float w  = s2 * x2 + b2;
    const int64_t row = (int64_t) t * hc + c;
    const float *    res   = residual + row * n_embd;
    const uint16_t * res16 = res_in_bf16  ? res_in_bf16  + row * n_embd : nullptr;
    float *          dst   = out_res      + row * n_embd;
    uint16_t *       dst16 = res_out_bf16 ? res_out_bf16 + row * n_embd : nullptr;
    const float *    blk   = block_out + (int64_t) t * n_embd;
    const uint16_t * blk16 = blk_in_bf16 ? blk_in_bf16 + (int64_t) t * n_embd : nullptr;
    constexpr int KP = (HC_CN_MAX_EMB / 2 + HC_CN_BLOCK2 - 1) / HC_CN_BLOCK2;
    float xs[2 * KP];
    float tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < KP; ++k) {
        const int col = (tid + k * HC_CN_BLOCK2) * 2;
        xs[2 * k] = 0.0f; xs[2 * k + 1] = 0.0f;
        if (col + 1 < n_embd) {
            float r0, r1, m0, m1;
            if (res16) { const uint32_t u = *(const uint32_t *)(res16 + col); r0 = hc_lo(u); r1 = hc_hi(u); }
            else       { const float2   v = *(const float2 *)(res + col);     r0 = v.x;      r1 = v.y; }
            if (blk16) { const uint32_t u = *(const uint32_t *)(blk16 + col); m0 = hc_lo(u); m1 = hc_hi(u); }
            else       { const float2   v = *(const float2 *)(blk + col);     m0 = v.x;      m1 = v.y; }
            const float a0 = hc_add_rn(r0, hc_mul_rn(m0, w));
            const float a1 = hc_add_rn(r1, hc_mul_rn(m1, w));
            if (dst16) *(uint32_t *)(dst16 + col) = hc_pack2(a0, a1);
            else       *(float2 *)(dst + col)     = make_float2(a0, a1);
            xs[2 * k] = a0; xs[2 * k + 1] = a1;
            tmp += a0 * a0 + a1 * a1;
        } else if (col < n_embd) {
            const float r0 = res16 ? hc_bf2f32(res16[col]) : res[col];
            const float m0 = blk16 ? hc_bf2f32(blk16[col]) : blk[col];
            const float a0 = hc_add_rn(r0, hc_mul_rn(m0, w));
            if (dst16) dst16[col] = hc_f2bf32(a0); else dst[col] = a0;
            xs[2 * k] = a0;
            tmp += a0 * a0;
        }
    }
    tmp = block_reduce<block_reduce_method::SUM, HC_CN_BLOCK2>(tmp, s_sum);
    const float mean  = tmp / n_embd;
    const float scale = rsqrtf(mean + eps);
    const float * g  = gamma  + (int64_t) c * n_embd;
    float *       xn = out_xn + row * n_embd;
    uint16_t *    xh = out_xn_bf16 ? out_xn_bf16 + row * n_embd : nullptr;
#pragma unroll
    for (int k = 0; k < KP; ++k) {
        const int col = (tid + k * HC_CN_BLOCK2) * 2;
        if (col + 1 < n_embd) {
            const float2 gv = *(const float2 *)(g + col);
            const float v0 = scale * xs[2 * k] * gv.x, v1 = scale * xs[2 * k + 1] * gv.y;
            if (store_xn_f32) *(float2 *)(xn + col) = make_float2(v0, v1);
            if (xh) *(uint32_t *)(xh + col) = hc_pack2(v0, v1);
            if (out_xn_q8) {
                // per-32-element-block absmax over the 16-lane group (j = col/2), then the int8 codes and scale
                float amax = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
                for (int off = 1; off < 16; off <<= 1) amax = fmaxf(amax, __shfl_xor(amax, off));
                const float d = amax / 127.0f;
                const float id = d > 0.0f ? 1.0f / d : 0.0f;
                const int j = col >> 1;
                const int kb = c * q8_blocks + (j >> 4);
                uint8_t * q8p = out_xn_q8 + ((size_t) kb * n_tokens + t) * 32 + ((j & 15) << 1);
                q8p[0] = (uint8_t)(int8_t) roundf(v0 * id);
                q8p[1] = (uint8_t)(int8_t) roundf(v1 * id);
                if ((j & 15) == 0) out_xn_scale[(size_t) kb * n_tokens + t] = d;
            }
            if constexpr (INJECT) {
#pragma unroll
                for (int r = 0; r < HC_INJ_HC; ++r) {
                    const float2 wv = *(const float2 *)(w_inj + (int64_t) r * hc * n_embd + (int64_t) c * n_embd + col);
                    inj[r] += wv.x * v0 + wv.y * v1;
                }
            }
        } else if (col < n_embd) {
            const float v0 = scale * xs[2 * k] * g[col];
            if (store_xn_f32) xn[col] = v0;
            if (xh) xh[col] = hc_f2bf32(v0);
            if constexpr (INJECT) {
#pragma unroll
                for (int r = 0; r < HC_INJ_HC; ++r) {
                    inj[r] += w_inj[(int64_t) r * hc * n_embd + (int64_t) c * n_embd + col] * v0;
                }
            }
        }
    }
}

static __global__ void __launch_bounds__(HC_CN_BLOCK2, 4) hc_combine_norm_f32_b256(
        const float * inject, const float * residual,
        const float * block_out, const float * gamma,
        float * out_res, float * out_xn, uint16_t * out_xn_bf16, const bool store_xn_f32,
        const uint16_t * res_in_bf16, uint16_t * res_out_bf16, const uint16_t * blk_in_bf16,
        const int n_embd, const float s1, const float b1, const float s2, const float b2, const float eps,
        uint8_t * out_xn_q8, float * out_xn_scale, const int q8_blocks, const int n_tokens) {
    __shared__ float s_sum[32];
    hc_combine_norm_row_b256<false>(blockIdx.x, blockIdx.y, gridDim.x, s_sum, inject, residual, block_out, gamma,
        out_res, out_xn, out_xn_bf16, store_xn_f32, res_in_bf16, res_out_bf16, blk_in_bf16,
        n_embd, s1, b1, s2, b2, eps, nullptr, nullptr, out_xn_q8, out_xn_scale, q8_blocks, n_tokens);
}

// One block per token over all HC_INJ_HC streams: the rows of hc_combine_norm_f32_b256 plus the next mix's inject projection.
static __global__ void __launch_bounds__(HC_CN_BLOCK2, 2) hc_combine_norm_inject_f32_b256(
        const float * inject, const float * residual,
        const float * block_out, const float * gamma,
        float * out_res, float * out_xn, uint16_t * out_xn_bf16, const bool store_xn_f32,
        const uint16_t * res_in_bf16, uint16_t * res_out_bf16, const uint16_t * blk_in_bf16,
        const int n_embd, const float s1, const float b1, const float s2, const float b2, const float eps,
        const float * w_inject, float * out_inj,
        uint8_t * out_xn_q8, float * out_xn_scale, const int q8_blocks, const int n_tokens) {
    __shared__ float s_sum[HC_INJ_HC][32];
    __shared__ float s_inj[HC_INJ_HC][HC_CN_BLOCK2 / WARP_SIZE];
    const int t = blockIdx.x;
    float inj[HC_INJ_HC] = {};
#pragma unroll
    for (int c = 0; c < HC_INJ_HC; ++c) {
        hc_combine_norm_row_b256<true>(c, t, HC_INJ_HC, s_sum[c], inject, residual, block_out, gamma,
            out_res, out_xn, out_xn_bf16, store_xn_f32, res_in_bf16, res_out_bf16, blk_in_bf16,
            n_embd, s1, b1, s2, b2, eps, w_inject, inj, out_xn_q8, out_xn_scale, q8_blocks, n_tokens);
    }
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
#pragma unroll
    for (int r = 0; r < HC_INJ_HC; ++r) {
        const float v = warp_reduce_sum(inj[r]);
        if (lane == 0) {
            s_inj[r][warp] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < HC_INJ_HC) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < HC_CN_BLOCK2 / WARP_SIZE; ++w) {
            sum += s_inj[threadIdx.x][w];
        }
        out_inj[(int64_t) t * HC_INJ_HC + threadIdx.x] = sum;
    }
}

// ---------------------------------------------------------------------------------------------------------
// hc_combine_norm_inject_v2 (hc_combine_norm_inject_f32_b256 when hc_v2_supported refuses the shape): combine +
// grouped norm + inject of one token per block. Thread i owns columns [8i, 8i+8) of all HC_INJ_HC streams
// (blockDim = n_embd/8):
//   * every load of the token (4 x 16 B residual, block_out once for the 4 streams) is issued before any compute,
//   * the 4 per-stream sums share one block reduction (one barrier instead of four serialized ones),
//   * stores are 16 B (residual / xn bf16) and 8 B (q8),
//   * the int8 copy of xn goes to the token-grouped layout [t/32][kb][32 tokens][32] (+ scales [t/32][kb][32]):
//     the legacy [kb][t][32] layout spread each token's 320 blocks 64 KB apart (one 32 B and one 4 B store per
//     block), which cost ~0.3 ms per call on its own. The W8A8 HC down GEMM reads the grouped layout (bit-identical).
// The element-wise math (combine, rms scale * x * gamma, bf16 and q8 rounding) uses the b256 kernel's expressions;
// only the summation order of the rms and inject reductions differs (near-bitwise).
#define HC_V2_COLS 8
#define HC_V2_QGS  5   // q8 token group 1 << 5 = 32 tokens

template <bool RIN16, bool BLK16>
static __global__ void __launch_bounds__(1024) hc_combine_norm_inject_v2(
        const float * __restrict__ inject, const void * res_in, const void * __restrict__ blk_in,
        const float * __restrict__ gamma,
        float * out_res32, uint16_t * out_res16, float * __restrict__ out_xn, uint16_t * __restrict__ out_xn16,
        const int n_embd, const float s1, const float b1, const float s2, const float b2, const float eps,
        const float * __restrict__ w_inject, float * __restrict__ out_inj,
        uint8_t * __restrict__ out_q8, float * __restrict__ out_q8s, const int q8_blocks) {
    constexpr int HC = HC_INJ_HC;
    constexpr int NC = HC_V2_COLS;
    __shared__ float s_red[HC][32];
    __shared__ float s_inj[HC][32];
    const int tid  = threadIdx.x;
    const int nw   = blockDim.x / WARP_SIZE;
    const int warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    const int e0   = tid * NC;
    const int t    = blockIdx.x;

    // 1. all loads of the token up front (res_in may alias out_res: each element is read before it is written)
    uint4  rr16[HC];
    float4 rr32[HC][2];
    uint4  bb16;
    float4 bb32[2];
    float  wc[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        // identical expressions to scale_f32 / op_sigmoid / scale_f32
        const float x1 = s1 * inject[(int64_t) t * HC + c] + b1;
        const float x2 = hc_sigmoid(x1);
        wc[c] = s2 * x2 + b2;
    }
    if constexpr (BLK16) bb16 = *(const uint4 *)((const uint16_t *) blk_in + (int64_t) t * n_embd + e0);
    else { const float4 * b = (const float4 *)((const float *) blk_in + (int64_t) t * n_embd + e0); bb32[0] = b[0]; bb32[1] = b[1]; }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t off = ((int64_t) t * HC + c) * n_embd + e0;
        if constexpr (RIN16) rr16[c] = *(const uint4 *)((const uint16_t *) res_in + off);
        else { const float4 * r = (const float4 *)((const float *) res_in + off); rr32[c][0] = r[0]; rr32[c][1] = r[1]; }
    }

    // 2. combine, residual stores, per-stream partial sums
    float m[NC];
    if constexpr (BLK16) {
        const uint32_t u[4] = { bb16.x, bb16.y, bb16.z, bb16.w };
#pragma unroll
        for (int q = 0; q < 4; ++q) { m[2 * q] = hc_lo(u[q]); m[2 * q + 1] = hc_hi(u[q]); }
    } else {
        m[0] = bb32[0].x; m[1] = bb32[0].y; m[2] = bb32[0].z; m[3] = bb32[0].w;
        m[4] = bb32[1].x; m[5] = bb32[1].y; m[6] = bb32[1].z; m[7] = bb32[1].w;
    }
    float a[HC][NC];
    float ss[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        float r[NC];
        if constexpr (RIN16) {
            const uint32_t u[4] = { rr16[c].x, rr16[c].y, rr16[c].z, rr16[c].w };
#pragma unroll
            for (int q = 0; q < 4; ++q) { r[2 * q] = hc_lo(u[q]); r[2 * q + 1] = hc_hi(u[q]); }
        } else {
            r[0] = rr32[c][0].x; r[1] = rr32[c][0].y; r[2] = rr32[c][0].z; r[3] = rr32[c][0].w;
            r[4] = rr32[c][1].x; r[5] = rr32[c][1].y; r[6] = rr32[c][1].z; r[7] = rr32[c][1].w;
        }
        float s = 0.0f;
#pragma unroll
        for (int q = 0; q < NC; ++q) {
            a[c][q] = hc_add_rn(r[q], hc_mul_rn(m[q], wc[c]));
            s += a[c][q] * a[c][q];
        }
        ss[c] = s;
        const int64_t off = ((int64_t) t * HC + c) * n_embd + e0;
        if (out_res16) {
            uint4 o;
            o.x = hc_pack2(a[c][0], a[c][1]); o.y = hc_pack2(a[c][2], a[c][3]);
            o.z = hc_pack2(a[c][4], a[c][5]); o.w = hc_pack2(a[c][6], a[c][7]);
            *(uint4 *)(out_res16 + off) = o;
        } else {
            float4 * d = (float4 *)(out_res32 + off);
            d[0] = make_float4(a[c][0], a[c][1], a[c][2], a[c][3]);
            d[1] = make_float4(a[c][4], a[c][5], a[c][6], a[c][7]);
        }
    }

    // 3. one block reduction for the HC sums; the gamma loads overlap the barrier
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_reduce_sum(ss[c]);
        if (lane == 0) s_red[c][warp] = v;
    }
    float g[HC][NC];
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float4 * gp = (const float4 *)(gamma + (int64_t) c * n_embd + e0);
        const float4 g0 = gp[0], g1 = gp[1];
        g[c][0] = g0.x; g[c][1] = g0.y; g[c][2] = g0.z; g[c][3] = g0.w;
        g[c][4] = g1.x; g[c][5] = g1.y; g[c][6] = g1.z; g[c][7] = g1.w;
    }
    __syncthreads();
    float scale[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        float v = lane < nw ? s_red[c][lane] : 0.0f;
        v = warp_reduce_sum(v);
        const float mean = v / n_embd;
        scale[c] = rsqrtf(mean + eps);
    }

    // 4. xn (bf16 / f32), q8, inject projection
    float inj[HC] = {};
    const int nkb = HC * q8_blocks;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        float v[NC];
#pragma unroll
        for (int q = 0; q < NC; ++q) v[q] = scale[c] * a[c][q] * g[c][q];
        const int64_t off = ((int64_t) t * HC + c) * n_embd + e0;
        if (out_xn16) {
            uint4 o;
            o.x = hc_pack2(v[0], v[1]); o.y = hc_pack2(v[2], v[3]);
            o.z = hc_pack2(v[4], v[5]); o.w = hc_pack2(v[6], v[7]);
            *(uint4 *)(out_xn16 + off) = o;
        }
        if (out_xn) {
            float4 * d = (float4 *)(out_xn + off);
            d[0] = make_float4(v[0], v[1], v[2], v[3]);
            d[1] = make_float4(v[4], v[5], v[6], v[7]);
        }
        if (out_q8) {
            // per-32-element-block absmax over the 4 lanes of the block, then the int8 codes and scale
            float amax = 0.0f;
#pragma unroll
            for (int q = 0; q < NC; ++q) amax = fmaxf(amax, fabsf(v[q]));
            amax = fmaxf(amax, __shfl_xor(amax, 1));
            amax = fmaxf(amax, __shfl_xor(amax, 2));
            const float d  = amax / 127.0f;
            const float id = d > 0.0f ? 1.0f / d : 0.0f;
            const int kb = c * q8_blocks + (e0 >> 5);
            const size_t qi = ((((size_t) t >> HC_V2_QGS) * nkb + kb) << HC_V2_QGS) + (t & ((1 << HC_V2_QGS) - 1));
            uint32_t lo = 0, hi = 0;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                lo |= (uint32_t) (uint8_t) (int8_t) roundf(v[q]     * id) << (8 * q);
                hi |= (uint32_t) (uint8_t) (int8_t) roundf(v[q + 4] * id) << (8 * q);
            }
            *(uint2 *)(out_q8 + qi * 32 + (e0 & 31)) = make_uint2(lo, hi);
            if ((tid & 3) == 0) out_q8s[qi] = d;
        }
#pragma unroll
        for (int r = 0; r < HC; ++r) {
            const float4 * wp = (const float4 *)(w_inject + (int64_t) r * HC * n_embd + (int64_t) c * n_embd + e0);
            const float4 w0 = wp[0], w1 = wp[1];
            inj[r] += w0.x * v[0] + w0.y * v[1] + w0.z * v[2] + w0.w * v[3] +
                      w1.x * v[4] + w1.y * v[5] + w1.z * v[6] + w1.w * v[7];
        }
    }
#pragma unroll
    for (int r = 0; r < HC; ++r) {
        const float v = warp_reduce_sum(inj[r]);
        if (lane == 0) s_inj[r][warp] = v;
    }
    __syncthreads();
    if (tid < HC) {
        float sum = 0.0f;
        for (int w = 0; w < nw; ++w) sum += s_inj[tid][w];
        out_inj[(int64_t) t * HC + tid] = sum;
    }
}

static bool hc_v2_supported(const int64_t n_embd, const int64_t hc) {
    return hc == HC_INJ_HC && n_embd % 256 == 0 && n_embd / HC_V2_COLS <= 1024;
}

static void hc_v2_launch(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_combine_norm_args & a,
        uint8_t * q8, float * q8s, float * out_res32, uint16_t * out_res16, float * out_xn, uint16_t * out_xn16, float * out_inj) {
    const int n_embd   = (int) a.out_res->ne[0];
    const int n_tokens = (int) (a.out_res->ne[2] * a.out_res->ne[3]);
    if (q8) ggml_cuda_mmb_xnq8_set_layout(ctx, HC_V2_QGS);
    const ggml_cuda_kernel_launch_params lp(dim3(n_tokens, 1, 1), n_embd / HC_V2_COLS, 0, ctx.stream());
    const void * rin = a.res_in_bf16 ? (const void *) a.res_in_bf16 : a.residual->data;
    const void * bin = a.blk_in_bf16 ? (const void *) a.blk_in_bf16 : a.block_out->data;
#define HC_V2_ARGS (const float *) a.inject->data, rin, bin, (const float *) a.gamma->data, \
        out_res32, out_res16, out_xn, out_xn16, n_embd, a.s1, a.b1, a.s2, a.b2, a.eps, \
        (const float *) a.w_inject->data, out_inj, q8, q8s, n_embd / 32
    if (a.res_in_bf16) {
        if (a.blk_in_bf16) ggml_cuda_kernel_launch(hc_combine_norm_inject_v2<true,  true>,  lp, HC_V2_ARGS);
        else               ggml_cuda_kernel_launch(hc_combine_norm_inject_v2<true,  false>, lp, HC_V2_ARGS);
    } else {
        if (a.blk_in_bf16) ggml_cuda_kernel_launch(hc_combine_norm_inject_v2<false, true>,  lp, HC_V2_ARGS);
        else               ggml_cuda_kernel_launch(hc_combine_norm_inject_v2<false, false>, lp, HC_V2_ARGS);
    }
#undef HC_V2_ARGS
}

bool ggml_cuda_hc_combine_norm_supported(const ggml_cuda_hc_combine_norm_args & a, const int warp_size) {
    const int64_t n_embd = a.out_res->ne[0];
    const int64_t hc     = a.out_res->ne[1];
    return warp_size == 32 && n_embd >= 1024 && n_embd <= HC_CN_MAX_EMB && hc >= 1 && hc <= 16 &&
        ggml_nelements(a.gamma) == n_embd * hc;
}

void ggml_cuda_op_hc_combine_norm(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_combine_norm_args & a) {
    const int64_t n_embd   = a.out_res->ne[0];
    const int64_t hc       = a.out_res->ne[1];
    const int64_t n_tokens = a.out_res->ne[2] * a.out_res->ne[3];

    GGML_ASSERT(a.out_res->ne[3] == 1);
    GGML_ASSERT(ggml_are_same_shape(a.out_res, a.out_xn) && ggml_are_same_shape(a.out_res, a.residual));
    GGML_ASSERT(ggml_nelements(a.block_out) == n_embd * n_tokens && ggml_nelements(a.gamma) == n_embd * hc &&
                ggml_nelements(a.inject) == hc * n_tokens);

    // int8 copy of xn for the W8A8 HC down: allocated once and written by the row kernels below; the HC down
    // GEMM reads it through ggml_cuda_mmb_xnq8_get when its src1 root matches.
    ggml_cuda_hc_combine_norm_args aa = a;
    if (!a.single_block && (n_embd % 32) == 0 && n_embd <= HC_CN_MAX_EMB && n_tokens >= 512) {
        // the v2 rows kernel writes token groups of 1 << HC_V2_QGS: the last group is padded, not truncated
        const int q8_blocks = (int) (n_embd / 32);
        const int64_t n_tok_pad = GGML_PAD(n_tokens, 1 << HC_V2_QGS);
        const size_t q8b = (size_t) q8_blocks * hc * n_tok_pad * 32;
        const size_t q8s = (size_t) q8_blocks * hc * n_tok_pad * sizeof(float);
        const ggml_tensor * xroot = a.out_xn->view_src ? a.out_xn->view_src : a.out_xn;
        aa.out_xn_q8    = ggml_cuda_mmb_xnq8_buf(ctx, xroot, q8b, q8s);
        aa.out_xn_scale = ggml_cuda_mmb_xnq8_scale(ctx);
    }

    if (a.single_block) {
        GGML_ASSERT(hc <= HC_CN_SINGLE_MAX_HC);
        GGML_ASSERT(a.store_xn_f32 && !a.out_xn_bf16 && !a.res_in_bf16 && !a.res_out_bf16 && !a.blk_in_bf16);
        const ggml_cuda_kernel_launch_params launch_params(dim3((int) n_tokens, 1, 1), HC_CN_BLOCK, 0, ctx.stream());
        ggml_cuda_kernel_launch(hc_combine_norm_single_f32, launch_params,
            (const float *) a.inject->data, (const float *) a.residual->data,
            (const float *) a.block_out->data, (const float *) a.gamma->data,
            (float *) a.out_res->data, (float *) a.out_xn->data,
            (int) n_embd, (int) hc, a.s1, a.b1, a.s2, a.b2, a.eps);
        return;
    }

    if (a.out_inject) {
        GGML_ASSERT(hc == HC_INJ_HC && a.w_inject && ggml_nelements(a.out_inject) == hc * n_tokens &&
                    a.w_inject->ne[0] == n_embd * hc && a.w_inject->ne[1] == hc);
        // the grouped q8 layout pads the last 32-token group (see the q8 buffer above), so any token count takes it
        const bool v2 = hc_v2_supported(n_embd, hc);
        float *    o_res32 = a.res_out_bf16 ? nullptr : (float *) a.out_res->data;
        uint16_t * o_res16 = a.res_out_bf16;
        float *    o_xn32  = a.store_xn_f32 ? (float *) a.out_xn->data : nullptr;
        if (v2) {
            hc_v2_launch(ctx, aa, aa.out_xn_q8, aa.out_xn_scale, o_res32, o_res16, o_xn32, a.out_xn_bf16, (float *) a.out_inject->data);
            return;
        }
        const ggml_cuda_kernel_launch_params lpi(dim3((int) n_tokens, 1, 1), HC_CN_BLOCK2, 0, ctx.stream());
        ggml_cuda_kernel_launch(hc_combine_norm_inject_f32_b256, lpi,
            (const float *) a.inject->data, (const float *) a.residual->data,
            (const float *) a.block_out->data, (const float *) a.gamma->data,
            (float *) a.out_res->data, (float *) a.out_xn->data, a.out_xn_bf16, a.store_xn_f32,
            a.res_in_bf16, a.res_out_bf16, a.blk_in_bf16,
            (int) n_embd, a.s1, a.b1, a.s2, a.b2, a.eps,
            (const float *) a.w_inject->data, (float *) a.out_inject->data,
            aa.out_xn_q8, aa.out_xn_scale, (int) (n_embd / 32), (int) n_tokens);
        return;
    }

    // 256 threads x 2 packed elements per iteration (the shipped layout); the BF16 options ride on this kernel
    const ggml_cuda_kernel_launch_params lp2(dim3((int) hc, (int) n_tokens, 1), HC_CN_BLOCK2, 0, ctx.stream());
    ggml_cuda_kernel_launch(hc_combine_norm_f32_b256, lp2,
        (const float *) a.inject->data, (const float *) a.residual->data,
        (const float *) a.block_out->data, (const float *) a.gamma->data,
        (float *) a.out_res->data, (float *) a.out_xn->data, a.out_xn_bf16, a.store_xn_f32,
        a.res_in_bf16, a.res_out_bf16, a.blk_in_bf16,
        (int) n_embd, a.s1, a.b1, a.s2, a.b2, a.eps,
        aa.out_xn_q8, aa.out_xn_scale, (int) (n_embd / 32), (int) n_tokens);
}
