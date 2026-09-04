#include "hyperconn.cuh"
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
    const int     threads = 256;
    const int     blocks  = (int) ((n_items + threads - 1) / threads);
    const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());
    ggml_cuda_kernel_launch(hc_mix_reduce_f32, launch_params,
        (const float *) xn->data, (const float *) gate->data, out, n_embd, n_tokens, args.hc, args.scale, args.bias);

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

// Same computation for one token with all streams in a single block, streams processed one after the other
// with exactly the per-stream expressions and block reduction above. Every input element is loaded into
// registers before the first store, so the outputs may alias any of the inputs (the graph allocator reuses
// the block_out buffer for xn when block_out dies at the combine).
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

    // 3. stores
#pragma unroll
    for (int c = 0; c < HC_CN_SINGLE_MAX_HC; ++c) {
        if (c < hc) {
            const float mean  = tmp[c] / n_embd;
            const float scale = rsqrtf(mean + eps);
            const int64_t row = (int64_t) t * hc + c;
            float *       dst = out_res + row * n_embd;
            float *       xn  = out_xn  + row * n_embd;
            const float * g   = gamma   + (int64_t) c * n_embd;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                const int col = tid + k * HC_CN_BLOCK;
                if (col < n_embd) {
                    dst[col] = xs[c][k];
                    xn[col]  = scale * xs[c][k] * g[col];
                }
            }
        }
    }
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

    if (a.single_block) {
        GGML_ASSERT(hc <= HC_CN_SINGLE_MAX_HC);
        const ggml_cuda_kernel_launch_params launch_params(dim3((int) n_tokens, 1, 1), HC_CN_BLOCK, 0, ctx.stream());
        ggml_cuda_kernel_launch(hc_combine_norm_single_f32, launch_params,
            (const float *) a.inject->data, (const float *) a.residual->data,
            (const float *) a.block_out->data, (const float *) a.gamma->data,
            (float *) a.out_res->data, (float *) a.out_xn->data,
            (int) n_embd, (int) hc, a.s1, a.b1, a.s2, a.b2, a.eps);
        return;
    }

    const ggml_cuda_kernel_launch_params launch_params(dim3((int) hc, (int) n_tokens, 1), HC_CN_BLOCK, 0, ctx.stream());
    ggml_cuda_kernel_launch(hc_combine_norm_f32, launch_params,
        (const float *) a.inject->data, (const float *) a.residual->data,
        (const float *) a.block_out->data, (const float *) a.gamma->data,
        (float *) a.out_res->data, (float *) a.out_xn->data,
        (int) n_embd, a.s1, a.b1, a.s2, a.b2, a.eps);
}
