#include "dsv4.cuh"

#include "convert.cuh"

#include <cfloat>

#define CUDA_DSV4_BLOCK_SIZE 256
#define CUDA_DSV4_MAX_GRIDDIM_X 0x7FFFFFFF

struct dsv4_rope_corr_dims {
    float v[2];
};

static __device__ __forceinline__ float dsv4_sigmoidf(const float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __device__ float dsv4_rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

static __device__ void dsv4_rope_yarn(
        const float theta_extrap,
        const float freq_scale,
        const dsv4_rope_corr_dims corr_dims,
        const int64_t i0,
        const float ext_factor,
        float mscale,
        const float sin_sign,
        float & cos_theta,
        float & sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float ramp_mix = dsv4_rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale * sin_sign;
}

static __device__ float dsv4_e4m3fn_dequant(const float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);

    int best = 0;
    float best_diff = ax;

    for (int i = 1; i < 127; ++i) {
        const int exp  = (i >> 3) & 0x0f;
        const int mant = i & 0x07;
        const float val = exp == 0
            ? ldexpf(float(mant), -9)
            : ldexpf(1.0f + float(mant) / 8.0f, exp - 7);
        const float diff = fabsf(ax - val);
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }

    const int exp  = (best >> 3) & 0x0f;
    const int mant = best & 0x07;
    const float val = exp == 0
        ? ldexpf(float(mant), -9)
        : ldexpf(1.0f + float(mant) / 8.0f, exp - 7);

    return sign * val;
}

static int64_t dsv4_nblocks(const int64_t n) {
    return MIN((int64_t) CUDA_DSV4_MAX_GRIDDIM_X, (n + CUDA_DSV4_BLOCK_SIZE - 1) / CUDA_DSV4_BLOCK_SIZE);
}

static __global__ void dsv4_hc_split_sinkhorn_f32(
        const float * __restrict__ mixes,
        const float * __restrict__ scale,
        const float * __restrict__ base,
        float * __restrict__ dst,
        const int n_hc,
        const int sinkhorn_iters,
        const float eps,
        const int64_t n_rows,
        const int64_t mix_s0,
        const int64_t mix_s1,
        const int64_t dst_s0,
        const int64_t dst_s1) {
    for (int64_t r = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; r < n_rows; r += (int64_t) blockDim.x * gridDim.x) {
        const float pre_scale  = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        float c[16*16];

        for (int i = 0; i < n_hc; ++i) {
            const float z = mixes[r*mix_s1 + i*mix_s0] * pre_scale + base[i];
            dst[r*dst_s1 + i*dst_s0] = dsv4_sigmoidf(z) + eps;
        }

        for (int i = 0; i < n_hc; ++i) {
            const int off = n_hc + i;
            const float z = mixes[r*mix_s1 + off*mix_s0] * post_scale + base[off];
            dst[r*dst_s1 + off*dst_s0] = 2.0f * dsv4_sigmoidf(z);
        }

        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            float row_max = ggml_cuda_negative_infinity();
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                const int idx = src_hc + dst_hc*n_hc;
                const int off = 2*n_hc + idx;
                const float v = mixes[r*mix_s1 + off*mix_s0] * comb_scale + base[off];
                c[idx] = v;
                row_max = fmaxf(row_max, v);
            }

            float row_sum = 0.0f;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                const int idx = src_hc + dst_hc*n_hc;
                const float v = expf(c[idx] - row_max);
                c[idx] = v;
                row_sum += v;
            }

            const float inv_sum = 1.0f / row_sum;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                const int idx = src_hc + dst_hc*n_hc;
                c[idx] = c[idx] * inv_sum + eps;
            }
        }

        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                sum += c[src_hc + dst_hc*n_hc];
            }

            const float inv_denom = 1.0f / (sum + eps);
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                c[src_hc + dst_hc*n_hc] *= inv_denom;
            }
        }

        for (int iter = 1; iter < sinkhorn_iters; ++iter) {
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                float sum = 0.0f;
                for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                    sum += c[src_hc + dst_hc*n_hc];
                }

                const float inv_denom = 1.0f / (sum + eps);
                for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                    c[src_hc + dst_hc*n_hc] *= inv_denom;
                }
            }

            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                float sum = 0.0f;
                for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                    sum += c[src_hc + dst_hc*n_hc];
                }

                const float inv_denom = 1.0f / (sum + eps);
                for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                    c[src_hc + dst_hc*n_hc] *= inv_denom;
                }
            }
        }

        for (int i = 0; i < n_hc*n_hc; ++i) {
            const int off = 2*n_hc + i;
            dst[r*dst_s1 + off*dst_s0] = c[i];
        }
    }
}

static __global__ void dsv4_hc_weighted_sum_f32(
        const float * __restrict__ x,
        const float * __restrict__ weights,
        float * __restrict__ dst,
        const int64_t n_embd,
        const int64_t n_hc,
        const int64_t n_tokens,
        const int64_t x_s0,
        const int64_t x_s1,
        const int64_t x_s2,
        const int64_t w_s0,
        const int64_t w_s1,
        const int64_t dst_s0,
        const int64_t dst_s1) {
    const int64_t n_elem = n_embd * n_tokens;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n_elem; i += (int64_t) blockDim.x * gridDim.x) {
        const int64_t d = i % n_embd;
        const int64_t t = i / n_embd;

        float acc = 0.0f;
        for (int64_t h = 0; h < n_hc; ++h) {
            acc += x[d*x_s0 + h*x_s1 + t*x_s2] * weights[h*w_s0 + t*w_s1];
        }

        dst[d*dst_s0 + t*dst_s1] = acc;
    }
}

static __global__ void dsv4_hc_expand_f32(
        const float * __restrict__ block_out,
        const float * __restrict__ residual,
        const float * __restrict__ post,
        const float * __restrict__ comb,
        float * __restrict__ dst,
        const int64_t n_embd,
        const int64_t n_hc,
        const int64_t n_tokens,
        const int64_t block_s0,
        const int64_t block_s1,
        const int64_t residual_s0,
        const int64_t residual_s1,
        const int64_t residual_s2,
        const int64_t post_s0,
        const int64_t post_s1,
        const int64_t comb_s0,
        const int64_t comb_s1,
        const int64_t comb_s2,
        const int64_t dst_s0,
        const int64_t dst_s1,
        const int64_t dst_s2) {
    const int64_t n_elem = n_embd * n_hc * n_tokens;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n_elem; i += (int64_t) blockDim.x * gridDim.x) {
        const int64_t d      = i % n_embd;
        const int64_t tmp    = i / n_embd;
        const int64_t dst_hc = tmp % n_hc;
        const int64_t t      = tmp / n_hc;

        float acc = block_out[d*block_s0 + t*block_s1] * post[dst_hc*post_s0 + t*post_s1];
        for (int64_t src_hc = 0; src_hc < n_hc; ++src_hc) {
            acc += comb[dst_hc*comb_s0 + src_hc*comb_s1 + t*comb_s2] *
                   residual[d*residual_s0 + src_hc*residual_s1 + t*residual_s2];
        }

        dst[d*dst_s0 + dst_hc*dst_s1 + t*dst_s2] = acc;
    }
}

static __global__ void dsv4_fp8_kv_quantize_f32(
        const float * __restrict__ src,
        float * __restrict__ dst,
        const int64_t head_dim,
        const int64_t n_nope,
        const int64_t n_rows,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t src_s0,
        const int64_t src_s1,
        const int64_t src_s2,
        const int64_t src_s3,
        const int64_t dst_s0,
        const int64_t dst_s1,
        const int64_t dst_s2,
        const int64_t dst_s3) {
    const int64_t n_blocks_per_row = n_nope / 64;
    const int64_t n_tasks = n_rows * n_blocks_per_row;
    for (int64_t task = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; task < n_tasks; task += (int64_t) blockDim.x * gridDim.x) {
        const int64_t row = task / n_blocks_per_row;
        const int64_t block = task % n_blocks_per_row;
        const int64_t off0 = block * 64;

        const int64_t i1 = row % ne1;
        const int64_t i2 = (row / ne1) % ne2;
        const int64_t i3 = row / (ne1 * ne2);

        const int64_t src_row = i1*src_s1 + i2*src_s2 + i3*src_s3;
        const int64_t dst_row = i1*dst_s1 + i2*dst_s2 + i3*dst_s3;

        float amax = 0.0f;
        for (int64_t i = 0; i < 64; ++i) {
            amax = fmaxf(amax, fabsf(src[src_row + (off0 + i)*src_s0]));
        }

        amax = fmaxf(amax, 1.0e-4f);
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        for (int64_t i = 0; i < 64; ++i) {
            const float v = src[src_row + (off0 + i)*src_s0];
            dst[dst_row + (off0 + i)*dst_s0] =
                dsv4_e4m3fn_dequant(fminf(fmaxf(v / scale, -448.0f), 448.0f)) * scale;
        }
    }

    const int64_t n_tail_tasks = n_rows * (head_dim - n_nope);
    for (int64_t task = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; task < n_tail_tasks; task += (int64_t) blockDim.x * gridDim.x) {
        const int64_t row = task / (head_dim - n_nope);
        const int64_t off = n_nope + task % (head_dim - n_nope);

        const int64_t i1 = row % ne1;
        const int64_t i2 = (row / ne1) % ne2;
        const int64_t i3 = row / (ne1 * ne2);

        dst[i1*dst_s1 + i2*dst_s2 + i3*dst_s3 + off*dst_s0] =
            src[i1*src_s1 + i2*src_s2 + i3*src_s3 + off*src_s0];
    }
}

template <typename T>
static __global__ void dsv4_rope_tail_kernel(
        const T * __restrict__ src,
        const int32_t * __restrict__ pos,
        const float * __restrict__ freq_factors,
        T * __restrict__ dst,
        const int64_t ne0,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3,
        const int64_t src_s1,
        const int64_t src_s2,
        const int64_t src_s3,
        const int64_t dst_s1,
        const int64_t dst_s2,
        const int64_t dst_s3,
        const int n_dims,
        const int mode,
        const float freq_scale,
        const float ext_factor,
        const float attn_factor,
        const dsv4_rope_corr_dims corr_dims,
        const float theta_scale,
        const float sin_sign) {
    const int64_t n_elem = ne0 * ne1 * ne2 * ne3;
    const int64_t n_nope = ne0 - n_dims;

    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; idx < n_elem; idx += (int64_t) blockDim.x * gridDim.x) {
        const int64_t i0 = idx % ne0;
        const int64_t tmp1 = idx / ne0;
        const int64_t i1 = tmp1 % ne1;
        const int64_t tmp2 = tmp1 / ne1;
        const int64_t i2 = tmp2 % ne2;
        const int64_t i3 = tmp2 / ne2;

        const int64_t src_base = i1*src_s1 + i2*src_s2 + i3*src_s3;
        const int64_t dst_base = i1*dst_s1 + i2*dst_s2 + i3*dst_s3;

        if (i0 < n_nope) {
            dst[dst_base + i0] = src[src_base + i0];
            continue;
        }

        const int64_t tail_i = i0 - n_nope;
        if (mode == GGML_ROPE_TYPE_NORMAL) {
            if ((tail_i & 1) != 0) {
                continue;
            }

            const int64_t d0 = n_nope + tail_i;
            const float theta_base = pos[i2] * powf(theta_scale, tail_i / 2.0f);
            const float freq_factor = freq_factors ? freq_factors[tail_i / 2] : 1.0f;

            float cos_theta;
            float sin_theta;
            dsv4_rope_yarn(theta_base / freq_factor, freq_scale, corr_dims, tail_i, ext_factor, attn_factor, sin_sign, cos_theta, sin_theta);

            const float x0 = ggml_cuda_cast<float>(src[src_base + d0 + 0]);
            const float x1 = ggml_cuda_cast<float>(src[src_base + d0 + 1]);
            dst[dst_base + d0 + 0] = ggml_cuda_cast<T>(x0*cos_theta - x1*sin_theta);
            dst[dst_base + d0 + 1] = ggml_cuda_cast<T>(x0*sin_theta + x1*cos_theta);
        } else if (mode == GGML_ROPE_TYPE_NEOX) {
            if (tail_i >= n_dims / 2) {
                continue;
            }

            const int64_t d0 = n_nope + tail_i;
            const int64_t d1 = d0 + n_dims / 2;
            const int64_t rope_i = 2 * tail_i;
            const float theta_base = pos[i2] * powf(theta_scale, rope_i / 2.0f);
            const float freq_factor = freq_factors ? freq_factors[rope_i / 2] : 1.0f;

            float cos_theta;
            float sin_theta;
            dsv4_rope_yarn(theta_base / freq_factor, freq_scale, corr_dims, rope_i, ext_factor, attn_factor, sin_sign, cos_theta, sin_theta);

            const float x0 = ggml_cuda_cast<float>(src[src_base + d0]);
            const float x1 = ggml_cuda_cast<float>(src[src_base + d1]);
            dst[dst_base + d0] = ggml_cuda_cast<T>(x0*cos_theta - x1*sin_theta);
            dst[dst_base + d1] = ggml_cuda_cast<T>(x0*sin_theta + x1*cos_theta);
        }
    }
}

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type  == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);

    const int n_hc           = ggml_get_op_params_i32(dst, 0);
    const int sinkhorn_iters = ggml_get_op_params_i32(dst, 1);
    const float eps          = ggml_get_op_params_f32(dst, 2);
    const int64_t n_rows     = ggml_nrows(mixes);

    dsv4_hc_split_sinkhorn_f32<<<dsv4_nblocks(n_rows), CUDA_DSV4_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
        n_hc, sinkhorn_iters, eps, n_rows,
        mixes->nb[0] / (int64_t) sizeof(float), mixes->nb[1] / (int64_t) sizeof(float),
        dst->nb[0] / (int64_t) sizeof(float), dst->nb[1] / (int64_t) sizeof(float));
}

void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type       == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type     == GGML_TYPE_F32);

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_hc     = x->ne[1];
    const int64_t n_tokens = dst->ne[1];
    const int64_t n_elem   = n_embd * n_tokens;

    dsv4_hc_weighted_sum_f32<<<dsv4_nblocks(n_elem), CUDA_DSV4_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) weights->data, (float *) dst->data,
        n_embd, n_hc, n_tokens,
        x->nb[0] / (int64_t) sizeof(float), x->nb[1] / (int64_t) sizeof(float), x->nb[2] / (int64_t) sizeof(float),
        weights->nb[0] / (int64_t) sizeof(float), weights->nb[1] / (int64_t) sizeof(float),
        dst->nb[0] / (int64_t) sizeof(float), dst->nb[1] / (int64_t) sizeof(float));
}

void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * block_out = dst->src[0];
    const ggml_tensor * residual  = dst->src[1];
    const ggml_tensor * post      = dst->src[2];
    const ggml_tensor * comb      = dst->src[3];

    GGML_ASSERT(block_out->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type  == GGML_TYPE_F32);
    GGML_ASSERT(post->type      == GGML_TYPE_F32);
    GGML_ASSERT(comb->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type       == GGML_TYPE_F32);

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_hc     = dst->ne[1];
    const int64_t n_tokens = dst->ne[2];
    const int64_t n_elem   = n_embd * n_hc * n_tokens;

    dsv4_hc_expand_f32<<<dsv4_nblocks(n_elem), CUDA_DSV4_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) block_out->data, (const float *) residual->data, (const float *) post->data, (const float *) comb->data,
        (float *) dst->data, n_embd, n_hc, n_tokens,
        block_out->nb[0] / (int64_t) sizeof(float), block_out->nb[1] / (int64_t) sizeof(float),
        residual->nb[0] / (int64_t) sizeof(float), residual->nb[1] / (int64_t) sizeof(float), residual->nb[2] / (int64_t) sizeof(float),
        post->nb[0] / (int64_t) sizeof(float), post->nb[1] / (int64_t) sizeof(float),
        comb->nb[0] / (int64_t) sizeof(float), comb->nb[1] / (int64_t) sizeof(float), comb->nb[2] / (int64_t) sizeof(float),
        dst->nb[0] / (int64_t) sizeof(float), dst->nb[1] / (int64_t) sizeof(float), dst->nb[2] / (int64_t) sizeof(float));
}

void ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t n_rot = ggml_get_op_params_i32(dst, 0);
    const int64_t head_dim = src0->ne[0];
    const int64_t n_nope = head_dim - n_rot;
    const int64_t n_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const int64_t n_tasks = n_rows * (n_nope / 64 + n_rot);

    dsv4_fp8_kv_quantize_f32<<<dsv4_nblocks(n_tasks), CUDA_DSV4_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) src0->data, (float *) dst->data,
        head_dim, n_nope, n_rows, src0->ne[1], src0->ne[2],
        src0->nb[0] / (int64_t) sizeof(float), src0->nb[1] / (int64_t) sizeof(float),
        src0->nb[2] / (int64_t) sizeof(float), src0->nb[3] / (int64_t) sizeof(float),
        dst->nb[0] / (int64_t) sizeof(float), dst->nb[1] / (int64_t) sizeof(float),
        dst->nb[2] / (int64_t) sizeof(float), dst->nb[3] / (int64_t) sizeof(float));
}

void ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const int n_dims     = ((int32_t *) dst->op_params)[0];
    const int mode       = ((int32_t *) dst->op_params)[1];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[2];
    const bool inverse   = ((int32_t *) dst->op_params)[3] != 0;

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *) dst->op_params + 4, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 9, sizeof(float));

    dsv4_rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    const int64_t n_elem = ggml_nelements(src0);
    const int64_t type_size = ggml_type_size(src0->type);
    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];
    const float sin_sign = inverse ? -1.0f : 1.0f;
    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    const float * freq_factors = src2 == nullptr ? nullptr : (const float *) src2->data;

    if (src0->type == GGML_TYPE_F32) {
        dsv4_rope_tail_kernel<float><<<dsv4_nblocks(n_elem), CUDA_DSV4_BLOCK_SIZE, 0, ctx.stream()>>>(
            (const float *) src0->data, (const int32_t *) src1->data, freq_factors, (float *) dst->data,
            ne0, ne1, ne2, ne3,
            src0->nb[1] / type_size, src0->nb[2] / type_size, src0->nb[3] / type_size,
            dst->nb[1] / type_size, dst->nb[2] / type_size, dst->nb[3] / type_size,
            n_dims, mode, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale, sin_sign);
    } else if (src0->type == GGML_TYPE_F16) {
        dsv4_rope_tail_kernel<half><<<dsv4_nblocks(n_elem), CUDA_DSV4_BLOCK_SIZE, 0, ctx.stream()>>>(
            (const half *) src0->data, (const int32_t *) src1->data, freq_factors, (half *) dst->data,
            ne0, ne1, ne2, ne3,
            src0->nb[1] / type_size, src0->nb[2] / type_size, src0->nb[3] / type_size,
            dst->nb[1] / type_size, dst->nb[2] / type_size, dst->nb[3] / type_size,
            n_dims, mode, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale, sin_sign);
    } else {
        GGML_ABORT("unsupported DeepSeek V4 RoPE tail type");
    }
}
