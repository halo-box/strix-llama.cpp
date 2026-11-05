#include "unary.cuh"
#include "convert.cuh"

static __device__ __forceinline__ float op_abs(float x) {
    return fabsf(x);
}

static __device__ __forceinline__ float op_sgn(float x) {
    return (x > 0.f ? 1.f : ((x < 0.f ? -1.f : 0.f)));
}

static __device__ __forceinline__ float op_neg(float x) {
    return -x;
}

static __device__ __forceinline__ float op_step(float x) {
    return x > 0.0f;
}

static __device__ __forceinline__ float op_gelu(float x) {
    return ggml_cuda_op_gelu_single(x);
}

static __device__ __forceinline__ float op_gelu_erf(float x) {
    const float SQRT_2_INV = 0.70710678118654752440084436210484f;

    return 0.5f*x*(1.0f + erff(x*SQRT_2_INV));
}

static __device__ __forceinline__ float op_gelu_quick(float x) {
    const float GELU_QUICK_COEF = -1.702f;

    return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

static __device__ __forceinline__ float op_silu(float x) {
    return ggml_cuda_op_silu_single(x);
}

static __device__ __forceinline__ float op_tanh(float x) {
    return tanhf(x);
}

static __device__ __forceinline__ float op_relu(float x) {
    return fmaxf(x, 0);
}

static __device__ __forceinline__ float op_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __device__ __forceinline__ float op_hardsigmoid(float x) {
    return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_hardswish(float x) {
    return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_exp(float x) {
    return expf(x);
}

static __device__ __forceinline__ float op_sqr(float x) {
    return x * x;
}

static __device__ __forceinline__ float op_relu_sqr(float x) {
    const float r = fmaxf(x, 0.0f);
    return r * r;
}

static __device__ __forceinline__ float op_sqrt(float x) {
    return sqrtf(x);
}

static __device__ __forceinline__ float op_sin(float x) {
    return sinf(x);
}

static __device__ __forceinline__ float op_cos(float x) {
    return cosf(x);
}

static __device__ __forceinline__ float op_log(float x) {
    return logf(x);
}

static __device__ __forceinline__ float op_expm1(float x) {
    return expm1f(x);
}

static __device__ __forceinline__ float op_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

static __device__ __forceinline__ float op_elu(float x) {
    return (x > 0.f) ? x : expm1f(x);
}

static __device__ __forceinline__ float op_floor(float x) {
    return floorf(x);
}

static __device__ __forceinline__ float op_ceil(float x) {
    return ceilf(x);
}

static __device__ __forceinline__ float op_round(float x) {
    return roundf(x);
}

static __device__ __forceinline__ float op_trunc(float x) {
    return trunc(x);
}

template <float (*op)(float), typename T>
static __global__ void unary_op_kernel(const T * x, T * dst, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    dst[i] = ggml_cuda_cast<T>(op(ggml_cuda_cast<float>(x[i])));
}

template <float (*op)(float), typename T>
static void unary_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(unary_op_kernel<op, T>, launch_params, x, dst, k);
}

template <float (*op)(float), typename T>
static __global__ void unary_op_kernel_strided(const T * x, T * dst, const int k,const int64_t ne00,const int64_t ne01,const int64_t ne02,const size_t nb00,const size_t nb01,const size_t nb02,const size_t nb03) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    int64_t rem = i;
    const int64_t i0 = rem % ne00; rem /= ne00;
    const int64_t i1 = rem % ne01; rem /= ne01;
    const int64_t i2 = rem % ne02;
    const int64_t i3 = rem / ne02;
    const size_t src_byte_offset = i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03;
    const T * src_ptr = (const T *)((const char *)x + src_byte_offset);

    ggml_cuda_pdl_sync();
    dst[i] = ggml_cuda_cast<T>(op(ggml_cuda_cast<float>(*src_ptr)));
}

template <float (*op)(float), typename T>
static void unary_cuda_strided(const T * x, T * dst, const int k,const int64_t ne00,const int64_t ne01,const int64_t ne02,const size_t nb00,const size_t nb01,const size_t nb02,const size_t nb03, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(unary_op_kernel_strided<op, T>, launch_params, x, dst, k, ne00,ne01,ne02,nb00,nb01,nb02,nb03);
}

template <float (*op)(float)>
void ggml_cuda_op_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src0->type == dst->type);

    if (ggml_is_contiguous(src0)) {
        if (src0->type == GGML_TYPE_F16) {
            unary_cuda<op>((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
        } else if (src0->type == GGML_TYPE_BF16) {
            unary_cuda<op>((const nv_bfloat16 *)src0_d, (nv_bfloat16 *)dst_d, ggml_nelements(src0), stream);
        } else {
            unary_cuda<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
        }
    } else {
        if (src0->type == GGML_TYPE_F16) {
            unary_cuda_strided<op>((const half *)src0_d, (half *)dst_d, ggml_nelements(src0),
                src0->ne[0], src0->ne[1], src0->ne[2],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], stream);
        } else if (src0->type == GGML_TYPE_BF16) {
            unary_cuda_strided<op>((const nv_bfloat16 *)src0_d, (nv_bfloat16 *)dst_d, ggml_nelements(src0),
                src0->ne[0], src0->ne[1], src0->ne[2],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], stream);
        } else {
            unary_cuda_strided<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0),
                src0->ne[0], src0->ne[1], src0->ne[2],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], stream);
        }
    }
}

void ggml_cuda_op_abs(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_abs>(ctx, dst);
}

void ggml_cuda_op_sgn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sgn>(ctx, dst);
}

void ggml_cuda_op_neg(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_neg>(ctx, dst);
}

void ggml_cuda_op_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_step>(ctx, dst);
}

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu>(ctx, dst);
}

void ggml_cuda_op_gelu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu_erf>(ctx, dst);
}

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu_quick>(ctx, dst);
}

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_silu>(ctx, dst);
}

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_tanh>(ctx, dst);
}

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_relu>(ctx, dst);
}

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sigmoid>(ctx, dst);
}

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_hardsigmoid>(ctx, dst);
}

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_hardswish>(ctx, dst);
}

void ggml_cuda_op_exp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_exp>(ctx, dst);
}

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sqr>(ctx, dst);
}

void ggml_cuda_op_sqrt(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sqrt>(ctx, dst);
}

void ggml_cuda_op_sin(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sin>(ctx, dst);
}

void ggml_cuda_op_cos(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_cos>(ctx, dst);
}

void ggml_cuda_op_log(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_log>(ctx, dst);
}

void ggml_cuda_op_elu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_elu>(ctx, dst);
}

void ggml_cuda_op_floor(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_floor>(ctx, dst);
}

void ggml_cuda_op_ceil(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_ceil>(ctx, dst);
}

void ggml_cuda_op_round(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_round>(ctx, dst);
}

void ggml_cuda_op_trunc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_trunc>(ctx, dst);
}

void ggml_cuda_op_expm1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_expm1>(ctx, dst);
}

void ggml_cuda_op_softplus(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_softplus>(ctx, dst);
}
/* gated ops */

template <float (*op)(float), typename T>
static __global__ void unary_gated_op_kernel(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1,
        uint16_t * dst16, const bool dst16_f16) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    // perform base op and multiply with gate (either offset in same tensor or a separate one)
    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ggml_cuda_pdl_sync();
    const T v = ggml_cuda_cast<T>(op(ggml_cuda_cast<float>(x[j0])) * ggml_cuda_cast<float>(g[j1]));
    dst[i] = v;
    if (dst16) {
        dst16[i] = ggml_cuda_f32_to_act16(ggml_cuda_cast<float>(v), dst16_f16);
    }
}

template <float (*op)(float), typename T>
static void unary_gated_cuda(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, cudaStream_t stream,
        uint16_t * dst16 = nullptr, const bool dst16_f16 = false) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(unary_gated_op_kernel<op, T>, launch_params, x, g, dst, k, n, o0, o1, dst16, dst16_f16);
}

template <float (*op)(float)>
void ggml_cuda_op_unary_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst, uint16_t * dst16 = nullptr) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    const int32_t swapped = ((const int32_t *) dst->op_params)[1];

    if (src0->type == GGML_TYPE_F16) {
        half * src0_p = (half *) src0_d;
        half * src1_p = (half *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_cuda<op>(src0_p, src1_p, (half *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(half), src1_o / sizeof(half), stream);
    } else if (src0->type == GGML_TYPE_BF16) {
        nv_bfloat16 * src0_p = (nv_bfloat16 *) src0_d;
        nv_bfloat16 * src1_p = (nv_bfloat16 *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_cuda<op>(src0_p, src1_p, (nv_bfloat16 *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(nv_bfloat16), src1_o / sizeof(nv_bfloat16), stream);
    } else {
        float * src0_p = (float *) src0_d;
        float * src1_p = (float *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_cuda<op>(src0_p, src1_p, (float *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), stream, dst16);
    }
}

void ggml_cuda_op_reglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_relu>(ctx, dst);
}

void ggml_cuda_op_geglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu>(ctx, dst);
}

void ggml_cuda_op_swiglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst, uint16_t * dst16) {
    ggml_cuda_op_unary_gated<op_silu>(ctx, dst, dst16);
}

void ggml_cuda_op_geglu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu_erf>(ctx, dst);
}

void ggml_cuda_op_geglu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu_quick>(ctx, dst);
}

// swiglu_oai

template <typename T>
static __global__ void swiglu_oai_kernel(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, float alpha, float limit) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    // perform base op and multiply with gate (either offset in same tensor or a separate one)
    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    float xi = x[j0];
    float gi = g[j1];

    dst[i] = ggml_cuda_op_swiglu_oai_single(xi, gi, alpha, limit);
}

template <typename T>
static void swiglu_oai_cuda(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, const float alpha, const float limit, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    swiglu_oai_kernel<<<num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(x, g, dst, k, n, o0, o1, alpha, limit);
}

void ggml_cuda_op_swiglu_oai(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float alpha = ggml_get_op_params_f32(dst, 2);
    const float limit = ggml_get_op_params_f32(dst, 3);

    float * src0_p = (float *) src0_d;
    float * src1_p = (float *) src1_d;

    if (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_cuda(src0_p, src1_p, (float *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), alpha, limit, stream);
}

// swiglu_clamp

template <typename T>
static __global__ void swiglu_clamp_kernel(const T * gate, const T * up, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, float limit) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    dst[i] = (T) ggml_cuda_op_swiglu_clamp_single((float) gate[j0], (float) up[j1], limit);
}

template <typename T>
static void swiglu_clamp_cuda(const T * gate, const T * up, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, const float limit, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    swiglu_clamp_kernel<<<num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(gate, up, dst, k, n, o0, o1, limit);
}

void ggml_cuda_op_swiglu_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float limit = ggml_get_op_params_f32(dst, 3);

    if (src0->type == GGML_TYPE_F16) {
        half * src0_p = (half *) src0_d;
        half * src1_p = (half *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_cuda(src0_p, src1_p, (half *) dst_d, ggml_nelements(dst), nc, src0_o / sizeof(half), src1_o / sizeof(half), limit, stream);
    } else {
        float * src0_p = (float *) src0_d;
        float * src1_p = (float *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_cuda(src0_p, src1_p, (float *) dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), limit, stream);
    }
}

/* CUDA kernel + launcher for xIELU */

template <typename T>
static __global__ void xielu_kernel(const T * x, T * dst, const int k, float alpha_n, float alpha_p, float beta, float eps) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const float xi = ggml_cuda_cast<float>(x[i]);

    const float gate_pos = (xi > 0.0f);
    const float y_pos = alpha_p * xi * xi + beta * xi;
    const float min_v_eps = fminf(xi, eps);
    const float y_neg = (expm1f(min_v_eps) - xi) * alpha_n + beta * xi;
    const float out = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;

    dst[i] = ggml_cuda_cast<T>(out);
}

template <typename T>
static void xielu_cuda(const T * x, T * dst, const int k, float alpha_n, float alpha_p, float beta, float eps, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_XIELU_BLOCK_SIZE) / CUDA_XIELU_BLOCK_SIZE;
    xielu_kernel<<<num_blocks, CUDA_XIELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, alpha_n, alpha_p, beta, eps);
}

void ggml_cuda_op_xielu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16 ||  dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(src0->type == dst->type);

    const float alpha_n = ggml_get_op_params_f32(dst, 1);
    const float alpha_p = ggml_get_op_params_f32(dst, 2);
    const float beta    = ggml_get_op_params_f32(dst, 3);
    const float eps     = ggml_get_op_params_f32(dst, 4);

    if (src0->type == GGML_TYPE_F16) {
        xielu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    } else if (src0->type == GGML_TYPE_BF16) {
        xielu_cuda((const nv_bfloat16 *)src0_d, (nv_bfloat16 *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    } else {
        xielu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    }
}



/* silu_back */

static __device__ __forceinline__ float op_silu_back(float grad, float x) {
    const float s = 1.0f / (1.0f + expf(-x));
    return grad * s * (1.0f + x * (1.0f - s));
}

template <class T>
static __global__ void silu_back_kernel(const T * grad, const T * xf, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op_silu_back((float)grad[i], (float)xf[i]);
}

template <class T>
static void silu_back_cuda(const T * grad, const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BACK_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_back_kernel<<<num_blocks, CUDA_SILU_BACK_BLOCK_SIZE, 0, stream>>>(grad, x, dst, k);
}

void ggml_cuda_op_silu_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // input from forward pass
    const ggml_tensor * src1 = dst->src[1]; // grads of forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        silu_back_cuda((const half *)src0_d, (const half *)src1_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        silu_back_cuda((const float*)src0_d, (const float*)src1_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

/* leaky relu */

static __device__ __forceinline__ float op_leaky_relu(float x, const float negative_slope) {
    return fmaxf(x, 0) + fminf(x, 0.0f) * negative_slope;
}

template <class T>
static __global__ void leaky_relu_kernel(const T * x, T * dst, const int k, const float negative_slope) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op_leaky_relu((float)x[i], negative_slope);
}

template <class T>
static void leaky_relu_cuda(const T * x, T * dst, const int k, const float negative_slope, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    leaky_relu_kernel<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, negative_slope);
}

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    if (src0->type == GGML_TYPE_F16) {
        leaky_relu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), negative_slope, stream);
    } else {
        leaky_relu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), negative_slope, stream);
    }
}

/* fused unary + mul */

template <float (*op)(float)>
static void ggml_cuda_op_unary_mul_impl(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node) {
    // unary_node: UNARY op applied to unary_node->src[0]
    // mul_node:   MUL(a, b) where one of a/b is unary_node
    // Output goes to mul_node->data

    const ggml_tensor * unary_src = unary_node->src[0];  // input to the unary op
    const ggml_tensor * other_src = (mul_node->src[0] == unary_node) ? mul_node->src[1] : mul_node->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(unary_src));
    GGML_ASSERT(unary_src->nb[0] == ggml_element_size(unary_src));
    GGML_ASSERT(ggml_is_contiguous_1(other_src));
    GGML_ASSERT(other_src->nb[0] == ggml_element_size(other_src));
    GGML_ASSERT(ggml_are_same_shape(unary_src, other_src));

    GGML_ASSERT(unary_src->type == GGML_TYPE_F32 || unary_src->type == GGML_TYPE_F16 || unary_src->type == GGML_TYPE_BF16);
    GGML_ASSERT(unary_src->type == other_src->type);
    GGML_ASSERT(unary_src->type == mul_node->type);

    cudaStream_t stream = ctx.stream();

    const int64_t k  = ggml_nelements(mul_node);
    const int64_t nc = unary_src->ne[0];
    const int64_t unary_stride = unary_src->nb[1];
    const int64_t other_stride = other_src->nb[1];

    if (unary_src->type == GGML_TYPE_F16) {
        unary_gated_cuda<op>((const half *) unary_src->data, (const half *) other_src->data,
                             (half *) mul_node->data, k, nc,
                             unary_stride / sizeof(half), other_stride / sizeof(half), stream);
    } else if (unary_src->type == GGML_TYPE_BF16) {
        unary_gated_cuda<op>((const nv_bfloat16 *) unary_src->data, (const nv_bfloat16 *) other_src->data,
                             (nv_bfloat16 *) mul_node->data, k, nc,
                             unary_stride / sizeof(nv_bfloat16), other_stride / sizeof(nv_bfloat16), stream);
    } else {
        unary_gated_cuda<op>((const float *) unary_src->data, (const float *) other_src->data,
                             (float *) mul_node->data, k, nc,
                             unary_stride / sizeof(float), other_stride / sizeof(float), stream);
    }
}

void ggml_cuda_op_unary_mul(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node) {
    switch (ggml_get_unary_op(unary_node)) {
        case GGML_UNARY_OP_SILU:
            ggml_cuda_op_unary_mul_impl<op_silu>(ctx, unary_node, mul_node);
            break;
        case GGML_UNARY_OP_SIGMOID:
            ggml_cuda_op_unary_mul_impl<op_sigmoid>(ctx, unary_node, mul_node);
            break;
        case GGML_UNARY_OP_SOFTPLUS:
            ggml_cuda_op_unary_mul_impl<op_softplus>(ctx, unary_node, mul_node);
            break;
        default:
            GGML_ABORT("Unsupported unary op for fused unary+mul");
    }
}

// The gated attention output sigmoid(gate) * attn in float4 lanes with 32-bit indexing (the scalar kernel, kept for
// unaligned or oversized buffers, spends a 64-bit division per element); same expression per element, so bit-identical.
// With store_f32 == false only the 16-bit copy for the MMB GEMM is written (the F32 output has no other reader).
template <bool F32>
static __global__ void cont_sigmoid_mul_v4_kernel(const float * __restrict__ gate, const float * __restrict__ x, float * __restrict__ dst,
        uint16_t * __restrict__ dst16, const bool f16, const int n4, const int gs4, const int k4) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k4) {
        return;
    }
    const int row = i / n4;
    const int c   = i - row * n4;
    const float4 g = ((const float4 *) gate)[(size_t) row * gs4 + c];
    const float4 v = ((const float4 *) x)[i];
    float4 r;
    r.x = op_sigmoid(g.x) * v.x;
    r.y = op_sigmoid(g.y) * v.y;
    r.z = op_sigmoid(g.z) * v.z;
    r.w = op_sigmoid(g.w) * v.w;
    if (F32) {
        ((float4 *) dst)[i] = r;
    }
    if (dst16) {
        uint2 h;
        h.x = (uint32_t) ggml_cuda_f32_to_act16(r.x, f16) | ((uint32_t) ggml_cuda_f32_to_act16(r.y, f16) << 16);
        h.y = (uint32_t) ggml_cuda_f32_to_act16(r.z, f16) | ((uint32_t) ggml_cuda_f32_to_act16(r.w, f16) << 16);
        ((uint2 *) dst16)[i] = h;
    }
}

void ggml_cuda_op_cont_sigmoid_mul(
        ggml_backend_cuda_context & ctx, ggml_tensor * cont_node, ggml_tensor * unary_node, ggml_tensor * mul_node, uint16_t * dst16,
        bool dst16_f16, bool store_f32) {
    const ggml_tensor * gate = cont_node->src[0];
    const ggml_tensor * other = mul_node->src[0] == unary_node ? mul_node->src[1] : mul_node->src[0];

    GGML_ASSERT(ggml_get_unary_op(unary_node) == GGML_UNARY_OP_SIGMOID);
    GGML_ASSERT(gate->type == GGML_TYPE_F32 && other->type == GGML_TYPE_F32 && mul_node->type == GGML_TYPE_F32);
    GGML_ASSERT(gate->nb[0] == sizeof(float) && gate->nb[2] == gate->nb[1] * gate->ne[1]);
    GGML_ASSERT(ggml_is_contiguous(other) && ggml_is_contiguous(mul_node));
    GGML_ASSERT(ggml_nelements(gate) == ggml_nelements(other));

    const int64_t k = ggml_nelements(other);
    const int64_t n = gate->ne[0];
    const int64_t gs = gate->nb[1] / sizeof(float);
    if (!store_f32) {
        GGML_ASSERT(dst16 != nullptr);
    }
    if (n % 4 == 0 && gs % 4 == 0 && k / 4 < INT32_MAX && (k / n) * gs < INT32_MAX &&
            ((uintptr_t) gate->data % 16) == 0 && ((uintptr_t) other->data % 16) == 0 && ((uintptr_t) mul_node->data % 16) == 0 &&
            ((uintptr_t) dst16 % 8) == 0) {
        const int k4 = (int) (k / 4);
        const dim3 grid((k4 + 255) / 256);
        if (store_f32) {
            cont_sigmoid_mul_v4_kernel<true><<<grid, 256, 0, ctx.stream()>>>((const float *) gate->data, (const float *) other->data,
                (float *) mul_node->data, dst16, dst16_f16, (int) (n / 4), (int) (gs / 4), k4);
        } else {
            cont_sigmoid_mul_v4_kernel<false><<<grid, 256, 0, ctx.stream()>>>((const float *) gate->data, (const float *) other->data,
                (float *) mul_node->data, dst16, dst16_f16, (int) (n / 4), (int) (gs / 4), k4);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // the scalar kernel always stores F32 next to the 16-bit copy: harmless when the F32 output has no reader
    unary_gated_cuda<op_sigmoid>((const float *) gate->data, (const float *) other->data, (float *) mul_node->data,
        k, n, gate->nb[1] / sizeof(float), n, ctx.stream(), dst16, dst16_f16);
}

/* fused relu + sqr */

void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node) {
    const ggml_tensor * src = relu_node->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    GGML_ASSERT(src->type == sqr_node->type);

    const int k = ggml_nelements(src);
    if (src->type == GGML_TYPE_F16) {
        unary_cuda<op_relu_sqr>((const half *)src->data, (half *)sqr_node->data, k, stream);
    } else {
        unary_cuda<op_relu_sqr>((const float *)src->data, (float *)sqr_node->data, k, stream);
    }
}

/* fused GDN gate: softplus(x + b) * a */

static __global__ void gdn_gate_kernel(const char * __restrict__ x, const int64_t nb1, const float * __restrict__ b, const float * __restrict__ a,
        float * __restrict__ dst, const int ne0, const int n) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int r = i / ne0, c = i - r * ne0;
    // op_add -> op_softplus -> op_mul, as the three unfused kernels
    const float s = *(const float *) (x + (int64_t) r * nb1 + (int64_t) c * sizeof(float)) + b[c];
    dst[i] = op_softplus(s) * a[c];
}

static bool gdn_gate_row(const ggml_tensor * t, const int64_t ne0) {
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[0] == ne0 && ggml_nrows(t) == 1;
}

bool ggml_cuda_gdn_gate_fusable(const ggml_tensor * add, const ggml_tensor * sp, const ggml_tensor * mul) {
    const ggml_tensor * x = add->src[0];
    return add->op == GGML_OP_ADD && sp->op == GGML_OP_UNARY && ggml_get_unary_op(sp) == GGML_UNARY_OP_SOFTPLUS && mul->op == GGML_OP_MUL &&
        sp->src[0] == add && mul->src[0] == sp && add->type == GGML_TYPE_F32 && sp->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 &&
        x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) && x->ne[2] == 1 && x->ne[3] == 1 && x->nb[1] % sizeof(float) == 0 &&
        gdn_gate_row(add->src[1], x->ne[0]) && gdn_gate_row(mul->src[1], x->ne[0]) &&
        ggml_are_same_shape(add, x) && ggml_are_same_shape(mul, x) && ggml_is_contiguous(mul) && ggml_nelements(mul) <= INT_MAX;
}

void ggml_cuda_op_gdn_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * add, ggml_tensor * mul) {
    const ggml_tensor * x = add->src[0];
    const int n = (int) ggml_nelements(mul), ne0 = (int) x->ne[0];
    const int nb = (n + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    gdn_gate_kernel<<<nb, CUDA_NEG_BLOCK_SIZE, 0, ctx.stream()>>>((const char *) x->data, (int64_t) x->nb[1], (const float *) add->src[1]->data,
        (const float *) mul->src[1]->data, (float *) mul->data, ne0, n);
    CUDA_CHECK(cudaGetLastError());
}

/* fused scale + unary */

template <float (*op)(float)>
static __global__ void scale_unary_kernel(const float * x, float * dst, const float scale, const float bias, const int k, uint16_t * dst16) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    // identical expression to scale_f32 in scale.cu, followed by the unary op
    const float v = op(scale * x[i] + bias);
    dst[i] = v;
    if (dst16) {
        dst16[i] = ggml_cuda_f32_to_bf16_rne(v);
    }
}

template <float (*op)(float)>
static void scale_unary_cuda(const float * x, float * dst, const float scale, const float bias, const int k, cudaStream_t stream,
        uint16_t * dst16) {
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(scale_unary_kernel<op>, launch_params, x, dst, scale, bias, k, dst16);
}

void ggml_cuda_op_scale_unary(ggml_backend_cuda_context & ctx, ggml_tensor * scale_node, ggml_tensor * unary_node, uint16_t * dst16) {
    const ggml_tensor * src = scale_node->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32 && scale_node->type == GGML_TYPE_F32 && unary_node->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(unary_node));
    GGML_ASSERT(unary_node->src[0] == scale_node);

    const float scale = ggml_get_op_params_f32(scale_node, 0);
    const float bias  = ggml_get_op_params_f32(scale_node, 1);
    const int   k     = ggml_nelements(src);

    switch (ggml_get_unary_op(unary_node)) {
        case GGML_UNARY_OP_SILU:
            scale_unary_cuda<op_silu>((const float *) src->data, (float *) unary_node->data, scale, bias, k, ctx.stream(), dst16);
            break;
        case GGML_UNARY_OP_SIGMOID:
            scale_unary_cuda<op_sigmoid>((const float *) src->data, (float *) unary_node->data, scale, bias, k, ctx.stream(), dst16);
            break;
        default:
            GGML_ABORT("unsupported unary op for fused scale+unary");
    }
}

/* prefill shared-expert merge: out = other + src * sigmoid(gate[t]) with one gate value per row (token); the product
   and the sum are rounded separately (no fma contraction), as the unfused MUL and ADD nodes -> bit-identical */
#if defined(GGML_USE_HIP)
static __device__ __forceinline__ float sgma_mul_rn(const float a, const float b) { float r; asm("v_mul_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r; }
static __device__ __forceinline__ float sgma_add_rn(const float a, const float b) { float r; asm("v_add_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r; }
#else
static __device__ __forceinline__ float sgma_mul_rn(const float a, const float b) { return __fmul_rn(a, b); }
static __device__ __forceinline__ float sgma_add_rn(const float a, const float b) { return __fadd_rn(a, b); }
#endif
static __global__ void sigmoid_gate_mul_add_f32(const float * __restrict__ src, const float * __restrict__ other,
        const float * __restrict__ gate, float * __restrict__ out, const int64_t ncols, const int64_t n) {
    const int64_t i = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) {
        return;
    }
    if (i + 4 <= n && (ncols % 4) == 0) {
        const float  g = op_sigmoid(gate[i / ncols]);
        const float4 a = *(const float4 *) (src + i), b = *(const float4 *) (other + i);
        *(float4 *) (out + i) = make_float4(sgma_add_rn(b.x, sgma_mul_rn(a.x, g)), sgma_add_rn(b.y, sgma_mul_rn(a.y, g)),
                                            sgma_add_rn(b.z, sgma_mul_rn(a.z, g)), sgma_add_rn(b.w, sgma_mul_rn(a.w, g)));
    } else {
        for (int64_t j = i; j < n && j < i + 4; ++j) {
            out[j] = sgma_add_rn(other[j], sgma_mul_rn(src[j], op_sigmoid(gate[j / ncols])));
        }
    }
}

void ggml_cuda_op_sigmoid_gate_mul_add(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * src,
        const ggml_tensor * other, ggml_tensor * out) {
    const int64_t n = ggml_nelements(out), ncols = out->ne[0];
    sigmoid_gate_mul_add_f32<<<(unsigned) ((n / 4 + 255) / 256 + 1), 256, 0, ctx.stream()>>>((const float *) src->data,
        (const float *) other->data, (const float *) gate->data, (float *) out->data, ncols, n);
    CUDA_CHECK(cudaGetLastError());
}
