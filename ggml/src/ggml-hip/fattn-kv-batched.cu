#include "../ggml-cuda/fattn-common.cuh"

#include <algorithm>
#include <cstdint>

namespace {

constexpr int64_t FATTN_KV_CONV_BATCH = 1024;
constexpr int     TURBO_FWHT_SIZE       = 128;
constexpr float   TURBO_FWHT_SCALE      = 0.08838834764831844f;

static bool is_turbo_type(ggml_type type) {
    return type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0;
}

template <int nbits>
static __global__ void k_turbo_to_f16_slice(
        const char * __restrict__ src,
        half       * __restrict__ dst,
        int64_t ne0,
        int64_t kv_start,
        int64_t kv_count,
        int64_t ne2,
        int64_t nb1,
        int64_t nb2,
        int64_t nb3) {
    __shared__ float values[TURBO_FWHT_SIZE];

    const int tid = threadIdx.x;
    const int64_t chunks_per_row = ne0 / TURBO_FWHT_SIZE;

    int64_t block = blockIdx.x;
    const int64_t chunk = block % chunks_per_row;
    block /= chunks_per_row;
    const int64_t token = block % kv_count;
    block /= kv_count;
    const int64_t head = block % ne2;
    const int64_t batch = block / ne2;

    const char * row = src + batch*nb3 + head*nb2 + (kv_start + token)*nb1;
    float norm;

    if constexpr (nbits == 3) {
        const block_turbo3_0 * blocks = (const block_turbo3_0 *) row + chunk*4;
        const block_turbo3_0 & x = blocks[tid / QK_TURBO3];
        const int elem = tid % QK_TURBO3;
        const int bit_offset = elem*3;
        const int byte_index = bit_offset / 8;
        const int shift = bit_offset % 8;
        uint16_t raw = (uint16_t) x.qs[byte_index] >> shift;
        if (shift > 5 && byte_index + 1 < 12) {
            raw |= (uint16_t) x.qs[byte_index + 1] << (8 - shift);
        }
        values[tid] = dc_codebook_3bit[raw & 0x07];
        norm = __half2float(blocks[0].d);
    } else {
        const block_turbo4_0 * blocks = (const block_turbo4_0 *) row + chunk*4;
        const block_turbo4_0 & x = blocks[tid / QK_TURBO4];
        const int elem = tid % QK_TURBO4;
        const uint8_t packed = x.qs[elem / 2];
        const uint8_t index = (elem & 1) ? packed >> 4 : packed & 0x0F;
        values[tid] = dc_codebook_4bit[index];
        norm = __half2float(blocks[0].d);
    }
    __syncthreads();

#pragma unroll
    for (int h = 1; h < TURBO_FWHT_SIZE; h <<= 1) {
        if (tid < TURBO_FWHT_SIZE/2) {
            const int group = tid / h;
            const int pos = tid % h;
            const int i = group*2*h + pos;
            const float a = values[i];
            const float b = values[i + h];
            values[i] = a + b;
            values[i + h] = a - b;
        }
        __syncthreads();
    }

    const int64_t out = (((batch*ne2 + head)*kv_count + token)*ne0 +
                         chunk*TURBO_FWHT_SIZE + tid);
    dst[out] = __float2half(values[tid] * (TURBO_FWHT_SCALE*norm));
}

static void convert_kv_slice(
        const ggml_tensor * src,
        half * dst,
        int64_t kv_start,
        int64_t kv_count,
        cudaStream_t stream) {
    GGML_ASSERT(src->ne[0] == 128 || src->ne[0] == 256);
    GGML_ASSERT(src->nb[0] == ggml_element_size(src));

    if (is_turbo_type(src->type)) {
        const int64_t nblocks = src->ne[3] * src->ne[2] * kv_count *
                                (src->ne[0] / TURBO_FWHT_SIZE);
        if (src->type == GGML_TYPE_TURBO3_0) {
            k_turbo_to_f16_slice<3><<<nblocks, TURBO_FWHT_SIZE, 0, stream>>>(
                (const char *) src->data, dst, src->ne[0], kv_start, kv_count,
                src->ne[2], src->nb[1], src->nb[2], src->nb[3]);
        } else {
            k_turbo_to_f16_slice<4><<<nblocks, TURBO_FWHT_SIZE, 0, stream>>>(
                (const char *) src->data, dst, src->ne[0], kv_start, kv_count,
                src->ne[2], src->nb[1], src->nb[2], src->nb[3]);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    GGML_ASSERT(src->type == GGML_TYPE_Q8_0);
    const size_t type_size = ggml_type_size(src->type);
    GGML_ASSERT(src->nb[1] % type_size == 0);
    GGML_ASSERT(src->nb[2] % type_size == 0);
    GGML_ASSERT(src->nb[3] % type_size == 0);

    to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(src->type);
    GGML_ASSERT(to_fp16 != nullptr);
    to_fp16(
        (const char *) src->data + kv_start*src->nb[1], dst,
        src->ne[0], kv_count, src->ne[2], src->ne[3],
        src->nb[1]/type_size, src->nb[2]/type_size, src->nb[3]/type_size,
        stream);
}

static __global__ void k_merge_fattn_parts(
        float * __restrict__ acc,
        float2 * __restrict__ acc_meta,
        const float * __restrict__ next,
        const float2 * __restrict__ next_meta,
        int64_t ne0) {
    const int64_t row = blockIdx.x;
    __shared__ float2 meta[2];
    if (threadIdx.x == 0) {
        meta[0] = acc_meta[row];
        meta[1] = next_meta[row];
    }
    __syncthreads();

    const float2 ma = meta[0];
    const float2 mb = meta[1];
    const float m = fmaxf(ma.x, mb.x);
    const float sa = expf(ma.x - m);
    const float sb = expf(mb.x - m);

    for (int64_t i = threadIdx.x; i < ne0; i += blockDim.x) {
        acc[row*ne0 + i] = sa*acc[row*ne0 + i] + sb*next[row*ne0 + i];
    }
    if (threadIdx.x == 0) {
        acc_meta[row] = make_float2(m, sa*ma.y + sb*mb.y);
    }
}

static __global__ void k_normalize_fattn_parts(
        float * __restrict__ dst,
        const float2 * __restrict__ meta,
        int64_t ne0) {
    const int64_t row = blockIdx.x;
    const float inv_sum = 1.0f / meta[row].y;
    for (int64_t i = threadIdx.x; i < ne0; i += blockDim.x) {
        dst[row*ne0 + i] *= inv_sum;
    }
}

} // namespace

bool ggml_cuda_fattn_kv_batched(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        fattn_kernel_t fattn_kernel,
        int nwarps,
        size_t nbytes_shared,
        int warp_size,
        int ncols1,
        int ncols2) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const bool k_supported = K->type == GGML_TYPE_Q8_0 || is_turbo_type(K->type);
    const bool v_supported = V->type == GGML_TYPE_Q8_0 || is_turbo_type(V->type);
    const bool has_turbo = is_turbo_type(K->type) || is_turbo_type(V->type);
    const bool q8_turbo_mixed = (K->type == GGML_TYPE_Q8_0 && is_turbo_type(V->type)) ||
                                (V->type == GGML_TYPE_Q8_0 && is_turbo_type(K->type));

    if (!has_turbo || !k_supported || !v_supported || Q->ne[1] <= 8 ||
        (K->ne[1] <= FATTN_KV_CONV_BATCH && !q8_turbo_mixed)) {
        return false;
    }

    GGML_ASSERT(K->ne[0] == 128 || K->ne[0] == 256);
    GGML_ASSERT(V->ne[0] == K->ne[0] && Q->ne[0] == K->ne[0]);
    GGML_ASSERT(V->ne[1] == K->ne[1]);
    GGML_ASSERT(K->ne[1] % FATTN_KQ_STRIDE == 0);
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));
    GGML_ASSERT(ncols2 == 1 || mask != nullptr);
    GGML_ASSERT(ggml_is_contiguous(dst));

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();

    const int64_t kv_total = K->ne[1];
    const int64_t kv_cap = std::min(FATTN_KV_CONV_BATCH, kv_total);
    const int64_t n_batches = (kv_total + kv_cap - 1) / kv_cap;
    const int64_t n_rows = ggml_nrows(dst);

    ggml_cuda_pool_alloc<half> K_f16(pool);
    ggml_cuda_pool_alloc<half> V_f16(pool);
    ggml_cuda_pool_alloc<float2> acc_meta(pool);
    ggml_cuda_pool_alloc<float> next_parts(pool);
    ggml_cuda_pool_alloc<float2> next_meta(pool);

    K_f16.alloc(kv_cap*K->ne[0]*K->ne[2]*K->ne[3]);
    V_f16.alloc(kv_cap*V->ne[0]*V->ne[2]*V->ne[3]);
    acc_meta.alloc(n_rows);
    if (n_batches > 1) {
        next_parts.alloc(ggml_nelements(dst));
        next_meta.alloc(n_rows);
    }

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const float m0 = powf(2.0f, -max_bias / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias/2.0f) / n_head_log2);
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    const int ntiles_x = (Q->ne[1] + ncols1 - 1) / ncols1;
    const int ntiles_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;
    const dim3 blocks(ntiles_x, 1, ntiles_z_gqa*K->ne[2]*Q->ne[3]);
    const dim3 threads(warp_size, nwarps, 1);

    auto run_batch = [&](int64_t batch, float * out, float2 * meta) {
        const int64_t kv_start = batch*kv_cap;
        const int64_t kv_count = std::min(kv_cap, kv_total - kv_start);

        convert_kv_slice(K, K_f16.ptr, kv_start, kv_count, stream);
        convert_kv_slice(V, V_f16.ptr, kv_start, kv_count, stream);

        const size_t knb1 = K->ne[0]*sizeof(half);
        const size_t knb2 = kv_count*knb1;
        const size_t knb3 = K->ne[2]*knb2;
        const size_t vnb1 = V->ne[0]*sizeof(half);
        const size_t vnb2 = kv_count*vnb1;
        const size_t vnb3 = V->ne[2]*vnb2;
        const char * mask_data = mask ? (const char *) mask->data + kv_start*mask->nb[0] : nullptr;
        const char * sink_data = batch == 0 && sinks ? (const char *) sinks->data : nullptr;

        fattn_kernel<<<blocks, threads, nbytes_shared, stream>>>(
            (const char *) Q->data,
            (const char *) K_f16.ptr,
            (const char *) V_f16.ptr,
            mask_data,
            sink_data,
            nullptr,
            out,
            meta,
            scale, max_bias, m0, m1, n_head_log2, logit_softcap,
            Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K->ne[0], kv_count, K->ne[2], K->ne[3], knb1, knb2, knb3,
            vnb1, vnb2, vnb3,
            mask ? (int32_t) mask->ne[1] : 0,
            mask ? (int32_t) mask->ne[2] : 0,
            mask ? (int32_t) mask->ne[3] : 0,
            mask ? (int32_t) mask->nb[1] : 0,
            mask ? (int32_t) mask->nb[2] : 0,
            mask ? (int64_t) mask->nb[3] : 0);
        CUDA_CHECK(cudaGetLastError());
    };

    run_batch(0, (float *) dst->data, acc_meta.ptr);
    const int combine_threads = std::min<int64_t>(dst->ne[0], 256);
    for (int64_t batch = 1; batch < n_batches; ++batch) {
        run_batch(batch, next_parts.ptr, next_meta.ptr);
        k_merge_fattn_parts<<<n_rows, combine_threads, 0, stream>>>(
            (float *) dst->data, acc_meta.ptr, next_parts.ptr, next_meta.ptr, dst->ne[0]);
        CUDA_CHECK(cudaGetLastError());
    }

    k_normalize_fattn_parts<<<n_rows, combine_threads, 0, stream>>>(
        (float *) dst->data, acc_meta.ptr, dst->ne[0]);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
