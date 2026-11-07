#include "argsort.cuh"
#include "top-k.cuh"

// Adjusted implementation thresholds from #28547, can be overridden at build time
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
// not measured on HIP/MUSA, keep the old split
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 1024
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 512
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC

#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT
#    define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT 4096
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT

// bitonic up to this width while nrows fits in one wave of SMs, 0 disables
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 0
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 1024
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS

#ifdef GGML_CUDA_USE_CUB
#    ifdef GGML_CUDA_CUB_IS_HIPCUB
#        include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#    else
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
#    endif  // GGML_CUDA_CUB_IS_HIPCUB
#endif      // GGML_CUDA_USE_CUB

// max rows for the per-row DeviceTopK / CUB argsort path before switching to radix / bitonic
#ifndef GGML_CUDA_TOP_K_NROWS_THRESHOLD
#    ifdef CUB_TOP_K_AVAILABLE
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 2
#    else
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 1
#    endif
#endif // GGML_CUDA_TOP_K_NROWS_THRESHOLD

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#endif                            // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int64_t col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE, bool STABLE_TIES = false>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int64_t col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if constexpr (!STABLE_TIES) {
            if (key == state->prefix) {
                const int pos = atomicAdd(&state->equal_count, 1);
                if (pos < state->rank) {
                    row_dst[k - state->rank + pos] = col;
                }
            }
        }
    }
}

#if defined(GGML_USE_HIP)
// ties with the k-th score, taken in increasing column order: one block per row scans the row in order,
// so the selected set does not depend on the scheduling of the atomic gather
template<int BLOCK_SIZE>
static __device__ void top_k_gather_equal(
        const float * src, int * dst, int ncols, uint32_t threshold, int limit, int offset) {
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int warp = tid / warpSize;
    const int nwarps = BLOCK_SIZE / warpSize;
    __shared__ int warp_counts[32];
    int count = 0;

    for (int base = 0; base < ncols && count < limit; base += BLOCK_SIZE) {
        const int col = base + tid;
        const bool equal = col < ncols && top_k_float_to_ordered(src[col]) == threshold;
        const unsigned long long mask = __ballot(equal);
        if (lane == 0) {
            warp_counts[warp] = __popcll(mask);
        }
        __syncthreads();
        int before = count;
        for (int w = 0; w < nwarps; ++w) {
            if (w < warp) {
                before += warp_counts[w];
            }
            count += warp_counts[w];
        }
        const unsigned long long lane_mask = (1ULL << lane) - 1;
        const int pos = before + __popcll(mask & lane_mask);
        if (equal && pos < limit) {
            dst[offset + pos] = col;
        }
        __syncthreads();
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather_equal(
        const float * src, int * dst, const top_k_radix_state * states, int ncols, int k) {
    const int row = blockIdx.x;
    const top_k_radix_state state = states[row];
    top_k_gather_equal<BLOCK_SIZE>(src + (size_t) row*ncols, dst + (size_t) row*k,
                                  ncols, state.prefix, state.rank, k - state.rank);
}
#endif // defined(GGML_USE_HIP)

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = (int) std::min<int64_t>(((int64_t) ncols + 1023) / 1024, 64);

    // chunk the rows to bound the histogram memory to 64 MB
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) blocks_per_row * NBINS * sizeof(int), nrows);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, chunk_nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) chunk_nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        top_k_radix_init<<<(iter_nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, iter_nrows, k);

        const dim3 row_grid(blocks_per_row * iter_nrows);
        for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
            top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    src, states, histograms, ncols, blocks_per_row, shift);
            top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
                <<<iter_nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
        }

        top_k_radix_reset_counters
            <<<(iter_nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, iter_nrows);
#if defined(GGML_USE_HIP)
        // HIP: ties at the cut by increasing column index, QSA block scores often tie at exactly zero
        top_k_radix_gather<BLOCK_SIZE, true>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, dst, states, ncols, k, blocks_per_row);
        top_k_radix_gather_equal<BLOCK_SIZE>
            <<<iter_nrows, BLOCK_SIZE, 0, stream>>>(src, dst, states, ncols, k);
#else
        top_k_radix_gather<BLOCK_SIZE>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, dst, states, ncols, k, blocks_per_row);
#endif // defined(GGML_USE_HIP)

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}

static void top_k_argsort_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, bool use_cub, cudaStream_t stream) {
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) ncols * sizeof(int), nrows);

    ggml_cuda_pool_alloc<int> tmp_alloc(pool, (size_t) ncols * chunk_nrows);
    int * tmp = tmp_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        if (use_cub) {
#ifdef GGML_CUDA_USE_CUB
            argsort_f32_i32_cuda_cub(pool, src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
#else
            GGML_ABORT("CUB is not available");
#endif // GGML_CUDA_USE_CUB
        } else {
            argsort_f32_i32_cuda_bitonic(src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst, k * sizeof(int), tmp, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}

#if defined(GGML_USE_HIP)
// One workgroup per row (RDNA3.5 prefill, e.g. the qwen4exp QSA block selection: 512 of ~62k blocks per query).
// The radix path above reads every row ~5.5 times (four 8-bit histogram passes, a gather and a tie gather); here a
// 12-bit LDS histogram locates the threshold bucket, a second pass writes the keys above it and compacts the bucket
// into LDS, and the remaining 20 bits are resolved on the compacted candidates. A bucket larger than the LDS buffer
// falls back to further histogram passes over the row. The selected set is the same as top_k_radix_cuda with
// stable_ties: every key above the k-th largest, then the lowest-column keys equal to it. Like that path, the order
// of the keys above the threshold is unspecified; the equal keys follow them in column order.
#define TOPK_WG_THREADS 1024
#define TOPK_WG_CAP     4096

// bin b of hist (NBINS bins, descending search) with sum(hist[> b]) < rank <= sum(hist[>= b]); returns b and the
// rank left inside it. Every thread gets the result.
template<int NBINS>
static __device__ __forceinline__ void top_k_wg_find(const int * hist, int rank, int * red, int & bin, int & rank_in) {
    constexpr int PER = NBINS / TOPK_WG_THREADS;
    const int tid = threadIdx.x;
    // thread t owns bins [NBINS - (t+1)*PER, NBINS - t*PER): thread 0 the highest
    int own = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i) { own += hist[NBINS - 1 - (tid * PER + i)]; }
    red[tid] = own;
    __syncthreads();
    // inclusive scan over threads (Hillis-Steele)
    for (int off = 1; off < TOPK_WG_THREADS; off <<= 1) {
        const int v = tid >= off ? red[tid - off] : 0;
        __syncthreads();
        red[tid] += v;
        __syncthreads();
    }
    const int incl = red[tid], excl = incl - own;
    __shared__ int s_bin, s_rank;
    if (excl < rank && rank <= incl) {
        int r = rank - excl, b = NBINS - 1 - tid * PER;
        for (int i = 0; i < PER; ++i, --b) {
            const int h = hist[b];
            if (r <= h) break;
            r -= h;
        }
        s_bin = b; s_rank = r;
    }
    __syncthreads();
    bin = s_bin; rank_in = s_rank;
    __syncthreads();
}

static __global__ void __launch_bounds__(TOPK_WG_THREADS) top_k_wg_kernel(
        const float * __restrict__ src, int * __restrict__ dst, const int ncols, const int k) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const float * x = src + (size_t) row * ncols;
    int * out = dst + (size_t) row * k;

    __shared__ int hist[4096];
    __shared__ int red[TOPK_WG_THREADS];
    __shared__ uint32_t ckey[TOPK_WG_CAP];
    __shared__ int      ccol[TOPK_WG_CAP];
    __shared__ int n_out, n_cand;

    for (int i = tid; i < 4096; i += TOPK_WG_THREADS) hist[i] = 0;
    if (tid == 0) { n_out = 0; n_cand = 0; }
    __syncthreads();

    // pass 1: top 12 bits (four independent loads in flight per thread)
    {
        int c = tid;
        for (; c + 3 * TOPK_WG_THREADS < ncols; c += 4 * TOPK_WG_THREADS) {
            const float v0 = x[c], v1 = x[c + TOPK_WG_THREADS], v2 = x[c + 2 * TOPK_WG_THREADS], v3 = x[c + 3 * TOPK_WG_THREADS];
            atomicAdd(&hist[top_k_float_to_ordered(v0) >> 20], 1);
            atomicAdd(&hist[top_k_float_to_ordered(v1) >> 20], 1);
            atomicAdd(&hist[top_k_float_to_ordered(v2) >> 20], 1);
            atomicAdd(&hist[top_k_float_to_ordered(v3) >> 20], 1);
        }
        for (; c < ncols; c += TOPK_WG_THREADS) {
            atomicAdd(&hist[top_k_float_to_ordered(x[c]) >> 20], 1);
        }
    }
    __syncthreads();
    int b12, rank;
    top_k_wg_find<4096>(hist, k, red, b12, rank);
    const int bucket = hist[b12];
    uint32_t prefix = (uint32_t) b12 << 20, pmask = 0xFFF00000u;
    __syncthreads();

    // pass 2: keys above the bucket go out, the bucket is compacted when it fits
    const bool fits = bucket <= TOPK_WG_CAP;
    auto place = [&](const int c, const float v) {
        const uint32_t key = top_k_float_to_ordered(v);
        const uint32_t top = key >> 20;
        if (top > (uint32_t) b12) {
            out[atomicAdd(&n_out, 1)] = c;
        } else if (fits && top == (uint32_t) b12) {
            const int p = atomicAdd(&n_cand, 1);
            ckey[p] = key; ccol[p] = c;
        }
    };
    {
        int c = tid;
        for (; c + 3 * TOPK_WG_THREADS < ncols; c += 4 * TOPK_WG_THREADS) {
            const float v0 = x[c], v1 = x[c + TOPK_WG_THREADS], v2 = x[c + 2 * TOPK_WG_THREADS], v3 = x[c + 3 * TOPK_WG_THREADS];
            place(c, v0); place(c + TOPK_WG_THREADS, v1); place(c + 2 * TOPK_WG_THREADS, v2); place(c + 3 * TOPK_WG_THREADS, v3);
        }
        for (; c < ncols; c += TOPK_WG_THREADS) place(c, x[c]);
    }
    __syncthreads();

    // remaining 20 bits: 10 + 10, on the candidates or on the row
    const int ncand = fits ? n_cand : 0;
    for (int shift = 10; shift >= 0; shift -= 10) {
        for (int i = tid; i < 1024; i += TOPK_WG_THREADS) hist[i] = 0;
        __syncthreads();
        if (fits) {
            for (int i = tid; i < ncand; i += TOPK_WG_THREADS) {
                const uint32_t key = ckey[i];
                if ((key & pmask) == prefix) atomicAdd(&hist[(key >> shift) & 1023], 1);
            }
        } else {
            for (int c = tid; c < ncols; c += TOPK_WG_THREADS) {
                const uint32_t key = top_k_float_to_ordered(x[c]);
                if ((key & pmask) == prefix) atomicAdd(&hist[(key >> shift) & 1023], 1);
            }
        }
        __syncthreads();
        int b10, r2;
        top_k_wg_find<1024>(hist, rank, red, b10, r2);
        rank = r2;
        prefix |= (uint32_t) b10 << shift;
        pmask  |= 1023u << shift;
        __syncthreads();
    }
    const uint32_t thr = prefix;   // the k-th largest key; `rank` keys equal to it are taken

    // keys above the threshold inside the bucket
    if (fits) {
        for (int i = tid; i < ncand; i += TOPK_WG_THREADS) {
            if (ckey[i] > thr) out[atomicAdd(&n_out, 1)] = ccol[i];
        }
    } else {
        for (int c = tid; c < ncols; c += TOPK_WG_THREADS) {
            const uint32_t key = top_k_float_to_ordered(x[c]);
            if ((key >> 20) == (uint32_t) b12 && key > thr) out[atomicAdd(&n_out, 1)] = c;
        }
    }
    __syncthreads();
    const int base = n_out;   // == k - rank

    // equal keys, lowest columns first
    if (fits) {
        // gather the equal columns, then pick the `rank` smallest by counting (each is unique)
        __shared__ int n_eq;
        if (tid == 0) n_eq = 0;
        __syncthreads();
        // the histogram is free now and holds up to TOPK_WG_CAP (>= ncand) columns
        for (int i = tid; i < ncand; i += TOPK_WG_THREADS) {
            if (ckey[i] == thr) { const int p = atomicAdd(&n_eq, 1); hist[p] = ccol[i]; }
        }
        __syncthreads();
        const int ne = n_eq;
        for (int i = tid; i < ne; i += TOPK_WG_THREADS) {
            const int col = hist[i];
            int less = 0;
            for (int j = 0; j < ne; ++j) less += hist[j] < col;
            if (less < rank) out[base + less] = col;
        }
    } else {
        const int lane = tid % warpSize, warp = tid / warpSize;
        constexpr int NW = TOPK_WG_THREADS / 32;
        int taken = 0;
        for (int c0 = 0; c0 < ncols && taken < rank; c0 += TOPK_WG_THREADS) {
            const int c = c0 + tid;
            const bool eq = c < ncols && top_k_float_to_ordered(x[c]) == thr;
            const unsigned long long m = __ballot(eq);
            if (lane == 0) red[warp] = __popcll(m);
            __syncthreads();
            int before = taken;
            for (int w = 0; w < NW; ++w) { if (w < warp) before += red[w]; }
            int total = 0;
            for (int w = 0; w < NW; ++w) total += red[w];
            const int pos = before + __popcll(m & ((1ULL << lane) - 1));
            if (eq && pos < rank) out[base + pos] = c;
            taken += total;
            __syncthreads();
        }
    }
}
#endif // defined(GGML_USE_HIP)

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

    const int device = ggml_cuda_get_device();

#if defined(GGML_USE_HIP)
    // RDNA3.5, wide batched rows (e.g. the qwen4exp QSA block selection): one workgroup per row selects the same set
    // as the stable-tie radix path in fewer passes.
    {
        const bool wide = ncols >= 8192 || (ncols >= 4096 && nrows >= 128) || (ncols >= 2048 && nrows >= 512);
        if (wide && GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[device].cc) &&
                nrows > 1 && k <= ncols && ncols <= INT_MAX && nrows <= INT_MAX) {
            top_k_wg_kernel<<<(unsigned) nrows, TOPK_WG_THREADS, 0, stream>>>(src0_d, dst_d, (int) ncols, (int) k);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
#endif // defined(GGML_USE_HIP)

#ifdef CUB_TOP_K_AVAILABLE
    // a single row always uses DeviceTopK if available
    const bool bitonic_short    = nrows > 1 && ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
#else
    const bool bitonic_short    = ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
#endif // CUB_TOP_K_AVAILABLE
    const bool bitonic_few_rows = nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD &&
                                  ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS &&
                                  nrows <= ggml_cuda_info().devices[device].nsm;

    if (bitonic_short || bitonic_few_rows) {
        // the padded row must fit in shared memory
        const int ncols_pad = next_power_of_2(ncols);
        if (ncols_pad * sizeof(int) <= ggml_cuda_info().devices[device].smpb) {
            top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, false, stream);
            return;
        }
    }

    if (nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }

#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Assess perf of `DeviceBatchedTopK` for multi-row TopK & CCCL >= 3.5.0, re-running perf sweep of https://github.com/ggml-org/llama.cpp/pull/28713
    for (int64_t i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    if (ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT) {
        top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, true, stream);
    } else {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    }
#else                             // GGML_CUDA_USE_CUB
    top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
#endif                            // CUB_TOP_K_AVAILABLE
}
