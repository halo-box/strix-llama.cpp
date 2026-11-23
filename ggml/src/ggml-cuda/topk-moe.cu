#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "topk-moe.cuh"

#include <cmath>
#include <initializer_list>

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
    bool use_sigmoid;
    bool use_sqrt_softplus;
    bool with_norm;
    bool delayed_softmax;
    int  n_groups;
    int  n_group_used;
};

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * warp_size;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max<warp_size>(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * warp_size;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum<warp_size>(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * warp_size;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * warp_size;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sqrt_softplus_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * warp_size;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? sqrtf(vals[i] > 20.0f ? vals[i] : log1pf(expf(vals[i]))) : -INFINITY;
    }
}

// Set -inf for experts outside the top n_group_used groups. Group score is the sum of its top-2 experts (DeepSeek V3).
template <int n_experts, int experts_per_thread>
__device__ void group_mask_warp_inplace(float (&vals)[experts_per_thread], const int n_groups, const int n_group_used, const int lane) {
    const int n_per_group = n_experts / n_groups;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    float score = -INFINITY; // lane g holds the score of group g
    for (int g = 0; g < n_groups; g++) {
        float top1 = -INFINITY;
        float top2 = -INFINITY;
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            if ((i * warp_size + lane) / n_per_group == g) {
                top2 = fmaxf(top2, fminf(top1, vals[i]));
                top1 = fmaxf(top1, vals[i]);
            }
        }
#pragma unroll
        for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
            const float o1 = ggml_cuda_shfl_xor(top1, offset, warp_size);
            const float o2 = ggml_cuda_shfl_xor(top2, offset, warp_size);
            top2 = fmaxf(fminf(top1, o1), fmaxf(top2, o2));
            top1 = fmaxf(top1, o1);
        }
        if (lane == g) {
            score = top1 + top2;
        }
    }

    // rank of each group, lower index wins ties
    int rank = 0;
    for (int g = 0; g < n_groups; g++) {
        const float other = ggml_cuda_shfl(score, g, warp_size);
        rank += other > score || (other == score && g < lane);
    }
    const auto keep = ggml_cuda_ballot(lane < n_groups && rank < n_group_used);

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int g = (i * warp_size + lane) / n_per_group;
        if (g >= n_groups || !((keep >> g) & 1)) {
            vals[i] = -INFINITY;
        }
    }
}

static __device__ constexpr int calc_experts_per_thread(int n_experts, int warp_size)
{
    if(n_experts <= warp_size) {
        return 1;
    }
    if(n_experts % warp_size == 0) {
        return n_experts / warp_size;
    }
    return n_experts / warp_size + 1;
}

/*
    This kernel does the following:
    1. optionally softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
// The selection part of topk_moe_cuda for one row whose logits are in wt[] (lane threadIdx.x holds experts
// threadIdx.x + i*warp_size), for topk_moe_router_cuda. topk_moe_cuda keeps its own copy: calling this function
// changes its code generation.
template <int n_experts, bool has_bias, bool small_topk>
static __device__ __forceinline__ void topk_moe_select(
        float (&wt)[calc_experts_per_thread(n_experts, ggml_cuda_get_physical_warp_size())],
        float * weights, int32_t * ids, const float * bias, const int n_expert_used,
        const float clamp_val, const float scale_val, const topk_moe_config & config) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int experts_per_thread = calc_experts_per_thread(n_experts, warp_size);

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
           sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else if (config.use_sqrt_softplus) {
           sqrt_softplus_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else {
           softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
    // NaN comparisons always return false, which would cause the same expert to be
    // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
    // -INFINITY sentinel used after each selection round.
    // More relevant for the cuBLAS path. See https://github.com/ggml-org/llama.cpp/issues/19659
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    [[maybe_unused]] float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += warp_size) {
            const int expert = i + threadIdx.x;
            selection_wt[i / warp_size] =
                (n_experts % warp_size == 0 || expert < n_experts) ? wt[i / warp_size] + bias[expert] : -INFINITY;
            if (__isnanf(selection_wt[i / warp_size])) {
                selection_wt[i / warp_size] = -FLT_MAX;
            }
        }
    }

    // without bias, masking wt is safe: the matcher ensures the kept groups hold all used experts
    if (config.n_groups > 0) {
        if constexpr (has_bias) {
            group_mask_warp_inplace<n_experts>(selection_wt, config.n_groups, config.n_group_used, threadIdx.x);
        } else {
            group_mask_warp_inplace<n_experts>(wt, config.n_groups, config.n_group_used, threadIdx.x);
        }
    }

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    constexpr int output_weights_per_thread = small_topk ? 1 : experts_per_thread;
    float output_weights[output_weights_per_thread];

#pragma unroll
    for (int i = 0; i < output_weights_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ggml_cuda_pdl_lc();
    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * warp_size;
                if ((n_experts % warp_size == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = warp_size / 2; mask > 0; mask /= 2) {
                const float val    = ggml_cuda_shfl_xor(max_val, mask, warp_size);
                const float val_s  = ggml_cuda_shfl_xor(max_val_s, mask, warp_size);
                const int   expert = ggml_cuda_shfl_xor(max_expert, mask, warp_size);
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val    = val;
                    max_val_s  = val_s;
                    max_expert = expert;
                }
            }

            if ((max_expert & (warp_size - 1)) == threadIdx.x) {
                selection_wt[max_expert / warp_size] = -INFINITY;
            }
        } else {
#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * warp_size;
                if ((n_experts % warp_size == 0 || expert < n_experts) && wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = warp_size / 2; mask > 0; mask /= 2) {
                const float val    = ggml_cuda_shfl_xor(max_val, mask, warp_size);
                const int   expert = ggml_cuda_shfl_xor(max_expert, mask, warp_size);
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            }

            if ((max_expert & (warp_size - 1)) == threadIdx.x) {
                wt[max_expert / warp_size] = -INFINITY;
            }
        }

        if ((k & (warp_size - 1)) == threadIdx.x) {
            output_weights[small_topk ? 0 : k / warp_size] = max_val;
        }

        if ((max_expert & (warp_size - 1)) == threadIdx.x) {
            ids[k] = max_expert;
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        wt_sum              = warp_reduce_sum<warp_size>(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < output_weights_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<output_weights_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < output_weights_per_thread; i++) {
        const int idx = i * warp_size + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template <int n_experts, bool has_bias, bool small_topk = false>
__launch_bounds__(TOPK_MOE_ROWS_PER_BLOCK * ggml_cuda_get_physical_warp_size(), 1)
__global__ void topk_moe_cuda(const float *         logits,
                              float *               weights,
                              int32_t *             ids,
                              float *               bias,
                              const int             n_rows,
                              const int             n_expert_used,
                              const float           clamp_val,
                              const float           scale_val,
                              const topk_moe_config config) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(GGML_USE_MUSA)
    // MUSA: every warp of a partially filled block must reach the barrier below.
    const int row = MIN(blockIdx.x * blockDim.y + threadIdx.y, n_rows - 1);
#else
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
#endif // defined(GGML_USE_MUSA)
    if (row >= n_rows) {
        return;
    }

    logits += n_experts * row;
    weights += n_expert_used * row;
    ids += n_experts * row;

    constexpr int experts_per_thread = calc_experts_per_thread(n_experts, warp_size);

    float wt[experts_per_thread];

    // Initialize all slots to -INFINITY
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < n_experts; i += warp_size) {
        const int expert  = i + threadIdx.x;
        wt[i / warp_size] = (n_experts % warp_size == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    // Weights and IDs can alias logits, so wait until every row in the block reads its logits.
    __syncthreads();

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
           sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else if (config.use_sqrt_softplus) {
           sqrt_softplus_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else {
           softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
    // NaN comparisons always return false, which would cause the same expert to be
    // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
    // -INFINITY sentinel used after each selection round.
    // More relevant for the cuBLAS path. See https://github.com/ggml-org/llama.cpp/issues/19659
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    [[maybe_unused]] float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += warp_size) {
            const int expert = i + threadIdx.x;
            selection_wt[i / warp_size] =
                (n_experts % warp_size == 0 || expert < n_experts) ? wt[i / warp_size] + bias[expert] : -INFINITY;
            if (__isnanf(selection_wt[i / warp_size])) {
                selection_wt[i / warp_size] = -FLT_MAX;
            }
        }
    }

    // without bias, masking wt is safe: the matcher ensures the kept groups hold all used experts
    if (config.n_groups > 0) {
        if constexpr (has_bias) {
            group_mask_warp_inplace<n_experts>(selection_wt, config.n_groups, config.n_group_used, threadIdx.x);
        } else {
            group_mask_warp_inplace<n_experts>(wt, config.n_groups, config.n_group_used, threadIdx.x);
        }
    }

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    constexpr int output_weights_per_thread = small_topk ? 1 : experts_per_thread;
    float output_weights[output_weights_per_thread];

#pragma unroll
    for (int i = 0; i < output_weights_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ggml_cuda_pdl_lc();
    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * warp_size;
                if ((n_experts % warp_size == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = warp_size / 2; mask > 0; mask /= 2) {
                const float val    = ggml_cuda_shfl_xor(max_val, mask, warp_size);
                const float val_s  = ggml_cuda_shfl_xor(max_val_s, mask, warp_size);
                const int   expert = ggml_cuda_shfl_xor(max_expert, mask, warp_size);
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val    = val;
                    max_val_s  = val_s;
                    max_expert = expert;
                }
            }

            if ((max_expert & (warp_size - 1)) == threadIdx.x) {
                selection_wt[max_expert / warp_size] = -INFINITY;
            }
        } else {
#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * warp_size;
                if ((n_experts % warp_size == 0 || expert < n_experts) && wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = warp_size / 2; mask > 0; mask /= 2) {
                const float val    = ggml_cuda_shfl_xor(max_val, mask, warp_size);
                const int   expert = ggml_cuda_shfl_xor(max_expert, mask, warp_size);
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            }

            if ((max_expert & (warp_size - 1)) == threadIdx.x) {
                wt[max_expert / warp_size] = -INFINITY;
            }
        }

        if ((k & (warp_size - 1)) == threadIdx.x) {
            output_weights[small_topk ? 0 : k / warp_size] = max_val;
        }

        if ((max_expert & (warp_size - 1)) == threadIdx.x) {
            ids[k] = max_expert;
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        wt_sum              = warp_reduce_sum<warp_size>(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < output_weights_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<output_weights_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < output_weights_per_thread; i++) {
        const int idx = i * warp_size + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

#if defined(GGML_USE_HIP)
// RDNA3.5 decode: the F32 router matvec of one token and topk_moe in one launch. Block b computes logit b with the
// arithmetic of mul_mat_vec_f<float, float, 1, 256> (same loads, fma order and block reduction, so the logits are
// bit-identical); the last block to store its logit then runs the selection of topk_moe_cuda on all logits.
template <int n_experts, bool has_bias>
__launch_bounds__(256, 1)
__global__ void topk_moe_router_cuda(const float *         x,
                                     const float *         y,
                                     const int             stride_row,
                                     const int             ncols2,
                                     float *               logits,
                                     float *               weights,
                                     int32_t *             ids,
                                     const float *         bias,
                                     unsigned int *        counter,
                                     const int             n_expert_used,
                                     const float           clamp_val,
                                     const float           scale_val,
                                     const topk_moe_config config) {
    constexpr int block_size = 256;
    constexpr int warp_size  = ggml_cuda_get_physical_warp_size();
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float buf_iw[warp_size];
    __shared__ bool  is_last;
    if (tid < warp_size) {
        buf_iw[tid] = 0.0f;
    }
    __syncthreads();

    const float2 * x2 = (const float2 *) (x + (int64_t) row*stride_row);
    const float2 * y2 = (const float2 *) y;
    float sumf = 0.0f;
    int col2 = tid;
    constexpr int pf = 4;
    for (; col2 + (pf - 1)*block_size < ncols2; col2 += pf*block_size) {
        float2 tmpx[pf];
        float2 tmpy[pf];
#pragma unroll
        for (int i = 0; i < pf; ++i) {
            tmpx[i] = x2[col2 + i*block_size];
        }
#pragma unroll
        for (int i = 0; i < pf; ++i) {
            tmpy[i] = y2[col2 + i*block_size];
        }
#pragma unroll
        for (int i = 0; i < pf; ++i) {
            ggml_cuda_mad(sumf, tmpx[i].x, tmpy[i].x);
            ggml_cuda_mad(sumf, tmpx[i].y, tmpy[i].y);
        }
    }
    for (; col2 < ncols2; col2 += block_size) {
        const float2 tmpx = x2[col2];
        const float2 tmpy = y2[col2];
        ggml_cuda_mad(sumf, tmpx.x, tmpy.x);
        ggml_cuda_mad(sumf, tmpx.y, tmpy.y);
    }

    sumf = warp_reduce_sum<warp_size>(sumf);
    buf_iw[tid/warp_size] = sumf;
    __syncthreads();
    if (tid < warp_size) {
        sumf = buf_iw[tid];
        sumf = warp_reduce_sum<warp_size>(sumf);
    }
    if (tid == 0) {
        logits[row] = sumf;
        __threadfence();
        is_last = atomicAdd(counter, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!is_last || tid >= warp_size) {
        return;
    }
    __threadfence();
    if (tid == 0) {
        *counter = 0;
    }

    constexpr int experts_per_thread = calc_experts_per_thread(n_experts, warp_size);
    float wt[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }
#pragma unroll
    for (int i = 0; i < n_experts; i += warp_size) {
        const int expert = i + tid;
        if (n_experts % warp_size == 0 || expert < n_experts) {
            wt[i / warp_size] = __hip_atomic_load(logits + expert, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
    topk_moe_select<n_experts, has_bias, false>(wt, weights, ids, bias, n_expert_used, clamp_val, scale_val, config);
}
#endif // defined(GGML_USE_HIP)

template<bool has_bias>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 float *                     bias,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used,
                                 const float                 clamp_val,
                                 const float                 scale_val,
                                 const topk_moe_config       config) {
    GGML_ASSERT(!(config.with_norm && config.delayed_softmax) &&
                "delayed softmax is not supported with weight normalization");
    const int    rows_per_block = TOPK_MOE_ROWS_PER_BLOCK;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(warp_size, rows_per_block, 1);
    cudaStream_t stream = ctx.stream();
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    switch (n_expert) {
        case 1:
            ggml_cuda_kernel_launch(topk_moe_cuda<1, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 2:
            ggml_cuda_kernel_launch(topk_moe_cuda<2, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 4:
            ggml_cuda_kernel_launch(topk_moe_cuda<4, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 8:
            ggml_cuda_kernel_launch(topk_moe_cuda<8, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 16:
            ggml_cuda_kernel_launch(topk_moe_cuda<16, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 32:
            ggml_cuda_kernel_launch(topk_moe_cuda<32, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 64:
            ggml_cuda_kernel_launch(topk_moe_cuda<64, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 128:
            ggml_cuda_kernel_launch(topk_moe_cuda<128, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 256:
            ggml_cuda_kernel_launch(topk_moe_cuda<256, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 288: // StepFun 3.7
            ggml_cuda_kernel_launch(topk_moe_cuda<288, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 384: // DeepSeek V4.1
            ggml_cuda_kernel_launch(topk_moe_cuda<384, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 512:
            if (n_expert_used <= warp_size) {
                ggml_cuda_kernel_launch(topk_moe_cuda<512, has_bias, true>, launch_params,
                    logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            } else {
                ggml_cuda_kernel_launch(topk_moe_cuda<512, has_bias>, launch_params,
                    logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            }
            break;
        case 576:
            ggml_cuda_kernel_launch(topk_moe_cuda<576, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        default:
            GGML_ASSERT(false && "fatal error");
    }
}

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    float *       bias_d    = bias ? (float *) bias->data : nullptr;

    float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const int n_expert_used = weights->ne[1];

    const bool with_norm = clamp != nullptr;

    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, 0);
    }

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = with_norm;
    config.delayed_softmax   = args.delayed_softmax;
    config.n_groups          = args.n_groups;
    config.n_group_used      = args.n_group_used;

    if (bias) {
        launch_topk_moe_cuda<true>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    } else {
        launch_topk_moe_cuda<false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    }
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids) {
    // must match an instantiation of launch_topk_moe_cuda: a power of 2 up to 512,
    // or one of the non-power-of-2 expert counts of supported models
    const int n_expert = ids->nb[1] / ids->nb[0];
    if (((n_expert & (n_expert - 1)) != 0 || n_expert > 512) && n_expert != 288 && n_expert != 384 && n_expert != 576) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        const ggml_tensor * softmax  = gating_op;
        float               scale    = 1.0f;
        float               max_bias = 0.0f;

        memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(softmax->src[0])) {
            return false;
        }

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        ggml_unary_op op = ggml_get_unary_op(gating_op);

        if (op != GGML_UNARY_OP_SIGMOID && op != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }
    }

    return true;
}

#if defined(GGML_USE_HIP)
// arrival counter of topk_moe_router_cuda; the last block resets it. Allocated outside of a graph capture (the
// first decode graph of a context runs uncaptured).
static unsigned int * ggml_cuda_topk_moe_counter(ggml_backend_cuda_context & ctx) {
    if (ctx.topk_counter == nullptr) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMalloc((void **) &ctx.topk_counter, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemsetAsync(ctx.topk_counter, 0, sizeof(unsigned int), ctx.stream()));
    }
    return ctx.topk_counter;
}

bool ggml_cuda_topk_moe_router_ready(ggml_backend_cuda_context & ctx) {
    if (ctx.topk_counter != nullptr) {
        return true;
    }
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &status));
    return status == cudaStreamCaptureStatusNone;
}

// the block size launch_mul_mat_vec_f_cuda picks on RDNA (32-wide warps, at most 256 threads) for a row of ncols
static int64_t topk_moe_router_mmvf_block_size(const int64_t ncols) {
    int64_t block_size_best = 32;
    int64_t niter_best      = (ncols + 2*32 - 1) / (2*32);
    for (int64_t block_size = 64; block_size <= 256; block_size += 32) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }
    return block_size_best;
}

bool ggml_cuda_topk_moe_router_ok(const ggml_tensor * router, const ggml_tensor * weights, const ggml_tensor * ids,
                                  const ggml_cuda_topk_moe_args & args) {
    const ggml_tensor * x = router->src[0];
    const ggml_tensor * y = router->src[1];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int n_experts = (int) router->ne[0];
    return GGML_CUDA_CC_IS_RDNA3_5(cc) && router->op == GGML_OP_MUL_MAT && router->type == GGML_TYPE_F32 &&
        x->type == GGML_TYPE_F32 && y->type == GGML_TYPE_F32 && x->ne[1] == n_experts && ggml_is_matrix(x) &&
        x->nb[0] == sizeof(float) && x->nb[1] % 8 == 0 && (uintptr_t) x->data % 8 == 0 &&
        ggml_is_contiguous(y) && ggml_nelements(y) == x->ne[0] && x->ne[0] % 2 == 0 && (uintptr_t) y->data % 8 == 0 &&
        ggml_is_contiguous(router) && ggml_nrows(router) == 1 && topk_moe_router_mmvf_block_size(x->ne[0]) == 256 &&
        (n_experts == 64 || n_experts == 128 || n_experts == 256) && weights->ne[1] <= 32 &&
        !args.delayed_softmax && args.n_groups == 0 && ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts;
}

void ggml_cuda_op_topk_moe_router(ggml_backend_cuda_context & ctx, ggml_tensor * router, ggml_tensor * weights,
                                  ggml_tensor * ids, const ggml_tensor * clamp, const ggml_tensor * scale,
                                  const ggml_tensor * bias, const ggml_cuda_topk_moe_args & args) {
    const ggml_tensor * x = router->src[0];
    const ggml_tensor * y = router->src[1];
    const int n_experts     = (int) router->ne[0];
    const int n_expert_used = (int) weights->ne[1];

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = clamp != nullptr;
    config.delayed_softmax   = false;
    config.n_groups          = 0;
    config.n_group_used      = 0;
    const float clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    const float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    unsigned int * counter = ggml_cuda_topk_moe_counter(ctx);
    const ggml_cuda_kernel_launch_params params(dim3(n_experts, 1, 1), dim3(256, 1, 1), 0, ctx.stream());
    auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, params,
            (const float *) x->data, (const float *) y->data, (int) (x->nb[1] / sizeof(float)), (int) (x->ne[0] / 2),
            (float *) router->data, (float *) weights->data, (int32_t *) ids->data,
            bias ? (const float *) bias->data : nullptr, counter, n_expert_used, clamp_val, scale_val, config);
    };
    switch (n_experts) {
        case 64:  bias ? launch(topk_moe_router_cuda<64,  true>) : launch(topk_moe_router_cuda<64,  false>); break;
        case 128: bias ? launch(topk_moe_router_cuda<128, true>) : launch(topk_moe_router_cuda<128, false>); break;
        case 256: bias ? launch(topk_moe_router_cuda<256, true>) : launch(topk_moe_router_cuda<256, false>); break;
        default:  GGML_ABORT("fatal error");
    }
}
#endif // defined(GGML_USE_HIP)
