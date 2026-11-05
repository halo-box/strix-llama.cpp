#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"
#if defined(GGML_USE_HIP)
#include "qsa-prefill.cuh"
#include "qsa-decode.cuh"
#endif

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// one list per group of ncols1 queries: a column is selected if any query of the group can see it
template <int ncols1, bool oob>
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, int32_t * counts_ptr, const int ne30, const int n_queries,
        const int n_kv_max, const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int values_per_lane = 8;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int lane     = tid % WARP_SIZE;
    const int sequence = blockIdx.y;
    const int group    = blockIdx.x;

    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);

    const half * mask = mask_ptr + sequence*s33 + q0*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + group)*n_kv_max;

    __shared__ int warp_offsets[256/WARP_SIZE];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        uint32_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            bool selected = false;
            if (i < ne30) {
#pragma unroll
                for (int q = 0; q < ncols1; ++q) {
                    selected |= (!oob || q < q1 - q0) && isfinite(__half2float(mask[q*s31 + i]));
                }
            }
            selected_warp[item] = __ballot_sync(0xFFFFFFFF, selected);
            warp_count += __popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/WARP_SIZE; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const uint32_t lane_mask = lane == 0 ? 0 : (1u << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + __popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & (uint32_t(1) << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += __popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = min(row_count, n_kv_max);
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    if (tid == 0) {
        counts_ptr[int64_t(sequence)*gridDim.x + group] = count;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t * counts, int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(mask, indices, counts, n_queries, ncols1, n_kv_max, stream);
    GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");
#else
    const int64_t s31 = mask->nb[1] / sizeof(half);
    const int64_t s33 = mask->nb[3] / sizeof(half);
    const dim3 blocks_num((n_queries + ncols1 - 1)/ncols1, mask->ne[3], 1);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    // the last group of queries is partial only if ncols1 does not divide n_queries
    GGML_ASSERT(ncols1 == 1 || ncols1 == 8);
    const auto kernel = ncols1 == 1       ? flash_attn_mask_to_sparse_indices<1, false> :
                        n_queries % 8 != 0 ? flash_attn_mask_to_sparse_indices<8, true>  :
                                             flash_attn_mask_to_sparse_indices<8, false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
        (const half *) mask->data, indices, counts, int(mask->ne[0]), n_queries, n_kv_max, s31, s33);
    CUDA_CHECK(cudaGetLastError());
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int cc, const ggml_tensor * dst, const int ncols1, const int ncols2) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(cc, dst, ncols1, ncols2);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    // the dense kernel handles up to 64/ncols2 queries per K/V pass, the single-query gather has to beat that
    const int64_t n_gather = (ncols1 == 1 ? std::min<int64_t>(Q->ne[1], 64/ncols2) : ncols1) * (int64_t) n_kv_max;

    return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, 2*n_gather);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        // a sparse variant at the full tile width gathers the union of its queries once, prefer it for large batches
        constexpr bool has_wide_sparse = ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 64/ncols2, ncols2);
        if (!(has_wide_sparse && Q->ne[1] > 32/ncols2) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, ncols2)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    // On RDNA it is preferable to minimize wasted compute vs. duplicate I/O for the mask.
    if (amd_wmma_available(cc)) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20 && GGML_CUDA_CC_IS_NVIDIA(cc)) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}

#define FATTN_VEC_CASE(D, type_K_case, type_V_case)                                                                                \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                                                                    \
        const bool type_K_okay = type_K == GGML_TYPE_##type_K_case || (type_K == GGML_TYPE_F32 && GGML_TYPE_##type_K_case == GGML_TYPE_F16); \
        const bool type_V_okay = type_V == GGML_TYPE_##type_V_case || (type_V == GGML_TYPE_F32 && GGML_TYPE_##type_V_case == GGML_TYPE_F16); \
        if (head_size == (D) && type_K_okay && type_V_okay) {                                                                      \
            return ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>;                         \
        }                                                                                                                          \
    }                                                                                                                              \

#define FATTN_VEC_CASES_ALL_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE( 64, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(128, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(256, type_K_case, type_V_case)       \

typedef void (* fattn_vec_case_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Vector kernel for the given head size and K/V types, nullptr if its template instance was not compiled:
static fattn_vec_case_t ggml_cuda_get_fattn_vec_case(const int64_t head_size, const ggml_type type_K, const ggml_type type_V) {
    FATTN_VEC_CASES_ALL_D(F16,  F16)
    FATTN_VEC_CASES_ALL_D(Q4_0, F16)
    FATTN_VEC_CASES_ALL_D(Q4_1, F16)
    FATTN_VEC_CASES_ALL_D(Q5_0, F16)
    FATTN_VEC_CASES_ALL_D(Q5_1, F16)
    FATTN_VEC_CASES_ALL_D(Q8_0, F16)
    FATTN_VEC_CASES_ALL_D(BF16, F16)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q8_0)

    FATTN_VEC_CASES_ALL_D(F16,  BF16)
    FATTN_VEC_CASES_ALL_D(Q4_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q4_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q8_0, BF16)
    FATTN_VEC_CASES_ALL_D(BF16, BF16)

    return nullptr;
}

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_vec_case_t vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type);
    if (vec_case == nullptr) {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_WARN("%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
                "Add \"%s-%s\" to GGML_CUDA_FA_QUANTS to compile it.\n",
                __func__, ggml_type_name(K->type), ggml_type_name(V->type), ggml_type_name(K->type), ggml_type_name(V->type));
            warned = true;
        }
        vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(vec_case != nullptr);
    vec_case(ctx, dst);
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

// K/V types for which there is a vector kernel template instance, other kernels convert these to f16:
static bool ggml_cuda_fattn_kv_type_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

static bool ggml_cuda_fattn_tile_q8_0_KV_supported(const ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    return GGML_CUDA_CC_IS_RDNA3_5(cc) && Q->ne[1] == 1 && K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0 && Q->ne[0] == K->ne[0] && K->ne[0] == V->ne[0] &&
        (Q->ne[0] == 64 || Q->ne[0] == 128 || Q->ne[0] == 256);
#else
    GGML_UNUSED(dst);
    return false;
#endif // GGML_USE_HIP
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                // the sparse gather exists only in the MMA kernel: (DKQ, DV, 1, 8) with GQA > 4
                const bool sparse_decode = gqa_opt_applies && gqa_ratio > 4 &&
                    ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(K->ne[0], V->ne[0], 1, 8) &&
                    ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 8);
                if (!sparse_decode && cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 &&
                        !(gqa_ratio > 4 && (Q->ne[0] >= 256 || K->ne[1] >= 8192))) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (max_bias == 0.0f && gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (ggml_cuda_fattn_tile_q8_0_KV_supported(dst) && gqa_opt_applies && gqa_ratio_eff >= 2) {
        return BEST_FATTN_KERNEL_TILE;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if (Q->ne[0] > 256 && gqa_opt_applies && Q->ne[1] * gqa_ratio_eff > 128) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is faster than the tile kernel if the wide tiles with high arithmetic intensity can be utilized.
    // Only head sizes that have WMMA device code in flash_attn_ext_f16 (fattn-mma-f16.cuh, AMD_WMMA_AVAILABLE guard:
    // DKQ <= 128 or DKQ == 256) may be routed here; any other head size (e.g. 192) hits NO_DEVICE_CODE -> __trap().
    if ((amd_wmma_available(cc) && gqa_opt_applies && (Q->ne[0] <= 128 || Q->ne[0] == 256)) && Q->ne[0] != 40 && Q->ne[0] != 72 &&
            Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // On RDNA3.5 the D=256 tile kernel is FMA-bound for prefill batches; the WMMA kernel with 64 columns is ~30-45% faster.
    // For smaller batches (ncols 16/32 configs) the tile kernel is still faster.
    if (GGML_CUDA_CC_IS_RDNA3_5(cc) && gqa_opt_applies && Q->ne[0] == 256 && V->ne[0] == 256 && Q->ne[1] * gqa_ratio_eff > 32) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
            if (ggml_cuda_fattn_tile_q8_0_KV_supported(dst)) {
                break;
            }
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC: {
            const bool f16_fallback = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type) == nullptr;
            need_f16_K = K->type == GGML_TYPE_F32 || f16_fallback;
            need_f16_V = V->type == GGML_TYPE_F32 || f16_fallback;
        } break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

// ---------------------------------------------------------------------------------------------
// fa7: GQA-aware WMMA flash attention for gfx1151, causal prefill only (NH=1 NK=32 PD=1, verified against a naive GPU
// reference and a CPU fp64 reference on all heads).
// Shape-restricted: HQ=24, HKV=2, D=256, Q f32, K/V f16, n_kv == n_q, no ALiBi / softcap / sinks / selected keys. The
// mask is not read: the graph must mark it as the plain causal one (ggml_flash_attn_ext_set_causal_prefix). Everything
// else keeps the existing path. K/V are first packed (pack_kv) into WMMA-fragment order in a pool buffer:
//   Kp[kvh][T/16][16 db][16 key][16 d], Vp[kvh][T/16][16 db][16 d][16 slot] (slots: keys 0,2,..,14 then 1,3,..,15).
// pp2048 1706 -> 1735 t/s; NMSE vs fp64 causal reference ~3e-9 (existing path ~3e-5).
#if defined(GGML_USE_HIP) && defined(RDNA3_5)
#define FA7_ENABLED 1
#endif
namespace fa7 {
constexpr int HQ = 24, HKV = 2, D = 256, G = HQ / HKV;
constexpr int NH = 1, NK = 32, PD = 1;
constexpr int NKB = NK / 16, NW = 2 * NH, NTH = NW * 32, HDB = 8;
constexpr float LAZY = 8.0f;
static_assert(8 % (PD + 1) == 0 && G % NH == 0, "fa7 config");
typedef __attribute__((ext_vector_type(16))) _Float16 v16h;
typedef __attribute__((ext_vector_type(8))) float v8f;
typedef __attribute__((ext_vector_type(4))) unsigned int u4;
typedef __attribute__((ext_vector_type(8))) unsigned int u8v;
typedef __attribute__((ext_vector_type(2))) _Float16 h2;
#if defined(FA7_ENABLED)
static __device__ __forceinline__ unsigned xchu(unsigned x) {
    return __builtin_amdgcn_permlanex16(x, x, 0x76543210, 0xfedcba98, false, false);
}
static __device__ __forceinline__ float xch(float x) { return __builtin_bit_cast(float, xchu(__builtin_bit_cast(unsigned, x))); }
static __device__ __forceinline__ void lds_barrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
#define FA7_SCHED_FENCE() __builtin_amdgcn_sched_barrier(0)
#endif

// Q: element (d, t, hq) at Q[t * sq1 + hq * sq2 + d] (floats); O: [T][HQ][D] f32; Kp/Vp: packed, one stream; T % 32 == 0
// GATED: O = sigmoid(gate) * O (gate element (d, hq, t) at gate[t * sg1 + hq * sg2 + d]), O16 the 16-bit copy, O may be null
template <bool GATED = false>
__global__ void __launch_bounds__(NTH) fa_kernel(const float * __restrict__ Q, const _Float16 * __restrict__ Kp,
        const _Float16 * __restrict__ Vp, float * __restrict__ O, const float scale_log2, const int T,
        const int64_t sq1, const int64_t sq2, const float * __restrict__ gate = nullptr, const int64_t sg1 = 0, const int64_t sg2 = 0,
        uint16_t * __restrict__ O16 = nullptr, const bool o16_f16 = false) {
#if defined(FA7_ENABLED)
    constexpr int NHG = G / NH;
    __shared__ float4 xs[2][NW][NKB * 2][32];
    const int bid = blockIdx.x;
    const int tb = (T / 16 - 1) - bid / (HKV * NHG);   // heavy first
    const int rem = bid % (HKV * NHG);
    const int kvh = rem / NHG, hg = rem % NHG;
    const int tq0 = tb * 16;
    const int tid = threadIdx.x, w = tid >> 5, l = tid & 31, lq = l & 15, hi = l >> 4;
    const int hd = w & 1, hq = kvh * G + hg * NH + (w >> 1);
    const int tq = tq0 + lq;

    v16h q[HDB];
#pragma unroll
    for (int j = 0; j < HDB; ++j) {
        const float * qp = Q + (int64_t) tq * sq1 + (int64_t) hq * sq2 + (hd * HDB + j) * 16;
#pragma unroll
        for (int e = 0; e < 16; ++e) q[j][e] = (_Float16) qp[e];
    }
    v8f o[HDB];
#pragma unroll
    for (int i = 0; i < HDB; ++i) o[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, lsum = 0.f;

    const int nt = (tq0 + 16 + NK - 1) / NK;
    const _Float16 * Kh = Kp + (size_t) kvh * T * D;
    const _Float16 * Vh = Vp + (size_t) kvh * T * D;
    const int loff = hd * HDB * 256 + lq * 16;
    const int voff0 = loff + hi * 8, voff1 = loff + 8 - hi * 8;
    // buffer loads: per-lane byte offset in a VGPR (constant), tile offset in an SGPR, frag offset as immediate
    const __amdgpu_buffer_rsrc_t rk = __builtin_amdgcn_make_buffer_rsrc((void *) Kh, 0, T * D * 2, 0x31004000);
    const __amdgpu_buffer_rsrc_t rv = __builtin_amdgcn_make_buffer_rsrc((void *) Vh, 0, T * D * 2, 0x31004000);
    const int lk = loff * 2, lv0 = voff0 * 2, lv1 = voff1 * 2;
    auto ldk = [&](int t, int db, int kb) -> v16h {
        const int so = (t * NK + kb * 16) * D * 2 + db * 512;
        const u4 a = __builtin_bit_cast(u4, __builtin_amdgcn_raw_buffer_load_b128(rk, lk, so, 0));
        const u4 b = __builtin_bit_cast(u4, __builtin_amdgcn_raw_buffer_load_b128(rk, lk + 16, so, 0));
        return __builtin_bit_cast(v16h, u8v{a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]});
    };
    auto ldv = [&](int t, int st, int i) -> v16h {   // frag i of PV step st, tile t; halves swapped in lanes 16-31
        const int db = (st >> 1) * 2 + i, kb = st & 1;   // step = (d-block pair, key block): independent WMMA chains
        const int so = (t * NK + kb * 16) * D * 2 + db * 512;
        const u4 a = __builtin_bit_cast(u4, __builtin_amdgcn_raw_buffer_load_b128(rv, lv0, so, 0));
        const u4 b = __builtin_bit_cast(u4, __builtin_amdgcn_raw_buffer_load_b128(rv, lv1, so, 0));
        return __builtin_bit_cast(v16h, u8v{a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]});
    };

    v16h buf[PD + 1][NKB];
    v16h pb[NKB];
    v8f s[NKB];

    auto s_phase = [&](int t, auto && pref) {
#pragma unroll
        for (int kb = 0; kb < NKB; ++kb) s[kb] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int st = 0; st < 8; ++st) {
            pref(st);
#pragma unroll
            for (int kb = 0; kb < NKB; ++kb)
                s[kb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(buf[st % (PD + 1)][kb], q[st], s[kb]);
            FA7_SCHED_FENCE();
        }
    };
    auto pv_phase = [&](auto && pref) {
#pragma unroll
        for (int st = 0; st < 8; ++st) {
            pref(st);
#pragma unroll
            for (int kb = 0; kb < NKB; ++kb)
                o[(st >> 1) * 2 + kb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(buf[st % (PD + 1)][kb], pb[st & 1], o[(st >> 1) * 2 + kb]);
            FA7_SCHED_FENCE();
        }
    };
    auto put_s = [&](int t) {
#pragma unroll
        for (int kb = 0; kb < NKB; ++kb)
#pragma unroll
            for (int c = 0; c < 2; ++c)
                xs[t & 1][w][kb * 2 + c][l] = float4{s[kb][4 * c], s[kb][4 * c + 1], s[kb][4 * c + 2], s[kb][4 * c + 3]};
    };
    auto softmax = [&](int t) {
#pragma unroll
        for (int kb = 0; kb < NKB; ++kb)
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const float4 y = xs[t & 1][w ^ 1][kb * 2 + c][l];
                s[kb][4 * c] += y.x; s[kb][4 * c + 1] += y.y; s[kb][4 * c + 2] += y.z; s[kb][4 * c + 3] += y.w;
            }
        const int key0 = t * NK;
        if (key0 + NK - 1 > tq0) {   // diagonal tile (uniform)
#pragma unroll
            for (int kb = 0; kb < NKB; ++kb)
#pragma unroll
                for (int v = 0; v < 8; ++v) if (key0 + kb * 16 + 2 * v + hi > tq) s[kb][v] = -INFINITY;
        }
        float mx = -INFINITY;
#pragma unroll
        for (int kb = 0; kb < NKB; ++kb)
#pragma unroll
            for (int v = 0; v < 8; ++v) mx = fmaxf(mx, s[kb][v]);
        mx = fmaxf(mx, xch(mx)) * scale_log2;
        if (__builtin_amdgcn_ballot_w32(mx > m + LAZY)) {   // lazy rescale: p <= 2^LAZY
            const float mn = fmaxf(m, mx);
            const float alpha = __builtin_amdgcn_exp2f(m - mn);
            m = mn;
            lsum *= alpha;
#pragma unroll
            for (int i = 0; i < HDB; ++i) o[i] *= alpha;
        }
        float ps = 0.f;
#pragma unroll
        for (int kb = 0; kb < NKB; ++kb) {
#pragma unroll
            for (int v = 0; v < 8; ++v) { const float p = __builtin_amdgcn_exp2f(fmaf(s[kb][v], scale_log2, -m)); s[kb][v] = p; ps += p; }
            unsigned u[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                u[i] = __builtin_bit_cast(unsigned, h2{(_Float16) s[kb][2 * i], (_Float16) s[kb][2 * i + 1]});
                u[4 + i] = xchu(u[i]);
            }
            pb[kb] = __builtin_bit_cast(v16h, u);
        }
        lsum += ps;
    };

    const int tl = nt - 1;
#pragma unroll
    for (int p = 0; p < PD; ++p)
#pragma unroll
        for (int kb = 0; kb < NKB; ++kb) buf[p][kb] = ldk(0, p, kb);
    for (int t = 0; t < nt; ++t) {
        s_phase(t, [&](int st) {
            const int sn = st + PD;
#pragma unroll
            for (int kb = 0; kb < NKB; ++kb) buf[sn % (PD + 1)][kb] = sn < 8 ? ldk(t, sn, kb) : ldv(t, sn - 8, kb);
        });
        put_s(t);
        lds_barrier();
        softmax(t);
        const int tn = min(t + 1, tl);
        pv_phase([&](int st) {
            const int sn = st + PD;
#pragma unroll
            for (int kb = 0; kb < NKB; ++kb) buf[sn % (PD + 1)][kb] = sn < 8 ? ldv(t, sn, kb) : ldk(tn, sn - 8, kb);
        });
    }
    lsum += xch(lsum);
    const float inv = 1.f / lsum;
    if constexpr (GATED) {
        const size_t ob = ((size_t) tq * HQ + hq) * D + hd * HDB * 16;
        (void) 0;
        const float * grow = gate + (int64_t) tq * sg1 + (int64_t) hq * sg2 + hd * HDB * 16;
#pragma unroll
        for (int j = 0; j < HDB; ++j) {
            // 16 contiguous gate floats of this d-block (16-byte loads); the lane takes the even (hi=0) or odd (hi=1) ones
            float gv[16];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float4 g4 = *(const float4 *) (grow + j * 16 + 4 * q);
                gv[4 * q] = g4.x; gv[4 * q + 1] = g4.y; gv[4 * q + 2] = g4.z; gv[4 * q + 3] = g4.w;
            }
            float r[8];
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float g = hi ? gv[2 * v + 1] : gv[2 * v];
                r[v] = (1.0f / (1.0f + expf(-g))) * (o[j][v] * inv);   // op_sigmoid(gate) * O, as unary_gated_op_kernel
            }
            // pair up with the other half-wave (lanes 16-31 hold the odd d) so each lane stores 2 adjacent values
            float rp[8];
#pragma unroll
            for (int v = 0; v < 8; ++v) rp[v] = xch(r[v]);
            // hi=0: d = 2v, 2v+1 for v < 4 ; hi=1: d = 2v, 2v+1 for v >= 4
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int v = hi ? k + 4 : k;
                const float a = hi ? rp[v] : r[v];      // even d = 2v
                const float b = hi ? r[v]  : rp[v];     // odd d = 2v + 1
                const int c = j * 16 + 2 * v;
                if (O) *(float2 *) (O + ob + c) = make_float2(a, b);
                if (O16) *(uint32_t *) (O16 + ob + c) = (uint32_t) ggml_cuda_f32_to_act16(a, o16_f16) | ((uint32_t) ggml_cuda_f32_to_act16(b, o16_f16) << 16);
            }
        }
    } else {
    float * orow = O + ((size_t) tq * HQ + hq) * D + hd * HDB * 16;
#pragma unroll
    for (int j = 0; j < HDB; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) orow[j * 16 + 2 * v + hi] = o[j][v] * inv;
    }
#else
    GGML_UNUSED_VARS(Q, Kp, Vp, O, scale_log2, T, sq1, sq2, gate, sg1, sg2, O16, o16_f16);
    NO_DEVICE_CODE;
#endif
}

// pack K/V (element (d, key, h) at K[key * sk1 + h * sk2 + d], halves) -> Kp / Vp. One block per (16-key block, kv head).
__global__ void pack_kv(const _Float16 * __restrict__ K, const _Float16 * __restrict__ V, _Float16 * __restrict__ Kp, _Float16 * __restrict__ Vp,
        const int T, const int64_t sk1, const int64_t sk2, const int64_t sv1, const int64_t sv2) {
#if defined(FA7_ENABLED)
    __shared__ _Float16 vs[16][D + 8];
    const int blk = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int db = tid >> 4, r = tid & 15;
    const size_t ob = ((size_t) h * (T / 16) + blk) * D * 16;
    const _Float16 * kr = K + (int64_t) (blk * 16 + r) * sk1 + (int64_t) h * sk2 + db * 16;
    *(u4 *) (Kp + ob + (db * 16 + r) * 16) = *(const u4 *) kr;
    *(u4 *) (Kp + ob + (db * 16 + r) * 16 + 8) = *(const u4 *) (kr + 8);
    const _Float16 * vr = V + (int64_t) (blk * 16 + r) * sv1 + (int64_t) h * sv2 + db * 16;
    *(u4 *) &vs[r][db * 16] = *(const u4 *) vr;
    *(u4 *) &vs[r][db * 16 + 8] = *(const u4 *) (vr + 8);
    __syncthreads();
    _Float16 tmp[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) tmp[j] = vs[j < 8 ? 2 * j : 2 * (j - 8) + 1][db * 16 + r];   // [even keys, odd keys]
    *(u4 *) (Vp + ob + (db * 16 + r) * 16) = *(const u4 *) tmp;
    *(u4 *) (Vp + ob + (db * 16 + r) * 16 + 8) = *(const u4 *) (tmp + 8);
#else
    GGML_UNUSED_VARS(K, V, Kp, Vp, T, sk1, sk2, sv1, sv2);
    NO_DEVICE_CODE;
#endif
}
} // namespace fa7

static bool ggml_cuda_fa7_supported(const ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    static const bool on = getenv("GGML_FA7") ? atoi(getenv("GGML_FA7")) != 0 : true;   // GGML_FA7=0 disables it
    if (!on) return false;
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3];
    if (!GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[ctx.device].cc)) return false;
    if (!K || !V || !mask || dst->src[4] || dst->src[5] || dst->src[6] || dst->src[7]) return false;
    if (!ggml_flash_attn_ext_get_causal_prefix(dst)) return false;   // the kernel applies index-causal masking itself
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32) return false;
    const int64_t T = Q->ne[1];
    if (Q->ne[0] != fa7::D || K->ne[0] != fa7::D || V->ne[0] != fa7::D || Q->ne[2] != fa7::HQ || K->ne[2] != fa7::HKV || V->ne[2] != fa7::HKV) return false;
    // n_kv == n_q: the ubatch is the whole visible sequence (key j visible to query i iff j <= i)
    if (K->ne[1] != T || V->ne[1] != T || T < 32 || T % 32 != 0 || (int64_t) T * fa7::D * 2 >= INT32_MAX) return false;
    if (K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3] || mask->ne[0] != T) return false;
    float scale, max_bias, softcap;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) return false;
    // element strides: d contiguous, 16-byte aligned rows (the packer / Q loads are vectorized)
    if (Q->nb[0] != 4 || K->nb[0] != 2 || V->nb[0] != 2 || !ggml_is_contiguous(dst)) return false;
    if (Q->nb[1] % 16 || Q->nb[2] % 16 || K->nb[1] % 16 || K->nb[2] % 16 || V->nb[1] % 16 || V->nb[2] % 16) return false;
    if (((uintptr_t) Q->data) % 16 || ((uintptr_t) K->data) % 16 || ((uintptr_t) V->data) % 16) return false;
    return true;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    return false;
#endif
}

// The gated attention output in the FA epilogue. Same expression per element as cont_sigmoid_mul_v4, so
// bit-identical (wikitext PPL unchanged at 2048 and 4096 context on gfx1151).

bool ggml_cuda_qsa_gate_ok(const ggml_backend_cuda_context & ctx, const ggml_tensor * fa) {
#if defined(GGML_USE_HIP)
    // the fused op must reach the prefill kernel (decode is tried first) and the gate spans one stream
    return fa->op == GGML_OP_FLASH_ATTN_EXT && fa->src[0]->ne[3] == 1 &&
        !ggml_cuda_flash_attn_ext_qsa_decode_supported(const_cast<ggml_backend_cuda_context &>(ctx), fa) &&
        ggml_cuda_flash_attn_ext_qsa_prefill_supported(const_cast<ggml_backend_cuda_context &>(ctx), fa);
#else
    GGML_UNUSED(ctx); GGML_UNUSED(fa);
    return false;
#endif
}

bool ggml_cuda_fa7_gate_ok(const ggml_backend_cuda_context & ctx, const ggml_tensor * fa) {
    if (fa->op != GGML_OP_FLASH_ATTN_EXT || fa->src[0]->ne[3] != 1) return false;
#if defined(GGML_USE_HIP)
    return ggml_cuda_fa7_supported(ctx, fa) &&
        !ggml_cuda_flash_attn_ext_qsa_decode_supported(const_cast<ggml_backend_cuda_context &>(ctx), fa) &&
        !ggml_cuda_flash_attn_ext_qsa_prefill_supported(const_cast<ggml_backend_cuda_context &>(ctx), fa);
#else
    return false;
#endif
}

void ggml_cuda_fa7_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * gate, float * out32,
        uint16_t * dst16, bool dst16_f16) {
    ggml_cuda_set_device(ctx.device);
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2];
    cudaStream_t stream = ctx.stream();
    const int T = (int) Q->ne[1];
    GGML_ASSERT(Q->ne[3] == 1 && (dst16 != nullptr || out32 != nullptr) && gate->nb[0] == 4);
    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const float scale_log2 = scale * 1.4426950408889634f;
    const size_t nkv = (size_t) fa7::HKV * T * fa7::D;
    ggml_cuda_pool_alloc<_Float16> Kp(ctx.pool(), nkv), Vp(ctx.pool(), nkv);
    const int grid = (T / 16) * fa7::HKV * (fa7::G / fa7::NH);
    fa7::pack_kv<<<dim3(T / 16, fa7::HKV), 256, 0, stream>>>((const _Float16 *) K->data, (const _Float16 *) V->data, Kp.get(), Vp.get(), T,
        K->nb[1] / 2, K->nb[2] / 2, V->nb[1] / 2, V->nb[2] / 2);
    fa7::fa_kernel<true><<<grid, fa7::NTH, 0, stream>>>((const float *) Q->data, Kp.get(), Vp.get(), out32, scale_log2, T, Q->nb[1] / 4, Q->nb[2] / 4,
        (const float *) gate->data, gate->nb[2] / 4, gate->nb[1] / 4, dst16, dst16_f16);
    CUDA_CHECK(cudaGetLastError());
}

static void ggml_cuda_fa7(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2];
    cudaStream_t stream = ctx.stream();
    const int T = (int) Q->ne[1];
    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const float scale_log2 = scale * 1.4426950408889634f;
    const size_t nkv = (size_t) fa7::HKV * T * fa7::D;
    ggml_cuda_pool_alloc<_Float16> Kp(ctx.pool(), nkv), Vp(ctx.pool(), nkv);
    const int grid = (T / 16) * fa7::HKV * (fa7::G / fa7::NH);
    for (int64_t s = 0; s < Q->ne[3]; ++s) {
        const _Float16 * k = (const _Float16 *) ((const char *) K->data + s * K->nb[3]);
        const _Float16 * v = (const _Float16 *) ((const char *) V->data + s * V->nb[3]);
        const float * q = (const float *) ((const char *) Q->data + s * Q->nb[3]);
        float * o = (float *) dst->data + (size_t) s * T * fa7::HQ * fa7::D;
        fa7::pack_kv<<<dim3(T / 16, fa7::HKV), 256, 0, stream>>>(k, v, Kp.get(), Vp.get(), T,
            K->nb[1] / 2, K->nb[2] / 2, V->nb[1] / 2, V->nb[2] / 2);
        fa7::fa_kernel<false><<<grid, fa7::NTH, 0, stream>>>(q, Kp.get(), Vp.get(), o, scale_log2, T, Q->nb[1] / 4, Q->nb[2] / 4);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
#if defined(GGML_USE_HIP)
    if (ggml_cuda_flash_attn_ext_qsa_decode_supported(ctx, dst)) {
        ggml_cuda_flash_attn_ext_qsa_decode(ctx, dst);
        return;
    }
    if (ggml_cuda_flash_attn_ext_qsa_prefill_supported(ctx, dst)) {
        ggml_cuda_flash_attn_ext_qsa_prefill(ctx, dst);
        return;
    }
    {
        const bool fa7_ok = ggml_cuda_fa7_supported(ctx, dst);
        if (fa7_ok) {
            ggml_cuda_fa7(ctx, dst);
            return;
        }
    }
#endif
    // a selected-key op without a mask carries its visibility only in src[5], which the dense kernels ignore:
    // reaching them would attend to the whole padded cache
    if (dst->src[5] && !dst->src[3]) {
        const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * ids = dst->src[5];
        GGML_ABORT("flash_attn_ext: maskless selected-key attention was not taken by a QSA kernel (%s): q [%lld,%lld,%lld,%lld] %s nb %zu/%zu, "
                   "k [%lld,%lld,%lld,%lld] %s nb %zu/%zu/%zu, v %s nb %zu/%zu, ids [%lld,%lld,%lld,%lld] nb %zu, src6 %d src7 %d p4 %d dst contiguous %d",
                   dst->name, (long long) q->ne[0], (long long) q->ne[1], (long long) q->ne[2], (long long) q->ne[3], ggml_type_name(q->type), q->nb[1], q->nb[2],
                   (long long) k->ne[0], (long long) k->ne[1], (long long) k->ne[2], (long long) k->ne[3], ggml_type_name(k->type), k->nb[0], k->nb[1], k->nb[2],
                   ggml_type_name(v->type), v->nb[1], v->nb[2], (long long) ids->ne[0], (long long) ids->ne[1], (long long) ids->ne[2], (long long) ids->ne[3], ids->nb[1],
                   dst->src[6] != nullptr, dst->src[7] != nullptr, ggml_get_op_params_i32(dst, 4), ggml_is_contiguous(dst));
    }


    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    // a maskless selected-key op (visibility only in src[5]) runs on the QSA kernels alone: an op neither takes must
    // go to another backend instead of the abort in ggml_cuda_flash_attn_ext
    if (dst->src[5] && !dst->src[3]) {
#if defined(GGML_USE_HIP)
        return ggml_cuda_flash_attn_ext_qsa_decode_supported_dev(device, dst) ||
               ggml_cuda_flash_attn_ext_qsa_prefill_supported_dev(device, dst);
#else
        return false;
#endif
    }
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
