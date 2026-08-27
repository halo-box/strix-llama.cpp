#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstring>

#ifndef GGML_ROCMFP4_RDNA35_NWARPS
#define GGML_ROCMFP4_RDNA35_NWARPS 2
#endif

#if GGML_ROCMFP4_RDNA35_NWARPS != 1 && GGML_ROCMFP4_RDNA35_NWARPS != 2 && \
    GGML_ROCMFP4_RDNA35_NWARPS != 4 && GGML_ROCMFP4_RDNA35_NWARPS != 8
#error "GGML_ROCMFP4_RDNA35_NWARPS must be one of: 1, 2, 4, 8"
#endif

#ifndef GGML_ROCMFP4_RDNA35_NWARPS_MAX_NCOLS
#define GGML_ROCMFP4_RDNA35_NWARPS_MAX_NCOLS 2
#endif

#if GGML_ROCMFP4_RDNA35_NWARPS_MAX_NCOLS < 1 || GGML_ROCMFP4_RDNA35_NWARPS_MAX_NCOLS > MMVQ_MAX_BATCH_SIZE
#error "GGML_ROCMFP4_RDNA35_NWARPS_MAX_NCOLS must be between 1 and MMVQ_MAX_BATCH_SIZE"
#endif

#ifndef GGML_ROCMFP4_RDNA35_MMID_MAX_BATCH
#define GGML_ROCMFP4_RDNA35_MMID_MAX_BATCH MMVQ_MAX_BATCH_SIZE
#endif

#if GGML_ROCMFP4_RDNA35_MMID_MAX_BATCH < 1 || GGML_ROCMFP4_RDNA35_MMID_MAX_BATCH > MMVQ_MAX_BATCH_SIZE
#error "GGML_ROCMFP4_RDNA35_MMID_MAX_BATCH must be between 1 and MMVQ_MAX_BATCH_SIZE"
#endif

#ifndef GGML_ROCMFP4_RDNA35_RPB_WIDE
#define GGML_ROCMFP4_RDNA35_RPB_WIDE 1
#endif

#if GGML_ROCMFP4_RDNA35_RPB_WIDE != 1 && GGML_ROCMFP4_RDNA35_RPB_WIDE != 2
#error "GGML_ROCMFP4_RDNA35_RPB_WIDE must be 1 or 2"
#endif

#ifndef GGML_ROCMFP4_RDNA35_RPB_WIDE_DUAL
#define GGML_ROCMFP4_RDNA35_RPB_WIDE_DUAL GGML_ROCMFP4_RDNA35_RPB_WIDE
#endif

#ifndef GGML_ROCMFP4_RDNA35_RPB_WIDE_FAST
#define GGML_ROCMFP4_RDNA35_RPB_WIDE_FAST GGML_ROCMFP4_RDNA35_RPB_WIDE
#endif

#ifndef GGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK
#define GGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK 2
#endif

#if GGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK < 1 || GGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK > 4
#error "GGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK must be between 1 and 4"
#endif

#ifndef GGML_ROCMFPX_RDNA35_NWARPS
#define GGML_ROCMFPX_RDNA35_NWARPS 1
#endif

#ifndef GGML_Q8_0_RDNA35_NWARPS
#define GGML_Q8_0_RDNA35_NWARPS 1
#endif

#if GGML_Q8_0_RDNA35_NWARPS != 1 && GGML_Q8_0_RDNA35_NWARPS != 2 && \
    GGML_Q8_0_RDNA35_NWARPS != 4 && GGML_Q8_0_RDNA35_NWARPS != 8
#error "GGML_Q8_0_RDNA35_NWARPS must be one of: 1, 2, 4, 8"
#endif

#if GGML_ROCMFPX_RDNA35_NWARPS != 1 && GGML_ROCMFPX_RDNA35_NWARPS != 2 && \
    GGML_ROCMFPX_RDNA35_NWARPS != 4 && GGML_ROCMFPX_RDNA35_NWARPS != 8
#error "GGML_ROCMFPX_RDNA35_NWARPS must be one of: 1, 2, 4, 8"
#endif

#ifndef GGML_ROCMFPX_RDNA35_NWARPS_MAX_NCOLS
#define GGML_ROCMFPX_RDNA35_NWARPS_MAX_NCOLS 2
#endif

#if GGML_ROCMFPX_RDNA35_NWARPS_MAX_NCOLS < 1 || GGML_ROCMFPX_RDNA35_NWARPS_MAX_NCOLS > MMVQ_MAX_BATCH_SIZE
#error "GGML_ROCMFPX_RDNA35_NWARPS_MAX_NCOLS must be between 1 and MMVQ_MAX_BATCH_SIZE"
#endif

#ifndef GGML_ROCMFP2_RDNA35_NWARPS
#define GGML_ROCMFP2_RDNA35_NWARPS GGML_ROCMFPX_RDNA35_NWARPS
#endif

#if GGML_ROCMFP2_RDNA35_NWARPS != 1 && GGML_ROCMFP2_RDNA35_NWARPS != 2 && \
    GGML_ROCMFP2_RDNA35_NWARPS != 4 && GGML_ROCMFP2_RDNA35_NWARPS != 8
#error "GGML_ROCMFP2_RDNA35_NWARPS must be one of: 1, 2, 4, 8"
#endif

#ifndef GGML_ROCMFP2_RDNA35_NWARPS_MAX_NCOLS
#define GGML_ROCMFP2_RDNA35_NWARPS_MAX_NCOLS GGML_ROCMFPX_RDNA35_NWARPS_MAX_NCOLS
#endif

#ifndef GGML_ROCMFP2_RDNA35_NWARPS_MIN_NCOLS
#define GGML_ROCMFP2_RDNA35_NWARPS_MIN_NCOLS 1
#endif

#if GGML_ROCMFP2_RDNA35_NWARPS_MIN_NCOLS < 1 || GGML_ROCMFP2_RDNA35_NWARPS_MIN_NCOLS > GGML_ROCMFP2_RDNA35_NWARPS_MAX_NCOLS
#error "GGML_ROCMFP2_RDNA35_NWARPS_MIN_NCOLS must be between 1 and GGML_ROCMFP2_RDNA35_NWARPS_MAX_NCOLS"
#endif

#if GGML_ROCMFP2_RDNA35_NWARPS_MAX_NCOLS < 1 || GGML_ROCMFP2_RDNA35_NWARPS_MAX_NCOLS > MMVQ_MAX_BATCH_SIZE
#error "GGML_ROCMFP2_RDNA35_NWARPS_MAX_NCOLS must be between 1 and MMVQ_MAX_BATCH_SIZE"
#endif

#ifndef GGML_ROCMFPX_RDNA35_MMID_MAX_BATCH
#define GGML_ROCMFPX_RDNA35_MMID_MAX_BATCH MMVQ_MAX_BATCH_SIZE
#endif

#if GGML_ROCMFPX_RDNA35_MMID_MAX_BATCH < 1 || GGML_ROCMFPX_RDNA35_MMID_MAX_BATCH > MMVQ_MAX_BATCH_SIZE
#error "GGML_ROCMFPX_RDNA35_MMID_MAX_BATCH must be between 1 and MMVQ_MAX_BATCH_SIZE"
#endif

#ifndef GGML_ROCMFPX_RDNA35_RPB_WIDE
#define GGML_ROCMFPX_RDNA35_RPB_WIDE 1
#endif

#if GGML_ROCMFPX_RDNA35_RPB_WIDE != 1 && GGML_ROCMFPX_RDNA35_RPB_WIDE != 2
#error "GGML_ROCMFPX_RDNA35_RPB_WIDE must be 1 or 2"
#endif

#ifndef GGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK
#define GGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK 2
#endif

#if GGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK < 1 || GGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK > 4
#error "GGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK must be between 1 and 4"
#endif

#ifndef GGML_ROCMFPX_HY3_ADAPTIVE_MOE_RPB
#define GGML_ROCMFPX_HY3_ADAPTIVE_MOE_RPB 0
#endif

#if GGML_ROCMFPX_HY3_ADAPTIVE_MOE_RPB != 0 && GGML_ROCMFPX_HY3_ADAPTIVE_MOE_RPB != 1
#error "GGML_ROCMFPX_HY3_ADAPTIVE_MOE_RPB must be 0 or 1"
#endif

#if GGML_ROCMFP4_RDNA35_RPB_WIDE_DUAL != 1 && GGML_ROCMFP4_RDNA35_RPB_WIDE_DUAL != 2
#error "GGML_ROCMFP4_RDNA35_RPB_WIDE_DUAL must be 1 or 2"
#endif

#if GGML_ROCMFP4_RDNA35_RPB_WIDE_FAST != 1 && GGML_ROCMFP4_RDNA35_RPB_WIDE_FAST != 2
#error "GGML_ROCMFP4_RDNA35_RPB_WIDE_FAST must be 1 or 2"
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_Q4_0_ROCMFP4:
                                return vec_dot_rocmfp4_q8_1;
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
                                return vec_dot_rocmfp4_fast_q8_1;
        case GGML_TYPE_Q3_0_ROCMFPX:
                                return vec_dot_rocmfpx_fp3_q8_1;
        case GGML_TYPE_Q2_0_ROCMFPX:
                                return vec_dot_rocmfpx_fp2_q8_1;
        case GGML_TYPE_Q6_0_ROCMFPX:
                                return vec_dot_rocmfpx_fp6_q8_1;
        case GGML_TYPE_Q8_0_ROCMFPX:
                                return vec_dot_rocmfpx_fp8_q8_1;
        case GGML_TYPE_Q7_0_ROCMFPX:
                                return vec_dot_rocmfpx_q7_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0_ROCMFP4:
                                return VDR_ROCMFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
                                return VDR_ROCMFP4_FAST_Q8_1_MMVQ;
        case GGML_TYPE_Q3_0_ROCMFPX:
                                return VDR_ROCMFP3_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0_ROCMFPX:
                                return VDR_ROCMFP2_Q8_1_MMVQ;
        case GGML_TYPE_Q6_0_ROCMFPX:
                                return VDR_ROCMFP6_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0_ROCMFPX:
                                return VDR_ROCMFP8_Q8_1_MMVQ;
        case GGML_TYPE_Q7_0_ROCMFPX:
                                return VDR_ROCMFP7_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA3_5,
    MMVQ_PARAMETERS_RDNA4
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA3_5;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA3_5;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_Q4_0_ROCMFP4:
                                return 4;
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
                                return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_Q4_0_ROCMFP4:
                                return 7;
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
                                return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3_5(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0_ROCMFP4:
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
                                return GGML_ROCMFP4_RDNA35_MMID_MAX_BATCH;
        case GGML_TYPE_Q3_0_ROCMFPX:
        case GGML_TYPE_Q2_0_ROCMFPX:
        case GGML_TYPE_Q6_0_ROCMFPX:
        case GGML_TYPE_Q8_0_ROCMFPX:
        case GGML_TYPE_Q7_0_ROCMFPX:
                                return GGML_ROCMFPX_RDNA35_MMID_MAX_BATCH;
        default:                return get_mmvq_mmid_max_batch_rdna3(type);
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_Q4_0_ROCMFP4:
                                return 5;
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
                                return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
            return get_mmvq_mmid_max_batch_rdna3_5(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3_5)
    return get_mmvq_mmid_max_batch_rdna3_5(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_5) {
        if (type == GGML_TYPE_Q8_0 && ncols_dst == 1) {
            return GGML_Q8_0_RDNA35_NWARPS;
        }
        if (ncols_dst < 1) {
            return 1;
        }
        switch (type) {
            case GGML_TYPE_Q4_0_ROCMFP4:
            case GGML_TYPE_Q4_0_ROCMFP4_FAST:
                return ncols_dst <= GGML_ROCMFP4_RDNA35_NWARPS_MAX_NCOLS ? GGML_ROCMFP4_RDNA35_NWARPS : 1;
            case GGML_TYPE_Q2_0_ROCMFPX:
                return ncols_dst >= GGML_ROCMFP2_RDNA35_NWARPS_MIN_NCOLS &&
                       ncols_dst <= GGML_ROCMFP2_RDNA35_NWARPS_MAX_NCOLS ? GGML_ROCMFP2_RDNA35_NWARPS : 1;
            case GGML_TYPE_Q3_0_ROCMFPX:
            case GGML_TYPE_Q6_0_ROCMFPX:
            case GGML_TYPE_Q8_0_ROCMFPX:
            case GGML_TYPE_Q7_0_ROCMFPX:
                return ncols_dst <= GGML_ROCMFPX_RDNA35_NWARPS_MAX_NCOLS ? GGML_ROCMFPX_RDNA35_NWARPS : 1;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_5) {
        if (ncols_dst >= 5 && ncols_dst <= 8) {
            switch (type) {
                case GGML_TYPE_Q4_0_ROCMFP4:
                    return GGML_ROCMFP4_RDNA35_RPB_WIDE_DUAL;
                case GGML_TYPE_Q4_0_ROCMFP4_FAST:
                    return GGML_ROCMFP4_RDNA35_RPB_WIDE_FAST;
                case GGML_TYPE_Q3_0_ROCMFPX:
                case GGML_TYPE_Q2_0_ROCMFPX:
                case GGML_TYPE_Q6_0_ROCMFPX:
                case GGML_TYPE_Q8_0_ROCMFPX:
                case GGML_TYPE_Q7_0_ROCMFPX:
                    return GGML_ROCMFPX_RDNA35_RPB_WIDE;
                default:
                    break;
            }
        }
    }
    return 1;
}

// FP2 has a very small payload but a non-trivial byte-to-int8 expansion.  The
// generic multi-column loop calls vec_dot once per destination column, causing
// that expansion (and the FP2 scale decode) to be inlined once per column.
// MTP verification uses exactly these small multi-column shapes.  Expand the
// target weights once and reuse them against every Q8_1 activation column.
template <int ncols_dst, int rows_per_cuda_block>
static __device__ __forceinline__ void vec_dot_rocmfpx_fp2_q8_1_ncols(
        const void * __restrict__ vx,
        const block_q8_1 * __restrict__ y,
        const uint32_t stride_col_y,
        const int kbx,
        const int kby,
        const int kqs,
        float (&tmp)[ncols_dst][rows_per_cuda_block],
        const int row) {
    const block_rocmfp2 * bq2 = (const block_rocmfp2 *) vx + kbx;

    int values[VDR_ROCMFP2_Q8_1_MMVQ];
#pragma unroll
    for (int i = 0; i < VDR_ROCMFP2_Q8_1_MMVQ; ++i) {
        values[i] = rocmfpx_pack4_fp2_vec_cuda(bq2->qs[kqs + i]);
    }

#if VDR_ROCMFP2_Q8_1_MMVQ <= 4
    const float dx = rocmfpx_ue4m3_to_fp32_finite(bq2->e[kqs / 4]);
#endif

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        const block_q8_1 * bq8 = &y[j*stride_col_y + kby];
        const int * q8 = (const int *) bq8->qs;

#if VDR_ROCMFP2_Q8_1_MMVQ <= 4
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < VDR_ROCMFP2_Q8_1_MMVQ; ++i) {
            sumi = ggml_cuda_dp4a(values[i], q8[kqs + i], sumi);
        }
        tmp[j][row] += dx * __low2float(bq8->ds) * sumi;
#else
        int sumi0 = 0;
        int sumi1 = 0;
#pragma unroll
        for (int i = 0; i < VDR_ROCMFP2_Q8_1_MMVQ; ++i) {
            const int group = kqs + i;
            if (group < QI_ROCMFP2/2) {
                sumi0 = ggml_cuda_dp4a(values[i], q8[group], sumi0);
            } else {
                sumi1 = ggml_cuda_dp4a(values[i], q8[group], sumi1);
            }
        }
        const float dx0 = rocmfpx_ue4m3_to_fp32_finite(bq2->e[0]);
        const float dx1 = rocmfpx_ue4m3_to_fp32_finite(bq2->e[1]);
        tmp[j][row] += __low2float(bq8->ds) * (dx0*sumi0 + dx1*sumi1);
#endif
    }
}

template <ggml_type type>
static constexpr int calc_moe_mmvq_rows_per_block() {
#if defined(GGML_USE_HIP)
    if constexpr (type == GGML_TYPE_Q4_0_ROCMFP4 || type == GGML_TYPE_Q4_0_ROCMFP4_FAST) {
        return GGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK;
    }
    if constexpr (type == GGML_TYPE_Q2_0_ROCMFPX || type == GGML_TYPE_Q3_0_ROCMFPX ||
                  type == GGML_TYPE_Q6_0_ROCMFPX || type == GGML_TYPE_Q8_0_ROCMFPX ||
                  type == GGML_TYPE_Q7_0_ROCMFPX) {
        return GGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK;
    }
#endif
    return 2;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids, const ggml_cuda_mm_fusion_args_device fusion, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id);
    constexpr int rows_per_cuda_block = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    ggml_glu_op active_glu;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
    }


    float x_biases[ncols_dst]    = { 0.0f };
    float gate_biases[ncols_dst] = { 0.0f };
    if constexpr (has_fusion) {
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (use_bias) {
            x_bias = x_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            // 1. Hide latency by prefetching bias and gate here
            // 2. load only on threads that won't die after partial sum calculation
            if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
        if (use_gate_bias) {
            gate_bias = gate_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

        if constexpr (type == GGML_TYPE_Q2_0_ROCMFPX) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                vec_dot_rocmfpx_fp2_q8_1_ncols(
                    vx, y, stride_col_y, kbx_offset + i*stride_row_x + kbx, kby, kqs, tmp, i);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        vec_dot_rocmfpx_fp2_q8_1_ncols(
                            vgate, y, stride_col_y, kbx_offset + i*stride_row_x + kbx, kby, kqs, tmp_gate, i);
                    }
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    tmp[j][i] += vec_dot_q_cuda(
                        vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_gate[j][i] += vec_dot_q_cuda(
                                vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                        }
                    }
                }
            }
        }
    }

    if constexpr (nwarps > 1) {
        __shared__ float tmp_shared[nwarps-1][ncols_dst][rows_per_cuda_block][warp_size];
        __shared__ float tmp_shared_gate[has_fusion ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
        if constexpr (!has_fusion) {
            (void) tmp_shared_gate;
        } else if (!use_gate) {
            (void) tmp_shared_gate;
        }

        if (threadIdx.y > 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                        }
                    }
                }
            }
        }

        __syncthreads();
        if (threadIdx.y > 0) {
            return;
        }

        // sum up partial sums from the other warps before the warp reduction
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
                for (int l = 0; l < nwarps-1; ++l) {
                    tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                        }
                    }
                }
            }
        }
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // reduce the remaining warp-local partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            float result = tmp[j][threadIdx.x];
            if constexpr (has_fusion) {
                if (use_bias) {
                    result += x_biases[j];
                }
                if (use_gate) {
                    float gate_value = tmp_gate[j][threadIdx.x];
                    if (use_gate_bias) {
                        gate_value += gate_biases[j];
                    }
                    switch (active_glu) {
                        case GGML_GLU_OP_SWIGLU:
                            result *= ggml_cuda_op_silu_single(gate_value);
                            break;
                        case GGML_GLU_OP_GEGLU:
                            result *= ggml_cuda_op_gelu_single(gate_value);
                            break;
                        case GGML_GLU_OP_SWIGLU_OAI: {
                            result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                            break;
                        }
                        default:
                            result = result * gate_value;
                            break;
                    }
                }
            }
            dst[j*stride_col_dst + threadIdx.x] = result;
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, active_glu, gate_bias, x_bias, tmp_gate);
    }
}

// gfx1151 standard-Q8 register outer-product experiment for small-column MMVQ.
//
// The stock kernel assigns one wave to one output row. This 2-D tile assigns a
// wave to a small row x token rectangle so both weights and activations can be
// reused without an inter-wave reduction or LDS. The common N=4, T=2, R=2
// instance is:
//
//   wave 0: rows [r, r+1] x tokens [0, 1]
//   wave 1: rows [r, r+1] x tokens [2, 3]
//
// The canonical block_q8_0 and block_q8_1 representations are consumed
// directly. Arithmetic remains signed dot4 with the original FP16 scales. To
// limit live ranges, the shorter tile dimension is retained while the longer
// dimension is streamed through the dot-product body.
template <int ncols_dst, int tokens_per_wave, int rows_per_block>
__launch_bounds__((ncols_dst/tokens_per_wave)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q8_0_rdna35_2d(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float * __restrict__ dst,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_col_dst,
        const uint3 channel_ratio,
        const uint32_t stride_channel_x,
        const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst,
        const uint3 sample_ratio,
        const uint32_t stride_sample_x,
        const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst) {
    static_assert(
        tokens_per_wave == 1 || tokens_per_wave == 2 || tokens_per_wave == 4,
        "tokens_per_wave must be a small power of two");
    static_assert(ncols_dst % tokens_per_wave == 0);
    static_assert(rows_per_block >= 1 && rows_per_block <= 8);

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int qk = QK8_0;
    constexpr int qi = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int blocks_per_iter = vdr*warp_size/qi;

    const int lane = threadIdx.x;
    const int token0 = threadIdx.y*tokens_per_wave;
    const int row0 = rows_per_block*blockIdx.x;
    const int blocks_per_row_x = ncols_x/qk;
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst = blockIdx.z;
    const uint32_t channel_x = fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y = channel_dst;
    const uint32_t sample_x = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y = sample_dst;

    const block_q8_1 * y =
        (const block_q8_1 *) vy +
        sample_y*stride_sample_y +
        channel_y*stride_channel_y;
    const int kbx_offset =
        sample_x*stride_sample_x +
        channel_x*stride_channel_x +
        row0*stride_row_x;

    float tmp[tokens_per_wave][rows_per_block] = {{0.0f}};

    for (int kbx = lane/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kqs = vdr*(lane % (qi/vdr));

        if constexpr (rows_per_block <= tokens_per_wave) {
            // Retain the smaller row side, then stream one activation tuple at
            // a time. This is the R2T4 schedule used by the strongest N=4/8
            // candidates.
            int w0[rows_per_block];
            int w1[rows_per_block];
            float wd[rows_per_block];
#pragma unroll
            for (int row = 0; row < rows_per_block; ++row) {
                const int safe_row = min(row0 + row, int(nrows_x) - 1);
                const block_q8_0 * wb =
                    (const block_q8_0 *) vx +
                    kbx_offset +
                    (safe_row - row0)*stride_row_x +
                    kbx;
                w0[row] = get_int_b2(wb->qs, kqs + 0);
                w1[row] = get_int_b2(wb->qs, kqs + 1);
                wd[row] = __half2float(wb->d);
            }

#pragma unroll
            for (int token = 0; token < tokens_per_wave; ++token) {
                const block_q8_1 * ab =
                    y + (token0 + token)*stride_col_y + kbx;
                const int a0 = get_int_b4(ab->qs, kqs + 0);
                const int a1 = get_int_b4(ab->qs, kqs + 1);
                const float ad = __half2float(__low2half(ab->ds));
#pragma unroll
                for (int row = 0; row < rows_per_block; ++row) {
                    int sumi = ggml_cuda_dp4a(w0[row], a0, 0);
                    sumi = ggml_cuda_dp4a(w1[row], a1, sumi);
                    tmp[token][row] += wd[row]*ad*float(sumi);
                }
            }
        } else {
            // Retain the smaller token side, then stream one weight tuple at a
            // time. This is the R4T2 schedule and the N=1 row-folding schedule.
            int a0[tokens_per_wave];
            int a1[tokens_per_wave];
            float ad[tokens_per_wave];
#pragma unroll
            for (int token = 0; token < tokens_per_wave; ++token) {
                const block_q8_1 * ab =
                    y + (token0 + token)*stride_col_y + kbx;
                a0[token] = get_int_b4(ab->qs, kqs + 0);
                a1[token] = get_int_b4(ab->qs, kqs + 1);
                ad[token] = __half2float(__low2half(ab->ds));
            }

#pragma unroll
            for (int row = 0; row < rows_per_block; ++row) {
                const int safe_row = min(row0 + row, int(nrows_x) - 1);
                const block_q8_0 * wb =
                    (const block_q8_0 *) vx +
                    kbx_offset +
                    (safe_row - row0)*stride_row_x +
                    kbx;
                const int w0 = get_int_b2(wb->qs, kqs + 0);
                const int w1 = get_int_b2(wb->qs, kqs + 1);
                const float wd = __half2float(wb->d);
#pragma unroll
                for (int token = 0; token < tokens_per_wave; ++token) {
                    int sumi = ggml_cuda_dp4a(w0, a0[token], 0);
                    sumi = ggml_cuda_dp4a(w1, a1[token], sumi);
                    tmp[token][row] += wd*ad[token]*float(sumi);
                }
            }
        }
    }

#pragma unroll
    for (int token = 0; token < tokens_per_wave; ++token) {
#pragma unroll
        for (int row = 0; row < rows_per_block; ++row) {
            tmp[token][row] = warp_reduce_sum<warp_size>(tmp[token][row]);
        }
        if (lane < rows_per_block && uint32_t(row0 + lane) < nrows_x) {
            dst[
                sample_dst*stride_sample_dst +
                channel_dst*stride_channel_dst +
                (token0 + token)*stride_col_dst +
                row0 +
                lane] = tmp[token][lane];
        }
    }
}

// gfx1151 standard-Q8 N=1 K-split row-pair experiment.
//
// Two waves compute two output rows while splitting K between them. This keeps
// the same total wave count as stock MMVQ (M waves), but each Q8_1 activation
// block is consumed once for the row pair instead of once per row. Each wave
// first reduces its two row partials to scalars; only four floats cross LDS.
// The barrier cost is intended to be amortized by long-K decode matrices.
template <bool scalar_exchange>
__launch_bounds__(2*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q8_0_rdna35_k2r2(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float * __restrict__ dst,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_col_dst,
        const uint3 channel_ratio,
        const uint32_t stride_channel_x,
        const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst,
        const uint3 sample_ratio,
        const uint32_t stride_sample_x,
        const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int qk = QK8_0;
    constexpr int qi = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int rows_per_block = 2;
    constexpr int nwarps = 2;
    constexpr int blocks_per_wave_iter = vdr*warp_size/qi;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int row0 = rows_per_block*blockIdx.x;
    const int blocks_per_row_x = ncols_x/qk;
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst = blockIdx.z;
    const uint32_t channel_x = fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y = channel_dst;
    const uint32_t sample_x = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y = sample_dst;

    const block_q8_1 * y =
        (const block_q8_1 *) vy +
        sample_y*stride_sample_y +
        channel_y*stride_channel_y;
    const int kbx_offset =
        sample_x*stride_sample_x +
        channel_x*stride_channel_x +
        row0*stride_row_x;

    float tmp[rows_per_block] = {0.0f};

    for (int kbx =
            warp*blocks_per_wave_iter + lane/(qi/vdr);
            kbx < blocks_per_row_x;
            kbx += nwarps*blocks_per_wave_iter) {
        const int kqs = vdr*(lane % (qi/vdr));
        const block_q8_1 * ab = y + kbx;
        const int a0 = get_int_b4(ab->qs, kqs + 0);
        const int a1 = get_int_b4(ab->qs, kqs + 1);
        const float ad = __half2float(__low2half(ab->ds));

#pragma unroll
        for (int row = 0; row < rows_per_block; ++row) {
            const int safe_row = min(row0 + row, int(nrows_x) - 1);
            const block_q8_0 * wb =
                (const block_q8_0 *) vx +
                kbx_offset +
                (safe_row - row0)*stride_row_x +
                kbx;
            const int w0 = get_int_b2(wb->qs, kqs + 0);
            const int w1 = get_int_b2(wb->qs, kqs + 1);
            int sumi = ggml_cuda_dp4a(w0, a0, 0);
            sumi = ggml_cuda_dp4a(w1, a1, sumi);
            tmp[row] += __half2float(wb->d)*ad*float(sumi);
        }
    }

    if constexpr (scalar_exchange) {
#pragma unroll
        for (int row = 0; row < rows_per_block; ++row) {
            tmp[row] = warp_reduce_sum<warp_size>(tmp[row]);
        }

        __shared__ float partial[nwarps][rows_per_block];
        if (lane == 0) {
#pragma unroll
            for (int row = 0; row < rows_per_block; ++row) {
                partial[warp][row] = tmp[row];
            }
        }

        __syncthreads();

        if (warp == 0 && lane < rows_per_block &&
                uint32_t(row0 + lane) < nrows_x) {
            dst[
                sample_dst*stride_sample_dst +
                channel_dst*stride_channel_dst +
                row0 +
                lane] =
                partial[0][lane] + partial[1][lane];
        }
    } else {
        // Warp 1 exports its unreduced lane partials. Warp 0 combines them
        // with its registers and performs the same two reductions as stock.
        __shared__ float partial[rows_per_block][warp_size];
        if (warp == 1) {
#pragma unroll
            for (int row = 0; row < rows_per_block; ++row) {
                partial[row][lane] = tmp[row];
            }
        }

        __syncthreads();

        if (warp == 0) {
#pragma unroll
            for (int row = 0; row < rows_per_block; ++row) {
                tmp[row] += partial[row][lane];
                tmp[row] = warp_reduce_sum<warp_size>(tmp[row]);
            }

            if (lane < rows_per_block &&
                    uint32_t(row0 + lane) < nrows_x) {
                dst[
                    sample_dst*stride_sample_dst +
                    channel_dst*stride_channel_dst +
                    row0 +
                    lane] = tmp[lane];
            }
        }
    }

    GGML_UNUSED(stride_col_y);
    GGML_UNUSED(stride_col_dst);
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    const uint32_t channel_x = ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
        }
    }

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id);
    const int rpb = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

static int gfx1151_q8_0_2d_mode(const int cc) {
#if defined(GGML_USE_HIP)
    static const char * value = std::getenv("GGML_ROCM_GFX1151_Q8_0_2D");
    static const int mode =
        value == nullptr ? 0 :
        (std::strcmp(value, "1") == 0 || std::strcmp(value, "T2R2") == 0) ? 1 :
        (std::strcmp(value, "2") == 0 || std::strcmp(value, "T4R2") == 0) ? 2 :
        (std::strcmp(value, "3") == 0 || std::strcmp(value, "T2R4") == 0) ? 3 :
        (std::strcmp(value, "4") == 0 || std::strcmp(value, "T1R2") == 0) ? 4 :
        (std::strcmp(value, "5") == 0 || std::strcmp(value, "T1R4") == 0) ? 5 :
        (std::strcmp(value, "6") == 0 || std::strcmp(value, "T1R8") == 0) ? 6 :
        (std::strcmp(value, "7") == 0 || std::strcmp(value, "auto") == 0) ? 7 :
        (std::strcmp(value, "8") == 0 || std::strcmp(value, "K2R2") == 0) ? 8 :
        (std::strcmp(value, "9") == 0 || std::strcmp(value, "K2R2_LANE") == 0) ? 9 :
        0;
    return cc == GGML_CUDA_CC_OFFSET_AMD + 0x1151 ? mode : 0;
#else
    GGML_UNUSED(cc);
    return 0;
#endif
}

struct q8_0_2d_launch_args {
    const void * vx;
    const void * vy;
    float * dst;
    int ncols_x;
    int nrows_x;
    int nchannels_dst;
    int nsamples_dst;
    int stride_row_x;
    int stride_col_y;
    int stride_col_dst;
    uint3 channel_ratio;
    int stride_channel_x;
    int stride_channel_y;
    int stride_channel_dst;
    uint3 sample_ratio;
    int stride_sample_x;
    int stride_sample_y;
    int stride_sample_dst;
    int warp_size;
};

template <bool scalar_exchange>
static void launch_gfx1151_q8_0_k2r2(
        const q8_0_2d_launch_args & args,
        cudaStream_t stream) {
    const dim3 block_nums(
        (args.nrows_x + 1)/2,
        args.nchannels_dst,
        args.nsamples_dst);
    const dim3 block_dims(args.warp_size, 2, 1);
    mul_mat_vec_q8_0_rdna35_k2r2<scalar_exchange><<<block_nums, block_dims, 0, stream>>>(
        args.vx, args.vy, args.dst,
        args.ncols_x, args.nrows_x,
        args.stride_row_x, args.stride_col_y, args.stride_col_dst,
        args.channel_ratio,
        args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
        args.sample_ratio,
        args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst);
}

template <int ncols_dst, int tokens_per_wave, int rows_per_block>
static void launch_gfx1151_q8_0_2d(
        const q8_0_2d_launch_args & args,
        cudaStream_t stream) {
    const dim3 block_nums(
        (args.nrows_x + rows_per_block - 1)/rows_per_block,
        args.nchannels_dst,
        args.nsamples_dst);
    const dim3 block_dims(args.warp_size, ncols_dst/tokens_per_wave, 1);
    mul_mat_vec_q8_0_rdna35_2d<ncols_dst, tokens_per_wave, rows_per_block>
        <<<block_nums, block_dims, 0, stream>>>(
            args.vx, args.vy, args.dst,
            args.ncols_x, args.nrows_x,
            args.stride_row_x, args.stride_col_y, args.stride_col_dst,
            args.channel_ratio,
            args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
            args.sample_ratio,
            args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst);
}

template<ggml_type type, int c_ncols_dst, bool small_k = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            mul_mat_vec_q<type, c_ncols_dst, true, small_k><<<block_nums, block_dims, nbytes_shared, stream>>>
                (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    mul_mat_vec_q<type, c_ncols_dst, false, small_k><<<block_nums, block_dims, nbytes_shared, stream>>>
        (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type, int rows_per_block>
static void mul_mat_vec_q_moe_launch_rpb(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst);
    const dim3 block_dims(warp_size, ncols_dst);

    mul_mat_vec_q_moe<type, rows_per_block><<<block_nums, block_dims, 0, stream>>>(
        vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        ncols_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

#if defined(GGML_USE_HIP) && GGML_ROCMFPX_HY3_ADAPTIVE_MOE_RPB
    // HY3's 192-expert matrices favor different launch shapes for decode and
    // multi-token verification. Keep this opt-in so established FPX recipes
    // retain the generic launch policy.
    if constexpr (type == GGML_TYPE_Q2_0_ROCMFPX) {
        if (ncols_dst == 1) {
            if (nrows_x <= 2048) {
                mul_mat_vec_q_moe_launch_rpb<type, 3>(vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x,
                    stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y,
                    stride_channel_dst, ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
            } else {
                mul_mat_vec_q_moe_launch_rpb<type, 1>(vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x,
                    stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y,
                    stride_channel_dst, ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
            }
        } else {
            mul_mat_vec_q_moe_launch_rpb<type, 3>(vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        }
        return;
    }

#endif

    constexpr int rows_per_block = calc_moe_mmvq_rows_per_block<type>();
    mul_mat_vec_q_moe_launch_rpb<type, rows_per_block>(vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y,
        stride_channel_dst, ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    const bool has_ids = ids != nullptr;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
        constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
        constexpr int vdr                   = get_vdr_mmvq(type);
        const int     blocks_per_row_x      = ncols_x / qk;
        const int     blocks_per_iter_1warp = vdr * warp_size / qi;
        const int     nwarps                = calc_nwarps(type, c_ncols_dst, table_id);
        bool          use                   = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    if constexpr (type == GGML_TYPE_Q8_0) {
        if (!has_ids && !has_fusion) {
            const int mode = gfx1151_q8_0_2d_mode(cc);
            int selected_mode = mode;

            if (mode == 7) {
                // Evidence-gated gfx1151 lookup. These four M x K cells were
                // measured independently against stock MMVQ; all other shapes
                // deliberately retain the upstream kernel until characterized.
                const bool measured_m =
                    nrows_x == 4096 || nrows_x == 8192;
                const bool measured_k =
                    ncols_x == 2048 || ncols_x == 4096;

                selected_mode = 0;
                if (measured_m && measured_k) {
                    if (ncols_dst == 1) {
                        selected_mode = 4; // T1R2
                    } else if (ncols_dst == 2) {
                        selected_mode = 1; // T2R2
                    } else if (ncols_dst == 4) {
                        selected_mode =
                            nrows_x == 8192 ? 3 : 2; // T2R4 or T4R2
                    } else if (ncols_dst == 8 &&
                            !(nrows_x == 8192 && ncols_x == 4096)) {
                        selected_mode = 3; // T2R4
                    }
                }
            }

            const q8_0_2d_launch_args args = {
                vx,
                vy,
                dst,
                ncols_x,
                nrows_x,
                nchannels_dst,
                nsamples_dst,
                stride_row_x,
                stride_col_y,
                stride_col_dst,
                channel_ratio_fd,
                stride_channel_x,
                stride_channel_y,
                stride_channel_dst,
                sample_ratio_fd,
                stride_sample_x,
                stride_sample_y,
                stride_sample_dst,
                warp_size,
            };

            if (ncols_dst == 1 && selected_mode == 4) {
                launch_gfx1151_q8_0_2d<1, 1, 2>(args, stream);
                return;
            }
            if (ncols_dst == 1 && selected_mode == 5) {
                launch_gfx1151_q8_0_2d<1, 1, 4>(args, stream);
                return;
            }
            if (ncols_dst == 1 && selected_mode == 6) {
                launch_gfx1151_q8_0_2d<1, 1, 8>(args, stream);
                return;
            }
            if (ncols_dst == 1 && selected_mode == 8) {
                launch_gfx1151_q8_0_k2r2<true>(args, stream);
                return;
            }
            if (ncols_dst == 1 && selected_mode == 9) {
                launch_gfx1151_q8_0_k2r2<false>(args, stream);
                return;
            }
            if (ncols_dst == 2 && selected_mode == 1) {
                launch_gfx1151_q8_0_2d<2, 2, 2>(args, stream);
                return;
            }
            if (ncols_dst == 2 && selected_mode == 3) {
                launch_gfx1151_q8_0_2d<2, 2, 4>(args, stream);
                return;
            }
            if (ncols_dst == 4 && selected_mode == 1) {
                launch_gfx1151_q8_0_2d<4, 2, 2>(args, stream);
                return;
            }
            if (ncols_dst == 4 && selected_mode == 2) {
                launch_gfx1151_q8_0_2d<4, 4, 2>(args, stream);
                return;
            }
            if (ncols_dst == 4 && selected_mode == 3) {
                launch_gfx1151_q8_0_2d<4, 2, 4>(args, stream);
                return;
            }
            if (ncols_dst == 8 && selected_mode == 2) {
                launch_gfx1151_q8_0_2d<8, 4, 2>(args, stream);
                return;
            }
            if (ncols_dst == 8 && selected_mode == 3) {
                launch_gfx1151_q8_0_2d<8, 2, 4>(args, stream);
                return;
            }
        }
    }

    switch (ncols_dst) {
        case 1: {
            constexpr int c_ncols_dst = 1;

            bool use_small_k = should_use_small_k(c_ncols_dst);

            if (use_small_k) {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id, true);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, true>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            } else {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }

    GGML_UNUSED(has_fusion);
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0_ROCMFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0_ROCMFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0_ROCMFP4_FAST>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_0_ROCMFPX:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_0_ROCMFPX>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0_ROCMFPX:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0_ROCMFPX>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_0_ROCMFPX:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_0_ROCMFPX>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0_ROCMFPX:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0_ROCMFPX>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q7_0_ROCMFPX:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q7_0_ROCMFPX>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        fusion_local.glu_op = fusion->glu_op;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
