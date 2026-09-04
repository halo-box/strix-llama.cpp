#include "mmvq.cuh"
#include "ggml-backend-impl.h"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <type_traits>

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
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
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
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
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_5,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
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
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
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
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
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
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 5;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
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

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_RDNA3_5) {
        return 1;
    }
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
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
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
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

static_assert(calc_nwarps(GGML_TYPE_Q5_1, 1, MMVQ_PARAMETERS_RDNA3_5) == 1);
static_assert(calc_nwarps(GGML_TYPE_Q8_0, 1, MMVQ_PARAMETERS_RDNA3_5) == 1);
static_assert(calc_nwarps(GGML_TYPE_MXFP4, 1, MMVQ_PARAMETERS_RDNA3_5) == 1);
static_assert(calc_nwarps(GGML_TYPE_Q4_K, 1, MMVQ_PARAMETERS_RDNA3_5) == 1);
static_assert(calc_nwarps(GGML_TYPE_Q5_K, 1, MMVQ_PARAMETERS_RDNA3_5) == 1);
static_assert(calc_nwarps(GGML_TYPE_Q6_K, 1, MMVQ_PARAMETERS_RDNA3_5) == 1);
static_assert(calc_nwarps(GGML_TYPE_Q1_0, 1, MMVQ_PARAMETERS_RDNA3_5) == 1);
static_assert(calc_nwarps(GGML_TYPE_Q2_0, 1, MMVQ_PARAMETERS_RDNA3_5) == 1);
static_assert(calc_nwarps(GGML_TYPE_Q4_0, 1, MMVQ_PARAMETERS_RDNA3_5) == 1);
static_assert(calc_nwarps(GGML_TYPE_IQ3_S, 1, MMVQ_PARAMETERS_RDNA3_5) == 1);

static constexpr __host__ __device__ bool is_rdna3_5_q4_columns_type(ggml_type type) {
    return type == GGML_TYPE_Q1_0 || type == GGML_TYPE_Q2_0 || type == GGML_TYPE_Q5_1 ||
           type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K ||
           type == GGML_TYPE_IQ3_S;
}

static_assert(is_rdna3_5_q4_columns_type(GGML_TYPE_Q1_0));
static_assert(is_rdna3_5_q4_columns_type(GGML_TYPE_Q2_0));
static_assert(is_rdna3_5_q4_columns_type(GGML_TYPE_Q5_1));
static_assert(is_rdna3_5_q4_columns_type(GGML_TYPE_Q4_K));
static_assert(is_rdna3_5_q4_columns_type(GGML_TYPE_Q5_K));
static_assert(is_rdna3_5_q4_columns_type(GGML_TYPE_Q6_K));
static_assert(is_rdna3_5_q4_columns_type(GGML_TYPE_IQ3_S));

template <ggml_type type>
__launch_bounds__(2 * ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q4_columns(
        const void * vx_ptr, const void * vy_ptr, float * dst_ptr,
        const uint32_t ncols_x, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {
    static_assert(type == GGML_TYPE_Q8_0);
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const int column0 = 0;
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst = blockIdx.z;
    const uint32_t channel_x = fastdiv(channel_dst, channel_ratio);
    const uint32_t sample_x = fastdiv(sample_dst, sample_ratio);

    const void * vx = vx_ptr;
    const block_q8_1 * y = (const block_q8_1 *) vy_ptr + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    float * dst = dst_ptr + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_iter = vdr * warp_size / qi;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row*stride_row_x;
    float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    ggml_cuda_pdl_sync();
    for (int kbx = lane / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (lane % (qi/vdr));
        const block_q8_0 * bx = (const block_q8_0 *) vx + kbx_offset + kbx;
        int xv[VDR_Q8_0_Q8_1_MMVQ];
#pragma unroll
        for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
            xv[i] = get_int_b2(bx->qs, kqs + i);
        }
        const float dx = __half2float(bx->d);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const block_q8_1 * by = &y[(column0 + j)*stride_col_y + kby];
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
                sumi = ggml_cuda_dp4a(xv[i], get_int_b4(by->qs, kqs + i), sumi);
            }
            tmp[j] += dx * __half2float(__low2half(by->ds)) * (float) sumi;
        }
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        tmp[j] = warp_reduce_sum<warp_size>(tmp[j]);
        if (lane == 0) {
            dst[(column0 + j)*stride_col_dst] = tmp[j];
        }
    }
}

template <ggml_type type>
__launch_bounds__(calc_nwarps(type, 1, get_device_table_id()) * ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q4_columns_rdna3_5(
        const void * vx_ptr, const void * vy_ptr, float * dst_ptr,
        const uint32_t ncols_x, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {
    static_assert(is_rdna3_5_q4_columns_type(type));
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int qi        = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr       = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = calc_nwarps(type, 1, get_device_table_id());

    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const int tid  = warp_size*threadIdx.y + lane;
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);

    const block_q8_1 * y = (const block_q8_1 *) vy_ptr + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    float * dst = dst_ptr + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_iter  = vdr * nwarps*warp_size / qi;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row*stride_row_x;
    float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    ggml_cuda_pdl_sync();
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));

        if constexpr (type == GGML_TYPE_Q1_0) {
            const block_q1_0 * bx = (const block_q1_0 *) vx_ptr + kbx_offset + kbx;
            const int16_t * qs = (const int16_t *) bx->qs + kqs*2;
            int xv[8];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int q = qs[i];
                const int n0 = __byte_perm(0x11100100, 0x11100100, q >> 0);
                const int n1 = __byte_perm(0x11100100, 0x11100100, q >> 2);
                const int s0 = __byte_perm(0x01FF, 0x01FF, n0 >>  0);
                const int s1 = __byte_perm(0x01FF, 0x01FF, n1 >>  0);
                const int s2 = __byte_perm(0x01FF, 0x01FF, n0 >> 16);
                const int s3 = __byte_perm(0x01FF, 0x01FF, n1 >> 16);
                xv[4*i+0] = __byte_perm(s0, s1, 0x5410);
                xv[4*i+1] = __byte_perm(s0, s1, 0x7632);
                xv[4*i+2] = __byte_perm(s2, s3, 0x5410);
                xv[4*i+3] = __byte_perm(s2, s3, 0x7632);
            }
            const float dx = bx->d;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const block_q8_1 * by = &y[j*stride_col_y + kby + kqs];
                int sumi = 0;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    sumi = ggml_cuda_dp4a(xv[4*i+0], get_int_b4(by->qs, i*4+0), sumi);
                    sumi = ggml_cuda_dp4a(xv[4*i+1], get_int_b4(by->qs, i*4+1), sumi);
                    sumi = ggml_cuda_dp4a(xv[4*i+2], get_int_b4(by->qs, i*4+2), sumi);
                    sumi = ggml_cuda_dp4a(xv[4*i+3], get_int_b4(by->qs, i*4+3), sumi);
                }
                const float d8 = __low2float(by->ds);
                tmp[j] += dx * d8 * sumi;
            }
        } else if constexpr (type == GGML_TYPE_Q2_0) {
            const block_q2_0 * bx = (const block_q2_0 *) vx_ptr + kbx_offset + kbx;
            const int16_t * qs = (const int16_t *) bx->qs + kqs*4;
            int xv[4];
            int xw[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int q = qs[i];
#if defined(GGML_USE_HIP)
                const uint32_t qx_indices = (q & 0x03) | ((q & 0x0C) << 6) | ((q & 0x30) << 12) | ((q & 0xC0) << 18);
                const uint32_t qy_bits    = q >> 8;
                const uint32_t qy_indices = (qy_bits & 0x03) | ((qy_bits & 0x0C) << 6) | ((qy_bits & 0x30) << 12) | ((qy_bits & 0xC0) << 18);
                xv[i] = __builtin_amdgcn_perm(0x020100FF, 0x020100FF, qx_indices);
                xw[i] = __builtin_amdgcn_perm(0x020100FF, 0x020100FF, qy_indices);
#else
                const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
                const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
                xv[i] = __byte_perm(qe, qo, 0x5140);
                xw[i] = __byte_perm(qe, qo, 0x7362);
#endif // defined(GGML_USE_HIP)
            }
            const float dx = bx->d;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const block_q8_1 * by = &y[j*stride_col_y + kby + kqs];
                int sumi = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sumi = ggml_cuda_dp4a(get_int_b4(by->qs, i*2+0), xv[i], sumi);
                    sumi = ggml_cuda_dp4a(get_int_b4(by->qs, i*2+1), xw[i], sumi);
                }
                const float d8 = __low2float(by->ds);
                tmp[j] += dx * d8 * sumi;
            }
        } else if constexpr (type == GGML_TYPE_IQ3_S) {
            const block_iq3_s * bx = (const block_iq3_s *) vx_ptr + kbx_offset + kbx;
            const int2 qs_packed = make_int2(get_int_b2(bx->qs, kqs + 0), get_int_b2(bx->qs, kqs + 1));
            const uint8_t * qs = (const uint8_t *) &qs_packed;
            const int qh = bx->qh[kqs/2];
            const int signs_packed_32 = get_int_b2(bx->signs, kqs/2);
            const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;
            int2 xv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int l0 = 2*i;
                const int2 grid_pos = make_int2(
                    iq3s_grid[qs[l0 + 0] | ((qh << (8-l0)) & 0x100)],
                    iq3s_grid[qs[l0 + 1] | ((qh << (7-l0)) & 0x100)]);
                const int signs0 = __vcmpne4(((signs_packed_8[i] & 0x03) << 7) | ((signs_packed_8[i] & 0x0C) << 21), 0x00000000);
                const int signs1 = __vcmpne4(((signs_packed_8[i] & 0x30) << 3) | ((signs_packed_8[i] & 0xC0) << 17), 0x00000000);
                xv[i] = make_int2(__vsub4(grid_pos.x ^ signs0, signs0), __vsub4(grid_pos.y ^ signs1, signs1));
            }
            const int ls = 1 + 2*((bx->scales[kqs/4] >> ((kqs << 1) & 0x04)) & 0x0F);
            const float dx = __half2float(bx->d);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const block_q8_1 * by = &y[j*stride_col_y + kby + kqs/2];
                int sumi = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sumi = ggml_cuda_dp4a(xv[i].x, get_int_b4(by->qs, 2*i + 0), sumi);
                    sumi = ggml_cuda_dp4a(xv[i].y, get_int_b4(by->qs, 2*i + 1), sumi);
                }
                sumi *= ls;
                const float d = dx * __low2float(by->ds);
                tmp[j] += d * sumi;
            }
        } else if constexpr (type == GGML_TYPE_Q5_1) {
            const block_q5_1 * bx = (const block_q5_1 *) vx_ptr + kbx_offset + kbx;
            int vl[VDR_Q5_1_Q8_1_MMVQ];
            int vh[VDR_Q5_1_Q8_1_MMVQ];
#pragma unroll
            for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
                vl[i] = get_int_b4(bx->qs, kqs + i);
                vh[i] = get_int_b4(bx->qh, 0) >> (4 * (kqs + i));
            }
            const half2 dm = bx->dm;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const block_q8_1 * by = &y[j*stride_col_y + kby];
                int u[2*VDR_Q5_1_Q8_1_MMVQ];
#pragma unroll
                for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
                    u[2*i+0] = get_int_b4(by->qs, kqs + i);
                    u[2*i+1] = get_int_b4(by->qs, kqs + i + QI5_1);
                }
                // ROCm 7.14 changes gfx1151 Q5_1 rounding under four-column register pressure. Keep both paths materialized until LLVM preserves this expression.
                volatile float dot = vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, dm, by->ds);
                tmp[j] = __fadd_rn(tmp[j], dot);
            }
        } else if constexpr (type == GGML_TYPE_Q4_K) {
            const block_q4_K * bx = (const block_q4_K *) vx_ptr + kbx_offset + kbx;
            const int bq8_offset = QR4_K * ((kqs/2) / (QI8_1/2));
            const int * q4 = (const int *) (bx->qs + 16*bq8_offset + 4*((kqs/2)%4));
            int xv[2] = {q4[0], q4[4]};
            const uint16_t * scales = (const uint16_t *) bx->scales;
            uint16_t aux[2];
            const int is = bq8_offset/2;
            if (is < 2) {
                aux[0] = scales[is+0] & 0x3f3f;
                aux[1] = scales[is+2] & 0x3f3f;
            } else {
                aux[0] = ((scales[is+2] >> 0) & 0x0f0f) | ((scales[is-2] & 0xc0c0) >> 2);
                aux[1] = ((scales[is+2] >> 4) & 0x0f0f) | ((scales[is-0] & 0xc0c0) >> 2);
            }
            const uint8_t * sc = (const uint8_t *) aux;
            const uint8_t * m  = sc + 2;
            const half2 dm = bx->dm;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const block_q8_1 * by = &y[j*stride_col_y + kby];
                int u[2*QR4_K];
                float d8[QR4_K];
#pragma unroll
                for (int i = 0; i < QR4_K; ++i) {
                    const block_q8_1 * byi = by + bq8_offset + i;
                    d8[i] = __low2float(byi->ds);
                    const int * q8 = (const int *) byi->qs + ((kqs/2)%4);
                    u[2*i+0] = q8[0];
                    u[2*i+1] = q8[4];
                }
                tmp[j] += vec_dot_q4_K_q8_1_impl_vmmq(xv, u, sc, m, dm, d8);
            }
        } else if constexpr (type == GGML_TYPE_Q5_K) {
            const block_q5_K * bx = (const block_q5_K *) vx_ptr + kbx_offset + kbx;
            const int bq8_offset = QR5_K * ((kqs/2) / (QI8_1/2));
            const int * ql = (const int *) (bx->qs + 16*bq8_offset + 4*((kqs/2)%4));
            const int * qh = (const int *) (bx->qh + 4*((kqs/2)%4));
            int vl[2] = {ql[0], ql[4]};
            int vh[2] = {qh[0] >> bq8_offset, qh[4] >> bq8_offset};
            const uint16_t * scales = (const uint16_t *) bx->scales;
            uint16_t aux[2];
            const int is = bq8_offset/2;
            if (is < 2) {
                aux[0] = scales[is+0] & 0x3f3f;
                aux[1] = scales[is+2] & 0x3f3f;
            } else {
                aux[0] = ((scales[is+2] >> 0) & 0x0f0f) | ((scales[is-2] & 0xc0c0) >> 2);
                aux[1] = ((scales[is+2] >> 4) & 0x0f0f) | ((scales[is-0] & 0xc0c0) >> 2);
            }
            const uint8_t * sc = (const uint8_t *) aux;
            const uint8_t * m  = sc + 2;
            const half2 dm = bx->dm;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const block_q8_1 * by = &y[j*stride_col_y + kby];
                int u[2*QR5_K];
                float d8[QR5_K];
#pragma unroll
                for (int i = 0; i < QR5_K; ++i) {
                    const block_q8_1 * byi = by + bq8_offset + i;
                    d8[i] = __low2float(byi->ds);
                    const int * q8 = (const int *) byi->qs + ((kqs/2)%4);
                    u[2*i+0] = q8[0];
                    u[2*i+1] = q8[4];
                }
                tmp[j] += vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, dm, d8);
            }
        } else if constexpr (type == GGML_TYPE_Q6_K) {
            const block_q6_K * bx = (const block_q6_K *) vx_ptr + kbx_offset + kbx;
            const int bq8_offset = 2*QR6_K*(kqs/(QI6_K/2)) + (kqs%(QI6_K/2))/(QI6_K/4);
            const int scale_offset = (QI6_K/4)*(kqs/(QI6_K/2)) + (kqs%(QI6_K/2))/(QI6_K/8);
            const int vh_shift = 2*((kqs%(QI6_K/2))/(QI6_K/4));
            const int vl = get_int_b2(bx->ql, kqs);
            const int vh = get_int_b2(bx->qh, (QI6_K/4)*(kqs/(QI6_K/2)) + kqs%(QI6_K/4)) >> vh_shift;
            const int8_t * scales = bx->scales + scale_offset;
            const float dx = bx->d;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const block_q8_1 * by = &y[j*stride_col_y + kby];
                int u[QR6_K];
                float d8[QR6_K];
#pragma unroll
                for (int i = 0; i < QR6_K; ++i) {
                    u[i]  = get_int_b4(by[bq8_offset + 2*i].qs, kqs % QI8_1);
                    d8[i] = __low2float(by[bq8_offset + 2*i].ds);
                }
                tmp[j] += vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, dx, d8);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][4][warp_size];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            tmp_shared[threadIdx.y-1][j][lane] = tmp[j];
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int i = 0; i < nwarps-1; ++i) {
            tmp[j] += tmp_shared[i][j][lane];
        }
        tmp[j] = warp_reduce_sum<warp_size>(tmp[j]);
        if (lane == 0) {
            dst[j*stride_col_dst] = tmp[j];
        }
    }
}

#ifndef MMVQ_IQ3_S_ROWS_PER_BLOCK
#define MMVQ_IQ3_S_ROWS_PER_BLOCK 1
#endif

// gfx1151 IQ3_S decode kernel. Each warp owns an independent output row, as in
// the baseline one-warp MMVQ kernel. Grouping adjacent rows into a block keeps
// their shared activation vector and 2 KiB IQ3 codebook hot without introducing
// a cross-warp reduction or changing the dot-product order.
template <bool has_gate, int rows_per_block>
__launch_bounds__(rows_per_block * ggml_cuda_get_physical_warp_size(), 8)
static __global__ void mul_mat_vec_iq3_s_rows_rdna3_5(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids,
        const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint32_t stride_sample_x,
        const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {
    constexpr int qk        = QK_K;
    constexpr int qi        = ggml_cuda_type_traits<GGML_TYPE_IQ3_S>::qi;
    constexpr int vdr       = VDR_IQ3_S_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int lane = threadIdx.x;
    const int row  = rows_per_block * blockIdx.x + threadIdx.y;
    if (row >= nrows_x) {
        return;
    }

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = ids[channel_dst];
    const uint32_t channel_y   = fastmodulo(channel_dst, nchannels_y);
    const uint32_t sample_x    = sample_dst;
    const uint32_t sample_y    = sample_dst;

    const block_q8_1 * y = (const block_q8_1 *) vy_ptr
        + sample_y * stride_sample_y + channel_y * stride_channel_y;
    const int kbx_offset = sample_x * stride_sample_x
        + channel_x * stride_channel_x + row * stride_row_x;
    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    float sum      = 0.0f;
    float sum_gate = 0.0f;
    for (int kbx = lane / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (lane % (qi/vdr));
        sum += vec_dot_iq3_s_q8_1(vx_ptr, &y[kby], kbx_offset + kbx, kqs);
        if constexpr (has_gate) {
            sum_gate += vec_dot_iq3_s_q8_1(fusion.gate, &y[kby], kbx_offset + kbx, kqs);
        }
    }

    sum = warp_reduce_sum<warp_size>(sum);
    if constexpr (has_gate) {
        sum_gate = warp_reduce_sum<warp_size>(sum_gate);
    }

    if (lane == 0) {
        float result = sum;
        if constexpr (has_gate) {
            result *= ggml_cuda_op_silu_single(sum_gate);
        }
        dst_ptr[sample_dst * stride_sample_dst
            + channel_dst * stride_channel_dst + row] = result;
    }
    GGML_UNUSED(stride_col_dst);
}

// vec_dot_iq3_s_q8_1 with the 512-entry codebook read from a caller-provided (shared memory) copy instead of
// the __device__ table: the eight per-lane gathers per block then go to the LDS pipe instead of the
// texture-address unit, which the direct kernel saturates (MemUnitBusy ~91% on gfx1151).
static __device__ __forceinline__ float vec_dot_iq3_s_q8_1_grid(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs,
    const uint32_t * __restrict__ grid) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq3->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// Direct kernel (same lane layout and accumulation order as mul_mat_vec_iq3_s_rows_rdna3_5) with the codebook
// staged in shared memory once per block.
template <bool has_gate, int rows_per_block>
__launch_bounds__(rows_per_block * ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_iq3_s_grid_rdna3_5(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids,
        const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint32_t stride_sample_x,
        const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {
    constexpr int qk        = QK_K;
    constexpr int qi        = ggml_cuda_type_traits<GGML_TYPE_IQ3_S>::qi;
    constexpr int vdr       = VDR_IQ3_S_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    __shared__ uint32_t s_grid[512];

    const int lane = threadIdx.x;
    const int tid  = threadIdx.y * warp_size + lane;
    for (int i = tid; i < 512; i += rows_per_block * warp_size) {
        s_grid[i] = iq3s_grid[i];
    }

    const int row = rows_per_block * blockIdx.x + threadIdx.y;
    const bool row_ok = row < nrows_x;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = ids[channel_dst];
    const uint32_t channel_y   = fastmodulo(channel_dst, nchannels_y);
    const uint32_t sample_x    = sample_dst;
    const uint32_t sample_y    = sample_dst;

    const block_q8_1 * y = (const block_q8_1 *) vy_ptr
        + sample_y * stride_sample_y + channel_y * stride_channel_y;
    const int kbx_offset = sample_x * stride_sample_x
        + channel_x * stride_channel_x + (row_ok ? row : 0) * stride_row_x;
    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    __syncthreads();
    if (!row_ok) {
        return;
    }

    float sum      = 0.0f;
    float sum_gate = 0.0f;
    for (int kbx = lane / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (lane % (qi/vdr));
        sum += vec_dot_iq3_s_q8_1_grid(vx_ptr, &y[kby], kbx_offset + kbx, kqs, s_grid);
        if constexpr (has_gate) {
            sum_gate += vec_dot_iq3_s_q8_1_grid(fusion.gate, &y[kby], kbx_offset + kbx, kqs, s_grid);
        }
    }

    sum = warp_reduce_sum<warp_size>(sum);
    if constexpr (has_gate) {
        sum_gate = warp_reduce_sum<warp_size>(sum_gate);
    }

    if (lane == 0) {
        float result = sum;
        if constexpr (has_gate) {
            result *= ggml_cuda_op_silu_single(sum_gate);
        }
        dst_ptr[sample_dst * stride_sample_dst
            + channel_dst * stride_channel_dst + row] = result;
    }
    GGML_UNUSED(stride_col_dst);
}

// gfx1151 IQ3_S decode kernel with LDS staging, ncols == 2560 (10 blocks, 1100 B per row). The direct
// per-lane field loads of vec_dot_iq3_s_q8_1 (about 18 vector-memory instructions per block per lane, most of
// them for the Q8_1 activations) saturate the texture-address unit (MemUnitBusy ~91%) long before DRAM
// bandwidth. Here each wave first copies its up (and gate) row into shared memory with 9 dword loads per
// lane, the block copies the quantized activations once, and the dot products then read from LDS.
// The per-lane accumulation and reduction order is the same as mul_mat_vec_iq3_s_rows_rdna3_5.
#define MMVQ_IQ3_LDS_NCOLS      2560
#define MMVQ_IQ3_LDS_ROW_DWORDS 275   // 10 blocks * 110 B / 4
#define MMVQ_IQ3_LDS_ROW_SLOT   276
#define MMVQ_IQ3_LDS_Y_DWORDS   720   // 80 Q8_1 blocks * 36 B / 4

template <bool has_gate, int rows_per_block>
__launch_bounds__(rows_per_block * ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_iq3_s_lds_rdna3_5(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids,
        const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint32_t stride_sample_x,
        const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {
    constexpr int qk        = QK_K;
    constexpr int qi        = ggml_cuda_type_traits<GGML_TYPE_IQ3_S>::qi;
    constexpr int vdr       = VDR_IQ3_S_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_row  = MMVQ_IQ3_LDS_NCOLS / qk;
    constexpr int blocks_per_iter = vdr * warp_size / qi;
    constexpr int row_loads       = (MMVQ_IQ3_LDS_ROW_DWORDS + warp_size - 1) / warp_size; // 9

    __shared__ int s_y[MMVQ_IQ3_LDS_Y_DWORDS];
    __shared__ int s_rows[rows_per_block][has_gate ? 2 : 1][MMVQ_IQ3_LDS_ROW_SLOT];

    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int tid  = wave * warp_size + lane;
    const int row  = rows_per_block * blockIdx.x + wave;
    const bool row_ok = row < nrows_x;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = ids[channel_dst];
    const uint32_t channel_y   = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = (const block_q8_1 *) vy_ptr + sample_dst * stride_sample_y + channel_y * stride_channel_y;
    const int kbx_offset = sample_dst * stride_sample_x + channel_x * stride_channel_x + (row_ok ? row : 0) * stride_row_x;

    // 1. weight rows -> registers (dword loads; rows are 4 B aligned: 1100 B rows, even block offsets)
    const int * gx = (const int *) ((const block_iq3_s *) vx_ptr + kbx_offset);
    int rx[row_loads];
    int rg[row_loads];
#pragma unroll
    for (int i = 0; i < row_loads; ++i) {
        const int idx = i * warp_size + lane;
        rx[i] = idx < MMVQ_IQ3_LDS_ROW_DWORDS ? gx[idx] : 0;
    }
    if constexpr (has_gate) {
        const int * gg = (const int *) ((const block_iq3_s *) fusion.gate + kbx_offset);
#pragma unroll
        for (int i = 0; i < row_loads; ++i) {
            const int idx = i * warp_size + lane;
            rg[i] = idx < MMVQ_IQ3_LDS_ROW_DWORDS ? gg[idx] : 0;
        }
    }

    // 2. quantized activations -> LDS (shared by the rows of the block)
    {
        const int * gy = (const int *) y;
        for (int i = tid; i < MMVQ_IQ3_LDS_Y_DWORDS; i += rows_per_block * warp_size) {
            s_y[i] = gy[i];
        }
    }

    // 3. rows -> LDS
#pragma unroll
    for (int i = 0; i < row_loads; ++i) {
        const int idx = i * warp_size + lane;
        if (idx < MMVQ_IQ3_LDS_ROW_DWORDS) {
            s_rows[wave][0][idx] = rx[i];
            if constexpr (has_gate) {
                s_rows[wave][1][idx] = rg[i];
            }
        }
    }
    __syncthreads();

    if (!row_ok) {
        return;
    }

    // 4. dot products from LDS, same lane -> block assignment as the direct kernel
    const block_q8_1 * y_l = (const block_q8_1 *) s_y;
    const void *       x_l = s_rows[wave][0];
    float sum      = 0.0f;
    float sum_gate = 0.0f;
#pragma unroll
    for (int kbx = lane / (qi/vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (lane % (qi/vdr));
        sum += vec_dot_iq3_s_q8_1(x_l, &y_l[kby], kbx, kqs);
        if constexpr (has_gate) {
            sum_gate += vec_dot_iq3_s_q8_1(s_rows[wave][1], &y_l[kby], kbx, kqs);
        }
    }

    sum = warp_reduce_sum<warp_size>(sum);
    if constexpr (has_gate) {
        sum_gate = warp_reduce_sum<warp_size>(sum_gate);
    }

    if (lane == 0) {
        float result = sum;
        if constexpr (has_gate) {
            result *= ggml_cuda_op_silu_single(sum_gate);
        }
        dst_ptr[sample_dst * stride_sample_dst + channel_dst * stride_channel_dst + row] = result;
    }
    GGML_UNUSED(stride_col_dst);
    GGML_UNUSED(ncols_x);
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_RDNA3_5) {
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
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
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
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

    ggml_cuda_pdl_sync();
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
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

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                if constexpr (type == GGML_TYPE_IQ3_S && has_fusion) {
                    if (use_gate) {
                        float dot;
                        float dot_gate;
                        vec_dot_iq3_s_q8_1_pair(
                            dot, dot_gate, vx, vgate, &y[j*stride_col_y + kby],
                            kbx_offset + i*stride_row_x + kbx, kqs);
                        tmp[j][i]      += dot;
                        tmp_gate[j][i] += dot_gate;
                        continue;
                    }
                }
                if constexpr (type == GGML_TYPE_Q5_1 && table_id == MMVQ_PARAMETERS_RDNA3_5) {
                    volatile float dot = vec_dot_q_cuda(
                        vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    tmp[j][i] = __fadd_rn(tmp[j][i], dot);
                } else {
                    tmp[j][i] += vec_dot_q_cuda(
                        vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
                if constexpr (has_fusion) {
                    if (use_gate) {
                        if constexpr (type == GGML_TYPE_Q5_1 && table_id == MMVQ_PARAMETERS_RDNA3_5) {
                            volatile float dot_gate = vec_dot_q_cuda(
                                vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                            tmp_gate[j][i] = __fadd_rn(tmp_gate[j][i], dot_gate);
                        } else {
                            tmp_gate[j][i] += vec_dot_q_cuda(
                                vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                        }
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

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

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
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
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            case GGML_GLU_OP_SWIGLU_CLAMP:
                                result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// ---------------------------------------------------------------------------------------------
// RDNA3.5 single-column MMVQ with the Q8_1 activation quantization fused into the matvec kernel.
//
// Token generation on gfx1151 pays a ~2 us launch gap for every kernel, so the quantize_q8_1 launch in front
// of each matvec costs about as much as the whole quantize kernel. A block of MMVQ_FQ_NWARPS waves (one output
// row per wave) quantizes the activation vector once into shared memory, replicating quantize_q8_1 exactly
// (same amax, same IEEE division and roundf), and then runs the per-wave K loop of mul_mat_vec_q<type, 1> in
// the same order against it, so the outputs are bit-identical to the two-kernel path.
//
// Measured on gfx1151 with Qwen3.6-35B-A3B Q8_0 decode (2026-09-03): 16 waves per block and 2 prefetched
// K iterations; 8 waves or deeper prefetch (more VGPRs, less occupancy) were slower.

#define MMVQ_FQ_NWARPS 16
// whether the fused-quantize path also serves the IQ3_S/IQ4_NL/IQ4_XS expert matvecs (see mmvq_fq_type_ok)
#ifndef MMVQ_FQ_IQ_TYPES
#define MMVQ_FQ_IQ_TYPES false
#endif

static constexpr __host__ __device__ bool mmvq_fq_type_ok(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q6_K:
            // Q4_K/Q5_K (block sum via mmvq_fq_block_sum) are exact too but measured neutral on gfx1151
            // with the generic, non-prefetched loop (Qwen3.6 UD-Q4_K_XL TG128 60.69 -> 60.63 t/s).
            return true;
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            // MoE expert types of the Unsloth qwen4exp IQ4_XS mix: their vec_dot only reads the block scale, so
            // the in-kernel quantization is exact; used to drop the quantize launch in front of every expert matvec
            return MMVQ_FQ_IQ_TYPES;
        default:
            return false;
    }
}

// Whether the vec_dot of the type reads the Q8_1 block sum (ds.y) in addition to the scale.
static constexpr __host__ __device__ bool mmvq_fq_needs_sum(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return false;
        default:
            return true;
    }
}

// Sum of the 32 values of a Q8_1 block in exactly the order of warp_reduce_sum<32> in quantize_q8_1, where lane
// l holds value l. Here lane (sub) of a group of 4 holds values 8*sub .. 8*sub+7 in v[].
static __device__ __forceinline__ float mmvq_fq_block_sum(const float * v, const int width) {
    float a[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        // offset 16: a_i = x_i + x_{i^16}, x_{i^16} lives two lanes over
        a[k] = v[k] + __shfl_xor_sync(0xffffffff, v[k], 2, width);
    }
    float b[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        // offset 8: b_i = a_i + a_{i^8}, one lane over
        b[k] = a[k] + __shfl_xor_sync(0xffffffff, a[k], 1, width);
    }
    float c[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        c[k] = b[k] + b[k + 4]; // offset 4
    }
    const float d0 = c[0] + c[2]; // offset 2
    const float d1 = c[1] + c[3];
    return d0 + d1;               // offset 1
}

// Register-staged Q8_0 weight loads: the K loop of one wave is split into chunks of MMVQ_FQ_PF iterations whose
// weight blocks are loaded up front, so the DRAM stream starts before the activation quantization and the
// barrier, and each wave keeps MMVQ_FQ_PF loads in flight instead of one.
#define MMVQ_FQ_PF 2
// Long rows with few output rows (e.g. the [10240 x 320] hyper-connection down projections of qwen4exp) have
// too few waves in flight to cover DRAM latency with 2 loads per wave; they use a deeper prefetch instead.
// The accumulation order per lane is the same for any depth, so the results do not change.
#define MMVQ_FQ_PF_LONG 8

template <int pf>
struct mmvq_fq_q8_0_chunk {
    int  qs[pf][VDR_Q8_0_Q8_1_MMVQ];
    half d[pf];
};

template <int pf>
static __device__ __forceinline__ void mmvq_fq_q8_0_load(
        mmvq_fq_q8_0_chunk<pf> & c, const block_q8_0 * GGML_CUDA_RESTRICT x, const int kbx0, const int kqs, const int blocks_per_row_x) {
    constexpr int blocks_per_iter = VDR_Q8_0_Q8_1_MMVQ * ggml_cuda_get_physical_warp_size() / QI8_0;
#pragma unroll
    for (int i = 0; i < pf; ++i) {
        const int kbx = kbx0 + i*blocks_per_iter;
        if (kbx < blocks_per_row_x) {
#pragma unroll
            for (int j = 0; j < VDR_Q8_0_Q8_1_MMVQ; ++j) {
                c.qs[i][j] = get_int_b2(x[kbx].qs, kqs + j);
            }
            c.d[i] = x[kbx].d;
        }
    }
}

template <int pf>
static __device__ __forceinline__ void mmvq_fq_q8_0_dot(
        float & tmp, const mmvq_fq_q8_0_chunk<pf> & c, const block_q8_1 * y, const int kbx0, const int kqs, const int blocks_per_row_x) {
    constexpr int blocks_per_iter = VDR_Q8_0_Q8_1_MMVQ * ggml_cuda_get_physical_warp_size() / QI8_0;
#pragma unroll
    for (int i = 0; i < pf; ++i) {
        const int kbx = kbx0 + i*blocks_per_iter;
        if (kbx < blocks_per_row_x) {
            const block_q8_1 * bq8_1 = &y[kbx]; // qk == QK8_1 for Q8_0
            int u[VDR_Q8_0_Q8_1_MMVQ];
#pragma unroll
            for (int j = 0; j < VDR_Q8_0_Q8_1_MMVQ; ++j) {
                u[j] = get_int_b4(bq8_1->qs, kqs + j);
            }
            tmp += vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMVQ>(c.qs[i], u, c.d[i], __low2half(bq8_1->ds));
        }
    }
}

template <ggml_type type, bool has_fusion, int nwarps, int pf>
__launch_bounds__(nwarps * ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_fq(
        const void * vx_ptr, const float * y_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint3 nchannels_y, const uint32_t stride_row_x,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const float y_scale, const float y_bias, const int y_op) {
    static_assert(mmvq_fq_type_ok(type));
    constexpr bool q8_0_path = type == GGML_TYPE_Q8_0;
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_iter = vdr * warp_size / qi;
    static_assert(!q8_0_path || qk == QK8_1);
    [[maybe_unused]] constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    extern __shared__ char mmvq_fq_smem[];
    block_q8_1 * y_q8 = (block_q8_1 *) mmvq_fq_smem;

    const int lane = threadIdx.x;
    const int tid  = warp_size*threadIdx.y + lane;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y   = ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    const uint32_t row      = blockIdx.x*nwarps + threadIdx.y;
    const bool     row_ok   = row < nrows_x;
    const uint32_t row_safe = row_ok ? row : nrows_x - 1;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
    }

    // 1. start streaming the weights of this wave's row before anything else
    const int blocks_per_row_x = ncols_x / qk;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row_safe*stride_row_x;
    const int kqs  = vdr * (lane % (qi/vdr));
    const int kbx0 = lane / (qi/vdr);

    [[maybe_unused]] const block_q8_0 * x  = (const block_q8_0 *) vx + kbx_offset;
    [[maybe_unused]] const block_q8_0 * xg = nullptr;

    [[maybe_unused]] mmvq_fq_q8_0_chunk<pf> cx;
    [[maybe_unused]] mmvq_fq_q8_0_chunk<pf> cg;
    if constexpr (q8_0_path) {
        mmvq_fq_q8_0_load(cx, x, kbx0, kqs, blocks_per_row_x);
        if constexpr (has_fusion) {
            if (use_gate) {
                xg = (const block_q8_0 *) vgate + kbx_offset;
                mmvq_fq_q8_0_load(cg, xg, kbx0, kqs, blocks_per_row_x);
            }
        }
    }

    // 2. quantize the activation vector into shared memory: 4 lanes per Q8_1 block, 8 values per lane
    {
        const float * y = y_ptr + sample_y*stride_sample_y + channel_y*stride_channel_y;
        const int nby = ncols_x / QK8_1;
        for (int i = tid; i < 4*nby; i += nwarps*warp_size) {
            const int ib  = i >> 2;
            const int sub = i & 3;
            const float4 * yv = (const float4 *) (y + ib*QK8_1 + sub*8);
            const float4 v0 = yv[0];
            const float4 v1 = yv[1];
            float v[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
            if (y_op == 3) {
                // qwen4exp GDN output: per-head (128) rms norm * weight, gated by sigmoid(z). The 16 lanes of a
                // head are consecutive threads of one wave (4 Q8_1 blocks x 4 lanes).
                float ss = 0.0f;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    ss = fmaf(v[k], v[k], ss);
                }
                ss += __shfl_xor_sync(0xffffffff, ss, 1, warp_size);
                ss += __shfl_xor_sync(0xffffffff, ss, 2, warp_size);
                ss += __shfl_xor_sync(0xffffffff, ss, 4, warp_size);
                ss += __shfl_xor_sync(0xffffffff, ss, 8, warp_size);
                const float rms_scale = rsqrtf(ss / 128.0f + fusion.y_eps);
                const float * z = (const float *) fusion.y_gate + ib*QK8_1 + sub*8;
                const float * w = (const float *) fusion.y_norm_w + (ib*QK8_1 + sub*8) % 128;
                const float4 * zv = (const float4 *) z;
                const float4 * wv = (const float4 *) w;
                const float4 z0 = zv[0], z1 = zv[1], w0 = wv[0], w1 = wv[1];
                const float zz[8] = {z0.x, z0.y, z0.z, z0.w, z1.x, z1.y, z1.z, z1.w};
                const float ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    // rms_norm_f32<.., true>: scale * x * w ; unary_gated: sigmoid(z) * normed
                    v[k] = (1.0f / (1.0f + expf(-zz[k]))) * (rms_scale * v[k] * ww[k]);
                }
            } else if (y_op != 0) {
                // fused activation prologue (scale -> unary), same expressions as scale_f32 / op_silu / op_sigmoid
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const float t = y_scale * v[k] + y_bias;
                    v[k] = y_op == 1 ? ggml_cuda_op_silu_single(t) : 1.0f / (1.0f + expf(-t));
                }
            }

            float amax = fabsf(v[0]);
#pragma unroll
            for (int k = 1; k < 8; ++k) {
                amax = fmaxf(amax, fabsf(v[k]));
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, warp_size));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, warp_size));

            const float sum = mmvq_fq_needs_sum(type) ? mmvq_fq_block_sum(v, warp_size) : 0.0f;

            const float d = amax / 127.0f;
            int q[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int8_t qk8 = amax == 0.0f ? 0 : roundf(v[k] / d);
                q[k] = (int) qk8 & 0xff;
            }
            int * qs = (int *) y_q8[ib].qs;
            qs[2*sub + 0] = q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
            qs[2*sub + 1] = q[4] | (q[5] << 8) | (q[6] << 16) | (q[7] << 24);
            if (sub == 0) {
                y_q8[ib].ds = make_half2(d, sum);
            }
        }
    }
    __syncthreads();

    if (!row_ok) {
        return;
    }

    [[maybe_unused]] float x_biases    = 0.0f;
    [[maybe_unused]] float gate_biases = 0.0f;
    if constexpr (has_fusion) {
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (lane == 0) {
            if (use_bias) {
                x_biases = x_bias[sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row];
            }
            if (use_gate_bias) {
                gate_biases = gate_bias[sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row];
            }
        }
    }

    // 3. exact per-wave dot products in the k order of mul_mat_vec_q<type, 1>, one chunk ahead in memory
    float tmp      = 0.0f;
    float tmp_gate = 0.0f;

    const block_q8_1 * y = y_q8;
    if constexpr (!q8_0_path) {
        // generic types: the per-wave loop of mul_mat_vec_q<type, 1> against the shared-memory activations
        for (int kbx = kbx0; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            tmp += vec_dot_q_cuda(vx, &y[kby], kbx_offset + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + kbx, kqs);
                }
            }
        }
    }
    constexpr int chunk_blocks = pf*blocks_per_iter;
    for (int c0 = kbx0; q8_0_path && c0 < blocks_per_row_x; c0 += chunk_blocks) {
        mmvq_fq_q8_0_chunk<pf> nx;
        [[maybe_unused]] mmvq_fq_q8_0_chunk<pf> ng;
        const int c1 = c0 + chunk_blocks;
        if (c1 < blocks_per_row_x) {
            mmvq_fq_q8_0_load(nx, x, c1, kqs, blocks_per_row_x);
            if constexpr (has_fusion) {
                if (use_gate) {
                    mmvq_fq_q8_0_load(ng, xg, c1, kqs, blocks_per_row_x);
                }
            }
        }
        mmvq_fq_q8_0_dot(tmp, cx, y, c0, kqs, blocks_per_row_x);
        if constexpr (has_fusion) {
            if (use_gate) {
                mmvq_fq_q8_0_dot(tmp_gate, cg, y, c0, kqs, blocks_per_row_x);
            }
        }
        cx = nx;
        if constexpr (has_fusion) {
            if (use_gate) {
                cg = ng;
            }
        }
    }

    tmp = warp_reduce_sum<warp_size>(tmp);
    if constexpr (has_fusion) {
        if (use_gate) {
            tmp_gate = warp_reduce_sum<warp_size>(tmp_gate);
        }
    }

    if (lane == 0) {
        float result = tmp;
        if constexpr (has_fusion) {
            result += x_biases;
            if (use_gate) {
                float gate_value = tmp_gate;
                gate_value += gate_biases;
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, active_glu, glu_limit, gate_bias, x_bias, tmp_gate);
    }
    GGML_UNUSED(stride_col_dst);
}

template <ggml_type type>
static void mul_mat_vec_q_fq_launch(
        const void * vx, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        cudaStream_t stream, const float y_scale, const float y_bias, const int y_op) {
    const int device    = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst / nsamples_x);

    // One output row per wave. Narrow matrices (e.g. the 320-row hyper-connection down projections of
    // qwen4exp) would give only nrows/16 blocks, leaving most CUs idle while each wave streams a long row.
    // Use fewer waves per block there so at least ~2 blocks per CU are in flight; every row is still reduced
    // by a single wave and the activation quantization is replicated per block, so results are unchanged.
    const int nsm = ggml_cuda_info().devices[device].nsm;
    int nwarps = MMVQ_FQ_NWARPS;
    while (nwarps > 4 && ((nrows_x + nwarps - 1) / nwarps) * nchannels_dst * nsamples_dst < 2 * nsm) {
        nwarps /= 2;
    }

    const dim3 block_nums((nrows_x + nwarps - 1) / nwarps, nchannels_dst, nsamples_dst);
    const dim3 block_dims(warp_size, nwarps, 1);
    const int  nbytes_shared = (ncols_x / QK8_1) * sizeof(block_q8_1);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);

    auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, launch_params,
            vx, y, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_dst,
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, y_scale, y_bias, y_op);
    };

    // deeper weight prefetch for long rows when the grid is too small to hide DRAM latency with many waves
    constexpr int blocks_per_iter = VDR_Q8_0_Q8_1_MMVQ * 32 / QI8_0;
    const bool long_rows = type == GGML_TYPE_Q8_0 && (ncols_x / QK8_0) >= 16 * blocks_per_iter &&
        (int64_t) nrows_x * nchannels_dst * nsamples_dst <= 1024;

    switch (nwarps) {
        case 16:
            if (has_fusion) { launch(mul_mat_vec_q_fq<type, true,  16, MMVQ_FQ_PF>); }
            else            { launch(mul_mat_vec_q_fq<type, false, 16, MMVQ_FQ_PF>); }
            break;
        case 8:
            if (long_rows) {
                if (has_fusion) { launch(mul_mat_vec_q_fq<type, true,  8, MMVQ_FQ_PF_LONG>); }
                else            { launch(mul_mat_vec_q_fq<type, false, 8, MMVQ_FQ_PF_LONG>); }
            } else {
                if (has_fusion) { launch(mul_mat_vec_q_fq<type, true,  8, MMVQ_FQ_PF>); }
                else            { launch(mul_mat_vec_q_fq<type, false, 8, MMVQ_FQ_PF>); }
            }
            break;
        case 4:
            if (long_rows) {
                if (has_fusion) { launch(mul_mat_vec_q_fq<type, true,  4, MMVQ_FQ_PF_LONG>); }
                else            { launch(mul_mat_vec_q_fq<type, false, 4, MMVQ_FQ_PF_LONG>); }
            } else {
                if (has_fusion) { launch(mul_mat_vec_q_fq<type, true,  4, MMVQ_FQ_PF>); }
                else            { launch(mul_mat_vec_q_fq<type, false, 4, MMVQ_FQ_PF>); }
            }
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static void mul_mat_vec_q_fq_switch_type(
        const void * vx, const ggml_type type_x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        cudaStream_t stream, const float y_scale, const float y_bias, const int y_op) {
#define MMVQ_FQ_LAUNCH(T) \
        mul_mat_vec_q_fq_launch<T>(vx, y, ids, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_dst, \
            nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, \
            nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, y_scale, y_bias, y_op)
    switch (type_x) {
        case GGML_TYPE_Q8_0:   MMVQ_FQ_LAUNCH(GGML_TYPE_Q8_0);   break;
        case GGML_TYPE_Q6_K:   MMVQ_FQ_LAUNCH(GGML_TYPE_Q6_K);   break;
#if MMVQ_FQ_IQ_TYPES
        case GGML_TYPE_IQ3_S:  MMVQ_FQ_LAUNCH(GGML_TYPE_IQ3_S);  break;
        case GGML_TYPE_IQ4_NL: MMVQ_FQ_LAUNCH(GGML_TYPE_IQ4_NL); break;
        case GGML_TYPE_IQ4_XS: MMVQ_FQ_LAUNCH(GGML_TYPE_IQ4_XS); break;
#endif
        default:
            GGML_ABORT("fatal error");
            break;
    }
#undef MMVQ_FQ_LAUNCH
}

// ---------------------------------------------------------------------------------------------
// RDNA3.5 grouped single-column matvec: several independent matvecs that share the same activation vector
// (e.g. the qwen4exp GDN qkv / gate / alpha / beta projections, or a hyper-connection down projection with its
// F32 inject rows) are launched as one kernel. Each segment is a plain [ncols x nrows] weight matrix (Q8_0, with
// an optional Q8_0 gate + GLU epilogue, or F32) with its own destination; blocks are assigned to segments by a
// prefix over the per-segment block counts. Q8_0 rows use the fused-quantize path of mul_mat_vec_q_fq (same
// per-lane accumulation order), F32 rows are one wave per row with float4 loads.

#define MMVQ_GROUP_MAX 4
#define MMVQ_GROUP_TYPE_Q8_0 0
#define MMVQ_GROUP_TYPE_F32  1

struct mmvq_group_seg_dev {
    const void * vx;
    const void * gate;
    float      * dst;
    int          nrows;
    int          stride_row;      // in blocks (Q8_0) or floats (F32)
    int          gate_stride_row; // in blocks
    int          blocks_begin;
    int          type;
    int          glu_op;
    float        glu_limit;
};

struct mmvq_group_args_dev {
    mmvq_group_seg_dev seg[MMVQ_GROUP_MAX];
    int nseg;
};

template <int nwarps, int pf>
__launch_bounds__(nwarps * ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_fq_group(
        const mmvq_group_args_dev args, const float * GGML_CUDA_RESTRICT y_ptr, const int ncols_x,
        const float y_scale, const float y_bias, const int y_op) {
    constexpr int qk  = QK8_0;
    constexpr int qi  = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    extern __shared__ char mmvq_fq_smem[];
    block_q8_1 * y_q8 = (block_q8_1 *) mmvq_fq_smem;

    const int lane = threadIdx.x;
    const int tid  = warp_size*threadIdx.y + lane;

    // segment lookup, uniform per block
    int s = 0;
#pragma unroll
    for (int k = 1; k < MMVQ_GROUP_MAX; ++k) {
        if (k < args.nseg && (int) blockIdx.x >= args.seg[k].blocks_begin) {
            s = k;
        }
    }
    const mmvq_group_seg_dev seg = args.seg[s];

    const int  row    = ((int) blockIdx.x - seg.blocks_begin)*nwarps + threadIdx.y;
    const bool row_ok = row < seg.nrows;

    if (seg.type == MMVQ_GROUP_TYPE_F32) {
        if (!row_ok) {
            return;
        }
        const float4 * w = (const float4 *) ((const float *) seg.vx + (size_t) row*seg.stride_row);
        const float4 * y = (const float4 *) y_ptr;
        float acc = 0.0f;
#pragma unroll 4
        for (int k = lane; k < ncols_x/4; k += warp_size) {
            float4 w4 = w[k];
            float4 y4 = y[k];
            if (y_op != 0) {
                float v[4] = {y4.x, y4.y, y4.z, y4.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float t = y_scale * v[j] + y_bias;
                    v[j] = y_op == 1 ? ggml_cuda_op_silu_single(t) : 1.0f / (1.0f + expf(-t));
                }
                y4 = make_float4(v[0], v[1], v[2], v[3]);
            }
            acc = fmaf(w4.x, y4.x, acc);
            acc = fmaf(w4.y, y4.y, acc);
            acc = fmaf(w4.z, y4.z, acc);
            acc = fmaf(w4.w, y4.w, acc);
        }
        acc = warp_reduce_sum<warp_size>(acc);
        if (lane == 0) {
            seg.dst[row] = acc;
        }
        return;
    }

    // Q8_0 path: identical to mul_mat_vec_q_fq<GGML_TYPE_Q8_0, ...>
    const int  row_safe = row_ok ? row : seg.nrows - 1;
    const bool use_gate = seg.gate != nullptr;

    const int blocks_per_row_x = ncols_x / qk;
    const int kqs  = vdr * (lane % (qi/vdr));
    const int kbx0 = lane / (qi/vdr);

    const block_q8_0 * x  = (const block_q8_0 *) seg.vx + (size_t) row_safe*seg.stride_row;
    const block_q8_0 * xg = use_gate ? (const block_q8_0 *) seg.gate + (size_t) row_safe*seg.gate_stride_row : nullptr;

    mmvq_fq_q8_0_chunk<pf> cx;
    mmvq_fq_q8_0_chunk<pf> cg;
    mmvq_fq_q8_0_load(cx, x, kbx0, kqs, blocks_per_row_x);
    if (use_gate) {
        mmvq_fq_q8_0_load(cg, xg, kbx0, kqs, blocks_per_row_x);
    }

    {
        const float * y = y_ptr;
        const int nby = ncols_x / QK8_1;
        for (int i = tid; i < 4*nby; i += nwarps*warp_size) {
            const int ib  = i >> 2;
            const int sub = i & 3;
            const float4 * yv = (const float4 *) (y + ib*QK8_1 + sub*8);
            const float4 v0 = yv[0];
            const float4 v1 = yv[1];
            float v[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
            if (y_op != 0) {
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const float t = y_scale * v[k] + y_bias;
                    v[k] = y_op == 1 ? ggml_cuda_op_silu_single(t) : 1.0f / (1.0f + expf(-t));
                }
            }

            float amax = fabsf(v[0]);
#pragma unroll
            for (int k = 1; k < 8; ++k) {
                amax = fmaxf(amax, fabsf(v[k]));
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, warp_size));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, warp_size));

            const float d = amax / 127.0f;
            int q[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int8_t qk8 = amax == 0.0f ? 0 : roundf(v[k] / d);
                q[k] = (int) qk8 & 0xff;
            }
            int * qs = (int *) y_q8[ib].qs;
            qs[2*sub + 0] = q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
            qs[2*sub + 1] = q[4] | (q[5] << 8) | (q[6] << 16) | (q[7] << 24);
            if (sub == 0) {
                y_q8[ib].ds = make_half2(d, 0.0f);
            }
        }
    }
    __syncthreads();

    if (!row_ok) {
        return;
    }

    float tmp      = 0.0f;
    float tmp_gate = 0.0f;

    const block_q8_1 * y = y_q8;
    constexpr int chunk_blocks = pf*blocks_per_iter;
    for (int c0 = kbx0; c0 < blocks_per_row_x; c0 += chunk_blocks) {
        mmvq_fq_q8_0_chunk<pf> nx;
        mmvq_fq_q8_0_chunk<pf> ng;
        const int c1 = c0 + chunk_blocks;
        if (c1 < blocks_per_row_x) {
            mmvq_fq_q8_0_load(nx, x, c1, kqs, blocks_per_row_x);
            if (use_gate) {
                mmvq_fq_q8_0_load(ng, xg, c1, kqs, blocks_per_row_x);
            }
        }
        mmvq_fq_q8_0_dot(tmp, cx, y, c0, kqs, blocks_per_row_x);
        if (use_gate) {
            mmvq_fq_q8_0_dot(tmp_gate, cg, y, c0, kqs, blocks_per_row_x);
        }
        cx = nx;
        if (use_gate) {
            cg = ng;
        }
    }

    tmp = warp_reduce_sum<warp_size>(tmp);
    if (use_gate) {
        tmp_gate = warp_reduce_sum<warp_size>(tmp_gate);
    }

    if (lane == 0) {
        float result = tmp;
        if (use_gate) {
            switch (seg.glu_op) {
                case GGML_GLU_OP_SWIGLU:
                    result *= ggml_cuda_op_silu_single(tmp_gate);
                    break;
                case GGML_GLU_OP_GEGLU:
                    result *= ggml_cuda_op_gelu_single(tmp_gate);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    result = ggml_cuda_op_swiglu_oai_single(tmp_gate, result);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    result = ggml_cuda_op_swiglu_clamp_single(tmp_gate, result, seg.glu_limit);
                    break;
                default:
                    result = result * tmp_gate;
                    break;
            }
        }
        seg.dst[row] = result;
    }
}

bool ggml_cuda_mmv_group_seg_ok(const ggml_tensor * w, const ggml_tensor * y) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return false;
    }
    if (y->type != GGML_TYPE_F32 || y->ne[1] != 1 || y->ne[2] != 1 || y->ne[3] != 1 || !ggml_is_contiguous(y) ||
            (uintptr_t) y->data % 16 != 0 || y->ne[0] != w->ne[0]) {
        return false;
    }
    if (w->ne[2] != 1 || w->ne[3] != 1 || w->nb[0] != ggml_type_size(w->type) || w->ne[1] > INT32_MAX ||
            (w->buffer && ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE)) {
        return false;
    }
    const int64_t ncols = w->ne[0];
    if (w->type == GGML_TYPE_Q8_0) {
        return ncols % QK8_1 == 0 && (ncols / QK8_1) * sizeof(block_q8_1) <= 16384 && w->nb[1] % sizeof(block_q8_0) == 0;
    }
    return false;
}

void ggml_cuda_mmv_group(ggml_backend_cuda_context & ctx, const ggml_tensor * y, const ggml_cuda_mmv_group_seg * segs, const int nseg) {
    GGML_ASSERT(nseg >= 1 && nseg <= MMVQ_GROUP_MAX);
    const int device    = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const int nsm       = ggml_cuda_info().devices[device].nsm;
    GGML_ASSERT(warp_size == 32);

    const int ncols = y->ne[0];

    int64_t total_q8_rows = 0;
    int64_t max_rows      = 0;
    for (int i = 0; i < nseg; ++i) {
        GGML_ASSERT(ggml_cuda_mmv_group_seg_ok(segs[i].w, y));
        GGML_ASSERT(segs[i].dst->type == GGML_TYPE_F32 && ggml_is_contiguous(segs[i].dst) && ggml_nelements(segs[i].dst) == segs[i].w->ne[1]);
        if (segs[i].gate) {
            GGML_ASSERT(segs[i].w->type == GGML_TYPE_Q8_0 && segs[i].gate->type == GGML_TYPE_Q8_0 &&
                ggml_are_same_shape(segs[i].w, segs[i].gate) && segs[i].gate->nb[0] == sizeof(block_q8_0) &&
                segs[i].gate->nb[1] % sizeof(block_q8_0) == 0);
        }
        if (segs[i].w->type == GGML_TYPE_Q8_0) {
            total_q8_rows += segs[i].w->ne[1];
        }
        max_rows = std::max<int64_t>(max_rows, segs[i].w->ne[1]);
    }

    // same block-size heuristic as mul_mat_vec_q_fq_launch, applied to the largest segment
    int nwarps = MMVQ_FQ_NWARPS;
    while (nwarps > 4 && ((max_rows + nwarps - 1) / nwarps) < 2 * nsm) {
        nwarps /= 2;
    }

    mmvq_group_args_dev args{};
    args.nseg = nseg;
    int nblocks = 0;
    for (int i = 0; i < nseg; ++i) {
        mmvq_group_seg_dev & d = args.seg[i];
        const ggml_tensor * w = segs[i].w;
        d.vx           = w->data;
        d.gate         = segs[i].gate ? segs[i].gate->data : nullptr;
        d.dst          = (float *) segs[i].dst->data;
        d.nrows        = (int) w->ne[1];
        d.type         = w->type == GGML_TYPE_Q8_0 ? MMVQ_GROUP_TYPE_Q8_0 : MMVQ_GROUP_TYPE_F32;
        d.stride_row   = (int) (w->nb[1] / ggml_type_size(w->type));
        d.gate_stride_row = segs[i].gate ? (int) (segs[i].gate->nb[1] / sizeof(block_q8_0)) : 0;
        d.blocks_begin = nblocks;
        d.glu_op       = (int) segs[i].glu_op;
        d.glu_limit    = segs[i].glu_limit;
        nblocks += (d.nrows + nwarps - 1) / nwarps;
    }

    const dim3 block_nums(nblocks, 1, 1);
    const dim3 block_dims(warp_size, nwarps, 1);
    const int  nbytes_shared = total_q8_rows > 0 ? (ncols / QK8_1) * sizeof(block_q8_1) : 0;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, ctx.stream());

    constexpr int blocks_per_iter = VDR_Q8_0_Q8_1_MMVQ * 32 / QI8_0;
    const bool long_rows = (ncols / QK8_0) >= 16 * blocks_per_iter && total_q8_rows <= 1024;

    auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, launch_params, args, (const float *) y->data, ncols, 1.0f, 0.0f, 0);
    };

    switch (nwarps) {
        case 16: launch(mul_mat_vec_fq_group<16, MMVQ_FQ_PF>); break;
        case 8:
            if (long_rows) { launch(mul_mat_vec_fq_group<8, MMVQ_FQ_PF_LONG>); }
            else           { launch(mul_mat_vec_fq_group<8, MMVQ_FQ_PF>); }
            break;
        case 4:
            if (long_rows) { launch(mul_mat_vec_fq_group<4, MMVQ_FQ_PF_LONG>); }
            else           { launch(mul_mat_vec_fq_group<4, MMVQ_FQ_PF>); }
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// ---------------------------------------------------------------------------------------------
// qwen4exp hyper-connection up projection + stream mix, one token (RDNA3.5):
//     gate = W_up * y'   (y' = op(y_scale*y + y_bias), Q8_0 weights [ncols, hc*n_embd], fused quantization)
//     mixed[e] = mix_scale * ( sum_c xn[c*n_embd + e] * sigmoid(gate[c*n_embd + e]) ) + mix_bias
// One wave owns element e and computes its hc gate rows {c*n_embd + e}: the hc*blocks_per_row Q8_0 blocks
// of those rows are spread flat over the lanes (4 lanes per block, 8 blocks per pass), so the short rows
// (ncols = 320 -> 10 blocks) keep all lanes busy. The gate itself is never written; the mix uses the same
// rounding sequence as hc_mix_reduce_f32 (separate mul / add roundings, streams summed in order).

#define MMVQ_HCMIX_MAX_PASSES 16
#define MMVQ_HCMIX_MAX_HC     4
#define MMVQ_HCMIX_EPW        2   // elements (mixed outputs) per wave: 2*hc adjacent-row pairs stream per wave

#if defined(__HIP_PLATFORM_AMD__)
static __device__ __forceinline__ float mmvq_hc_mul_rn(const float a, const float b) {
    float result;
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}
static __device__ __forceinline__ float mmvq_hc_add_rn(const float a, const float b) {
    float result;
    asm("v_add_f32_e32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}
#else
static __device__ __forceinline__ float mmvq_hc_mul_rn(const float a, const float b) { return __fmul_rn(a, b); }
static __device__ __forceinline__ float mmvq_hc_add_rn(const float a, const float b) { return __fadd_rn(a, b); }
#endif

template <int nwarps>
__launch_bounds__(nwarps * ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_fq_hcmix(
        const void * GGML_CUDA_RESTRICT vx_ptr, const float * GGML_CUDA_RESTRICT y_ptr, const float * xn, float * dst,
        const int ncols_x, const int n_embd, const int hc, const int stride_row_x,
        const float y_scale, const float y_bias, const int y_op, const float mix_scale, const float mix_bias) {
    constexpr int qi  = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int lanes_per_row = qi / vdr;                 // 4
    constexpr int rows_per_wave = warp_size / lanes_per_row; // 8 = hc_max * epw
    constexpr int epw = MMVQ_HCMIX_EPW;
    static_assert(rows_per_wave == MMVQ_HCMIX_MAX_HC * epw, "lane layout");

    extern __shared__ char mmvq_fq_smem[];
    block_q8_1 * y_q8 = (block_q8_1 *) mmvq_fq_smem;

    const int lane = threadIdx.x;
    const int tid  = warp_size*threadIdx.y + lane;

    const int e0 = (blockIdx.x*nwarps + threadIdx.y) * epw;   // first element of this wave

    // lane group g = lane/4 owns wave row r = g: stream c = r / epw, element e0 + (r % epw)
    const int r   = lane / lanes_per_row;
    const int kqs = vdr * (lane % lanes_per_row);
    const int c   = r / epw;
    const int j   = r - c*epw;
    const bool row_ok = c < hc && e0 + j < n_embd;
    const int row = c*n_embd + min(e0 + j, n_embd - 1); // clamped so that every lane reads valid memory

    const block_q8_0 * x = (const block_q8_0 *) vx_ptr + (size_t) (row_ok ? row : 0)*stride_row_x;
    const int blocks_per_row = ncols_x / QK8_0;

    // 1. the whole row of this lane group first (the DRAM stream; everything else is tiny)
    int  qs[MMVQ_HCMIX_MAX_PASSES][vdr];
    half d[MMVQ_HCMIX_MAX_PASSES];
#pragma unroll
    for (int p = 0; p < MMVQ_HCMIX_MAX_PASSES; ++p) {
        if (p < blocks_per_row) {
#pragma unroll
            for (int jj = 0; jj < vdr; ++jj) {
                qs[p][jj] = get_int_b2(x[p].qs, kqs + jj);
            }
            d[p] = x[p].d;
        }
    }

    // 2. quantize the activation vector into shared memory (same as mul_mat_vec_q_fq)
    {
        const int nby = ncols_x / QK8_1;
        for (int i = tid; i < 4*nby; i += nwarps*warp_size) {
            const int ib  = i >> 2;
            const int sub = i & 3;
            const float4 * yv = (const float4 *) (y_ptr + ib*QK8_1 + sub*8);
            const float4 v0 = yv[0];
            const float4 v1 = yv[1];
            float v[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
            if (y_op != 0) {
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const float t = y_scale * v[k] + y_bias;
                    v[k] = y_op == 1 ? ggml_cuda_op_silu_single(t) : 1.0f / (1.0f + expf(-t));
                }
            }
            float amax = fabsf(v[0]);
#pragma unroll
            for (int k = 1; k < 8; ++k) {
                amax = fmaxf(amax, fabsf(v[k]));
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, warp_size));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, warp_size));
            const float dq = amax / 127.0f;
            int q[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int8_t qk8 = amax == 0.0f ? 0 : roundf(v[k] / dq);
                q[k] = (int) qk8 & 0xff;
            }
            int * qsy = (int *) y_q8[ib].qs;
            qsy[2*sub + 0] = q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
            qsy[2*sub + 1] = q[4] | (q[5] << 8) | (q[6] << 16) | (q[7] << 24);
            if (sub == 0) {
                y_q8[ib].ds = make_half2(dq, 0.0f);
            }
        }
        __syncthreads();
    }

    if (e0 >= n_embd) {
        return;
    }

    // 3. dot product of this lane group's row
    float acc = 0.0f;
#pragma unroll
    for (int p = 0; p < MMVQ_HCMIX_MAX_PASSES; ++p) {
        if (p < blocks_per_row) {
            const block_q8_1 * bq8_1 = &y_q8[p];
            int u[vdr];
#pragma unroll
            for (int jj = 0; jj < vdr; ++jj) {
                u[jj] = get_int_b4(bq8_1->qs, kqs + jj);
            }
            acc += vec_dot_q8_0_q8_1_impl<float, vdr>(qs[p], u, d[p], __low2half(bq8_1->ds));
        }
    }
    acc += __shfl_xor_sync(0xffffffff, acc, 1, warp_size);
    acc += __shfl_xor_sync(0xffffffff, acc, 2, warp_size);
    // lane 4*r now holds the gate of wave row r = c*epw + j

    float g[MMVQ_HCMIX_MAX_HC];
#pragma unroll
    for (int cc = 0; cc < MMVQ_HCMIX_MAX_HC; ++cc) {
        // element j = lane (for lane < epw): its stream-cc gate sits in lane group cc*epw + lane
        g[cc] = __shfl_sync(0xffffffff, acc, lanes_per_row * (cc*epw + (lane % epw)), warp_size);
    }

    if (lane < epw && e0 + lane < n_embd) {
        const int e = e0 + lane;
        // same expressions as hc_mix_reduce_f32: sigmoid, rounded mul, streams added in order, then scale_f32
        float mixed = mmvq_hc_mul_rn(xn[e], 1.0f / (1.0f + expf(-g[0])));
#pragma unroll
        for (int cc = 1; cc < MMVQ_HCMIX_MAX_HC; ++cc) {
            if (cc < hc) {
                mixed = mmvq_hc_add_rn(mixed, mmvq_hc_mul_rn(xn[cc*n_embd + e], 1.0f / (1.0f + expf(-g[cc]))));
            }
        }
        dst[e] = mix_scale * mixed + mix_bias;
    }
}

bool ggml_cuda_mul_mat_vec_q_fq_hcmix_ok(const ggml_tensor * w, const ggml_tensor * y, const ggml_tensor * xn, const ggml_tensor * dst, const int hc) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_RDNA3_5(cc) || hc < 1 || hc > MMVQ_HCMIX_MAX_HC) {
        return false;
    }
    if (w->type != GGML_TYPE_Q8_0 || w->ne[2] != 1 || w->ne[3] != 1 || w->nb[0] != sizeof(block_q8_0) || w->nb[1] % sizeof(block_q8_0) != 0 ||
            (w->buffer && ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE)) {
        return false;
    }
    const int64_t ncols  = w->ne[0];
    const int64_t n_embd = dst->ne[0];
    if (ncols % QK8_0 != 0 || w->ne[1] != hc * n_embd || (ncols / QK8_0) > MMVQ_HCMIX_MAX_PASSES) {
        return false;
    }
    if (y->type != GGML_TYPE_F32 || !ggml_is_contiguous(y) || ggml_nelements(y) != ncols || (uintptr_t) y->data % 16 != 0) {
        return false;
    }
    if (xn->type != GGML_TYPE_F32 || !ggml_is_contiguous(xn) || ggml_nelements(xn) != hc * n_embd ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || ggml_nelements(dst) != n_embd) {
        return false;
    }
    // every block reads y: dst must not overlap it
    {
        const char * d0 = (const char *) dst->data;
        const char * d1 = d0 + ggml_backend_buft_get_alloc_size(dst->buffer->buft, dst);
        const char * y0 = (const char *) y->data;
        const char * y1 = y0 + ggml_backend_buft_get_alloc_size(y->buffer->buft, y);
        if (d0 < y1 && y0 < d1) {
            return false;
        }
    }
    // dst may alias xn only on a whole stream (element e is then read and written by the same wave)
    {
        const char * d0 = (const char *) dst->data;
        const char * d1 = d0 + ggml_backend_buft_get_alloc_size(dst->buffer->buft, dst);
        const char * x0 = (const char *) xn->data;
        const char * x1 = x0 + ggml_backend_buft_get_alloc_size(xn->buffer->buft, xn);
        if (d0 < x1 && x0 < d1) {
            const ptrdiff_t off = d0 - x0;
            if (off < 0 || off % (n_embd * (ptrdiff_t) sizeof(float)) != 0 || off + (ptrdiff_t) ggml_nbytes(dst) > (ptrdiff_t) ggml_nbytes(xn)) {
                return false;
            }
        }
    }
    return true;
}

void ggml_cuda_mul_mat_vec_q_fq_hcmix(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * y, const ggml_tensor * xn,
        ggml_tensor * dst, const int hc, const float y_scale, const float y_bias, const int y_op, const float mix_scale, const float mix_bias) {
    GGML_ASSERT(ggml_cuda_mul_mat_vec_q_fq_hcmix_ok(w, y, xn, dst, hc));
    const int device    = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    GGML_ASSERT(warp_size == 32);

    const int ncols  = (int) w->ne[0];
    const int n_embd = (int) dst->ne[0];

    constexpr int nwarps = 32;
    const int n_waves = (n_embd + MMVQ_HCMIX_EPW - 1) / MMVQ_HCMIX_EPW;
    const dim3 block_nums((n_waves + nwarps - 1) / nwarps, 1, 1);
    const dim3 block_dims(warp_size, nwarps, 1);
    const int  nbytes_shared = (ncols / QK8_1) * sizeof(block_q8_1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, nbytes_shared, ctx.stream());
    ggml_cuda_kernel_launch(mul_mat_vec_q_fq_hcmix<nwarps>, launch_params,
        w->data, (const float *) y->data, (const float *) xn->data, (float *) dst->data,
        ncols, n_embd, hc, (int) (w->nb[1] / sizeof(block_q8_0)), y_scale, y_bias, y_op, mix_scale, mix_bias);
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

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

    ggml_cuda_pdl_sync();
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

    ggml_cuda_pdl_lc();

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
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst);
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block>, launch_params,
        vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        ncols_dst, ids_stride);
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

    const bool has_ids = ids != nullptr;

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

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

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
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

    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            if constexpr (type == GGML_TYPE_IQ3_S) {
                constexpr int c_rows = MMVQ_IQ3_S_ROWS_PER_BLOCK;
                const bool no_extra_fusion = fusion.x_bias == nullptr && fusion.gate_bias == nullptr &&
                    fusion.x_scale == nullptr && fusion.gate_scale == nullptr;
                const bool gate_ok = fusion.gate == nullptr || fusion.glu_op == GGML_GLU_OP_SWIGLU;
                if (table_id == MMVQ_PARAMETERS_RDNA3_5 && ids != nullptr && ncols_x == 2560 &&
                        no_extra_fusion && gate_ok && nsamples_dst == 1) {
                    static const int rows_env = getenv("GGML_IQ3_ROWS") ? atoi(getenv("GGML_IQ3_ROWS")) : c_rows;
                    auto launch_rows = [&](auto kernel, int rows) {
                        const dim3 block_nums((nrows_x + rows - 1) / rows, nchannels_dst, nsamples_dst);
                        const dim3 block_dims(warp_size, rows, 1);
                        const ggml_cuda_kernel_launch_params params(block_nums, block_dims, 0, stream);
                        ggml_cuda_kernel_launch(kernel, params,
                            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
                            stride_row_x, stride_col_dst, stride_channel_x, stride_channel_y,
                            stride_channel_dst, stride_sample_x, stride_sample_y, stride_sample_dst);
                    };
                    static const int lds_env  = getenv("GGML_IQ3_LDS")  ? atoi(getenv("GGML_IQ3_LDS"))  : 0;
                    static const int grid_env = getenv("GGML_IQ3_GRID") ? atoi(getenv("GGML_IQ3_GRID")) : 4;
                    if (fusion.gate != nullptr && grid_env > 0) {
                        switch (grid_env) {
                            case 1: launch_rows(mul_mat_vec_iq3_s_grid_rdna3_5<true, 1>, 1); break;
                            case 2: launch_rows(mul_mat_vec_iq3_s_grid_rdna3_5<true, 2>, 2); break;
                            case 8: launch_rows(mul_mat_vec_iq3_s_grid_rdna3_5<true, 8>, 8); break;
                            default: launch_rows(mul_mat_vec_iq3_s_grid_rdna3_5<true, 4>, 4); break;
                        }
                    } else if (fusion.gate != nullptr && lds_env > 0) {
                        switch (lds_env) {
                            case 1: launch_rows(mul_mat_vec_iq3_s_lds_rdna3_5<true, 1>, 1); break;
                            case 2: launch_rows(mul_mat_vec_iq3_s_lds_rdna3_5<true, 2>, 2); break;
                            case 8: launch_rows(mul_mat_vec_iq3_s_lds_rdna3_5<true, 8>, 8); break;
                            default: launch_rows(mul_mat_vec_iq3_s_lds_rdna3_5<true, 4>, 4); break;
                        }
                    } else if (fusion.gate != nullptr) {
                        switch (rows_env) {
                            case 2: launch_rows(mul_mat_vec_iq3_s_rows_rdna3_5<true, 2>, 2); break;
                            case 4: launch_rows(mul_mat_vec_iq3_s_rows_rdna3_5<true, 4>, 4); break;
                            case 8: launch_rows(mul_mat_vec_iq3_s_rows_rdna3_5<true, 8>, 8); break;
                            default: launch_rows(mul_mat_vec_iq3_s_rows_rdna3_5<true, 1>, 1); break;
                        }
                    } else {
                        launch_rows(mul_mat_vec_iq3_s_rows_rdna3_5<false, c_rows>, c_rows);
                    }
                    break;
                }
            }

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;

                const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                              nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            };

            if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{});
            } else {
                launch(std::false_type{}, std::false_type{});
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
            if constexpr (is_rdna3_5_q4_columns_type(type)) {
                const bool no_fusion = fusion.gate == nullptr && fusion.x_bias == nullptr && fusion.gate_bias == nullptr &&
                    fusion.x_scale == nullptr && fusion.gate_scale == nullptr;
                if (table_id == MMVQ_PARAMETERS_RDNA3_5 && ids == nullptr && no_fusion) {
                    constexpr int c_nwarps = calc_nwarps(type, 1, MMVQ_PARAMETERS_RDNA3_5);
                    const dim3 block_nums(nrows_x, nchannels_dst, nsamples_dst);
                    const dim3 block_dims(warp_size, c_nwarps, 1);
                    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
                    ggml_cuda_kernel_launch(mul_mat_vec_q4_columns_rdna3_5<type>, launch_params,
                        vx, vy, dst, ncols_x, stride_row_x, stride_col_y, stride_col_dst,
                        channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                        sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst);
                    break;
                }
            }
            if constexpr (type == GGML_TYPE_Q8_0) {
                const bool no_fusion = fusion.gate == nullptr && fusion.x_bias == nullptr && fusion.gate_bias == nullptr &&
                    fusion.x_scale == nullptr && fusion.gate_scale == nullptr;
                if (table_id == MMVQ_PARAMETERS_RDNA3_5 && ids == nullptr && no_fusion) {
                    const dim3 block_nums(nrows_x, nchannels_dst, nsamples_dst);
                    const dim3 block_dims(warp_size, 1, 1);
                    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
                    ggml_cuda_kernel_launch(mul_mat_vec_q4_columns<type>, launch_params,
                        vx, vy, dst, ncols_x, stride_row_x, stride_col_y, stride_col_dst,
                        channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                        sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst);
                    break;
                }
            }
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
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
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

// RDNA3.5 token-generation fast path: quantize the activations inside the matvec kernel. Optionally applies an
// elementwise prologue (scale -> silu/sigmoid) to the activations first, so that the two small kernels in front
// of the matvec disappear as well. Returns false when the shape/type is not served by this path.
static bool mul_mat_vec_q_fq_try(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion, const float y_scale, const float y_bias, const int y_op, const bool launch) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || nb00 != ts_src0 || nb10 != ts_src1 || nb0 != ts_dst) {
        return false;
    }

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int64_t ncols_dst_fq = ids ? ne2 : ne1;
    const bool aligned = (uintptr_t) src1->data % 16 == 0 && nb11 % 16 == 0 && nb12 % 16 == 0 && nb13 % 16 == 0;
    const bool fusion_ok = !fusion || (fusion->x_scale == nullptr && fusion->gate_scale == nullptr);
    if (!(GGML_CUDA_CC_IS_RDNA3_5(cc) && ncols_dst_fq == 1 && mmvq_fq_type_ok(src0->type) && aligned && fusion_ok &&
            ne10 % QK8_1 == 0 && (ne10 / QK8_1) * sizeof(block_q8_1) <= 16384)) {
        return false;
    }
    if (!launch) {
        return true;
    }

    cudaStream_t stream = ctx.stream();

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    if (fusion) {
        GGML_ASSERT(!ids || (fusion->x_bias == nullptr && fusion->gate_bias == nullptr));
        if (fusion->x_bias) {
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        fusion_local.glu_op    = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
        fusion_local.y_gate    = fusion->y_gate   ? fusion->y_gate->data   : nullptr;
        fusion_local.y_norm_w  = fusion->y_norm_w ? fusion->y_norm_w->data : nullptr;
        fusion_local.y_eps     = fusion->y_eps;
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

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s1  = dst->nb[1] / ts_dst;
    const int64_t s2  = dst->nb[2] / ts_dst;
    const int64_t s3  = dst->nb[3] / ts_dst;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    mul_mat_vec_q_fq_switch_type(
        src0->data, src0->type, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, s01, stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03, ne3, s03, s13, s3, stream, y_scale, y_bias, y_op);
    return true;
}

bool ggml_cuda_mul_mat_vec_q_fq_prologue_ok(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, const ggml_tensor * y) {
    if (dst->op != GGML_OP_MUL_MAT || !ggml_is_quantized(dst->src[0]->type) || y->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(y) || !ggml_are_same_shape(y, dst->src[1])) {
        return false;
    }
    return mul_mat_vec_q_fq_try(ctx, dst->src[0], y, nullptr, const_cast<ggml_tensor *>(dst), nullptr, 1.0f, 0.0f, 0, /*launch=*/false);
}

void ggml_cuda_mul_mat_vec_q_fq_prologue(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * y,
        const float y_scale, const float y_bias, const int y_op) {
    const bool ok = mul_mat_vec_q_fq_try(ctx, dst->src[0], y, nullptr, dst, nullptr, y_scale, y_bias, y_op, /*launch=*/true);
    GGML_ASSERT(ok);
}

bool ggml_cuda_mul_mat_vec_q_fq_gdn_gate_ok(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, const ggml_tensor * z) {
    const ggml_tensor * y = dst->src[1];
    if (dst->op != GGML_OP_MUL_MAT || !ggml_is_quantized(dst->src[0]->type) || y->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(y) || ggml_nrows(y) != 1 || y->ne[0] % 128 != 0 ||
            z->type != GGML_TYPE_F32 || !ggml_is_contiguous(z) || ggml_nelements(z) != y->ne[0] || (uintptr_t) z->data % 16 != 0) {
        return false;
    }
    // every block reads z while the waves write dst
    const char * d0 = (const char *) dst->data;
    const char * d1 = d0 + ggml_backend_buft_get_alloc_size(dst->buffer->buft, dst);
    const char * z0 = (const char *) z->data;
    const char * z1 = z0 + ggml_backend_buft_get_alloc_size(z->buffer->buft, z);
    if (d0 < z1 && z0 < d1) {
        return false;
    }
    return mul_mat_vec_q_fq_try(ctx, dst->src[0], y, nullptr, const_cast<ggml_tensor *>(dst), nullptr, 1.0f, 0.0f, 0, /*launch=*/false);
}

void ggml_cuda_mul_mat_vec_q_fq_gdn_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const float * attn,
        const ggml_tensor * z, const ggml_tensor * norm_w, const float eps) {
    GGML_ASSERT((uintptr_t) attn % 16 == 0 && (uintptr_t) norm_w->data % 16 == 0 && ggml_nelements(norm_w) == 128);
    ggml_tensor y = *dst->src[1]; // same shape/strides as the (elided) gated norm output, data = pre-norm scratch
    y.data = (void *) attn;
    ggml_cuda_mm_fusion_args_host fusion{};
    fusion.y_gate   = z;
    fusion.y_norm_w = norm_w;
    fusion.y_eps    = eps;
    const bool ok = mul_mat_vec_q_fq_try(ctx, dst->src[0], &y, nullptr, dst, &fusion, 1.0f, 0.0f, 3, /*launch=*/true);
    GGML_ASSERT(ok);
}

template <int n_expert_used>
__launch_bounds__(32, 8)
static __global__ void mul_mat_id_iq4_nl_weighted_rdna3_5(
        const void * vx_ptr, const block_q8_1 * y, const int32_t * ids, const float * weights, float * dst,
        const int nrows, const int blocks_per_row, const int stride_row_x, const int stride_channel_x,
        const int stride_y) {
    constexpr int qi        = ggml_cuda_type_traits<GGML_TYPE_IQ4_NL>::qi;
    constexpr int vdr       = VDR_Q4_0_Q8_1_MMVQ;
    constexpr int warp_size = 32;
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    const int lane = threadIdx.x;
    const int row  = blockIdx.x;
    if (row >= nrows) {
        return;
    }

    float result = 0.0f;
#pragma unroll
    for (int ex = 0; ex < n_expert_used; ++ex) {
        const int channel = ids[ex];
        const int x_off = channel * stride_channel_x + row * stride_row_x;
        float sum = 0.0f;
        for (int kbx = lane / (qi/vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
            const int kqs = vdr * (lane % (qi/vdr));
            sum += vec_dot_iq4_nl_q8_1(vx_ptr, y + ex * stride_y + kbx, x_off + kbx, kqs);
        }
        sum = warp_reduce_sum<warp_size>(sum);
        const float term = mmvq_hc_mul_rn(sum, weights[ex]);
        result = ex == 0 ? term : mmvq_hc_add_rn(result, term);
    }

    if (lane == 0) {
        dst[row] = result;
    }
}

template <int n_expert_used>
__launch_bounds__(32, 8)
static __global__ void mul_mat_id_q8_0_weighted_rdna3_5(
        const void * vx_ptr, const block_q8_1 * y, const int32_t * ids, const float * weights, float * dst,
        const int nrows, const int blocks_per_row, const int stride_row_x, const int stride_channel_x,
        const int stride_y) {
    constexpr int qi        = QI8_0;
    constexpr int vdr       = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = 32;
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    const int lane = threadIdx.x;
    const int row  = blockIdx.x;
    if (row >= nrows) {
        return;
    }

    float result = 0.0f;
#pragma unroll
    for (int ex = 0; ex < n_expert_used; ++ex) {
        const int channel = ids[ex];
        const int x_off = channel * stride_channel_x + row * stride_row_x;
        float sum = 0.0f;
        for (int kbx = lane / (qi/vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
            const int kqs = vdr * (lane % (qi/vdr));
            sum += vec_dot_q8_0_q8_1(vx_ptr, y + ex * stride_y + kbx, x_off + kbx, kqs);
        }
        sum = warp_reduce_sum<warp_size>(sum);
        const float term = mmvq_hc_mul_rn(sum, weights[ex]);
        result = ex == 0 ? term : mmvq_hc_add_rn(result, term);
    }

    if (lane == 0) {
        dst[row] = result;
    }
}

bool ggml_cuda_mul_mat_id_weighted_rdna3_5_ok(
        const ggml_tensor * experts, const ggml_tensor * weights, const ggml_tensor * dst) {
    if (experts->op != GGML_OP_MUL_MAT_ID ||
            (experts->src[0]->type != GGML_TYPE_IQ4_NL && experts->src[0]->type != GGML_TYPE_Q8_0) ||
            experts->src[1]->type != GGML_TYPE_F32 || experts->src[2]->type != GGML_TYPE_I32 ||
            experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor * w   = experts->src[0];
    const ggml_tensor * y   = experts->src[1];
    const ggml_tensor * ids = experts->src[2];
    const int64_t n_used = ids->ne[0];
    return GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) &&
        n_used == 10 && w->ne[0] == 640 && w->ne[1] == 2560 && w->ne[2] == 512 && w->ne[3] == 1 &&
        y->ne[0] == 640 && ggml_nelements(y) == 640 * n_used && ggml_is_contiguous(y) &&
        ggml_nelements(ids) == n_used && ggml_is_contiguous(ids) &&
        weights->ne[0] == 1 && weights->ne[1] == n_used && ggml_nelements(weights) == n_used && ggml_is_contiguous(weights) &&
        ggml_nelements(experts) == 2560 * n_used && ggml_is_contiguous(experts) &&
        ggml_nelements(dst) == 2560 && ggml_is_contiguous(dst);
}

void ggml_cuda_mul_mat_id_weighted_rdna3_5(
        ggml_backend_cuda_context & ctx, const ggml_tensor * experts, const ggml_tensor * weights, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_mul_mat_id_weighted_rdna3_5_ok(experts, weights, dst));
    const ggml_tensor * w   = experts->src[0];
    const ggml_tensor * y   = experts->src[1];
    const ggml_tensor * ids = experts->src[2];
    constexpr int n_used = 10;
    constexpr int ncols  = 640;
    constexpr int nrows  = 2560;
    constexpr int nblocks = ncols / QK8_1;

    ggml_cuda_pool_alloc<block_q8_1> y_q8(ctx.pool(), n_used * nblocks);
    quantize_row_q8_1_cuda(
        (const float *) y->data, nullptr, y_q8.get(), w->type,
        ncols,
        y->nb[1] / sizeof(float), y->nb[2] / sizeof(float), y->nb[3] / sizeof(float),
        y->ne[0], y->ne[1], y->ne[2], y->ne[3], ctx.stream());

    const ggml_cuda_kernel_launch_params params(nrows, 32, 0, ctx.stream());
    if (w->type == GGML_TYPE_IQ4_NL) {
        ggml_cuda_kernel_launch(mul_mat_id_iq4_nl_weighted_rdna3_5<n_used>, params,
            w->data, y_q8.get(), (const int32_t *) ids->data, (const float *) weights->data, (float *) dst->data,
            nrows, nblocks, (int) (w->nb[1] / ggml_type_size(w->type)),
            (int) (w->nb[2] / ggml_type_size(w->type)), nblocks);
    } else {
        ggml_cuda_kernel_launch(mul_mat_id_q8_0_weighted_rdna3_5<n_used>, params,
            w->data, y_q8.get(), (const int32_t *) ids->data, (const float *) weights->data, (float *) dst->data,
            nrows, nblocks, (int) (w->nb[1] / ggml_type_size(w->type)),
            (int) (w->nb[2] / ggml_type_size(w->type)), nblocks);
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
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

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
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
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

    if (mul_mat_vec_q_fq_try(ctx, src0, src1, ids, dst, fusion, 1.0f, 0.0f, 0, /*launch=*/true)) {
        return;
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
