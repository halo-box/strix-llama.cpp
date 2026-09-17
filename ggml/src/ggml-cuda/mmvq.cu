#include "mmvq.cuh"
#include "ggml-backend-impl.h"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <vector>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static __device__ __forceinline__ void mmvq_prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif

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

// LEVER (2026-09-10, results/2026-09-10-lever-mmqrouted/): runtime override of the per-type
//     MMVQ/MMQ crossover on RDNA3.5, so that one binary can serve both arms of a comparison.
// Format: GGML_MMVQ_THR="<ggml type id>:<max ne11 kept on MMVQ>[,...]", e.g. "18:6" for IQ3_XXS.
// Unset (the default) changes nothing; a value of 0 for a type also means "no override".
// The table below is tuned per type and was measured on a tree that predates the MMVQ
//     rows-per-block work, hence this switch.
static const int * ggml_cuda_mmvq_thr_override() {
    static const std::vector<int> tbl = []() {
        std::vector<int> t(GGML_TYPE_COUNT, 0);
        const char * s = getenv("GGML_MMVQ_THR");
        if (s == nullptr) {
            return t;
        }
        const char * p = s;
        while (*p) {
            char * end = nullptr;
            const long ty = strtol(p, &end, 10);
            if (end == p || *end != ':') {
                break;
            }
            p = end + 1;
            const long thr = strtol(p, &end, 10);
            if (end == p) {
                break;
            }
            p = end;
            if (ty >= 0 && ty < GGML_TYPE_COUNT && thr > 0) {
                t[ty] = (int) thr;
            }
            if (*p == ',') {
                ++p;
            } else {
                break;
            }
        }
        return t;
    }();
    return tbl.data();
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne01, int64_t ne11, bool exact_batch) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // Below one MMQ tile there is no tiled kernel to fill. The smallest MMQ tile on RDNA3.5 is
    //     I = 64 rows, so ne01 = 48 (the GDN alpha/beta projections) launches a single block and
    //     leaves 39 of the 40 CUs idle, while MMVQ launches a block per row (or row pair). The
    //     per-type thresholds below are all crossovers measured at m >= 5120, where MMQ fills the
    //     device; they do not describe this regime.
    if (GGML_CUDA_CC_IS_RDNA3_5(cc) && ne01 < 64 && ne11 <= MMVQ_MAX_BATCH_SIZE) {
        return true;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
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
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ORIN) {
        switch (type) { // tuned for Jetson Orin
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return ne11 <= 1;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        // Tuned on gfx1151 (Strix Halo) by timing both paths at the same ne11 with a temporary
        //     dispatch override, on real model shapes. MMVQ re-reads the src1 column and redoes
        //     the dot per output column while the weight unpack is hoisted out of the ncols_dst
        //     loop, so its cost is A + ncols_dst*D with D dominated by the type's vec_dot. MMQ's
        //     smallest instantiated tile here is J = 16, so one column tile covers the whole
        //     ne11 = 1..16 range and MMQ's cost is flat across it (measured within 5 %). The
        //     crossover is therefore a per-type constant, and it does not move with m or k once
        //     MMQ fills a wave (ceil(m/64) >= 80): Q5_K measures 2.34 / 2.33 / 2.31 / 2.35 at
        //     m = 5120 (k 6144), 5120 (k 17408), 17408 and 248320, i.e. from 1.0 to 48.5 waves.
        if (exact_batch) {
            // GGML_HINT_EXACT_BATCH promises that mul_mat over n columns equals n single-column
            // mul_mats exactly. A per-type threshold inside [1, MMVQ_MAX_BATCH_SIZE] would send the
            // batch to MMQ while the singles stay on MMVQ, and the two accumulate differently.
            // Same reasoning as the BF16/MMVF case in ggml-cuda.cu.
            return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
        // LEVER: per-type crossover override, off unless GGML_MMVQ_THR is set. See above.
        {
            const int thr_env = ggml_cuda_mmvq_thr_override()[type];
            if (thr_env > 0) {
                return ne11 <= thr_env;
            }
        }
        // 2026-09-11, results/2026-09-10-crossover-retune/: the whole table below was re-measured.
        //     Instrument: GGML_MMVQ_THR above, so ONE binary served both arms and the only
        //     difference between them was the environment. Two arms (every type forced to MMVQ,
        //     every type forced to MMQ), 4-6 position-balanced arms per side, at the (ne00 x ne01)
        //     pairs each type actually takes in Qwen3.8-27B-UD-IQ4_XS.gguf. An f16 anchor moved
        //     +0.28 % between arms and ne11 = 1, which no threshold can reach, is null for every
        //     type, so the arms are comparable.
        //
        //     What is recorded per type is the COST FUNCTION, not just the crossover, because a
        //     crossover is not durable -- it moves whenever either kernel changes, which is how
        //     this table went stale twice. MMVQ costs A + ne11*D and MMQ costs a flat Q (measured:
        //     Q varies 0.4-2.6 % over ne11 = 2..8, so "flat" is now checked rather than assumed),
        //     giving crossover = floor((Q - A)/D). Units are us/run at ne00 = 5120, ne01 = 17408.
        //
        //             A       D       Q     (Q-A)/D   bound   previous
        //     IQ2_S    82.3   34.3   506.8   12.4       8       8   (cap, not a crossover)
        //     IQ3_XXS 112.3   34.0   406.7    8.7       8       6
        //     IQ4_XS  156.1   26.9   410.3    9.5       8       6
        //     IQ3_S   135.2   34.2   428.5    8.6       8       6
        //     Q8_0    383.8   19.3   498.6    6.0       5       4
        //     Q6_K    250.7   80.1   635.3    4.8       4       3
        //     Q3_K    246.1   71.5   497.0    3.5       3       3   RE-MEASURED, UNCHANGED
        //     Q4_K    119.3  109.7   411.1    2.7       2       2   RE-MEASURED, UNCHANGED
        //     Q5_K    163.1  109.0   433.4    2.5       2       2   RE-MEASURED, UNCHANGED
        //
        //     The k-quants keeping 2-3 is the result, not an omission. Their per-column slope D is
        //     107-110 us against 27-34 for the IQ types at the same shape, a factor of 3-4, which
        //     is the "k-quants are expensive to decode and mvq redoes that per column" comment
        //     above holding up under measurement. Q5_K was additionally measured at output.weight
        //     (5120 x 248320, 834 MiB, the largest matmul in the file and live at ne11 = 6 during
        //     speculative verification): D = 1069 us/col there and the crossover is 2.24, i.e. the
        //     one shape that could have overturned the bound of 2 confirms it hardest.
        //
        //     Q4_K, Q5_K and Q6_K were re-measured a second time on top of 6cb4ed016 (the J = 16
        //     MMQ prefetch, which makes the MMQ side of exactly these three cheaper). Same bounds:
        //     2, 2 and 4. Q6_K's crossover moves 4.80 -> 4.58, still clear of 4.
        //
        //     The eleven types no local GGUF contains were swept afterwards at the generic 27B
        //     shapes (5120x17408, 17408x5120, 5120x6144), same instrument and design. Their
        //     bounds below are therefore measured at REPRESENTATIVE shapes, not at shapes from a
        //     real file, and unlike the nine above they have no end-to-end confirmation, because
        //     there is no model on the tuning box to run one on. Only Q1_0 and Q2_0 remain
        //     entirely unmeasured.
        //
        //             A       D       Q     (Q-A)/D   bound   previous
        //     IQ2_XS   83.1   33.0   501.1   12.7       8       8   RE-MEASURED, UNCHANGED
        //     Q5_0    237.3   26.0   498.0   10.1       8       6
        //     IQ2_XXS  71.3   33.1   400.3    9.9       8       8   RE-MEASURED, UNCHANGED
        //     Q4_0    179.0   24.4   413.9    9.6       8       6
        //     IQ1_S    45.7   31.0   337.0    9.4       8       6
        //     IQ4_NL  184.4   27.8   407.3    8.0       6       5
        //     MXFP4   168.3   28.2   386.0    7.7       6       5
        //     Q4_1    212.4   22.0   378.4    7.5       6       5
        //     Q2_K     55.5   97.3   736.0    7.0       6       4
        //     NVFP4   103.7  112.1   572.8    4.2       4       8   LOWERED
        //     Q5_1     49.9  151.3   437.4    2.6       2       5   LOWERED
        //
        //     Two of those are the table being wrong in the direction that costs, not the
        //     direction that leaves something on the table. Q5_1 has the second-largest
        //     per-column slope of any type here (151 us, above even Q4_K and Q5_K) and its bound
        //     of 5 was sending it to MMVQ at ne11 = 3, 4 and 5 where MMVQ measures 25-33 %, 6-8 %
        //     and 81-102 % SLOWER. NVFP4's bound of 8 was doing the same at ne11 = 5..8, where
        //     MMVQ is 12-15 % slower at 5 and 37 % slower at 6; the previous comment's "NVFP4
        //     does not cross below 9" does not reproduce. Both are resolved on all three shapes
        //     (t = +5.9 .. +49.7).
        //
        //     KNOWN INCOMPLETE: the crossover is not a per-type constant, it is a per-(type, ne01)
        //     constant, and one number per type cannot express that. At ne01 = 1024 (attn_k,
        //     attn_v) MMQ launches 16 tiles against 40 CUs and the measured crossovers are Q4_K 4,
        //     Q5_K 5, Q6_K >= 8, Q8_0 >= 8 -- two to four columns above the values below. The
        //     bounds below are the ne01 >= 5120 values, which is the conservative choice because
        //     the FFN tensors carry far more bytes; the cost is that attn_k/attn_v stay on MMQ at
        //     ne11 = 3-5 where MMVQ is 8-18 % faster.
        switch (type) {
            case GGML_TYPE_Q5_1:
                // 5 -> 2. Crossover 2.50-2.58 over three shapes, D = 141-151 us/col. MMVQ is
                //     3.7-11.0 % faster at ne11 = 2 and 25-33 % SLOWER at 3, so the previous
                //     bound of 5 was a regression on every Q5_1 tensor at three to five columns.
                //     Representative shapes; no local GGUF has Q5_1, so no end-to-end check.
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                // Crossover 2.5-2.8 (Q4_K) and 2.2-2.6 (Q5_K) across four and five shapes
                //     respectively, on both d6e477a70 and 6cb4ed016. MMVQ is 6-12 % faster at
                //     ne11 = 2 and 10-25 % slower at ne11 = 3. Unchanged.
                return ne11 <= 2;
            case GGML_TYPE_Q3_K:
                // Crossover 3.5-4.0 over three shapes. MMVQ -9.9 % at ne11 = 3, +0.9 % at 4
                //     (+6.3 / -3.0 / -0.7 per shape, i.e. no consistent win). Unchanged.
                return ne11 <= 3;
            case GGML_TYPE_NVFP4:
                // 8 -> 4. Crossover 4.19-4.27 over three shapes, D = 41-112 us/col. MMVQ is
                //     3.7-8.0 % faster at ne11 = 4 and 11.7-14.8 % slower at 5 (t = +5.9 .. +14.5)
                //     and 36.6-37.5 % slower at 6. The previous bound of 8 also ran the
                //     ncols_dst 7 and 8 kernels, which rdna3_5_rows4_max_ncols_dst already
                //     records as spilling 24 and 68 registers under four rows per block.
                //     Representative shapes; no local GGUF has NVFP4, so no end-to-end check.
            case GGML_TYPE_Q6_K:
                // 3 -> 4. Crossover 4.7-4.9 over three shapes. MMVQ is 10.8-11.7 % faster at
                //     ne11 = 4 (t = -7.6 .. -12.2, every CI clear of zero) and 1.9-5.4 % slower at
                //     5. Re-measured on 6cb4ed016 as well, where the J = 16 prefetch makes MMQ
                //     cheaper: crossover 4.58-4.62, bound still 4.
                return ne11 <= 4;
            case GGML_TYPE_Q8_0:
                // 4 -> 5. Crossover 5.7-7.5 over three shapes. MMVQ is 6.4-11.4 % faster at
                //     ne11 = 5 (t = -9.8 .. -14.7) and mixed at 6 (-6.6 / -1.8 / +2.3), so 5 is
                //     where the win is consistent. Not affected by the J = 16 prefetch, whose
                //     Q8_0 entries are J = 48 and J = 128.
                return ne11 <= 5;
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_MXFP4:
            case GGML_TYPE_IQ4_NL:
                // Q2_K 4 -> 6, the other three 5 -> 6. Crossovers 6.5-7.0 (Q2_K), 6.5-8.3 (Q4_1),
                //     6.9-8.6 (MXFP4), 7.1-9.0 (IQ4_NL); in each case ne11 = 7 wins on two of the
                //     three shapes but not on 5120x6144, so 6 is where the win is consistent.
                //     Representative shapes; no local GGUF has them, so no end-to-end check.
                return ne11 <= 6;
            default:
                // IQ4_XS, IQ3_S and IQ3_XXS moved 6 -> 8 here and fall through to this arm.
                //     MMVQ is faster at ne11 = 7 on all nine (type, shape) cells by 9.7-20.0 %,
                //     every CI clear of zero, and at ne11 = 8 on all nine by mean, resolved on
                //     seven. The fitted crossovers are 8.6-9.5, so 8 is MMVQ_MAX_BATCH_SIZE
                //     biting, not a measured meeting point -- there is no ncols_dst > 8 kernel.
                //     These three are 68.5 % of a 27B UD-IQ4_XS file's matmul weights and they
                //     all left the vector path in the same step at a verification width of 7,
                //     taking the file from 29.6 % to 98.1 % on MMQ, which is why a draft length
                //     of n_max = 6 measured 9 % SLOWER than 5 on the server before this change.
                // Q4_0 (6 -> 8), Q5_0 (6 -> 8) and IQ1_S (6 -> 8) also land here: crossovers
                //     8.4-10.6, 8.8-10.9 and 9.3-10.2 at the generic shapes, MMVQ faster at
                //     ne11 = 8 by 8.7 %, 8.9 % and 10.5 % on average. IQ2_XS (12.7) and IQ2_XXS
                //     (9.9) were re-measured and keep 8. IQ2_S was too: crossover 12.4. Q1_0 and
                //     Q2_0 are the only two types in this function never measured on gfx1151.
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

// Two adjacent output rows per block for the four-column RDNA3.5 kernels, mirroring what
// calc_rows_per_block does for the generic MMVQ kernel: a block owns rows row0 and row0+1 and
// reuses one load of the four q8_1 activation columns for both. The per-lane K decomposition and
// the accumulation order for a given (row, column) are unchanged, so results are bit-exact.
#ifndef MMVQ_Q4_COLUMNS_ROWS_PER_BLOCK
#define MMVQ_Q4_COLUMNS_ROWS_PER_BLOCK 2
#endif

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
    constexpr int rpb = MMVQ_Q4_COLUMNS_ROWS_PER_BLOCK;

    const int row0 = rpb * blockIdx.x;
    const int lane = threadIdx.x;
    const int column0 = 0;
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst = blockIdx.z;
    const uint32_t channel_x = fastdiv(channel_dst, channel_ratio);
    const uint32_t sample_x = fastdiv(sample_dst, sample_ratio);

    const void * vx = vx_ptr;
    const block_q8_1 * y = (const block_q8_1 *) vy_ptr + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    float * dst = dst_ptr + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_iter = vdr * warp_size / qi;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;
    float tmp[4][rpb] = {{0.0f}};

    ggml_cuda_pdl_sync();
    for (int kbx = lane / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (lane % (qi/vdr));
        int xv[rpb][VDR_Q8_0_Q8_1_MMVQ];
        float dx[rpb];
#pragma unroll
        for (int r = 0; r < rpb; ++r) {
            const block_q8_0 * bx = (const block_q8_0 *) vx + kbx_offset + r*stride_row_x + kbx;
#pragma unroll
            for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
                xv[r][i] = get_int_b2(bx->qs, kqs + i);
            }
            dx[r] = __half2float(bx->d);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const block_q8_1 * by = &y[(column0 + j)*stride_col_y + kby];
            const float d8 = __half2float(__low2half(by->ds));
            int u[VDR_Q8_0_Q8_1_MMVQ];
#pragma unroll
            for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
                u[i] = get_int_b4(by->qs, kqs + i);
            }
#pragma unroll
            for (int r = 0; r < rpb; ++r) {
                int sumi = 0;
#pragma unroll
                for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
                    sumi = ggml_cuda_dp4a(xv[r][i], u[i], sumi);
                }
                tmp[j][r] += dx[r] * d8 * (float) sumi;
            }
        }
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int r = 0; r < rpb; ++r) {
            tmp[j][r] = warp_reduce_sum<warp_size>(tmp[j][r]);
            if (lane == r && (rpb == 1 || uint32_t(row0 + r) < stride_col_dst)) {
                dst[(column0 + j)*stride_col_dst + r] = tmp[j][r];
            }
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
    constexpr int rpb       = MMVQ_Q4_COLUMNS_ROWS_PER_BLOCK;

    const int row0 = rpb * blockIdx.x;
    const int lane = threadIdx.x;
    const int tid  = warp_size*threadIdx.y + lane;
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);

    const block_q8_1 * y = (const block_q8_1 *) vy_ptr + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    float * dst = dst_ptr + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_iter  = vdr * nwarps*warp_size / qi;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;
    float tmp[4][rpb] = {{0.0f}};

    ggml_cuda_pdl_sync();
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));

        if constexpr (type == GGML_TYPE_Q1_0) {
            int   xv[rpb][8];
            float dx[rpb];
#pragma unroll
            for (int r = 0; r < rpb; ++r) {
                const block_q1_0 * bx = (const block_q1_0 *) vx_ptr + kbx_offset + r*stride_row_x + kbx;
                const int16_t * qs = (const int16_t *) bx->qs + kqs*2;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int q = qs[i];
                    const int n0 = __byte_perm(0x11100100, 0x11100100, q >> 0);
                    const int n1 = __byte_perm(0x11100100, 0x11100100, q >> 2);
                    const int s0 = __byte_perm(0x01FF, 0x01FF, n0 >>  0);
                    const int s1 = __byte_perm(0x01FF, 0x01FF, n1 >>  0);
                    const int s2 = __byte_perm(0x01FF, 0x01FF, n0 >> 16);
                    const int s3 = __byte_perm(0x01FF, 0x01FF, n1 >> 16);
                    xv[r][4*i+0] = __byte_perm(s0, s1, 0x5410);
                    xv[r][4*i+1] = __byte_perm(s0, s1, 0x7632);
                    xv[r][4*i+2] = __byte_perm(s2, s3, 0x5410);
                    xv[r][4*i+3] = __byte_perm(s2, s3, 0x7632);
                }
                dx[r] = bx->d;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const block_q8_1 * by = &y[j*stride_col_y + kby + kqs];
                const float d8 = __low2float(by->ds);
                int u[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    u[i] = get_int_b4(by->qs, i);
                }
#pragma unroll
                for (int r = 0; r < rpb; ++r) {
                    int sumi = 0;
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        sumi = ggml_cuda_dp4a(xv[r][4*i+0], u[i*4+0], sumi);
                        sumi = ggml_cuda_dp4a(xv[r][4*i+1], u[i*4+1], sumi);
                        sumi = ggml_cuda_dp4a(xv[r][4*i+2], u[i*4+2], sumi);
                        sumi = ggml_cuda_dp4a(xv[r][4*i+3], u[i*4+3], sumi);
                    }
                    tmp[j][r] += dx[r] * d8 * sumi;
                }
            }
        } else if constexpr (type == GGML_TYPE_Q2_0) {
            int   xv[rpb][4];
            int   xw[rpb][4];
            float dx[rpb];
#pragma unroll
            for (int r = 0; r < rpb; ++r) {
                const block_q2_0 * bx = (const block_q2_0 *) vx_ptr + kbx_offset + r*stride_row_x + kbx;
                const int16_t * qs = (const int16_t *) bx->qs + kqs*4;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int q = qs[i];
#if defined(GGML_USE_HIP)
                    const uint32_t qx_indices = (q & 0x03) | ((q & 0x0C) << 6) | ((q & 0x30) << 12) | ((q & 0xC0) << 18);
                    const uint32_t qy_bits    = q >> 8;
                    const uint32_t qy_indices = (qy_bits & 0x03) | ((qy_bits & 0x0C) << 6) | ((qy_bits & 0x30) << 12) | ((qy_bits & 0xC0) << 18);
                    xv[r][i] = __builtin_amdgcn_perm(0x020100FF, 0x020100FF, qx_indices);
                    xw[r][i] = __builtin_amdgcn_perm(0x020100FF, 0x020100FF, qy_indices);
#else
                    const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
                    const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
                    xv[r][i] = __byte_perm(qe, qo, 0x5140);
                    xw[r][i] = __byte_perm(qe, qo, 0x7362);
#endif // defined(GGML_USE_HIP)
                }
                dx[r] = bx->d;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const block_q8_1 * by = &y[j*stride_col_y + kby + kqs];
                const float d8 = __low2float(by->ds);
                int u[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    u[i] = get_int_b4(by->qs, i);
                }
#pragma unroll
                for (int r = 0; r < rpb; ++r) {
                    int sumi = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sumi = ggml_cuda_dp4a(u[i*2+0], xv[r][i], sumi);
                        sumi = ggml_cuda_dp4a(u[i*2+1], xw[r][i], sumi);
                    }
                    tmp[j][r] += dx[r] * d8 * sumi;
                }
            }
        } else if constexpr (type == GGML_TYPE_IQ3_S) {
            int2  xv[rpb][4];
            float dx[rpb];
#pragma unroll
            for (int r = 0; r < rpb; ++r) {
                const block_iq3_s * bx = (const block_iq3_s *) vx_ptr + kbx_offset + r*stride_row_x + kbx;
                const int2 qs_packed = make_int2(get_int_b2(bx->qs, kqs + 0), get_int_b2(bx->qs, kqs + 1));
                const uint8_t * qs = (const uint8_t *) &qs_packed;
                const int qh = bx->qh[kqs/2];
                const int signs_packed_32 = get_int_b2(bx->signs, kqs/2);
                const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int l0 = 2*i;
                    const int2 grid_pos = make_int2(
                        iq3s_grid[qs[l0 + 0] | ((qh << (8-l0)) & 0x100)],
                        iq3s_grid[qs[l0 + 1] | ((qh << (7-l0)) & 0x100)]);
                    xv[r][i] = make_int2(apply_signs4_nz(grid_pos.x, signs_packed_8[i]), apply_signs4_nz(grid_pos.y, signs_packed_8[i] >> 4));
                }
                // L-EPI (results/2026-09-10-lever-mmvq6/): the integer block scale is folded
                // into the float scale here exactly as vec_dot_iq3_s_q8_1 now does it, so the
                // four-column kernel and the generic kernel stay in agreement.
                const int ls = 1 + 2*((bx->scales[kqs/4] >> ((kqs << 1) & 0x04)) & 0x0F);
                dx[r] = __half2float(bx->d) * (float) ls;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const block_q8_1 * by = &y[j*stride_col_y + kby + kqs/2];
                const float d8 = __low2float(by->ds);
                int u[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    u[i] = get_int_b4(by->qs, i);
                }
#pragma unroll
                for (int r = 0; r < rpb; ++r) {
                    int sumi = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sumi = ggml_cuda_dp4a(xv[r][i].x, u[2*i + 0], sumi);
                        sumi = ggml_cuda_dp4a(xv[r][i].y, u[2*i + 1], sumi);
                    }
                    const float d = dx[r] * d8;
                    tmp[j][r] += d * (float) sumi;
                }
            }
        } else if constexpr (type == GGML_TYPE_Q5_1) {
            int   vl[rpb][VDR_Q5_1_Q8_1_MMVQ];
            int   vh[rpb][VDR_Q5_1_Q8_1_MMVQ];
            half2 dm[rpb];
#pragma unroll
            for (int r = 0; r < rpb; ++r) {
                const block_q5_1 * bx = (const block_q5_1 *) vx_ptr + kbx_offset + r*stride_row_x + kbx;
#pragma unroll
                for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
                    vl[r][i] = get_int_b4(bx->qs, kqs + i);
                    vh[r][i] = get_int_b4(bx->qh, 0) >> (4 * (kqs + i));
                }
                dm[r] = bx->dm;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const block_q8_1 * by = &y[j*stride_col_y + kby];
                int u[2*VDR_Q5_1_Q8_1_MMVQ];
#pragma unroll
                for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
                    u[2*i+0] = get_int_b4(by->qs, kqs + i);
                    u[2*i+1] = get_int_b4(by->qs, kqs + i + QI5_1);
                }
#pragma unroll
                for (int r = 0; r < rpb; ++r) {
                    // ROCm 7.14 changes gfx1151 Q5_1 rounding under four-column register pressure. Keep both paths materialized until LLVM preserves this expression.
                    volatile float dot = vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl[r], vh[r], u, dm[r], by->ds);
                    tmp[j][r] = __fadd_rn(tmp[j][r], dot);
                }
            }
        } else if constexpr (type == GGML_TYPE_Q4_K) {
            const int bq8_offset = QR4_K * ((kqs/2) / (QI8_1/2));
            const int is = bq8_offset/2;
            int      xv[rpb][2];
            uint16_t aux[rpb][2];
            half2    dm[rpb];
#pragma unroll
            for (int r = 0; r < rpb; ++r) {
                const block_q4_K * bx = (const block_q4_K *) vx_ptr + kbx_offset + r*stride_row_x + kbx;
                const int * q4 = (const int *) (bx->qs + 16*bq8_offset + 4*((kqs/2)%4));
                xv[r][0] = q4[0];
                xv[r][1] = q4[4];
                const uint16_t * scales = (const uint16_t *) bx->scales;
                if (is < 2) {
                    aux[r][0] = scales[is+0] & 0x3f3f;
                    aux[r][1] = scales[is+2] & 0x3f3f;
                } else {
                    aux[r][0] = ((scales[is+2] >> 0) & 0x0f0f) | ((scales[is-2] & 0xc0c0) >> 2);
                    aux[r][1] = ((scales[is+2] >> 4) & 0x0f0f) | ((scales[is-0] & 0xc0c0) >> 2);
                }
                dm[r] = bx->dm;
            }
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
#pragma unroll
                for (int r = 0; r < rpb; ++r) {
                    const uint8_t * sc = (const uint8_t *) aux[r];
                    const uint8_t * m  = sc + 2;
                    tmp[j][r] += vec_dot_q4_K_q8_1_impl_vmmq(xv[r], u, sc, m, dm[r], d8);
                }
            }
        } else if constexpr (type == GGML_TYPE_Q5_K) {
            const int bq8_offset = QR5_K * ((kqs/2) / (QI8_1/2));
            const int is = bq8_offset/2;
            int      vl[rpb][2];
            int      vh[rpb][2];
            uint16_t aux[rpb][2];
            half2    dm[rpb];
#pragma unroll
            for (int r = 0; r < rpb; ++r) {
                const block_q5_K * bx = (const block_q5_K *) vx_ptr + kbx_offset + r*stride_row_x + kbx;
                const int * ql = (const int *) (bx->qs + 16*bq8_offset + 4*((kqs/2)%4));
                const int * qh = (const int *) (bx->qh + 4*((kqs/2)%4));
                vl[r][0] = ql[0];
                vl[r][1] = ql[4];
                vh[r][0] = qh[0] >> bq8_offset;
                vh[r][1] = qh[4] >> bq8_offset;
                const uint16_t * scales = (const uint16_t *) bx->scales;
                if (is < 2) {
                    aux[r][0] = scales[is+0] & 0x3f3f;
                    aux[r][1] = scales[is+2] & 0x3f3f;
                } else {
                    aux[r][0] = ((scales[is+2] >> 0) & 0x0f0f) | ((scales[is-2] & 0xc0c0) >> 2);
                    aux[r][1] = ((scales[is+2] >> 4) & 0x0f0f) | ((scales[is-0] & 0xc0c0) >> 2);
                }
                dm[r] = bx->dm;
            }
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
#pragma unroll
                for (int r = 0; r < rpb; ++r) {
                    const uint8_t * sc = (const uint8_t *) aux[r];
                    const uint8_t * m  = sc + 2;
                    tmp[j][r] += vec_dot_q5_K_q8_1_impl_vmmq(vl[r], vh[r], u, sc, m, dm[r], d8);
                }
            }
        } else if constexpr (type == GGML_TYPE_Q6_K) {
            const int bq8_offset   = 2*QR6_K*(kqs/(QI6_K/2)) + (kqs%(QI6_K/2))/(QI6_K/4);
            const int scale_offset = (QI6_K/4)*(kqs/(QI6_K/2)) + (kqs%(QI6_K/2))/(QI6_K/8);
            const int vh_shift     = 2*((kqs%(QI6_K/2))/(QI6_K/4));
            int            vl[rpb];
            int            vh[rpb];
            const int8_t * scales[rpb];
            float          dx[rpb];
#pragma unroll
            for (int r = 0; r < rpb; ++r) {
                const block_q6_K * bx = (const block_q6_K *) vx_ptr + kbx_offset + r*stride_row_x + kbx;
                vl[r]     = get_int_b2(bx->ql, kqs);
                vh[r]     = get_int_b2(bx->qh, (QI6_K/4)*(kqs/(QI6_K/2)) + kqs%(QI6_K/4)) >> vh_shift;
                scales[r] = bx->scales + scale_offset;
                dx[r]     = bx->d;
            }
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
#pragma unroll
                for (int r = 0; r < rpb; ++r) {
                    tmp[j][r] += vec_dot_q6_K_q8_1_impl_mmvq(vl[r], vh[r], u, scales[r], dx[r], d8);
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][4][rpb][warp_size];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int r = 0; r < rpb; ++r) {
                tmp_shared[threadIdx.y-1][j][r][lane] = tmp[j][r];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int r = 0; r < rpb; ++r) {
#pragma unroll
            for (int i = 0; i < nwarps-1; ++i) {
                tmp[j][r] += tmp_shared[i][j][r][lane];
            }
            tmp[j][r] = warp_reduce_sum<warp_size>(tmp[j][r]);
            if (lane == r && (rpb == 1 || uint32_t(row0 + r) < stride_col_dst)) {
                dst[j*stride_col_dst + r] = tmp[j][r];
            }
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

        const int grid_l = apply_signs4_nz(grid_pos.x, signs_packed_8[l0/2]);
        const int grid_h = apply_signs4_nz(grid_pos.y, signs_packed_8[l0/2] >> 4);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    // L-EPI: kept in step with vec_dot_iq3_s_q8_1.
    const int ls = 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = (__half2float(bq3->d) * (float) ls) * __low2float(bq8_1[iqs/2].ds);
    return d * (float) sumi;
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

// RDNA3.5 only: the largest ncols_dst at which a block should own FOUR output rows instead of the
// two that shipped in 5bc829e0c. 0 means "never promote", i.e. the type keeps two rows everywhere.
//
// Source of the numbers: results/rocm-fa-depth/night43/05-round2.md, lever "R2B", which set four
// rows for EVERY type and every ncols_dst 2..8 and measured (a) the change in the per-column cost
// slope D on top of the shipped two-row stack and (b) the static VGPR count of
// mul_mat_vec_q<type,6> read out of the built object. Waves/SIMD below are derived from the VGPR
// count (gfx11 wave32, 1536 VGPR per SIMD, granule 8, hardware ceiling 16).
//
// R2B was NOT shipped as a blanket change because two of its cells are negative, and both of those
// are excluded here rather than averaged away. This table is the surviving subset.
static constexpr __host__ __device__ int rdna3_5_rows4_max_ncols_dst(ggml_type type) {
    switch (type) {
        // Codebook / IQ types: D fell 13.2-28.6 % and the register growth still leaves 7-12 waves
        //     per SIMD. D: iq1_m -28.6, iq2_xs -20.7, iq2_s -19.8, iq1_s -18.1, iq2_xxs -17.8,
        //     iq4_xs -13.6, iq3_s -13.2. VGPR at ncols_dst=6, two rows -> four rows:
        //     127->199, 105->155, 105->169, 91->118, 104->151, 116->164, 103->166. None spills.
        // Capped at 6 because the ncols_dst 7 and 8 kernels are where R2B's register cliff sits
        //     (see the default arm) and nothing in that sweep could price it.
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
            return 6;

        // Q2_K: D fell 18.8 %, the largest gain outside the IQ set, but it is also the type with
        //     the worst register growth -- 124->252 VGPR at ncols_dst=6, i.e. 12->6 waves per SIMD,
        //     and mul_mat_vec_q<q2_K,8> goes to 256 VGPR with 20 spills / 84 B of scratch.
        // Bounded at 4, not 6, because 4 is also ggml_cuda_should_use_mmvq's Q2_K threshold on this
        //     arch: ncols_dst 5..8 for Q2_K is reachable only through the ne01 < 64 GDN path, which
        //     the R2B sweep contains no case for. The measured -18.8 % comes entirely from
        //     ncols_dst <= 4, so that is exactly how far it is applied.
        case GGML_TYPE_Q2_K:
            return 4;

        // Deliberate negatives, listed so the measurement that excludes them is on the record:
        //   q5_1  REGRESSED +30.7 % at ncols_dst=2, reproduced on two builds. Never promote.
        //   q2_0  D rose 3.2 % (slower), so there is nothing to buy.
        //   q1_0  D fell only 4.9 %, smaller than the 2.55 pp cross-sweep RMS of that experiment,
        //         and it costs 16->11 waves per SIMD. Not established, so not taken.
        //   q8_0  the two-point D fit was unusable (13.37 -> 15.30, i.e. pointing the wrong way).
        //         It is also the one type with a dedicated q4-columns kernel at ncols_dst=4, so the
        //         generic kernel only sees it at 2 and 3 below its threshold of 4.
        //   NVFP4 mul_mat_vec_q<NVFP4,7> and <NVFP4,8> spill 24 and 68 registers (100 B / 276 B of
        //         scratch) under R2B and no case in that sweep reaches them.
        // Every type not named above -- k-quants other than Q2_K, Q4_0/Q4_1/Q5_0, MXFP4, IQ3_XXS,
        //     IQ4_NL -- was not moved by R2B outside its noise band and keeps two rows as today.
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_NVFP4:
        default:
            return 0;
    }
}

static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_RDNA3_5) {
        // nwarps is 1 here, so a block is a single wave and every lane re-reads the whole q8_1
        // activation block for the row it owns. Two rows per block share those loads; the types in
        // rdna3_5_rows4_max_ncols_dst share them four ways.
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return ncols_dst <= rdna3_5_rows4_max_ncols_dst(type) ? 4 : 2;
            default:
                return 1;
        }
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

// The promoted cells, one per justification group.
static_assert(calc_rows_per_block(GGML_TYPE_IQ1_M,   2, MMVQ_PARAMETERS_RDNA3_5) == 4);
static_assert(calc_rows_per_block(GGML_TYPE_IQ1_M,   6, MMVQ_PARAMETERS_RDNA3_5) == 4);
static_assert(calc_rows_per_block(GGML_TYPE_IQ4_XS,  6, MMVQ_PARAMETERS_RDNA3_5) == 4);
static_assert(calc_rows_per_block(GGML_TYPE_Q2_K,    4, MMVQ_PARAMETERS_RDNA3_5) == 4);

// The cells the table deliberately refuses.
static_assert(calc_rows_per_block(GGML_TYPE_IQ1_M,   7, MMVQ_PARAMETERS_RDNA3_5) == 2);  // spill risk unpriced at 7/8
static_assert(calc_rows_per_block(GGML_TYPE_IQ1_M,   8, MMVQ_PARAMETERS_RDNA3_5) == 2);
static_assert(calc_rows_per_block(GGML_TYPE_Q2_K,    5, MMVQ_PARAMETERS_RDNA3_5) == 2);  // above Q2_K's MMVQ threshold
static_assert(calc_rows_per_block(GGML_TYPE_Q2_K,    8, MMVQ_PARAMETERS_RDNA3_5) == 2);  // 20 spills under R2B
static_assert(calc_rows_per_block(GGML_TYPE_Q5_1,    2, MMVQ_PARAMETERS_RDNA3_5) == 2);  // +30.7 % regression
static_assert(calc_rows_per_block(GGML_TYPE_Q2_0,    4, MMVQ_PARAMETERS_RDNA3_5) == 2);
static_assert(calc_rows_per_block(GGML_TYPE_Q8_0,    2, MMVQ_PARAMETERS_RDNA3_5) == 2);
static_assert(calc_rows_per_block(GGML_TYPE_NVFP4,   7, MMVQ_PARAMETERS_RDNA3_5) == 2);  // 24 spills under R2B
static_assert(calc_rows_per_block(GGML_TYPE_Q4_K,    6, MMVQ_PARAMETERS_RDNA3_5) == 2);  // never listed
static_assert(calc_rows_per_block(GGML_TYPE_IQ1_M,   1, MMVQ_PARAMETERS_RDNA3_5) == 1);
static_assert(calc_rows_per_block(GGML_TYPE_IQ1_M,   9, MMVQ_PARAMETERS_RDNA3_5) == 1);

// Every other parameter table must be type-independent and bit-identical to what shipped: the type
// argument is only ever read inside the RDNA3.5 arm.
static_assert(calc_rows_per_block(GGML_TYPE_IQ1_M,   6, MMVQ_PARAMETERS_GENERIC) == 2);
static_assert(calc_rows_per_block(GGML_TYPE_Q5_1,    6, MMVQ_PARAMETERS_GENERIC) == 2);
static_assert(calc_rows_per_block(GGML_TYPE_IQ1_M,   1, MMVQ_PARAMETERS_GENERIC) == 1);
static_assert(calc_rows_per_block(GGML_TYPE_IQ1_M,   1, MMVQ_PARAMETERS_GENERIC, true, 4) == 4);
static_assert(calc_rows_per_block(GGML_TYPE_IQ1_M,   6, MMVQ_PARAMETERS_GCN) == 2);
static_assert(calc_rows_per_block(GGML_TYPE_IQ1_M,   6, MMVQ_PARAMETERS_TURING) == 2);
static_assert(calc_rows_per_block(GGML_TYPE_IQ1_M,   6, MMVQ_PARAMETERS_GB10) == 2);
static_assert(calc_rows_per_block(GGML_TYPE_IQ1_M,   6, MMVQ_PARAMETERS_RDNA4) == 1);
static_assert(calc_rows_per_block(GGML_TYPE_IQ1_M,   6, MMVQ_PARAMETERS_RDNA3_0) == 1);
static_assert(calc_rows_per_block(GGML_TYPE_IQ1_M,   6, MMVQ_PARAMETERS_RDNA2) == 1);

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
    constexpr int rows_per_cuda_block = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    // The epilogue writes row row0 + i from lane i, so a block can never own more rows than a wave
    // has lanes. Two rows never came close; the RDNA3.5 four-row table makes this worth pinning.
    static_assert(rows_per_cuda_block <= warp_size);

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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            constexpr int pf_dist = 2; // loop iterations, not blocks
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    mmvq_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            mmvq_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

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
        if (c1 < blocks_per_row_x) {
            cx = nx;
            if constexpr (has_fusion) {
                if (use_gate) {
                    cg = ng;
                }
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
// Q8_0 matvecs are launched as one kernel. Each segment is a plain [ncols x nrows] weight matrix with an optional
// Q8_0 gate and GLU epilogue. Blocks are assigned to segments by a prefix over the segment block counts.

#define MMVQ_GROUP_MAX 4

struct mmvq_group_seg_dev {
    const void * vx;
    const void * gate;
    float      * dst;
    int          nrows;
    int          stride_row;      // in Q8_0 blocks
    int          gate_stride_row; // in blocks
    int          blocks_begin;
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
        if (c1 < blocks_per_row_x) {
            cx = nx;
            if (use_gate) {
                cg = ng;
            }
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
        total_q8_rows += segs[i].w->ne[1];
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

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
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

    // fuse gate, bias, scales, and glu_op into the up projection
    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    const float * x_scale    = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
    }

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
    float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + threadIdx.x;

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (x_scale) {
                    result *= x_scale[channel_x];
                }
            }
            if (x_bias) {
                result += x_bias[bias_idx];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (gate_scale) {
                        gate_value *= gate_scale[channel_x];
                    }
                }
                if (gate_bias) {
                    gate_value += gate_bias[bias_idx];
                }
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
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale);
    } else if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(x_scale, gate_scale);
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
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
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
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

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;

    if (has_fusion) {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    }
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
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
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
                    constexpr int c_rpb    = MMVQ_Q4_COLUMNS_ROWS_PER_BLOCK;
                    const dim3 block_nums((nrows_x + c_rpb - 1) / c_rpb, nchannels_dst, nsamples_dst);
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
                    constexpr int c_rpb = MMVQ_Q4_COLUMNS_ROWS_PER_BLOCK;
                    const dim3 block_nums((nrows_x + c_rpb - 1) / c_rpb, nchannels_dst, nsamples_dst);
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
    const bool fusion_ok = !fusion ||
        (fusion->x_scale == nullptr && fusion->gate_scale == nullptr &&
         (!ids || (fusion->x_bias == nullptr && fusion->gate_bias == nullptr)));
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
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT( !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc));
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
