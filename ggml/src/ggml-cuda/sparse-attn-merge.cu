// sa_merge_splits: the deterministic split-K merge of the G1 sparse attend (PLAN-v3 §6.2 "sparse-attn-merge.cu").
//
// One block of 256 threads per output row R = (s * n_q + i) * n_head + h; thread d owns dim d. The partial rows of R
// (sa_attend_g1 with splits > 1: [splits][258] = 256 unnormalised accumulators, the split's running max m_z and its
// normaliser l_z) are combined in split order, so the result is a pure function of the partials:
//   M = max_z m_z;  f_z = exp(m_z - M) for the splits with l_z > 0;  out = sum_z f_z acc_z / sum_z f_z l_z  (0 if empty).
// This is PR #219's qsa_decode_merge arithmetic (qsa-decode.cu:65-79: f32, expf, the same order), with the splits whose
// normaliser is 0 skipped by a select instead of multiplied by a 0 factor (their accumulators are 0 anyway: an empty
// split writes zeros). The rot_o epilogue of PLAN-v3 §6.2 lands with the in-op rotation (W10): supports_op rejects
// rot_o != 0 until then.

#include "sparse-attn.cuh"
#include "sparse-attn-census.h"

#include <climits>

#if defined(GGML_USE_HIP)

static __global__ void __launch_bounds__(256) sa_merge_splits(const float * partial, float * out, const int splits) {
    const size_t row = blockIdx.x;
    const int    d   = threadIdx.x;
    const float * src = partial + row * (size_t) splits * 258;
    float maximum = -INFINITY;
    for (int z = 0; z < splits; ++z) {
        maximum = fmaxf(maximum, src[z * 258 + 256]);
    }
    float sum = 0.0f, value = 0.0f;
    for (int z = 0; z < splits; ++z) {
        const float count = src[z * 258 + 257];
        if (count > 0.0f) {
            const float factor = expf(src[z * 258 + 256] - maximum);
            value += src[z * 258 + d] * factor;
            sum   += count * factor;
        }
    }
    out[row * 256 + d] = sum > 0.0f ? value / sum : 0.0f;
}

void ggml_cuda_sa_merge_splits(cudaStream_t stream, const float * partial, float * out, int64_t n_rows, int splits) {
    GGML_ASSERT(n_rows > 0 && n_rows <= INT_MAX && splits > 1);
    sa_merge_splits<<<(unsigned) n_rows, 256, 0, stream>>>(partial, out, splits);
    ggml_cuda_sa_census_add(GGML_CUDA_SA_CENSUS_MERGE);
}

#endif // defined(GGML_USE_HIP)
