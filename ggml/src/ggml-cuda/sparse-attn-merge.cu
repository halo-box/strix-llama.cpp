// sa_merge_splits: deterministic split-K merge of the G1 attend. One block per output row R, thread d owns dim d; the
// [splits][258] partials (256 accumulators, running max m_z, normaliser l_z) are combined in split order:
//   M = max_z m_z;  f_z = exp(m_z - M) for l_z != 0;  out = sum_z f_z acc_z / sum_z f_z l_z  (0 unless > 0).
// A NaN l_z is kept so the row is 0, as in the CPU op; skipping it would make the result depend on the split count.
// With rot_o != 0, H_o is applied to the merged row in shared memory (the partials are unrotated).

#include "sparse-attn.cuh"
#include "sparse-attn-census.h"
#include "sparse-attn-fwht.cuh"

#include <climits>

#if defined(GGML_USE_HIP)

static __global__ void __launch_bounds__(256) sa_merge_splits(const float * partial, float * out, const int splits,
                                                               const int rot_o, const float rot_o_scale) {
    __shared__ float rowbuf[256];
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
        if (count != 0.0f) {   // true for NaN: a poisoned split makes the row 0
            const float factor = expf(src[z * 258 + 256] - maximum);
            value += src[z * 258 + d] * factor;
            sum   += count * factor;
        }
    }
    float o = sum > 0.0f ? value / sum : 0.0f;
    if (rot_o != 0) {   // uniform over the grid
        rowbuf[d] = o;
        sa_fwht_smem<256>(rowbuf, 1, 256, rot_o, rot_o_scale, d, 256);
        o = rowbuf[d];
    }
    out[row * 256 + d] = o;
}

void ggml_cuda_sa_merge_splits(cudaStream_t stream, const float * partial, float * out, int64_t n_rows, int splits, int rot_o) {
    GGML_ASSERT(n_rows > 0 && n_rows <= INT_MAX && splits > 1);
    GGML_ASSERT(rot_o == 0 || rot_o == 64 || rot_o == 128 || rot_o == 256);
    const float rot_o_scale = ggml_cuda_sa_rot_scale(rot_o);
    sa_merge_splits<<<(unsigned) n_rows, 256, 0, stream>>>(partial, out, splits, rot_o, rot_o_scale);
    ggml_cuda_sa_census_add(GGML_CUDA_SA_CENSUS_MERGE);
}

#endif // defined(GGML_USE_HIP)
