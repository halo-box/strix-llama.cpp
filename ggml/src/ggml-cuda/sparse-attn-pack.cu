// sa_pack_v<T>: V of an S_fast type to the packed f16 layout the G4 attend reads with SA_V_PACKED (PLAN-v3 §6.2, D8).
//
// Layout: f16 [n_stream][n_head_kv][n_quads][256 dims][4 cells], the PR #219 tree's qsa_pack_v indexing
// (qsa-prefill.cu: element (quad, dim) at quad * 1024 + dim * 4) with the stream and the kv head as outer dimensions, so
// one uint2 load gives the 4 cells of an entry for one dim. Every cell 0 .. 4 * n_quads - 1 is converted (the pack is
// O(n_kv), independent of the selection), with the conversions of sparse-attn-q8.cuh, so packed and direct V deliver the
// same halves (non-finite halves excepted: the pack writes them as 0 and flags their quad for the exact fixup). The destination is the attend dst over-allocation (ggml_cuda_sparse_attn_get_alloc_size), never the pool
// (D12).

#include "sparse-attn.cuh"
#include "sparse-attn-census.h"

#if defined(GGML_USE_HIP)

#include "sparse-attn-loaders.cuh"

static constexpr int SA_PACK_THREADS = 256;

// one thread per (stream, kv head, quad, dim): 4 cells of one dim -> one ushort4; one block per (stream, kv head, quad)
// (256 threads = the 256 dims). A non-finite half (stored inf / NaN, NaN scale, f16 overflow) is written as 0, and the
// block writes the quad's flag (0 or 1, every quad written: no clearing needed) and, when set, ORs 1 into *any (cleared
// before the launch): sa_attend_fixup recomputes the groups that touch a flagged quad (sparse-attn-attend-g4.cuh)
template <int T>
static __global__ void __launch_bounds__(SA_PACK_THREADS) sa_pack_v(
        const char * __restrict__ v, uint16_t * __restrict__ dst, const size_t nb1, const size_t nb2, const size_t nb3,
        const int n_head_kv, const uint32_t n_quads, const uint64_t count, uint8_t * __restrict__ qflags,
        uint32_t * __restrict__ any) {
    static_assert(SA_PACK_THREADS == 256, "one block per quad: the 256 dims of a row");
    const uint64_t i = (uint64_t) blockIdx.x * SA_PACK_THREADS + threadIdx.x;
    if (blockIdx.x * (uint64_t) SA_PACK_THREADS >= count) {
        return;
    }
    const uint32_t d    = (uint32_t) (i % 256);
    const uint64_t blk  = i / 256;                      // (s * n_head_kv + kvh) * n_quads + quad
    const uint32_t quad = (uint32_t) (blk % n_quads);
    const uint64_t sh   = blk / n_quads;
    const uint32_t kvh  = (uint32_t) (sh % (uint64_t) n_head_kv);
    const uint32_t s    = (uint32_t) (sh / (uint64_t) n_head_kv);
    const char * in = v + (size_t) s * nb3 + (size_t) kvh * nb2 + (size_t) (4 * quad) * nb1;
    const int w = (int) (d / 32), dd = (int) (d % 32);
    uint32_t h[4];
    int bad = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t g = sa_gather_load<T>(in + (size_t) j * nb1, w, dd);
        h[j] = sa_gather_pair<T>(g, g) & 0xffffu;
        if ((h[j] & 0x7c00u) == 0x7c00u) {
            h[j] = 0;
            bad = 1;
        }
    }
    *reinterpret_cast<uint2 *>(dst + blk * 1024 + d * 4) = make_uint2(h[0] | (h[1] << 16), h[2] | (h[3] << 16));
    bad = __syncthreads_or(bad);
    if (threadIdx.x == 0) {
        qflags[blk] = bad ? 1 : 0;
        if (bad) {
            atomicOr(any, 1u);
        }
    }
}

void ggml_cuda_sparse_attn_pack_v(cudaStream_t stream, const ggml_tensor * v, const int64_t n_quads, uint16_t * dst,
                                  uint8_t * qflags, uint32_t * any) {
    GGML_ASSERT(v->ne[0] == 256 && n_quads >= 1 && n_quads <= ((int64_t) 1 << 22));
    GGML_ASSERT(4 * n_quads <= v->ne[1]);
    const uint64_t count  = (uint64_t) n_quads * 256 * (uint64_t) v->ne[2] * (uint64_t) v->ne[3];
    const uint64_t blocks = (count + SA_PACK_THREADS - 1) / SA_PACK_THREADS;
    GGML_ASSERT(blocks <= (uint64_t) INT32_MAX);
    const char * src = (const char *) v->data;
    CUDA_CHECK(cudaMemsetAsync(any, 0, sizeof(uint32_t), stream));
    const int nhkv = (int) v->ne[2];
    switch (v->type) {
#define SA_PACK_CASE(gt, tid, key)                                                                                         \
        case gt:                                                                                                           \
            sa_pack_v<tid><<<(unsigned) blocks, SA_PACK_THREADS, 0, stream>>>(src, dst, v->nb[1], v->nb[2], v->nb[3], nhkv, \
                                                                              (uint32_t) n_quads, count, qflags, any);     \
            ggml_cuda_sa_census_add(key);                                                                                  \
            break;
        SA_PACK_CASE(GGML_TYPE_F16,  SA_T_F16,  GGML_CUDA_SA_CENSUS_PACK_f16)
        SA_PACK_CASE(GGML_TYPE_BF16, SA_T_BF16, GGML_CUDA_SA_CENSUS_PACK_bf16)
        SA_PACK_CASE(GGML_TYPE_Q8_0, SA_T_Q8_0, GGML_CUDA_SA_CENSUS_PACK_q8_0)
        SA_PACK_CASE(GGML_TYPE_Q4_0, SA_T_Q4_0, GGML_CUDA_SA_CENSUS_PACK_q4_0)
#undef SA_PACK_CASE
        default:
            GGML_ABORT("SPARSE_ATTN: no V pack kernel for %s", ggml_type_name(v->type));
    }
}

#endif // defined(GGML_USE_HIP)
