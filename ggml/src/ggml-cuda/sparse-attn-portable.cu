// sa_attend_portable: the portable tier of GGML_OP_SPARSE_ATTN on HIP (PLAN-v3 §5 "HIP tier selection per arch", §6.2
// "sparse-attn-portable.cu").
//
// One kernel for every HIP arch: no WMMA / MFMA, no v_permlanex16, no cross-lane operation (only __syncthreads and
// shared memory, so neither the wave size, 32 or 64, nor the arch is assumed), every K / V type of S converted to its
// contract f16 value by the runtime switch sa_any_elem (sparse-attn-loaders.cuh). It is what every HIP arch other than
// RDNA3.5 runs (the WMMA tiers are selected on GGML_CUDA_CC_IS_RDNA3_5 only), and what GGML_CUDA_SA_FORCE=portable
// selects on RDNA3.5 (the tests of PLAN-v3 §13 W5). Its speed on other archs is UNPROVEN (U15).
//
// Arithmetic: the CPU op's (ggml-cpu/sparse-attn.cpp, O_CON), cell by cell, with FP contraction off in the kernel (the CPU
// TU is built with -ffp-contract=off):
//   q f16(q * scale) (f32 product, one rounding); K / V elements f16 (sa_any_elem); each score an f32 sum over the 256
//   dims in sequential order; per cell, in the plan's ascending cell order: if s > m { alpha = (m == -inf) ? 0 :
//   expf(m - s); acc *= alpha; l *= alpha; m = s }; p = expf(s - m); l += p; acc += f16(p) * v; out = l > 0 ? acc / l : 0.
//   The only arithmetic difference from the CPU op is the device expf (ocml) against the host's.
//   Rotations (PLAN-v3 §8, D13): H_q of q (f32, before the product with scale) and H_o of acc / l through shared memory,
//   sa_fwht_smem (sparse-attn-fwht.cuh: the arithmetic of fwht.cu and of the CPU op's sa_fwht).
// Masked-off and padding cells are never loaded (the cells are collected from this query's mask bits), so their bytes
// cannot reach the output (contract item 4).
//
// Work: block (stream, query, kv head), 256 threads, thread d owns dim d. The group's plan entries are read in batches of
// 64; the cells this query attends are compacted in order into shared memory; each tile of 16 cells is staged (K as f32 of
// the f16 value), the 16 x GQA scores are computed one per thread, thread h < GQA runs the online softmax of head h over
// the tile's cells, and every thread applies the per-cell rescale and f16(p) * v to its GQA accumulators. Shared memory:
// 12 x 257 + 16 x 257 floats + 3 x 16 x 12 floats + small = about 32 KiB.

#include "sparse-attn.cuh"
#include "sparse-attn-census.h"

#include <climits>

#if defined(GGML_USE_HIP)

#include "sparse-attn-loaders.cuh"
#include "sparse-attn-fwht.cuh"

#define SA_P_D     256   // head size (supports_op)
#define SA_P_TILE  16    // cells per staged tile
#define SA_P_BATCH 64    // plan entries per batch (<= 4 cells each)
#define SA_P_RMAX  12    // largest GQA (supports_op: 4, 8, 12)

struct sa_portable_args {
    const char *     q;         // F32 [256, n_q, n_head, n_stream]
    size_t           q_nb1, q_nb2, q_nb3;
    const char *     k;         // [256, cells, n_head_kv, n_stream], ggml type kt_g
    size_t           k_nb1, k_nb2, k_nb3;
    const char *     v;         // ggml type vt_g
    size_t           v_nb1, v_nb2, v_nb3;
    const uint32_t * counts;    // plan sections (ggml.h layout)
    const uint32_t * quads;
    const uint16_t * masks;
    int              n_groups, cap, G, n_q, n_head, n_head_kv, gqa, kt_g, vt_g;
    float            scale;
    int              rot_q, rot_o;              // H_q / H_o block sizes (0, 64, 128, 256)
    float            rot_q_scale, rot_o_scale;  // 1.0f / sqrtf(rot), as fwht.cu
    int              rot_dump;                  // test hook: H_q q into out, nothing else
    float *          out;       // dst F32 [256, n_head, n_q, n_stream]
};

static __global__ void __launch_bounds__(256) sa_attend_portable(const sa_portable_args a) {
#pragma clang fp contract(off)
    const int d  = threadIdx.x;
    const int bq = blockIdx.x / a.n_head_kv, kvh = blockIdx.x - bq * a.n_head_kv;
    const int s  = bq / a.n_q, i = bq - s * a.n_q;
    const int g  = i / a.G, jq = i - g * a.G;   // group, and the query's index in it (mask bits 4 * jq .. 4 * jq + 3)
    const int r  = a.gqa;

    __shared__ float qs[SA_P_RMAX][SA_P_D + 1];
    __shared__ float ks[SA_P_TILE][SA_P_D + 1];
    __shared__ float sc[SA_P_TILE][SA_P_RMAX];
    __shared__ float al[SA_P_TILE][SA_P_RMAX];   // the rescale of the accumulator before cell c (1 when the max is kept)
    __shared__ float pw[SA_P_TILE][SA_P_RMAX];   // f16(p) of cell c
    __shared__ float m_s[SA_P_RMAX], l_s[SA_P_RMAX];
    __shared__ int   cnt[SA_P_BATCH + 1];
    __shared__ int   cells[4 * SA_P_BATCH];

    if (a.rot_q != 0) {   // uniform over the grid
        for (int h = 0; h < r; ++h) {
            qs[h][d] = *reinterpret_cast<const float *>(a.q + (size_t) s * a.q_nb3 + (size_t) i * a.q_nb1 +
                                                        (size_t) (kvh * r + h) * a.q_nb2 + (size_t) d * sizeof(float));
        }
        sa_fwht_smem<SA_P_D>(&qs[0][0], r, SA_P_D + 1, a.rot_q, a.rot_q_scale, d, SA_P_D);
        if (a.rot_dump) {
            for (int h = 0; h < r; ++h) {
                a.out[(((size_t) s * a.n_q + i) * a.n_head + (size_t) (kvh * r + h)) * SA_P_D + d] = qs[h][d];
            }
            return;
        }
        for (int h = 0; h < r; ++h) {
            qs[h][d] = sa_h2f_bits(sa_f2h(qs[h][d] * a.scale));
        }
    } else {
        for (int h = 0; h < r; ++h) {
            const float x = *reinterpret_cast<const float *>(a.q + (size_t) s * a.q_nb3 + (size_t) i * a.q_nb1 +
                                                             (size_t) (kvh * r + h) * a.q_nb2 + (size_t) d * sizeof(float));
            qs[h][d] = sa_h2f_bits(sa_f2h(x * a.scale));
        }
    }
    if (d < SA_P_RMAX) {
        m_s[d] = -INFINITY;
        l_s[d] = 0.0f;
    }
    float acc[SA_P_RMAX];
#pragma unroll
    for (int h = 0; h < SA_P_RMAX; ++h) {
        acc[h] = 0.0f;
    }

    const size_t     ent   = (size_t) s * a.n_groups + g;
    const uint32_t * gq    = a.quads + ent * a.cap;
    const uint16_t * gm    = a.masks + ent * a.cap;
    const int        n_ent = (int) a.counts[ent];
    const char *     kb    = a.k + (size_t) s * a.k_nb3 + (size_t) kvh * a.k_nb2;
    const char *     vb    = a.v + (size_t) s * a.v_nb3 + (size_t) kvh * a.v_nb2;
    __syncthreads();

    for (int e0 = 0; e0 < n_ent; e0 += SA_P_BATCH) {
        // this query's cells of entries e0 .. e0 + 63, compacted in plan order (ascending cells)
        uint32_t bits = 0, quad = 0;
        if (d < SA_P_BATCH && e0 + d < n_ent) {
            bits = ((uint32_t) gm[e0 + d] >> (4 * jq)) & 0xfu;
            quad = gq[e0 + d];
        }
        if (d < SA_P_BATCH) {
            cnt[d] = __popc(bits);
        }
        __syncthreads();
        if (d < SA_P_BATCH) {
            int off = 0;
            for (int t = 0; t < d; ++t) {
                off += cnt[t];
            }
            for (int c = 0; c < 4; ++c) {
                if ((bits >> c) & 1u) {
                    cells[off++] = (int) (4 * quad + c);
                }
            }
            if (d == SA_P_BATCH - 1) {
                cnt[SA_P_BATCH] = off;
            }
        }
        __syncthreads();
        const int n_c = cnt[SA_P_BATCH];

        for (int t0 = 0; t0 < n_c; t0 += SA_P_TILE) {
            const int nt = min(SA_P_TILE, n_c - t0);
            // K of the tile: element d of each cell, its contract f16 value
            for (int c = 0; c < nt; ++c) {
                ks[c][d] = sa_h2f_bits(sa_any_elem(a.kt_g, kb + (size_t) cells[t0 + c] * a.k_nb1, d));
            }
            __syncthreads();
            // scores: thread (cell c, head h), an f32 sum over the 256 dims in sequential order
            if (d < SA_P_TILE * r) {
                const int c = d % SA_P_TILE, h = d / SA_P_TILE;
                if (c < nt) {
                    float x = 0.0f;
                    for (int e = 0; e < SA_P_D; ++e) {
                        x += qs[h][e] * ks[c][e];
                    }
                    sc[c][h] = x;
                }
            }
            __syncthreads();
            // online softmax of head h over the tile's cells, one cell at a time (the CPU op's order and branches)
            if (d < r) {
                float mx = m_s[d], l = l_s[d];
                for (int c = 0; c < nt; ++c) {
                    const float x = sc[c][d];
                    float alpha = 1.0f;
                    if (x > mx) {
                        alpha = mx == -INFINITY ? 0.0f : expf(mx - x);
                        l    *= alpha;
                        mx    = x;
                    }
                    const float p = expf(x - mx);
                    l += p;
                    al[c][d] = alpha;
                    pw[c][d] = sa_h2f_bits(sa_f2h(p));
                }
                m_s[d] = mx;
                l_s[d] = l;
            }
            __syncthreads();
            // PV: acc *= alpha (exact when alpha = 1), then acc += f16(p) * v
            for (int c = 0; c < nt; ++c) {
                const float vv = sa_h2f_bits(sa_any_elem(a.vt_g, vb + (size_t) cells[t0 + c] * a.v_nb1, d));
#pragma unroll
                for (int h = 0; h < SA_P_RMAX; ++h) {
                    if (h < r) {
                        acc[h] = acc[h] * al[c][h];
                        acc[h] = acc[h] + pw[c][h] * vv;
                    }
                }
            }
            __syncthreads();   // ks, sc, al, pw are rewritten by the next tile
        }
        __syncthreads();       // cnt and cells are rewritten by the next batch
    }

    if (a.rot_o != 0) {   // uniform over the grid; qs is free (the last tile ended on a barrier)
#pragma unroll
        for (int h = 0; h < SA_P_RMAX; ++h) {
            if (h < r) {
                const float l = l_s[h];
                qs[h][d] = l > 0.0f ? acc[h] / l : 0.0f;
            }
        }
        sa_fwht_smem<SA_P_D>(&qs[0][0], r, SA_P_D + 1, a.rot_o, a.rot_o_scale, d, SA_P_D);
        for (int h = 0; h < r; ++h) {
            a.out[(((size_t) s * a.n_q + i) * a.n_head + (size_t) (kvh * r + h)) * SA_P_D + d] = qs[h][d];
        }
        return;
    }
#pragma unroll
    for (int h = 0; h < SA_P_RMAX; ++h) {
        if (h < r) {
            const float l = l_s[h];
            a.out[(((size_t) s * a.n_q + i) * a.n_head + (size_t) (kvh * r + h)) * SA_P_D + d] = l > 0.0f ? acc[h] / l : 0.0f;
        }
    }
}

void ggml_cuda_sa_portable_launch(cudaStream_t stream, const ggml_cuda_sa_portable_launch_params & p) {
    GGML_ASSERT(p.gqa == 4 || p.gqa == 8 || p.gqa == 12);
    GGML_ASSERT(p.G == 1 || p.G == 4);
    GGML_ASSERT((int64_t) p.n_q * p.n_stream * p.n_head_kv <= INT_MAX);
    sa_portable_args a = {};
    a.q = p.q; a.q_nb1 = p.q_nb[1]; a.q_nb2 = p.q_nb[2]; a.q_nb3 = p.q_nb[3];
    a.k = p.k; a.k_nb1 = p.k_nb[1]; a.k_nb2 = p.k_nb[2]; a.k_nb3 = p.k_nb[3];
    a.v = p.v; a.v_nb1 = p.v_nb[1]; a.v_nb2 = p.v_nb[2]; a.v_nb3 = p.v_nb[3];
    a.counts    = p.counts;
    a.quads     = p.quads;
    a.masks     = p.masks;
    a.n_groups  = p.n_groups;
    a.cap       = p.cap;
    a.G         = p.G;
    a.n_q       = p.n_q;
    a.n_head    = p.n_head;
    a.n_head_kv = p.n_head_kv;
    a.gqa       = p.gqa;
    a.kt_g      = p.kt_g;
    a.vt_g      = p.vt_g;
    a.scale     = p.scale;
    a.rot_q       = p.rot_q;
    a.rot_o       = p.rot_o;
    a.rot_q_scale = ggml_cuda_sa_rot_scale(p.rot_q);
    a.rot_o_scale = ggml_cuda_sa_rot_scale(p.rot_o);
    a.rot_dump    = p.rot_dump;
    a.out       = p.out;
    const unsigned blocks = (unsigned) ((int64_t) p.n_q * p.n_stream * p.n_head_kv);
    sa_attend_portable<<<blocks, 256, 0, stream>>>(a);
    ggml_cuda_sa_census_add(p.G == 1 ? GGML_CUDA_SA_CENSUS_PORTABLE_G1 : GGML_CUDA_SA_CENSUS_PORTABLE_G4);
}

#endif // defined(GGML_USE_HIP)
