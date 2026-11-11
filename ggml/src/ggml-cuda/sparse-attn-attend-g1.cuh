#pragma once

// sa_attend_g1<KT, VT, GQA, ROT>: GGML_OP_SPARSE_ATTN with a G = 1 plan on RDNA3 WMMA. Block = one query and one kv head: rows are its GQA query heads (padded to 16 with zero q), grid.x = (stream, query, kv head) with kv head fastest, grid.y = split. Per 16-cell chunk: QK WMMA, online softmax in wave 0, PV WMMA. V is LDS-staged and transposed through a wave-private 1 KB slot; K is direct per lane for f16 / q8_0, split for q4_0 / bf16 / the generic tier.
//   - cells come from the plan in ascending order, so the output is a function of the op's inputs;
//   - q * scale in f32, then f16;
//   - masked-off and padding cells are never loaded (zero fragments) and score -inf. All rows belong to one query, so no
//     row-dependent masking is needed in the PV WMMA (no fixup);
//   - split z takes chunks [z * len, min((z + 1) * len, C)), len = ceil(C / splits). One split normalises and writes dst;
//     otherwise it writes a partial row (256 values, max, normaliser) for sa_merge_splits;
//   - rotations (ROT = true instances only): H_q on whole query rows in f32 before the scale; H_o in the one-split
//     epilogue (8 heads at a time through shared memory), else after the merge.
// Every plan entry read is inside the plan dst and every K / V cell inside the views. Offsets are size_t.

#include <algorithm>
#include <cstdint>
#include <cstddef>

// ---- shape-only split count ---- Decode split rule on n_sel keys: up to 8 queries every split covers 64 keys, with more queries the splits get longer so the grid stays near its 8-query size. Clamped to the plan's chunks (cap / 4), so no split is empty, and to grid.y.
// It depends on (n_q, n_sel, n_kv) only, never on the plan's data.
static inline int64_t sa_g1_splits(const int64_t n_q, const int64_t n_sel, const int64_t n_kv) {
    const int64_t step = 64;
    int64_t chunk = step;
    if (n_q > 8) {
        const int64_t target = std::max<int64_t>(1, (8 * ((n_sel + step - 1) / step) + n_q - 1) / n_q);
        chunk = ((n_sel + target - 1) / target + 15) / 16 * 16;
    }
    const int64_t pr_splits = (n_sel + chunk - 1) / chunk;
    const int64_t quads     = (n_kv + 3) / 4;
    const int64_t cap       = ((std::min(n_sel, quads) + 3) / 4) * 4;   // G1 plan cap (ggml.h layout)
    return std::max<int64_t>(1, std::min(std::min(pr_splits, cap / 4), (int64_t) 65535));   // 65535: grid.y
}

// the census split class of a split count (1, 2-5, 6-16, 17-33, > 33)
static inline int sa_g1_split_class(const int64_t splits) {
    return splits <= 1 ? 0 : splits <= 5 ? 1 : splits <= 16 ? 2 : splits <= 33 ? 3 : 4;
}
#define SA_G1_SPLIT_CLASSES 5

// bytes of the split-K partials: [n_stream][n_q][n_head][splits][258] f32, none with one split
static inline size_t sa_g1_partial_bytes(const int64_t n_q, const int64_t n_head, const int64_t n_stream, const int64_t splits) {
    return splits > 1 ? (size_t) n_stream * (size_t) n_q * (size_t) n_head * (size_t) splits * 258 * sizeof(float) : 0;
}

// the largest partial buffer any runtime G1 op with n_q' <= n_q, n_kv' <= n_kv can use. The bytes are not monotone in n_q, hence the loop; splits is 1 for every n_q >= 8 * ceil(n_sel / 64), so it stops there
static inline size_t sa_g1_partial_bytes_max(const int64_t n_q, const int64_t n_sel, const int64_t n_kv, const int64_t n_head,
                                             const int64_t n_stream) {
    size_t m = 0;
    const int64_t last = std::min<int64_t>(n_q, 8 * ((n_sel + 63) / 64));
    for (int64_t nq = 1; nq <= last; ++nq) {
        m = std::max(m, sa_g1_partial_bytes(nq, n_head, n_stream, sa_g1_splits(nq, n_sel, n_kv)));
    }
    return m;
}

#if defined(GGML_USE_HIP)

#include "sparse-attn-loaders.cuh"
#include "sparse-attn-fwht.cuh"

#ifndef SA_L2E
#define SA_L2E 1.4426950408889634f
#endif

struct sa_g1_args {
    const char *     q;         // F32 [256, n_q, n_head, n_stream], nb1 = query, nb2 = head, nb3 = stream
    size_t           q_nb1, q_nb2, q_nb3;
    const char *     k;         // [256, cells, n_head_kv, n_stream]
    size_t           k_nb1, k_nb2, k_nb3;
    const char *     v;
    size_t           v_nb1, v_nb2, v_nb3;
    const uint32_t * counts;    // plan sections (ggml.h layout), G = 1: group = query
    const uint32_t * quads;
    const uint16_t * masks;
    int              n_groups;  // = n_q
    int              cap;
    int              n_q;       // queries per stream
    int              n_head;
    int              n_head_kv;
    int              n_stream;
    int              splits;
    float            scale;
    float *          out;       // splits == 1: dst F32 [256, n_head, n_q, n_stream]
    float *          partial;   // splits > 1: [n_stream][n_q][n_head][splits][258]
    int              kt_g;      // generic tier: the ggml types of K and V
    int              vt_g;
    int              rot_q;     // H_q block size (0, 64, 128 or 256)
    int              rot_o;     // H_o block size; applied here with one split, by sa_merge_splits otherwise
    float            rot_q_scale;   // 1.0f / sqrtf(rot_q), as fwht.cu
    float            rot_o_scale;   // 1.0f / sqrtf(rot_o)
    int              rot_dump;  // test switch (rot_q != 0): write H_q q (f32) to out and stop
};

typedef void (*sa_g1_kernel_t)(const sa_g1_args);

// the generic tier's G1 instances, index 2 * gqa_index + ROT
extern const sa_g1_kernel_t ggml_cuda_sa_gen_g1_kernels[];

template <int KT, int VT, int GQA, bool ROT>
__global__ void __launch_bounds__(256) sa_attend_g1(const sa_g1_args a) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3)
    GGML_UNUSED(a);
    NO_DEVICE_CODE;
#else
    static_assert(GQA == 4 || GQA == 8 || GQA == 12, "GQA must be 4, 8 or 12");
    // K direct for f16 and q8_0, split for q4_0 and bf16
    constexpr bool K_DIRECT = KT == SA_T_F16 || KT == SA_T_Q8_0;

    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, r = lane & 15, hi = lane >> 4;
    // grid: x = (stream, query, kv head), kv head fastest, y = split
    const int bq = blockIdx.x / a.n_head_kv, kvh = blockIdx.x - bq * a.n_head_kv;
    const int s = bq / a.n_q, query = bq - s * a.n_q, split = blockIdx.y;
    __shared__ union {
        uint16_t query_stage[8][16][32];
        float4   scores[8][64];
        float    ostage[8][256];   // rot_o epilogue: acc / l of 8 heads, whole rows
    } shared;
    __shared__ __align__(16) uint16_t probabilities[16][16];
    __shared__ float alpha_s[16];
    __shared__ float l_s[16];
    char * tile = reinterpret_cast<char *>(&shared.scores[w][0]);   // wave-private 1 KB slot (free outside the score exchange)

    const size_t     ent  = (size_t) s * a.n_groups + query;
    const uint32_t * gblk = a.quads + ent * a.cap;
    const uint16_t * gmsk = a.masks + ent * a.cap;
    // split z takes chunks [z * len, min((z + 1) * len, C)), len = ceil(C / splits) (32-bit: C <= 2^22, splits <= 65535)
    const uint32_t   nchk = a.counts[ent] >> 2;
    const uint32_t   len  = (nchk + (uint32_t) a.splits - 1) / (uint32_t) a.splits;
    const int        c0   = (int) min((uint32_t) split * len, nchk);
    const int        c1   = (int) min((uint32_t) split * len + len, nchk);
    const char *     kb_  = a.k + (size_t) s * a.k_nb3 + (size_t) kvh * a.k_nb2;
    const char *     vb_  = a.v + (size_t) s * a.v_nb3 + (size_t) kvh * a.v_nb2;

    // Q prologue: f16(q * scale) (f32 product, one rounding), rows >= GQA zero
    sa_v16s qf[2];
    if (ROT && a.rot_q != 0) {
        // H_q: wave w takes heads w and w + 8 whole (lane l holds dims 32 j + l), transforms them in f32 and writes f16(x * scale) of dim 32 j + l into the stage of wave j
        for (int head = w; head < 16; head += 8) {
            float x[8];
            if (head < GQA) {   // uniform over the wave
                const float * src = (const float *) (a.q + (size_t) s * a.q_nb3 + (size_t) query * a.q_nb1 + (size_t) (kvh * GQA + head) * a.q_nb2);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    x[j] = src[32*j + lane];
                }
                fwht_row<256, 32>(x, lane, a.rot_q, a.rot_q_scale);
                if (a.rot_dump) {
                    float * dst = a.out + (((size_t) s * a.n_q + query) * a.n_head + (size_t) (kvh * GQA + head)) * 256;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        dst[32*j + lane] = x[j];
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    x[j] = 0.0f;
                }
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                shared.query_stage[j][head][lane] = sa_f2h(x[j] * a.scale);
            }
        }
        if (a.rot_dump) {
            return;   // uniform over the block: the test hook stops after the transform
        }
    } else {
        for (int j = lane; j < 16*32; j += 32) {
            const int head = j / 32, d = j % 32;
            const float value = head < GQA ? *(const float *) (a.q + (size_t) s * a.q_nb3 + (size_t) query * a.q_nb1 +
                                                               (size_t) (kvh * GQA + head) * a.q_nb2 + (size_t) (w * 32 + d) * 4) : 0.0f;
            shared.query_stage[w][head][d] = sa_f2h(value * a.scale);
        }
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        const uint16_t * src = &shared.query_stage[w][r][t*16];
        qf[t] = __builtin_bit_cast(sa_v16s, (uint4[2]){*(const uint4 *) src, *(const uint4 *) (src + 8)});
    }
    __syncthreads();
    sa_v8f output[2] = {};
    float maximum = -INFINITY, normalizer = 0.0f;
    for (int c = c0; c < c1; ++c) {
        // the 16 cells of chunk c: cell kk = entry 4 * c + (kk >> 2), cell 4 * quad + (kk & 3), -1 when its mask bit is clear. The chunk's quads and masks are uniform over the block (scalar loads)
        const uint4 bb = *reinterpret_cast<const uint4 *>(gblk + 4 * c);
        const uint2 mm = *reinterpret_cast<const uint2 *>(gmsk + 4 * c);
        auto cell_of = [&](const int kk) -> int {
            const int j = kk >> 2, cc = kk & 3;
            const uint32_t qd = j == 0 ? bb.x : j == 1 ? bb.y : j == 2 ? bb.z : bb.w;
            const uint32_t mk = j < 2 ? mm.x : mm.y;
            return (mk >> (16 * (j & 1) + cc)) & 1u ? (int) (4 * qd + cc) : -1;
        };
        sa_v8f scores = {};
        sa_v16s kf[2];
        if constexpr (K_DIRECT) {
            const int key = cell_of(r);
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                kf[t] = sa_v16s{};
                if (key >= 0) {
                    if constexpr (KT == SA_T_Q8_0) {
                        const char * blk = kb_ + (size_t) key * a.k_nb1 + 34 * w;
                        const _Float16 d = __builtin_bit_cast(_Float16, *reinterpret_cast<const uint16_t *>(blk));
                        const sa_half2 d2 = sa_half2{d, d};
                        const uint16_t * qs = reinterpret_cast<const uint16_t *>(blk + 2) + 8*t;
                        uint32_t h[8];
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const uint32_t x = qs[j];
                            h[j] = sa_q8x2_to_h2((x & 0xFFu) | ((x & 0xFF00u) << 8), d2);
                        }
                        kf[t] = __builtin_bit_cast(sa_v16s, h);
                    } else {
                        const uint16_t * src = reinterpret_cast<const uint16_t *>(kb_ + (size_t) key * a.k_nb1) + 32*w + 16*t;
                        kf[t] = __builtin_bit_cast(sa_v16s, (uint4[2]){*(const uint4 *) src, *(const uint4 *) (src + 8)});
                    }
                }
            }
        } else {
            // split: lane half p loads and converts piece p of key r once, the halves are exchanged with v_permlanex16
            const int key = cell_of(r);
            uint32_t o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (key >= 0) {
                if constexpr (KT == SA_T_GEN) {   // generic tier: converted at load
                    sa_any_piece(a.kt_g, kb_ + (size_t) key * a.k_nb1, w, hi, o);
                } else {
                    uint32_t raw[sa_piece<KT>::N];
                    sa_piece_load<KT>(kb_ + (size_t) key * a.k_nb1, w, hi, raw);
                    sa_piece_cvt<KT>(raw, w, o);
                }
            }
            uint32_t x[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                x[i] = sa_xchg16(o[i]);
            }
            uint32_t k0[8], k1[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                k0[i] = hi ? x[i] : o[i];       k0[4+i] = hi ? o[i] : x[i];
                k1[i] = hi ? x[4+i] : o[4+i];   k1[4+i] = hi ? o[4+i] : x[4+i];
            }
            kf[0] = __builtin_bit_cast(sa_v16s, k0);
            kf[1] = __builtin_bit_cast(sa_v16s, k1);
        }
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            scores = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kf[t], qf[t], scores);
        }
        shared.scores[w][lane*2]     = make_float4(scores[0], scores[1], scores[2], scores[3]);
        shared.scores[w][lane*2 + 1] = make_float4(scores[4], scores[5], scores[6], scores[7]);
        __syncthreads();
        if (w == 0) {
            float score[8] = {};
#pragma unroll
            for (int sw = 0; sw < 8; ++sw) {
                const float4 lo = shared.scores[sw][lane*2], high = shared.scores[sw][lane*2 + 1];
                score[0] += lo.x;   score[1] += lo.y;   score[2] += lo.z;   score[3] += lo.w;
                score[4] += high.x; score[5] += high.y; score[6] += high.z; score[7] += high.w;
            }
            float local = -INFINITY;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                // select: a cell outside the plan's mask scores -inf (independent of its zero K fragment)
                if (cell_of(2*e + hi) < 0) {
                    score[e] = -INFINITY;
                }
                local = fmaxf(local, score[e]);
            }
            local = fmaxf(local, __shfl_xor(local, 16));
            const float next  = fmaxf(maximum, local);
            const float alpha = next == -INFINITY ? 1.0f : exp2f((maximum - next) * SA_L2E);
            // fmaxf drops a NaN score, so next stays -inf in a chunk of only -inf / NaN scores: exponentials are then taken against 0, a -inf cell gets 0 and a NaN cell keeps its NaN, so the normaliser is NaN and the row 0 (op contract)
            const float ref   = next == -INFINITY ? 0.0f : next;
            float sum = 0.0f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float p = exp2f((score[e] - ref) * SA_L2E);
                sum += p;
                probabilities[r][2*e + hi] = sa_f2h(p);
            }
            sum += __shfl_xor(sum, 16);
            normalizer = normalizer * alpha + sum;
            maximum = next;
            if (hi == 0) {
                alpha_s[r] = alpha;
            }
        }
        __syncthreads();
        const float alpha = alpha_s[r];
        const sa_v16s pf = __builtin_bit_cast(sa_v16s, (uint4[2]){
            *(const uint4 *) &probabilities[r][0], *(const uint4 *) &probabilities[r][8]});
        // V, LDS-staged: lane l loads piece l & 1 of cell l >> 1 (coalesced), converts it once and writes it transposed into the wave's 1 KB slot (bank swizzle); the WMMA A fragment is read back with 2 x b128 per t
        sa_v16s vf[2];
        {
            const int kk = lane >> 1, p = lane & 1;
            const int key = cell_of(kk);
            uint32_t o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (key >= 0) {
                if constexpr (VT == SA_T_GEN) {
                    sa_any_piece(a.vt_g, vb_ + (size_t) key * a.v_nb1, w, p, o);
                } else {
                    uint32_t raw[sa_piece<VT>::N];
                    sa_piece_load<VT>(vb_ + (size_t) key * a.v_nb1, w, p, raw);
                    sa_piece_cvt<VT>(raw, w, o);
                }
            }
            const int kh = (kk >> 3) ^ p, koff = (kk & 7) * 2;
            char * b0 = tile + 256 * p + (kh << 4) + koff;
            char * b1 = tile + 256 * p + ((kh ^ 1) << 4) + koff;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                char * bb = ((i >> 2) & 1) ? b1 : b0;
                const uint32_t wa = o[i >> 1], wb = o[4 + (i >> 1)];
                *reinterpret_cast<uint16_t *>(bb + 32 * i)       = (i & 1) ? (uint16_t) (wa >> 16) : (uint16_t) (wa & 0xffffu);
                *reinterpret_cast<uint16_t *>(bb + 512 + 32 * i) = (i & 1) ? (uint16_t) (wb >> 16) : (uint16_t) (wb & 0xffffu);
            }
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int dim = 16*t + r, sw = ((dim >> 2) ^ (dim >> 3)) & 1;
                const uint4 v0 = *reinterpret_cast<const uint4 *>(tile + dim * 32 + ((0 ^ sw) << 4));
                const uint4 v1 = *reinterpret_cast<const uint4 *>(tile + dim * 32 + ((1 ^ sw) << 4));
                vf[t] = __builtin_bit_cast(sa_v16s, (uint4[2]){v0, v1});
            }
            __builtin_amdgcn_wave_barrier();
        }
#pragma unroll
        for (int t = 0; t < 2; ++t) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                output[t][e] *= alpha;
            }
            output[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vf[t], pf, output[t]);
        }
        __syncthreads();
    }
    if (a.splits == 1) {
        // one split: normalise and write dst
        if (w == 0 && hi == 0) {
            l_s[r] = normalizer;
        }
        __syncthreads();
        if (ROT && a.rot_o != 0) {
            // H_o: 8 heads at a time, acc / l staged whole in shared memory; wave w transforms head h0 + w and stores it.
            // The chunk loop ended on a barrier, so the union is free
            for (int h0 = 0; h0 < GQA; h0 += 8) {
                if (r >= h0 && r < h0 + 8 && r < GQA) {
                    const float l = l_s[r];
#pragma unroll
                    for (int t = 0; t < 2; ++t) {
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            shared.ostage[r - h0][32*w + 16*t + 2*e + hi] = l > 0.0f ? output[t][e] / l : 0.0f;
                        }
                    }
                }
                __syncthreads();
                const int head = h0 + w;
                if (head < GQA) {   // uniform over the wave
                    float x[8];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        x[j] = shared.ostage[w][32*j + lane];
                    }
                    fwht_row<256, 32>(x, lane, a.rot_o, a.rot_o_scale);
                    float * dst = a.out + (((size_t) s * a.n_q + query) * a.n_head + (size_t) (kvh * GQA + head)) * 256;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        dst[32*j + lane] = x[j];
                    }
                }
                __syncthreads();
            }
        } else if (r < GQA) {
            const float l = l_s[r];
            float * dst = a.out + (((size_t) s * a.n_q + query) * a.n_head + (size_t) (kvh * GQA + r)) * 256;
#pragma unroll
            for (int t = 0; t < 2; ++t) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    dst[32*w + 16*t + 2*e + hi] = l > 0.0f ? output[t][e] / l : 0.0f;
                }
            }
        }
    } else if (r < GQA) {
        float * dst = a.partial + ((((size_t) s * a.n_q + query) * a.n_head + (size_t) (kvh * GQA + r)) * a.splits + split) * 258;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                dst[32*w + 16*t + 2*e + hi] = output[t][e];
            }
        }
        if (w == 0 && hi == 0) {
            dst[256] = maximum;
            dst[257] = normalizer;
        }
    }
#endif // defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3)
}

#endif // defined(GGML_USE_HIP)
