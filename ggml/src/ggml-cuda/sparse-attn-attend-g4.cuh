#pragma once

// sa_attend_g4<KT, VT, VL, EPI, GQA>: GGML_OP_SPARSE_ATTN with a G = 4 plan on RDNA3 WMMA (PLAN-v3 §6.2, D8, D10, D11).
//
// The body is PR #219's qsa3_attn_kernel (qsa-prefill.cu of 94e086847, as C18's c18_attn_kernel took it over line by
// line: Q prologue, WMMA order, online softmax, 2 barriers per 16-cell chunk, epilogue), with:
//   - no mask path and no runtime gate / o16 branches: EPI is a template parameter (C18: the PR kernel is 2-8.5 % slower
//     than the same loaders without those paths);
//   - the K / V loaders of sparse-attn-loaders.cuh: K split for every type (sa_k_loader_of), V packed (sa_pack_v) or
//     split-gather direct (D8);
//   - rows = 4 queries x GQA heads (GQA in {4, 8, 12}: 1, 2 or 3 WMMA row tiles of 16), streams on grid.z;
//   - the ggml.h plan layout (padding entries repeat the last quad with mask 0: no sentinel);
//   - the op contract (ggml.h): q * scale in f32, then f16 (the PR rounded q and scale to f16 first: identical while
//     scale is a power of two, 1/16 for qwen4exp);
//   - masked-off cells contribute nothing whatever their bytes (contract item 4): K / V of a cell no query of the group
//     attends are zeroed by select, and the score of a cell the row's query does not attend is replaced by -inf. The PV
//     WMMA is shared by the 4 queries of the group, so a V element that is not finite (a stored inf / NaN, a NaN scale,
//     an f16 overflow of the conversion) of a cell attended by only some of them would give 0 * inf = NaN in the rows
//     of the others. Packed V: sa_pack_v replaces every non-finite half by 0 and flags its quad; direct V: a wave whose
//     V fragment holds a non-finite half (one packed FMA per dword detects it) takes it out of the WMMA and flags the
//     block. sa_attend_fixup then recomputes every block that touches a flagged quad or was flagged, with the CPU op's
//     per-cell arithmetic (p * v added only to the rows whose query attends the cell). The flags and the fixup keep the
//     WMMA loop free of any exact-path code, which did not fit the 192 VGPRs of the GQA 12 instances (W3 resource
//     checks).
//
// Work: block (group g, kv head, stream), 8 waves; wave w owns dims 32w .. 32w+31. Scores: each wave computes the
// partial scores of all rows over its 32 dims, waves w < NT (NT = GQA / 4 row tiles) reduce tile w and run the online
// softmax for its 16 rows. Every count / quad / mask entry the kernel reads is inside the plan dst; every K / V cell it
// reads is a cell 4 * quad + c with quad < ceil(n_kv / 4), inside the views (ggml_sparse_attn asserts the view rows).

#if defined(GGML_USE_HIP)

#include "common.cuh"
#include "sparse-attn-loaders.cuh"

#define SA_L2E 1.4426950408889634f

enum sa_epilogue { SA_EPI_PLAIN = 0, SA_EPI_GATED = 1, SA_EPI_COUNT };

struct sa_g4_args {
    const char *     q;         // F32 [256, n_q, n_head, n_stream], nb1 = query, nb2 = head, nb3 = stream
    size_t           q_nb1, q_nb2, q_nb3;
    const char *     k;         // [256, cells, n_head_kv, n_stream]
    size_t           k_nb1, k_nb2, k_nb3;
    const char *     v;         // direct V (SA_V_SPLITGATHER)
    size_t           v_nb1, v_nb2, v_nb3;
    const uint16_t * pv;        // packed V (SA_V_PACKED): f16 [n_stream][n_head_kv][pv_nquads][256][4]
    size_t           pv_nquads;
    const uint32_t * counts;    // plan sections (ggml.h layout)
    const uint32_t * quads;
    const uint16_t * masks;
    int              n_groups;
    int              cap;
    int              n_q;       // queries per stream
    int              n_head;
    int              n_head_kv;
    float            scale;
    float *          out;       // plain: dst; gated: the MUL output (nullptr: only the 16-bit copy)
    const char *     gate;      // gated: row R = (s * n_q + i) * n_head + h at gate + R * gate_nb1
    size_t           gate_nb1;
    uint16_t *       o16;       // gated: 16-bit copy of the gated output for an MMB GEMM reader (nullptr: none)
    int              o16_f16;   // 1: f16 (saturating RNE), 0: bf16 (RNE)
    uint8_t *        flags;     // direct V: [n_stream][n_groups][n_head_kv][8 waves]: 1 = the wave took a non-finite V
                                // half out of the PV WMMA (the block is recomputed by sa_attend_fixup)
    const uint8_t *  qflags;    // packed V: [n_stream][n_head_kv][pv_nquads]: 1 = sa_pack_v replaced a non-finite half
                                // of that quad by 0 (nullptr for direct V)
    const uint32_t * qany;      // packed V: nonzero when any quad is flagged (cleared before the pack)
};

typedef void (*sa_g4_kernel_t)(const sa_g4_args);

// the instance tables of sparse-attn-attend-g4-<kt>.cu, in GGML_CUDA_SA_G4_KT_ order (sparse-attn-census.h):
// index ((vt * 2 + vl) * 2 + epi) * 3 + gqa_index, gqa_index 0 / 1 / 2 for GQA 4 / 8 / 12
extern const sa_g4_kernel_t ggml_cuda_sa_g4_kernels_f16[];
extern const sa_g4_kernel_t ggml_cuda_sa_g4_kernels_bf16[];
extern const sa_g4_kernel_t ggml_cuda_sa_g4_kernels_q8_0[];
extern const sa_g4_kernel_t ggml_cuda_sa_g4_kernels_q4_0[];

template <int KT, int VT, int VL, int EPI, int GQA>
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8))) sa_attend_g4(const sa_g4_args a) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3)
    GGML_UNUSED(a);
    NO_DEVICE_CODE;
#else
    static_assert(GQA == 4 || GQA == 8 || GQA == 12, "GQA must be 4, 8 or 12");
    constexpr int  NT     = GQA / 4;            // WMMA row tiles of 16 rows (4 queries x GQA heads)
    constexpr int  NPART  = NT > 2 ? NT : 2;    // the epilogue stages 8 waves x 16 rows x 32 floats in part
    constexpr int  KL     = sa_k_loader_of(KT);
    constexpr int  KRN    = sa_piece<KT>::N;
    constexpr bool PACKED = VL == SA_V_PACKED;

    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, r = lane & 15, hi = lane >> 4;
    const int g = blockIdx.x, kvh = blockIdx.y, s = blockIdx.z;
    __shared__ float4 part[NPART][8][64];
    __shared__ uint4  ptile[NT][32];
    __shared__ float  alpha_s[16 * NT];
    __shared__ float  l_s[16 * NT];

    const size_t     ent     = (size_t) s * a.n_groups + g;
    const uint32_t * gblk    = a.quads + ent * a.cap;
    const uint16_t * gmsk    = a.masks + ent * a.cap;
    const int        nchunks = (int) (a.counts[ent] >> 2);
    const char *     kbase   = a.k + (size_t) s * a.k_nb3 + (size_t) kvh * a.k_nb2;
    const char *     vbase   = PACKED ? nullptr : a.v + (size_t) s * a.v_nb3 + (size_t) kvh * a.v_nb2;
    const uint16_t * pvg     = PACKED ? a.pv + ((size_t) s * a.n_head_kv + kvh) * a.pv_nquads * 1024 : nullptr;

    // Q prologue: f16(q * scale) (f32 product, one rounding), staged per wave, read back as the WMMA B fragments
    sa_v16s qf[NT][2];
    {
        uint16_t * stage = reinterpret_cast<uint16_t *>(&part[0][0][0]) + w * (16 * NT * 32);
#pragma unroll
        for (int c = 0; c < 4 * NT; ++c) {
            const int row = 4*c + (lane >> 3), dq = 4 * (lane & 7);
            const int qi = row / GQA, h = row - GQA*qi;
            const int query = 4*g + qi;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (query < a.n_q) {
                v = *reinterpret_cast<const float4 *>(a.q + (size_t) s * a.q_nb3 + (size_t) query * a.q_nb1 +
                                                      (size_t) (kvh * GQA + h) * a.q_nb2 + (size_t) (32*w + dq) * 4);
            }
            const uint32_t h0 = sa_f2h(v.x * a.scale), h1 = sa_f2h(v.y * a.scale);
            const uint32_t h2 = sa_f2h(v.z * a.scale), h3 = sa_f2h(v.w * a.scale);
            *reinterpret_cast<uint2 *>(stage + row * 32 + dq) = make_uint2(h0 | (h1 << 16), h2 | (h3 << 16));
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < NT; ++i) {
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const uint16_t * src = stage + (16*i + r) * 32 + 16*t;
                qf[i][t] = __builtin_bit_cast(sa_v16s, (uint4[2]){*reinterpret_cast<const uint4 *>(src), *reinterpret_cast<const uint4 *>(src + 8)});
            }
        }
        __syncthreads();
    }
    sa_v8f O[NT][2];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
#pragma unroll
        for (int t = 0; t < 2; ++t) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                O[i][t][e] = 0.f;
            }
        }
    }
    float m = -INFINITY, l = 0.f;
    const int own_row = 16*w + r, own_qi = own_row / GQA;

    auto load_desc = [&](const int c, uint32_t * b, uint32_t & m01, uint32_t & m23) {
        const uint4 bb = *reinterpret_cast<const uint4 *>(gblk + 4*c);
        const uint2 mm = *reinterpret_cast<const uint2 *>(gmsk + 4*c);
        b[0] = bb.x; b[1] = bb.y; b[2] = bb.z; b[3] = bb.w; m01 = mm.x; m23 = mm.y;
    };
    // mask of entry j of the chunk (bit 4 * query + cell)
    auto entry_mask = [](const uint32_t m01, const uint32_t m23, const int j) -> uint32_t {
        return ((j < 2 ? m01 : m23) >> (16 * (j & 1))) & 0xffffu;
    };
    // some query of the group attends cell kk = 4 * j + c of the chunk
    auto cell_on = [&](const uint32_t m01, const uint32_t m23, const int kk) -> bool {
        return ((entry_mask(m01, m23, kk >> 2) >> (kk & 3)) & 0x1111u) != 0;
    };

    // ---- K: direct (f16), immediate ----
    auto load_k_direct = [&](const uint32_t * b, const uint32_t m01, const uint32_t m23, sa_v16s * kf) {
        if (!cell_on(m01, m23, r)) {
            kf[0] = sa_v16s{};
            kf[1] = sa_v16s{};
            return;
        }
        const char * row = kbase + (size_t) (4*b[r >> 2] + (r & 3)) * a.k_nb1 + (size_t) 64 * w;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const char * qq = row + (size_t) t * 32;
            kf[t] = __builtin_bit_cast(sa_v16s, (uint4[2]){*reinterpret_cast<const uint4 *>(qq), *reinterpret_cast<const uint4 *>(qq + 16)});
        }
    };
    // ---- K: split, deferred (raw piece loaded before the barriers, converted and exchanged after) ----
    auto k_raw = [&](const uint32_t * b, const uint32_t m01, const uint32_t m23, uint32_t * kr, bool & kon) {
        kon = cell_on(m01, m23, r);
        if (!kon) {
            return;
        }
        sa_piece_load<KT>(kbase + (size_t) (4*b[r >> 2] + (r & 3)) * a.k_nb1, w, hi, kr);
    };
    auto k_fin = [&](const uint32_t * kr, const bool kon, sa_v16s * kf) {
        uint32_t o[8];
        if (kon) {
            sa_piece_cvt<KT>(kr, w, o);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                o[i] = 0;
            }
        }
        uint32_t x[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            x[i] = sa_xchg16(o[i]);
        }
        // half p holds dims 8p.. of each 16-dim fragment t; the other half's dims come from x
        uint32_t k0[8], k1[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            k0[i] = hi ? x[i] : o[i];       k0[4+i] = hi ? o[i] : x[i];
            k1[i] = hi ? x[4+i] : o[4+i];   k1[4+i] = hi ? o[4+i] : x[4+i];
        }
        kf[0] = __builtin_bit_cast(sa_v16s, k0);
        kf[1] = __builtin_bit_cast(sa_v16s, k1);
    };

    // ---- V: split-gather (direct): raw elements before barrier 2, conversion + exchange after ----
    auto v_raw = [&](const uint32_t * b, uint32_t * vr) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                vr[4*j + c] = sa_gather_load<VT>(vbase + (size_t) (4*b[j] + c) * a.v_nb1, w, 16*hi + r);
            }
        }
    };
    auto v_fin = [&](const uint32_t * vr, const uint32_t m01, const uint32_t m23, sa_v16s * vf) {
        uint32_t o[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint2 v = make_uint2(sa_gather_pair<VT>(vr[4*j + 0], vr[4*j + 1]), sa_gather_pair<VT>(vr[4*j + 2], vr[4*j + 3]));
            const uint2 mv = sa_mask_cells(v, entry_mask(m01, m23, j));
            o[2*j] = mv.x;
            o[2*j + 1] = mv.y;
        }
        uint32_t x[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            x[i] = sa_xchg16(o[i]);
        }
        uint32_t v0[8], v1[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v0[i] = hi ? x[i] : o[i];
            v1[i] = hi ? o[i] : x[i];
        }
        vf[0] = __builtin_bit_cast(sa_v16s, v0);
        vf[1] = __builtin_bit_cast(sa_v16s, v1);
    };

    uint32_t b[4] = { 0, 0, 0, 0 };
    uint32_t m01 = 0, m23 = 0;
    bool     flagged = false;
    sa_v16s  kf[2];
    uint32_t kr[KRN];
    bool     kon = false;
    if (nchunks > 0) {
        load_desc(0, b, m01, m23);
        if constexpr (KL == SA_K_DIRECT) {
            load_k_direct(b, m01, m23, kf);
        } else {
            k_raw(b, m01, m23, kr, kon);
            k_fin(kr, kon, kf);
        }
    }

    for (int c = 0; c < nchunks; ++c) {
        sa_v8f sc[NT];
#pragma unroll
        for (int i = 0; i < NT; ++i) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                sc[i][e] = 0.f;
            }
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                sc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kf[t], qf[i][t], sc[i]);
            }
        }
        sa_v16s  vf[2];
        uint32_t vr[PACKED ? 1 : 16];
        if constexpr (PACKED) {
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int d = 32*w + 16*t + r;
                const uint2 v0 = sa_mask_cells(*reinterpret_cast<const uint2 *>(pvg + (size_t) b[0] * 1024 + d * 4), m01 & 0xffffu);
                const uint2 v1 = sa_mask_cells(*reinterpret_cast<const uint2 *>(pvg + (size_t) b[1] * 1024 + d * 4), m01 >> 16);
                const uint2 v2 = sa_mask_cells(*reinterpret_cast<const uint2 *>(pvg + (size_t) b[2] * 1024 + d * 4), m23 & 0xffffu);
                const uint2 v3 = sa_mask_cells(*reinterpret_cast<const uint2 *>(pvg + (size_t) b[3] * 1024 + d * 4), m23 >> 16);
                vf[t] = __builtin_bit_cast(sa_v16s, (uint2[4]){v0, v1, v2, v3});
            }
        } else {
            v_raw(b, vr);
        }
        // next chunk's descriptor and K (direct: loaded now; split: raw now, converted at the end of the iteration)
        uint32_t nb[4] = { 0, 0, 0, 0 };
        uint32_t nm01 = 0, nm23 = 0;
        sa_v16s  kfn[2];
        uint32_t nkr[KRN];
        bool     nkon = false;
        if constexpr (KL == SA_K_DIRECT) {
            if (c + 1 < nchunks) {
                load_desc(c + 1, nb, nm01, nm23);
                load_k_direct(nb, nm01, nm23, kfn);
            } else {
                kfn[0] = kf[0];
                kfn[1] = kf[1];
            }
        } else {
            if (c + 1 < nchunks) {
                load_desc(c + 1, nb, nm01, nm23);
                k_raw(nb, nm01, nm23, nkr, nkon);
            }
        }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            if (i != w) {
                part[i][w][lane*2 + 0] = make_float4(sc[i][0], sc[i][1], sc[i][2], sc[i][3]);
                part[i][w][lane*2 + 1] = make_float4(sc[i][4], sc[i][5], sc[i][6], sc[i][7]);
            }
        }
        __syncthreads();
        if (w < NT) {
            float scf[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                scf[e] = 0.f;
            }
#pragma unroll
            for (int w2 = 0; w2 < 8; ++w2) {
                if (w2 == w) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        scf[e] += sc[w][e];
                    }
                } else {
                    const float4 x = part[w][w2][lane*2 + 0], y = part[w][w2][lane*2 + 1];
                    scf[0] += x.x; scf[1] += x.y; scf[2] += x.z; scf[3] += x.w; scf[4] += y.x; scf[5] += y.y; scf[6] += y.z; scf[7] += y.w;
                }
            }
            // select: a cell this row's query does not attend scores -inf (a NaN score of a poisoned cell included)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int j = e >> 1, cc = 2*(e & 1) + hi;
                if (!((entry_mask(m01, m23, j) >> (own_qi * 4 + cc)) & 1u)) {
                    scf[e] = -INFINITY;
                }
            }
            float mloc = scf[0];
#pragma unroll
            for (int e = 1; e < 8; ++e) {
                mloc = fmaxf(mloc, scf[e]);
            }
            mloc = fmaxf(mloc, __shfl_xor(mloc, 16));
            const float mnew  = fmaxf(m, mloc);
            const float alpha = mnew == -INFINITY ? 1.f : exp2f((m - mnew) * SA_L2E);
            float p[8], ls = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                p[e] = mnew == -INFINITY ? 0.f : exp2f((scf[e] - mnew) * SA_L2E);
                ls += p[e];
            }
            ls += __shfl_xor(ls, 16);
            l = l * alpha + ls;
            m = mnew;
            uint32_t pp[4], po[4];
#pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) {
                pp[e2] = (uint32_t) sa_f2h(p[2*e2]) | ((uint32_t) sa_f2h(p[2*e2 + 1]) << 16);
                po[e2] = (uint32_t) __shfl_xor((int) pp[e2], 16);
            }
            uint16_t ph[16];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint16_t mine  = (e & 1) ? (uint16_t) (pp[e >> 1] >> 16) : (uint16_t) (pp[e >> 1] & 0xffffu);
                const uint16_t other = (e & 1) ? (uint16_t) (po[e >> 1] >> 16) : (uint16_t) (po[e >> 1] & 0xffffu);
                ph[2*e]     = hi ? other : mine;
                ph[2*e + 1] = hi ? mine  : other;
            }
            if (hi == 0) {
                ptile[w][r*2 + 0] = make_uint4((uint32_t) ph[0] | ((uint32_t) ph[1] << 16), (uint32_t) ph[2] | ((uint32_t) ph[3] << 16),
                                               (uint32_t) ph[4] | ((uint32_t) ph[5] << 16), (uint32_t) ph[6] | ((uint32_t) ph[7] << 16));
                ptile[w][r*2 + 1] = make_uint4((uint32_t) ph[8] | ((uint32_t) ph[9] << 16), (uint32_t) ph[10] | ((uint32_t) ph[11] << 16),
                                               (uint32_t) ph[12] | ((uint32_t) ph[13] << 16), (uint32_t) ph[14] | ((uint32_t) ph[15] << 16));
                alpha_s[own_row] = alpha;
            }
        }
        __syncthreads();
        float al[NT];
        bool  rescale = false;
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            al[i] = alpha_s[16*i + r];
            rescale = rescale || al[i] != 1.f;
        }
        if (__ballot(rescale)) {
#pragma unroll
            for (int i = 0; i < NT; ++i) {
#pragma unroll
                for (int t = 0; t < 2; ++t) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        O[i][t][e] *= al[i];
                    }
                }
            }
        }
        if constexpr (!PACKED) {
            v_fin(vr, m01, m23, vf);
        }
        // direct V: a non-finite V half in this wave's fragment (x * 0 + acc is NaN exactly then, one packed FMA per
        // dword). Packed V needs no check here: sa_pack_v already replaced every non-finite half by 0 and flagged its
        // quad, and the fixup recomputes the blocks that touch a flagged quad (W3 lab: the check costs about 4 % of the
        // packed kernel time on the C18 rows, scripts/qsa-validation/summaries/w3_kernel_time.md)
        if constexpr (!PACKED) {
            sa_half2 nf = sa_half2{(_Float16) 0, (_Float16) 0};
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const uint4 * vv = reinterpret_cast<const uint4 *>(&vf[t]);
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    nf = __builtin_bit_cast(sa_half2, vv[k].x) * sa_half2{(_Float16) 0, (_Float16) 0} + nf;
                    nf = __builtin_bit_cast(sa_half2, vv[k].y) * sa_half2{(_Float16) 0, (_Float16) 0} + nf;
                    nf = __builtin_bit_cast(sa_half2, vv[k].z) * sa_half2{(_Float16) 0, (_Float16) 0} + nf;
                    nf = __builtin_bit_cast(sa_half2, vv[k].w) * sa_half2{(_Float16) 0, (_Float16) 0} + nf;
                }
            }
            // a wave whose V fragment holds a non-finite half: the shared PV WMMA would give 0 * inf = NaN to the rows whose
            // query does not attend that cell, so the WMMA takes the finite halves only and the block is flagged; the exact
            // fixup kernel (sa_attend_fixup) then recomputes the flagged blocks with the CPU op's per-cell arithmetic
            if (__builtin_expect(__ballot((__builtin_bit_cast(uint32_t, nf) & 0x7fff7fffu) != 0) != 0, 0)) {
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    uint32_t x[8];
                    const uint32_t * src = reinterpret_cast<const uint32_t *>(&vf[t]);
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        uint32_t y = src[k];
                        if ((y & 0x00007c00u) == 0x00007c00u) y &= 0xffff0000u;
                        if ((y & 0x7c000000u) == 0x7c000000u) y &= 0x0000ffffu;
                        x[k] = y;
                    }
                    vf[t] = __builtin_bit_cast(sa_v16s, x);
                }
                flagged = true;
            }
        }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            const sa_v16s pf = __builtin_bit_cast(sa_v16s, (uint4[2]){ptile[i][r*2 + 0], ptile[i][r*2 + 1]});
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                O[i][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vf[t], pf, O[i][t]);
            }
        }
        if constexpr (KL == SA_K_DIRECT) {
            kf[0] = kfn[0];
            kf[1] = kfn[1];
        } else {
            k_fin(nkr, nkon, kf);   // nkon is false past the last chunk (zeros, unused)
        }
        b[0] = nb[0]; b[1] = nb[1]; b[2] = nb[2]; b[3] = nb[3]; m01 = nm01; m23 = nm23;
    }

    if (!PACKED && lane == 0) {   // one byte per (block, wave), written whether set or not: no clearing needed
        a.flags[(ent * a.n_head_kv + kvh) * 8 + w] = flagged ? 1 : 0;
    }
    if (w < NT && hi == 0) {
        l_s[own_row] = l;
    }
    __syncthreads();
    float * ostage = reinterpret_cast<float *>(&part[0][0][0]) + w * (16 * 32);
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        const float li = l_s[16*i + r];
#pragma unroll
        for (int t = 0; t < 2; ++t) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                ostage[r * 32 + 16*t + 2*e + hi] = li > 0.f ? O[i][t][e] / li : 0.f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int rr = 4*c + (lane >> 3), dq = 4 * (lane & 7);
            const int row = 16*i + rr, qi = row / GQA, h = row - GQA*qi;
            const int query = 4*g + qi;
            if (query < a.n_q) {
                const size_t R = ((size_t) s * a.n_q + query) * a.n_head + (size_t) (kvh * GQA + h);
                float4 v = *reinterpret_cast<const float4 *>(ostage + rr * 32 + dq);
                if constexpr (EPI == SA_EPI_GATED) {
                    // sigmoid(gate) * O with op_sigmoid's expression, as cont_sigmoid_mul_v4 and the PR's gated epilogue
                    const float4 gg = *reinterpret_cast<const float4 *>(a.gate + R * a.gate_nb1 + (size_t) (32*w + dq) * 4);
                    v.x = 1.0f / (1.0f + expf(-gg.x)) * v.x;
                    v.y = 1.0f / (1.0f + expf(-gg.y)) * v.y;
                    v.z = 1.0f / (1.0f + expf(-gg.z)) * v.z;
                    v.w = 1.0f / (1.0f + expf(-gg.w)) * v.w;
                    if (a.o16) {
                        const bool f16 = a.o16_f16 != 0;
                        uint2 hv;
                        hv.x = (uint32_t) ggml_cuda_f32_to_act16(v.x, f16) | ((uint32_t) ggml_cuda_f32_to_act16(v.y, f16) << 16);
                        hv.y = (uint32_t) ggml_cuda_f32_to_act16(v.z, f16) | ((uint32_t) ggml_cuda_f32_to_act16(v.w, f16) << 16);
                        *reinterpret_cast<uint2 *>(a.o16 + R * 256 + 32*w + dq) = hv;
                    }
                    if (a.out) {
                        *reinterpret_cast<float4 *>(a.out + R * 256 + 32*w + dq) = v;
                    }
                } else {
                    *reinterpret_cast<float4 *>(a.out + R * 256 + 32*w + dq) = v;
                }
            }
        }
        __syncthreads();
    }
#endif // defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3)
}


// one K / V element (dim d of a row) as its contract f16 value, widened to f32 (sparse-attn-q8.cuh conversions)
template <int T>
static __device__ __forceinline__ float sa_elem_f16(const char * row, const int d) {
    if constexpr (T == SA_T_Q8_0) {
        const char * blk = row + 34 * (d / 32);
        const _Float16 dh = __builtin_bit_cast(_Float16, *reinterpret_cast<const uint16_t *>(blk));
        return (float) ((_Float16) *reinterpret_cast<const int8_t *>(blk + 2 + d % 32) * dh);
    } else if constexpr (T == SA_T_Q4_0) {
        const char * blk = row + 18 * (d / 32);
        const _Float16 dh = __builtin_bit_cast(_Float16, *reinterpret_cast<const uint16_t *>(blk));
        const int j = d % 32;
        const uint8_t byte = *reinterpret_cast<const uint8_t *>(blk + 2 + (j & 15));
        return (float) ((_Float16) ((j < 16 ? (byte & 15) : (byte >> 4)) - 8) * dh);
    } else if constexpr (T == SA_T_BF16) {
        return (float) __builtin_bit_cast(_Float16, sa_bf1_to_h(*reinterpret_cast<const uint16_t *>(row + 2 * d)));
    } else {
        return (float) __builtin_bit_cast(_Float16, *reinterpret_cast<const uint16_t *>(row + 2 * d));
    }
}

// sa_attend_fixup<KT, VT, EPI>: exact recomputation of the blocks sa_attend_g4 flagged (a non-finite V half taken out of
// its PV WMMA), with the CPU op's arithmetic (ggml-cpu/sparse-attn.cpp): per row (query, head) the cells of the group
// that row's query attends, in the plan's ascending order, one at a time: f32 score (block sum over the 256 dims in a
// fixed order), online softmax with expf, P rounded to f16 for PV, l summing the f32 p, f32 PV, acc / l. Same grid as
// sa_attend_g4; a block with nothing flagged returns after one load (direct V: its 8 wave flags; packed V: the pack's
// any-flag, and only when that is set the quad flags of its entries). Thread d owns dim d.
template <int KT, int VT, int EPI>
__global__ void __launch_bounds__(256) sa_attend_fixup(const sa_g4_args a, const int gqa) {
    const int d = threadIdx.x, lane = d & 31, w = d >> 5;
    const int g = blockIdx.x, kvh = blockIdx.y, s = blockIdx.z;
    const size_t ent = (size_t) s * a.n_groups + g;
    __shared__ float red[8];
    const uint32_t * gblk  = a.quads + ent * a.cap;
    const uint16_t * gmsk  = a.masks + ent * a.cap;
    const uint32_t   n_ent = a.counts[ent];
    if (a.qflags != nullptr) {
        // packed V: recompute only when a quad of this group was flagged by the pack
        if (*a.qany == 0) {
            return;
        }
        const uint8_t * qf = a.qflags + ((size_t) s * a.n_head_kv + kvh) * a.pv_nquads;
        int hit = 0;
        for (uint32_t e = d; e < n_ent; e += 256) {
            hit |= qf[gblk[e]] != 0;
        }
        if (!__syncthreads_or(hit)) {
            return;
        }
    } else if (*reinterpret_cast<const uint64_t *>(a.flags + (ent * a.n_head_kv + kvh) * 8) == 0) {
        return;
    }
    const char *     kbase = a.k + (size_t) s * a.k_nb3 + (size_t) kvh * a.k_nb2;
    const char *     vbase = a.v + (size_t) s * a.v_nb3 + (size_t) kvh * a.v_nb2;
    for (int row = 0; row < 4 * gqa; ++row) {
        const int qi = row / gqa, h = row - gqa * qi, query = 4*g + qi;
        if (query >= a.n_q) {
            break;   // rows are ordered by query
        }
        const int hq = kvh * gqa + h;
        const float qv = *reinterpret_cast<const float *>(a.q + (size_t) s * a.q_nb3 + (size_t) query * a.q_nb1 + (size_t) hq * a.q_nb2 + (size_t) d * 4);
        const float qd = (float) __builtin_bit_cast(_Float16, sa_f2h(qv * a.scale));
        float mx = -INFINITY, l = 0.f, acc = 0.f;
        for (uint32_t e = 0; e < n_ent; ++e) {
            const uint32_t bits = ((uint32_t) gmsk[e] >> (4 * qi)) & 0xfu;
            if (bits == 0) {
                continue;
            }
            for (int c = 0; c < 4; ++c) {
                if (!((bits >> c) & 1u)) {
                    continue;
                }
                const size_t cell = 4 * (size_t) gblk[e] + c;
                float prod = qd * sa_elem_f16<KT>(kbase + cell * a.k_nb1, d);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    prod += __shfl_xor(prod, o);
                }
                if (lane == 0) {
                    red[w] = prod;
                }
                __syncthreads();
                float sc = 0.f;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    sc += red[k];
                }
                __syncthreads();
                if (sc > mx) {
                    const float alpha = mx == -INFINITY ? 0.0f : expf(mx - sc);
                    acc *= alpha;
                    l   *= alpha;
                    mx   = sc;
                }
                const float p  = expf(sc - mx);
                const float ph = (float) __builtin_bit_cast(_Float16, sa_f2h(p));
                l   += p;
                acc += ph * sa_elem_f16<VT>(vbase + cell * a.v_nb1, d);
            }
        }
        float v = l > 0.f ? acc / l : 0.f;
        const size_t R = ((size_t) s * a.n_q + query) * a.n_head + (size_t) hq;
        if constexpr (EPI == SA_EPI_GATED) {
            const float gg = *reinterpret_cast<const float *>(a.gate + R * a.gate_nb1 + (size_t) d * 4);
            v = 1.0f / (1.0f + expf(-gg)) * v;
            if (a.o16) {
                a.o16[R * 256 + d] = ggml_cuda_f32_to_act16(v, a.o16_f16 != 0);
            }
            if (a.out) {
                a.out[R * 256 + d] = v;
            }
        } else {
            a.out[R * 256 + d] = v;
        }
    }
}

typedef void (*sa_fixup_kernel_t)(const sa_g4_args, const int);

// the fixup instances of sparse-attn-attend-g4-<kt>.cu, index vt * 2 + epi
extern const sa_fixup_kernel_t ggml_cuda_sa_fixup_kernels_f16[];
extern const sa_fixup_kernel_t ggml_cuda_sa_fixup_kernels_bf16[];
extern const sa_fixup_kernel_t ggml_cuda_sa_fixup_kernels_q8_0[];
extern const sa_fixup_kernel_t ggml_cuda_sa_fixup_kernels_q4_0[];

#endif // defined(GGML_USE_HIP)
