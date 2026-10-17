#include "fattn-rdna35.cuh"
#include "fattn-common.cuh"
#include "convert.cuh"

// Flash attention for RDNA3.5 WMMA (v_wmma_f32_16x16x16_f16), D = 64, 128, 256 or 512, f16 K/V. For prefill a K/V cache of
// another type (q8_0, q4_0, ...) is converted to f16 first, into the scratch space behind dst: at 32k keys and 4096
// tokens that is ~1% of the attention time, while the generic kernels such a cache fell back to are ~2x slower.
// Small batches at depth read q8_0 K/V directly (dequantized while staged to LDS): they are bound by the K/V reads.
// Absorbed MLA (deepseek2 family, DKQ = kv_lora_rank + rope, DV = kv_lora_rank) is served the same way: K spans
// DKQ dims while V reads its first DV dims of the same (strided or converted) cache rows; the O tile follows DV.
//
// The generic MMA kernel keeps fp32 accumulators for 16 query columns x all V dims per warp (128 VGPRs at
// D = 256), so Q cannot stay in registers, every WMMA re-reads both operands from LDS and the kernel
// spills; D = 512 is not supported by it at all on RDNA and falls back to the FMA tile kernel.
// Here a block of 4*DQ warps owns 64 query columns (4 column tiles of 16) and walks the KV cache in steps of
// 64 / 32 / 32 / 16 keys at D = 64 / 128 / 256 / 512 (32 WMMAs per warp and step):
//   - S^T = K Q^T: the DQ = D/128 warps of a column tile split D (one warp at D <= 128); each keeps its share
//     of Q in registers (64 VGPRs), scores all key tiles of the step over its share (one LDS operand per WMMA) and the shares
//     are summed through LDS. Every warp of a column tile then holds all scores of its columns, so the
//     softmax needs no max exchange.
//   - P stays in the lane that owns the column; one permlanex16 turns the WMMA C layout into the B operand
//     layout.
//   - O^T += V^T P: warp w owns V dims [32w, 32w+32) for all 64 columns (64 VGPRs of accumulators). V is
//     transposed once per block while it is staged into LDS.
//   - K and V^T have separate LDS buffers: the next K and this step's V^T are both staged between the two
//     barriers of a step, from registers that were loaded while the scores were computed.
//   - a pre-pass classifies the mask per (token tile, 32 keys): all-zero steps ignore the mask, all -inf
//     steps are skipped, only mixed steps (near the diagonal) read it.
//   - the head group varies fastest across blocks, so the blocks in flight read whole KV cache rows.
//   - small batches (decode, speculative verification) have too few column tiles to fill the GPU: the key steps are
//     split over gridDim.y blocks, which write unnormalized partials and (max, row sum) for flash_attn_combine_results.
// Column c of a block is (token jt*NCOLS1 + c/NCOLS2, Q head head0 + c%NCOLS2); all NCOLS2 heads share
// one K/V head.

#if defined(GGML_USE_HIP) && defined(RDNA3_5)
#define FATTN_RDNA35_AVAILABLE
#endif // defined(GGML_USE_HIP) && defined(RDNA3_5)

namespace fattn_rdna35 {

constexpr int CLS_KEYS   = 32;     // granularity of the mask classes
constexpr int MAX_CLS    = 8192;   // mask classes kept in LDS (2 bits each): up to 262144 keys
constexpr float m_init   = -1e30f; // finite so that fully masked columns never produce inf - inf
constexpr float log2e    = 1.44269504088896340736f;
constexpr float T_LAZY   = 8.0f;   // OWN: the reference max moves when a step max is more than this above it (log2 units)
constexpr float T_HI     = 14.0f;  // OWN: above this P would leave the f16 range -> redo the step against the new max

// DKQ = Q/K head dim, DV = V head dim. The generic GQA models have DV == DKQ; MLA (absorbed deepseek2-family
// attention) has DKQ == kv_lora_rank + rope (576) and DV == kv_lora_rank (512), with V a strided view of the
// K cache rows (both read from one buffer, V only its first DV dims). The K-side constants (SEGS/SQK/QCH and
// the sK tile) follow DKQ, the V^T staging and the O epilogue follow DV.
template <int DKQ, int DV = DKQ>
struct cfg {
    static constexpr int NCOLS  = DV >= 512 ? 32 : 64;  // query columns per block (32 at DV = 512: 16.3 vs 25.6 TFLOPS, K/V reuse halves)
    static constexpr int NCT    = NCOLS / 16;           // column tiles
    static constexpr int DQ     = DKQ > 128 ? DKQ / 128 : 1; // warps per column tile, splitting D in the score phase
    static constexpr int QCH    = DKQ / DQ / 16;        // 16-dim chunks of Q per warp
    static constexpr int NWARPS = NCT * DQ;
    static constexpr int NTHR   = NWARPS * 32;
    static constexpr int OT     = DV / NWARPS / 16;     // warp w owns V dims [16*OT*w, 16*OT*(w+1))
    static constexpr int NKEYS  = DV >= 512 ? 16 : DKQ == 64 ? 64 : 32; // keys per step: 32 WMMAs per warp and step
    static constexpr int KT     = NKEYS / 16;           // key tiles per step
    static constexpr int SEGS   = DKQ / 8;              // 16 B segments per K row
    static constexpr int SEGSV  = DV / 8;               // 16 B segments per V row
    static constexpr int WPRV   = SEGSV / 8;            // warps per V row
    static constexpr int NKP    = (NKEYS*SEGS + NTHR - 1) / NTHR; // K rows (keys) per thread and step (last pass may be partial)
    static constexpr int KPT    = (NKEYS*SEGSV) / NTHR; // V keys per thread and step
    // D = 512 runs at 192 VGPRs (2 blocks per WGP). With V prefetched as well and all P operands of a key tile loaded at
    // once, the loop spilled and its scratch reloads waited for the K/V loads (vmcnt is in order): 26.7 TFLOPS. Loading V
    // after the first barrier, one score chain and P one operand at a time: 29.5. At D = 256 both cost ~1%.
    static constexpr bool TIGHT = DV >= 512;
    // Fewer VGPRs can only be reached that way (live state in the source): on gfx1151 with the ROCm core-10.0 toolchain
    // neither the __launch_bounds__ minimum-waves argument nor __attribute__((amdgpu_waves_per_eu)) drives the register
    // budget (standalone check: same 256-float kernel, same num_vgpr with and without a 4-waves/EU cap). The q8_0 decode
    // path at D <= 128 runs 256 VGPRs = 8 waves per CU and is latency-bound there: more split blocks help a little
    // (LFM2.5, 64k keys: 532 -> 470 us per call at 240 blocks), fewer live columns are the real lever.
    // D = 256: warp dq of a column tile finishes key tile dq of the step (half the score exchange and softmax per warp).
    static constexpr bool OWN   = KT == DQ;
    static constexpr int SQK    = DKQ + 8;              // row stride (halves) of sK:     conflict-free 16-row reads
    static constexpr int SVT    = NKEYS + 8;            // row stride (halves) of sVt/sP: conflict-free 16-row reads
    static constexpr int NUNITS = NKEYS / 8;            // 8-key units of a V^T row (swizzled)

    static constexpr int SZ_K   = NKEYS * SQK * 2;
    static constexpr int SZ_VT  = DV * SVT * 2;
    static constexpr int SZ_P   = NCOLS * SVT * 2;
    // partial scores [warp][key tile][lane][8]; one warp per column tile (D <= 128) needs no exchange, only the column maxima
    static constexpr int SZ_SX  = DQ == 1 ? NCOLS * 4 : NWARPS * (OWN ? 1 : KT) * 32 * 8 * 4;
    static constexpr int SZ_RED = (OWN ? 4 : 2) * NCOLS * 4;            // per column: step maxima and row sums (OWN), or rescale factors and row sums
    static constexpr int SZ_CLS = MAX_CLS / 4;

    static constexpr int OFF_K   = 0;
    static constexpr int OFF_VT  = OFF_K   + SZ_K;
    static constexpr int OFF_P   = OFF_VT  + SZ_VT;
    static constexpr int OFF_SX  = OFF_P   + SZ_P;
    static constexpr int OFF_RED = OFF_SX  + SZ_SX;
    static constexpr int OFF_CLS = OFF_RED + SZ_RED;
    static constexpr int SZ_SMEM = OFF_CLS + SZ_CLS;

    static_assert(DV <= DKQ, "V head dim cannot exceed the K/Q head dim");
    static_assert(KPT == 2 || KPT == 4, "V transpose handles 2 or 4 keys per thread");
    static_assert(SEGS % 8 == 0 && SEGSV % 8 == 0 && NTHR % SEGSV == 0, "thread layout");
    static_assert(NWARPS % WPRV == 0 && NKEYS == KPT*4*(NWARPS / WPRV), "V staging key coverage");
    static_assert(QCH == 8 || (DQ == 1 && QCH == 4) || (DQ == 4 && QCH == 9), "Q chunks per warp");
    static_assert(NKEYS % CLS_KEYS == 0 || CLS_KEYS % NKEYS == 0, "mask classes per step");
    static_assert(SZ_SMEM + 16 <= 65536, "LDS budget");
};

typedef _Float16 half16_t __attribute__((ext_vector_type(16)));
typedef float    float8_t __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4_t  __attribute__((ext_vector_type(4))); // plain vector: stays in registers (HIP's int4 class may not)

#ifdef FATTN_RDNA35_AVAILABLE

static __device__ __forceinline__ float8_t wmma(const half16_t a, const half16_t b, const float8_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// Barrier that only orders LDS: __syncthreads() also waits for all outstanding global loads (vmcnt(0)) and
// invalidates the vector L0 cache, which would cut the K/V prefetch short at every barrier.
static __device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// value of the lane in the other half of the wave (lane ^ 16)
static __device__ __forceinline__ int swap_halves(const int x) {
    return __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xfedcba98u, false, false);
}
static __device__ __forceinline__ float swap_halves(const float x) {
    return __int_as_float(swap_halves(__float_as_int(x)));
}

// q8_0 K/V (small batches, see the launcher): a 16 B f16 segment of 8 dims is a quarter of a q8_0 block. Its 8 quants
// are loaded raw with the block scale and dequantized when staged to LDS (same f16 values as dequantize_q8_0).
// The quants of segment s start at byte 34*(s/4) + 2 + 8*(s%4) of the row: 2 mod 4 in even blocks, where a third dword
// is loaded and the two data dwords are funnel-shifted out; rows and heads start at multiples of 4 B (host check).
static __device__ __forceinline__ u32x4_t load_seg_q8(const char * row, const int seg) {
    const uint32_t  blk = seg / 4;
    const uint32_t  off = 34u*blk + 2u + 8u*(seg % 4);
    const uint32_t * p  = (const uint32_t *) (row + (off & ~3u));
    u32x4_t r;
    r[0] = p[0];
    r[1] = p[1];
    r[2] = blk % 2 == 0 ? p[2] : 0u;
    r[3] = *(const uint16_t *) (row + 34u*blk);
    return r;
}
static __device__ __forceinline__ u32x4_t dequant_seg_q8(const u32x4_t r, const int seg) {
    typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
    const uint32_t sh = (seg / 4) % 2 == 0 ? 2u : 0u;
    // q + 128 as the low byte of an f16 with exponent 10: 1152 + q, exact
    const uint32_t lo = __builtin_amdgcn_alignbyte(r[1], r[0], sh) ^ 0x80808080u;
    const uint32_t hi = __builtin_amdgcn_alignbyte(r[2], r[1], sh) ^ 0x80808080u;
    const uint32_t b[4] = {
        __builtin_amdgcn_perm(0x64646464u, lo, 0x04010400u), __builtin_amdgcn_perm(0x64646464u, lo, 0x04030402u),
        __builtin_amdgcn_perm(0x64646464u, hi, 0x04010400u), __builtin_amdgcn_perm(0x64646464u, hi, 0x04030402u),
    };
    const h2_t d = __builtin_bit_cast(h2_t, r[3] * 0x00010001u);
    const h2_t o = {(_Float16) -1152.0f, (_Float16) -1152.0f};
    u32x4_t out;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        out[e] = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(h2_t, b[e]) + o) * d); // q exact, one rounding of d*q
    }
    return out;
}

// K tile: NKP x 16 B per thread; the lanes of a wave read 512 contiguous bytes of a key row. The last pass is
// partial when the K row segments per step (DKQ wide) do not divide evenly by the threads (MLA DKQ = 576).
template <class C, bool Q8>
static __device__ __forceinline__ void load_K(u32x4_t * regs, const char * K, const uint32_t nb11, const int k0) {
    constexpr int NITEMS = C::NKEYS*C::SEGS;
#pragma unroll
    for (int p = 0; p < C::NKP; ++p) {
        const int idx = p*C::NTHR + threadIdx.x;
        if (NITEMS % C::NTHR == 0 || idx < NITEMS) {
            if constexpr (Q8) {
                regs[p] = load_seg_q8(K + uint32_t(k0 + idx/C::SEGS)*nb11, idx % C::SEGS);
            } else {
                regs[p] = *(const u32x4_t *) (K + (uint32_t(k0 + idx/C::SEGS)*nb11 + uint32_t(idx % C::SEGS)*16u));
            }
        }
    }
}
template <class C, bool Q8>
static __device__ __forceinline__ void store_K(const u32x4_t * regs, _Float16 * sK) {
    constexpr int NITEMS = C::NKEYS*C::SEGS;
#pragma unroll
    for (int p = 0; p < C::NKP; ++p) {
        const int idx = p*C::NTHR + threadIdx.x;
        if (NITEMS % C::NTHR == 0 || idx < NITEMS) {
            *(u32x4_t *) (sK + (idx/C::SEGS)*C::SQK + (idx % C::SEGS)*8) = Q8 ? dequant_seg_q8(regs[p], idx % C::SEGS) : regs[p];
        }
    }
}

// V tile: thread = (group kg of KPT keys, 8-dim segment seg). Lanes 8j..8j+7 of a wave read 8 consecutive
// segments of one key, i.e. full 128 B cache lines.
template <int D>
static __device__ __forceinline__ void v_thread_coords(int & kg, int & seg) {
    constexpr int WPR = cfg<D>::SEGS / 8; // warps per V row
    const int lane = threadIdx.x % 32;
    const int w    = threadIdx.x / 32;
    seg = lane % 8 + 8*(w % WPR);
    kg  = lane / 8 + 4*(w / WPR);
}
template <int D, bool Q8>
static __device__ __forceinline__ void load_V(u32x4_t * regs, const char * V, const uint32_t nb21, const int k0) {
    using C = cfg<D>;
    int kg, seg;
    v_thread_coords<D>(kg, seg);
#pragma unroll
    for (int kk = 0; kk < C::KPT; ++kk) {
        if constexpr (Q8) {
            regs[kk] = load_seg_q8(V + uint32_t(k0 + C::KPT*kg + kk)*nb21, seg);
        } else {
            regs[kk] = *(const u32x4_t *) (V + (uint32_t(k0 + C::KPT*kg + kk)*nb21 + uint32_t(seg)*16u));
        }
    }
}
// V^T[dim][key] in LDS: within a row the 8-key units are swizzled with the dim's segment, so that the lanes
// of a store (same key group, consecutive segments) spread over more banks.
template <int D>
static __device__ __forceinline__ int vt_unit(const int unit, const int dim) {
    return unit ^ ((dim / 8) % cfg<D>::NUNITS);
}
// registers -> LDS as V^T: the KPT keys x 8 dims block of the thread is transposed with v_perm_b32.
template <int D, bool Q8>
static __device__ __forceinline__ void store_Vt(const u32x4_t * regs_in, _Float16 * sVt) {
    using C = cfg<D>;
    int kg, seg;
    v_thread_coords<D>(kg, seg);
    u32x4_t regs[C::KPT];
#pragma unroll
    for (int kk = 0; kk < C::KPT; ++kk) {
        regs[kk] = Q8 ? dequant_seg_q8(regs_in[kk], seg) : regs_in[kk];
    }
    // regs[kk][q]: dims (2q, 2q+1) of key kk
    const int col = 8*vt_unit<D>(C::KPT*kg/8, 8*seg) + (C::KPT*kg) % 8;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        _Float16 * row_even = sVt + (8*seg + 2*q + 0)*C::SVT + col;
        _Float16 * row_odd  = sVt + (8*seg + 2*q + 1)*C::SVT + col;
        const uint32_t e01 = __builtin_amdgcn_perm(regs[1][q], regs[0][q], 0x05040100u); // dim 2q:   keys 0, 1
        const uint32_t o01 = __builtin_amdgcn_perm(regs[1][q], regs[0][q], 0x07060302u); // dim 2q+1: keys 0, 1
        if constexpr (C::KPT == 4) {
            const uint32_t e23 = __builtin_amdgcn_perm(regs[3][q], regs[2][q], 0x05040100u);
            const uint32_t o23 = __builtin_amdgcn_perm(regs[3][q], regs[2][q], 0x07060302u);
            *(uint2 *) row_even = make_uint2(e01, e23);
            *(uint2 *) row_odd  = make_uint2(o01, o23);
        } else {
            *(uint32_t *) row_even = e01;
            *(uint32_t *) row_odd  = o01;
        }
    }
}
// A operand of V^T for dims [dim0, dim0+16) (dim0 % 16 == 0) and keys [16c, 16c+16): lane row dim0 + l16.
template <int D>
static __device__ __forceinline__ half16_t load_Vt_operand(const _Float16 * sVt, const int dim, const int c) {
    typedef int int8v_t __attribute__((ext_vector_type(8)));
    const _Float16 * row = sVt + dim*cfg<D>::SVT;
    const int4 lo = *(const int4 *) (row + 8*vt_unit<D>(2*c + 0, dim));
    const int4 hi = *(const int4 *) (row + 8*vt_unit<D>(2*c + 1, dim));
    const int8v_t v = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    return __builtin_bit_cast(half16_t, v);
}

#endif // FATTN_RDNA35_AVAILABLE

// D = 512: a 16-wave block at up to 256 VGPRs fits once per WGP, so the SIMDs idle while its warps wait at the barriers.
// Capped at 192 VGPRs (8 waves per SIMD) a second block fits (LDS allows two) and fills these gaps (see cfg::TIGHT).
template <int DKQ, int DV, int NCOLS2, bool Q8>
__launch_bounds__((cfg<DKQ, DV>::NTHR), (cfg<DKQ, DV>::NWARPS == 16 ? 8 : 1)) // HIP: the second argument is the minimum number of waves per SIMD
static __global__ void flash_attn_rdna35(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const char * __restrict__ mask, const uint8_t * __restrict__ mask_cls, const int * __restrict__ KV_max, float * __restrict__ dst,
        float * __restrict__ dst_parts, float2 * __restrict__ dst_meta, const float * __restrict__ sinks,
        const float scale_q, const float softcap_log2, const int ne01, const int ne02, const int ne11, const int gqa_ratio,
        const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int64_t nb21, const int64_t nb22, const int64_t nb23,
        const int ne31, const int ne33, const int64_t nb31, const int64_t nb33) {
#ifdef FATTN_RDNA35_AVAILABLE
    using C = cfg<DKQ, DV>;
    constexpr int NCOLS  = C::NCOLS;
    constexpr int NCOLS1 = NCOLS / NCOLS2;
    constexpr int NUSED  = NCOLS1 * NCOLS2; // NCOLS2 = 6: 60 of the 64 columns
    constexpr int NCT    = C::NCT;
    constexpr int OT     = C::OT;
    constexpr int KT     = C::KT;
    constexpr int DQ     = C::DQ;
    constexpr int NKEYS  = C::NKEYS;
    constexpr int SVT    = C::SVT;

    __shared__ __attribute__((aligned(16))) char smem[C::SZ_SMEM];
    __shared__ int sRescaleStep; // last step in which some column max moved
    __shared__ int sRedo;        // OWN: last step whose P had to be redone against a new max
    _Float16 * sK     = (_Float16 *) (smem + C::OFF_K);
    _Float16 * sVt    = (_Float16 *) (smem + C::OFF_VT);
    _Float16 * sP     = (_Float16 *) (smem + C::OFF_P);
    float    * sSx    = (float    *) (smem + C::OFF_SX);
    float    * sAlpha = (float    *) (smem + C::OFF_RED);
    float    * sMax   = sAlpha;                              // OWN: [key tile][l16][column tile]
    float    * sL     = sAlpha + (C::OWN ? 2*NCOLS : NCOLS); // OWN: [key tile][column]

    const int lane = threadIdx.x % 32;
    const int w    = threadIdx.x / 32;
    const int l16  = lane % 16;
    const int hh   = lane / 16;
    const int ct   = w / DQ;      // column tile this warp scores
    const int dq   = w % DQ;      // share of D this warp scores: dims [128dq, 128dq+128)
    const int c_s  = 16*ct + l16; // column of this lane in the score phase

    const int n_hg   = ne02 / NCOLS2;
    const int ntiles = gridDim.x / n_hg;
    const int jt     = blockIdx.x / n_hg;
    const int head0  = (blockIdx.x % n_hg)*NCOLS2;
    const int seq    = blockIdx.z;
    const int z_KV   = head0 / gqa_ratio;

    const char * K_h = K + seq*nb13 + z_KV*nb12;
    const char * V_h = V + seq*nb23 + z_KV*nb22;
    const uint32_t nb11u = (uint32_t) nb11; // row offsets of one head stay below 4 GiB (checked on the host)
    const uint32_t nb21u = (uint32_t) nb21;
    const half * mask_h = mask ? (const half *) (mask + (seq % ne33)*nb33) : nullptr;
    const int64_t s31 = nb31 / sizeof(half);

    const int kv_end  = KV_max ? min(ne11, KV_max[seq*ntiles + jt]) : ne11;
    const int n_steps = (kv_end + NKEYS - 1) / NKEYS;

    // split-KV: this block takes the key steps [st_begin, st_end)
    const int nsplit   = gridDim.y;
    const int st_begin = int(int64_t(n_steps)*blockIdx.y/nsplit);
    const int st_end   = int(int64_t(n_steps)*(blockIdx.y + 1)/nsplit);

    // mask classes per 32 keys (see mask_classify), uniform across the block; no mask = all zero
    uint8_t * cls = mask_cls ? (uint8_t *) (smem + C::OFF_CLS) : nullptr;
    if (cls) {
        const int n_cls  = (n_steps*NKEYS + CLS_KEYS - 1) / CLS_KEYS;
        const int stride = (ne11/CLS_KEYS + 3) / 4;
        const uint8_t * cls_g = mask_cls + (int64_t(seq % ne33)*ntiles + jt)*stride;
        for (int i = threadIdx.x; i < (n_cls + 3)/4; i += C::NTHR) {
            cls[i] = cls_g[i];
        }
    }
    if (threadIdx.x == 0) {
        sRescaleStep = -1;
        sRedo        = -1;
    }
    auto step_class = [&](const int st) -> int {
        if (!cls) {
            return mask_h ? 2 : 0;
        }
        const int i = st*NKEYS / CLS_KEYS;
        int c = (cls[i/4] >> (2*(i % 4))) & 3;
#pragma unroll
        for (int j = 1; j < NKEYS / CLS_KEYS; ++j) { // a step of 64 keys (D = 64) spans two classes
            const int cj = (cls[(i + j)/4] >> (2*((i + j) % 4))) & 3;
            c = c == cj ? c : 2;
        }
        return c;
    };
    auto next_active = [&](int st) -> int {
        while (st < st_end && step_class(st) == 1) {
            ++st;
        }
        return st;
    };

    // Q of column c_s, dims [128dq, 128dq+128) -> registers as B operands, pre-scaled by scale*log2(e)
    // so that the softmax can use exp2 (with a logit softcap by scale/softcap, see softcap below).
    half16_t qb[C::QCH];
    {
        const int  token = jt*NCOLS1 + c_s/NCOLS2;
        const bool valid = token < ne01 && c_s < NUSED;
        const float * qrow = (const float *) (Q + seq*nb03 + int64_t(head0 + c_s % NCOLS2)*nb02 + int64_t(valid ? token : 0)*nb01) + (DKQ/DQ)*dq;
#pragma unroll
        for (int k = 0; k < C::QCH; ++k) {
#pragma unroll
            for (int e = 0; e < 16; e += 4) {
                const float4 v = valid ? *(const float4 *) (qrow + 16*k + e) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                qb[k][e + 0] = (_Float16) (v.x*scale_q);
                qb[k][e + 1] = (_Float16) (v.y*scale_q);
                qb[k][e + 2] = (_Float16) (v.z*scale_q);
                qb[k][e + 3] = (_Float16) (v.w*scale_q);
            }
        }
    }
    __syncthreads(); // classes and sRescaleStep visible

    u32x4_t kregs[C::NKP];
    u32x4_t vregs[C::KPT];
    int cur = next_active(st_begin);
    if (cur < st_end) {
        load_K<C, Q8>(kregs, K_h, nb11u, cur*NKEYS);
        store_K<C, Q8>(kregs, sK);
    }
    __syncthreads();

    float8_t O[OT][NCT]; // [dim tile][column tile]
#pragma unroll
    for (int i = 0; i < OT; ++i) {
#pragma unroll
        for (int t = 0; t < NCT; ++t) {
            O[i][t] = float8_t{};
        }
    }
    float m = m_init; // running max and row sum of column c_s (identical in all warps of a column tile)
    float l = 0.0f;   // OWN: reference max of column c_s and row sum over key tile dq
    float l_prev = 0.0f;
    float mref[NCT];  // OWN: reference max of the PV columns 16t + l16
#pragma unroll
    for (int t = 0; t < NCT; ++t) {
        mref[t] = m_init;
    }

    // OWN: P = exp2(S - mr) of key tile dq -> B operand layout in sP; returns the column sum of P.
    // own[q] = (key 4q + hh, key 4q + 2 + hh), other[q] from lane ^ 16 = (key 4q + 1 - hh, key 4q + 3 - hh)
    auto store_P = [&](const float8_t & sc, const float mr) -> float {
        float p[8];
        float rowsum = 0.0f;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            p[r] = __builtin_amdgcn_exp2f(sc[r] - mr);
            rowsum += p[r];
        }
        uint32_t Pb[8];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const half2 h2 = __floats2half2_rn(p[2*q], p[2*q + 1]);
            const uint32_t own   = __builtin_bit_cast(uint32_t, h2);
            const uint32_t other = (uint32_t) swap_halves((int) own);
            const uint32_t lo_src = hh ? other : own;
            const uint32_t hi_src = hh ? own   : other;
            Pb[2*q + 0] = __builtin_amdgcn_perm(hi_src, lo_src, 0x05040100u);
            Pb[2*q + 1] = __builtin_amdgcn_perm(hi_src, lo_src, 0x07060302u);
        }
        if (hh == 0) {
            int4 * dstp = (int4 *) (sP + c_s*SVT + 16*dq);
            dstp[0] = make_int4(Pb[0], Pb[1], Pb[2], Pb[3]);
            dstp[1] = make_int4(Pb[4], Pb[5], Pb[6], Pb[7]);
        }
        return rowsum + swap_halves(rowsum);
    };

    // logit softcap: Q carries scale/softcap, the full scores become softcap*tanh(s) in log2 units (before the mask)
    auto softcap = [&](float8_t & s) {
        if (softcap_log2 != 0.0f) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                s[r] = softcap_log2*tanhf(s[r]);
            }
        }
    };

    while (cur < st_end) {
        const int  k0       = cur*NKEYS;
        const int  nxt      = next_active(cur + 1);
        const bool has_next = nxt < st_end;
        const bool mixed    = step_class(cur) == 2;

        // V of this step (not at D = 512) and K of the next one: in flight while the scores are computed.
        if constexpr (!C::TIGHT) {
            load_V<DV, Q8>(vregs, V_h, nb21u, k0);
        }
        if (has_next) {
            load_K<C, Q8>(kregs, K_h, nb11u, nxt*NKEYS);
        }

        // ---- partial S^T = K Q^T over dims [128dq, 128dq+128), all key tiles of the step ----
        float8_t S[KT];
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            S[kt] = float8_t{};
        }
        if constexpr (!C::TIGHT) {
            // K operands double-buffered: the loads of chunk k + 1 are in flight during the WMMAs of chunk k
            const _Float16 * ka[KT];
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                ka[kt] = sK + (16*(C::OWN ? kt ^ dq : kt) + l16)*C::SQK + (DKQ/DQ)*dq;
            }
            half16_t a[KT];
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                a[kt] = *(const half16_t *) ka[kt];
            }
#pragma unroll
            for (int k = 0; k < C::QCH; ++k) {
                half16_t an[KT];
                if (k < C::QCH - 1) {
#pragma unroll
                    for (int kt = 0; kt < KT; ++kt) {
                        an[kt] = *(const half16_t *) (ka[kt] + 16*(k + 1));
                    }
                }
                __builtin_amdgcn_sched_barrier(0); // keep the loads of chunk k + 1 ahead of the WMMAs of chunk k
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
                    S[kt] = wmma(a[kt], qb[k], S[kt]);
                    if (k < C::QCH - 1) {
                        a[kt] = an[kt];
                    }
                }
            }
        } else {
#pragma unroll
            for (int k = 0; k < C::QCH; ++k) {
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
                    S[kt] = wmma(*(const half16_t *) (sK + (16*(C::OWN ? kt ^ dq : kt) + l16)*C::SQK + (DKQ/DQ)*dq + 16*k), qb[k], S[kt]);
                }
            }
        }
        // S[kt][r]: key k0 + 16kt + 2r + hh, column c_s (OWN: S[0] is key tile dq, S[1] key tile dq ^ 1)
#pragma unroll
        for (int kt = C::OWN ? 1 : 0; kt < KT && DQ > 1; ++kt) {
            float4 * px = (float4 *) (sSx + ((C::OWN ? w : KT*w + kt)*32 + lane)*8);
            px[0] = make_float4(S[kt][0], S[kt][1], S[kt][2], S[kt][3]);
            px[1] = make_float4(S[kt][4], S[kt][5], S[kt][6], S[kt][7]);
        }
        lds_barrier(); // B1: partial scores visible, all warps are done reading K

        if constexpr (C::OWN) {
            // Own key tile: add the share of the partner warp. The reference max m moves lazily (as in FlashAttention-4),
            // after V^T P and only when a step max is more than T_LAZY above it; P up to 2^T_HI is exact in f16.
            {
                const float4 * px = (const float4 *) (sSx + ((ct*DQ + (dq ^ 1))*32 + lane)*8);
                const float4 x0 = px[0];
                const float4 x1 = px[1];
                S[0][0] += x0.x; S[0][1] += x0.y; S[0][2] += x0.z; S[0][3] += x0.w;
                S[0][4] += x1.x; S[0][5] += x1.y; S[0][6] += x1.z; S[0][7] += x1.w;
            }
            softcap(S[0]);
            if (mixed) {
                const int row = min(jt*NCOLS1 + c_s/NCOLS2, ne31 - 1);
                const int4 * mp = (const int4 *) (mask_h + row*s31 + k0 + 16*dq);
                const int4 m0 = mp[0];
                const int4 m1 = mp[1];
                const uint32_t md[8] = {(uint32_t) m0.x, (uint32_t) m0.y, (uint32_t) m0.z, (uint32_t) m0.w,
                                        (uint32_t) m1.x, (uint32_t) m1.y, (uint32_t) m1.z, (uint32_t) m1.w};
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const uint16_t bits = (uint16_t) (hh ? md[r] >> 16 : md[r] & 0xFFFFu);
                    S[0][r] += __half2float(__ushort_as_half(bits)) * log2e;
                }
            }
            float mx = S[0][0];
#pragma unroll
            for (int r = 1; r < 8; ++r) {
                mx = fmaxf(mx, S[0][r]);
            }
            mx = fmaxf(mx, swap_halves(mx));
            if (hh == 0) {
                sMax[(dq*16 + l16)*NCT + ct] = mx;
            }
            if (mx > m + T_HI) {
                sRedo = cur;
            }
            l_prev = l;
            l += store_P(S[0], m);
        } else if (dq == 0) {
            // With one key tile (D = 512) only warp dq = 0 of a column tile uses the softmax (P, rescale factors, row sums);
            // at D <= 128 one warp owns a column tile.
            // ---- full scores: add the shares of the other warps of the column tile ----
#pragma unroll
            for (int j = 1; j < DQ; ++j) {
                const int wp = ct*DQ + (dq + j) % DQ;
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
                    const float4 * px = (const float4 *) (sSx + ((KT*wp + kt)*32 + lane)*8);
                    const float4 x0 = px[0];
                    const float4 x1 = px[1];
                    S[kt][0] += x0.x; S[kt][1] += x0.y; S[kt][2] += x0.z; S[kt][3] += x0.w;
                    S[kt][4] += x1.x; S[kt][5] += x1.y; S[kt][6] += x1.z; S[kt][7] += x1.w;
                }
            }
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                softcap(S[kt]);
            }
            if (mixed) {
                // lane needs mask[token of column c_s][k0 + 16kt + 2r + hh], r = 0..7
                const int row = min(jt*NCOLS1 + c_s/NCOLS2, ne31 - 1);
                const int4 * mp = (const int4 *) (mask_h + row*s31 + k0);
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
                    const int4 m0 = mp[2*kt + 0];
                    const int4 m1 = mp[2*kt + 1];
                    const uint32_t md[8] = {(uint32_t) m0.x, (uint32_t) m0.y, (uint32_t) m0.z, (uint32_t) m0.w,
                                            (uint32_t) m1.x, (uint32_t) m1.y, (uint32_t) m1.z, (uint32_t) m1.w};
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        // dword r holds keys 2r (low half) and 2r + 1 (high half)
                        const uint16_t bits = (uint16_t) (hh ? md[r] >> 16 : md[r] & 0xFFFFu);
                        S[kt][r] += __half2float(__ushort_as_half(bits)) * log2e; // S is in log2 units
                    }
                }
            }

            // ---- softmax over the keys of the step ----
            {
                float mx = S[0][0];
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        mx = fmaxf(mx, S[kt][r]);
                    }
                }
                const float m_new = fmaxf(m, fmaxf(mx, swap_halves(mx)));
                const float alpha = __builtin_amdgcn_exp2f(m - m_new);
                m = m_new;

                float rowsum = 0.0f;
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        S[kt][r] = __builtin_amdgcn_exp2f(S[kt][r] - m_new);
                        rowsum += S[kt][r];
                    }
                }
                l = l*alpha + rowsum + swap_halves(rowsum);
                if (dq == 0 && hh == 0) {
                    sAlpha[c_s] = alpha;
                    if (alpha != 1.0f) {
                        sRescaleStep = cur;
                    }
                }

                // P of the key tiles kt % DQ == dq -> B operand layout.
                // own[q] = (key 4q + hh, key 4q + 2 + hh), other[q] from lane ^ 16 = (key 4q + 1 - hh, key 4q + 3 - hh)
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
                    if (kt % DQ != dq) {
                        continue;
                    }
                    uint32_t Pb[8];
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const half2 h2 = __floats2half2_rn(S[kt][2*q], S[kt][2*q + 1]);
                        const uint32_t own   = __builtin_bit_cast(uint32_t, h2);
                        const uint32_t other = (uint32_t) swap_halves((int) own);
                        const uint32_t lo_src = hh ? other : own; // provides the even key of each pair
                        const uint32_t hi_src = hh ? own   : other;
                        Pb[2*q + 0] = __builtin_amdgcn_perm(hi_src, lo_src, 0x05040100u); // keys 4q,   4q+1
                        Pb[2*q + 1] = __builtin_amdgcn_perm(hi_src, lo_src, 0x07060302u); // keys 4q+2, 4q+3
                    }
                    if (hh == 0) {
                        int4 * dstp = (int4 *) (sP + c_s*SVT + 16*kt);
                        dstp[0] = make_int4(Pb[0], Pb[1], Pb[2], Pb[3]);
                        dstp[1] = make_int4(Pb[4], Pb[5], Pb[6], Pb[7]);
                    }
                }
            }
        }

        // ---- stage V^T of this step and K of the next one (both buffers are free after B1) ----
        if constexpr (C::TIGHT) {
            load_V<DV, Q8>(vregs, V_h, nb21u, k0);
        }
        store_Vt<DV, Q8>(vregs, sVt);
        if (has_next) {
            store_K<C, Q8>(kregs, sK);
        }
        lds_barrier(); // B2: P, V^T, rescale factors and the next K visible

        // ---- O^T = alpha*O^T + V^T P for dims [32w, 32w+32) ----
        float alpha[NCT];
        bool  redo  = false;
        bool  moved = false;
        if constexpr (C::OWN) {
            redo = sRedo == cur;
#pragma unroll
            for (int t = 0; t < NCT; ++t) {
                const float smax = fmaxf(sMax[l16*NCT + t], sMax[(16 + l16)*NCT + t]);
                const float mn   = redo || smax > mref[t] + T_LAZY ? fmaxf(mref[t], smax) : mref[t];
                alpha[t] = __builtin_amdgcn_exp2f(mref[t] - mn);
                moved    = moved || mn != mref[t];
                mref[t]  = mn;
            }
            float alpha_s = alpha[0];
            float m_s     = mref[0];
#pragma unroll
            for (int t = 1; t < NCT; ++t) {
                if (ct == t) {
                    alpha_s = alpha[t];
                    m_s     = mref[t];
                }
            }
            m = m_s;
            if (redo) {
                // P of this step left the f16 range against the old max: rescale first and redo P (first steps only, in practice)
#pragma unroll
                for (int t = 0; t < NCT; ++t) {
#pragma unroll
                    for (int i = 0; i < OT; ++i) {
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            O[i][t][r] *= alpha[t];
                        }
                    }
                }
                l = l_prev*alpha_s + store_P(S[0], m);
                lds_barrier();
            } else {
                l *= alpha_s;
            }
        } else if (sRescaleStep == cur) {
#pragma unroll
            for (int t = 0; t < NCT; ++t) {
                const float alpha = sAlpha[16*t + l16];
#pragma unroll
                for (int i = 0; i < OT; ++i) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        O[i][t][r] *= alpha;
                    }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < KT && !C::TIGHT; ++c) {
            half16_t b[NCT];
#pragma unroll
            for (int t = 0; t < NCT; ++t) {
                b[t] = *(const half16_t *) (sP + (16*t + l16)*SVT + 16*c);
            }
#pragma unroll
            for (int i = 0; i < OT; ++i) {
                const half16_t a = load_Vt_operand<DV>(sVt, 16*OT*w + 16*i + l16, c);
#pragma unroll
                for (int t = 0; t < NCT; ++t) {
                    O[i][t] = wmma(a, b[t], O[i][t]);
                }
            }
        }
#pragma unroll
        for (int c = 0; c < KT && C::TIGHT; ++c) {
            half16_t a[OT];
#pragma unroll
            for (int i = 0; i < OT; ++i) {
                a[i] = load_Vt_operand<DV>(sVt, 16*OT*w + 16*i + l16, c);
            }
#pragma unroll
            for (int t = 0; t < NCT; ++t) {
                const half16_t b = *(const half16_t *) (sP + (16*t + l16)*SVT + 16*c);
#pragma unroll
                for (int i = 0; i < OT; ++i) {
                    O[i][t] = wmma(a[i], b, O[i][t]);
                }
            }
        }
        if (C::OWN && !redo && __any(moved)) {
#pragma unroll
            for (int t = 0; t < NCT; ++t) {
#pragma unroll
                for (int i = 0; i < OT; ++i) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        O[i][t][r] *= alpha[t];
                    }
                }
            }
        }
        cur = nxt;
    }

    // ---- normalize by the row sums and write (split-KV: partials and (max, row sum) in natural log units) ----
    float * sM = sSx; // column maxima (not OWN), nobody reads sSx after the last B2
    if (C::OWN ? hh == 0 : dq == 0 && hh == 0) {
        sL[(C::OWN ? dq*NCOLS : 0) + c_s] = l;
        if (!C::OWN) {
            sM[c_s] = m;
        }
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < NCT; ++t) {
        const int c = 16*t + l16;
        const int token = jt*NCOLS1 + c/NCOLS2;
        if (token >= ne01 || c >= NUSED) {
            continue;
        }
        const int head = head0 + c % NCOLS2;
        float l_sum = C::OWN ? sL[c] + sL[NCOLS + c] : sL[c];
        float m_c   = C::OWN ? mref[t] : sM[c];
        float o_scl = 1.0f;
        if (sinks && blockIdx.y == 0) {
            // the sink logit of the head joins the softmax denominator (once, in the first split)
            const float s = sinks[head]*log2e;
            if (s > m_c) {
                o_scl = __builtin_amdgcn_exp2f(m_c - s);
                l_sum = l_sum*o_scl + 1.0f;
                m_c   = s;
            } else {
                l_sum += __builtin_amdgcn_exp2f(s - m_c);
            }
        }
        const int64_t j_dst = (int64_t(seq)*ne01 + token)*ne02 + head;
        if (nsplit > 1) {
            float * part = dst_parts + (j_dst*nsplit + blockIdx.y)*DV;
#pragma unroll
            for (int i = 0; i < OT; ++i) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    part[16*OT*w + 16*i + 2*r + hh] = O[i][t][r]*o_scl;
                }
            }
            if (w == 0 && hh == 0) {
                dst_meta[j_dst*nsplit + blockIdx.y] = make_float2(m_c * (1.0f/log2e), l_sum);
            }
            continue;
        }
        const float inv = l_sum > 0.0f ? o_scl/l_sum : 0.0f;
        float * dst_c = dst + j_dst*DV;
#pragma unroll
        for (int i = 0; i < OT; ++i) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                dst_c[16*OT*w + 16*i + 2*r + hh] = O[i][t][r]*inv;
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, mask_cls, KV_max, dst, dst_parts, dst_meta, sinks, scale_q, softcap_log2, ne01, ne02, ne11, gqa_ratio,
        nb01, nb02, nb03, nb11, nb12, nb13, nb21, nb22, nb23, ne31, ne33, nb31, nb33);
    NO_DEVICE_CODE;
#endif // FATTN_RDNA35_AVAILABLE
}

// Classifies every (tile of NCOLS1 tokens, 32 keys) of the mask: 0 = all zero (mask can be ignored),
// 1 = all -inf (keys can be skipped), 2 = mixed, packed as 2 bits each.
// A wave covers 256 keys of a row with 16 B per lane (4 lanes per class) and walks the NCOLS1 rows of the tile.
// The mask is read once per FA call: at 32k keys this is ~140 MB, so the loads must be wide.
constexpr int CLS_WAVES = 4;
constexpr int CLS_BLOCK_KEYS = CLS_WAVES*32*8;

template <int NCOLS1>
__launch_bounds__(CLS_WAVES*32, 1)
static __global__ void mask_classify(const half * __restrict__ mask, uint8_t * __restrict__ cls,
        const int ne31, const int64_t s31, const int64_t s33, const int n_cls, const int n_bytes) {
    const int jt   = blockIdx.x;
    const int seq  = blockIdx.z;
    const int lane = threadIdx.x % 32;
    const int i0   = (blockIdx.y*CLS_BLOCK_KEYS + (threadIdx.x/32)*256) / CLS_KEYS; // first class of the wave

    bool zero = true;
    bool ninf = true;
    if (i0 + lane/4 < n_cls) {
        const half * m = mask + seq*s33 + int64_t(i0)*CLS_KEYS + 8*lane;
#pragma unroll 8
        for (int j = 0; j < NCOLS1; ++j) {
            const uint4 v = *(const uint4 *) (m + int64_t(min(jt*NCOLS1 + j, ne31 - 1))*s31);
            zero = zero && ((v.x | v.y | v.z | v.w) & 0x7FFF7FFFu) == 0;
            ninf = ninf && (v.x & v.y & v.z & v.w) == 0xFC00FC00u && (v.x | v.y | v.z | v.w) == 0xFC00FC00u;
        }
    }
    const uint32_t bz = (uint32_t) __ballot(zero);
    const uint32_t bi = (uint32_t) __ballot(ninf);
    if (lane == 0) {
        uint32_t packed = 0; // 8 classes, 2 bits each
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const uint32_t z = (bz >> 4*c) & 0xF;
            const uint32_t n = (bi >> 4*c) & 0xF;
            packed |= (n == 0xF ? 1u : (z == 0xF ? 0u : 2u)) << 2*c;
        }
        uint8_t * dst = cls + (int64_t(seq)*gridDim.x + jt)*n_bytes;
        const int b0 = i0/4;
        if (b0 < n_bytes) {
            dst[b0] = (uint8_t) packed;
        }
        if (b0 + 1 < n_bytes) {
            dst[b0 + 1] = (uint8_t) (packed >> 8);
        }
    }
}

} // namespace fattn_rdna35

static int fattn_rdna35_ncols2(const int gqa_ratio) {
    for (int n : {8, 4, 2}) {
        if (gqa_ratio % n == 0) {
            return n;
        }
    }
    return 1;
}

// f16, or a type with a strided f16 converter (see fattn_rdna35_to_f16)
static bool fattn_rdna35_kv_type_ok(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F16 || (ggml_get_to_fp16_nc_cuda(t->type) != nullptr && t->nb[0] == ggml_type_size(t->type));
}

// row stride of t as the kernel reads it (after the conversion to contiguous f16)
static int64_t fattn_rdna35_nb1(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F16 ? t->nb[1] : t->ne[0]*int64_t(sizeof(half));
}

// t -> contiguous f16 [ne0, ne1, ne2, ne3] at dst_f16, nb = its byte strides
static void fattn_rdna35_to_f16(const ggml_tensor * t, half * dst_f16, int64_t * nb, cudaStream_t stream) {
    const int64_t ts = ggml_type_size(t->type);
    ggml_get_to_fp16_nc_cuda(t->type)(t->data, dst_f16, t->ne[0], t->ne[1], t->ne[2], t->ne[3],
        t->nb[1]/ts, t->nb[2]/ts, t->nb[3]/ts, stream);
    nb[1] = t->ne[0]*int64_t(sizeof(half));
    nb[2] = t->ne[1]*nb[1];
    nb[3] = t->ne[2]*nb[2];
}

// q8_0 K and V of a small batch are read and dequantized by the kernel, which is bound by the K/V reads there.
// Prefill is bound by the WMMAs and converts them to f16 first (the dequantization would cost more than that).
// D = 64 single-token batches stay on the generic tile kernel: this kernel holds 256 VGPRs (8 waves per CU),
// and at that occupancy the q8 staging does not hide the read latency (LFM2.5 24B-A2B, 64k keys, gfx1151:
// 532 us per call against 471 us for flash_attn_tile; D = 128 keeps ~2x on this kernel, see the launcher note).
static bool fattn_rdna35_q8_direct(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    auto aligned4 = [](const ggml_tensor * t) { return t->nb[1] % 4 == 0 && t->nb[2] % 4 == 0 && t->nb[3] % 4 == 0; };
    return (Q->ne[0] != 64 || Q->ne[1] > 1) &&
        Q->ne[1]*fattn_rdna35_ncols2(Q->ne[2] / K->ne[2]) < 256 &&
        K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0 && aligned4(K) && aligned4(V);
}

size_t ggml_cuda_flash_attn_ext_rdna35_alloc_size(const ggml_tensor * dst) {
    const bool convert = !fattn_rdna35_q8_direct(dst);
    return ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, convert, convert).end - (uintptr_t) dst->data;
}

bool ggml_cuda_flash_attn_ext_rdna35_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    return ggml_cuda_flash_attn_ext_rdna35_supported(ctx.device, dst);
}

bool ggml_cuda_flash_attn_ext_rdna35_supported(const int device, const ggml_tensor * dst) {
    static const int mode = getenv("GGML_FA_RDNA35") ? atoi(getenv("GGML_FA_RDNA35")) : 1; // 0 off, 1 auto, 2 force
    if (mode == 0) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return false;
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    const int64_t DKQ = Q->ne[0];
    const int64_t DV  = V->ne[0];
    // absorbed MLA (deepseek2 family): all cache state is in the K cache (kv_lora_rank + rope wide), V is a
    // strided view of those rows reading only its first kv_lora_rank dims; n_head_kv == 1 (MQA). Both K and V
    // reach this op through the same permute of the cache views, so they alias at the buffer base and share the
    // key stride (V->nb[1] == K->nb[1]) while V's rows are only kv_lora_rank wide
    const bool mla = DKQ == 576 && K->ne[0] == 576 && DV == 512 &&
        V->data == K->data && V->nb[1] == K->nb[1] && V->nb[3] == K->nb[3] &&
        V->ne[1] == K->ne[1] && V->ne[2] == K->ne[2] && V->type == K->type;
    if (!mla && ((DKQ != 64 && DKQ != 128 && DKQ != 256 && DKQ != 512) || K->ne[0] != DKQ || DV != DKQ)) return false;
    if (Q->type != GGML_TYPE_F32 || !fattn_rdna35_kv_type_ok(K) || !fattn_rdna35_kv_type_ok(V)) return false;
    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) return false;
    if (dst->src[5] || max_bias != 0.0f) return false;
    const ggml_tensor * sinks = dst->src[4];
    if (sinks && (sinks->type != GGML_TYPE_F32 || !ggml_is_contiguous(sinks) || sinks->ne[0] < Q->ne[2])) return false;
    if (K->ne[1] % 64 != 0 || K->ne[1] > int64_t(fattn_rdna35::CLS_KEYS)*fattn_rdna35::MAX_CLS || V->ne[1] != K->ne[1]) return false;
    if (Q->ne[2] % K->ne[2] != 0 || V->ne[2] != K->ne[2] || K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) return false;
    if (Q->nb[0] != sizeof(float)) return false;
    const int64_t nbK1 = fattn_rdna35_nb1(K);
    const int64_t nbV1 = fattn_rdna35_nb1(V);
    if (Q->nb[1] % 16 || Q->nb[2] % 16 || nbK1 % 16 || nbV1 % 16) return false;
    if (K->ne[1]*nbK1 >= (int64_t(1) << 32) || V->ne[1]*nbV1 >= (int64_t(1) << 32)) return false; // 32-bit row offsets
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->nb[1] % 16 || mask->nb[3] % 16 || mask->ne[1] < Q->ne[1])) return false;

    // prefill: enough columns per K/V head to fill the 64-column tiles
    const int gqa_ratio = Q->ne[2] / K->ne[2];
    if (mode == 2 || Q->ne[1]*fattn_rdna35_ncols2(gqa_ratio) >= 256) {
        return true;
    }
    // small batches at depth (decode, speculative verification), key steps split over blocks: faster than the generic
    // kernels from 2 tokens with f16 K/V and from 1 token with q8_0 K/V (read by the kernel). Per call at 32k keys,
    // 8 tokens, f16 / q8_0: Qwen3.6 35B-A3B 577 / 865 -> 330 / 188 us, Qwen3.8 27B 1635 / 2750 -> 597 / 385 us.
    // Single-token f16 decode stays on the tile kernel (279 vs 325 us); below 4k keys the generic kernels are as fast.
    const bool f16 = K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;
    // MQA decode (MLA): one block streams the whole cache once for all Q heads, unlike the GQA case where the
    // tile kernel's re-reads are as cheap as the split
    return K->ne[1] >= 4096 && ((f16 && (Q->ne[1] >= 2 || DV != DKQ)) || fattn_rdna35_q8_direct(dst));
}

template <int DKQ, int DV, int NCOLS2>
static void launch_fattn_rdna35(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    using namespace fattn_rdna35;
    constexpr int NCOLS1 = cfg<DKQ, DV>::NCOLS / NCOLS2;

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    cudaStream_t stream = ctx.stream();

    float scale, logit_softcap;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int ntiles = (Q->ne[1] + NCOLS1 - 1) / NCOLS1;

    // KV_max skips the fully masked tail of the KV cache (causal mask) per tile of NCOLS1 tokens, the mask classes per
    // 32 keys let the kernel skip hidden keys and ignore the mask where it is all zero. Both depend only on the mask
    // and the tile height: computed by the first attention layer of a graph compute that reads the mask, reused by
    // the others (Qwen3.6 @ 32k: 2.6 of the 65 ms per layer and 4096-token ubatch).
    int *     kv_max_ptr   = nullptr;
    uint8_t * mask_cls_ptr = nullptr;
    ggml_cuda_pool_alloc<char> mask_info_tmp(ctx.pool());
    if (mask) {
        const bool    want_kv_max  = K->ne[1] % FATTN_KQ_STRIDE == 0 && mask->ne[1] >= int64_t(ntiles)*NCOLS1; // the scan reads whole tiles
        const int     n_cls        = K->ne[1] / CLS_KEYS;
        const int     n_bytes      = (n_cls + 3) / 4;
        const size_t  kv_max_bytes = want_kv_max ? GGML_PAD(size_t(ntiles)*Q->ne[3]*sizeof(int), 256) : 0;
        const size_t  size         = kv_max_bytes + size_t(ntiles)*n_bytes*mask->ne[3];
        const int64_t key[9] = { (int64_t) (uintptr_t) mask->data, mask->ne[1], mask->ne[3], (int64_t) mask->nb[1],
                                 (int64_t) mask->nb[3], K->ne[1], NCOLS1, Q->ne[1], Q->ne[3] };

        char * buf   = nullptr;
        bool   fresh = true;
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        if (capture != cudaStreamCaptureStatusNone) { // no allocations while capturing a graph: per call
            buf = mask_info_tmp.alloc(size);
        } else {
            ggml_backend_cuda_context::fattn_mask_info * slot = nullptr;
            for (auto & m : ctx.fattn_masks) {
                if (m.epoch == ctx.compute_epoch && memcmp(m.key, key, sizeof(key)) == 0 && m.size >= size) {
                    slot = &m; fresh = false; break;
                }
            }
            if (fresh) {
                for (auto & m : ctx.fattn_masks) { // a slot of an earlier compute
                    if (m.epoch != ctx.compute_epoch) {
                        slot = &m;
                        break;
                    }
                }
                if (slot == nullptr) { // more masks in this compute than slots: the next reader of the evicted one recomputes it
                    slot = &ctx.fattn_masks[0];
                }
                if (slot->size < size) {
                    if (slot->buf != nullptr) {
                        CUDA_CHECK(cudaFree(slot->buf)); // synchronizes with the device: no kernel reads it anymore
                    }
                    slot->size = size + size/4;
                    CUDA_CHECK(cudaMalloc(&slot->buf, slot->size));
                }
                slot->epoch = ctx.compute_epoch;
                memcpy(slot->key, key, sizeof(key));
            }
            buf = (char *) slot->buf;
        }
        kv_max_ptr   = want_kv_max ? (int *) buf : nullptr;
        mask_cls_ptr = (uint8_t *) buf + kv_max_bytes;

        if (fresh) {
            if (want_kv_max) {
                const int64_t s31 = mask->nb[1] / sizeof(half2);
                const int64_t s33 = mask->nb[3] / sizeof(half2);
                flash_attn_mask_to_KV_max<NCOLS1><<<dim3(ntiles, Q->ne[3], 1), dim3(FATTN_KQ_STRIDE/2, 1, 1), 0, stream>>>(
                    (const half2 *) mask->data, kv_max_ptr, K->ne[1] / FATTN_KQ_STRIDE, s31, s33);
                CUDA_CHECK(cudaGetLastError());
            }
            const int n_blk = (K->ne[1] + CLS_BLOCK_KEYS - 1) / CLS_BLOCK_KEYS;
            mask_classify<NCOLS1><<<dim3(ntiles, n_blk, mask->ne[3]), dim3(CLS_WAVES*32, 1, 1), 0, stream>>>(
                (const half *) mask->data, mask_cls_ptr, mask->ne[1], mask->nb[1] / sizeof(half), mask->nb[3] / sizeof(half), n_cls, n_bytes);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // small batches: split the key steps over more blocks (>= 16 steps each) to fill the GPU
    const int nblk      = ntiles*(Q->ne[2] / NCOLS2)*Q->ne[3];
    const int nsm       = ggml_cuda_info().devices[ctx.device].nsm;
    const int max_split = std::max(1, int(K->ne[1] / (16*cfg<DKQ, DV>::NKEYS)));
    const int nsplit    = std::min(max_split, std::max(1, (2*nsm + nblk - 1) / nblk));

    const bool q8_direct = fattn_rdna35_q8_direct(dst);

    // a K/V cache of another type: f16 copies in the scratch space behind dst. For absorbed MLA, V reads the
    // first kv_lora_rank dims of the same (possibly wider) rows as K: the tensors reach this op through permutes
    // of the same cache view, so the aliasing is decided from the assigned pointers, not from view_src
    const char * K_data = (const char *) K->data;
    const char * V_data = (const char *) V->data;
    int64_t nbK[4] = {0, (int64_t) K->nb[1], (int64_t) K->nb[2], (int64_t) K->nb[3]};
    int64_t nbV[4] = {0, (int64_t) V->nb[1], (int64_t) V->nb[2], (int64_t) V->nb[3]};
    const bool v_of_k = V->data == K->data && V->type == K->type &&
        V->ne[1] == K->ne[1] && V->ne[2] == K->ne[2] && V->ne[3] == K->ne[3] &&
        V->nb[1] == K->nb[1] && V->nb[2] == K->nb[2] && V->nb[3] == K->nb[3];
    if (!q8_direct && (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16)) {
        const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra = ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, true, true);
        if (K->type != GGML_TYPE_F16) {
            fattn_rdna35_to_f16(K, (half *) f16_extra.K, nbK, stream);
            K_data = (const char *) f16_extra.K;
        }
        if (V->type != GGML_TYPE_F16) {
            if (v_of_k) {
                memcpy(nbV, nbK, sizeof(nbK)); // V reads the converted K rows
                V_data = K_data;
            } else {
                fattn_rdna35_to_f16(V, (half *) f16_extra.V, nbV, stream);
                V_data = (const char *) f16_extra.V;
            }
        }
    }

    ggml_cuda_pool_alloc<float>  parts(ctx.pool());
    ggml_cuda_pool_alloc<float2> meta(ctx.pool());
    if (nsplit > 1) {
        parts.alloc(size_t(nsplit)*Q->ne[1]*Q->ne[2]*Q->ne[3]*DV);
        meta.alloc(size_t(nsplit)*Q->ne[1]*Q->ne[2]*Q->ne[3]);
    }

    const dim3 grid(ntiles*(Q->ne[2] / NCOLS2), nsplit, Q->ne[3]);
    auto * kernel = q8_direct ? flash_attn_rdna35<DKQ, DV, NCOLS2, true> : flash_attn_rdna35<DKQ, DV, NCOLS2, false>;
    kernel<<<grid, dim3(cfg<DKQ, DV>::NTHR, 1, 1), 0, stream>>>(
        (const char *) Q->data, K_data, V_data,
        mask ? (const char *) mask->data : nullptr, mask_cls_ptr, kv_max_ptr, (float *) dst->data, parts.ptr, meta.ptr,
        dst->src[4] ? (const float *) dst->src[4]->data : nullptr,
        logit_softcap != 0.0f ? scale/logit_softcap : scale*log2e, logit_softcap*log2e,
        Q->ne[1], Q->ne[2], K->ne[1], Q->ne[2] / K->ne[2],
        Q->nb[1], Q->nb[2], Q->nb[3],
        nbK[1], nbK[2], nbK[3],
        nbV[1], nbV[2], nbV[3],
        mask ? mask->ne[1] : 0, mask ? mask->ne[3] : 1, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0);
    CUDA_CHECK(cudaGetLastError());

    if (nsplit > 1) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
            dim3(Q->ne[1], Q->ne[2], Q->ne[3]), dim3(DV, 1, 1), nsplit*sizeof(float2), stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<DV>, launch_params, parts.ptr, meta.ptr, (float *) dst->data, nsplit);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <int DKQ, int DV = DKQ>
static void launch_fattn_rdna35_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int gqa_ratio = dst->src[0]->ne[2] / dst->src[1]->ne[2];
    // GQA 16 small batches (decode, speculative verification): the 8-head grouping reads every K/V row twice,
    // all 16 Q heads of a K/V head fit one block. Batches of 16+ tokens keep the 8-head grouping (prefill path).
    if (gqa_ratio % 16 == 0 && dst->src[0]->ne[1]*16 < 256) {
        launch_fattn_rdna35<DKQ, DV, 16>(ctx, dst);
        return;
    }
    // small batches of GQA 6 models: all 6 Q heads of a K/V head in one block, so that every K/V row is read once
    // (60 of 64 columns); prefill keeps 2 heads x 32 tokens
    if (gqa_ratio % 6 == 0 && dst->src[0]->ne[1] <= 32) {
        launch_fattn_rdna35<DKQ, DV, 6>(ctx, dst);
        return;
    }
    if (DV != DKQ && dst->src[0]->ne[1] <= 12) {
        // absorbed MLA (MQA) small batches: one block takes all Q heads of the single K/V head, so every cache
        // row is read once; prefill uses the pow2 grouping below (8 tokens x 4 heads)
        const int n2 = std::min((int) gqa_ratio, fattn_rdna35::cfg<DKQ, DV>::NCOLS);
        GGML_ASSERT(dst->src[0]->ne[2] % n2 == 0 && "unexpected MLA head grouping");
        switch (n2) {
            case 1:  launch_fattn_rdna35<DKQ, DV,  1>(ctx, dst); break;
            case 2:  launch_fattn_rdna35<DKQ, DV,  2>(ctx, dst); break;
            case 4:  launch_fattn_rdna35<DKQ, DV,  4>(ctx, dst); break;
            case 8:  launch_fattn_rdna35<DKQ, DV,  8>(ctx, dst); break;
            case 16: launch_fattn_rdna35<DKQ, DV, 16>(ctx, dst); break;
            case 20: launch_fattn_rdna35<DKQ, DV, 20>(ctx, dst); break;
            case 32: launch_fattn_rdna35<DKQ, DV, 32>(ctx, dst); break;
            default: GGML_ABORT("unsupported MLA head grouping");
        }
        return;
    }
    switch (fattn_rdna35_ncols2(gqa_ratio)) {
        case 8:  launch_fattn_rdna35<DKQ, DV, 8>(ctx, dst); break;
        case 4:  launch_fattn_rdna35<DKQ, DV, 4>(ctx, dst); break;
        case 2:  launch_fattn_rdna35<DKQ, DV, 2>(ctx, dst); break;
        default: launch_fattn_rdna35<DKQ, DV, 1>(ctx, dst); break;
    }
}

void ggml_cuda_flash_attn_ext_rdna35(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (dst->src[2]->ne[0] != dst->src[0]->ne[0]) { // MLA absorbed: V is a strided view of the K cache rows
        launch_fattn_rdna35_ncols2<576, 512>(ctx, dst);
        return;
    }
    switch (dst->src[0]->ne[0]) {
        case 64:  launch_fattn_rdna35_ncols2< 64>(ctx, dst); break;
        case 128: launch_fattn_rdna35_ncols2<128>(ctx, dst); break;
        case 256: launch_fattn_rdna35_ncols2<256>(ctx, dst); break;
        default:  launch_fattn_rdna35_ncols2<512>(ctx, dst); break;
    }
}
