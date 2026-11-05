#include <type_traits>
// C5 prototype attend kernel, derived line by line from the PR's qsa3_attn_kernel<DIRECT=true, 8, DIRECT_V=false, Q8>
// (PR 94e086847 ggml/src/ggml-cuda/qsa-prefill.cu:254-601): K read direct (f16 or q8_0), V from the qsa_pack_v layout,
// no mask, no gate. Knobs:
//   NH   halves per workgroup: 1 = G4 (256 threads), 2 = G8 (512 threads: two G4 halves walk one G8 union, each with
//        its own 4 queries, Q fragments, softmax state and accumulators; K/V of a chunk is fetched by both halves)
//   RPQ  rows per query in the WMMA row space: 12 (today's packing, 48 rows = 3 tiles) or 16 (one 16-row tile per query,
//        heads 12..15 are zero pad rows: plan-B.md:10 "tS/q")
//   SKIP 1 = per chunk scalar branches; 2 = tile-set specialised loops; 3 = full body + branch body; 4 = two sequential
//        loops (BUCKET plans); 16+T = timing oracle with the fixed tile set T. Per chunk, skip the row tiles whose queries select none of the chunk's 4 quads (QK WMMA, LDS partials, softmax,
//        rescale, PV WMMA); K/V loads are never skipped beyond the PR's per-lane K mask
//   MT   per-entry mask type, uint16_t (bit 4*qi + cell); for G8 the plan holds two mask arrays per group, [g][half][cap]
// With NH=1, RPQ=12, SKIP=false it performs the same arithmetic in the same order as the PR kernel (checked bit-exact).
template <int RPQ> struct c5_tiles {
    static constexpr int NT = (4 * RPQ) / 16;
    // query-presence bits (4 queries of a half) that make tile i active
    static __device__ __forceinline__ uint32_t tmask(const int i) {
        uint32_t m = 0;
        for (int j = 0; j < 4; ++j) { const int r0 = RPQ * j, r1 = RPQ * j + 11; if (r1 >= 16 * i && r0 <= 16 * i + 15) m |= 1u << j; }
        return m;
    }
};

template<int NH, int RPQ, int SKIP, bool Q8, typename MT> __global__ __launch_bounds__(256 * NH) __attribute__((amdgpu_waves_per_eu(8)))
void c5_attn_kernel(
        const float * __restrict__ q, const uint16_t * __restrict__ pk, const uint16_t * __restrict__ pv,
        const uint32_t * __restrict__ ublk, const MT * __restrict__ umask,
        const int * __restrict__ ucount, float * __restrict__ out, const qsa3_layout s) {
    constexpr int NT = c5_tiles<RPQ>::NT;
    constexpr int NR = 16 * NT;
    const int tid = threadIdx.x, half = NH == 1 ? 0 : __builtin_amdgcn_readfirstlane(tid >> 8), lt = NH == 1 ? tid : (tid & 255), lane = lt & 31, w = lt >> 5, r = lane & 15, hi = lane >> 4;
    const int g = blockIdx.x, kvh = blockIdx.y;
    __shared__ float4 part_all[NH][NT][8][64];
    __shared__ uint4  ptile_all[NH][NT][32];
    __shared__ float  alpha_all[NH][NR];
    __shared__ float  l_all[NH][NR];
    float4 (*part)[8][64] = part_all[half];
    uint4  (*ptile)[32]   = ptile_all[half];
    float  * alpha_s      = alpha_all[half];
    float  * l_s          = l_all[half];

    const size_t nblk = (size_t) s.nk / 4;
    const char * kbase = reinterpret_cast<const char *>(pk) + (size_t) kvh * s.knb2;
    const uint16_t * pvg = pv + (size_t) kvh * nblk * 1024;
    const uint32_t * gblk = ublk + (size_t) g * s.cap;
    const MT * gmsk = umask + ((size_t) g * NH + half) * s.cap;   // G8: per group two u16 mask arrays, one per half
    const int nchunks = ucount[g] >> 2;
    const int qbase = 4 * NH * g + 4 * half;

    uint32_t tm[NT];
#pragma unroll
    for (int i = 0; i < NT; ++i) { tm[i] = c5_tiles<RPQ>::tmask(i); }

    v16s qf[NT][2];
    {
        uint16_t * stage = reinterpret_cast<uint16_t *>(&part[0][0][0]) + w * (NR * 32);
        const _Float16 hs = (_Float16) s.scale;
#pragma unroll
        for (int c = 0; c < NR / 4; ++c) {
            const int row = 4*c + (lane >> 3), dq = 4 * (lane & 7);
            const int qi = row / RPQ, h = row - RPQ*qi;
            const int query = qbase + qi;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (query < s.n_q && (RPQ == 12 || h < 12)) {
                v = *reinterpret_cast<const float4 *>(reinterpret_cast<const char *>(q) + (size_t) query * s.q1 + (size_t) (kvh * s.gqa + h) * s.q2 + (size_t) (32*w + dq) * 4);
            }
            const uint16_t h0 = __builtin_bit_cast(uint16_t, (_Float16) v.x * hs), h1 = __builtin_bit_cast(uint16_t, (_Float16) v.y * hs);
            const uint16_t h2 = __builtin_bit_cast(uint16_t, (_Float16) v.z * hs), h3 = __builtin_bit_cast(uint16_t, (_Float16) v.w * hs);
            *reinterpret_cast<uint2 *>(stage + row * 32 + dq) = make_uint2((uint32_t) h0 | ((uint32_t) h1 << 16), (uint32_t) h2 | ((uint32_t) h3 << 16));
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const uint16_t * src = stage + (16*i + r) * 32 + 16*t;
                qf[i][t] = __builtin_bit_cast(v16s, (uint4[2]){*reinterpret_cast<const uint4 *>(src), *reinterpret_cast<const uint4 *>(src + 8)});
            }
        __syncthreads();
    }
    v8f O[NT][2];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) { O[i][t][e] = 0.f; }
    float m = -INFINITY, l = 0.f;
    const int own_row = 16*w + r, own_qi = own_row / RPQ;

    auto load_desc = [&](const int c, uint32_t * b, uint32_t & m01, uint32_t & m23) {
        const uint4 bb = *reinterpret_cast<const uint4 *>(gblk + 4*c);
        b[0] = bb.x; b[1] = bb.y; b[2] = bb.z; b[3] = bb.w;
        static_assert(sizeof(MT) == 2, "u16 masks");
        const uint2 mm = *reinterpret_cast<const uint2 *>(gmsk + 4*c);
        m01 = mm.x; m23 = mm.y;
    };
    auto blk_of = [&](const uint32_t * b, const int j) -> int {
        return b[j] == QSA3_BLK_SENT ? 0 : (int) b[j];
    };
    auto load_k = [&](const uint32_t * b, const uint32_t m01, const uint32_t m23, v16s * kf) {
        const int j = r >> 2;
        const uint32_t mk = ((j < 2 ? m01 : m23) >> (16 * (j & 1))) & 0xffffu;
        if (((mk >> (r & 3)) & 0x1111u) == 0) {
            kf[0] = v16s{};
            kf[1] = v16s{};
            return;
        }
        const int kb = blk_of(b, j);
        if constexpr (Q8) {
            const int key = 4*kb + (r & 3);
            const uint32_t * a = reinterpret_cast<const uint32_t *>(kbase + (size_t)key * s.knb1 + (size_t)(34*w & ~3));
            uint32_t dw[9], qd[8];
#pragma unroll
            for (int i = 0; i < 9; ++i) { dw[i] = a[i]; }
            uint16_t dh;
            if (w & 1) {
                dh = (uint16_t) (dw[0] >> 16);
#pragma unroll
                for (int i = 0; i < 8; ++i) { qd[i] = dw[i+1]; }
            } else {
                dh = (uint16_t) dw[0];
#pragma unroll
                for (int i = 0; i < 8; ++i) { qd[i] = __builtin_amdgcn_alignbyte(dw[i+1], dw[i], 2); }
            }
            const _Float16 d = __builtin_bit_cast(_Float16, dh);
            const qsa3_half2 d2 = qsa3_half2{d, d};
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                uint32_t h[8];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t x = qd[4*t + j];
                    const uint32_t e = qsa3_q8x2_to_h2(x, d2), o = qsa3_q8x2_to_h2(x >> 8, d2);
                    h[2*j]   = __builtin_amdgcn_perm(o, e, 0x05040100u);
                    h[2*j+1] = __builtin_amdgcn_perm(o, e, 0x07060302u);
                }
                kf[t] = __builtin_bit_cast(v16s, h);
            }
        } else {
            const int key = 4*kb + (r & 3);
            const char * p = kbase + (size_t)key * s.knb1 + (size_t)((2*w) * 16) * 2;
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const char * q = p + (size_t)t * 32;
                kf[t] = __builtin_bit_cast(v16s, (uint4[2]){*reinterpret_cast<const uint4 *>(q), *reinterpret_cast<const uint4 *>(q + 16)});
            }
        }
    };
    auto mask_values = [&](uint2 v, uint32_t mk) {
        uint32_t lo = 0, hi = 0;
        if (mk & 0x1111u) lo |= 0x0000ffffu;
        if (mk & 0x2222u) lo |= 0xffff0000u;
        if (mk & 0x4444u) hi |= 0x0000ffffu;
        if (mk & 0x8888u) hi |= 0xffff0000u;
        return make_uint2(v.x & lo, v.y & hi);
    };
    // query-presence of the chunk for this half (4 bits), from the 4 entries' 16-bit masks
    auto presence = [&](const uint32_t m01, const uint32_t m23) -> uint32_t {
        const uint32_t mo = (m01 | (m01 >> 16) | m23 | (m23 >> 16)) & 0xffffu;
        return (mo & 0x000fu ? 1u : 0u) | (mo & 0x00f0u ? 2u : 0u) | (mo & 0x0f00u ? 4u : 0u) | (mo & 0xf000u ? 8u : 0u);
    };

    uint32_t b[4] = { 0, 0, 0, 0 };
    uint32_t m01 = 0, m23 = 0;
    v16s kf[2];
    if (nchunks > 0) { load_desc(0, b, m01, m23); load_k(b, m01, m23, kf); }

    // SKIP 2: tile-set specialised chunk loops. The tile set of a chunk is block-uniform (scalar); a run of chunks with the
    // same set (a bucket, in BUCKET order) executes straight-line code with only that set's WMMA/LDS/softmax work
    auto tileset = [&](const uint32_t a, const uint32_t b2) __attribute__((always_inline)) -> uint32_t {
        const uint32_t qp = __builtin_amdgcn_readfirstlane(presence(a, b2));
        uint32_t ts = 0;
#pragma unroll
        for (int i = 0; i < NT; ++i) { if (qp & tm[i]) { ts |= 1u << i; } }
        return ts;
    };
    auto step = [&](auto TSc, const int c) __attribute__((always_inline)) {
        // readfirstlane: the chunk descriptor is block-uniform; scalar branches keep the WMMA path out of exec-mask control flow
        constexpr uint32_t TS = decltype(TSc)::value;
        // TS == 0x80: tile set decided at run time from the chunk masks (scalar branches); otherwise compile-time set
        constexpr bool RT = SKIP == 1 || ((SKIP == 3 || SKIP == 4) && TS == 0x80u);
        const uint32_t qp = RT ? __builtin_amdgcn_readfirstlane(presence(m01, m23)) : 0xfu;
        bool act[NT];
#pragma unroll
        for (int i = 0; i < NT; ++i) { act[i] = SKIP == 0 ? true : RT ? (qp & tm[i]) != 0 : ((TS >> i) & 1u) != 0; }
        v8f sc[NT];
#pragma unroll
        for (int i = 0; i < NT; ++i) {
#pragma unroll
            for (int e = 0; e < 8; ++e) { sc[i][e] = 0.f; }
            if (act[i]) {
#pragma unroll
                for (int t = 0; t < 2; ++t) { sc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kf[t], qf[i][t], sc[i]); }
            }
        }
        v16s vf[2];
        {   // V gather always unconditional, as the PR kernel: a branch around it serialises the loads (ISA: vmcnt(0) after each)
            const int kb0 = blk_of(b, 0), kb1 = blk_of(b, 1), kb2 = blk_of(b, 2), kb3 = blk_of(b, 3);
            const uint32_t mb0 = m01 & 0xffffu, mb1 = m01 >> 16, mb2 = m23 & 0xffffu, mb3 = m23 >> 16;
            auto ldv = [&](const int kb, const uint32_t mb, const int d) -> uint2 {
                return mask_values(*reinterpret_cast<const uint2 *>(pvg + (size_t) kb * 1024 + d * 4), mb);
            };
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int d = 32*w + 16*t + r;
                vf[t] = __builtin_bit_cast(v16s, (uint2[4]){ldv(kb0, mb0, d), ldv(kb1, mb1, d), ldv(kb2, mb2, d), ldv(kb3, mb3, d)});
            }
        }
        uint32_t nb[4] = { 0, 0, 0, 0 };
        uint32_t nm01 = 0, nm23 = 0;
        v16s kfn[2];
        if (c + 1 < nchunks) { load_desc(c + 1, nb, nm01, nm23); load_k(nb, nm01, nm23, kfn); }
        else { kfn[0] = kf[0]; kfn[1] = kf[1]; }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            if (i != w && act[i]) {
                part[i][w][lane*2+0] = make_float4(sc[i][0], sc[i][1], sc[i][2], sc[i][3]);
                part[i][w][lane*2+1] = make_float4(sc[i][4], sc[i][5], sc[i][6], sc[i][7]);
            }
        }
        __syncthreads();
        bool myact = false;
#pragma unroll
        for (int i = 0; i < NT; ++i) { if (i == w) { myact = act[i]; } }
        if (w < NT && (SKIP == 0 || myact)) {
            float scf[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) { scf[e] = 0.f; }
#pragma unroll
            for (int w2 = 0; w2 < 8; ++w2) {
                if (w2 == w) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) { scf[e] += sc[w][e]; }
                } else {
                    const float4 a = part[w][w2][lane*2+0], bq = part[w][w2][lane*2+1];
                    scf[0] += a.x; scf[1] += a.y; scf[2] += a.z; scf[3] += a.w; scf[4] += bq.x; scf[5] += bq.y; scf[6] += bq.z; scf[7] += bq.w;
                }
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int j = e >> 1;
                const uint32_t mk = ((j < 2 ? m01 : m23) >> (16 * (j & 1))) & 0xFFFFu;
                const int cc = 2*(e & 1) + hi;
                const bool kv = (mk >> (own_qi * 4 + cc)) & 1u;
                if (!kv) { scf[e] = -INFINITY; }
            }
            float mloc = scf[0];
#pragma unroll
            for (int e = 1; e < 8; ++e) { mloc = fmaxf(mloc, scf[e]); }
            mloc = fmaxf(mloc, __shfl_xor(mloc, 16));
            const float mnew  = fmaxf(m, mloc);
            const float alpha = mnew == -INFINITY ? 1.f : exp2f((m - mnew) * QSA3_L2E);
            float p[8], ls = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) { p[e] = mnew == -INFINITY ? 0.f : exp2f((scf[e] - mnew) * QSA3_L2E); ls += p[e]; }
            ls += __shfl_xor(ls, 16);
            l = l * alpha + ls;
            m = mnew;
            uint32_t pp[4], po[4];
#pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) {
                pp[e2] = (uint32_t) qsa3_f2h(p[2*e2]) | ((uint32_t) qsa3_f2h(p[2*e2+1]) << 16);
                po[e2] = (uint32_t) __shfl_xor((int) pp[e2], 16);
            }
            uint16_t ph[16];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint16_t mine  = (e & 1) ? (uint16_t) (pp[e>>1] >> 16) : (uint16_t) (pp[e>>1] & 0xffffu);
                const uint16_t other = (e & 1) ? (uint16_t) (po[e>>1] >> 16) : (uint16_t) (po[e>>1] & 0xffffu);
                ph[2*e]   = hi ? other : mine;
                ph[2*e+1] = hi ? mine  : other;
            }
            if (hi == 0) {
                ptile[w][r*2+0] = make_uint4((uint32_t) ph[0] | ((uint32_t) ph[1] << 16), (uint32_t) ph[2] | ((uint32_t) ph[3] << 16),
                                             (uint32_t) ph[4] | ((uint32_t) ph[5] << 16), (uint32_t) ph[6] | ((uint32_t) ph[7] << 16));
                ptile[w][r*2+1] = make_uint4((uint32_t) ph[8] | ((uint32_t) ph[9] << 16), (uint32_t) ph[10] | ((uint32_t) ph[11] << 16),
                                             (uint32_t) ph[12] | ((uint32_t) ph[13] << 16), (uint32_t) ph[14] | ((uint32_t) ph[15] << 16));
                alpha_s[own_row] = alpha;
            }
        }
        __syncthreads();
        float al[NT];
        bool anyr = false;
#pragma unroll
        for (int i = 0; i < NT; ++i) { al[i] = act[i] ? alpha_s[16*i + r] : 1.f; anyr = anyr || al[i] != 1.f; }
        if (__ballot(anyr)) {
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int t = 0; t < 2; ++t)
#pragma unroll
                    for (int e = 0; e < 8; ++e) { O[i][t][e] *= al[i]; }
        }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            if (act[i]) {
                const v16s pf = __builtin_bit_cast(v16s, (uint4[2]){ptile[i][r*2+0], ptile[i][r*2+1]});
#pragma unroll
                for (int t = 0; t < 2; ++t) { O[i][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vf[t], pf, O[i][t]); }
            }
        }
        kf[0] = kfn[0]; kf[1] = kfn[1];
        b[0] = nb[0]; b[1] = nb[1]; b[2] = nb[2]; b[3] = nb[3]; m01 = nm01; m23 = nm23;
    };
    if constexpr (SKIP == 2) {
        int c = 0;
        while (c < nchunks) {
            const uint32_t ts = tileset(m01, m23);
            auto run = [&](auto TSc) __attribute__((always_inline)) { do { step(TSc, c); ++c; } while (c < nchunks && tileset(m01, m23) == decltype(TSc)::value); };
#define C5_CASE(k) case k: run(std::integral_constant<uint32_t, k>{}); break;
            if constexpr (NT == 3) {
                switch (ts) { C5_CASE(0) C5_CASE(1) C5_CASE(2) C5_CASE(3) C5_CASE(4) C5_CASE(5) C5_CASE(6) default: run(std::integral_constant<uint32_t, 7>{}); }
            } else {
                switch (ts) { C5_CASE(0) C5_CASE(1) C5_CASE(2) C5_CASE(3) C5_CASE(4) C5_CASE(5) C5_CASE(6) C5_CASE(7)
                              C5_CASE(8) C5_CASE(9) C5_CASE(10) C5_CASE(11) C5_CASE(12) C5_CASE(13) C5_CASE(14) default: run(std::integral_constant<uint32_t, 15>{}); }
            }
#undef C5_CASE
        }
    } else if constexpr (SKIP == 3) {
        // two loop bodies only: chunks with every tile active run the straight-line full body (a run of them is the
        // all-queries bucket in BUCKET order); any other chunk runs the scalar-branch body
        constexpr uint32_t FULL = (1u << NT) - 1;
        int c = 0;
        while (c < nchunks) {
            if (tileset(m01, m23) == FULL) {
                do { step(std::integral_constant<uint32_t, FULL>{}, c); ++c; } while (c < nchunks && tileset(m01, m23) == FULL);
            } else {
                step(std::integral_constant<uint32_t, 0x80u>{}, c); ++c;
            }
        }
    } else if constexpr (SKIP == 4) {
        // BUCKET plans only (patterns ascending, so the all-4-queries bucket is last): two sequential loops, the partial
        // buckets with scalar-branch tile skipping, then the all-queries bucket in the straight-line full body
        int c = 0;
        for (; c < nchunks && __builtin_amdgcn_readfirstlane(presence(m01, m23)) != 0xfu; ++c) { step(std::integral_constant<uint32_t, 0x80u>{}, c); }
        for (; c < nchunks; ++c) { step(std::integral_constant<uint32_t, (1u << NT) - 1>{}, c); }
    } else if constexpr (SKIP >= 16) {
        // ORACLE (timing bound only, wrong output): every chunk runs the fixed tile set SKIP-16 in straight-line code,
        // i.e. the cost of a chunk with that many tiles and zero skip overhead
        for (int c = 0; c < nchunks; ++c) { step(std::integral_constant<uint32_t, (uint32_t) (SKIP - 16)>{}, c); }
    } else {
        for (int c = 0; c < nchunks; ++c) { step(std::integral_constant<uint32_t, 15>{}, c); }
    }

    if (w < NT && hi == 0) { l_s[own_row] = l; }
    __syncthreads();
    float * ostage = reinterpret_cast<float *>(&part[0][0][0]) + w * (16 * 32);
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        const float li = l_s[16*i + r];
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) { ostage[r * 32 + 16*t + 2*e + hi] = li > 0.f ? O[i][t][e] / li : 0.f; }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int rr = 4*c + (lane >> 3), dq = 4 * (lane & 7);
            const int row = 16*i + rr, qi = row / RPQ, h = row - RPQ*qi;
            const int query = qbase + qi;
            if (query < s.n_q && (RPQ == 12 || h < 12)) {
                const int hq = kvh * s.gqa + h;
                const float4 v = *reinterpret_cast<const float4 *>(ostage + rr * 32 + dq);
                float * dst = reinterpret_cast<float *>(reinterpret_cast<char *>(out) + (size_t) query * s.o2 + (size_t) hq * s.o1) + 32*w + dq;
                *reinterpret_cast<float4 *>(dst) = v;
            }
        }
        __syncthreads();
    }
}
