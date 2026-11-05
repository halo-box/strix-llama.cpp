typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f  __attribute__((ext_vector_type(8)));

#define QSA3_L2E 1.4426950408889634f
#define QSA3_G 4

static __device__ __forceinline__ uint16_t qsa3_f2h(const float f) { return __builtin_bit_cast(uint16_t, (_Float16) f); }
static __device__ __forceinline__ float qsa3_h2f(const uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }

typedef _Float16 qsa3_half2 __attribute__((ext_vector_type(2)));
// two q8_0 values in byte 0 and byte 2 of x to packed halves times d (as qd_q8x2_to_h2 of the decode kernel):
// 0x6400 | (q + 128) is 1024 + q + 128 in half, exact, so the product matches dequantize_block_q8_0_f16
static __device__ __forceinline__ uint32_t qsa3_q8x2_to_h2(const uint32_t x, const qsa3_half2 d2) {
    const qsa3_half2 v = __builtin_bit_cast(qsa3_half2, ((x ^ 0x00800080u) & 0x00FF00FFu) | 0x64006400u);
    return __builtin_bit_cast(uint32_t, (v - qsa3_half2{(_Float16) 1152, (_Float16) 1152}) * d2);
}

struct qsa3_layout {
    size_t q1, q2, o1, o2, m1;
    size_t knb1, vnb1, knb2, vnb2;   // raw f16 / q8_0 cache strides (DIRECT reads only)
    int nk, n_q, gqa, cap;
    float scale;
    int hq, gs;                      // total q heads; gate row stride in floats (row = query * hq + head)
};

// WPE > 0 asks the compiler for that many waves per SIMD (fewer VGPRs per wave): more CTAs in flight to cover the
// two barriers per 16 keys. 8 (192 VGPRs, no spill) is 4.5% faster than 7 on gfx1151 pp4096 and is what the launches
// use. Same arithmetic either way.
// Q8: the direct reads take q8_0 K and V rows and dequantize them in registers to the halves the pack would produce
// (a direct K chunk, dims 32w .. 32w+31 of a key, is exactly q8_0 block w of the row)
template<bool DIRECT, int WPE = 0, bool DIRECT_V = DIRECT, bool Q8 = false> __global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(WPE > 0 ? WPE : 1)))
void qsa3_attn_kernel(
        const float * __restrict__ q, const uint16_t * __restrict__ pk, const uint16_t * __restrict__ pv,
        const uint16_t * __restrict__ mask, const uint32_t * __restrict__ ublk, const uint16_t * __restrict__ umask,
        const int * __restrict__ ucount, float * __restrict__ out, const qsa3_layout s,
        const float * __restrict__ gate, uint16_t * __restrict__ o16, const bool o16_f16) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3)
    NO_DEVICE_CODE;
#else
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, r = lane & 15, hi = lane >> 4;
    const int g = blockIdx.x, kvh = blockIdx.y;
    __shared__ float4 part[3][8][64];
    __shared__ uint4  ptile[3][32];
    __shared__ float  alpha_s[48];
    __shared__ float  l_s[48];

    const size_t nblk = (size_t) s.nk / 4;
    const uint16_t * pkg = nullptr;
    const uint16_t * pvg = nullptr;
    const char * kbase = nullptr;
    const char * vbase = nullptr;
    if constexpr (DIRECT) {
        kbase = reinterpret_cast<const char *>(pk) + (size_t) kvh * s.knb2;
    } else {
        pkg = pk + (size_t) kvh * nblk * 1024;
    }
    if constexpr (DIRECT_V) {
        vbase = reinterpret_cast<const char *>(pv) + (size_t) kvh * s.vnb2;
    } else {
        pvg = pv + (size_t) kvh * nblk * 1024;
    }
    const uint32_t * gblk = ublk + (size_t) g * s.cap;
    const uint16_t * gmsk = umask + (size_t) g * s.cap;
    const int nchunks = ucount[g] >> 2;

    v16s qf[3][2];
    {
        uint16_t * stage = reinterpret_cast<uint16_t *>(&part[0][0][0]) + w * (48 * 32);
        const _Float16 hs = (_Float16) s.scale;
#pragma unroll
        for (int c = 0; c < 12; ++c) {
            const int row = 4*c + (lane >> 3), dq = 4 * (lane & 7);
            const int qi = row / 12, h = row - 12*qi;
            const int query = QSA3_G * g + qi;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (query < s.n_q) {
                v = *reinterpret_cast<const float4 *>(reinterpret_cast<const char *>(q) + (size_t) query * s.q1 + (size_t) (kvh * s.gqa + h) * s.q2 + (size_t) (32*w + dq) * 4);
            }
            const uint16_t h0 = __builtin_bit_cast(uint16_t, (_Float16) v.x * hs), h1 = __builtin_bit_cast(uint16_t, (_Float16) v.y * hs);
            const uint16_t h2 = __builtin_bit_cast(uint16_t, (_Float16) v.z * hs), h3 = __builtin_bit_cast(uint16_t, (_Float16) v.w * hs);
            *reinterpret_cast<uint2 *>(stage + row * 32 + dq) = make_uint2((uint32_t) h0 | ((uint32_t) h1 << 16), (uint32_t) h2 | ((uint32_t) h3 << 16));
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 3; ++i)
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const uint16_t * src = stage + (16*i + r) * 32 + 16*t;
                qf[i][t] = __builtin_bit_cast(v16s, (uint4[2]){*reinterpret_cast<const uint4 *>(src), *reinterpret_cast<const uint4 *>(src + 8)});
            }
        __syncthreads();
    }
    v8f O[3][2];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) { O[i][t][e] = 0.f; }
    float m = -INFINITY, l = 0.f;
    const int own_row = 16*w + r, own_qi = own_row / 12;
    const int own_query = QSA3_G * g + own_qi;
    const uint16_t * maskq = (mask && w < 3 && own_query < s.n_q)
        ? reinterpret_cast<const uint16_t *>(reinterpret_cast<const char *>(mask) + (size_t) own_query * s.m1) : nullptr;

    auto load_desc = [&](const int c, uint32_t * b, uint32_t & m01, uint32_t & m23) {
        const uint4 bb = *reinterpret_cast<const uint4 *>(gblk + 4*c);
        const uint2 mm = *reinterpret_cast<const uint2 *>(gmsk + 4*c);
        b[0] = bb.x; b[1] = bb.y; b[2] = bb.z; b[3] = bb.w; m01 = mm.x; m23 = mm.y;
    };
    // a padded slot reads block 0 but carries no mask bits, so its contribution is discarded downstream
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
        if constexpr (DIRECT && Q8) {
            // block w sits 34w bytes into a 16-byte aligned row: its quants are dword aligned for odd w (wave-uniform),
            // 2 bytes past a dword for even w. Aligned dword loads either way, realigned with alignbyte
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
                    // bytes 0 and 2, then 1 and 3, back into order: (q0, q1), (q2, q3)
                    const uint32_t x = qd[4*t + j];
                    const uint32_t e = qsa3_q8x2_to_h2(x, d2), o = qsa3_q8x2_to_h2(x >> 8, d2);
                    h[2*j]   = __builtin_amdgcn_perm(o, e, 0x05040100u);
                    h[2*j+1] = __builtin_amdgcn_perm(o, e, 0x07060302u);
                }
                kf[t] = __builtin_bit_cast(v16s, h);
            }
        } else if constexpr (DIRECT) {
            // the packed 16-half chunk at (key, w, t) is dims (2w+t)*16 .. +15 of that key's row
            const int key = 4*kb + (r & 3);
            const char * p = kbase + (size_t)key * s.knb1 + (size_t)((2*w) * 16) * 2;
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const char * q = p + (size_t)t * 32;
                kf[t] = __builtin_bit_cast(v16s, (uint4[2]){*reinterpret_cast<const uint4 *>(q), *reinterpret_cast<const uint4 *>(q + 16)});
            }
        } else {
            const uint16_t * krow = pkg + (size_t) kb * 1024 + (r & 3) * 16 + w * 128;
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const uint4 lo  = *reinterpret_cast<const uint4 *>(krow + t*64);
                const uint4 hi4 = *reinterpret_cast<const uint4 *>(krow + t*64 + 8);
                kf[t] = __builtin_bit_cast(v16s, (uint4[2]){lo, hi4});
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

    uint32_t b[4] = { 0, 0, 0, 0 };
    uint32_t m01 = 0, m23 = 0;
    v16s kf[2];
    if (nchunks > 0) { load_desc(0, b, m01, m23); load_k(b, m01, m23, kf); }

    for (int c = 0; c < nchunks; ++c) {
        v8f sc[3];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int e = 0; e < 8; ++e) { sc[i][e] = 0.f; }
#pragma unroll
            for (int t = 0; t < 2; ++t) { sc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kf[t], qf[i][t], sc[i]); }
        }
        v16s vf[2];
        {
            const int kb0 = blk_of(b, 0), kb1 = blk_of(b, 1), kb2 = blk_of(b, 2), kb3 = blk_of(b, 3);
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int d = 32*w + 16*t + r;
                if constexpr (DIRECT_V && Q8) {
                    // dim d is qs[16t + r] of q8_0 block w; each key row has its own scale
                    auto kv4 = [&](const int kb) -> uint2 {
                        const char * row = vbase + (size_t)(4*kb) * s.vnb1 + (size_t) w * sizeof(block_q8_0);
                        uint32_t qv[4];
                        _Float16 dv[4];
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const auto * blk = reinterpret_cast<const block_q8_0 *>(row + (size_t) j * s.vnb1);
                            dv[j] = __builtin_bit_cast(_Float16, blk->d);
                            qv[j] = (uint8_t) blk->qs[16*t + r];
                        }
                        return make_uint2(qsa3_q8x2_to_h2(qv[0] | (qv[1] << 16), qsa3_half2{dv[0], dv[1]}),
                                          qsa3_q8x2_to_h2(qv[2] | (qv[3] << 16), qsa3_half2{dv[2], dv[3]}));
                    };
                    vf[t] = __builtin_bit_cast(v16s, (uint2[4]){
                        mask_values(kv4(kb0), m01 & 0xffffu), mask_values(kv4(kb1), m01 >> 16),
                        mask_values(kv4(kb2), m23 & 0xffffu), mask_values(kv4(kb3), m23 >> 16)});
                } else if constexpr (DIRECT_V) {
                    // the packed uint2 holds the four keys of block kb at dim d; gather them from the rows
                    const char * col = vbase + (size_t)d * 2;
                    auto kv4 = [&](const int kb) -> uint2 {
                        const char * b = col + (size_t)(4*kb) * s.vnb1;
                        const uint32_t a0 = *reinterpret_cast<const uint16_t *>(b);
                        const uint32_t a1 = *reinterpret_cast<const uint16_t *>(b + s.vnb1);
                        const uint32_t a2 = *reinterpret_cast<const uint16_t *>(b + 2*s.vnb1);
                        const uint32_t a3 = *reinterpret_cast<const uint16_t *>(b + 3*s.vnb1);
                        return make_uint2(a0 | (a1 << 16), a2 | (a3 << 16));
                    };
                    vf[t] = __builtin_bit_cast(v16s, (uint2[4]){
                        mask_values(kv4(kb0), m01 & 0xffffu), mask_values(kv4(kb1), m01 >> 16),
                        mask_values(kv4(kb2), m23 & 0xffffu), mask_values(kv4(kb3), m23 >> 16)});
                } else {
                    const uint2 v0 = mask_values(*reinterpret_cast<const uint2 *>(pvg + (size_t) kb0 * 1024 + d * 4), m01 & 0xffffu);
                    const uint2 v1 = mask_values(*reinterpret_cast<const uint2 *>(pvg + (size_t) kb1 * 1024 + d * 4), m01 >> 16);
                    const uint2 v2 = mask_values(*reinterpret_cast<const uint2 *>(pvg + (size_t) kb2 * 1024 + d * 4), m23 & 0xffffu);
                    const uint2 v3 = mask_values(*reinterpret_cast<const uint2 *>(pvg + (size_t) kb3 * 1024 + d * 4), m23 >> 16);
                    vf[t] = __builtin_bit_cast(v16s, (uint2[4]){v0, v1, v2, v3});
                }
            }
        }
        uint32_t nb[4] = { 0, 0, 0, 0 };
        uint32_t nm01 = 0, nm23 = 0;
        v16s kfn[2];
        if (c + 1 < nchunks) { load_desc(c + 1, nb, nm01, nm23); load_k(nb, nm01, nm23, kfn); }
        else { kfn[0] = kf[0]; kfn[1] = kf[1]; }
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            if (i != w) {
                part[i][w][lane*2+0] = make_float4(sc[i][0], sc[i][1], sc[i][2], sc[i][3]);
                part[i][w][lane*2+1] = make_float4(sc[i][4], sc[i][5], sc[i][6], sc[i][7]);
            }
        }
        __syncthreads();
        if (w < 3) {
            float scf[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) { scf[e] = 0.f; }
#pragma unroll
            for (int w2 = 0; w2 < 8; ++w2) {
                if (w2 == w) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) { scf[e] += sc[w][e]; }
                } else {
                    const float4 a = part[w][w2][lane*2+0], b = part[w][w2][lane*2+1];
                    scf[0] += a.x; scf[1] += a.y; scf[2] += a.z; scf[3] += a.w; scf[4] += b.x; scf[5] += b.y; scf[6] += b.z; scf[7] += b.w;
                }
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int j = e >> 1, cc = 2*(e & 1) + hi;
                const uint32_t mk = ((j < 2 ? m01 : m23) >> (16 * (j & 1))) & 0xFFFFu;
                const bool kv = (mk >> (own_qi * 4 + cc)) & 1u;
                if (maskq && kv) { scf[e] += qsa3_h2f(maskq[4 * blk_of(b, j) + cc]); }
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
        float al[3];
#pragma unroll
        for (int i = 0; i < 3; ++i) { al[i] = alpha_s[16*i + r]; }
        if (__ballot(al[0] != 1.f || al[1] != 1.f || al[2] != 1.f)) {
#pragma unroll
            for (int i = 0; i < 3; ++i)
#pragma unroll
                for (int t = 0; t < 2; ++t)
#pragma unroll
                    for (int e = 0; e < 8; ++e) { O[i][t][e] *= al[i]; }
        }
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const v16s pf = __builtin_bit_cast(v16s, (uint4[2]){ptile[i][r*2+0], ptile[i][r*2+1]});
#pragma unroll
            for (int t = 0; t < 2; ++t) { O[i][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vf[t], pf, O[i][t]); }
        }
        kf[0] = kfn[0]; kf[1] = kfn[1];
        b[0] = nb[0]; b[1] = nb[1]; b[2] = nb[2]; b[3] = nb[3]; m01 = nm01; m23 = nm23;
    }

    if (w < 3 && hi == 0) { l_s[own_row] = l; }
    __syncthreads();
    float * ostage = reinterpret_cast<float *>(&part[0][0][0]) + w * (16 * 32);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const float li = l_s[16*i + r];
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) { ostage[r * 32 + 16*t + 2*e + hi] = li > 0.f ? O[i][t][e] / li : 0.f; }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int rr = 4*c + (lane >> 3), dq = 4 * (lane & 7);
            const int row = 16*i + rr, qi = row / 12, h = row - 12*qi;
            const int query = QSA3_G * g + qi;
            if (query < s.n_q) {
                // gated output: sigmoid(gate) * O with op_sigmoid's expression, as cont_sigmoid_mul_v4
                const int hq = kvh * s.gqa + h;
                float4 v = *reinterpret_cast<const float4 *>(ostage + rr * 32 + dq);
                if (gate) {
                    const float4 gg = *reinterpret_cast<const float4 *>(gate + ((size_t) query * s.hq + hq) * s.gs + 32*w + dq);
                    v.x = 1.0f / (1.0f + expf(-gg.x)) * v.x; v.y = 1.0f / (1.0f + expf(-gg.y)) * v.y;
                    v.z = 1.0f / (1.0f + expf(-gg.z)) * v.z; v.w = 1.0f / (1.0f + expf(-gg.w)) * v.w;
                }
                if (out) {
                    float * dst = reinterpret_cast<float *>(reinterpret_cast<char *>(out) + (size_t) query * s.o2 + (size_t) hq * s.o1) + 32*w + dq;
                    *reinterpret_cast<float4 *>(dst) = v;
                }
                if (o16) {
                    uint2 hv;
                    hv.x = (uint32_t) ggml_cuda_f32_to_act16(v.x, o16_f16) | ((uint32_t) ggml_cuda_f32_to_act16(v.y, o16_f16) << 16);
                    hv.y = (uint32_t) ggml_cuda_f32_to_act16(v.z, o16_f16) | ((uint32_t) ggml_cuda_f32_to_act16(v.w, o16_f16) << 16);
                    *reinterpret_cast<uint2 *>(o16 + ((size_t) query * s.hq + hq) * 256 + 32*w + dq) = hv;
                }
            }
        }
        __syncthreads();
    }
#endif
}
static __device__ __forceinline__ uint16_t qsa_q8_to_h(const int8_t q, const _Float16 d) {
    return __builtin_bit_cast(uint16_t, (_Float16) ((_Float16) q * d));
}

template <bool q8>
static __global__ void qsa_pack_k(const char * src, uint16_t * dst, size_t nb1, size_t nb2, int nk, uint32_t count) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count) return;
    const uint32_t d=(i%32)*8, key=(i/32)%nk, head=i/(uint32_t(nk)*32);
    const size_t group=size_t(head)*(nk/4)+key/4;
    auto * out=reinterpret_cast<uint4 *>(dst+group*1024+(d/16)*64+(key%4)*16+d%16);
    if constexpr (q8) {
        const auto * blk=reinterpret_cast<const block_q8_0 *>(src+size_t(head)*nb2+size_t(key)*nb1)+d/QK8_0;
        const _Float16 dh=__builtin_bit_cast(_Float16, blk->d);
        uint16_t h[8];
#pragma unroll
        for (int j=0; j<8; ++j) { h[j]=qsa_q8_to_h(blk->qs[d%QK8_0+j], dh); }
        *out=make_uint4(h[0]|(uint32_t(h[1])<<16), h[2]|(uint32_t(h[3])<<16), h[4]|(uint32_t(h[5])<<16), h[6]|(uint32_t(h[7])<<16));
    } else {
        *out=*reinterpret_cast<const uint4 *>(src+size_t(head)*nb2+size_t(key)*nb1+d*2);
    }
}

template <bool q8>
static __global__ void qsa_pack_v(const char * src, uint16_t * dst, size_t nb1, size_t nb2, int nk, uint32_t count) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count) return;
    const uint32_t d=i%256, group=i/256, head=group/(nk/4), key=(group%(nk/4))*4;
    const char * in=src+size_t(head)*nb2+size_t(key)*nb1;
    uint16_t h[4];
#pragma unroll
    for (int j=0; j<4; ++j) {
        if constexpr (q8) {
            const auto * blk=reinterpret_cast<const block_q8_0 *>(in+j*nb1)+d/QK8_0;
            h[j]=qsa_q8_to_h(blk->qs[d%QK8_0], __builtin_bit_cast(_Float16, blk->d));
        } else {
            h[j]=*reinterpret_cast<const uint16_t *>(in+j*nb1+d*2);
        }
    }
    *reinterpret_cast<ushort4 *>(dst+size_t(group)*1024+d*4)=make_ushort4(h[0], h[1], h[2], h[3]);
}
