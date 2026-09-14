// k_pair is the K coordinate measured in FLOAT_TYPEV2 elements.
uint a_shmem_index(uint m, uint k_pair) {
    if (APPLY_SLM_A_RESHAPE) {
        const uint tile_width = TK / 2;
        return (k_pair / tile_width) * BM * tile_width
             + m * tile_width
             + k_pair % tile_width;
    }
    return m * SHMEM_STRIDE + k_pair;
}

uint a_shmem_stride() {
    return APPLY_SLM_A_RESHAPE ? TK / 2 : SHMEM_STRIDE;
}

void store_a(uint m, uint k_pair, FLOAT_TYPEV2 value) {
    buf_a[a_shmem_index(m, k_pair)] = TO_BUF(value);
}

#if defined(DATA_A_ROCMFPX_FP3)
int32_t rocmfpx_mm_fp3_pack4_window(uint ib, uint idx) {
    const uint bit_pos = idx * 3u;
    const uint byte_pos = bit_pos >> 3u;
    const uint sh = bit_pos & 7u;
    uint bits = uint(data_a[ib].qs[byte_pos]) |
                (uint(data_a[ib].qs[byte_pos + 1u]) << 8);
    if (sh > 4u) {
        bits |= uint(data_a[ib].qs[byte_pos + 2u]) << 16;
    }
    bits = (bits >> sh) & 0xFFFu;
    return pack32(i8vec4(kvalues_rocmfpx_fp3_const[ bits        & 7u],
                         kvalues_rocmfpx_fp3_const[(bits >> 3) & 7u],
                         kvalues_rocmfpx_fp3_const[(bits >> 6) & 7u],
                         kvalues_rocmfpx_fp3_const[(bits >> 9) & 7u]));
}

vec4 rocmfpx_mm_fp3_vec4(uint ib, uint idx) {
    const float d = ue4m3_to_fp32(data_a[ib].e[idx >= 16u ? 1u : 0u]);
    return vec4(unpack8(rocmfpx_mm_fp3_pack4_window(ib, idx))) * d;
}
#endif

#if defined(DATA_A_ROCMFPX_FP6)
uint rocmfpx_mm_fp6_get_bits(uint ib, uint idx) {
    const uint bit_pos  = idx * 6u;
    const uint byte_pos = bit_pos >> 3u;
    const uint sh       = bit_pos & 7u;
    uint bits = uint(data_a[ib].qs[byte_pos]);
    if (sh > 2u) {
        bits |= uint(data_a[ib].qs[byte_pos + 1u]) << 8;
    }
    return (bits >> sh) & 0x3Fu;
}

float rocmfpx_mm_fp6_value(uint ib, uint idx) {
    const float d = ue4m3_to_fp32(data_a[ib].e[idx >= 16u ? 1u : 0u]);
    return float(rocmfpx_fp6_decode_code(rocmfpx_mm_fp6_get_bits(ib, idx))) * d;
}

vec4 rocmfpx_mm_fp6_vec4(uint ib, uint idx) {
    return vec4(rocmfpx_mm_fp6_value(ib, idx + 0u),
                rocmfpx_mm_fp6_value(ib, idx + 1u),
                rocmfpx_mm_fp6_value(ib, idx + 2u),
                rocmfpx_mm_fp6_value(ib, idx + 3u));
}
#endif

// ---- 8-wide q6_K / q3_K / q8_0 / q5_0 loaders (LOAD_VEC_A == 8, KHR coopmat variants) -------------
// One call covers 8 consecutive k of one row. Blocks are 210 / 110 / 34 / 22 bytes (2-byte aligned).
// fetch8 reads the 8 bytes at a 2-aligned byte offset b as two dwords when b is 4-aligned, else as
// dword(b-2), dword(b+2) and a 16-bit load of b+6: it never touches a byte outside [b-2, b+8), which
// stays inside the block for every group these loaders use (the misaligned case only occurs for
// odd block indices, whose groups start >= 2 bytes into the block). The first version read whole
// dwords past the group and faulted on the last block of an mmap'd tensor (GPUVM fault, 2026-09-13).
// Scales and d are read through the typed block members. Index math is split into a uniform part
// (pos_a: superblock column and the position inside the block are the same for every lane) and a
// lane-invariant part computed once before the K loop (a_lane_off / a_store_idx, see mul_mm.comp).
// pos_a is in LOAD_VEC_A units and a multiple of 4 (rows are multiples of 32 k, BK = 32).
#if LOAD_VEC_A == 8 && (defined(DATA_A_Q6_K) || defined(DATA_A_Q3_K) || defined(DATA_A_Q8_0) || defined(DATA_A_Q5_0))
uvec3 fetch8(const uint b) {
    // branchless: dwords at (b & ~3) and +4 cover [b-2, b+8) when b is misaligned and [b, b+8) when
    // aligned; the 16-bit load at b+6 is only consumed in the misaligned case (it is inside either way)
    const uint b4 = b & ~3u;
    return uvec3(data_a_u32[b4 / 4], data_a_u32[b4 / 4 + 1], uint(data_a_u16[(b + 6) / 2]));
}
// the 8 bytes as two dwords (low dword first)
uvec2 unpack8b(const uvec3 r, const bool odd) {
    return odd ? uvec2((r.x >> 16) | (r.y << 16), (r.y >> 16) | (r.z << 16)) : r.xy;
}
#endif
#if LOAD_VEC_A == 8 && defined(DATA_A_Q6_K)
#define A_PREFETCH 1
#define A_RAW_T uvec4[2]   // [0] = ql fetch8, qh.x; [1] = qh.yz, d bits | scale << 16, 0
uint a_lane_off(const uint row, const uint col) { return (col * (p.stride_a / 256)) * 210 + 8 * row; }   // byte offset of the lane's row within its superblock column, plus its 8-byte group
void fetch_a(const uint pos_a, const uint row, const uint col, const uint lane_off, out uvec4 raw[2]) {
    const uint kb = pos_a / 32;            // superblock column
    const uint g  = (pos_a % 32) / 4;      // 32-k group in the block: n = g/4, m = g%4
    const uint n  = g / 4;
    const uint m  = g % 4;
    const uint base = kb * 210 + lane_off - 8 * row;     // block byte base for this lane (lane_off carries 8*row)
    const uint ib   = base / 210;
    const uint bql = base + 64 * n + 32 * (m & 1) + 8 * row;
    const uint bqh = base + 128 + 32 * n + 8 * row;
    const uint is  = 8 * n + 2 * m + row / 2;
    const uvec3 ql = fetch8(bql);
    const uvec3 qh = fetch8(bqh);
    raw[0] = uvec4(ql.x, ql.y, ql.z, qh.x);
    raw[1] = uvec4(qh.y, qh.z, uint(float16BitsToUint16(data_a[ib].d)) | (uint(uint8_t(data_a[ib].scales[is])) << 16), 0);
}
void store_a_raw(const uint pos_a, const uint row, const uint col, const uint sidx, const uvec4 raw[2]) {
    const uint g  = (pos_a % 32) / 4;
    const uint m  = g % 4;
    const uint base = (pos_a / 32) * 210 + (col * (p.stride_a / 256)) * 210;
    const bool odd = (base & 2) != 0;      // every odd block starts 2 bytes into a dword
    const uvec2 ql = unpack8b(raw[0].xyz, odd);
    const uvec2 qh = unpack8b(uvec3(raw[0].w, raw[1].x, raw[1].y), odd);
    const int   sc = int(int8_t(uint8_t(raw[1].z >> 16)));
    const float d  = float(uint16BitsToFloat16(uint16_t(raw[1].z & 0xFFFF)));
    const float dscale = d * float(sc);
    const uint nib = 4 * (m >> 1);
    const uint hsh = 2 * m;
    const uint q0 = ((ql.x >> nib) & 0x0F0F0F0F) | (((qh.x >> hsh) & 0x03030303) << 4);
    const uint q1 = ((ql.y >> nib) & 0x0F0F0F0F) | (((qh.y >> hsh) & 0x03030303) << 4);
    const vec4 v0 = (vec4(unpack8(q0)) - 32.0f) * dscale;
    const vec4 v1 = (vec4(unpack8(q1)) - 32.0f) * dscale;
    buf_a[sidx]     = TO_BUF(FLOAT_TYPEV2(v0.xy));
    buf_a[sidx + 1] = TO_BUF(FLOAT_TYPEV2(v0.zw));
    buf_a[sidx + 2] = TO_BUF(FLOAT_TYPEV2(v1.xy));
    buf_a[sidx + 3] = TO_BUF(FLOAT_TYPEV2(v1.zw));
}
#elif LOAD_VEC_A == 8 && defined(DATA_A_Q3_K)
#define A_PREFETCH 1
#define A_RAW_T uvec4[2]   // [0] = qs fetch8, hmask.x; [1] = hmask.yz, sc0 | sc1 << 8 | d bits << 16, 0
uint a_lane_off(const uint row, const uint col) { return (col * (p.stride_a / 256)) * 110 + 8 * row; }
void fetch_a(const uint pos_a, const uint row, const uint col, const uint lane_off, out uvec4 raw[2]) {
    const uint kb = pos_a / 32;
    const uint g  = (pos_a % 32) / 4;      // n = g/4, j = g%4
    const uint n  = g / 4;
    const uint base = kb * 110 + lane_off - 8 * row;
    const uint ib   = base / 110;
    const uint bqs = base + 32 + 32 * n + 8 * row;
    const uint bhm = base + 8 * row;
    const uint is  = 2 * g + row / 2;       // 16-element scale index 0..15
    const uvec3 qs = fetch8(bqs);
    const uvec3 hm = fetch8(bhm);
    raw[0] = uvec4(qs.x, qs.y, qs.z, hm.x);
    raw[1] = uvec4(hm.y, hm.z, uint(data_a[ib].scales[is % 8]) | (uint(data_a[ib].scales[8 + (is % 4)]) << 8) | (uint(float16BitsToUint16(data_a[ib].d)) << 16), 0);
}
void store_a_raw(const uint pos_a, const uint row, const uint col, const uint sidx, const uvec4 raw[2]) {
    const uint g  = (pos_a % 32) / 4;
    const uint n  = g / 4;
    const uint j  = g % 4;
    const uint is = 2 * g + row / 2;
    const uint base = (pos_a / 32) * 110 + (col * (p.stride_a / 256)) * 110;
    const bool odd = (base & 2) != 0;
    const uvec2 qs = unpack8b(raw[0].xyz, odd);
    const uvec2 hm = unpack8b(uvec3(raw[0].w, raw[1].x, raw[1].y), odd);
    const uint sc0 = raw[1].z & 0xFF;
    const uint sc1 = (raw[1].z >> 8) & 0xFF;
    const int  us  = int(((sc0 >> (4 * (is / 8))) & 0xF) | (((sc1 >> (2 * (is / 4))) & 3) << 4));
    const float dl = float(uint16BitsToFloat16(uint16_t(raw[1].z >> 16))) * float(us - 32);
    const uint qsh = 2 * j;
    const uint hsh = 4 * n + j;
    const uint q0 = (qs.x >> qsh) & 0x03030303;
    const uint q1 = (qs.y >> qsh) & 0x03030303;
    const uint h0 = ((((hm.x >> hsh) & 0x01010101) ^ 0x01010101) << 2);
    const uint h1 = ((((hm.y >> hsh) & 0x01010101) ^ 0x01010101) << 2);
    const vec4 v0 = (vec4(unpack8(q0)) - vec4(unpack8(h0))) * dl;
    const vec4 v1 = (vec4(unpack8(q1)) - vec4(unpack8(h1))) * dl;
    buf_a[sidx]     = TO_BUF(FLOAT_TYPEV2(v0.xy));
    buf_a[sidx + 1] = TO_BUF(FLOAT_TYPEV2(v0.zw));
    buf_a[sidx + 2] = TO_BUF(FLOAT_TYPEV2(v1.xy));
    buf_a[sidx + 3] = TO_BUF(FLOAT_TYPEV2(v1.zw));
}
#elif LOAD_VEC_A == 8 && defined(DATA_A_Q8_0)
#define A_PREFETCH 1
#define A_RAW_T uvec4   // xyz: qs fetch8 (block of 34 B), w: d bits
uint a_lane_off(const uint row, const uint col) { return (col * (p.stride_a / 32)) * 34 + 8 * row; }   // block row base + 8-byte group
void fetch_a(const uint pos_a, const uint row, const uint col, const uint lane_off, out uvec4 raw) {
    const uint base = (pos_a / 4) * 34 + lane_off - 8 * row;   // block byte base (pos_a/4 = block column)
    const uint ib   = base / 34;
    const uvec3 qs = fetch8(base + 2 + 8 * row);
    raw = uvec4(qs.x, qs.y, qs.z, uint(float16BitsToUint16(data_a_packed16[ib].d)));
}
void store_a_raw(const uint pos_a, const uint row, const uint col, const uint sidx, const uvec4 raw) {
    const uint base = (pos_a / 4 + col * (p.stride_a / 32)) * 34;
    const bool odd = ((base + 2) & 2) != 0;
    const uvec2 q = unpack8b(raw.xyz, odd);
    const float d = float(uint16BitsToFloat16(uint16_t(raw.w)));
    // int8 via unsigned bytes: (q ^ 0x80) - 128
    const vec4 v0 = fma(vec4(unpack8(q.x ^ 0x80808080)), vec4(d), vec4(-128.0f * d));
    const vec4 v1 = fma(vec4(unpack8(q.y ^ 0x80808080)), vec4(d), vec4(-128.0f * d));
    buf_a[sidx]     = TO_BUF(FLOAT_TYPEV2(v0.xy));
    buf_a[sidx + 1] = TO_BUF(FLOAT_TYPEV2(v0.zw));
    buf_a[sidx + 2] = TO_BUF(FLOAT_TYPEV2(v1.xy));
    buf_a[sidx + 3] = TO_BUF(FLOAT_TYPEV2(v1.zw));
}
#elif LOAD_VEC_A == 8 && defined(DATA_A_Q5_0)
#define A_PREFETCH 1
#define A_RAW_T uvec4[2]   // [0].xyz: qs fetch8 (block of 22 B), [0].w: d bits; [1].x: qh
uint a_lane_off(const uint row, const uint col) { return (col * (p.stride_a / 32)) * 22 + 8 * (row % 2); }
void fetch_a(const uint pos_a, const uint row, const uint col, const uint lane_off, out uvec4 raw[2]) {
    const uint base = (pos_a / 4) * 22 + lane_off - 8 * (row % 2);
    const uint ib   = base / 22;
    const uvec3 qs = fetch8(base + 6 + 8 * (row % 2));      // row >= 2 are the high nibbles of the same bytes
    raw[0] = uvec4(qs.x, qs.y, qs.z, uint(float16BitsToUint16(data_a_packed16[ib].d)));
    raw[1] = uvec4(uint(data_a_packed16[ib].qh[0]) | (uint(data_a_packed16[ib].qh[1]) << 16), 0, 0, 0);
}
void store_a_raw(const uint pos_a, const uint row, const uint col, const uint sidx, const uvec4 raw[2]) {
    const uint base = (pos_a / 4 + col * (p.stride_a / 32)) * 22;
    const bool odd = ((base + 6) & 2) != 0;
    const uvec2 qs = unpack8b(raw[0].xyz, odd);
    const float d = float(uint16BitsToFloat16(uint16_t(raw[0].w)));
    const uint qh = raw[1].x;
    const uint nib = 4 * (row / 2);
    const uint hb  = (qh >> (8 * row)) & 0xFF;                     // the 8 high bits of this group
    const uint h0  = ((hb & 0xF) * 0x00204081u) & 0x01010101u;    // spread bits 0..3 into bytes
    const uint h1  = (((hb >> 4) & 0xF) * 0x00204081u) & 0x01010101u;
    const uint q0 = ((qs.x >> nib) & 0x0F0F0F0F) | (h0 << 4);
    const uint q1 = ((qs.y >> nib) & 0x0F0F0F0F) | (h1 << 4);
    const vec4 v0 = fma(vec4(unpack8(q0)), vec4(d), vec4(-16.0f * d));
    const vec4 v1 = fma(vec4(unpack8(q1)), vec4(d), vec4(-16.0f * d));
    buf_a[sidx]     = TO_BUF(FLOAT_TYPEV2(v0.xy));
    buf_a[sidx + 1] = TO_BUF(FLOAT_TYPEV2(v0.zw));
    buf_a[sidx + 2] = TO_BUF(FLOAT_TYPEV2(v1.xy));
    buf_a[sidx + 3] = TO_BUF(FLOAT_TYPEV2(v1.zw));
}
#endif

void load_a_to_shmem(const uint pos_a, const uint row, const uint col, const uint idx_m, const uint block, const uint end_k) {
#if defined(DATA_A_F32) || defined(DATA_A_F16)
#if LOAD_VEC_A == 8
            if (ALIGNED != 0) {
                const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
                const uint k_pair = row * LOAD_VEC_A / 2;
                FLOAT_TYPEV8 aa = FLOAT_TYPEV8(data_a[idx]);
                store_a(col, k_pair,     aa[0].xy);
                store_a(col, k_pair + 1, aa[0].zw);
                store_a(col, k_pair + 2, aa[1].xy);
                store_a(col, k_pair + 3, aa[1].zw);
                return;
            }
#elif LOAD_VEC_A == 4
            if (ALIGNED != 0) {
                const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
                const uint k_pair = row * LOAD_VEC_A / 2;
                FLOAT_TYPEV4 aa = FLOAT_TYPEV4(data_a[idx]);
                store_a(col, k_pair,     aa.xy);
                store_a(col, k_pair + 1, aa.zw);
                return;
            }
#endif
            const uint idx = pos_a + col * p.stride_a + row * 2;
            if (idx_m < p.M && block + row * 2 + 1 < end_k) {
                store_a(col, row, FLOAT_TYPEV2(data_a_scalar[idx],
                                               data_a_scalar[idx + 1]));
            } else if (idx_m < p.M && block + row * 2 < end_k) {
                store_a(col, row, FLOAT_TYPEV2(data_a_scalar[idx], 0.0f));
            } else {
                store_a(col, row, FLOAT_TYPEV2(0.0f));
            }
#elif defined(DATA_A_BF16)
#if LOAD_VEC_A == 4
            if (ALIGNED != 0) {
                const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
                const uint k_pair = row * LOAD_VEC_A / 2;
                FLOAT_TYPEV4 aa = FLOAT_TYPEV4(TO_FLOAT_TYPE(data_a[idx]));
                store_a(col, k_pair,     aa.xy);
                store_a(col, k_pair + 1, aa.zw);
                return;
            }
#endif
            const uint idx = pos_a + col * p.stride_a + row * 2;
            if (idx_m < p.M && block + row * 2 + 1 < end_k) {
                store_a(col, row, FLOAT_TYPEV2(TO_FLOAT_TYPE(data_a_scalar[idx]),
                                               TO_FLOAT_TYPE(data_a_scalar[idx + 1])));
            } else if (idx_m < p.M && block + row * 2 < end_k) {
                store_a(col, row, FLOAT_TYPEV2(TO_FLOAT_TYPE(data_a_scalar[idx]), 0.0f));
            } else {
                store_a(col, row, FLOAT_TYPEV2(0.0f));
            }
#elif defined(DATA_A_Q4_0)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 4;
            const uint iqs = idx & 0x03;

            const float d = float(data_a_packed16[ib].d);
            const uint vui = uint(data_a_packed16[ib].qs[2*iqs]) | (uint(data_a_packed16[ib].qs[2*iqs + 1]) << 16);
            const vec4 v0 = (vec4(unpack8(vui & 0x0F0F0F0F)) - 8.0f) * d;
            const vec4 v1 = (vec4(unpack8((vui >> 4) & 0x0F0F0F0F)) - 8.0f) * d;

            const uint k_pair = row * LOAD_VEC_A / 4;
            store_a(col, k_pair,     FLOAT_TYPEV2(v0.xy));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(v0.zw));
            store_a(col, k_pair + 8, FLOAT_TYPEV2(v1.xy));
            store_a(col, k_pair + 9, FLOAT_TYPEV2(v1.zw));
#elif defined(DATA_A_Q4_1)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 4;
            const uint iqs = idx & 0x03;

            const vec2 dm = vec2(data_a_packed32[ib].dm);
            const uint vui = data_a_packed32[ib].qs[iqs];
            const vec4 v0 = vec4(unpack8(vui & 0x0F0F0F0F)) * dm.x + dm.y;
            const vec4 v1 = vec4(unpack8((vui >> 4) & 0x0F0F0F0F)) * dm.x + dm.y;

            const uint k_pair = row * LOAD_VEC_A / 4;
            store_a(col, k_pair,     FLOAT_TYPEV2(v0.xy));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(v0.zw));
            store_a(col, k_pair + 8, FLOAT_TYPEV2(v1.xy));
            store_a(col, k_pair + 9, FLOAT_TYPEV2(v1.zw));
#elif defined(DATA_A_Q5_0)
#if LOAD_VEC_A == 8
            A_RAW_T raw;
            fetch_a(pos_a, row, col, a_lane_off(row, col), raw);
            store_a_raw(pos_a, row, col, a_shmem_index(col, row * LOAD_VEC_A / 2), raw);
#else
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 8;
            const uint iqs = idx & 0x07;

            const float d = float(data_a_packed16[ib].d);
            const uint uint_qh = uint(data_a_packed16[ib].qh[1]) << 16 | uint(data_a_packed16[ib].qh[0]);
            const ivec2 qh0 = ivec2(((uint_qh >> 2*iqs) << 4) & 0x10, (uint_qh >> (2*iqs + 12)) & 0x10);
            const ivec2 qh1 = ivec2(((uint_qh >> (2*iqs + 1)) << 4) & 0x10, (uint_qh >> (2*iqs + 13)) & 0x10);

            const uint vui = uint(data_a_packed16[ib].qs[iqs]);
            const vec4 v = (vec4((vui & 0xF) | qh0.x, ((vui >> 4) & 0xF) | qh0.y, ((vui >> 8) & 0xF) | qh1.x, (vui >> 12) | qh1.y) - 16.0f) * d;
            store_a(col, row,     FLOAT_TYPEV2(v.xz));
            store_a(col, row + 8, FLOAT_TYPEV2(v.yw));
#endif
#elif defined(DATA_A_Q5_1)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 4;
            const uint iqs = idx & 0x03;

            const vec2 dm = vec2(data_a_packed32[ib].dm);
            const uint uint_qh = data_a_packed32[ib].qh;
            const uvec2 qh0 = uvec2(((uint_qh >> 4*iqs) << 4) & 0x10, (uint_qh >> (4*iqs + 12)) & 0x10);
            const uvec2 qh1 = uvec2(((uint_qh >> (4*iqs + 1)) << 4) & 0x10, (uint_qh >> (4*iqs + 13)) & 0x10);
            const uvec2 qh2 = uvec2(((uint_qh >> (4*iqs + 2)) << 4) & 0x10, (uint_qh >> (4*iqs + 14)) & 0x10);
            const uvec2 qh3 = uvec2(((uint_qh >> (4*iqs + 3)) << 4) & 0x10, (uint_qh >> (4*iqs + 15)) & 0x10);

            const uint vui = data_a_packed32[ib].qs[iqs];
            const vec4 v0 = vec4((vui & 0xF) | qh0.x, ((vui >> 4) & 0xF) | qh0.y, ((vui >> 8) & 0xF) | qh1.x, ((vui >> 12) & 0xF) | qh1.y) * dm.x + dm.y;
            const vec4 v1 = vec4(((vui >> 16) & 0xF) | qh2.x, ((vui >> 20) & 0xF) | qh2.y, ((vui >> 24) & 0xF) | qh3.x, ((vui >> 28) & 0xF) | qh3.y) * dm.x + dm.y;

            const uint k_pair = row * LOAD_VEC_A / 4;
            store_a(col, k_pair,     FLOAT_TYPEV2(v0.xz));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(v1.xz));
            store_a(col, k_pair + 8, FLOAT_TYPEV2(v0.yw));
            store_a(col, k_pair + 9, FLOAT_TYPEV2(v1.yw));
#elif defined(DATA_A_Q8_0)
#if LOAD_VEC_A == 8
            A_RAW_T raw;
            fetch_a(pos_a, row, col, a_lane_off(row, col), raw);
            store_a_raw(pos_a, row, col, a_shmem_index(col, row * LOAD_VEC_A / 2), raw);
#else
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 8;
            const uint iqs = idx & 0x07;

            const float d = float(data_a_packed16[ib].d);
            const i8vec2 v0 = unpack8(int32_t(data_a_packed16[ib].qs[2*iqs])).xy; // vec4 used due to #12147
            const i8vec2 v1 = unpack8(int32_t(data_a_packed16[ib].qs[2*iqs + 1])).xy;
            const vec4 v = vec4(v0.x, v0.y, v1.x, v1.y) * d;

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2(v.xy));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(v.zw));
#endif
#elif defined(DATA_A_Q1_0)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 16;
            const uint iqs = idx & 0xfu;

            const float d = float(data_a[ib].d);
            const uint bits = uint(data_a[ib].qs[iqs]);

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2((bits & 0x01u) != 0u ? d : -d, (bits & 0x02u) != 0u ? d : -d));
            store_a(col, k_pair + 1, FLOAT_TYPEV2((bits & 0x04u) != 0u ? d : -d, (bits & 0x08u) != 0u ? d : -d));
            store_a(col, k_pair + 2, FLOAT_TYPEV2((bits & 0x10u) != 0u ? d : -d, (bits & 0x20u) != 0u ? d : -d));
            store_a(col, k_pair + 3, FLOAT_TYPEV2((bits & 0x40u) != 0u ? d : -d, (bits & 0x80u) != 0u ? d : -d));
#elif defined(DATA_A_Q2_0)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 16;
            const uint iqs = idx & 0xfu;

            const FLOAT_TYPE d = FLOAT_TYPE(data_a[ib].d);
            const uint bits = uint(data_a[ib].qs[iqs]);

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     d * (FLOAT_TYPEV2(bits & 3u, (bits >> 2u) & 3u) - FLOAT_TYPEV2(1.0f)));
            store_a(col, k_pair + 1, d * (FLOAT_TYPEV2((bits >> 4u) & 3u, bits >> 6u) - FLOAT_TYPEV2(1.0f)));
#elif defined(DATA_A_Q2_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 64;                          // 4 values per idx
            const uint iqs = (idx % 64) * 2;                   // 0,2,4..126

            const uint qsi = (iqs / 64) * 16 + (iqs % 16);     // 0..15
            const uint scalesi = iqs / 8;                      // 0..15
            const uint qsshift = ((iqs % 64) / 16) * 2;        // 0,2,4,6

            const vec4 qs = vec4(unpack8((data_a_packed32[ib].qs[qsi / 2] >> qsshift) & 0x03030303));
            const uint scales = data_a[ib].scales[scalesi];
            const vec2 dm = vec2(data_a[ib].dm);

            const vec4 v = dm.x * float(scales & 0xF) * qs - dm.y * float(scales >> 4);

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2(v.xy));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(v.zw));
#elif defined(DATA_A_TQ1_0)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib  = idx / 128;               // 2 values per idx
            const uint iqs = (idx % 128) * 2;         // element 0,2,4..254

            const float d = float(data_a[ib].d);
            vec2 v;
            for (uint kk = 0u; kk < 2u; ++kk) {
                const uint e = iqs + kk;
                const uint bidx = tq1_0_byte_of(e);
                const uint qbyte = uint(bidx < 48u ? data_a[ib].qs[bidx]
                                                   : data_a[ib].qh[bidx - 48u]);
                v[kk] = d * (float(tq1_0_trit(qbyte, tq1_0_digit_of(e))) - 1.0);
            }

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair, FLOAT_TYPEV2(v.xy));
#elif defined(DATA_A_TQ2_0)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 128;                         // 2 values per idx
            const uint iqs = (idx % 128) * 2;                  // elem 0,2,4..254

            const uint qsi   = (iqs / 128) * 32 + (iqs % 32);  // byte pair start
            const uint shift = 2 * ((iqs % 128) / 32);         // 0,2,4,6

            const uvec2 qs = uvec2(data_a[ib].qs[qsi], data_a[ib].qs[qsi + 1]);
            const float d = float(data_a[ib].d);

            const vec2 v = d * (vec2((qs >> shift) & 3) - 1.0);

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair, FLOAT_TYPEV2(v.xy));
#elif defined(DATA_A_Q3_K)
#if LOAD_VEC_A == 8
            A_RAW_T raw;
            fetch_a(pos_a, row, col, a_lane_off(row, col), raw);
            store_a_raw(pos_a, row, col, a_shmem_index(col, row * LOAD_VEC_A / 2), raw);
#else
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 128;                   // 2 values per idx
            const uint iqs = idx % 128;                  // 0..127

            const uint n = iqs / 64;                     // 0,1
            const uint qsi = n * 32 + (iqs % 16) * 2;    // 0,2,4..62
            const uint hmi =          (iqs % 16) * 2;    // 0,2,4..30
            const uint j = (iqs % 64) / 4;               // 0..3
            const uint is = iqs / 8;                     // 0..15
            const uint halfsplit = ((iqs % 64) / 16);    // 0,1,2,3
            const uint qsshift = halfsplit * 2;          // 0,2,4,6

            const int8_t us = int8_t(((data_a[ib].scales[is % 8] >> (4 * int(is / 8))) & 0xF)
                                  | (((data_a[ib].scales[8 + (is % 4)] >> (2 * int(is / 4))) & 3) << 4));
            const float dl = float(data_a[ib].d) * float(us - 32);

            const vec2 qs = vec2(unpack8((uint(data_a_packed16[ib].qs[qsi / 2]) >> qsshift) & 0x0303).xy);
            const vec2 hm = vec2(unpack8(((uint(data_a_packed16[ib].hmask[hmi / 2]) >> (4 * n + halfsplit)) & 0x0101 ^ 0x0101) << 2).xy);

            store_a(col, row * LOAD_VEC_A / 2, FLOAT_TYPEV2(dl * (qs.x - hm.x),
                                                              dl * (qs.y - hm.y)));
#endif
#elif defined(DATA_A_Q4_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 64;                  // 4 values per idx
            const uint iqs = (idx % 64) * 2;           // 0,2,4..126

            const uint n = iqs / 32;                   // 0,1,2,3
            const uint b = (iqs % 32) / 16;            // 0,1
            const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126

#ifdef SCACHE_DM
            // (d, m) precomputed per (tile row, sub-block) at superblock boundaries
            const vec2 dm = vec2(scache_dm[col * 8 + 2 * n + b]);
            const float d = dm.x;
            const float m = dm.y;
#else
            const uint is = 2 * n + b;                 // 0..7
            const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126

            const vec2 loadd = vec2(data_a[ib].dm);

            const uvec3 scales = uvec3(data_a_packed32[ib].scales[0],
                                       data_a_packed32[ib].scales[1],
                                       data_a_packed32[ib].scales[2]);
            const uint scalesoffs = (is & 3) * 8;

            const uint scidx0 = (is < 4) ? 0 : 2;
            const uint scidxshift0 = scalesoffs;
            const uint scidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;
            const uint mbidx0 = (is < 4) ? 1 : 2;
            const uint mbidxshift0 = (is < 4) ? scalesoffs : scalesoffs + 4;
            const uint mbidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;

            const uint8_t sc    = uint8_t(((scales[scidx0] >> scidxshift0) & 0xF) | ((scales[0] >> scidxshift1) & 0x30));
            const uint8_t mbyte = uint8_t(((scales[mbidx0] >> mbidxshift0) & 0xF) | ((scales[1] >> mbidxshift1) & 0x30));

            const float d = loadd.x * sc;
            const float m = -loadd.y * mbyte;

            const vec4 q = vec4(unpack8((data_a_packed32[ib].qs[qsi / 4] >> (b * 4)) & 0x0F0F0F0F));

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2(fma(d, q.x, m), fma(d, q.y, m)));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(fma(d, q.z, m), fma(d, q.w, m)));
#elif defined(DATA_A_Q5_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 64;                  // 4 values per idx
            const uint iqs = (idx % 64) * 2;           // 0,2,4..126

            const uint n = iqs / 32;                   // 0,1,2,3
            const uint b = (iqs % 32) / 16;            // 0,1
            const uint is = 2 * n + b;                 // 0..7
            const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126
            const uint qhi = (iqs % 16) * 2;           // 0,2,4..30

#ifdef SCACHE_DM
            // (d, m) precomputed per (tile row, sub-block) at superblock boundaries
            const vec2 dm = vec2(scache_dm[col * 8 + 2 * n + b]);
            const float d = dm.x;
            const float m = dm.y;
#else
            const uint is = 2 * n + b;                 // 0..7

            const vec2 loadd = vec2(data_a[ib].dm);

            const uvec3 scales = uvec3(data_a_packed32[ib].scales[0],
                                       data_a_packed32[ib].scales[1],
                                       data_a_packed32[ib].scales[2]);
            const uint scalesoffs = (is & 3) * 8;

            const uint scidx0 = (is < 4) ? 0 : 2;
            const uint scidxshift0 = scalesoffs;
            const uint scidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;
            const uint mbidx0 = (is < 4) ? 1 : 2;
            const uint mbidxshift0 = (is < 4) ? scalesoffs : scalesoffs + 4;
            const uint mbidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;

            const uint8_t sc    = uint8_t(((scales[scidx0] >> scidxshift0) & 0xF) | ((scales[0] >> scidxshift1) & 0x30));
            const uint8_t mbyte = uint8_t(((scales[mbidx0] >> mbidxshift0) & 0xF) | ((scales[1] >> mbidxshift1) & 0x30));

            const float d = loadd.x * sc;
            const float m = -loadd.y * mbyte;

            const uint qs = (data_a_packed32[ib].qs[qsi / 4] >> (b * 4)) & 0x0F0F0F0F;
            const uint qh = ((data_a_packed32[ib].qh[qhi / 4] >> (iqs / 16)) & 0x01010101) << 4;
            const vec4 q = vec4(unpack8(qs | qh));

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2(fma(d, q.x, m), fma(d, q.y, m)));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(fma(d, q.z, m), fma(d, q.w, m)));
#elif defined(DATA_A_Q6_K)
#if LOAD_VEC_A == 8
            A_RAW_T raw;
            fetch_a(pos_a, row, col, a_lane_off(row, col), raw);
            store_a_raw(pos_a, row, col, a_shmem_index(col, row * LOAD_VEC_A / 2), raw);
#else
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 128;                  // 2 values per idx
            const uint iqs = idx % 128;                 // 0..127

            const uint n = iqs / 64;                    // 0,1
            const uint b = ((iqs % 64) / 32) * 4;       // 0,4
            const uint is_b = (iqs % 16) / 8;           // 0,1
            const uint qhshift = ((iqs % 64) / 16) * 2; // 0,2,4,6
            const uint is = 8 * n + qhshift + is_b;     // 0..15
            const uint qsi = n * 32 + (iqs % 32);       // 0..63
            const uint qhi = n * 16 + (iqs % 16);       // 0..31

            const float dscale = float(data_a[ib].d) * float(data_a[ib].scales[is]);

            const uint ql = (uint(data_a_packed16[ib].ql[qsi]) >> b) & 0x0F0F;
            const uint qh = (uint(data_a_packed16[ib].qh[qhi]) >> qhshift) & 0x0303;
            const vec2 q = (vec2(unpack8(ql | (qh << 4)).xy) - 32) * dscale;

            store_a(col, row * LOAD_VEC_A / 2, FLOAT_TYPEV2(q.x, q.y));
#endif
#elif defined(DATA_A_IQ1_S)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 32;                  // 8 values per idx
            const uint ib32 = (idx % 32) / 4;         // 0..7
            const uint ib8 = idx % 32;

            const float d = float(data_a[ib].d);
            const uint qh = data_a[ib].qh[ib32];
            const uint qs = data_a[ib].qs[ib8];
            const float dl = d * (2 * bitfieldExtract(qh, 12, 3) + 1);
            const float delta = ((qh & 0x8000) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;
            const int16_t grid = int16_t(iq1s_grid[qs | (bitfieldExtract(qh, 3 * int(ib8 & 3), 3) << 8)]);

            const uint k_pair = row * LOAD_VEC_A / 2;
            [[unroll]] for (int k = 0; k < 4; ++k) {
                store_a(col, k_pair + k, FLOAT_TYPEV2(dl * (bitfieldExtract(grid, 4 * k    , 2) + delta),
                                                       dl * (bitfieldExtract(grid, 4 * k + 2, 2) + delta)));
            }
#elif defined(DATA_A_IQ1_M)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 32;  // 8 values per idx
            const uint ib8 = idx % 32;
            const uint ib16 = ib8 / 2;

            const uint16_t[4] scales = data_a[ib].scales;
            const u16vec4 s = u16vec4(scales[0], scales[1], scales[2], scales[3]) >> 12;
            const float d = float(unpackHalf2x16(s.x | (s.y << 4) | (s.z << 8) | (s.w << 12)).x);
            const uint sc = scales[ib8 / 8];
            const uint qs = data_a[ib].qs[ib8];
            const uint qh = data_a[ib].qh[ib16] >> (4 * (ib8 & 1));
            const float dl = d * (2 * bitfieldExtract(sc, 3 * int(ib16 & 3), 3) + 1);
            const float delta = ((qh & 8) != 0) ? -IQ1M_DELTA : IQ1M_DELTA;
            const int16_t grid = int16_t(iq1s_grid[qs | ((qh & 7) << 8)]);

            const uint k_pair = row * LOAD_VEC_A / 2;
            [[unroll]] for (int k = 0; k < 4; ++k) {
                store_a(col, k_pair + k, FLOAT_TYPEV2(dl * (bitfieldExtract(grid, 4 * k    , 2) + delta),
                                                       dl * (bitfieldExtract(grid, 4 * k + 2, 2) + delta)));
            }
#elif defined(DATA_A_IQ2_XXS)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 32;                 // 8 values per idx
            const uint ib32 = (idx % 32) / 4;         // 0..7
            const uint ib8 = idx % 4;

            const float d = float(data_a[ib].d);
            const uint qs = data_a[ib].qs[8 * ib32 + ib8];
            const uint signs = pack32(u8vec4(
                data_a[ib].qs[8*ib32 + 4],
                data_a[ib].qs[8*ib32 + 5],
                data_a[ib].qs[8*ib32 + 6],
                data_a[ib].qs[8*ib32 + 7]
            ));
            const FLOAT_TYPE db = FLOAT_TYPE(d * 0.25 * (0.5 + (signs >> 28)));
            const uint32_t sign7 = bitfieldExtract(signs, 7 * int(ib8), 7);
            const uint sign = sign7 | (bitCount(sign7) << 7);
            const uvec2 grid = iq2xxs_grid[qs];
            const vec4 grid0 = vec4(unpack8(grid.x));
            const vec4 grid1 = vec4(unpack8(grid.y));

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     db * FLOAT_TYPEV2((sign &   1) != 0 ? -grid0.x : grid0.x,
                                                        (sign &   2) != 0 ? -grid0.y : grid0.y));
            store_a(col, k_pair + 1, db * FLOAT_TYPEV2((sign &   4) != 0 ? -grid0.z : grid0.z,
                                                        (sign &   8) != 0 ? -grid0.w : grid0.w));
            store_a(col, k_pair + 2, db * FLOAT_TYPEV2((sign &  16) != 0 ? -grid1.x : grid1.x,
                                                        (sign &  32) != 0 ? -grid1.y : grid1.y));
            store_a(col, k_pair + 3, db * FLOAT_TYPEV2((sign &  64) != 0 ? -grid1.z : grid1.z,
                                                        (sign & 128) != 0 ? -grid1.w : grid1.w));
#elif defined(DATA_A_IQ2_XS)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 32;            // 8 values per idx
            const uint ib32 = (idx % 32) / 4;    // 0..7
            const uint ib8 = idx % 4;            // 0..3

            const float d = float(data_a[ib].d);
            const uint scale = (data_a[ib].scales[ib32] >> (2 * (ib8 & 2))) & 0xf;
            const FLOAT_TYPE db = FLOAT_TYPE(d * 0.25 * (0.5 + scale));
            const uint qs = data_a[ib].qs[4 * ib32 + ib8];
            const uint sign7 = qs >> 9;
            const uint sign = sign7 | (bitCount(sign7) << 7);
            const uvec2 grid = iq2xs_grid[qs & 511];
            const vec4 grid0 = vec4(unpack8(grid.x));
            const vec4 grid1 = vec4(unpack8(grid.y));

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     db * FLOAT_TYPEV2((sign &   1) != 0 ? -grid0.x : grid0.x,
                                                        (sign &   2) != 0 ? -grid0.y : grid0.y));
            store_a(col, k_pair + 1, db * FLOAT_TYPEV2((sign &   4) != 0 ? -grid0.z : grid0.z,
                                                        (sign &   8) != 0 ? -grid0.w : grid0.w));
            store_a(col, k_pair + 2, db * FLOAT_TYPEV2((sign &  16) != 0 ? -grid1.x : grid1.x,
                                                        (sign &  32) != 0 ? -grid1.y : grid1.y));
            store_a(col, k_pair + 3, db * FLOAT_TYPEV2((sign &  64) != 0 ? -grid1.z : grid1.z,
                                                        (sign & 128) != 0 ? -grid1.w : grid1.w));
#elif defined(DATA_A_IQ2_S)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 32;  // 8 values per idx
            const uint ib8 = idx % 32; // 0..31
            const uint ib32 = ib8 / 4; // 0..7

            const uint scale = (data_a[ib].scales[ib32] >> (2 * (ib8 & 2))) & 0xf;
            const uint qs = data_a[ib].qs[ib8];
            const uint qh = data_a[ib].qh[ib32];
            const uint qhshift = 2 * (ib8 % 4);
            const uint sign = data_a[ib].qs[QUANT_K / 8 + ib8];

            const float d = float(data_a[ib].d);
            const FLOAT_TYPE db = FLOAT_TYPE(d * 0.25 * (0.5 + scale));
            const uvec2 grid = iq2s_grid[qs | ((qh << (8 - qhshift)) & 0x300)];
            const vec4 grid0 = vec4(unpack8(grid.x));
            const vec4 grid1 = vec4(unpack8(grid.y));

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     db * FLOAT_TYPEV2((sign &   1) != 0 ? -grid0.x : grid0.x,
                                                        (sign &   2) != 0 ? -grid0.y : grid0.y));
            store_a(col, k_pair + 1, db * FLOAT_TYPEV2((sign &   4) != 0 ? -grid0.z : grid0.z,
                                                        (sign &   8) != 0 ? -grid0.w : grid0.w));
            store_a(col, k_pair + 2, db * FLOAT_TYPEV2((sign &  16) != 0 ? -grid1.x : grid1.x,
                                                        (sign &  32) != 0 ? -grid1.y : grid1.y));
            store_a(col, k_pair + 3, db * FLOAT_TYPEV2((sign &  64) != 0 ? -grid1.z : grid1.z,
                                                        (sign & 128) != 0 ? -grid1.w : grid1.w));
#elif defined(DATA_A_IQ3_XXS)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 64;            // 4 values per idx
            const uint iqs = idx % 64;           // 0..63
            const uint is = QUANT_K / 4 + 4 * (iqs / 8); // 8 values

            const float d = float(data_a[ib].d);
            const uint qs = data_a[ib].qs[iqs];
            const uint signs = pack32(u16vec2(
                data_a_packed16[ib].qs[is/2],
                data_a_packed16[ib].qs[is/2+1]
            ));
            const float db = d * 0.5 * (0.5 + (signs >> 28));
            const uint32_t sign7 = bitfieldExtract(signs, 7 * (int(iqs / 2) % 4), 7);
            const uint sign = (sign7 | (bitCount(sign7) << 7)) >> (4 * (idx % 2));
            const uint grid = iq3xxs_grid[qs];
            const vec4 v = db * vec4(unpack8(grid));

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2((sign &   1) != 0 ? -v.x : v.x,
                                                   (sign &   2) != 0 ? -v.y : v.y));
            store_a(col, k_pair + 1, FLOAT_TYPEV2((sign &   4) != 0 ? -v.z : v.z,
                                                   (sign &   8) != 0 ? -v.w : v.w));
#elif defined(DATA_A_IQ3_S)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 64;            // 4 values per idx
            const uint iqs = idx % 64;           // 0..63
            const uint iqh = iqs / 8;

            const float d = float(data_a[ib].d);
            const uint qs = data_a[ib].qs[iqs];
            const uint qh = data_a[ib].qh[iqh];
            const int8_t sign = int8_t(data_a[ib].signs[iqs / 2] >> (4 * (idx % 2)));
            const uint scale = data_a[ib].scales[iqs / 16];
            const i8vec2 sign01 = i8vec2(1 - (2 & i8vec2(sign << 1, sign)));
            const float db = d * (1 + 2 * ((scale >> (4 * (iqh & 1))) & 0xf));
            const uint32_t grid = iq3s_grid[qs | ((qh << (8 - (iqs % 8))) & 256)];
            const vec4 v = db * vec4(unpack8(grid));

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2((sign &   1) != 0 ? -v.x : v.x,
                                                   (sign &   2) != 0 ? -v.y : v.y));
            store_a(col, k_pair + 1, FLOAT_TYPEV2((sign &   4) != 0 ? -v.z : v.z,
                                                   (sign &   8) != 0 ? -v.w : v.w));
#elif defined(DATA_A_IQ4_XS)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 64;            // 4 values per idx
            const uint ib32 = (idx % 64) / 8;    // 0..7
            const uint iq = 4 * ib32 + (idx % 4);

            const uint sl = (data_a[ib].scales_l[ib32/2] >> (4 * (ib32 & 1))) & 0xF;
            const uint sh = ((data_a[ib].scales_h) >> (2 * ib32)) & 3;
            const uint qshift = idx & 4;
            u8vec4 qs = unpack8((uint(data_a_packed32[ib].qs[iq]) >> qshift) & 0x0F0F0F0F);

            const float d = float(data_a[ib].d);
            const vec4 v = d * float(int(sl | (sh << 4)) - 32) * vec4(kvalues_iq4nl[qs.x], kvalues_iq4nl[qs.y], kvalues_iq4nl[qs.z], kvalues_iq4nl[qs.w]);

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2(v.xy));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(v.zw));
#elif defined(DATA_A_IQ4_NL)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 8;
            const uint iqs = idx & 0x07;

            const FLOAT_TYPE d = FLOAT_TYPE(data_a_packed16[ib].d);
            const uint vui = uint(data_a_packed16[ib].qs[iqs]);

            const uint k_pair = row * LOAD_VEC_A / 4;
            store_a(col, k_pair,     d * FLOAT_TYPEV2(kvalues_iq4nl[vui & 0xF],
                                                       kvalues_iq4nl[bitfieldExtract(vui, 8, 4)]));
            store_a(col, k_pair + 8, d * FLOAT_TYPEV2(kvalues_iq4nl[bitfieldExtract(vui, 4, 4)],
                                                       kvalues_iq4nl[vui >> 12]));
#elif defined(DATA_A_MXFP4)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 8;
            const uint iqs = (idx & 0x07) * 2;

            const uint vui = uint(data_a[ib].qs[iqs]);
            const uint vui2 = uint(data_a[ib].qs[iqs+1]);

#ifdef USE_OCP_FP4
            const float d = e8m0_to_fp32(data_a[ib].e);
            const u8vec2 packed = u8vec2(vui, vui2);
            store_a(col, row,     FLOAT_TYPEV2(bitcastExtractfe2m1EXT(packed, 0u)) * FLOAT_TYPE(d));
            store_a(col, row + 8, FLOAT_TYPEV2(bitcastExtractfe2m1EXT(packed, 4u)) * FLOAT_TYPE(d));
#else
            const float d = e8m0_to_fp32(data_a[ib].e) * 0.5;
            store_a(col, row,     FLOAT_TYPEV2(kvalues_mxfp4[vui  & 0xF] * d,
                                              kvalues_mxfp4[vui2 & 0xF] * d));
            store_a(col, row + 8, FLOAT_TYPEV2(kvalues_mxfp4[vui  >>  4] * d,
                                              kvalues_mxfp4[vui2 >>  4] * d));
#endif
#elif defined(DATA_A_ROCMFP4) || defined(DATA_A_ROCMFP4_FAST)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 8;
            const uint iqs = (idx & 0x07) * 2;

            const uint vui  = uint(data_a[ib].qs[iqs]);
            const uint vui2 = uint(data_a[ib].qs[iqs+1]);

            // low nibbles hold the first half of the block (scale e[0]),
            // high nibbles the second half (scale e[1])
#if defined(DATA_A_ROCMFP4)
            const float d0 = ue4m3_to_fp32(data_a[ib].e[0]);
            const float d1 = ue4m3_to_fp32(data_a[ib].e[1]);
#else
            const float d0 = ue4m3_to_fp32(data_a[ib].e);
            const float d1 = d0;
#endif
            store_a(col, row,     FLOAT_TYPEV2(float(kvalues_rocmfp4[vui  & 0xF]) * d0,
                                               float(kvalues_rocmfp4[vui2 & 0xF]) * d0));
            store_a(col, row + 8, FLOAT_TYPEV2(float(kvalues_rocmfp4[vui  >>  4]) * d1,
                                               float(kvalues_rocmfp4[vui2 >>  4]) * d1));
#elif defined(DATA_A_ROCMFPX_FP3)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 8;
            const uint iqs = (idx & 0x07) * 4;
            const vec4 v = rocmfpx_mm_fp3_vec4(ib, iqs);

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2(v.xy));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(v.zw));
#elif defined(DATA_A_ROCMFPX_FP6)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 8;
            const uint iqs = (idx & 0x07) * 4;
            const vec4 v = rocmfpx_mm_fp6_vec4(ib, iqs);

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2(v.xy));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(v.zw));
#elif defined(DATA_A_ROCMFPX_FP8)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;

            const uint ib = idx / 8;
            const uint iqs = (idx & 0x07) * 4;

            const float d = ue4m3_to_fp32(data_a[ib].e);
            const vec4 v = vec4(float(data_a[ib].qs[iqs + 0u]),
                                float(data_a[ib].qs[iqs + 1u]),
                                float(data_a[ib].qs[iqs + 2u]),
                                float(data_a[ib].qs[iqs + 3u])) * d;

            const uint k_pair = row * LOAD_VEC_A / 2;
            store_a(col, k_pair,     FLOAT_TYPEV2(v.xy));
            store_a(col, k_pair + 1, FLOAT_TYPEV2(v.zw));
#elif defined(DATA_A_NVFP4)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint ib = idx / 16u;
            const uint sub = (idx & 0xC) >> 2;
            const uint iqs = (idx & 0xF) * 2;
            const uint vui = uint(data_a[ib].qs[iqs]);
            const uint vui2 = uint(data_a[ib].qs[iqs+1]);

            // lo and hi nibbles are 8 elements apart, which doesn't quite line up with
            // how the thread mapping and buf_idx calculation works for other types.
            const uint eff_row = (row & 3) + (row & ~3) * 2;
#ifdef USE_OCP_FP4
            const FLOAT_TYPE d = FLOAT_TYPE(ue4m3_from_bits(data_a[ib].d[sub]));
            const u8vec2 packed = u8vec2(vui, vui2);
            store_a(col, eff_row,     FLOAT_TYPEV2(bitcastExtractfe2m1EXT(packed, 0u)) * d);
            store_a(col, eff_row + 4, FLOAT_TYPEV2(bitcastExtractfe2m1EXT(packed, 4u)) * d);
#else
            const float d = ue4m3_to_fp32(data_a[ib].d[sub]) * 0.5;
            store_a(col, eff_row,     FLOAT_TYPEV2(kvalues_mxfp4[vui  & 0xF] * d,
                                                    kvalues_mxfp4[vui2 & 0xF] * d));
            store_a(col, eff_row + 4, FLOAT_TYPEV2(kvalues_mxfp4[vui  >>  4] * d,
                                                    kvalues_mxfp4[vui2 >>  4] * d));
#endif
#endif
}

// ---- register prefetch (dense coopmat path, aligned pipelines) ----------------------------------
// The fused load_*_to_shmem functions issue the global load and immediately consume it, so every BK
// step stalls for the full memory latency before the tile reaches LDS (ISA audit 2026-09-13: the
// f32-accumulator loop dropped from 526 to 340 instructions for +4% at n=512, i.e. the loop is
// latency-bound, not issue-bound). fetch_* only issue the loads for the NEXT tile into registers;
// store_*_raw dequantize/convert those registers into LDS one iteration later, after the WMMAs of
// the current tile have covered the latency. Types are added as their fetch/store split is written.
#if defined(COOPMAT) && (defined(DATA_A_F16) || defined(DATA_A_F32)) && LOAD_VEC_A == 8
// f16 / f32 A, aligned pipelines only (the unaligned ones keep the fused scalar loader; see mul_mm.comp)
#define A_PREFETCH 1
#define A_RAW_T FLOAT_TYPEV8
uint a_lane_off(const uint row, const uint col) { return col * p.stride_a / LOAD_VEC_A + row; }
void fetch_a(const uint pos_a, const uint row, const uint col, const uint lane_off, out A_RAW_T raw) {
    raw = FLOAT_TYPEV8(data_a[pos_a + lane_off]);
}
void store_a_raw(const uint pos_a, const uint row, const uint col, const uint sidx, const A_RAW_T raw) {
    buf_a[sidx]     = TO_BUF(raw[0].xy);
    buf_a[sidx + 1] = TO_BUF(raw[0].zw);
    buf_a[sidx + 2] = TO_BUF(raw[1].xy);
    buf_a[sidx + 3] = TO_BUF(raw[1].zw);
}
#elif defined(COOPMAT) && (defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)) && defined(SCACHE_DM)
#define A_PREFETCH 1
// Index math with the uniform part split out. For K-quants stride_a is a multiple of 256 and pos_a
// advances by BK/LOAD_VEC_A = 8 per step, so idx/64 = pos_a/64 + col*stride_a/256 and idx%64 =
// pos_a%64 + row (row < 8, no carry). pos_a is uniform, so the superblock column kb and the 32-wide
// sub-block sub are scalar; the per-lane part comes in precomputed (a_lane_off = the lane's
// superblock row offset in dwords, a_store_idx = its LDS index).
#if defined(DATA_A_Q4_K)
#define A_RAW_T uint
#else
#define A_RAW_T uvec2   // x: qs dword, y: qh dword
#endif
#if defined(DATA_A_Q4_K)
uint a_lane_off(const uint row, const uint col) { return (col * (p.stride_a / 256)) * 36 + row; }   // block_q4_K = 36 dwords
#else
uint a_lane_off(const uint row, const uint col) { return (col * (p.stride_a / 256)) * 44 + row; }   // block_q5_K = 44 dwords
#endif
void fetch_a(const uint pos_a, const uint row, const uint col, const uint lane_off, out A_RAW_T raw) {
    const uint kb  = pos_a / 64;
    const uint sub = (pos_a % 64) / 8;
#if defined(DATA_A_Q4_K)
    // dword layout of block_q4_K: dm (1), scales (3), qs (32)
    raw = data_a_u32[kb * 36 + lane_off + 4 + 8 * (sub / 2)];
#else
    // dword layout of block_q5_K: dm (1), scales (3), qh (8), qs (32); lane_off counts 44-dword blocks
    raw.x = data_a_u32[kb * 44 + lane_off + 12 + 8 * (sub / 2)];
    raw.y = data_a_u32[kb * 44 + lane_off + 4];
#endif
}
void store_a_raw(const uint pos_a, const uint row, const uint col, const uint sidx, const A_RAW_T raw) {
    const uint sub = (pos_a % 64) / 8;
    const vec2 dm = vec2(scache_dm[col * 8 + sub]);
#if defined(DATA_A_Q4_K)
    const vec4 q = vec4(unpack8((raw >> (4 * (sub % 2))) & 0x0F0F0F0F));
#else
    const uint qs = (raw.x >> (4 * (sub % 2))) & 0x0F0F0F0F;
    const uint qh = ((raw.y >> sub) & 0x01010101) << 4;
    const vec4 q = vec4(unpack8(qs | qh));
#endif
    buf_a[sidx]     = TO_BUF(FLOAT_TYPEV2(fma(dm.x, q.x, dm.y), fma(dm.x, q.y, dm.y)));
    buf_a[sidx + 1] = TO_BUF(FLOAT_TYPEV2(fma(dm.x, q.z, dm.y), fma(dm.x, q.w, dm.y)));
}
#elif defined(COOPMAT) && defined(DATA_A_Q8_0) && LOAD_VEC_A != 8
#define A_PREFETCH 1
#define A_RAW_T uvec2   // x: d bits, y: the lane's 4 int8 values
uint a_lane_off(const uint row, const uint col) { return 0; }
void fetch_a(const uint pos_a, const uint row, const uint col, const uint lane_off, out A_RAW_T raw) {
    const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
    const uint ib = idx / 8;
    const uint iqs = idx & 0x07;
    raw.x = uint(float16BitsToUint16(data_a_packed16[ib].d));
    raw.y = pack32(u16vec2(data_a_packed16[ib].qs[2*iqs], data_a_packed16[ib].qs[2*iqs + 1]));
}
void store_a_raw(const uint pos_a, const uint row, const uint col, const uint sidx, const A_RAW_T raw) {
    const float d = float(uint16BitsToFloat16(uint16_t(raw.x)));
    const i8vec4 q = unpack8(int(raw.y));
    const vec4 v = vec4(q) * d;
    const uint k_pair = row * LOAD_VEC_A / 2;
    store_a(col, k_pair,     FLOAT_TYPEV2(v.xy));
    store_a(col, k_pair + 1, FLOAT_TYPEV2(v.zw));
}
#elif defined(COOPMAT) && defined(DATA_A_Q5_0) && LOAD_VEC_A != 8
#define A_PREFETCH 1
#define A_RAW_T uvec2   // x: d bits | qs<<16, y: qh
uint a_lane_off(const uint row, const uint col) { return 0; }
void fetch_a(const uint pos_a, const uint row, const uint col, const uint lane_off, out A_RAW_T raw) {
    const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
    const uint ib = idx / 8;
    const uint iqs = idx & 0x07;
    raw.x = uint(float16BitsToUint16(data_a_packed16[ib].d)) | (uint(data_a_packed16[ib].qs[iqs]) << 16);
    raw.y = uint(data_a_packed16[ib].qh[1]) << 16 | uint(data_a_packed16[ib].qh[0]);
}
void store_a_raw(const uint pos_a, const uint row, const uint col, const uint sidx, const A_RAW_T raw) {
    const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
    const uint iqs = idx & 0x07;
    const float d = float(uint16BitsToFloat16(uint16_t(raw.x & 0xFFFF)));
    const uint uint_qh = raw.y;
    const ivec2 qh0 = ivec2(((uint_qh >> 2*iqs) << 4) & 0x10, (uint_qh >> (2*iqs + 12)) & 0x10);
    const ivec2 qh1 = ivec2(((uint_qh >> (2*iqs + 1)) << 4) & 0x10, (uint_qh >> (2*iqs + 13)) & 0x10);
    const uint vui = raw.x >> 16;
    const vec4 v = (vec4((vui & 0xF) | qh0.x, ((vui >> 4) & 0xF) | qh0.y, ((vui >> 8) & 0xF) | qh1.x, (vui >> 12) | qh1.y) - 16.0f) * d;
    store_a(col, row,     FLOAT_TYPEV2(v.xz));
    store_a(col, row + 8, FLOAT_TYPEV2(v.yw));
}
#elif defined(COOPMAT) && defined(DATA_A_Q6_K) && LOAD_VEC_A != 8
#define A_PREFETCH 1
#define A_RAW_T uvec2   // x: d bits | ql<<16, y: qh | scale<<16
uint a_lane_off(const uint row, const uint col) { return 0; }
void fetch_a(const uint pos_a, const uint row, const uint col, const uint lane_off, out A_RAW_T raw) {
    const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
    const uint ib = idx / 128;
    const uint iqs = idx % 128;
    const uint n = iqs / 64;
    const uint is_b = (iqs % 16) / 8;
    const uint qhshift = ((iqs % 64) / 16) * 2;
    const uint is = 8 * n + qhshift + is_b;
    const uint qsi = n * 32 + (iqs % 32);
    const uint qhi = n * 16 + (iqs % 16);
    raw.x = uint(float16BitsToUint16(data_a[ib].d)) | (uint(data_a_packed16[ib].ql[qsi]) << 16);
    raw.y = uint(data_a_packed16[ib].qh[qhi]) | (uint(uint8_t(data_a[ib].scales[is])) << 16);
}
void store_a_raw(const uint pos_a, const uint row, const uint col, const uint sidx, const A_RAW_T raw) {
    const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
    const uint iqs = idx % 128;
    const uint b = ((iqs % 64) / 32) * 4;
    const uint qhshift = ((iqs % 64) / 16) * 2;
    const float dscale = float(uint16BitsToFloat16(uint16_t(raw.x & 0xFFFF))) * float(int8_t(uint8_t(raw.y >> 16)));
    const uint ql = ((raw.x >> 16) >> b) & 0x0F0F;
    const uint qh = ((raw.y & 0xFFFF) >> qhshift) & 0x0303;
    const vec2 q = (vec2(unpack8(ql | (qh << 4)).xy) - 32) * dscale;
    store_a(col, row * LOAD_VEC_A / 2, FLOAT_TYPEV2(q.x, q.y));
}
#elif defined(COOPMAT) && defined(DATA_A_Q3_K) && LOAD_VEC_A != 8
#define A_PREFETCH 1
#define A_RAW_T uvec2   // x: d bits | qs<<16, y: hmask | sc0<<16 | sc1<<24
uint a_lane_off(const uint row, const uint col) { return 0; }
void fetch_a(const uint pos_a, const uint row, const uint col, const uint lane_off, out A_RAW_T raw) {
    const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
    const uint ib = idx / 128;
    const uint iqs = idx % 128;
    const uint n = iqs / 64;
    const uint qsi = n * 32 + (iqs % 16) * 2;
    const uint hmi =          (iqs % 16) * 2;
    const uint is = iqs / 8;
    raw.x = uint(float16BitsToUint16(data_a[ib].d)) | (uint(data_a_packed16[ib].qs[qsi / 2]) << 16);
    raw.y = uint(data_a_packed16[ib].hmask[hmi / 2]) | (uint(data_a[ib].scales[is % 8]) << 16) | (uint(data_a[ib].scales[8 + (is % 4)]) << 24);
}
void store_a_raw(const uint pos_a, const uint row, const uint col, const uint sidx, const A_RAW_T raw) {
    const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
    const uint iqs = idx % 128;
    const uint n = iqs / 64;
    const uint is = iqs / 8;
    const uint halfsplit = ((iqs % 64) / 16);
    const uint qsshift = halfsplit * 2;
    const uint sc0 = (raw.y >> 16) & 0xFF;
    const uint sc1 = (raw.y >> 24) & 0xFF;
    const int8_t us = int8_t(((sc0 >> (4 * int(is / 8))) & 0xF) | (((sc1 >> (2 * int(is / 4))) & 3) << 4));
    const float dl = float(uint16BitsToFloat16(uint16_t(raw.x & 0xFFFF))) * float(us - 32);
    const vec2 qs = vec2(unpack8(((raw.x >> 16) >> qsshift) & 0x0303).xy);
    const vec2 hm = vec2(unpack8((((raw.y & 0xFFFF) >> (4 * n + halfsplit)) & 0x0101 ^ 0x0101) << 2).xy);
    store_a(col, row * LOAD_VEC_A / 2, FLOAT_TYPEV2(dl * (qs.x - hm.x), dl * (qs.y - hm.y)));
}
#endif
#if defined(COOPMAT) && LOAD_VEC_B == 8 && !defined(DATA_B_BF16)
#define B_PREFETCH 1
void fetch_b(const uint pos_b, const uint row, const uint col, out B_TYPE raw) {
#ifdef MUL_MAT_ID
    const u16vec2 row_idx = row_ids[col];
    raw = data_b[pos_b + row_idx.y * p.batch_stride_b / LOAD_VEC_B + (row_idx.x % p.ne11) * p.stride_b / LOAD_VEC_B + row];
#else
    raw = data_b[pos_b + col * p.stride_b / LOAD_VEC_B + row];
#endif
}
void store_b_raw(const uint buf_idx, const B_TYPE raw) {
    FLOAT_TYPEV8 bb = FLOAT_TYPEV8(raw);
    buf_b[buf_idx + 0] = TO_BUF(bb[0].xy);
    buf_b[buf_idx + 1] = TO_BUF(bb[0].zw);
    buf_b[buf_idx + 2] = TO_BUF(bb[1].xy);
    buf_b[buf_idx + 3] = TO_BUF(bb[1].zw);
}
#endif

#if !defined(MUL_MAT_ID)
void load_b_to_shmem(const uint pos_b, const uint row, const uint col, const uint idx_n, const uint block, const uint end_k) {
#if LOAD_VEC_B == 8
            if (ALIGNED != 0) {
                // Not supported for b_type bf16 because bf16mat2x4 does not exist
                const uint idx = pos_b + col * p.stride_b / LOAD_VEC_B + row;
                const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
                FLOAT_TYPEV8 bb = FLOAT_TYPEV8(data_b[idx]);
                buf_b[buf_idx + 0] = TO_BUF(bb[0].xy);
                buf_b[buf_idx + 1] = TO_BUF(bb[0].zw);
                buf_b[buf_idx + 2] = TO_BUF(bb[1].xy);
                buf_b[buf_idx + 3] = TO_BUF(bb[1].zw);
                return;
            }
#elif LOAD_VEC_B == 4
            if (ALIGNED != 0) {
                const uint idx = pos_b + col * p.stride_b / LOAD_VEC_B + row;
                const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
#if defined(DATA_B_BF16)
                FLOAT_TYPEV4 bb = FLOAT_TYPEV4(TO_FLOAT_TYPE(data_b[idx]));
#else
                FLOAT_TYPEV4 bb = FLOAT_TYPEV4(data_b[idx]);
#endif
                buf_b[buf_idx + 0] = TO_BUF(bb.xy);
                buf_b[buf_idx + 1] = TO_BUF(bb.zw);
                return;
            }
#endif
            const uint idx = pos_b + col * p.stride_b + row * 2;
            const uint buf_idx = col * SHMEM_STRIDE + row;
            if (idx_n < p.N && block + row * 2 + 1 < end_k) {
                buf_b[buf_idx] = TO_BUF(FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]),
                                              TO_FLOAT_TYPE(data_b_scalar[idx + 1])));
            } else if (idx_n < p.N && block + row * 2 < end_k) {
                buf_b[buf_idx] = TO_BUF(FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]), 0.0f));
            } else {
                buf_b[buf_idx] = TO_BUF(FLOAT_TYPEV2(0.0f));
            }
}
#else
void load_b_to_shmem(const uint pos_b, const uint row, const uint col, const uint ic, const uint _ne1, const uint block, const uint end_k) {
#if LOAD_VEC_B == 8
            if (ALIGNED != 0) {
                // Not supported for b_type bf16 because bf16mat2x4 does not exist
                const u16vec2 row_idx = row_ids[col];
                const uint idx = pos_b + row_idx.y * p.batch_stride_b / LOAD_VEC_B + (row_idx.x % p.ne11) * p.stride_b / LOAD_VEC_B + row;
                const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
                FLOAT_TYPEV8 bb = FLOAT_TYPEV8(data_b[idx]);
                buf_b[buf_idx + 0] = TO_BUF(bb[0].xy);
                buf_b[buf_idx + 1] = TO_BUF(bb[0].zw);
                buf_b[buf_idx + 2] = TO_BUF(bb[1].xy);
                buf_b[buf_idx + 3] = TO_BUF(bb[1].zw);
                return;
            }
#elif LOAD_VEC_B == 4
            if (ALIGNED != 0) {
                const u16vec2 row_idx = row_ids[col];
                const uint idx = pos_b + row_idx.y * p.batch_stride_b / LOAD_VEC_B + (row_idx.x % p.ne11) * p.stride_b / LOAD_VEC_B + row;
                const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
#if defined(DATA_B_BF16)
                FLOAT_TYPEV4 bb = FLOAT_TYPEV4(TO_FLOAT_TYPE(data_b[idx]));
#else
                FLOAT_TYPEV4 bb = FLOAT_TYPEV4(data_b[idx]);
#endif
                buf_b[buf_idx + 0] = TO_BUF(bb.xy);
                buf_b[buf_idx + 1] = TO_BUF(bb.zw);
                return;
            }
#endif
            const uint row_i = ic * BN + col;
            const uint buf_idx = col * SHMEM_STRIDE + row;
            if (row_i < _ne1 && block + row * 2 + 1 < end_k) {
                const u16vec2 row_idx = row_ids[col];
                const uint idx = pos_b + row_idx.y * p.batch_stride_b + (row_idx.x % p.ne11) * p.stride_b + row * 2;
                buf_b[buf_idx] = TO_BUF(FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]),
                                              TO_FLOAT_TYPE(data_b_scalar[idx + 1])));
            } else if (row_i < _ne1 && block + row * 2 < end_k) {
                const u16vec2 row_idx = row_ids[col];
                const uint idx = pos_b + row_idx.y * p.batch_stride_b + (row_idx.x % p.ne11) * p.stride_b + row * 2;
                buf_b[buf_idx] = TO_BUF(FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]), 0.0f));
            } else {
                buf_b[buf_idx] = TO_BUF(FLOAT_TYPEV2(0.0f));
            }
}
#endif
