// Per-quant-type data structures and functions for the cm1 int8 coopmat path.
// Each quant type defines:
//   struct block_a_prefetch  — register data for one A-block per thread
//   block_a_load()           — load from global memory into a block_a_prefetch
//   block_a_to_shmem()       — unpack and write to shared memory

#if defined(DATA_A_IQ2_XXS) || defined(DATA_A_IQ2_XS) || defined(DATA_A_IQ2_S) || defined(DATA_A_IQ3_XXS) || defined(DATA_A_IQ3_S)
// 7 stored sign bits plus the implied parity bit (ksigns_iq2xs)
uint cm1_iq_signs8(uint s) {
    const uint s7 = s & 127;
    return s7 | ((bitCount(s7) & 1) << 7);
}

// negate the bytes of g selected by the low 4 bits of s; grid bytes are never 0, so +1 cannot carry
uint cm1_apply_signs4(uint g, uint s) {
    const uint m = (((s & 15) * 0x00204081u) & 0x01010101u) * 0xFFu;
    return (g ^ m) + (m & 0x01010101u);
}
#endif

#if defined(DATA_A_TQ1_0) || defined(DATA_A_TQ2_0) || defined(DATA_A_Q2_0) || defined(DATA_A_PTQ1_0)
// bytes 0..3 -> bytes - 1 as int8
uint cm1_sub1_bytes(uint v) {
    return ((v | 0x80808080u) - 0x01010101u) ^ 0x80808080u;
}
#endif

#if defined(DATA_A_Q2_0)
// the four 2-bit codes of byte b, one per output byte
uint cm1_spread2(uint b) {
    return (b | (b << 6) | (b << 12) | (b << 18)) & 0x03030303u;
}
#endif

#if defined(DATA_A_TQ1_0) || defined(DATA_A_PTQ1_0)
uint cm1_pow3(uint n) {
    return n < 2u ? (n == 0u ? 1u : 3u) : (n == 2u ? 9u : (n == 3u ? 27u : 81u));
}

// base-3 digit of each byte of w at multiplier p = 3^n: ((uint8(b * p)) * 3) >> 8, two bytes per 16-bit lane
uint cm1_trits4(uint w, uint p) {
    uint e = ((w & 0x00FF00FFu) * p) & 0x00FF00FFu;
    uint o = (((w >> 8) & 0x00FF00FFu) * p) & 0x00FF00FFu;
    e = ((e * 3u) >> 8) & 0x00030003u;
    o = ((o * 3u) >> 8) & 0x00030003u;
    return e | (o << 8);
}
#endif

#if defined(DATA_A_Q4_0)

struct block_a_prefetch {
    uint32_t qs;
    float16_t d;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = pack32(u16vec2(data_a_packed16[ib].qs[loadr * 2],
                             data_a_packed16[ib].qs[loadr * 2 + 1]));
    blk.d = data_a_packed16[ib].d;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    uint32_t lo4 = blk.qs & 0x0F0F0F0F;
    uint32_t hi4 = (blk.qs >> 4) & 0x0F0F0F0F;
    lo4 = ((lo4 | 0x80808080) - 0x08080808) ^ 0x80808080;
    hi4 = ((hi4 | 0x80808080) - 0x08080808) ^ 0x80808080;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = lo4;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = hi4;

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = float(blk.d);
    }
}

#elif defined(DATA_A_Q4_1)

struct block_a_prefetch {
    uint32_t qs;
    f16vec2 dm;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = data_a_packed32[ib].qs[loadr];
    blk.dm = data_a_packed32[ib].dm;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    // Store raw unsigned nibbles; the -8 offset is absorbed by the min term.
    uint32_t lo4 = blk.qs & 0x0F0F0F0F;
    uint32_t hi4 = (blk.qs >> 4) & 0x0F0F0F0F;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = lo4;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = hi4;

    if (loadr == 0) {
        buf_a_dm[ks * BM + buf_ib] = vec2(float(blk.dm.x), float(blk.dm.y));
    }
}

#elif defined(DATA_A_Q5_0)

struct block_a_prefetch {
    uint32_t qs;
    float16_t d;
    uint32_t qh;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = pack32(u16vec2(data_a_packed16[ib].qs[loadr * 2],
                             data_a_packed16[ib].qs[loadr * 2 + 1]));
    blk.d = data_a_packed16[ib].d;
    blk.qh = pack32(u16vec2(data_a_packed16[ib].qh[0], data_a_packed16[ib].qh[1]));
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    uint32_t lo4 = blk.qs & 0x0F0F0F0F;
    uint32_t hi4 = (blk.qs >> 4) & 0x0F0F0F0F;
    lo4 |= ((blk.qh >> (4u * loadr       )) & 0xFu) * 0x02040810u & 0x10101010u;
    hi4 |= ((blk.qh >> (4u * loadr + 16u )) & 0xFu) * 0x02040810u & 0x10101010u;
    lo4 = ((lo4 | 0x80808080) - 0x10101010) ^ 0x80808080;
    hi4 = ((hi4 | 0x80808080) - 0x10101010) ^ 0x80808080;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = lo4;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = hi4;

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = float(blk.d);
    }
}

#elif defined(DATA_A_Q5_1)

struct block_a_prefetch {
    uint32_t qs;
    f16vec2 dm;
    uint32_t qh;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = data_a_packed32[ib].qs[loadr];
    blk.dm = data_a_packed32[ib].dm;
    blk.qh = data_a_packed32[ib].qh;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    // Store raw unsigned 5-bit values; the -16 offset is absorbed by the min term.
    uint32_t lo4 = blk.qs & 0x0F0F0F0F;
    uint32_t hi4 = (blk.qs >> 4) & 0x0F0F0F0F;
    lo4 |= ((blk.qh >> (4u * loadr       )) & 0xFu) * 0x02040810u & 0x10101010u;
    hi4 |= ((blk.qh >> (4u * loadr + 16u )) & 0xFu) * 0x02040810u & 0x10101010u;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = lo4;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = hi4;

    if (loadr == 0) {
        buf_a_dm[ks * BM + buf_ib] = vec2(float(blk.dm.x), float(blk.dm.y));
    }
}

#elif defined(DATA_A_Q8_0)

struct block_a_prefetch {
    uint32_t qs;
    float16_t d;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = pack32(u16vec2(data_a_packed16[ib].qs[loadr * 2],
                             data_a_packed16[ib].qs[loadr * 2 + 1]));
    blk.d = data_a_packed16[ib].d;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr] = blk.qs;

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = float(blk.d);
    }
}

#elif defined(DATA_A_IQ4_NL)

struct block_a_prefetch {
    uint32_t qs;
    float16_t d;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = pack32(u16vec2(data_a_packed16[ib].qs[loadr * 2],
                             data_a_packed16[ib].qs[loadr * 2 + 1]));
    blk.d = data_a_packed16[ib].d;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    const u8vec4 lo_idx = unpack8(blk.qs & 0x0F0F0F0F);
    const u8vec4 hi_idx = unpack8((blk.qs >> 4) & 0x0F0F0F0F);
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] =
        pack32(i8vec4(cm1_kvalues[lo_idx.x], cm1_kvalues[lo_idx.y],
                      cm1_kvalues[lo_idx.z], cm1_kvalues[lo_idx.w]));
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] =
        pack32(i8vec4(cm1_kvalues[hi_idx.x], cm1_kvalues[hi_idx.y],
                      cm1_kvalues[hi_idx.z], cm1_kvalues[hi_idx.w]));

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = float(blk.d);
    }
}

#elif defined(DATA_A_IQ4_XS)

struct block_a_prefetch {
    uint32_t qs;
    float d;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;
    blk.qs = data_a_packed32[ib_k].qs[4 * ib32 + loadr];
    blk.d = 0.0;
    if (loadr == 0) {
        const uint sl = (data_a_packed32[ib_k].scales_l >> (4 * ib32)) & 0xF;
        const uint sh = (data_a_packed32[ib_k].scales_h >> (2 * ib32)) & 3;
        blk.d = float(data_a_packed32[ib_k].d) * float(int(sl | (sh << 4)) - 32);
    }
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    const u8vec4 lo_idx = unpack8(blk.qs & 0x0F0F0F0F);
    const u8vec4 hi_idx = unpack8((blk.qs >> 4) & 0x0F0F0F0F);
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] =
        pack32(i8vec4(cm1_kvalues[lo_idx.x], cm1_kvalues[lo_idx.y],
                      cm1_kvalues[lo_idx.z], cm1_kvalues[lo_idx.w]));
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] =
        pack32(i8vec4(cm1_kvalues[hi_idx.x], cm1_kvalues[hi_idx.y],
                      cm1_kvalues[hi_idx.z], cm1_kvalues[hi_idx.w]));

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = blk.d;
    }
}

#elif defined(DATA_A_MXFP4)

struct block_a_prefetch {
    uint32_t qs;
    uint8_t e;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = pack32(u8vec4(data_a[ib].qs[loadr * 4],
                            data_a[ib].qs[loadr * 4 + 1],
                            data_a[ib].qs[loadr * 4 + 2],
                            data_a[ib].qs[loadr * 4 + 3]));
    blk.e = data_a[ib].e;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    const u8vec4 lo_idx = unpack8(blk.qs & 0x0F0F0F0F);
    const u8vec4 hi_idx = unpack8((blk.qs >> 4) & 0x0F0F0F0F);
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] =
        pack32(i8vec4(cm1_kvalues[lo_idx.x], cm1_kvalues[lo_idx.y],
                      cm1_kvalues[lo_idx.z], cm1_kvalues[lo_idx.w]));
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] =
        pack32(i8vec4(cm1_kvalues[hi_idx.x], cm1_kvalues[hi_idx.y],
                      cm1_kvalues[hi_idx.z], cm1_kvalues[hi_idx.w]));

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = e8m0_to_fp32(blk.e) * 0.5;
    }
}

// LOAD_VEC_A=8 for k-quants and NVFP4: loadr has 4 positions, each writes 2 uint32

#elif defined(DATA_A_Q4_K)

struct block_a_prefetch {
    uint32_t qs0;
    uint32_t qs1;
    uint ib;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint sub = ib % 8;
    const uint qs_base = (sub >> 1) * 8;

    uint32_t raw0 = data_a_packed32[ib_k].qs[qs_base + loadr * 2];
    uint32_t raw1 = data_a_packed32[ib_k].qs[qs_base + loadr * 2 + 1];
    if ((sub & 1u) != 0u) {
        blk.qs0 = (raw0 >> 4) & 0x0F0F0F0F;
        blk.qs1 = (raw1 >> 4) & 0x0F0F0F0F;
    } else {
        blk.qs0 = raw0 & 0x0F0F0F0F;
        blk.qs1 = raw1 & 0x0F0F0F0F;
    }
    blk.ib = ib;

    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    // Store raw unsigned nibbles (blk.qs already masked); no -8 recentering needed.
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = blk.qs0;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = blk.qs1;

    if (loadr == 0) {
        const uint ib_k = blk.ib / 8;
        const uint sub = blk.ib % 8;
        const uint j = sub & 3u;
        const uint s_j  = uint(data_a[ib_k].scales[j]);
        const uint s_j4 = uint(data_a[ib_k].scales[j + 4]);
        const uint s_j8 = uint(data_a[ib_k].scales[j + 8]);
        const uint sc_val = (sub < 4) ? (s_j  & 0x3Fu) : ((s_j8 & 0x0Fu) | ((s_j  >> 6) << 4));
        const uint mn_val = (sub < 4) ? (s_j4 & 0x3Fu) : ((s_j8 >> 4)    | ((s_j4 >> 6) << 4));
        vec2 dm = vec2(data_a_packed32[ib_k].dm);
        float d_scaled = dm.x * float(sc_val);
        buf_a_dm[ks * BM + buf_ib] = vec2(d_scaled, -(dm.y * float(mn_val)));
    }
}

#elif defined(DATA_A_Q5_K)

struct block_a_prefetch {
    uint32_t qs0;
    uint32_t qs1;
    uint32_t qh0;
    uint32_t qh1;
    uint ib;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint sub = ib % 8;
    const uint qs_base = (sub >> 1) * 8;

    uint32_t raw0 = data_a_packed32[ib_k].qs[qs_base + loadr * 2];
    uint32_t raw1 = data_a_packed32[ib_k].qs[qs_base + loadr * 2 + 1];
    if ((sub & 1u) != 0u) {
        blk.qs0 = (raw0 >> 4) & 0x0F0F0F0F;
        blk.qs1 = (raw1 >> 4) & 0x0F0F0F0F;
    } else {
        blk.qs0 = raw0 & 0x0F0F0F0F;
        blk.qs1 = raw1 & 0x0F0F0F0F;
    }
    blk.qh0 = ((data_a_packed32[ib_k].qh[loadr * 2    ] >> sub) & 0x01010101) << 4;
    blk.qh1 = ((data_a_packed32[ib_k].qh[loadr * 2 + 1] >> sub) & 0x01010101) << 4;
    blk.ib = ib;

    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    // Store raw unsigned 5-bit values (qs nibble | qh bit); no -16 recentering needed.
    uint32_t v0 = blk.qs0 | blk.qh0;
    uint32_t v1 = blk.qs1 | blk.qh1;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = v0;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = v1;

    if (loadr == 0) {
        const uint ib_k = blk.ib / 8;
        const uint sub = blk.ib % 8;
        const uint j = sub & 3u;
        const uint s_j  = uint(data_a[ib_k].scales[j]);
        const uint s_j4 = uint(data_a[ib_k].scales[j + 4]);
        const uint s_j8 = uint(data_a[ib_k].scales[j + 8]);
        const uint sc_val = (sub < 4) ? (s_j  & 0x3Fu) : ((s_j8 & 0x0Fu) | ((s_j  >> 6) << 4));
        const uint mn_val = (sub < 4) ? (s_j4 & 0x3Fu) : ((s_j8 >> 4)    | ((s_j4 >> 6) << 4));
        vec2 dm = vec2(data_a_packed32[ib_k].dm);
        float d_scaled = dm.x * float(sc_val);
        buf_a_dm[ks * BM + buf_ib] = vec2(d_scaled, -(dm.y * float(mn_val)));
    }
}

#elif defined(DATA_A_Q6_K)

struct block_a_prefetch {
    uint32_t qs0;
    uint32_t qs1;
    uint ib;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint sub = ib % 8;
    const uint g = sub / 4;
    const uint j = sub % 4;

    const uint ql_u16 = g * 32 + (j & 1) * 16 + loadr * 4;
    const uint qh_u16 = g * 16 + loadr * 4;
    const uint qh_shift = j * 2;

    uint32_t ql0 = pack32(u16vec2(data_a_packed16[ib_k].ql[ql_u16    ],
                                   data_a_packed16[ib_k].ql[ql_u16 + 1]));
    uint32_t ql1 = pack32(u16vec2(data_a_packed16[ib_k].ql[ql_u16 + 2],
                                   data_a_packed16[ib_k].ql[ql_u16 + 3]));
    if (j >= 2) {
        ql0 = (ql0 >> 4) & 0x0F0F0F0F;
        ql1 = (ql1 >> 4) & 0x0F0F0F0F;
    } else {
        ql0 = ql0 & 0x0F0F0F0F;
        ql1 = ql1 & 0x0F0F0F0F;
    }

    uint32_t qh0 = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_u16    ],
                                   data_a_packed16[ib_k].qh[qh_u16 + 1]));
    uint32_t qh1 = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_u16 + 2],
                                   data_a_packed16[ib_k].qh[qh_u16 + 3]));

    blk.qs0 = ql0 | (((qh0 >> qh_shift) & 0x03030303) << 4);
    blk.qs1 = ql1 | (((qh1 >> qh_shift) & 0x03030303) << 4);
    blk.ib = ib;

    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    uint32_t v0 = ((blk.qs0 | 0x80808080) - 0x20202020) ^ 0x80808080;
    uint32_t v1 = ((blk.qs1 | 0x80808080) - 0x20202020) ^ 0x80808080;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = v0;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = v1;

    if (loadr == 0) {
        const uint ib_k = blk.ib / 8;
        const uint sub = blk.ib % 8;
        i8vec2 sc = unpack8(int32_t(int16_t(data_a_packed16[ib_k].scales[sub]))).xy;
        buf_a_d[(ks * KSCALES    ) * BM + buf_ib] = float(data_a_packed16[ib_k].d) * float(sc.x);
        buf_a_d[(ks * KSCALES + 1) * BM + buf_ib] = float(data_a_packed16[ib_k].d) * float(sc.y);
    }
}

#elif defined(DATA_A_Q3_K)

struct block_a_prefetch {
    uint32_t qs0;
    uint32_t qs1;
    uint ib;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint sub = ib % 8;
    const uint g = sub / 4;
    const uint j = sub % 4;
    const uint qs_shift = j * 2;
    const uint hm_bit = j + g * 4;

    const uint qs_u16 = g * 16 + loadr * 4;
    uint32_t qs0 = pack32(u16vec2(data_a_packed16[ib_k].qs[qs_u16    ],
                                   data_a_packed16[ib_k].qs[qs_u16 + 1]));
    uint32_t qs1 = pack32(u16vec2(data_a_packed16[ib_k].qs[qs_u16 + 2],
                                   data_a_packed16[ib_k].qs[qs_u16 + 3]));

    const uint hm_u16 = loadr * 4;
    uint32_t hm0 = pack32(u16vec2(data_a_packed16[ib_k].hmask[hm_u16    ],
                                   data_a_packed16[ib_k].hmask[hm_u16 + 1]));
    uint32_t hm1 = pack32(u16vec2(data_a_packed16[ib_k].hmask[hm_u16 + 2],
                                   data_a_packed16[ib_k].hmask[hm_u16 + 3]));

    blk.qs0 = ((qs0 >> qs_shift) & 0x03030303) | (((hm0 >> hm_bit) & 0x01010101) << 2);
    blk.qs1 = ((qs1 >> qs_shift) & 0x03030303) | (((hm1 >> hm_bit) & 0x01010101) << 2);
    blk.ib = ib;

    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    uint32_t v0 = ((blk.qs0 | 0x80808080) - 0x04040404) ^ 0x80808080;
    uint32_t v1 = ((blk.qs1 | 0x80808080) - 0x04040404) ^ 0x80808080;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = v0;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = v1;

    if (loadr == 0) {
        const uint ib_k = blk.ib / 8;
        const uint sub = blk.ib % 8;
        const uint is = sub * 2;
        uint lo = uint(data_a_packed16[ib_k].scales[(is % 8) / 2]);
        lo = (lo >> (4 * (is / 8))) & 0x0F0Fu;
        uint hi = uint(data_a_packed16[ib_k].scales[(8 + (is % 4)) / 2]);
        hi = (hi >> (2 * (is / 4))) & 0x0303u;
        uint combined = lo | (hi << 4);
        i8vec2 sc = unpack8(int32_t(combined)).xy;
        float d = float(data_a_packed16[ib_k].d);
        buf_a_d[(ks * KSCALES    ) * BM + buf_ib] = d * float(int(sc.x) - 32);
        buf_a_d[(ks * KSCALES + 1) * BM + buf_ib] = d * float(int(sc.y) - 32);
    }
}

#elif defined(DATA_A_NVFP4)

struct block_a_prefetch {
    uint32_t qs;
    uint8_t d0;
    uint8_t d1;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 2;
    const uint ihalf = ib % 2;
    const uint sub = ihalf * 2 + (loadr >> 1);
    const uint byte_group = loadr & 1u;

    blk.qs = pack32(u8vec4(data_a[ib_k].qs[sub * 8 + byte_group * 4],
                            data_a[ib_k].qs[sub * 8 + byte_group * 4 + 1],
                            data_a[ib_k].qs[sub * 8 + byte_group * 4 + 2],
                            data_a[ib_k].qs[sub * 8 + byte_group * 4 + 3]));
    blk.d0 = data_a[ib_k].d[ihalf * 2];
    blk.d1 = data_a[ib_k].d[ihalf * 2 + 1];

    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    const u8vec4 lo_idx = unpack8(blk.qs & 0x0F0F0F0F);
    const u8vec4 hi_idx = unpack8((blk.qs >> 4) & 0x0F0F0F0F);
    const uint sub_base = (loadr >> 1) * 4;
    const uint byte_group = loadr & 1u;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + sub_base + byte_group] =
        pack32(i8vec4(cm1_kvalues[lo_idx.x], cm1_kvalues[lo_idx.y],
                      cm1_kvalues[lo_idx.z], cm1_kvalues[lo_idx.w]));
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + sub_base + 2 + byte_group] =
        pack32(i8vec4(cm1_kvalues[hi_idx.x], cm1_kvalues[hi_idx.y],
                      cm1_kvalues[hi_idx.z], cm1_kvalues[hi_idx.w]));

    if (loadr == 0) {
        buf_a_d[(ks * KSCALES    ) * BM + buf_ib] = ue4m3_to_fp32(blk.d0) * 0.5;
        buf_a_d[(ks * KSCALES + 1) * BM + buf_ib] = ue4m3_to_fp32(blk.d1) * 0.5;
    }
}

// ===== Exact int8 views: the stored level, not the index, goes to the int8 WMMA =====
// Grid and ternary codes are small integers, so the int8 operand is exact and only the
// per-32 (or per-16) float scale is applied to the accumulator. LOAD_VEC_A is 8 for all of
// them: each thread writes eight consecutive levels of one 32-element block as two words.

#elif defined(DATA_A_IQ2_XXS)

struct block_a_prefetch {
    uint idx;
    uint aux;
    float16_t d;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint sub = ib % 8;
    blk.idx = uint(data_a[ib_k].qs[8 * sub + loadr]);
    blk.aux = pack32(u16vec2(data_a_packed16[ib_k].qs[4 * sub + 2], data_a_packed16[ib_k].qs[4 * sub + 3]));
    blk.d = data_a_packed16[ib_k].d;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    const uint s8 = cm1_iq_signs8(blk.aux >> (7 * loadr));
    const uvec2 g = iq2xxs_grid[blk.idx];
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = cm1_apply_signs4(g.x, s8);
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = cm1_apply_signs4(g.y, s8 >> 4);

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = float(blk.d) * 0.25 * (0.5 + float(blk.aux >> 28));
    }
}

#elif defined(DATA_A_IQ2_XS)

struct block_a_prefetch {
    uint q;
    uint sc;
    float16_t d;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint sub = ib % 8;
    blk.q = uint(data_a[ib_k].qs[4 * sub + loadr]);
    blk.sc = uint(data_a[ib_k].scales[sub]);
    blk.d = data_a[ib_k].d;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    const uint s8 = cm1_iq_signs8(blk.q >> 9);
    const uvec2 g = iq2xs_grid[blk.q & 511];
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = cm1_apply_signs4(g.x, s8);
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = cm1_apply_signs4(g.y, s8 >> 4);

    if (loadr == 0) {
        const float d = float(blk.d) * 0.25;
        buf_a_d[(ks * KSCALES    ) * BM + buf_ib] = d * (0.5 + float(blk.sc & 15));
        buf_a_d[(ks * KSCALES + 1) * BM + buf_ib] = d * (0.5 + float(blk.sc >> 4));
    }
}

#elif defined(DATA_A_IQ2_S)

struct block_a_prefetch {
    uint idx;
    uint s8;
    uint sc;
    float16_t d;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint sub = ib % 8;
    const uint qh = uint(data_a[ib_k].qh[sub]);
    blk.idx = uint(data_a[ib_k].qs[4 * sub + loadr]) | (((qh >> (2 * loadr)) & 3) << 8);
    blk.s8 = uint(data_a[ib_k].qs[QUANT_K_IQ2_S / 8 + 4 * sub + loadr]);
    blk.sc = uint(data_a[ib_k].scales[sub]);
    blk.d = data_a[ib_k].d;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    const uvec2 g = iq2s_grid[blk.idx];
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = cm1_apply_signs4(g.x, blk.s8);
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = cm1_apply_signs4(g.y, blk.s8 >> 4);

    if (loadr == 0) {
        const float d = float(blk.d) * 0.25;
        buf_a_d[(ks * KSCALES    ) * BM + buf_ib] = d * (0.5 + float(blk.sc & 15));
        buf_a_d[(ks * KSCALES + 1) * BM + buf_ib] = d * (0.5 + float(blk.sc >> 4));
    }
}

#elif defined(DATA_A_IQ3_XXS)

struct block_a_prefetch {
    uint idx;   // two grid indices in the low 16 bits
    uint aux;
    float16_t d;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint sub = ib % 8;
    blk.idx = uint(data_a_packed16[ib_k].qs[4 * sub + loadr]);
    blk.aux = pack32(u16vec2(data_a_packed16[ib_k].qs[QUANT_K_IQ3_XXS / 8 + 2 * sub],
                             data_a_packed16[ib_k].qs[QUANT_K_IQ3_XXS / 8 + 2 * sub + 1]));
    blk.d = data_a_packed16[ib_k].d;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    const uint s8 = cm1_iq_signs8(blk.aux >> (7 * loadr));
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = cm1_apply_signs4(iq3xxs_grid[blk.idx & 255], s8);
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = cm1_apply_signs4(iq3xxs_grid[blk.idx >> 8], s8 >> 4);

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = float(blk.d) * 0.5 * (0.5 + float(blk.aux >> 28));
    }
}

#elif defined(DATA_A_IQ3_S)

struct block_a_prefetch {
    uint idx;   // grid index pair, 9 bits each
    uint s8;
    uint sc;
    float16_t d;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint sub = ib % 8;
    const uint qh = uint(data_a[ib_k].qh[sub]) >> (2 * loadr);
    const uint q2 = uint(data_a_packed16[ib_k].qs[4 * sub + loadr]);
    blk.idx = ((q2 & 255) | ((qh & 1) << 8)) | (((q2 >> 8) | ((qh & 2) << 7)) << 16);
    blk.s8 = uint(data_a[ib_k].signs[4 * sub + loadr]);
    blk.sc = (uint(data_a[ib_k].scales[sub / 2]) >> (4 * (sub & 1))) & 15;
    blk.d = data_a[ib_k].d;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = cm1_apply_signs4(iq3s_grid[blk.idx & 0xFFFF], blk.s8);
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = cm1_apply_signs4(iq3s_grid[blk.idx >> 16], blk.s8 >> 4);

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = float(blk.d) * float(1 + 2 * blk.sc);
    }
}

#elif defined(DATA_A_TQ2_0)

struct block_a_prefetch {
    uint32_t qs0;
    uint32_t qs1;
    float16_t d;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint sub = ib % 8;
    // 32-element block sub = bit pair (sub % 4) of bytes (sub / 4) * 32 .. + 31
    const uint q16 = (sub / 4) * 16 + loadr * 4;
    const uint sh = 2 * (sub % 4);
    blk.qs0 = (pack32(u16vec2(data_a_packed16[ib_k].qs[q16    ], data_a_packed16[ib_k].qs[q16 + 1])) >> sh) & 0x03030303;
    blk.qs1 = (pack32(u16vec2(data_a_packed16[ib_k].qs[q16 + 2], data_a_packed16[ib_k].qs[q16 + 3])) >> sh) & 0x03030303;
    blk.d = data_a_packed16[ib_k].d;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = cm1_sub1_bytes(blk.qs0);
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = cm1_sub1_bytes(blk.qs1);

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = float(blk.d);
    }
}

#elif defined(DATA_A_Q2_0)

struct block_a_prefetch {
    uint q16;
    float16_t d;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    // 64-element blocks: the 32-element half (ib % 2) is bytes 8 * half .. + 7, two per thread
    blk.q16 = uint(data_a_packed16[ib / 2].qs[4 * (ib % 2) + loadr]);
    blk.d = data_a_packed16[ib / 2].d;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = cm1_sub1_bytes(cm1_spread2(blk.q16 & 255));
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = cm1_sub1_bytes(cm1_spread2(blk.q16 >> 8));

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = float(blk.d);
    }
}

#elif defined(DATA_A_PTQ1_0)

struct block_a_prefetch {
    uint32_t w0;
    uint32_t w1;
    uint n;     // trit position; 5 = the qh tail
    float16_t d;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 4;
    const uint e = 32 * (ib % 4) + 8 * loadr;
    // see ptq1_0.glsl: bytes 0..15 carry elements t*16+j, bytes 16..23 carry 80+t*8+j, qh the last 8
    if (e < 120) {
        const uint q = e < 80 ? (e & 15) : 16;
        blk.n = e < 80 ? (e >> 4) : ((e - 80) >> 3);
        blk.w0 = data_a_packed32[ib_k].qs[q >> 2];
        blk.w1 = data_a_packed32[ib_k].qs[(q >> 2) + 1];
    } else {
        blk.n = 5;
        blk.w0 = uint(data_a_packed32[ib_k].qh);
        blk.w1 = 0;
    }
    blk.d = data_a_packed32[ib_k].d;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    uint lo, hi;
    if (blk.n < 5) {
        const uint p = cm1_pow3(blk.n);
        lo = cm1_trits4(blk.w0, p);
        hi = cm1_trits4(blk.w1, p);
    } else {
        // qh tail: elements 120 + 2 * t + h come from byte h at trit t
        const uint b2 = blk.w0 & 0xFFFF;
        const uint pair = b2 | (b2 << 16);
        lo = (cm1_trits4(pair, 1) & 0x0000FFFF) | (cm1_trits4(pair, 3) & 0xFFFF0000);
        hi = (cm1_trits4(pair, 9) & 0x0000FFFF) | (cm1_trits4(pair, 27) & 0xFFFF0000);
    }
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = cm1_sub1_bytes(lo);
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = cm1_sub1_bytes(hi);

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = float(blk.d);
    }
}

#elif defined(DATA_A_TQ1_0)

struct block_a_prefetch {
    uint32_t w0;
    uint32_t w1;
    uint p01;   // trit multipliers 3^n for w0 (low 8 bits) and w1 (high 8 bits)
    float16_t d;
};

uint cm1_tq1_word(uint ib_k, uint byte0) {
    return pack32(u16vec2(data_a_tq1_16[ib_k].q[byte0 / 2], data_a_tq1_16[ib_k].q[byte0 / 2 + 1]));
}

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint sub = ib % 8;
    // elements 0..159: n * 32 + m from qs[m]; 160..239: 160 + n * 16 + m from qs[32 + m];
    // 240..255: 240 + n * 4 + m from qh[m] (qh starts at byte 48)
    if (sub < 5) {
        blk.w0 = cm1_tq1_word(ib_k, 8 * loadr);
        blk.w1 = cm1_tq1_word(ib_k, 8 * loadr + 4);
        blk.p01 = cm1_pow3(sub) * 0x101;
    } else if (sub < 7 || loadr < 2) {
        const uint n = 2 * (sub - 5) + (loadr >> 1);
        const uint m = 8 * (loadr & 1);
        blk.w0 = cm1_tq1_word(ib_k, 32 + m);
        blk.w1 = cm1_tq1_word(ib_k, 32 + m + 4);
        blk.p01 = cm1_pow3(n) * 0x101;
    } else {
        const uint n = 2 * (loadr - 2);
        blk.w0 = cm1_tq1_word(ib_k, 48);
        blk.w1 = blk.w0;
        blk.p01 = cm1_pow3(n) | (cm1_pow3(n + 1) << 8);
    }
    blk.d = data_a_tq1_16[ib_k].d;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = cm1_sub1_bytes(cm1_trits4(blk.w0, blk.p01 & 255));
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = cm1_sub1_bytes(cm1_trits4(blk.w1, blk.p01 >> 8));

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = float(blk.d);
    }
}

#endif

// ===== B-side: load and store =====

struct block_b_prefetch {
    ivec4 qs;
    float16_t d;
#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1) || defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
    float16_t s;
#endif
};

block_b_prefetch block_b_load(uint ib_outer, uint ib_inner, uint loadr) {
    block_b_prefetch blk;
    blk.qs = data_b[ib_outer].qs[ib_inner * 2 + loadr];
    blk.d = data_b[ib_outer].ds[ib_inner].x;
#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1) || defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
    blk.s = data_b[ib_outer].ds[ib_inner].y;
#endif
    return blk;
}

void block_b_to_shmem(block_b_prefetch blk, uint buf_ib, uint ks, uint loadr, bool in_bounds) {
    const ivec4 v = in_bounds ? blk.qs : ivec4(0);
    const uint base = buf_ib * QPITCH + ks * (BK / 4) + loadr * 4;
    buf_b_qs[base    ] = v.x;
    buf_b_qs[base + 1] = v.y;
    buf_b_qs[base + 2] = v.z;
    buf_b_qs[base + 3] = v.w;
    if (loadr == 0) {
        buf_b_d[ks * BN + buf_ib] = in_bounds ? float(blk.d) : 0.0f;
#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1) || defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
        buf_b_s[ks * BN + buf_ib] = in_bounds ? float(blk.s) : 0.0f;
#endif
    }
}

// ===== Framework macros =====

#ifdef MUL_MAT_ID
#define B_IB_CALC                                                                               \
            const u16vec2 row_idx = row_ids[buf_ib];                                            \
            const uint ib = pos_b_ib + row_idx.y * p.batch_stride_b / BK                        \
                          + (row_idx.x % p.ne11) * p.stride_b / BK;
#else
#define B_IB_CALC                                                                               \
            const uint ib = pos_b_ib + buf_ib * p.stride_b / BK;
#endif

#define PREFETCH_BLOCK(blk)                                                                     \
    [[unroll]] for (uint li = 0; li < A_LOADS; li++) {                                          \
        const uint buf_ib = loadc_a + li * loadstride_a;                                        \
        if (buf_ib < BM) {                                                                      \
            const uint ib = pos_a_ib + buf_ib * p.stride_a / BK;                                \
            [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {                                  \
                pre_a[li * BK_STEP + ks] = block_a_load(ib + ks, loadr_a);                      \
            }                                                                                   \
        }                                                                                       \
    }                                                                                           \
    [[unroll]] for (uint li = 0; li < B_LOADS; li++) {                                          \
        const uint buf_ib = loadc_b + li * loadstride_b;                                        \
        if (buf_ib < BN) {                                                                      \
            B_IB_CALC                                                                           \
            [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {                                  \
                const uint ib_k = ((blk) + ks * BK < end_k) ? (ib + ks) : ib;                   \
                pre_b[li * BK_STEP + ks] = block_b_load(ib_k / 4, ib_k % 4, loadr_b);          \
            }                                                                                   \
        }                                                                                       \
    }

#define STORE_BLOCK_TO_LDS(blk)                                                                 \
    [[unroll]] for (uint li = 0; li < A_LOADS; li++) {                                          \
        const uint buf_ib = loadc_a + li * loadstride_a;                                        \
        if (buf_ib < BM) {                                                                      \
            [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {                                  \
                block_a_to_shmem(pre_a[li * BK_STEP + ks], buf_ib, ks, loadr_a);                \
            }                                                                                   \
        }                                                                                       \
    }                                                                                           \
    [[unroll]] for (uint li = 0; li < B_LOADS; li++) {                                          \
        const uint buf_ib = loadc_b + li * loadstride_b;                                        \
        if (buf_ib < BN) {                                                                      \
            [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {                                  \
                const bool in_bounds = (blk) + ks * BK < end_k;                                 \
                block_b_to_shmem(pre_b[li * BK_STEP + ks], buf_ib, ks, loadr_b, in_bounds);     \
            }                                                                                   \
        }                                                                                       \
    }
