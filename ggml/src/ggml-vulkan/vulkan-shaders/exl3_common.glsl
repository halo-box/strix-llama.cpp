// EXL3 (exllamav3) trellis decode for the Vulkan EXL3 ops, see ggml/exl3/README.md.
//
// A tile is 16*K int16 (8*K uint32 words, read as little-endian pairs); ring bit p is bit 31 - p%32 of word p/32.
// Position i adds D(i) = K bits (half rates: floor(K) plus bit i%16 of 0xAAAA), its 16-bit state is the ring bits
// [S(i) - 16, S(i)), MSB first, wrapping. Position i = 8t + j is tile element (r, c), r = (t%4)*2 + (j&1) + 8*((j>>1)&1),
// c = t/4 + 8*((j>>2)&1). Lane l of a wave32 decodes positions 8l .. 8l + 7: rows r0, r0 + 1, r0 + 8, r0 + 9
// (r0 = (l%4)*2) of the columns l/4 (states 0..3) and l/4 + 8 (states 4..7), as the HIP dot kernel (exl3.cu).
// Every value is the exact fp16 codebook value of ggml_exl3_codebook_value().
//
// Needs: binding data_t (uint words of the trellis), spec constants 1..3 below.

layout (constant_id = 1) const uint KA   = 4; // K, or floor(K) of a half rate
layout (constant_id = 2) const uint HALF = 0; // 1: K + 0.5 (mul1 only)
layout (constant_id = 3) const uint CB   = 1; // 0 3inst, 1 mcg, 2 mul1

const uint EXL3_TILE_U32  = 8*KA + 4*HALF;      // uint32 words of a tile
const uint EXL3_TILE_BITS = 32*EXL3_TILE_U32;
// the 8 states of a lane span 7*KA (+4) + 16 bits from the first one's start
const uint EXL3_SPAN = 7*KA + 4*HALF + 16;
const uint EXL3_NAL  = (EXL3_SPAN + 31) / 32;  // words of the aligned window
const uint EXL3_NWL  = EXL3_NAL + 1;           // words loaded (any bit alignment)

// the 16-bit states of positions 8*lane .. 8*lane + 7 of the tile at word `base` of data_t
void exl3_lane_states(const uint base, const uint lane, out uint st[8]) {
    // start of the state of position 8*lane: S(8*lane) - 16 = 8*lane*KA + KA + 4*lane*HALF - 16, wrapping
    int b = int(8*lane*KA + KA + 4*lane*HALF) - 16;
    if (b < 0) {
        b += int(EXL3_TILE_BITS);
    }
    const uint w0 = uint(b) >> 5;
    const uint sh = uint(b) & 31;

    uint w[5];
    [[unroll]] for (uint i = 0; i < 5; ++i) {
        if (i < EXL3_NWL) {
            uint wi = w0 + i;
            if (wi >= EXL3_TILE_U32) {
                wi -= EXL3_TILE_U32;
            }
            w[i] = data_t[base + wi];
        }
    }
    // a[i]: 32 ring bits from the lane's first state bit on, MSB first
    uint a[4];
    [[unroll]] for (uint i = 0; i < 4; ++i) {
        if (i < EXL3_NAL) {
            a[i] = (w[i] << sh) | ((w[i + 1] >> 1) >> (31 - sh));
        }
    }
    [[unroll]] for (uint j = 0; j < 8; ++j) {
        // o: offset of state j from the lane's first state bit
        const uint o = j*KA + HALF*((j + 1)/2);
        const uint q = o >> 5;
        const uint s = o & 31;
        uint v;
        if (s <= 16) {
            v = a[q] >> (16 - s);
        } else {
            v = (a[q] << (s - 16)) | (a[q + 1] >> (48 - s));
        }
        st[j] = v & 0xffffu;
    }
}

// the codebook values of two states, each with the one fp16 rounding of ggml_exl3_codebook_value()
f16vec2 exl3_cb2(const uint s0, const uint s1) {
    if (CB == 2) {
        // mul1: fma(1024 + bytesum(s*0x83DCD12D), half(0x1EEE), half(0xC931)), one rounding (exact in f32 before it)
        const uint x0 = s0*0x83DCD12Du;
        const uint x1 = s1*0x83DCD12Du;
        uint t0 = (x0 & 0x00ff00ffu) + ((x0 >> 8) & 0x00ff00ffu);
        uint t1 = (x1 & 0x00ff00ffu) + ((x1 >> 8) & 0x00ff00ffu);
        t0 = (t0 + (t0 >> 16)) & 0xffffu;
        t1 = (t1 + (t1 >> 16)) & 0xffffu;
        // fp16 bits 0x6400 + bsum = 1024 + bsum (exact)
        const f16vec2 h = unpackFloat2x16((t0 | (t1 << 16)) + 0x64006400u);
        const f16vec2 k = unpackFloat2x16(0xc9311eeeu);
        precise f16vec2 r = fma(h, f16vec2(k.x), f16vec2(k.y));
        return r;
    } else {
        uint x0;
        uint x1;
        if (CB == 0) {
            x0 = s0*89226354u + 64248484u;
            x1 = s1*89226354u + 64248484u;
        } else {
            x0 = s0*0xCBAC1FEDu;
            x1 = s1*0xCBAC1FEDu;
        }
        // two fp16 halves in [2^-3, 2): their f16 sum is the exact sum rounded once
        const f16vec2 h0 = unpackFloat2x16((x0 & 0x8fff8fffu) ^ 0x3b603b60u);
        const f16vec2 h1 = unpackFloat2x16((x1 & 0x8fff8fffu) ^ 0x3b603b60u);
        precise f16vec2 r = f16vec2(h0.x, h1.x) + f16vec2(h0.y, h1.y);
        return r;
    }
}

// the 8 values of a lane: f[0] rows (r0, r0 + 1), f[1] rows (r0 + 8, r0 + 9) of column l/4; f[2], f[3] of column l/4 + 8
void exl3_lane_decode(const uint base, const uint lane, out f16vec2 f[4]) {
    uint st[8];
    exl3_lane_states(base, lane, st);
    [[unroll]] for (uint i = 0; i < 4; ++i) {
        f[i] = exl3_cb2(st[2*i], st[2*i + 1]);
    }
}

#ifdef EXL3_FWHT
// unnormalized 128-point Hadamard of v[i] = element i*32 + lane of one wave32: the stages h = 1, 2, ..., 64 in the
// order and with the roundings of fwht128 in ggml-cpu/exl3-ops.cpp
void exl3_fwht128(inout float v[4], const uint lane) {
    [[unroll]] for (uint h = 1; h < 32; h <<= 1) {
        [[unroll]] for (uint i = 0; i < 4; ++i) {
            const float o = subgroupShuffleXor(v[i], h);
            v[i] = (lane & h) != 0 ? o - v[i] : v[i] + o;
        }
    }
    const float a0 = v[0] + v[1];
    const float a1 = v[0] - v[1];
    const float a2 = v[2] + v[3];
    const float a3 = v[2] - v[3];
    v[0] = a0 + a2;
    v[1] = a1 + a3;
    v[2] = a0 - a2;
    v[3] = a1 - a3;
}
#endif
