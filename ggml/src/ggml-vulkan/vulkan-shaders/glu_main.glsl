// f16 builds (D_F16): the store equals the f32 op stored, then converted (and the MUL_MAT_ID SwiGLU f16 epilogue). Written
// plainly, the compiler may fold the last f32 multiply and the conversion into one single-rounding mixed-precision op; the
// value goes through an XOR with a push-constant zero first (mode is 0, 1 or 2).
#ifdef D_F16
float16_t store_d16(const float x) { return float16_t(uintBitsToFloat(floatBitsToUint(x) ^ (p.mode >> 2))); }
#define STORE_D(x) store_d16(x)
#else
#define STORE_D(x) D_TYPE(x)
#endif

void main() {
    const uint i = gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;

    if (i >= p.N) {
        return;
    }

    const uint i23 = fastdiv(i, p.ne2_012mp, p.ne2_012L);
    const uint i23_offset = i23 * p.ne22*p.ne21*p.ne20;
    const uint i22 = fastdiv(i - i23_offset, p.ne2_01mp, p.ne2_01L);
    const uint i22_offset = i22*p.ne21*p.ne20;
    const uint i21 = fastdiv(i - i23_offset - i22_offset, p.ne2_0mp, p.ne2_0L);
    const uint i20 = i - i23_offset - i22_offset - i21*p.ne20;

    const uint src_idx_a = get_aoffset() + i23 * p.nb03 + i22 * p.nb02 + i21 * p.nb01 + i20 * p.nb00;
    const uint src_idx_b = get_boffset() + i23 * p.nb13 + i22 * p.nb12 + i21 * p.nb11 + i20 * p.nb10;
    const uint dst_idx = get_doffset() + i23 * p.nb23 + i22 * p.nb22 + i21 * p.nb21 + i20 * p.nb20;

    if (p.mode == 0) {
        // Default
        const uint offset = (p.ne00 / 2) * p.nb00;
        const uint idx = src_idx_a;

        data_d[dst_idx] = STORE_D(op(float(data_a[idx]), float(data_a[idx + offset])));
    } else if (p.mode == 1) {
        // Swapped
        const uint offset = (p.ne00 / 2) * p.nb00;
        const uint idx = src_idx_a;

        data_d[dst_idx] = STORE_D(op(float(data_a[idx + offset]), float(data_a[idx])));
    } else {
        // Split
        data_d[dst_idx] = STORE_D(op(float(data_a[src_idx_a]), float(data_b[src_idx_b])));
    }
}
