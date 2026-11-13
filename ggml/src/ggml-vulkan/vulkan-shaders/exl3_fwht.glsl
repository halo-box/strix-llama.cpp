// needs GL_KHR_shader_subgroup_shuffle
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
