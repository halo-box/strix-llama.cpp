// Body of dequant_tqp5/6/7.comp (DATA_A_TQP5 / 6 / 7 select K).
#include "dequant_head.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_b[];};

// One invocation per two trellis steps (4 weights): 32 invocations per 128-weight block,
// 8 blocks (1024 weights) per workgroup.
void main() {
    init_iq_shmem(gl_WorkGroupSize);

    const uint item = gl_WorkGroupID.x * 256u + gl_LocalInvocationID.x;
    const uint ib = item / 32u;
    const uint t0 = (item % 32u) * 2u;
    if (ib >= p.nel / 128u) {
        return;
    }

    const float d = float(data_a[ib].d);
    vec2 v[2];
    [[unroll]] for (uint h = 0; h < 2u; ++h) {
        const uint off = tqp_off(t0 + h);
        const uint i0 = off >> 3u;
        const uint s = tqp_state_bytes(uint(data_a[ib].qs[i0]),
                                       uint(data_a[ib].qs[tqp_byte_wrap(i0 + 1u)]),
                                       uint(data_a[ib].qs[tqp_byte_wrap(i0 + 2u)]), off);
        v[h] = tqp_pair(s) * d;
    }

    const uint b_idx = 128u*ib + 2u*t0;
    data_b[b_idx + 0u] = D_TYPE(v[0].x);
    data_b[b_idx + 1u] = D_TYPE(v[0].y);
    data_b[b_idx + 2u] = D_TYPE(v[1].x);
    data_b[b_idx + 3u] = D_TYPE(v[1].y);
}
