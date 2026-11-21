// Body of dequant_tqk6.comp / dequant_tqk7.comp (DATA_A_TQK6 / DATA_A_TQK7 select K).
#include "dequant_head.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_b[];};

// One invocation per trellis step (4 weights): 32 invocations per 128-weight block,
// 8 blocks (1024 weights) per workgroup.
void main() {
    init_iq_shmem(gl_WorkGroupSize);

    const uint step = gl_WorkGroupID.x * 256u + gl_LocalInvocationID.x;
    const uint ib = step / 32u;
    const uint t  = step % 32u;
    if (ib >= p.nel / 128u) {
        return;
    }

    const float d = float(data_a[ib].d);
    const uint off = tqk_off(t);
    const uint i0 = off >> 3u;
    const uint s = tqk_state_bytes(uint(data_a[ib].qs[i0]),
                                   uint(data_a[ib].qs[tqk_byte_wrap(i0 + 1u)]),
                                   uint(data_a[ib].qs[tqk_byte_wrap(i0 + 2u)]), off);
    const vec4 v = tq2_t_step(s) * d;

    const uint b_idx = 128u*ib + 4u*t;
    data_b[b_idx + 0u] = D_TYPE(v.x);
    data_b[b_idx + 1u] = D_TYPE(v.y);
    data_b[b_idx + 2u] = D_TYPE(v.z);
    data_b[b_idx + 3u] = D_TYPE(v.w);
}
