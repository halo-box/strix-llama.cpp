// GGML_OP_SPARSE_ATTN on HIP: kernel selection, workspace size, supports_op and dispatch (PLAN-v3 §2.2, §5, §6.2, §7).
//
// Tiers in this file: the G4 WMMA attend for S_fast^2 (sa_attend_g4, sparse-attn-attend-g4.cuh; instances in
// sparse-attn-attend-g4-<kt>.cu), selected at runtime only on RDNA3.5 (GGML_CUDA_CC_IS_RDNA3_5, the arch PR #219 gates
// every QSA kernel to, qsa-decode.cu:85), with rot_q = rot_o = 0. The G1 tier (W4), the generic and portable tiers (W5)
// and the in-op rotation (W10) are not built yet: supports_op is false for those shapes.
//
// One shape-only function, sa_select, decides the kernel instance and the V loader; supports_op, get_alloc_size, the
// dispatch and the census all use it (PLAN-v3 §5 "kernel instance, V pack vs direct, split count").
//   V loader: the per-type pack-limit trait of sparse-attn-loaders.cuh (default CAPPED for every type: packed iff
//   n_kv < 64 * n_q, n_q per stream; OWNER decision 2026-11-05).
//   Workspace (the get_alloc_size over-allocation of dst, placed at round_up(nbytes(dst), 256); D12: no pool), G4,
//   each part padded to 256 bytes:
//     packed V: [V: ceil(n_kv / 4) quads x 2 KiB x n_head_kv x n_stream][quad flags: 1 B per (stream, kv head, quad)]
//               [any-flag: 4 B]  (sa_pack_v: non-finite halves -> 0, flagged)
//     direct V: [wave flags: 1 B per (stream, group, kv head, wave) = n_stream x ceil(n_q / 4) x n_head_kv x 8 B]
//   every flag byte is written at each launch (the any-flag is cleared by a 4-byte memset before the pack), and
//   sa_attend_fixup reads them.
//   get_alloc_size takes the max over every G of the policy range ({1, 4}: G1 has no workspace until W4 adds its split
//   partials) and every n_q' <= n_q at this n_kv, and over every runtime n_kv' <= n_kv: under CAPPED a runtime n_kv'
//   packs only when n_kv' < 64 * n_q, so the pack bound is min(4 * ceil(n_kv / 4), 64 * n_q) cells, and the flags grow
//   with n_q (PLAN-v3 §5 "workspace size", §7).
// Test switch GGML_CUDA_SA_FORCE (comma-separated): vpacked | vdirect force the V loader (PLAN-v3 §6.2); it changes the
// kernel choice inside the backend only, never placement, and get_alloc_size follows it; the census skip proc names the
// keys a forced process cannot dispatch (ggml_cuda_sparse_attn_census_skip).

#include "sparse-attn.cuh"
#include "sparse-attn-census.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(GGML_USE_HIP)

#include "sparse-attn-attend-g4.cuh"

static constexpr size_t SA_WS_ALIGN = 256;   // the workspace starts at round_up(nbytes(dst), 256)

static int sa_type_id_of(const ggml_type t) {
    switch (t) {
        case GGML_TYPE_F16:  return SA_T_F16;
        case GGML_TYPE_BF16: return SA_T_BF16;
        case GGML_TYPE_Q8_0: return SA_T_Q8_0;
        case GGML_TYPE_Q4_0: return SA_T_Q4_0;
        default:             return -1;
    }
}

static int sa_gqa_index(const int64_t gqa) {
    return gqa == 4 ? 0 : gqa == 8 ? 1 : gqa == 12 ? 2 : -1;
}

enum sa_force_vl { SA_FORCE_VL_NONE = 0, SA_FORCE_VL_PACKED, SA_FORCE_VL_DIRECT };

static int sa_force_vl(void) {
    static const int v = [] {
        const char * e = getenv("GGML_CUDA_SA_FORCE");
        int r = SA_FORCE_VL_NONE;
        if (e == nullptr) {
            return r;
        }
        std::string s(e);
        size_t p = 0;
        while (p <= s.size()) {
            const size_t q = std::min(s.find(',', p), s.size());
            const std::string tok = s.substr(p, q - p);
            if (tok == "vpacked") {
                r = SA_FORCE_VL_PACKED;
            } else if (tok == "vdirect") {
                r = SA_FORCE_VL_DIRECT;
            } else if (!tok.empty()) {
                GGML_LOG_WARN("%s: GGML_CUDA_SA_FORCE: '%s' is not a forcing switch of this build (vpacked, vdirect)\n", __func__, tok.c_str());
            }
            p = q + 1;
        }
        return r;
    }();
    return v;
}

// V loader of the G4 attend at (V type, n_q per stream, n_kv): the pack-limit trait, or the forcing switch
static int sa_g4_vl(const int vt, const int64_t n_q, const int64_t n_kv) {
    switch (sa_force_vl()) {
        case SA_FORCE_VL_PACKED: return SA_V_PACKED;
        case SA_FORCE_VL_DIRECT: return SA_V_SPLITGATHER;
        default: break;
    }
    if (sa_vpack_trait(vt) == SA_VPACK_ALWAYS) {
        return SA_V_PACKED;
    }
    return n_kv < 64 * n_q ? SA_V_PACKED : SA_V_SPLITGATHER;
}

// largest number of V cells the G4 pack can write for any runtime op whose shape is bounded by (n_q, n_kv): packed
// when n_kv' < 64 * n_q' <= 64 * n_q (CAPPED), at most 4 * ceil(n_kv' / 4) cells
static int64_t sa_g4_pack_cells_max(const int vt, const int64_t n_q, const int64_t n_kv) {
    const int64_t n_kv4 = 4 * ((n_kv + 3) / 4);
    switch (sa_force_vl()) {
        case SA_FORCE_VL_PACKED: return n_kv4;
        case SA_FORCE_VL_DIRECT: return 0;
        default: break;
    }
    if (sa_vpack_trait(vt) == SA_VPACK_ALWAYS) {
        return n_kv4;
    }
    return std::min(n_kv4, 64 * n_q);
}

struct sa_selection {
    bool    ok;           // a kernel instance exists for this op on this device
    int     G;
    int     kt, vt, vl;
    int     gqa_index;
    int64_t n_kv, n_quads, n_q, n_stream, n_head, n_head_kv, n_sel;
    size_t  ws_offset;     // round_up(nbytes(dst), 256)
    size_t  qflags_offset; // packed: quad flags, from dst->data
    size_t  any_offset;    // packed: any-flag, from dst->data
    size_t  flags_offset;  // direct: wave flags, from dst->data
    size_t  ws_bytes;      // workspace this op uses from ws_offset
    size_t  ws_max;        // workspace get_alloc_size reserves (max over the policy range, see the header)
    const char * why;     // when !ok
};

static size_t sa_g4_flags_bytes(const int64_t n_q, const int64_t n_head_kv, const int64_t n_stream) {
    return (size_t) ((n_q + 3) / 4) * (size_t) n_head_kv * (size_t) n_stream * 8;
}

// packed-V workspace: V, quad flags, any-flag, each padded to 256 bytes
static size_t sa_g4_packed_ws_bytes(const int64_t n_quads, const int64_t n_head_kv, const int64_t n_stream) {
    const size_t nq = (size_t) n_quads * (size_t) n_head_kv * (size_t) n_stream;
    return GGML_PAD(nq * 2048, SA_WS_ALIGN) + GGML_PAD(nq, SA_WS_ALIGN) + SA_WS_ALIGN;
}

// shape-only selection (no pointers, no device state other than the arch)
static sa_selection sa_select(const ggml_tensor * op, const int device) {
    sa_selection r = {};
    r.ok = false;
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2], * plan = op->src[3];
    if (q == nullptr || k == nullptr || v == nullptr || plan == nullptr || plan->op != GGML_OP_SPARSE_ATTN_PLAN || plan->src[0] == nullptr) {
        r.why = "malformed SPARSE_ATTN node";
        return r;
    }
    r.G         = ggml_get_op_params_i32(plan, 1);
    r.n_kv      = ggml_get_op_params_i32(plan, 0);
    r.n_quads   = (r.n_kv + 3) / 4;
    r.n_q       = q->ne[1];
    r.n_head    = q->ne[2];
    r.n_stream  = q->ne[3];
    r.n_head_kv = k->ne[2];
    r.n_sel     = plan->src[0]->ne[0];
    r.kt        = sa_type_id_of(k->type);
    r.vt        = sa_type_id_of(v->type);
    r.gqa_index = r.n_head_kv > 0 && r.n_head % r.n_head_kv == 0 ? sa_gqa_index(r.n_head / r.n_head_kv) : -1;
    r.ws_offset = GGML_PAD(ggml_nbytes(op), SA_WS_ALIGN);

    // workspace bound over the policy range (G in {1, 4}; G1 has none until W4) at this n_kv: the larger of the packed
    // layout at the largest pack any runtime op can need and the direct layout at this n_q
    if (r.vt >= 0) {
        const int64_t quads_max = sa_g4_pack_cells_max(r.vt, r.n_q, r.n_kv) / 4;
        const size_t  packed = quads_max > 0 ? sa_g4_packed_ws_bytes(quads_max, r.n_head_kv, r.n_stream) : 0;
        r.ws_max = std::max(packed, sa_g4_flags_bytes(r.n_q, r.n_head_kv, r.n_stream));
    }

    if (!GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[device].cc)) {
        r.why = "no SPARSE_ATTN tier for this HIP arch yet (the WMMA tiers run on RDNA3.5 only; portable tier: W5)";
        return r;
    }
    if (r.kt < 0 || r.vt < 0) {
        r.why = "K/V type outside S_fast (generic tier: W5)";
        return r;
    }
    if (r.gqa_index < 0) {
        r.why = "n_head / n_head_kv not in {4, 8, 12}";
        return r;
    }
    if (q->ne[0] != 256 || k->ne[0] != 256 || v->ne[0] != 256) {
        r.why = "head size other than 256";
        return r;
    }
    if (ggml_get_op_params_i32(op, 1) != 0 || ggml_get_op_params_i32(op, 2) != 0) {
        r.why = "rot_q / rot_o != 0 (in-op rotation: W10)";
        return r;
    }
    if (r.G != 4) {
        r.why = "G = 1 plan (G1 tier: W4)";
        return r;
    }
    r.vl = sa_g4_vl(r.vt, r.n_q, r.n_kv);
    if (r.vl == SA_V_PACKED) {
        const size_t nq = (size_t) r.n_quads * (size_t) r.n_head_kv * (size_t) r.n_stream;
        r.qflags_offset = r.ws_offset + GGML_PAD(nq * 2048, SA_WS_ALIGN);
        r.any_offset    = r.qflags_offset + GGML_PAD(nq, SA_WS_ALIGN);
        r.ws_bytes      = sa_g4_packed_ws_bytes(r.n_quads, r.n_head_kv, r.n_stream);
    } else {
        r.flags_offset  = r.ws_offset;
        r.ws_bytes      = sa_g4_flags_bytes(r.n_q, r.n_head_kv, r.n_stream);
    }
    GGML_ASSERT(r.ws_bytes <= r.ws_max);
    r.ok = true;
    return r;
}

static bool sa_strides_ok(const ggml_tensor * t) {
    return t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0;
}

bool ggml_cuda_sparse_attn_supported(int device, const ggml_tensor * op) {
    const sa_selection sel = sa_select(op, device);
    if (!sel.ok) {
        return false;
    }
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    // strides (PLAN-v3 §2.2 "Alignment"); base pointers are checked at compute
    if (q->type != GGML_TYPE_F32 || q->nb[0] != sizeof(float) || !sa_strides_ok(q) || !sa_strides_ok(k) || !sa_strides_ok(v)) {
        return false;
    }
    if (k->nb[0] != ggml_type_size(k->type) || v->nb[0] != ggml_type_size(v->type)) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
        return false;
    }
    // grid (n_groups, n_head_kv, n_stream) and int indexing of queries / groups / plan entries
    const int64_t n_groups = (sel.n_q + 3) / 4;
    if (sel.n_head_kv > 65535 || sel.n_stream > 65535 || n_groups > INT_MAX || sel.n_q > INT_MAX / 4 || sel.n_head > INT_MAX) {
        return false;
    }
    const ggml_sparse_attn_plan_layout L = ggml_sparse_attn_plan_get_layout(sel.n_sel, sel.n_q, sel.n_stream, sel.n_kv, sel.G);
    return L.cap <= INT_MAX;
}

size_t ggml_cuda_sparse_attn_get_alloc_size(int device, const ggml_tensor * op) {
    const sa_selection sel = sa_select(op, device);
    return sel.ws_max > 0 ? sel.ws_offset + sel.ws_max : ggml_nbytes(op);
}

const char * ggml_cuda_sparse_attn_census_skip(const char * key) {
    const std::string k(key);
    switch (sa_force_vl()) {
        case SA_FORCE_VL_PACKED:
            if (k.rfind("sa_attend/G4/", 0) == 0 && k.find("/direct/") != std::string::npos) {
                return "GGML_CUDA_SA_FORCE=vpacked: the direct-V G4 instances are not dispatched in this process";
            }
            break;
        case SA_FORCE_VL_DIRECT:
            if ((k.rfind("sa_attend/G4/", 0) == 0 && k.find("/packed/") != std::string::npos) || k.rfind("sa_pack/", 0) == 0) {
                return "GGML_CUDA_SA_FORCE=vdirect: the packed-V G4 instances and the V packs are not dispatched in this process";
            }
            break;
        default:
            break;
    }
    return nullptr;
}

bool ggml_cuda_sparse_attn_gate_ok(int device, const ggml_tensor * op) {
    const sa_selection sel = sa_select(op, device);
    return sel.ok && sel.G == 4;   // EPI = gated exists for the G4 instances
}

void ggml_cuda_sparse_attn_ws_range(int device, const ggml_tensor * op, size_t * offset, size_t * bytes) {
    const sa_selection sel = sa_select(op, device);
    *offset = sel.ws_offset;
    *bytes  = sel.ok ? sel.ws_bytes : 0;
}

static const sa_g4_kernel_t * sa_g4_table(const int kt) {
    switch (kt) {
        case SA_T_F16:  return ggml_cuda_sa_g4_kernels_f16;
        case SA_T_BF16: return ggml_cuda_sa_g4_kernels_bf16;
        case SA_T_Q8_0: return ggml_cuda_sa_g4_kernels_q8_0;
        case SA_T_Q4_0: return ggml_cuda_sa_g4_kernels_q4_0;
        default:        GGML_ABORT("SPARSE_ATTN: no G4 instance for K type id %d", kt);
    }
}

static const sa_fixup_kernel_t * sa_fixup_table(const int kt) {
    switch (kt) {
        case SA_T_F16:  return ggml_cuda_sa_fixup_kernels_f16;
        case SA_T_BF16: return ggml_cuda_sa_fixup_kernels_bf16;
        case SA_T_Q8_0: return ggml_cuda_sa_fixup_kernels_q8_0;
        case SA_T_Q4_0: return ggml_cuda_sa_fixup_kernels_q4_0;
        default:        GGML_ABORT("SPARSE_ATTN: no fixup instance for K type id %d", kt);
    }
}

// plain: out32 = dst->data, gate = nullptr; gated (gate fusion): out32 = the MUL output or nullptr, o16 its 16-bit copy
static void sa_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int epi, const ggml_tensor * gate,
                   float * out32, uint16_t * o16, const bool o16_f16) {
    const sa_selection sel = sa_select(dst, ctx.device);
    if (!sel.ok) {
        GGML_ABORT("SPARSE_ATTN: no HIP kernel for this node: %s", sel.why);
    }
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * plan = dst->src[3];
    const ggml_sparse_attn_plan_layout L = ggml_sparse_attn_plan_get_layout(sel.n_sel, sel.n_q, sel.n_stream, sel.n_kv, sel.G);
    GGML_ASSERT(ggml_nbytes(plan) == L.nbytes);
    // compute aborts only on a misaligned base pointer (PLAN-v3 §2.2 "Alignment")
    if ((uintptr_t) q->data % 16 != 0 || (uintptr_t) k->data % 16 != 0 || (uintptr_t) v->data % 16 != 0 ||
        (uintptr_t) plan->data % 16 != 0 || (uintptr_t) dst->data % 16 != 0) {
        GGML_ABORT("SPARSE_ATTN: q / k / v / plan / dst base pointer not 16-byte aligned");
    }
    GGML_ASSERT(dst->view_src == nullptr && dst->buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_get_alloc_size(dst->buffer, dst) >= sel.ws_offset + sel.ws_bytes &&
                "SPARSE_ATTN dst is not over-allocated by the HIP buffer type (get_alloc_size)");
    cudaStream_t stream = ctx.stream();

    sa_g4_args a = {};
    a.q = (const char *) q->data; a.q_nb1 = q->nb[1]; a.q_nb2 = q->nb[2]; a.q_nb3 = q->nb[3];
    a.k = (const char *) k->data; a.k_nb1 = k->nb[1]; a.k_nb2 = k->nb[2]; a.k_nb3 = k->nb[3];
    a.v = (const char *) v->data; a.v_nb1 = v->nb[1]; a.v_nb2 = v->nb[2]; a.v_nb3 = v->nb[3];
    if (sel.vl == SA_V_PACKED) {
        uint16_t * pv     = (uint16_t *) ((char *) dst->data + sel.ws_offset);
        uint8_t *  qflags = (uint8_t *) dst->data + sel.qflags_offset;
        uint32_t * any    = (uint32_t *) ((char *) dst->data + sel.any_offset);
        ggml_cuda_sparse_attn_pack_v(stream, v, sel.n_quads, pv, qflags, any);
        a.pv        = pv;
        a.pv_nquads = (size_t) sel.n_quads;
        a.qflags    = qflags;
        a.qany      = any;
    } else {
        a.flags = (uint8_t *) dst->data + sel.flags_offset;
    }
    const uint8_t * pd = (const uint8_t *) plan->data;
    a.counts    = (const uint32_t *) (pd + L.off_counts);
    a.quads     = (const uint32_t *) (pd + L.off_quads);
    a.masks     = (const uint16_t *) (pd + L.off_masks);
    a.n_groups  = (int) L.n_groups;
    a.cap       = (int) L.cap;
    a.n_q       = (int) sel.n_q;
    a.n_head    = (int) sel.n_head;
    a.n_head_kv = (int) sel.n_head_kv;
    a.scale     = ggml_get_op_params_f32(dst, 0);
    a.out       = out32;
    if (epi == SA_EPI_GATED) {
        a.gate     = (const char *) gate->data;
        a.gate_nb1 = gate->nb[1];
        a.o16      = o16;
        a.o16_f16  = o16_f16 ? 1 : 0;
    }

    const int idx = ((sel.vt * SA_V_COUNT + sel.vl) * SA_EPI_COUNT + epi) * 3 + sel.gqa_index;
    GGML_ASSERT(idx >= 0 && idx < GGML_CUDA_SA_G4_PER_KT);
    const sa_g4_kernel_t kernel = sa_g4_table(sel.kt)[idx];
    const dim3 grid((unsigned) L.n_groups, (unsigned) sel.n_head_kv, (unsigned) sel.n_stream);
    kernel<<<grid, 256, 0, stream>>>(a);
    ggml_cuda_sa_census_add((ggml_cuda_sa_census_id) (GGML_CUDA_SA_CENSUS_G4_FIRST + sel.kt * GGML_CUDA_SA_G4_PER_KT + idx));

    // exact recomputation of the blocks that took a non-finite V half out of the shared PV WMMA (returns at once for
    // the others; sparse-attn-attend-g4.cuh)
    const int fidx = sel.vt * SA_EPI_COUNT + epi;
    const sa_fixup_kernel_t fixup = sa_fixup_table(sel.kt)[fidx];
    fixup<<<grid, 256, 0, stream>>>(a, (int) (sel.n_head / sel.n_head_kv));
    ggml_cuda_sa_census_add((ggml_cuda_sa_census_id) (GGML_CUDA_SA_CENSUS_FIXUP_FIRST + sel.kt * GGML_CUDA_SA_FIXUP_PER_KT + fidx));
}

void ggml_cuda_sparse_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    sa_run(ctx, dst, SA_EPI_PLAIN, nullptr, (float *) dst->data, nullptr, false);
}

void ggml_cuda_sparse_attn_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * gate,
                                 float * out32, uint16_t * dst16, bool dst16_f16) {
    GGML_ASSERT(gate->type == GGML_TYPE_F32 && gate->nb[0] == sizeof(float) && gate->nb[1] % 16 == 0 &&
                (uintptr_t) gate->data % 16 == 0 && (out32 != nullptr || dst16 != nullptr));
    sa_run(ctx, dst, SA_EPI_GATED, gate, out32, dst16, dst16_f16);
}

#else // !defined(GGML_USE_HIP)

// CUDA builds: no sparse-attention kernels (PLAN-v3 §6.2, D22); supports_op is false, so these are never dispatched
bool ggml_cuda_sparse_attn_supported(int device, const ggml_tensor * op) {
    GGML_UNUSED(device);
    GGML_UNUSED(op);
    return false;
}

size_t ggml_cuda_sparse_attn_get_alloc_size(int device, const ggml_tensor * op) {
    GGML_UNUSED(device);
    return ggml_nbytes(op);
}

const char * ggml_cuda_sparse_attn_census_skip(const char * key) {
    GGML_UNUSED(key);
    return nullptr;
}

bool ggml_cuda_sparse_attn_gate_ok(int device, const ggml_tensor * op) {
    GGML_UNUSED(device);
    GGML_UNUSED(op);
    return false;
}

void ggml_cuda_sparse_attn_ws_range(int device, const ggml_tensor * op, size_t * offset, size_t * bytes) {
    GGML_UNUSED(device);
    *offset = ggml_nbytes(op);
    *bytes  = 0;
}

void ggml_cuda_sparse_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("SPARSE_ATTN is not supported by this CUDA build");
}

void ggml_cuda_sparse_attn_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * gate,
                                 float * out32, uint16_t * dst16, bool dst16_f16) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_UNUSED(gate);
    GGML_UNUSED(out32);
    GGML_UNUSED(dst16);
    GGML_UNUSED(dst16_f16);
    GGML_ABORT("SPARSE_ATTN is not supported by this CUDA build");
}

#endif // defined(GGML_USE_HIP)
