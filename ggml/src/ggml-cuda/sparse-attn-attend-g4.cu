// GGML_OP_SPARSE_ATTN on HIP: kernel selection, workspace size, supports_op and dispatch (PLAN-v3 §2.2, §5, §6.2, §7).
//
// Tiers (sa_select, PLAN-v3 §5 "HIP tier selection per arch", §6.1, §6.2):
//   fast      the G4 WMMA attend for S_fast^2 (sa_attend_g4, sparse-attn-attend-g4.cuh; instances in
//             sparse-attn-attend-g4-<kt>.cu) and the G1 WMMA attend for S_fast^2 (sa_attend_g1, sparse-attn-attend-g1.cuh;
//             instances and launcher in sparse-attn-attend-g1.cu, split merge in sparse-attn-merge.cu);
//   generic   the same two WMMA attends with the runtime-switch loader (SA_T_GEN, sparse-attn-generic.cu) for every pair
//             of S with a member outside S_fast (f32, q4_1, q5_0, q5_1, iq4_nl);
//   portable  the scalar attend of sparse-attn-portable.cu (any type of S, any plan G, no WMMA / cross-lane operation).
// The WMMA tiers are selected at runtime only on RDNA3.5 (GGML_CUDA_CC_IS_RDNA3_5, the arch PR #219 gates every QSA
// kernel to, qsa-decode.cu:85); every other HIP arch runs the portable tier (its speed there is UNPROVEN, U15). All tiers
// take rot_q = rot_o = 0 until the in-op rotation (W10): supports_op is false for rot != 0.
//
// One shape-only function, sa_select, decides the tier, the kernel instance and the V loader; supports_op, get_alloc_size,
// the dispatch and the census all use it (PLAN-v3 §5 "kernel instance, V pack vs direct, split count").
//   V loader (G4, fast and generic): the per-type pack-limit trait of sparse-attn-loaders.cuh (default CAPPED for every
//   type, S_generic included: packed iff n_kv < 64 * n_q, n_q per stream; OWNER decision 2026-11-05).
//   Workspace (the get_alloc_size over-allocation of dst, placed at round_up(nbytes(dst), 256); D12: no pool), G4 (fast and
//   generic: the generic pack writes the same f16 layout), each part padded to 256 bytes:
//     packed V: [V: ceil(n_kv / 4) quads x 2 KiB x n_head_kv x n_stream][quad flags: 1 B per (stream, kv head, quad)]
//               [any-flag: 4 B]  (sa_pack_v: non-finite halves -> 0, flagged)
//     direct V: [wave flags: 1 B per (stream, group, kv head, wave) = n_stream x ceil(n_q / 4) x n_head_kv x 8 B]
//   every flag byte is written at each launch (the any-flag is cleared by a 4-byte memset before the pack), and
//   sa_attend_fixup reads them.
//   Workspace, G1 (fast and generic): the split-K partials [n_stream][n_q][n_head][splits][258] f32 when the shape-only
//   split count (sa_g1_splits) is > 1, none with one split (the attend writes dst itself).
//   Portable: no workspace.
//   get_alloc_size takes the max over every G of the policy range ({1, 4}) and every n_q' <= n_q at this n_kv, and over
//   every runtime n_kv' <= n_kv: under CAPPED a runtime n_kv' packs only when n_kv' < 64 * n_q, so the pack bound is
//   min(4 * ceil(n_kv / 4), 64 * n_q) cells, the flags grow with n_q, and the G1 partials are maximised over n_q' by
//   sa_g1_partial_bytes_max (they are not monotone in n_q, R1-C) (PLAN-v3 §5 "workspace size", §7). On a device whose
//   tier is the portable one (another arch, or GGML_CUDA_SA_FORCE=portable) there is no workspace.
// Test switch GGML_CUDA_SA_FORCE (comma-separated, PLAN-v3 §6.2): vpacked | vdirect force the G4 V loader; generic forces
// the generic tier for every pair of S; portable forces the portable tier; g1 | g4 force the group policy
// (sparse-attn-attend-g1.cu). It changes the kernel choice inside the backend only, never placement, and get_alloc_size
// follows it; the census skip proc names the keys a forced process (or another arch) cannot dispatch
// (ggml_cuda_sparse_attn_census_skip).

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

// the types of S (PLAN-v3 §6.1): S_fast and S_generic, the cases of sa_any_elem (sparse-attn-loaders.cuh)
static bool sa_type_in_S(const ggml_type t) {
    switch (t) {
        case GGML_TYPE_F16: case GGML_TYPE_BF16: case GGML_TYPE_Q8_0: case GGML_TYPE_Q4_0:
        case GGML_TYPE_F32: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_IQ4_NL:
            return true;
        default:
            return false;
    }
}

static int sa_gqa_index(const int64_t gqa) {
    return gqa == 4 ? 0 : gqa == 8 ? 1 : gqa == 12 ? 2 : -1;
}

enum sa_force_vl   { SA_FORCE_VL_NONE = 0, SA_FORCE_VL_PACKED, SA_FORCE_VL_DIRECT };
enum sa_force_tier { SA_FORCE_TIER_NONE = 0, SA_FORCE_TIER_GENERIC, SA_FORCE_TIER_PORTABLE };
enum sa_tier       { SA_TIER_FAST = 0, SA_TIER_GENERIC, SA_TIER_PORTABLE };

struct sa_force_t {
    int vl;
    int tier;
};

// GGML_CUDA_SA_FORCE, parsed once: the V loader and the tier (g1 / g4 are the group policy's, read at every call by
// sparse-attn-attend-g1.cu)
static const sa_force_t & sa_force(void) {
    static const sa_force_t f = [] {
        sa_force_t r = { SA_FORCE_VL_NONE, SA_FORCE_TIER_NONE };
        const char * e = getenv("GGML_CUDA_SA_FORCE");
        if (e == nullptr) {
            return r;
        }
        std::string s(e);
        size_t p = 0;
        while (p <= s.size()) {
            const size_t q = std::min(s.find(',', p), s.size());
            const std::string tok = s.substr(p, q - p);
            if (tok == "vpacked") {
                r.vl = SA_FORCE_VL_PACKED;
            } else if (tok == "vdirect") {
                r.vl = SA_FORCE_VL_DIRECT;
            } else if (tok == "generic") {
                r.tier = SA_FORCE_TIER_GENERIC;
            } else if (tok == "portable") {
                r.tier = SA_FORCE_TIER_PORTABLE;
            } else if (tok == "g1" || tok == "g4") {
                // the group policy's (sparse-attn-attend-g1.cu)
            } else if (!tok.empty()) {
                GGML_LOG_WARN("%s: GGML_CUDA_SA_FORCE: '%s' is not a forcing switch of this build (vpacked, vdirect, generic, portable, g1, g4)\n", __func__, tok.c_str());
            }
            p = q + 1;
        }
        if (r.tier != SA_FORCE_TIER_NONE || r.vl != SA_FORCE_VL_NONE) {
            GGML_LOG_INFO("%s: GGML_CUDA_SA_FORCE=%s: tier %s, V loader %s\n", __func__, e,
                          r.tier == SA_FORCE_TIER_GENERIC ? "generic" : r.tier == SA_FORCE_TIER_PORTABLE ? "portable" : "policy",
                          r.vl == SA_FORCE_VL_PACKED ? "packed" : r.vl == SA_FORCE_VL_DIRECT ? "direct" : "policy");
        }
        return r;
    }();
    return f;
}

static int sa_force_vl(void) {
    return sa_force().vl;
}

// the tier of HIP device `device` before the type pair is known: portable on every arch other than RDNA3.5 and under
// GGML_CUDA_SA_FORCE=portable
static bool sa_device_portable(const int device) {
    return !GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[device].cc) || sa_force().tier == SA_FORCE_TIER_PORTABLE;
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
    int     tier;         // sa_tier
    int     G;
    int     kt, vt, vl;
    int     gqa_index;
    int64_t n_kv, n_quads, n_q, n_stream, n_head, n_head_kv, n_sel;
    size_t  ws_offset;     // round_up(nbytes(dst), 256)
    size_t  qflags_offset; // packed: quad flags, from dst->data
    size_t  any_offset;    // packed: any-flag, from dst->data
    size_t  flags_offset;  // direct: wave flags, from dst->data
    int64_t splits;        // G1: shape-only split count (sa_g1_splits); 1 for G4
    size_t  partial_offset;// G1 with splits > 1: split-K partials, from dst->data
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

    if (!sa_type_in_S(k->type) || !sa_type_in_S(v->type)) {
        r.why = "K/V type outside S (f16, bf16, q8_0, q4_0, f32, q4_1, q5_0, q5_1, iq4_nl)";
        return r;
    }
    // tier (PLAN-v3 §5): portable on every arch other than RDNA3.5 (and when forced); otherwise the generic tier for a pair
    // with a member outside S_fast (and when forced), else the fast tier
    if (sa_device_portable(device)) {
        r.tier = SA_TIER_PORTABLE;
    } else if (r.kt < 0 || r.vt < 0 || sa_force().tier == SA_FORCE_TIER_GENERIC) {
        r.tier = SA_TIER_GENERIC;
    } else {
        r.tier = SA_TIER_FAST;
    }

    // workspace bound over the policy range (G in {1, 4}) at this n_kv: the G4 packed layout at the largest pack any
    // runtime op can need, the G4 direct layout at this n_q, and the largest G1 partials of any n_q' <= n_q (the generic
    // tier uses the same layouts; the portable tier none)
    r.splits = 1;
    r.ws_max = 0;
    if (r.tier != SA_TIER_PORTABLE) {
        const int64_t quads_max = sa_g4_pack_cells_max(r.vt, r.n_q, r.n_kv) / 4;
        const size_t  packed = quads_max > 0 ? sa_g4_packed_ws_bytes(quads_max, r.n_head_kv, r.n_stream) : 0;
        r.ws_max = std::max(packed, sa_g4_flags_bytes(r.n_q, r.n_head_kv, r.n_stream));
        r.ws_max = std::max(r.ws_max, ggml_cuda_sa_g1_partial_bytes_max(r.n_q, r.n_sel, r.n_kv, r.n_head, r.n_stream));
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
    if (r.G != 1 && r.G != 4) {
        r.why = "plan G other than 1 or 4";
        return r;
    }
    if (r.tier == SA_TIER_PORTABLE) {
        r.vl       = -1;
        r.ws_bytes = 0;
        r.ok       = true;
        return r;
    }
    if (r.G == 1) {
        r.vl     = -1;
        r.splits = ggml_cuda_sa_g1_splits(r.n_q, r.n_sel, r.n_kv);
        r.ws_bytes = ggml_cuda_sa_g1_partial_bytes(r.n_q, r.n_head, r.n_stream, r.splits);
        r.partial_offset = r.ws_offset;
        GGML_ASSERT(r.ws_bytes <= r.ws_max);
        r.ok = true;
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
    // grid (G4: (n_groups, n_head_kv, n_stream); G1: (n_stream * n_q * n_head_kv, splits)) and int indexing of queries /
    // groups / plan entries
    const int64_t n_groups = (sel.n_q + 3) / 4;
    if (sel.n_head_kv > 65535 || sel.n_stream > 65535 || n_groups > INT_MAX || sel.n_q > INT_MAX / 4 || sel.n_head > INT_MAX) {
        return false;
    }
    // G1 and portable grids: x = n_stream * n_q * n_head_kv blocks
    if ((sel.G == 1 || sel.tier == SA_TIER_PORTABLE) && sel.n_stream * sel.n_q * sel.n_head_kv > INT_MAX) {
        return false;
    }
    const ggml_sparse_attn_plan_layout L = ggml_sparse_attn_plan_get_layout(sel.n_sel, sel.n_q, sel.n_stream, sel.n_kv, sel.G);
    return L.cap <= INT_MAX;
}

size_t ggml_cuda_sparse_attn_get_alloc_size(int device, const ggml_tensor * op) {
    const sa_selection sel = sa_select(op, device);
    return sel.ws_max > 0 ? sel.ws_offset + sel.ws_max : ggml_nbytes(op);
}

const char * ggml_cuda_sparse_attn_census_skip(const int device, const char * key) {
    const std::string k(key);
    auto under = [&](const char * prefix) { return k.rfind(prefix, 0) == 0; };
    const bool fast     = under("sa_attend/") || under("sa_fixup/") || under("sa_pack/");
    const bool generic  = under("sa_generic/");
    const bool merge    = under("sa_merge");
    const bool portable = under("sa_portable/");
    // the tier: the portable tier on every arch other than RDNA3.5 (device -1: not known, no arch rule) and when forced
    const bool arch_portable = device >= 0 && !GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[device].cc);
    if (arch_portable || sa_force().tier == SA_FORCE_TIER_PORTABLE) {
        if (fast || generic || merge) {
            return arch_portable ? "this HIP arch runs the portable tier (the WMMA tiers are selected on RDNA3.5 only)"
                                 : "GGML_CUDA_SA_FORCE=portable: only the portable tier is dispatched in this process";
        }
        return nullptr;
    }
    if (portable) {
        return "the portable tier runs on HIP archs other than RDNA3.5, and on RDNA3.5 only with GGML_CUDA_SA_FORCE=portable";
    }
    if (sa_force().tier == SA_FORCE_TIER_GENERIC && fast) {
        return "GGML_CUDA_SA_FORCE=generic: the S_fast instances are not dispatched in this process";
    }
    switch (sa_force_vl()) {
        case SA_FORCE_VL_PACKED:
            if ((under("sa_attend/G4/") && k.find("/direct/") != std::string::npos) || under("sa_generic/G4/direct")) {
                return "GGML_CUDA_SA_FORCE=vpacked: the direct-V G4 instances are not dispatched in this process";
            }
            break;
        case SA_FORCE_VL_DIRECT:
            if ((under("sa_attend/G4/") && k.find("/packed/") != std::string::npos) || under("sa_pack/") ||
                under("sa_generic/G4/packed") || under("sa_generic/pack")) {
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
    return sel.ok && sel.G == 4 && sel.tier == SA_TIER_FAST;   // EPI = gated exists for the fast G4 instances only
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
    GGML_ASSERT((sel.ws_bytes == 0 || ggml_backend_buffer_get_alloc_size(dst->buffer, dst) >= sel.ws_offset + sel.ws_bytes) &&
                "SPARSE_ATTN dst is not over-allocated by the HIP buffer type (get_alloc_size)");
    cudaStream_t stream = ctx.stream();
    const bool generic = sel.tier == SA_TIER_GENERIC;

    if (sel.tier == SA_TIER_PORTABLE) {
        GGML_ASSERT(epi == SA_EPI_PLAIN && "the gate fusion has fast G4 instances only (ggml_cuda_sparse_attn_gate_ok)");
        const uint8_t * pd = (const uint8_t *) plan->data;
        ggml_cuda_sa_portable_launch_params p = {};
        p.q = (const char *) q->data; p.k = (const char *) k->data; p.v = (const char *) v->data;
        for (int i = 0; i < 4; ++i) {
            p.q_nb[i] = q->nb[i]; p.k_nb[i] = k->nb[i]; p.v_nb[i] = v->nb[i];
        }
        p.counts    = (const uint32_t *) (pd + L.off_counts);
        p.quads     = (const uint32_t *) (pd + L.off_quads);
        p.masks     = (const uint16_t *) (pd + L.off_masks);
        p.n_groups  = (int) L.n_groups;
        p.cap       = (int) L.cap;
        p.G         = sel.G;
        p.n_q       = (int) sel.n_q;
        p.n_head    = (int) sel.n_head;
        p.n_head_kv = (int) sel.n_head_kv;
        p.n_stream  = (int) sel.n_stream;
        p.gqa       = (int) (sel.n_head / sel.n_head_kv);
        p.kt_g      = (int) k->type;
        p.vt_g      = (int) v->type;
        p.scale     = ggml_get_op_params_f32(dst, 0);
        p.out       = out32;
        ggml_cuda_sa_portable_launch(stream, p);
        return;
    }

    if (sel.G == 1) {
        GGML_ASSERT(epi == SA_EPI_PLAIN && "the gate fusion has fast G4 instances only (ggml_cuda_sparse_attn_gate_ok)");
        const uint8_t * pd = (const uint8_t *) plan->data;
        ggml_cuda_sa_g1_launch_params p = {};
        p.q = (const char *) q->data; p.k = (const char *) k->data; p.v = (const char *) v->data;
        for (int i = 0; i < 4; ++i) {
            p.q_nb[i] = q->nb[i]; p.k_nb[i] = k->nb[i]; p.v_nb[i] = v->nb[i];
        }
        p.counts    = (const uint32_t *) (pd + L.off_counts);
        p.quads     = (const uint32_t *) (pd + L.off_quads);
        p.masks     = (const uint16_t *) (pd + L.off_masks);
        p.n_groups  = (int) L.n_groups;
        p.cap       = (int) L.cap;
        p.n_q       = (int) sel.n_q;
        p.n_head    = (int) sel.n_head;
        p.n_head_kv = (int) sel.n_head_kv;
        p.n_stream  = (int) sel.n_stream;
        p.splits    = (int) sel.splits;
        p.kt        = sel.kt;
        p.vt        = sel.vt;
        p.gqa_index = sel.gqa_index;
        p.kt_g      = (int) k->type;
        p.vt_g      = (int) v->type;
        p.scale     = ggml_get_op_params_f32(dst, 0);
        p.out       = out32;
        p.partial   = sel.splits > 1 ? (float *) ((char *) dst->data + sel.partial_offset) : nullptr;
        if (generic) {
            ggml_cuda_sa_gen_g1_launch(stream, p);
        } else {
            ggml_cuda_sa_g1_launch(stream, p);
        }
        if (sel.splits > 1) {
            ggml_cuda_sa_merge_splits(stream, p.partial, out32, sel.n_stream * sel.n_q * sel.n_head, (int) sel.splits);
        }
        return;
    }

    sa_g4_args a = {};
    a.q = (const char *) q->data; a.q_nb1 = q->nb[1]; a.q_nb2 = q->nb[2]; a.q_nb3 = q->nb[3];
    a.k = (const char *) k->data; a.k_nb1 = k->nb[1]; a.k_nb2 = k->nb[2]; a.k_nb3 = k->nb[3];
    a.v = (const char *) v->data; a.v_nb1 = v->nb[1]; a.v_nb2 = v->nb[2]; a.v_nb3 = v->nb[3];
    if (sel.vl == SA_V_PACKED) {
        uint16_t * pv     = (uint16_t *) ((char *) dst->data + sel.ws_offset);
        uint8_t *  qflags = (uint8_t *) dst->data + sel.qflags_offset;
        uint32_t * any    = (uint32_t *) ((char *) dst->data + sel.any_offset);
        ggml_cuda_sparse_attn_pack_v(stream, v, sel.n_quads, pv, qflags, any, generic);
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
    a.kt_g      = (int) k->type;
    a.vt_g      = (int) v->type;
    if (epi == SA_EPI_GATED) {
        a.gate     = (const char *) gate->data;
        a.gate_nb1 = gate->nb[1];
        a.o16      = o16;
        a.o16_f16  = o16_f16 ? 1 : 0;
    }

    const dim3 grid((unsigned) L.n_groups, (unsigned) sel.n_head_kv, (unsigned) sel.n_stream);
    if (generic) {
        GGML_ASSERT(epi == SA_EPI_PLAIN && "the gate fusion has fast G4 instances only (ggml_cuda_sparse_attn_gate_ok)");
        const int gidx = sel.vl * 3 + sel.gqa_index;
        GGML_ASSERT(gidx >= 0 && gidx < GGML_CUDA_SA_GEN_G4_N);
        ggml_cuda_sa_gen_g4_kernels[gidx]<<<grid, 256, 0, stream>>>(a);
        ggml_cuda_sa_census_add((ggml_cuda_sa_census_id) (GGML_CUDA_SA_CENSUS_GEN_G4_FIRST + gidx));
        ggml_cuda_sa_gen_fixup_kernel<<<grid, 256, 0, stream>>>(a, (int) (sel.n_head / sel.n_head_kv));
        ggml_cuda_sa_census_add(GGML_CUDA_SA_CENSUS_GEN_FIXUP);
        return;
    }

    const int idx = ((sel.vt * SA_V_COUNT + sel.vl) * SA_EPI_COUNT + epi) * 3 + sel.gqa_index;
    GGML_ASSERT(idx >= 0 && idx < GGML_CUDA_SA_G4_PER_KT);
    const sa_g4_kernel_t kernel = sa_g4_table(sel.kt)[idx];
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

const char * ggml_cuda_sparse_attn_census_skip(int device, const char * key) {
    GGML_UNUSED(device);
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
