// f16 store exactness (wave 6b): a fused Vulkan dispatch that ends in f32 arithmetic and stores f16 must equal the unfused
// chain, which stores the f32 result and then converts it (CPY / SET_ROWS / the f32 twin of the op + cast): two roundings,
// f32 then f16. A store written D_TYPE(a * b) can be compiled into one single-rounding mixed-precision multiply (RADV/ACO), one
// f16 ulp off in a few values per 100000.
//
// Per case: the fused graph and a reference graph over shared inputs (the reference keeps an intermediate as a graph output,
// so the fusion cannot match, or flips the fusion's per-call knob), then
//   byte-equal  fused f16 bytes == reference f16 bytes
//   rte         fused f16 == RTE(the f32 value the same kernel family computes before the conversion), host conversion;
//               separates rounding from accumulation order (n/a where no such f32 value exists)
//   maxulp      largest f16 ulp distance fused vs reference
//   on/off      fused-dispatch count of the fusion in the fused / reference graph (engagement; must be expect / 0)
//   run2        the fused graph run again gives the same bytes
//   control     one perturbed input changes the fused output (the comparison can fail)
// Status: ok, FAIL, or acc-order: the reference is a chain whose reduction order differs from the fused kernel by design
// (e.g. rms_norm_small vs rms_norm.comp), byte-equal is reported but not required there; every such line has a strict
// same-kernel line next to it where one exists. Exit code 1 on any FAIL. Skips (exit 0) without a Vulkan device.
//
// Uses only ops and knobs that exist before the store fix, so it builds on the unpatched tree (positive control).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>

// the Vulkan dispatch census test hooks (off until the first reset)
typedef uint64_t (*fusion_count_t)(const char *);
typedef void     (*fusion_reset_t)(void);

static fusion_count_t g_count = nullptr;
static fusion_reset_t g_reset = nullptr;
static int g_n_ok = 0, g_n_acc = 0, g_n_fail = 0;

static uint64_t fnv1a(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= ((const uint8_t *) p)[i]; h *= 1099511628211ull; }
    return h;
}

// distance in f16 ulps (sign-magnitude mapped onto a line)
static int ulp16(ggml_fp16_t a, ggml_fp16_t b) {
    auto key = [](ggml_fp16_t h) { return (h & 0x8000) ? -(int) (h & 0x7fff) : (int) h; };
    return std::abs(key(a) - key(b));
}

static void set_vals(ggml_tensor * t, const std::vector<float> & v) {
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(v.size());
        for (size_t i = 0; i < v.size(); ++i) h[i] = ggml_fp32_to_fp16(v[i]);
        ggml_backend_tensor_set(t, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
    } else {
        std::vector<uint8_t> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, v.data(), q.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    }
}

static void fill(ggml_tensor * t, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(ggml_nelements(t));
    for (auto & x : v) x = u(rng);
    set_vals(t, v);
}

// quantized expert weights, cached per (type, k, rows, seed): iq3_xxs / K-quant quantization is slow on the host
static void fill_quant_cached(ggml_tensor * t, uint32_t seed) {
    static std::map<std::tuple<int, int64_t, int64_t, uint32_t>, std::vector<uint8_t>> cache;
    const auto key = std::make_tuple((int) t->type, t->ne[0], ggml_nrows(t), seed);
    auto it = cache.find(key);
    if (it == cache.end()) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::vector<float> v(ggml_nelements(t));
        for (auto & x : v) x = u(rng);
        std::vector<uint8_t> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, v.data(), q.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
        it = cache.emplace(key, std::move(q)).first;
    }
    ggml_backend_tensor_set(t, it->second.data(), 0, it->second.size());
}

// add 0.5 to element 0 of an f32 / f16 tensor
static void bump(ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32) {
        float v; ggml_backend_tensor_get(t, &v, 0, 4); v += 0.5f; ggml_backend_tensor_set(t, &v, 0, 4);
    } else {
        ggml_fp16_t h; ggml_backend_tensor_get(t, &h, 0, 2); h = ggml_fp32_to_fp16(ggml_fp16_to_fp32(h) + 0.5f); ggml_backend_tensor_set(t, &h, 0, 2);
    }
}

static std::vector<float> get_f32(ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

// rte(fused f16 values, matches, checked): compare with RTE of the reference f32 (tensors the reference runs wrote)
typedef std::function<void(const std::vector<ggml_fp16_t> &, size_t &, size_t &)> rte_fn;

// elementwise RTE against an f32 tensor laid out like the f16 output
static rte_fn rte_of(ggml_tensor * ref32) {
    return [ref32](const std::vector<ggml_fp16_t> & o, size_t & match, size_t & total) {
        const std::vector<float> r = get_f32(ref32);
        total = std::min(o.size(), r.size());
        match = 0;
        for (size_t i = 0; i < total; ++i) match += o[i] == ggml_fp32_to_fp16(r[i]);
    };
}

struct pair_case {
    std::string label;
    const char * fname = nullptr;            // fusion counter name; nullptr: a standalone op, no engagement check
    int expect = 1;                          // required count of fname in the fused graph
    std::vector<ggml_tensor *> roots_f;      // fused graph roots; roots_f[0] is the compared f16 tensor
    std::vector<ggml_tensor *> roots_r;      // reference graph roots; roots_r[0] is the compared f16 tensor
    std::vector<ggml_tensor *> roots_k;      // optional third graph run after the reference (same-kernel f32 for rte)
    bool strict = true;                      // byte-equal required
    rte_fn rte;                              // optional
    std::function<void()> perturb;           // positive control
    const char * knob = nullptr;             // env knob set to "1" for the fused runs and "0" for the reference runs
};

static void run_pair(ggml_backend_t be, ggml_context * ctx, const pair_case & c) {
    auto graph = [&](const std::vector<ggml_tensor *> & roots) {
        ggml_cgraph * g = ggml_new_graph_custom(ctx, 4096, false);
        for (ggml_tensor * t : roots) ggml_build_forward_expand(g, t);
        return g;
    };
    ggml_cgraph * gf = graph(c.roots_f);
    ggml_cgraph * gr = graph(c.roots_r);
    ggml_cgraph * gk = c.roots_k.empty() ? nullptr : graph(c.roots_k);
    ggml_tensor * out_f = c.roots_f[0];
    ggml_tensor * out_r = c.roots_r[0];
    GGML_ASSERT(out_f->type == GGML_TYPE_F16 && out_r->type == GGML_TYPE_F16 && ggml_nelements(out_f) == ggml_nelements(out_r));
    const size_t ne = ggml_nelements(out_f);
    std::vector<ggml_fp16_t> o_f(ne), o_f2(ne), o_r(ne), o_p(ne);
    auto run = [&](ggml_cgraph * g, bool fused, uint64_t * n) {
        if (c.knob) setenv(c.knob, fused ? "1" : "0", 1);
        g_reset();
        ggml_backend_graph_compute(be, g);
        if (n) *n = c.fname ? g_count(c.fname) : 0;
    };
    uint64_t n_f = 0, n_r = 0;
    run(gf, true, &n_f);
    ggml_backend_tensor_get(out_f, o_f.data(), 0, ne * 2);
    run(gr, false, &n_r);
    ggml_backend_tensor_get(out_r, o_r.data(), 0, ne * 2);
    if (gk) run(gk, false, nullptr);
    size_t r_match = 0, r_total = 0;
    if (c.rte) c.rte(o_f, r_match, r_total);
    run(gf, true, nullptr);
    ggml_backend_tensor_get(out_f, o_f2.data(), 0, ne * 2);
    size_t ndiff = 0; int maxulp = 0;
    for (size_t i = 0; i < ne; ++i) {
        if (o_f[i] != o_r[i]) { ndiff++; maxulp = std::max(maxulp, ulp16(o_f[i], o_r[i])); }
    }
    const bool run2 = memcmp(o_f.data(), o_f2.data(), ne * 2) == 0;
    c.perturb();
    run(gf, true, nullptr);
    ggml_backend_tensor_get(out_f, o_p.data(), 0, ne * 2);
    if (c.knob) unsetenv(c.knob);
    const bool ctl = memcmp(o_f.data(), o_p.data(), ne * 2) != 0;
    const bool engaged = c.fname == nullptr || (n_f == (uint64_t) c.expect && n_r == 0);
    const bool rte_ok = !c.rte || r_match == r_total;
    const bool beq = ndiff == 0;
    bool fail = !engaged || !run2 || !ctl || (c.strict && (!beq || !rte_ok));
    const char * status = fail ? "FAIL" : beq ? "ok" : "acc-order";
    if (fail) g_n_fail++; else if (beq) g_n_ok++; else g_n_acc++;
    char rte_s[64];
    if (c.rte) snprintf(rte_s, sizeof(rte_s), "%s(%zu/%zu)", rte_ok ? "yes" : "NO", r_match, r_total);
    else snprintf(rte_s, sizeof(rte_s), "n/a");
    printf("%-9s %-58s byte-equal=%s (%zu/%zu differ, maxulp %d) rte=%s fused dispatches on/off=%llu/%llu run2=%s control=%s hash=%016llx\n",
           status, c.label.c_str(), beq ? "yes" : "NO", ndiff, ne, maxulp, rte_s, (unsigned long long) n_f, (unsigned long long) n_r,
           run2 ? "same" : "DIFF", ctl ? "differs" : "SAME", (unsigned long long) fnv1a(o_f.data(), ne * 2));
    fflush(stdout);
}

static ggml_context * new_ctx() {
    ggml_init_params ip = { 256 * ggml_tensor_overhead() + 8 * ggml_graph_overhead_custom(4096, false), nullptr, true };
    return ggml_init(ip);
}

// ---------------------------------------------------------------------------------------------------------------------
// MUL_MAT_ID + MUL(router weights) + CPY(f16): the expert down projection, int8 MMQ (mul_mmq_cm1.comp d16) or the routed
// f16-B pf path (mul_mm.comp MMID_SCALE d16). Reference: the MUL_MAT_ID and MUL outputs are graph outputs.
static void case_mmid_mul(ggml_backend_t be, ggml_type t, int64_t m, int64_t k, int n_mats, int n_used, int64_t nt, bool b16) {
    ggml_context * ctx = new_ctx();
    ggml_tensor * as  = ggml_new_tensor_3d(ctx, t, k, m, n_mats);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, nt);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, b16 ? GGML_TYPE_F16 : GGML_TYPE_F32, k, n_used, nt);
    ggml_tensor * w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, nt);
    ggml_tensor * of  = ggml_cast(ctx, ggml_mul(ctx, ggml_mul_mat_id(ctx, as, b, ids), w), GGML_TYPE_F16);
    ggml_set_output(of);
    ggml_tensor * rmm = ggml_mul_mat_id(ctx, as, b, ids); ggml_set_output(rmm);
    ggml_tensor * rmu = ggml_mul(ctx, rmm, w);            ggml_set_output(rmu);
    ggml_tensor * orr = ggml_cast(ctx, rmu, GGML_TYPE_F16); ggml_set_output(orr);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    GGML_ASSERT(buf);
    std::mt19937 rng(4321 + (int) t * 131 + (int) m + (int) nt);
    fill_quant_cached(as, 99 + (uint32_t) t);
    std::vector<int32_t> idv(n_used * nt), perm(n_mats);
    for (int64_t i = 0; i < nt; ++i) {
        for (int e = 0; e < n_mats; ++e) perm[e] = e;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int j = 0; j < n_used; ++j) idv[i * n_used + j] = perm[j];
    }
    ggml_backend_tensor_set(ids, idv.data(), 0, idv.size() * sizeof(int32_t));
    fill(b, rng, -1.0f, 1.0f);
    fill(w, rng, 0.05f, 1.05f);
    pair_case c;
    char label[128];
    snprintf(label, sizeof(label), "MUL_MAT_ID_MUL_CPY16 %s %lldx%lld e%d/%d nt=%lld b=%s", ggml_type_name(t), (long long) m, (long long) k,
             n_used, n_mats, (long long) nt, b16 ? "f16" : "f32");
    c.label = label; c.fname = "MUL_MAT_ID_MUL_CPY16";
    c.roots_f = { of }; c.roots_r = { orr };
    c.rte = rte_of(rmu);
    c.perturb = [b]() { bump(b); };
    run_pair(be, ctx, c);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// MUL_MAT_ID + CPY(f16) + SWIGLU(gate, up): the SwiGLU in the gate/up GEMM's f16 store (int8 MMQ). Reference: both casts are
// graph outputs (MUL_MAT_ID_CPY16 each, then the f16 SWIGLU op). rte: against SWIGLU in f32 over the same f16 gate and up.
static void case_mmid_glu(ggml_backend_t be, ggml_type t, int64_t m, int64_t k, int n_mats, int n_used, int64_t nt) {
    ggml_context * ctx = new_ctx();
    ggml_tensor * ag  = ggml_new_tensor_3d(ctx, t, k, m, n_mats);
    ggml_tensor * au  = ggml_new_tensor_3d(ctx, t, k, m, n_mats);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, nt);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, nt);
    auto chain = [&](bool ref, ggml_tensor ** g16o, ggml_tensor ** u16o) {
        ggml_tensor * g16 = ggml_cast(ctx, ggml_mul_mat_id(ctx, ag, b, ids), GGML_TYPE_F16);
        ggml_tensor * u16 = ggml_cast(ctx, ggml_mul_mat_id(ctx, au, b, ids), GGML_TYPE_F16);
        if (ref) { ggml_set_output(g16); ggml_set_output(u16); }
        ggml_tensor * o = ggml_swiglu_split(ctx, g16, u16);
        ggml_set_output(o);
        if (g16o) *g16o = g16;
        if (u16o) *u16o = u16;
        return o;
    };
    ggml_tensor * of = chain(false, nullptr, nullptr);
    ggml_tensor * g16r = nullptr, * u16r = nullptr;
    ggml_tensor * orr = chain(true, &g16r, &u16r);
    ggml_tensor * s32 = ggml_swiglu_split(ctx, ggml_cast(ctx, g16r, GGML_TYPE_F32), ggml_cast(ctx, u16r, GGML_TYPE_F32));
    ggml_set_output(s32);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    GGML_ASSERT(buf);
    std::mt19937 rng(777 + (int) t * 17 + (int) nt);
    fill_quant_cached(ag, 1001 + (uint32_t) t);
    fill_quant_cached(au, 2002 + (uint32_t) t);
    std::vector<int32_t> idv(n_used * nt), perm(n_mats);
    for (int64_t i = 0; i < nt; ++i) {
        for (int e = 0; e < n_mats; ++e) perm[e] = e;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int j = 0; j < n_used; ++j) idv[i * n_used + j] = perm[j];
    }
    ggml_backend_tensor_set(ids, idv.data(), 0, idv.size() * sizeof(int32_t));
    fill(b, rng, -1.0f, 1.0f);
    pair_case c;
    char label[128];
    snprintf(label, sizeof(label), "MUL_MAT_ID_CPY16_SWIGLU %s %lldx%lld e%d/%d nt=%lld", ggml_type_name(t), (long long) m, (long long) k,
             n_used, n_mats, (long long) nt);
    c.label = label; c.fname = "MUL_MAT_ID_CPY16_SWIGLU";
    c.roots_f = { of }; c.roots_r = { orr, s32 };
    c.rte = rte_of(s32);
    c.perturb = [b]() { bump(b); };
    run_pair(be, ctx, c);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// RMS_NORM + MUL(gamma) + SIGMOID(z) + MUL + CPY(f16) (RMS_NORM_MUL_SIGMUL_CPY, rms_norm_small.comp f16), the GDN output norm.
// Reference: every intermediate is a graph output (RMS_NORM_MUL on the same short-row kernel, SIGMOID, MUL, CPY).
static void case_gdn_norm(ggml_backend_t be, int64_t ne0, int64_t ne1, int64_t nt) {
    ggml_context * ctx = new_ctx();
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, nt);
    ggml_tensor * g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    ggml_tensor * z = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, nt);
    ggml_tensor * last_r = nullptr;
    auto chain = [&](bool ref) {
        ggml_tensor * n = ggml_rms_norm(ctx, x, 1e-6f);
        ggml_tensor * m = ggml_mul(ctx, n, g);       if (ref) ggml_set_output(m);
        ggml_tensor * s = ggml_sigmoid(ctx, z);      if (ref) ggml_set_output(s);
        ggml_tensor * p = ggml_mul(ctx, m, s);       if (ref) { ggml_set_output(p); last_r = p; }
        ggml_tensor * o = ggml_cast(ctx, p, GGML_TYPE_F16);
        ggml_set_output(o);
        return o;
    };
    ggml_tensor * of = chain(false);
    ggml_tensor * orr = chain(true);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    GGML_ASSERT(buf);
    std::mt19937 rng(55 + (int) ne0 + (int) ne1 + (int) nt);
    fill(x, rng, -2.0f, 2.0f); fill(g, rng, -2.0f, 2.0f); fill(z, rng, -4.0f, 4.0f);
    pair_case c;
    c.label = "RMS_NORM_MUL_SIGMUL_CPY " + std::to_string(ne0) + "x" + std::to_string(ne1) + " nt=" + std::to_string(nt);
    c.fname = "RMS_NORM_MUL_SIGMUL_CPY";
    c.roots_f = { of }; c.roots_r = { orr };
    c.rte = rte_of(last_r);
    c.perturb = [x]() { bump(x); };
    run_pair(be, ctx, c);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// SCALE + SIGMOID + SCALE + DSV4_HC_POST + RMS_NORM + MUL + CPY(f16) (HC_POST_GATE_NORM_CPY, dsv4_hc_post_norm.comp OUT_F16), the
// qwen4exp hc combine with the next hc norm. Strict reference: the same kernel with an f32 norm output (HC_POST_GATE_NORM, the
// MUL is a graph output), then the CPY. Second line: the fully unfused chain (rms_norm.comp reduction order, acc-order allowed).
static void case_hc_post_norm(ggml_backend_t be, int64_t n_embd, int64_t nt) {
    const int64_t hc = 4;
    ggml_context * ctx = new_ctx();
    ggml_tensor * x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, nt);
    ggml_tensor * res   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, nt);
    ggml_tensor * inj   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, nt);
    ggml_tensor * gamma = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, hc);
    // mode 0: fused; 1: same-kernel reference (MUL output); 2: fully unfused
    auto chain = [&](int mode, ggml_tensor ** ro_o, ggml_tensor ** m_o) {
        ggml_tensor * w = ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, inj, 0.25f)), 2.0f);
        ggml_tensor * ro = ggml_dsv4_hc_post(ctx, x, res, w, nullptr);
        ggml_set_output(ro);   // the next combine reads it as its residual
        ggml_tensor * n = ggml_rms_norm(ctx, ro, 1e-6f);
        if (mode == 2) ggml_set_output(n);
        ggml_tensor * m = ggml_mul(ctx, n, gamma);
        if (mode >= 1) ggml_set_output(m);
        ggml_tensor * o = ggml_cast(ctx, m, GGML_TYPE_F16);
        ggml_set_output(o);
        *ro_o = ro; *m_o = m;
        return o;
    };
    ggml_tensor * ro_f, * m_f, * ro_k, * m_k, * ro_u, * m_u;
    ggml_tensor * of = chain(0, &ro_f, &m_f);
    ggml_tensor * ok = chain(1, &ro_k, &m_k);
    ggml_tensor * ou = chain(2, &ro_u, &m_u);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    GGML_ASSERT(buf);
    std::mt19937 rng(91 + (int) n_embd + (int) nt);
    fill(x, rng, -1.0f, 1.0f); fill(res, rng, -1.0f, 1.0f); fill(inj, rng, -2.0f, 2.0f); fill(gamma, rng, 0.0f, 2.0f);
    const std::string sh = std::to_string(n_embd) + "x4 nt=" + std::to_string(nt);
    pair_case c;
    c.label = "HC_POST_GATE_NORM_CPY " + sh + " vs same-kernel f32+CPY";
    c.fname = "HC_POST_GATE_NORM_CPY";
    c.roots_f = { of, ro_f }; c.roots_r = { ok, ro_k };
    c.rte = rte_of(m_k);
    c.perturb = [x]() { bump(x); };
    run_pair(be, ctx, c);
    pair_case u = c;
    u.label = "HC_POST_GATE_NORM_CPY " + sh + " vs unfused chain";
    u.roots_r = { ou, ro_u };
    u.rte = rte_of(m_u);
    u.strict = false;
    run_pair(be, ctx, u);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// RMS_NORM + MUL + CPY(f16) on the hc norm kernel without the combine (HC_NORM_ROWS_CPY, fusion name RMS_NORM_MUL_CPY): the
// qwen4exp hc norm [n_embd, hc, nt] with a [n_embd, hc] gamma. No f32 twin of this kernel is reachable from a graph, so the
// reference is the unfused chain (rms_norm.comp reduction order): acc-order allowed; the store line is the one the strict
// HC_POST_GATE_NORM_CPY case checks.
static void case_hc_norm_rows(ggml_backend_t be, int64_t n_embd, int64_t hc, int64_t nt) {
    ggml_context * ctx = new_ctx();
    ggml_tensor * x     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, nt);
    ggml_tensor * gamma = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, hc);
    ggml_tensor * m_r = nullptr;
    auto chain = [&](bool ref) {
        ggml_tensor * n = ggml_rms_norm(ctx, x, 1e-6f);
        if (ref) ggml_set_output(n);
        ggml_tensor * m = ggml_mul(ctx, n, gamma);
        if (ref) { ggml_set_output(m); m_r = m; }
        ggml_tensor * o = ggml_cast(ctx, m, GGML_TYPE_F16);
        ggml_set_output(o);
        return o;
    };
    ggml_tensor * of = chain(false);
    ggml_tensor * orr = chain(true);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    GGML_ASSERT(buf);
    std::mt19937 rng(13 + (int) n_embd + (int) hc + (int) nt);
    fill(x, rng, -2.0f, 2.0f); fill(gamma, rng, 0.0f, 2.0f);
    pair_case c;
    c.label = "RMS_NORM_MUL_CPY (hc rows) " + std::to_string(n_embd) + "x" + std::to_string(hc) + " nt=" + std::to_string(nt) + " vs unfused";
    c.fname = "RMS_NORM_MUL_CPY";
    c.roots_f = { of }; c.roots_r = { orr };
    c.rte = rte_of(m_r);
    c.strict = false;
    c.perturb = [x]() { bump(x); };
    run_pair(be, ctx, c);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// DSV4_HC_PRE gated with an f16 output (dsv4_hc_pre.comp D_F16), standalone, against its f32 output + CPY(f16)
static void case_hc_pre(ggml_backend_t be, int64_t n_embd, int64_t nt, bool g16) {
    const int64_t hc = 4;
    ggml_context * ctx = new_ctx();
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_embd, hc, nt);
    ggml_tensor * g = ggml_new_tensor_3d(ctx, g16 ? GGML_TYPE_F16 : GGML_TYPE_F32, n_embd, hc, nt);
    ggml_tensor * of = ggml_dsv4_hc_pre_gated_ext(ctx, x, g, 0.25f, GGML_TYPE_F16); ggml_set_output(of);
    ggml_tensor * r32 = ggml_dsv4_hc_pre_gated_ext(ctx, x, g, 0.25f, GGML_TYPE_F32); ggml_set_output(r32);
    ggml_tensor * orr = ggml_cast(ctx, r32, GGML_TYPE_F16); ggml_set_output(orr);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    GGML_ASSERT(buf);
    std::mt19937 rng(23 + (int) n_embd + (int) nt + (g16 ? 1 : 0));
    fill(x, rng, -2.0f, 2.0f); fill(g, rng, -3.0f, 3.0f);
    pair_case c;
    c.label = std::string("DSV4_HC_PRE f16 out ") + (g16 ? "g16 " : "g32 ") + std::to_string(n_embd) + " nt=" + std::to_string(nt) + " vs f32+CPY";
    c.roots_f = { of }; c.roots_r = { orr };
    c.rte = rte_of(r32);
    c.perturb = [x]() { bump(x); };
    run_pair(be, ctx, c);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// HC_UP_MIX with the f16 mixed stream (mul_mmq_cm1.comp HC_EPI d16): MUL_MAT(q8_0 hc up) + CPY(f16 gate) + DSV4_HC_PRE(f16 out).
// Reference: the gate GEMM + CPY, DSV4_HC_PRE with an f32 output, then CPY(f16); GGML_VK_HC_UP_MIX=0 for the reference runs.
static void case_hc_up_mix(ggml_backend_t be, int64_t n_embd, int64_t nt, int64_t lr) {
    const int64_t hc = 4;
    ggml_context * ctx = new_ctx();
    ggml_tensor * w  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, lr, hc * n_embd);
    ggml_tensor * lo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lr, nt);
    ggml_tensor * x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_embd, hc, nt);
    auto gate = [&]() { return ggml_reshape_3d(ctx, ggml_cast(ctx, ggml_mul_mat(ctx, w, lo), GGML_TYPE_F16), n_embd, hc, nt); };
    ggml_tensor * of = ggml_dsv4_hc_pre_gated_ext(ctx, x, gate(), 0.25f, GGML_TYPE_F16); ggml_set_output(of);
    ggml_tensor * gr = gate(); ggml_set_output(gr);
    ggml_tensor * m32 = ggml_dsv4_hc_pre_gated_ext(ctx, x, gr, 0.25f, GGML_TYPE_F32); ggml_set_output(m32);
    ggml_tensor * orr = ggml_cast(ctx, m32, GGML_TYPE_F16); ggml_set_output(orr);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    GGML_ASSERT(buf);
    std::mt19937 rng(31 + (int) n_embd + (int) nt);
    fill_quant_cached(w, 4242);
    fill(lo, rng, -1.0f, 1.0f); fill(x, rng, -1.0f, 1.0f);
    pair_case c;
    c.label = "HC_UP_MIX f16 mix " + std::to_string(n_embd) + " nt=" + std::to_string(nt) + " lr=" + std::to_string(lr) + " vs f32 hc_pre+CPY";
    c.fname = "HC_UP_MIX";
    c.roots_f = { of }; c.roots_r = { orr };
    c.rte = rte_of(m32);
    c.knob = "GGML_VK_HC_UP_MIX";
    c.perturb = [lo]() { bump(lo); };
    run_pair(be, ctx, c);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// scattered cache rows: distinct for nt <= kv (odd multiplier, power-of-two kv)
static std::vector<int64_t> cache_rows(int64_t nt, int64_t kv) {
    std::vector<int64_t> iv(nt);
    for (int64_t t = 0; t < nt; ++t) iv[t] = (t * 2731 + 911) % kv;
    return iv;
}

// RMS_NORM + VIEW + SET_ROWS into an f16 KV cache (RMS_NORM_VIEW_SET_ROWS, rms_norm_set_rows_f32_f16). byte-equal: the whole
// cache against the unfused chain (RMS_NORM f32 output, SET_ROWS f32 -> f16) with the same rows; rte: against the same kernel
// writing an f32 cache (rms_norm_set_rows_f32_f32). Rows <= 256 wide take rms_norm_small.comp unfused (another reduction
// order): acc-order allowed there; wider rows run rms_norm.comp both ways and are strict.
static void case_norm_set_rows(ggml_backend_t be, int64_t hd, int64_t nh, int64_t nt) {
    const int64_t kv = 4096;
    ggml_context * ctx = new_ctx();
    ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, nt);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, nt);
    ggml_tensor * cf  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hd * nh, kv);
    ggml_tensor * cr  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hd * nh, kv);
    ggml_tensor * ck  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd * nh, kv);
    auto chain = [&](ggml_tensor * cache, bool ref) {
        ggml_tensor * n = ggml_rms_norm(ctx, x, 1e-6f);
        if (ref) ggml_set_output(n);
        ggml_tensor * v = ggml_view_2d(ctx, n, hd * nh, nt, n->nb[2], 0);
        ggml_tensor * o = ggml_set_rows(ctx, cache, v, idx);
        ggml_set_output(o);
        return o;
    };
    ggml_tensor * of = chain(cf, false);
    ggml_tensor * orr = chain(cr, true);
    ggml_tensor * ok = chain(ck, false);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    GGML_ASSERT(buf);
    std::mt19937 rng(71 + (int) hd + (int) nh + (int) nt);
    fill(x, rng, -3.0f, 3.0f);
    // the same background in every cache: rows the write must not touch compare too
    std::vector<float> bg(hd * nh * kv);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto & v : bg) v = ggml_fp16_to_fp32(ggml_fp32_to_fp16(u(rng)));
    set_vals(cf, bg); set_vals(cr, bg); set_vals(ck, bg);
    const std::vector<int64_t> iv = cache_rows(nt, kv);
    ggml_backend_tensor_set(idx, iv.data(), 0, iv.size() * sizeof(int64_t));
    pair_case c;
    c.label = "RMS_NORM_VIEW_SET_ROWS f16 kv " + std::to_string(hd) + "x" + std::to_string(nh) + " nt=" + std::to_string(nt);
    c.fname = "RMS_NORM_VIEW_SET_ROWS";
    c.roots_f = { of }; c.roots_r = { orr }; c.roots_k = { ok };
    c.rte = rte_of(ck);
    c.strict = hd > 256;
    c.perturb = [x]() { bump(x); };
    run_pair(be, ctx, c);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// The rope KV write into an f16 cache: ROPE + VIEW + SET_ROWS (ROPE_VIEW_SET_ROWS, rope_*_f32_f16) or, norm = true,
// RMS_NORM + MUL + ROPE + VIEW + SET_ROWS (RMS_NORM_MUL_ROPE_VIEW_SET_ROWS, rms_norm_mul_rope_f32_f16). Strict reference: the
// rope output is a graph output (ROPE f32, or the same norm+rope kernel RMS_NORM_MUL_ROPE in f32), then SET_ROWS f32 -> f16.
// rte: the cache rows against RTE of that reference rope output. multi: ggml_rope_multi (mode IMROPE / MROPE).
static void case_rope_set_rows(ggml_backend_t be, bool norm, int mode, int64_t hd, int64_t nh, int64_t nt, int n_dims) {
    const int64_t kv = 4096;
    const bool multi = (mode & GGML_ROPE_TYPE_MROPE) != 0;
    ggml_context * ctx = new_ctx();
    ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, nt);
    ggml_tensor * w   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, multi ? nt * 4 : nt);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, nt);
    ggml_tensor * cf  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hd * nh, kv);
    ggml_tensor * cr  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hd * nh, kv);
    ggml_tensor * rope_r = nullptr;
    auto chain = [&](ggml_tensor * cache, bool ref) {
        ggml_tensor * a = x;
        if (norm) {
            a = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), w);
        }
        // Flash-Next: [11, 11, 10, 0] over 64 rotated dims
        int sections[GGML_MROPE_SECTIONS] = { n_dims / 2 - 2 * (n_dims / 6) - 1, n_dims / 6 + 1, n_dims / 6, 0 };
        ggml_tensor * r = multi ? ggml_rope_multi(ctx, a, pos, nullptr, n_dims, sections, mode, 262144, 10000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f)
                                : ggml_rope_ext(ctx, a, pos, nullptr, n_dims, mode, 262144, 10000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        if (ref) { ggml_set_output(r); rope_r = r; }
        ggml_tensor * v = ggml_view_2d(ctx, r, hd * nh, nt, r->nb[2], 0);
        ggml_tensor * o = ggml_set_rows(ctx, cache, v, idx);
        ggml_set_output(o);
        return o;
    };
    ggml_tensor * of = chain(cf, false);
    ggml_tensor * orr = chain(cr, true);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    GGML_ASSERT(buf);
    std::mt19937 rng(57 + (int) hd + mode + (norm ? 1 : 0) + (int) nt);
    fill(x, rng, -3.0f, 3.0f); fill(w, rng, 0.0f, 2.0f);
    std::vector<int32_t> pv(ggml_nelements(pos));
    for (size_t i = 0; i < pv.size(); ++i) pv[i] = 1000 + (int32_t) (i % nt) * 3 + (int32_t) (i / nt);
    ggml_backend_tensor_set(pos, pv.data(), 0, pv.size() * sizeof(int32_t));
    std::vector<float> bg(hd * nh * kv);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto & v : bg) v = u(rng);
    set_vals(cf, bg); set_vals(cr, bg);
    const std::vector<int64_t> iv = cache_rows(nt, kv);
    ggml_backend_tensor_set(idx, iv.data(), 0, iv.size() * sizeof(int64_t));
    const int64_t row = hd * nh;
    pair_case c;
    c.label = std::string(norm ? "RMS_NORM_MUL_ROPE_VIEW_SET_ROWS" : "ROPE_VIEW_SET_ROWS") + " mode=" + std::to_string(mode) + " " +
              std::to_string(hd) + "x" + std::to_string(nh) + " rot=" + std::to_string(n_dims) + " nt=" + std::to_string(nt);
    c.fname = norm ? "RMS_NORM_MUL_ROPE_VIEW_SET_ROWS" : "ROPE_VIEW_SET_ROWS";
    c.roots_f = { of }; c.roots_r = { orr };
    c.rte = [rope_r, iv, nt, row](const std::vector<ggml_fp16_t> & o, size_t & match, size_t & total) {
        const std::vector<float> r = get_f32(rope_r);
        match = 0; total = (size_t) (nt * row);
        for (int64_t t = 0; t < nt; ++t) {
            for (int64_t e = 0; e < row; ++e) match += o[iv[t] * row + e] == ggml_fp32_to_fp16(r[t * row + e]);
        }
    };
    c.perturb = [x]() { bump(x); };
    run_pair(be, ctx, c);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main() {
    // the routed f16-B pf path (mul_mm.comp MMID_SCALE d16) for iq3_xxs (no int8 MMQ pipeline) and q6_K; read once per process
    setenv("GGML_VK_MMID_F16COOPMAT", "iq3_xxs,q6_K", 0);
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("Vulkan0");
    if (!dev) { printf("test-f16-store-exact: no Vulkan0 device, skipped\n"); return 0; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    g_count = (fusion_count_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_dispatch_census_count");
    g_reset = (fusion_reset_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_dispatch_census_reset");
    if (!g_count || !g_reset) { printf("FAIL: no dispatch census counters\n"); return 1; }
    printf("test-f16-store-exact: GGML_VK_MMID_F16COOPMAT=%s\n", getenv("GGML_VK_MMID_F16COOPMAT"));

    // Flash-Next (qwen4exp): n_embd 2560, 10 of 320 experts (32 here, for host quantization time), expert ffn 640 (768 for the
    // 256-block types), GDN 128 x 48, K/V heads 256 x 2 with an imrope over 64 dims, hc 4 x 2560, low rank 320; ubatch 2048 and
    // a ragged 130. Qwen3.6-35B-A3B: n_embd 2048, 8 experts of ffn 512, GDN 128 x 32.
    for (int64_t nt : { 2048, 130 }) {
        case_mmid_mul(be, GGML_TYPE_Q8_0,    2560, 640, 32, 10, nt, false);   // int8 MMQ
        case_mmid_mul(be, GGML_TYPE_IQ4_NL,  2560, 640, 32, 10, nt, false);
        case_mmid_mul(be, GGML_TYPE_Q4_K,    2560, 768, 32, 10, nt, false);
        case_mmid_mul(be, GGML_TYPE_Q8_0,    2560, 640, 32, 10, nt, true);    // int8 MMQ, B quantized from f16
        case_mmid_mul(be, GGML_TYPE_IQ3_XXS, 2560, 768, 32, 10, nt, false);   // routed f16-B pf, f32 B
        case_mmid_mul(be, GGML_TYPE_IQ3_XXS, 2560, 768, 32, 10, nt, true);    // routed f16-B pf, f16 B
        case_mmid_mul(be, GGML_TYPE_Q6_K,    2560, 768, 32, 10, nt, true);
    }
    case_mmid_mul(be, GGML_TYPE_Q4_K, 2048, 512, 32, 8, 512, false);          // 35B down
    for (int64_t nt : { 2048, 130 }) {
        case_mmid_glu(be, GGML_TYPE_Q8_0,   640, 2560, 32, 10, nt);
        case_mmid_glu(be, GGML_TYPE_IQ4_NL, 640, 2560, 32, 10, nt);
        case_mmid_glu(be, GGML_TYPE_Q4_K,   768, 2560, 32, 10, nt);
    }
    case_mmid_glu(be, GGML_TYPE_Q4_K, 512, 2048, 32, 8, 512);                 // 35B gate/up

    case_gdn_norm(be, 128, 48, 2048);
    case_gdn_norm(be, 128, 48, 130);
    case_gdn_norm(be, 128, 32, 512);

    case_hc_post_norm(be, 2560, 2048);
    case_hc_post_norm(be, 2560, 130);
    case_hc_norm_rows(be, 2560, 4, 2048);
    case_hc_norm_rows(be, 2048, 1, 512);                                      // 35B-wide norm

    case_hc_pre(be, 2560, 2048, false);
    case_hc_pre(be, 2560, 2048, true);
    case_hc_pre(be, 2560, 130, true);
    case_hc_up_mix(be, 2560, 2048, 320);
    case_hc_up_mix(be, 2560, 130, 320);

    // the KV writes: norm into the cache (K/V head shapes), and the rope writes (FN K: imrope 64 of 256, 2 heads)
    for (int64_t nt : { 1, 7, 130 }) {
        case_norm_set_rows(be, 256, 2, nt);
        case_norm_set_rows(be, 128, 2, nt);
        case_norm_set_rows(be, 512, 1, nt);
        case_rope_set_rows(be, false, GGML_ROPE_TYPE_IMROPE, 256, 2, nt, 64);
        case_rope_set_rows(be, false, GGML_ROPE_TYPE_NEOX,   128, 8, nt, 128);
        case_rope_set_rows(be, false, GGML_ROPE_TYPE_NORMAL, 128, 8, nt, 128);
        case_rope_set_rows(be, true,  GGML_ROPE_TYPE_NEOX,   256, 2, nt, 256);
        case_rope_set_rows(be, true,  GGML_ROPE_TYPE_NORMAL, 128, 8, nt, 128);
    }

    ggml_backend_free(be);
    printf("test-f16-store-exact: %s (%d ok, %d acc-order, %d FAIL)\n", g_n_fail ? "FAIL" : "OK", g_n_ok, g_n_acc, g_n_fail);
    return g_n_fail ? 1 : 0;
}
