// Vulkan fused dispatches byte-equal with the unfused chain on the same device (test-backend-ops compares against the CPU).
// Usage: test-vk-exact [f16-store] [hc-up-mix] [mmid-frag] (default: the first two; mmid-frag is a tool for its .py).
// Exit 1 on any FAIL, 0 without Vulkan0; with GGML_VK_DISABLE_FUSION no case expects a fused dispatch.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>

// setenv / unsetenv are POSIX only; _putenv_s with an empty value removes the variable on Windows
static void set_env(const char * name, const char * value, bool overwrite = true) {
    if (!overwrite && getenv(name)) {
        return;
    }
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// the Vulkan dispatch census test hooks (off until the first reset)
typedef uint64_t (*fusion_count_t)(const char *);
typedef void     (*fusion_reset_t)(void);

static fusion_count_t g_count = nullptr;
static fusion_reset_t g_reset = nullptr;
static void (*g_env_reload)(void) = nullptr;   // the backend caches knobs it reads per graph; null on a backend without the hook

// a Vulkan knob flipped at run time: the backend re-reads it on its next use
static void set_vk_knob(const char * name, const char * value) {
    set_env(name, value);
    if (g_env_reload) {
        g_env_reload();
    }
}

static uint64_t fnv1a(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= ((const uint8_t *) p)[i]; h *= 1099511628211ull; }
    return h;
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

// ---------------------------------------------------------------------------------------------------------------------
// f16-store: a fused dispatch storing f16 must round like the unfused f32 store + convert (RADV can fold D_TYPE(a * b) into
// one single-rounding multiply). Per case: byte-equal, rte (vs RTE of the kernel's f32 value), engagement, run2, control;
// acc-order: the reference reduces in another order by design, byte-equal reported but not required.

namespace f16_store {

static int g_n_ok = 0, g_n_acc = 0, g_n_fail = 0;
static bool g_fusion = true;

// distance in f16 ulps (sign-magnitude mapped onto a line)
static int ulp16(ggml_fp16_t a, ggml_fp16_t b) {
    auto key = [](ggml_fp16_t h) { return (h & 0x8000) ? -(int) (h & 0x7fff) : (int) h; };
    return std::abs(key(a) - key(b));
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
        if (c.knob) set_vk_knob(c.knob, fused ? "1" : "0");
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
    if (c.knob) set_vk_knob(c.knob, nullptr);
    const bool ctl = memcmp(o_f.data(), o_p.data(), ne * 2) != 0;
    const bool engaged = c.fname == nullptr || (n_f == (g_fusion ? (uint64_t) c.expect : 0) && n_r == 0);
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

// RMS_NORM + MUL + CPY(f16) on the hc norm kernel (RMS_NORM_MUL_CPY): no f32 twin is reachable from a graph, so acc-order;
// its store is the one the strict HC_POST_GATE_NORM_CPY case checks
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

// RMS_NORM + VIEW + SET_ROWS into an f16 KV cache vs the unfused chain; rte vs the same kernel writing f32. Rows <= 256 run
// rms_norm_small.comp unfused (acc-order), wider rows are strict
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

// (RMS_NORM + MUL +) ROPE + VIEW + SET_ROWS into an f16 cache; strict reference: the rope (or norm + rope) output in f32,
// then SET_ROWS; rte against that output
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

static int run(ggml_backend_t be, bool fusion) {
    g_fusion = fusion;
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

    printf("f16-store: %s (%d ok, %d acc-order, %d FAIL)\n", g_n_fail ? "FAIL" : "OK", g_n_ok, g_n_acc, g_n_fail);
    return g_n_fail;
}

} // namespace f16_store

// ---------------------------------------------------------------------------------------------------------------------
// hc-up-mix: HC_UP_MIX (hc up GEMM + gated hc pre in one int8 MMQ epilogue) must be bit-exact with the unfused chain, or
// cached multi-turn and a full re-prefill diverge; GGML_VK_HC_UP_MIX=1 vs 0 per shape, engagement and a positive control

namespace hc_up_mix {

struct shape { int64_t n_embd, nt, k, pad; bool expect_fused; bool gate16 = true; bool mix16 = false; };   // gate16 false: the f32-gate form; mix16: the f16 mixed stream

static int run_shape(ggml_backend_t be, const shape & sh, bool fusion) {
    ggml_init_params ip = { 32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int64_t hc_dim = 4 * sh.n_embd;
    ggml_tensor * w   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, sh.k, hc_dim);
    ggml_tensor * lo  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.k, sh.nt);
    ggml_tensor * xb  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hc_dim + sh.pad, sh.nt);   // row-padded like LLAMA_HC_XN_PAD
    ggml_tensor * x   = ggml_view_3d(ctx, xb, sh.n_embd, 4, sh.nt, sh.n_embd * sizeof(ggml_fp16_t), xb->nb[1], 0);
    ggml_tensor * g   = ggml_mul_mat(ctx, w, lo);
    ggml_tensor * g16 = sh.gate16 ? ggml_cast(ctx, g, GGML_TYPE_F16) : g;   // (named g16 either way: the gate the HC pre reads)
    ggml_tensor * rs  = ggml_reshape_3d(ctx, g16, sh.n_embd, 4, sh.nt);
    ggml_tensor * mixed = ggml_dsv4_hc_pre_gated_ext(ctx, x, rs, 0.25f, sh.mix16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
    const size_t es = ggml_element_size(mixed), nb_out = ggml_nbytes(mixed);   // f16 mixed: the float vectors below hold raw bytes
    ggml_set_output(mixed);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, mixed);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "FAIL: alloc\n"); return 1; }

    std::mt19937 rng(1234 + (int) sh.n_embd + (int) sh.nt);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> wf(sh.k * hc_dim), lv(sh.k * sh.nt);
    for (auto & v : wf) v = u(rng);
    for (auto & v : lv) v = u(rng);
    std::vector<uint8_t> wq(ggml_nbytes(w));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, wf.data(), wq.data(), 0, hc_dim, sh.k, nullptr);
    std::vector<ggml_fp16_t> xv((hc_dim + sh.pad) * sh.nt);
    for (auto & v : xv) v = ggml_fp32_to_fp16(u(rng));
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(lo, lv.data(), 0, lv.size() * sizeof(float));
    ggml_backend_tensor_set(xb, xv.data(), 0, xv.size() * sizeof(ggml_fp16_t));

    std::vector<float> out_on(sh.n_embd * sh.nt), out_off(sh.n_embd * sh.nt), out_pert(sh.n_embd * sh.nt);
    auto run = [&](const char * knob, std::vector<float> & out, uint64_t & n) {
        set_vk_knob("GGML_VK_HC_UP_MIX", knob);
        g_reset();
        ggml_backend_graph_compute(be, gf);
        n = g_count(sh.gate16 ? "HC_UP_MIX" : "HC_UP_MIX_F32G");
        ggml_backend_tensor_get(mixed, out.data(), 0, nb_out);
    };
    uint64_t n_on = 0, n_off = 0, n_p = 0;
    run("1", out_on, n_on);
    run("0", out_off, n_off);
    int fails = 0;
    const bool eq = memcmp(out_on.data(), out_off.data(), nb_out) == 0;
    size_t ndiff = 0; for (size_t i = 0; i < out_on.size(); ++i) ndiff += memcmp((const char *) out_on.data() + i * es, (const char *) out_off.data() + i * es, es) != 0;
    if (!eq) { fprintf(stderr, "FAIL n_embd=%lld nt=%lld k=%lld pad=%lld: fused != unfused in %zu of %zu values\n",
                       (long long) sh.n_embd, (long long) sh.nt, (long long) sh.k, (long long) sh.pad, ndiff, out_on.size()); fails++; }
    const bool want_fused = sh.expect_fused && fusion;
    if (n_off != 0 || n_on != (want_fused ? 1u : 0u)) {
        fprintf(stderr, "FAIL n_embd=%lld nt=%lld: HC_UP_MIX dispatches on=%llu off=%llu (expected %d / 0)\n",
                (long long) sh.n_embd, (long long) sh.nt, (unsigned long long) n_on, (unsigned long long) n_off, want_fused ? 1 : 0); fails++; }
    // positive control: one perturbed input must change the fused output (the comparison can fail)
    lv[0] += 0.5f; ggml_backend_tensor_set(lo, lv.data(), 0, lv.size() * sizeof(float));
    run("1", out_pert, n_p);
    if (memcmp(out_on.data(), out_pert.data(), nb_out) == 0) { fprintf(stderr, "FAIL: positive control, a perturbed input left the output unchanged\n"); fails++; }
    printf("%s %s%s n_embd=%lld nt=%lld k=%lld pad=%lld: byte-equal=%s fused dispatches on/off=%llu/%llu control=%s\n", fails ? "FAIL" : "ok", sh.gate16 ? "f16-gate" : "f32-gate", sh.mix16 ? " f16-mix" : "",
           (long long) sh.n_embd, (long long) sh.nt, (long long) sh.k, (long long) sh.pad, eq ? "yes" : "NO",
           (unsigned long long) n_on, (unsigned long long) n_off, memcmp(out_on.data(), out_pert.data(), nb_out) ? "differs" : "SAME");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return fails;
}

static int run(ggml_backend_t be, bool fusion) {
    const shape shapes[] = {
        {   64,   64,  96,   0, true },
        {   64,  130,  96, 128, true },    // ragged N, padded x rows
        { 2560, 2048, 320, 128, true },    // Flash-Next: hc_dim 10240, ub 2048, low rank 320, LLAMA_HC_XN_PAD 128
        { 2560, 2051, 320, 128, true },    // the 2051 tail
        {  128,    8,  96,   0, false },   // below the MUL_MAT_CPY16 shapes: unfused both ways, still byte-equal
        // the f32-gate form (LLAMA_HC_GATE_F16=0): the unfused gate is the f32 accumulator, no rounding in the epilogue
        {   64,  130,  96, 128, true, false },
        { 2560, 2048, 320, 128, true, false },
        { 2560, 2051, 320, 128, true, false },
        {  128,    8,  96,   0, false, false },
        // the f16 mixed stream: both gate forms; the fused store and the unfused op both round once (RTE)
        {   64,  130,  96, 128, true, false, true },
        { 2560, 2051, 320, 128, true, false, true },
        {   64,  130,  96, 128, true, true,  true },
        { 2560, 2051, 320, 128, true, true,  true },
    };
    int fails = 0;
    for (const auto & sh : shapes) fails += run_shape(be, sh, fusion);
    printf("hc-up-mix: %s (%d failing checks)\n", fails ? "FAIL" : "OK", fails);
    return fails;
}

} // namespace hc_up_mix

// ---------------------------------------------------------------------------------------------------------------------
// mmid-frag: MUL_MAT_ID output hashes and pf / _mf / REG_A path counts, compared across a default, a
// GGML_VK_MMID_MANUAL_FRAG=0 and a GGML_VK_TEST_MF_SKEW=1 process: default == off, skew != off where _mf ran, mf == pf

namespace mmid_frag {


static std::vector<float> randv(size_t n, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto & x : v) x = u(rng);
    return v;
}

enum epi { EPI_F32, EPI_CPY16, EPI_MUL_CPY16 };

// a: [k, m, n_exp] experts; b: [k, n_used or 1 (bcast), n]; ids: [n_used, n] distinct experts per token
static void run_case(ggml_backend_t be, ggml_type ta, ggml_type tb, int64_t m, int64_t k, int n_exp, int n_used, int64_t n,
                     bool bcast, epi e, int64_t pad_b) {
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int64_t nb_rows = bcast ? 1 : n_used;
    ggml_tensor * a = ggml_new_tensor_3d(ctx, ta, k, m, n_exp);
    // padded B rows: B is a view of a [k + pad_b, ...] tensor (the row stride a padded activation buffer gives)
    ggml_tensor * bb = ggml_new_tensor_3d(ctx, tb, k + pad_b, nb_rows, n);
    ggml_tensor * b = pad_b ? ggml_view_3d(ctx, bb, k, nb_rows, n, bb->nb[1], bb->nb[2], 0) : bb;
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n);
    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, n);
    ggml_tensor * out = ggml_mul_mat_id(ctx, a, b, ids);
    if (e == EPI_MUL_CPY16) {
        out = ggml_mul(ctx, out, w);
    }
    if (e != EPI_F32) {
        out = ggml_cast(ctx, out, GGML_TYPE_F16);
    }
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    std::mt19937 rng(1234u + (uint32_t) ta * 7u + (uint32_t) m + (uint32_t) k + (uint32_t) n);
    set_vals(a, randv(ggml_nelements(a), rng));
    set_vals(bb, randv(ggml_nelements(bb), rng));
    set_vals(w, randv(ggml_nelements(w), rng));
    std::vector<int32_t> id((size_t) n_used * n);
    std::vector<int> perm(n_exp);
    for (int64_t t = 0; t < n; ++t) {
        for (int i = 0; i < n_exp; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int u = 0; u < n_used; ++u) id[(size_t) t * n_used + u] = perm[u];
    }
    ggml_backend_tensor_set(ids, id.data(), 0, id.size() * sizeof(int32_t));

    g_reset();
    ggml_backend_graph_compute(be, gf);
    ggml_backend_synchronize(be);
    std::vector<uint8_t> bytes(ggml_nbytes(out));
    ggml_backend_tensor_get(out, bytes.data(), 0, bytes.size());
    const char * en = e == EPI_F32 ? "f32" : e == EPI_CPY16 ? "cpy16" : "mul_cpy16";
    printf("case %s_x_%s m%lld k%lld E%d u%d n%lld %s %s pad%lld hash=%016llx all=%llu pf=%llu mf=%llu rega=%llu rega_mf=%llu\n",
           ggml_type_name(ta), ggml_type_name(tb), (long long) m, (long long) k, n_exp, n_used, (long long) n, bcast ? "bcast" : "rows", en,
           (long long) pad_b, (unsigned long long) fnv1a(bytes.data(), bytes.size()),
           (unsigned long long) g_count("path:MUL_MAT_ID_ALL"), (unsigned long long) g_count("path:MUL_MAT_ID_PF"),
           (unsigned long long) g_count("path:MUL_MAT_ID_MANUAL_FRAG"),
           (unsigned long long) g_count("path:MUL_MAT_ID_REGA"), (unsigned long long) g_count("path:MUL_MAT_ID_REGA_MF"));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static int run(ggml_backend_t be, bool fusion) {
    GGML_UNUSED(fusion);
    printf("mmid-frag: GGML_VK_MMID_MANUAL_FRAG=%s GGML_VK_TEST_MF_SKEW=%s\n",
           getenv("GGML_VK_MMID_MANUAL_FRAG") ? getenv("GGML_VK_MMID_MANUAL_FRAG") : "(default)",
           getenv("GGML_VK_TEST_MF_SKEW") ? getenv("GGML_VK_TEST_MF_SKEW") : "0");
    // Flash-Next experts: gate/up iq3_xxs [2560 -> 640], down iq4_nl [640 -> 2560], 320 of them top-10 at ub 2048 (64 rows per
    // expert: the REG_A medium tile), B broadcast for gate/up; q8_0 and the K-quants as the other pf types
    for (ggml_type tb : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
        run_case(be, GGML_TYPE_IQ3_XXS, tb, 640, 2560, 320, 10, 2048, true,  EPI_F32,       0);
        run_case(be, GGML_TYPE_IQ3_XXS, tb, 640, 2560, 320, 10, 2048, true,  EPI_CPY16,     0);
        run_case(be, GGML_TYPE_IQ4_NL,  tb, 2560, 640, 320, 10, 2048, false, EPI_MUL_CPY16, 0);
        for (int64_t n : { 2048, 130 }) {
            run_case(be, GGML_TYPE_Q8_0,    tb, 640, 2560, 32, 10, n, true,  EPI_F32,       0);
            run_case(be, GGML_TYPE_IQ3_XXS, tb, 640, 2560, 32, 10, n, false, EPI_F32,       0);
            run_case(be, GGML_TYPE_IQ4_NL,  tb, 2560, 640, 32, 10, n, false, EPI_MUL_CPY16, 0);
            run_case(be, GGML_TYPE_Q4_K,    tb, 768, 2560, 32, 10, n, true,  EPI_F32,       0);
            run_case(be, GGML_TYPE_Q6_K,    tb, 2560, 768, 32, 10, n, false, EPI_MUL_CPY16, 0);
        }
        // padded B rows, a k off the 64 alignment (unaligned pipeline: no _mf build, must be unchanged too)
        run_case(be, GGML_TYPE_IQ3_XXS, tb, 640, 2560, 32, 10, 512, false, EPI_F32, 64);
        run_case(be, GGML_TYPE_Q8_0,    tb, 640, 2592, 32, 10, 512, false, EPI_F32, 0);
    }
    printf("mmid-frag: done (compare three runs with test-vk-exact-mmid-frag.py)\n");
    return 0;
}

} // namespace mmid_frag

// the groups; mmid_f16coopmat: the GGML_VK_MMID_F16COOPMAT default the group sets (read once per process, kept if set)
struct exact_group {
    const char * name;
    bool         by_default;
    const char * mmid_f16coopmat;
    int       (* run)(ggml_backend_t be, bool fusion);
};

static const exact_group groups[] = {
    // the routed f16-B pf path (mul_mm.comp MMID_SCALE d16) for iq3_xxs (no int8 MMQ pipeline) and q6_K
    { "f16-store", true,  "iq3_xxs,q6_K", f16_store::run },
    { "hc-up-mix", true,  nullptr,        hc_up_mix::run },
    // every type here reaches the pipelines that have an "_mf" build
    { "mmid-frag", false, "all",          mmid_frag::run },
};

int main(int argc, char ** argv) {
    std::vector<const exact_group *> sel;
    for (int i = 1; i < argc; ++i) {
        const exact_group * g = nullptr;
        for (const auto & e : groups) {
            if (strcmp(argv[i], e.name) == 0) g = &e;
        }
        if (!g) {
            fprintf(stderr, "usage: %s [f16-store] [hc-up-mix] [mmid-frag]\n", argv[0]);
            return 1;
        }
        sel.push_back(g);
    }
    if (sel.empty()) {
        for (const auto & e : groups) {
            if (e.by_default) sel.push_back(&e);
        }
    }
    const char * f16coopmat = nullptr;
    for (const auto * g : sel) {
        if (g->mmid_f16coopmat && f16coopmat && strcmp(g->mmid_f16coopmat, f16coopmat) != 0) {
            fprintf(stderr, "test-vk-exact: %s needs its own process (GGML_VK_MMID_F16COOPMAT=%s)\n", g->name, g->mmid_f16coopmat);
            return 1;
        }
        f16coopmat = g->mmid_f16coopmat ? g->mmid_f16coopmat : f16coopmat;
    }
    if (f16coopmat) {
        set_env("GGML_VK_MMID_F16COOPMAT", f16coopmat, false);
    }

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("Vulkan0");
    if (!dev) {
        printf("test-vk-exact: no Vulkan0 device, skipped\n");
        return 0;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    g_count = (fusion_count_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_dispatch_census_count");
    g_reset = (fusion_reset_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_dispatch_census_reset");
    g_env_reload = (void (*)(void)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_env_reload");
    if (!g_count || !g_reset) {
        printf("FAIL: no dispatch census counters\n");
        return 1;
    }
    const bool fusion = getenv("GGML_VK_DISABLE_FUSION") == nullptr;
    printf("test-vk-exact: GGML_VK_MMID_F16COOPMAT=%s%s\n", getenv("GGML_VK_MMID_F16COOPMAT") ? getenv("GGML_VK_MMID_F16COOPMAT") : "(default)",
           fusion ? "" : " GGML_VK_DISABLE_FUSION");

    int fails = 0;
    for (const auto * g : sel) {
        fails += g->run(be, fusion);
    }
    ggml_backend_free(be);
    printf("test-vk-exact: %s (%d failing checks)\n", fails ? "FAIL" : "OK", fails);
    return fails ? 1 : 0;
}
