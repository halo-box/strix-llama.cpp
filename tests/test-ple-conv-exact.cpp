// LLAMA_PLE_CONV_FUSED exactness (wave 8, C5): qwen4exp's PLE layer runs a dilated depthwise conv (4 taps, dilation 3, a 9-token
// history), SiLU, and out = hidden + (gated + conv). The original graph prepends the history token-major and transposes every tap
// (CONT per tap); the C5 graph prepends it channels first and reads every tap as a row view, and the Vulkan backend fuses the ten
// tail nodes as PLE_CONV_SILU_ADD. On the Vulkan backend, per shape: the C5 graph with GGML_VK_PLE_CONV=1 (fused, one dispatch),
// with =0 (unfused), and the original formulation must all be byte-equal. Positive control: the C5 graph with the tap sum in the
// reverse order must differ (so the comparison sees a reassociation). Skips (exit 0) without a Vulkan device.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// setenv / unsetenv are POSIX only; _putenv_s with an empty value removes the variable on Windows
static void set_env(const char * name, const char * value) {
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

typedef uint64_t (*census_count_t)(const char *);
typedef void     (*census_reset_t)(void);
typedef void     (*env_reload_t)(void);   // the backend reads its knobs once: re-read them after a change
typedef const char * (*census_skip_t)(ggml_backend_t, const char *);

static const int64_t K = 4, DIL = 3, HIST = (K - 1) * DIL;

struct shape { int64_t c, t, s; };

enum variant { C5, C5_REVERSED, ORIGINAL };

struct inputs { std::vector<float> p, w, g, h; };   // p: channels first [c, HIST + t, s]; w: [K, c] as the file holds it

static std::vector<uint8_t> run(ggml_backend_t be, ggml_backend_reg_t reg, const shape & sh, const inputs & in, variant v,
                                const char * knob, uint64_t & n_fused) {
    auto count = (census_count_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_dispatch_census_count");
    auto reset = (census_reset_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_dispatch_census_reset");
    auto env_reload = (env_reload_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_env_reload");
    const int64_t C = sh.c, T = sh.t, S = sh.s, R = HIST + T;

    ggml_init_params ip = { 96 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, C);
    ggml_tensor * g  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, T, S);
    ggml_tensor * h  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, T, S);
    ggml_tensor * p  = v == ORIGINAL ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, R, C, S) : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, R, S);
    ggml_tensor * pin = ggml_scale(ctx, p, 1.0f);   // a computed padded buffer, as the model's concat is
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * out;
    if (v == ORIGINAL) {
        ggml_tensor * conv = nullptr;
        for (int64_t k = 0; k < K; ++k) {
            const int64_t start = HIST - (K - 1 - k) * DIL;
            ggml_tensor * sh_k = ggml_cont(ctx, ggml_transpose(ctx, ggml_view_3d(ctx, pin, T, C, S, pin->nb[1], pin->nb[2], start * sizeof(float))));
            ggml_tensor * wk = ggml_reshape_1d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, w, 1, C, w->nb[1], k * w->nb[0])), C);
            ggml_tensor * term = ggml_mul(ctx, sh_k, wk);
            conv = conv ? ggml_add(ctx, conv, term) : term;
        }
        conv = ggml_cont(ctx, ggml_silu(ctx, conv));
        out = ggml_add(ctx, h, ggml_add(ctx, g, conv));
    } else {
        ggml_tensor * w_cf = ggml_cont(ctx, ggml_transpose(ctx, w));
        std::vector<ggml_tensor *> taps, wks;
        for (int64_t k = 0; k < K; ++k) {
            const int64_t start = HIST - (K - 1 - k) * DIL;
            taps.push_back(ggml_view_3d(ctx, pin, C, T, S, pin->nb[1], pin->nb[2], start * pin->nb[1]));
            wks.push_back(ggml_view_1d(ctx, w_cf, C, k * w_cf->nb[1]));
            ggml_build_forward_expand(gf, taps.back());
            ggml_build_forward_expand(gf, wks.back());
        }
        ggml_tensor * conv = nullptr;
        for (int64_t i = 0; i < K; ++i) {
            const int64_t k = v == C5_REVERSED ? K - 1 - i : i;
            ggml_tensor * term = ggml_mul(ctx, taps[k], wks[k]);
            conv = conv ? ggml_add(ctx, conv, term) : term;
        }
        out = ggml_add(ctx, h, ggml_add(ctx, g, ggml_silu(ctx, conv)));
    }
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "FAIL: alloc\n"); exit(1); }

    std::vector<float> pv(in.p.size());
    if (v == ORIGINAL) {   // token-major copy of the same padded values
        for (int64_t s = 0; s < S; ++s) for (int64_t r = 0; r < R; ++r) for (int64_t c = 0; c < C; ++c) {
            pv[(s * C + c) * R + r] = in.p[(s * R + r) * C + c];
        }
    } else {
        pv = in.p;
    }
    ggml_backend_tensor_set(p, pv.data(), 0, pv.size() * sizeof(float));
    ggml_backend_tensor_set(w, in.w.data(), 0, in.w.size() * sizeof(float));
    ggml_backend_tensor_set(g, in.g.data(), 0, in.g.size() * sizeof(float));
    ggml_backend_tensor_set(h, in.h.data(), 0, in.h.size() * sizeof(float));
    set_env("GGML_VK_PLE_CONV", knob);
    if (env_reload) env_reload();
    if (reset) reset();
    ggml_backend_graph_compute(be, gf);
    n_fused = count ? count("PLE_CONV_SILU_ADD") : 0;
    std::vector<uint8_t> res(ggml_nbytes(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size());
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

static size_t ndiff(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); i += 4) n += memcmp(a.data() + i, b.data() + i, 4) != 0;
    return n;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t di = ggml_backend_dev_get(i);
        if (strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(di)), "Vulkan") == 0) { dev = di; break; }
    }
    if (!dev) { printf("no Vulkan device, skipping\n"); return 0; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const bool want_fused = getenv("GGML_VK_DISABLE_FUSION") == nullptr;
    // a device that cannot run the fused path at all skips (the backend says why); with fusion disabled the cases still run
    auto census_skip = (census_skip_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_dispatch_census_skip");
    const char * why = census_skip ? census_skip(be, "PLE_CONV_SILU_ADD") : nullptr;
    if (why != nullptr && want_fused) {
        printf("PLE_CONV_SILU_ADD not available on %s (%s), skipping\n", ggml_backend_dev_name(dev), why);
        ggml_backend_free(be);
        return 0;
    }

    // Flash-Next hc_dim 10240: decode, a small batch, a prefill ubatch; two sequences; an odd width
    const shape shapes[] = { { 10240, 1, 1 }, { 10240, 7, 1 }, { 10240, 2048, 1 }, { 10240, 33, 2 }, { 1000, 5, 3 } };
    int fails = 0;
    for (const auto & sh : shapes) {
        std::mt19937 rng(5 + (int) sh.t * 7 + (int) sh.s);
        std::uniform_real_distribution<float> u(-2.0f, 2.0f);
        inputs in;
        in.p.resize(sh.c * (HIST + sh.t) * sh.s); in.w.resize(K * sh.c); in.g.resize(sh.c * sh.t * sh.s); in.h.resize(in.g.size());
        for (auto * vec : { &in.p, &in.w, &in.g, &in.h }) for (auto & x : *vec) x = u(rng);

        uint64_t n_on = 0, n_off = 0, n_orig = 0, n_rev = 0;
        const auto on   = run(be, reg, sh, in, C5, "1", n_on);
        const auto off  = run(be, reg, sh, in, C5, "0", n_off);
        const auto orig = run(be, reg, sh, in, ORIGINAL, "1", n_orig);
        const auto rev  = run(be, reg, sh, in, C5_REVERSED, "0", n_rev);
        const size_t d_off = ndiff(on, off), d_orig = ndiff(on, orig), d_rev = ndiff(off, rev);
        printf("c=%lld t=%lld s=%lld: fused vs unfused %zu, fused vs original %zu differ (of %zu); PLE_CONV_SILU_ADD on=%llu off=%llu "
               "original=%llu (want %d/0/0); reversed-tap control %zu differ\n",
               (long long) sh.c, (long long) sh.t, (long long) sh.s, d_off, d_orig, on.size() / 4, (unsigned long long) n_on,
               (unsigned long long) n_off, (unsigned long long) n_orig, want_fused ? 1 : 0, d_rev);
        if (d_off || d_orig) { fprintf(stderr, "FAIL t=%lld s=%lld: outputs differ\n", (long long) sh.t, (long long) sh.s); fails++; }
        if (n_on != (want_fused ? 1u : 0u) || n_off != 0 || n_orig != 0) { fprintf(stderr, "FAIL t=%lld: fused dispatch counts\n", (long long) sh.t); fails++; }
        if (!d_rev) { fprintf(stderr, "FAIL t=%lld: reversed-tap control unchanged\n", (long long) sh.t); fails++; }
    }
    ggml_backend_free(be);
    printf("%s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
