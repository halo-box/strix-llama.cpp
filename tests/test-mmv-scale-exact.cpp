// MUL_MAT_VEC_SCALE exactness (wave 8b): the Vulkan backend applies a SCALE that follows a one-column quantised mat-vec in the mat-vec's
// store (qwen4exp's hc mix: silu(scale(W_down * xn, 1/hc)), 96 per decode token on Flash-Next). It must be BIT-EXACT with the separate
// SCALE. Each case runs with GGML_VK_MMV_SCALE=1 and =0 (re-read through the knob reset hook), requires byte-equal outputs and the expected fused-dispatch
// counts (cases the fusion must decline expect 0 both ways), and checks that a scale one f32 ulp away changes the output (positive
// control: the comparison sees a single rounding; for an f16 output, 2^-10 relative, the smallest step the f16 store cannot absorb).
// Skips (exit 0) when there is no Vulkan device.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
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

// chain: 0 = SCALE -> SILU (the hc mix); 1 = SCALE -> CPY f16 (the scaled result stored as f16 after the f32 SCALE); 2 = the mat-vec
// result cast to f16 and back before the SCALE (a mat-vec feeding a CPY16: not a MUL_MAT -> SCALE pair, must not fuse)
struct shape { ggml_type ta; int64_t m, k, n; ggml_type tb; float scale, bias; bool expect_fused; const char * what; int chain = 0; };

static std::vector<uint8_t> run(ggml_backend_t be, ggml_backend_reg_t reg, const shape & sh, float scale, const char * knob, uint64_t & n_fused,
                                const std::vector<uint8_t> & wq, const std::vector<float> & xv) {
    auto count = (census_count_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_dispatch_census_count");
    auto reset = (census_reset_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_dispatch_census_reset");
    auto env_reload = (env_reload_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_env_reload");
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, sh.ta, sh.k, sh.m);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, sh.tb, sh.k, sh.n);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (sh.chain == 2) {
        y = ggml_cast(ctx, ggml_cast(ctx, y, GGML_TYPE_F16), GGML_TYPE_F32);
    }
    y = sh.bias != 0.0f ? ggml_scale_bias(ctx, y, scale, sh.bias) : ggml_scale(ctx, y, scale);
    y = sh.chain == 1 ? ggml_cast(ctx, y, GGML_TYPE_F16) : ggml_silu(ctx, y);   // the hc mix's consumer, or an f16 store
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "FAIL: alloc\n"); exit(1); }
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    if (sh.tb == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(xv.size());
        for (size_t i = 0; i < xv.size(); ++i) h[i] = ggml_fp32_to_fp16(xv[i]);
        ggml_backend_tensor_set(x, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
    }
    set_env("GGML_VK_MMV_SCALE", knob);
    if (env_reload) env_reload();
    if (reset) reset();
    ggml_backend_graph_compute(be, gf);
    n_fused = count ? count("MUL_MAT_VEC_SCALE") : 0;
    std::vector<uint8_t> res(ggml_nbytes(y));
    ggml_backend_tensor_get(y, res.data(), 0, res.size());
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
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
    const bool fusion = getenv("GGML_VK_DISABLE_FUSION") == nullptr;
    // a device that cannot run the fused path at all skips (the backend says why); with fusion disabled the cases still run
    auto census_skip = (census_skip_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_dispatch_census_skip");
    const char * why = census_skip ? census_skip(be, "MUL_MAT_VEC_SCALE") : nullptr;
    if (why != nullptr && fusion) {
        printf("MUL_MAT_VEC_SCALE not available on %s (%s), skipping\n", ggml_backend_dev_name(dev), why);
        ggml_backend_free(be);
        return 0;
    }

    const shape shapes[] = {
        { GGML_TYPE_Q8_0, 320, 10240, 1, GGML_TYPE_F32, 0.25f, 0.0f, true,  "FN hc down, decode" },
        { GGML_TYPE_Q8_0, 320, 10240, 1, GGML_TYPE_F32, 0.3f,  0.0f, true,  "non power-of-two scale" },
        { GGML_TYPE_Q8_0, 320, 10240, 1, GGML_TYPE_F16, 0.25f, 0.0f, true,  "f16 B" },
        { GGML_TYPE_Q4_K, 640, 2560,  1, GGML_TYPE_F32, 0.25f, 0.0f, true,  "k-quant" },
        { GGML_TYPE_Q8_0, 4099, 512,  1, GGML_TYPE_F32, 0.25f, 0.0f, true,  "odd row count" },
        { GGML_TYPE_Q8_0, 320, 10240, 2, GGML_TYPE_F32, 0.25f, 0.0f, false, "two columns: declined" },
        { GGML_TYPE_Q8_0, 320, 10240, 1, GGML_TYPE_F32, 0.25f, 0.5f, false, "scale_bias: declined" },
        { GGML_TYPE_F16,  320, 10240, 1, GGML_TYPE_F32, 0.25f, 0.0f, false, "f16 weight: declined" },
        // f16 outputs: ggml_mul_mat's dst is always f32, so an f16 result only appears after a CPY. Scaling first and casting after
        // fuses (the cast reads the f32 scaled value either way); a mat-vec feeding a CPY16 before the SCALE is not a pair at all
        { GGML_TYPE_Q8_0, 320, 10240, 1, GGML_TYPE_F32, 0.25f, 0.0f, true,  "SCALE -> CPY f16", 1 },
        { GGML_TYPE_Q8_0, 320, 10240, 1, GGML_TYPE_F32, 0.25f, 0.0f, false, "mat-vec -> CPY16 -> SCALE", 2 },
        // large K: the mat-vec never splits K across workgroups (one workgroup per row block reduces all of K: mul_mat_vec_base.glsl
        // reduce_result, subgroupAdd then the shared-memory sum; the only other n=1 scratch path, transpose_d, is excluded once fused)
        { GGML_TYPE_Q8_0, 320, 16384, 1, GGML_TYPE_F32, 0.25f, 0.0f, true,  "K 16384" },
        { GGML_TYPE_Q8_0, 64,  65536, 1, GGML_TYPE_F32, 0.25f, 0.0f, true,  "K 65536" },
    };
    int fails = 0;
    for (const auto & sh : shapes) {
        std::mt19937 rng(11 + (int) sh.m + (int) sh.n * 3 + (int) sh.tb);
        std::normal_distribution<float> dw(0.f, 0.05f), dx(0.f, 1.f);
        std::vector<float> wf(sh.k * sh.m), xv(sh.k * sh.n);
        for (auto & v : wf) v = dw(rng);
        for (auto & v : xv) v = dx(rng);
        std::vector<uint8_t> wq(ggml_row_size(sh.ta, sh.k) * sh.m);
        ggml_quantize_chunk(sh.ta, wf.data(), wq.data(), 0, sh.m, sh.k, nullptr);
        uint64_t n_on = 0, n_off = 0, n_ulp = 0;
        const auto on  = run(be, reg, sh, sh.scale, "1", n_on, wq, xv);
        const auto off = run(be, reg, sh, sh.scale, "0", n_off, wq, xv);
        // the control's scale: one f32 ulp away, or for an f16 output the smallest step the f16 rounding cannot absorb (2^-10 relative)
        const float s_ctl = sh.chain == 1 ? sh.scale * (1.0f + 1.0f / 1024) : std::nextafter(sh.scale, 1.0f);
        const auto ulp = run(be, reg, sh, s_ctl, "1", n_ulp, wq, xv);
        const size_t es = sh.chain == 1 ? 2 : 4;   // compare per output element
        size_t nd = 0, nu = 0;
        for (size_t i = 0; i < on.size(); i += es) { nd += memcmp(on.data() + i, off.data() + i, es) != 0; nu += memcmp(off.data() + i, ulp.data() + i, es) != 0; }
        const uint64_t want = sh.expect_fused && fusion ? 1 : 0;
        printf("%-24s %s m=%lld k=%lld n=%lld B=%s scale=%g bias=%g: fused vs unfused %s (%zu of %zu differ); MUL_MAT_VEC_SCALE on=%llu off=%llu "
               "(want %llu/0); scale control (%s) %zu differ\n", sh.what, ggml_type_name(sh.ta), (long long) sh.m, (long long) sh.k, (long long) sh.n,
               ggml_type_name(sh.tb), sh.scale, sh.bias, nd ? "DIFFER" : "byte-equal", nd, on.size() / es, (unsigned long long) n_on,
               (unsigned long long) n_off, (unsigned long long) want, sh.chain == 1 ? "x(1+2^-10), f16 output" : "1 f32 ulp", nu);
        if (nd) { fprintf(stderr, "FAIL %s: outputs differ\n", sh.what); fails++; }
        if (n_on != want || n_off != 0) { fprintf(stderr, "FAIL %s: dispatch counts\n", sh.what); fails++; }
        if (!nu) { fprintf(stderr, "FAIL %s: scale control unchanged\n", sh.what); fails++; }
    }
    ggml_backend_free(be);
    printf("%s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
