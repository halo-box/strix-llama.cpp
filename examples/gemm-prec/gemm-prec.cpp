// gemm-prec: time and precision of one MUL_MAT / MUL_MAT_ID shape on the first non-CPU backend vs the CPU f32 result.
//   llama-gemm-prec <type_a> <m> <n> <k> [n_expert n_used]   (type_b f32, or f16 with GEMM_PREC_B16=1)
// prints: shape, us/run (min of iters; GEMM_PREC_SUSTAIN=<n> adds the mean of n back-to-back calls), max-abs-error, NMSE and the max-abs of the reference. Used for the GGML_VK_F32ACC A/B
// (run twice with the env set differently) so the PR carries a precision delta vs CPU, not only speed.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static ggml_type type_of(const char * s) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) { const char * n = ggml_type_name((ggml_type) t); if (n && strcmp(n, s) == 0) return (ggml_type) t; }
    fprintf(stderr, "unknown type %s\n", s); exit(1);
}
static std::vector<float> randf(size_t n, float scale, uint32_t seed) { std::mt19937 r(seed); std::normal_distribution<float> d(0.f, scale); std::vector<float> v(n); for (auto & x : v) x = d(r); return v; }
static void set_quant(ggml_tensor * t, const std::vector<float> & f) {
    if (t->type == GGML_TYPE_F32) { ggml_backend_tensor_set(t, f.data(), 0, f.size() * 4); return; }
    if (t->type == GGML_TYPE_F16) { std::vector<ggml_fp16_t> h(f.size()); for (size_t i = 0; i < f.size(); i++) h[i] = ggml_fp32_to_fp16(f[i]); ggml_backend_tensor_set(t, h.data(), 0, h.size() * 2); return; }
    std::vector<uint8_t> q(ggml_nbytes(t)); ggml_quantize_chunk(t->type, f.data(), q.data(), 0, f.size() / t->ne[0], t->ne[0], nullptr); ggml_backend_tensor_set(t, q.data(), 0, q.size());
}
struct run_t { std::vector<float> out; double us; double us_sus = 0; int n_sus = 0; };
static run_t run_on(ggml_backend_t be, ggml_type ta, ggml_type tb, int m, int n, int k, int ne_exp, int n_used,
                    const std::vector<float> & wa, const std::vector<float> & xb, const std::vector<int32_t> & ids, int iters) {
    ggml_init_params wp = { ggml_tensor_overhead() * 8, nullptr, true }; ggml_context * ctx = ggml_init(wp);
    ggml_tensor * a = ne_exp ? ggml_new_tensor_3d(ctx, ta, k, m, ne_exp) : ggml_new_tensor_2d(ctx, ta, k, m);
    ggml_tensor * b = ne_exp ? ggml_new_tensor_3d(ctx, tb, k, n_used, n) : ggml_new_tensor_2d(ctx, tb, k, n);
    ggml_tensor * id = ne_exp ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n) : nullptr;
    ggml_tensor * out = ne_exp ? ggml_mul_mat_id(ctx, a, b, id) : ggml_mul_mat(ctx, a, b);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    set_quant(a, wa); set_quant(b, xb); if (id) ggml_backend_tensor_set(id, ids.data(), 0, ids.size() * 4);
    ggml_init_params gp = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true }; ggml_context * gctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(gctx); ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be)); ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_graph_compute(be, gf); ggml_backend_synchronize(be);
    double best = 1e30;
    for (int i = 0; i < iters; i++) { auto t0 = std::chrono::steady_clock::now(); ggml_backend_graph_compute(be, gf); ggml_backend_synchronize(be);
        best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count()); }
    run_t r; r.us = best;
    // GEMM_PREC_SUSTAIN=<n>: n back-to-back computes with one sync at the end, mean per call. The min above is one burst at the
    // boost clock; a compute-dense tile (XL 256x256) is measured here at the clock the model sees under sustained load.
    static const int sus = [] { const char * e = getenv("GEMM_PREC_SUSTAIN"); return e ? atoi(e) : 0; }();
    if (sus > 0 && iters > 1) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < sus; i++) { ggml_backend_graph_compute(be, gf); }
        ggml_backend_synchronize(be);
        r.us_sus = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / sus; r.n_sus = sus;
    }
    r.out.resize(ggml_nelements(out)); ggml_backend_tensor_get(out, r.out.data(), 0, r.out.size() * 4);
    ggml_gallocr_free(ga); ggml_free(gctx); ggml_backend_buffer_free(buf); ggml_free(ctx); return r;
}
int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <type_a> <m> <n> <k> [n_expert n_used]\n", argv[0]); return 1; }
    ggml_type ta = type_of(argv[1]); int m = atoi(argv[2]), n = atoi(argv[3]), k = atoi(argv[4]);
    int ne_exp = argc > 6 ? atoi(argv[5]) : 0, n_used = argc > 6 ? atoi(argv[6]) : 0;
    ggml_type tb = getenv("GEMM_PREC_B16") ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_backend_load_all();
    ggml_backend_t gpu = nullptr;   // the 8060S registers as an integrated GPU: first non-CPU device
    for (size_t i = 0; i < ggml_backend_dev_count() && !gpu; i++) { auto d = ggml_backend_dev_get(i); if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_CPU) gpu = ggml_backend_dev_init(d, nullptr); }
    ggml_backend_t cpu = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(cpu, 16);
    if (!gpu) { fprintf(stderr, "no GPU backend\n"); return 1; }
    const size_t na = (size_t) k * m * (ne_exp ? ne_exp : 1), nb = (size_t) k * n * (ne_exp ? n_used : 1);
    auto wa = randf(na, 0.05f, 1), xb = randf(nb, 1.0f, 2);
    std::vector<int32_t> ids; if (ne_exp) { std::mt19937 r(3); ids.resize((size_t) n_used * n); for (int t = 0; t < n; t++) { std::vector<int> p(ne_exp); for (int e = 0; e < ne_exp; e++) p[e] = e; std::shuffle(p.begin(), p.end(), r); for (int u = 0; u < n_used; u++) ids[(size_t) t * n_used + u] = p[u]; } }
    // the CPU reference always takes an f32 B (ggml-cpu asserts src1 f32 for a quantised A with an f16 B); for the f16-B arm the
    // Vulkan side gets the f16-rounded values and the reference gets the same values as f32, so the error is the backend's alone
    std::vector<float> xb_ref = xb;
    if (tb == GGML_TYPE_F16) { for (auto & v : xb_ref) v = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)); }
    run_t ref = run_on(cpu, ta, GGML_TYPE_F32, m, n, k, ne_exp, n_used, wa, xb_ref, ids, 1);
    run_t dev = run_on(gpu, ta, tb, m, n, k, ne_exp, n_used, wa, xb, ids, 10);
    double maxabs = 0, maxref = 0, num = 0, den = 0;
    for (size_t i = 0; i < ref.out.size(); i++) { double d = (double) dev.out[i] - ref.out[i]; maxabs = std::max(maxabs, std::fabs(d)); maxref = std::max(maxref, (double) std::fabs(ref.out[i])); num += d * d; den += (double) ref.out[i] * ref.out[i]; }
    if (dev.n_sus) printf("gemm-prec sustained(%d back-to-back): %.1f us/run mean (min-of-iters %.1f, ratio %.3f)\n", dev.n_sus, dev.us_sus, dev.us, dev.us_sus / dev.us);
    printf("gemm-prec %s %s x %s m=%d n=%d k=%d%s: %.1f us/run  max_abs_err %.3e  (max|ref| %.3e)  NMSE %.3e  F32ACC=%s\n", ne_exp ? "MUL_MAT_ID" : "MUL_MAT", ggml_type_name(ta), ggml_type_name(tb), m, n, k,
           ne_exp ? (std::string(" E") + std::to_string(ne_exp) + " used " + std::to_string(n_used)).c_str() : "", dev.us, maxabs, maxref, num / std::max(den, 1e-30), getenv("GGML_VK_F32ACC") ? getenv("GGML_VK_F32ACC") : "default");
    ggml_backend_free(gpu); ggml_backend_free(cpu); return 0;
}
