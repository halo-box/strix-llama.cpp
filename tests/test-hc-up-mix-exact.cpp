// HC_UP_MIX exactness (wave 5): the Vulkan backend fuses MUL_MAT(q8_0 hc up weight, lo) + CPY(f16) + RESHAPE + DSV4_HC_PRE (gated) into
// one int8 MMQ GEMM whose epilogue applies the gated mix. It must be BIT-EXACT with the unfused chain (the fork's HC_UP_MIX was turned
// off because a non-exact epilogue made cached multi-turn and full re-prefill diverge). Runs each shape with GGML_VK_HC_UP_MIX=1 and =0
// (the backend reads the knob per graph), requires byte-equal mixed outputs and the expected fused-dispatch counts, then perturbs one
// input as a positive control (the output must change). Skips (exit 0) when there is no Vulkan device.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>
#include <vector>

// the Vulkan dispatch census test hooks (off until the first reset)
typedef uint64_t (*census_count_t)(const char *);
typedef void     (*census_reset_t)(void);

struct shape { int64_t n_embd, nt, k, pad; bool expect_fused; bool gate16 = true; bool mix16 = false; };   // gate16 false: the f32-gate form; mix16: the f16 mixed stream (wave 6)

static int run_shape(ggml_backend_t be, ggml_backend_reg_t reg, const shape & sh) {
    auto count = (census_count_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_dispatch_census_count");
    auto reset = (census_reset_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_dispatch_census_reset");
    if (!count || !reset) { fprintf(stderr, "FAIL: no dispatch census counters\n"); return 1; }

    ggml_init_params ip = { 32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int64_t M = 4 * sh.n_embd;
    ggml_tensor * w   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, sh.k, M);
    ggml_tensor * lo  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.k, sh.nt);
    ggml_tensor * xb  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, M + sh.pad, sh.nt);   // row-padded like LLAMA_HC_XN_PAD
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
    std::vector<float> wf(sh.k * M), lv(sh.k * sh.nt);
    for (auto & v : wf) v = u(rng);
    for (auto & v : lv) v = u(rng);
    std::vector<uint8_t> wq(ggml_nbytes(w));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, wf.data(), wq.data(), 0, M, sh.k, nullptr);
    std::vector<ggml_fp16_t> xv((M + sh.pad) * sh.nt);
    for (auto & v : xv) v = ggml_fp32_to_fp16(u(rng));
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(lo, lv.data(), 0, lv.size() * sizeof(float));
    ggml_backend_tensor_set(xb, xv.data(), 0, xv.size() * sizeof(ggml_fp16_t));

    std::vector<float> out_on(sh.n_embd * sh.nt), out_off(sh.n_embd * sh.nt), out_pert(sh.n_embd * sh.nt);
    auto run = [&](const char * knob, std::vector<float> & out, uint64_t & n) {
        setenv("GGML_VK_HC_UP_MIX", knob, 1);
        reset();
        ggml_backend_graph_compute(be, gf);
        n = count(sh.gate16 ? "HC_UP_MIX" : "HC_UP_MIX_F32G");
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
    if (n_off != 0 || n_on != (sh.expect_fused ? 1u : 0u)) {
        fprintf(stderr, "FAIL n_embd=%lld nt=%lld: HC_UP_MIX dispatches on=%llu off=%llu (expected %d / 0)\n",
                (long long) sh.n_embd, (long long) sh.nt, (unsigned long long) n_on, (unsigned long long) n_off, sh.expect_fused ? 1 : 0); fails++; }
    if (!eq && sh.gate16 && !sh.mix16 && getenv("HC_UP_MIX_DIAG") != nullptr) {
        // where the two runs part: the knob-off run left its f16 gate in g16; a separate f32 GEMM gives the accumulator before
        // rounding. host(g16) = the dsv4_hc_pre.comp expression on the host over that gate
        const int64_t ne = M * sh.nt;
        std::vector<ggml_fp16_t> gh(ne);
        ggml_backend_tensor_get(g16, gh.data(), 0, ne * sizeof(ggml_fp16_t));
        ggml_init_params ip2 = { 4 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * c2 = ggml_init(ip2);
        ggml_tensor * w2 = ggml_new_tensor_2d(c2, GGML_TYPE_Q8_0, sh.k, M);
        ggml_tensor * l2 = ggml_new_tensor_2d(c2, GGML_TYPE_F32, sh.k, sh.nt);
        ggml_tensor * gf = ggml_mul_mat(c2, w2, l2);
        ggml_set_output(gf);
        ggml_cgraph * g2 = ggml_new_graph(c2);
        ggml_build_forward_expand(g2, gf);
        ggml_backend_buffer_t b2 = ggml_backend_alloc_ctx_tensors(c2, be);
        ggml_backend_tensor_set(w2, wq.data(), 0, wq.size());
        ggml_backend_tensor_set(l2, lv.data(), 0, lv.size() * sizeof(float));
        ggml_backend_graph_compute(be, g2);
        std::vector<float> g32(ne);
        ggml_backend_tensor_get(gf, g32.data(), 0, ne * sizeof(float));
        size_t eq_rte = 0, eq_rtz = 0;
        for (int64_t i = 0; i < ne; ++i) {
            uint32_t u; memcpy(&u, &g32[i], 4);
            const uint32_t ut = u & 0xffffe000u; float t; memcpy(&t, &ut, 4);   // RTZ for normal f16 range
            eq_rte += ggml_fp32_to_fp16(g32[i]) == gh[i];
            eq_rtz += ggml_fp32_to_fp16(t) == gh[i];
        }
        auto host = [&](int64_t i0, int64_t t, int mode, float * gv) {   // gate: 0 = g16, 1 = RTE(f32 GEMM), 2 = f32 GEMM unrounded
            float r = 0.0f;
            for (int ih = 0; ih < 4; ++ih) {
                const int64_t gi = t * M + ih * sh.n_embd + i0;
                const float g = mode == 2 ? g32[gi] : mode == 1 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(g32[gi])) : ggml_fp16_to_fp32(gh[gi]);
                if (gv) gv[ih] = g;
                const float wv = 1.0f / (1.0f + expf(-g));
                r = fmaf(ggml_fp16_to_fp32(xv[t * (M + sh.pad) + ih * sh.n_embd + i0]), wv, r);
            }
            return 0.25f * r;
        };
        double d_on = 0, d_off = 0, d_unr = 0; size_t on_eq_host = 0, off_eq_host = 0, on_eq_unr = 0;
        for (int64_t t = 0; t < sh.nt; ++t) for (int64_t i0 = 0; i0 < sh.n_embd; ++i0) {
            const float h = host(i0, t, 0, nullptr); const float hu = host(i0, t, 2, nullptr); const size_t o = t * sh.n_embd + i0;
            d_unr = std::max(d_unr, (double) fabsf(out_on[o] - hu)); on_eq_unr += out_on[o] == hu;
            d_on = std::max(d_on, (double) fabsf(out_on[o] - h)); d_off = std::max(d_off, (double) fabsf(out_off[o] - h));
            on_eq_host += out_on[o] == h; off_eq_host += out_off[o] == h;
        }
        printf("DIAG n_embd=%lld nt=%lld: g16 == RTE(f32 GEMM) %zu/%lld, == RTZ %zu/%lld; vs host(g16): on equal %zu max|d| %.3g, off equal %zu max|d| %.3g; on vs host(unrounded gate): equal %zu max|d| %.3g\n",
               (long long) sh.n_embd, (long long) sh.nt, eq_rte, (long long) ne, eq_rtz, (long long) ne, on_eq_host, d_on, off_eq_host, d_off, on_eq_unr, d_unr);
        int shown = 0;
        for (int64_t t = 0; t < sh.nt && shown < 3; ++t) for (int64_t i0 = 0; i0 < sh.n_embd && shown < 3; ++i0) {
            const size_t o = t * sh.n_embd + i0;
            if (memcmp(&out_on[o], &out_off[o], 4) == 0) continue;
            float gv[4], gr[4];
            const float h = host(i0, t, 0, gv); const float hr = host(i0, t, 2, gr);
            printf("DIAG  i0=%lld t=%lld on=%.9g off=%.9g host(g16)=%.9g host(unrounded)=%.9g g16=[%g %g %g %g] f32=[%.9g %.9g %.9g %.9g]\n",
                   (long long) i0, (long long) t, out_on[o], out_off[o], h, hr, gv[0], gv[1], gv[2], gv[3], gr[0], gr[1], gr[2], gr[3]);
            shown++;
        }
        ggml_backend_buffer_free(b2);
        ggml_free(c2);
    }
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

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("Vulkan0");
    if (!dev) { printf("test-hc-up-mix-exact: no Vulkan0 device, skipped\n"); return 0; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
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
        // the f16 mixed stream (wave 6): both gate forms; the fused store and the unfused op both round once (RTE)
        {   64,  130,  96, 128, true, false, true },
        { 2560, 2051, 320, 128, true, false, true },
        {   64,  130,  96, 128, true, true,  true },
        { 2560, 2051, 320, 128, true, true,  true },
    };
    int fails = 0;
    for (const auto & sh : shapes) fails += run_shape(be, reg, sh);
    ggml_backend_free(be);
    printf("test-hc-up-mix-exact: %s (%d failing checks)\n", fails ? "FAIL" : "OK", fails);
    return fails ? 1 : 0;
}
