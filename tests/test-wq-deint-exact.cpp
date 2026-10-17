// LLAMA_WQ_DEINTERLEAVE: the attention gate from a de-interleaved wq (views, SIGMOID_MUL) must be byte-equal with the file order
// (strided view, CONT); swapping two heads must change it. argv[1] dumps the output for a GGML_VK_DISABLE_FUSION=1 comparison.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

typedef uint64_t (*census_count_t)(const char *);
typedef void     (*census_reset_t)(void);
typedef const char * (*census_skip_t)(ggml_backend_t, const char *);

struct shape { int64_t n_embd, d, n_head, nt; };

// rows of w: [q(d) | gate(d)] per head -> [q heads | gate heads]; swap != 0 exchanges the q rows of heads 0 and 1 (control)
static std::vector<uint8_t> deinterleave(const std::vector<uint8_t> & src, size_t row, int64_t d, int64_t n_head, bool swap) {
    std::vector<uint8_t> dst(src.size());
    for (int64_t h = 0; h < n_head; ++h) {
        const int64_t hs = swap && h < 2 ? 1 - h : h;
        memcpy(dst.data() + (size_t) (h * d) * row,            src.data() + (size_t) (2 * hs * d) * row,     (size_t) d * row);
        memcpy(dst.data() + (size_t) ((n_head + h) * d) * row, src.data() + (size_t) ((2 * h + 1) * d) * row, (size_t) d * row);
    }
    return dst;
}

// returns the gated output; n_sigmul gets the SIGMOID_MUL dispatches of the run
static std::vector<uint8_t> run_graph(ggml_backend_t be, ggml_backend_reg_t reg, const shape & sh, bool deint,
                                      const std::vector<uint8_t> & wbytes, const std::vector<float> & xv, uint64_t & n_sigmul) {
    auto count = (census_count_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_dispatch_census_count");
    auto reset = (census_reset_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_dispatch_census_reset");

    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int64_t d = sh.d, H = sh.n_head, T = sh.nt;
    ggml_tensor * w  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, sh.n_embd, 2 * d * H);
    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.n_embd, T);
    ggml_tensor * qf = ggml_mul_mat(ctx, w, x);   // Qcur_full
    const size_t es = ggml_element_size(qf);
    ggml_tensor * q, * gate;
    if (deint) {
        q    = ggml_view_3d(ctx, qf, d, H, T, es * d, qf->nb[1], 0);
        gate = ggml_view_2d(ctx, qf, d * H, T, qf->nb[1], es * d * H);
    } else {
        q    = ggml_view_3d(ctx, qf, d, H, T, es * d * 2, es * d * 2 * H, 0);
        gate = ggml_view_3d(ctx, qf, d, H, T, es * d * 2, es * d * 2 * H, es * d);
        gate = ggml_cont_2d(ctx, gate, d * H, T);
    }
    // the q side stands in for norm -> rope -> attention: a chain the gate does not depend on
    ggml_tensor * qn   = ggml_rms_norm(ctx, q, 1e-6f);
    ggml_tensor * qs   = ggml_scale(ctx, qn, 0.5f);
    ggml_tensor * attn = ggml_soft_max(ctx, ggml_reshape_2d(ctx, qs, d * H, T));
    ggml_tensor * out  = ggml_mul(ctx, attn, ggml_sigmoid(ctx, gate));
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "FAIL: alloc\n"); exit(1); }
    ggml_backend_tensor_set(w, wbytes.data(), 0, wbytes.size());
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
    if (reset) reset();
    ggml_backend_graph_compute(be, gf);
    n_sigmul = count ? count("SIGMOID_MUL") : 0;
    std::vector<uint8_t> res(ggml_nbytes(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size());
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t di = ggml_backend_dev_get(i);
        if (strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(di)), "Vulkan") == 0) { dev = di; break; }
    }
    if (!dev) { printf("no Vulkan device, skipping\n"); return 0; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    // graph_optimize (GGML_VK_UNARY_MUL_KEEP) runs under the scheduler only: here the graph keeps its build order
    const bool want_fused = getenv("GGML_VK_DISABLE_FUSION") == nullptr;
    // a device that cannot run the fused path at all skips (the backend says why); with fusion disabled the cases still run
    auto census_skip = (census_skip_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_dispatch_census_skip");
    const char * why = census_skip ? census_skip(be, "SIGMOID_MUL") : nullptr;
    if (why != nullptr && want_fused) {
        printf("SIGMOID_MUL not available on %s (%s), skipping\n", ggml_backend_dev_name(dev), why);
        ggml_backend_free(be);
        return 0;
    }

    // FN attention (d 256, 12 heads, n_embd 2560); decode, a small batch, and prefill widths
    const shape shapes[] = { { 2560, 256, 12, 1 }, { 2560, 256, 12, 7 }, { 2560, 256, 12, 64 }, { 2560, 256, 12, 512 }, { 512, 128, 4, 33 } };
    FILE * dump = argc > 1 ? fopen(argv[1], "wb") : nullptr;
    int fails = 0;
    for (const auto & sh : shapes) {
        const int64_t M = 2 * sh.d * sh.n_head;
        std::mt19937 rng(77 + (int) sh.nt + (int) sh.n_embd);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::vector<float> wf(sh.n_embd * M), xv(sh.n_embd * sh.nt);
        for (auto & v : wf) v = u(rng);
        for (auto & v : xv) v = u(rng);
        std::vector<uint8_t> wq(ggml_row_size(GGML_TYPE_Q8_0, sh.n_embd) * M);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, wf.data(), wq.data(), 0, M, sh.n_embd, nullptr);
        const size_t row = ggml_row_size(GGML_TYPE_Q8_0, sh.n_embd);

        uint64_t n_a = 0, n_b = 0, n_c = 0;
        const auto out_a = run_graph(be, reg, sh, false, wq, xv, n_a);
        const auto out_b = run_graph(be, reg, sh, true,  deinterleave(wq, row, sh.d, sh.n_head, false), xv, n_b);
        const auto out_c = run_graph(be, reg, sh, true,  deinterleave(wq, row, sh.d, sh.n_head, true),  xv, n_c);
        size_t ndiff = 0;
        for (size_t i = 0; i < out_a.size(); i += 4) ndiff += memcmp(out_a.data() + i, out_b.data() + i, 4) != 0;
        const bool ctrl = memcmp(out_a.data(), out_c.data(), out_a.size()) != 0;
        printf("n_embd=%lld d=%lld heads=%lld nt=%lld: file-order vs de-interleaved %s (%zu of %zu differ), SIGMOID_MUL file-order=%llu "
               "de-interleaved=%llu (want %d), swapped-head control %s\n",
               (long long) sh.n_embd, (long long) sh.d, (long long) sh.n_head, (long long) sh.nt, ndiff ? "DIFFER" : "byte-equal", ndiff,
               out_a.size() / 4, (unsigned long long) n_a, (unsigned long long) n_b, want_fused ? 1 : 0, ctrl ? "changed (ok)" : "UNCHANGED");
        if (ndiff) { fprintf(stderr, "FAIL nt=%lld: de-interleaved output differs\n", (long long) sh.nt); fails++; }
        if (n_b != (want_fused ? 1u : 0u)) { fprintf(stderr, "FAIL nt=%lld: SIGMOID_MUL=%llu\n", (long long) sh.nt, (unsigned long long) n_b); fails++; }
        if (!ctrl) { fprintf(stderr, "FAIL nt=%lld: positive control did not change the output\n", (long long) sh.nt); fails++; }
        if (dump) fwrite(out_b.data(), 1, out_b.size(), dump);
    }
    if (dump) fclose(dump);
    ggml_backend_free(be);
    printf("%s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
