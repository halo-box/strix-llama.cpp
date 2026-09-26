// TQ2_T reference-vector test.
//   test-tq2t [path/to/tq2t_ref.bin]
// The file (from agention-infer examples/tq2t_vectors.rs) is: u32 n_blocks, then
// n_blocks x 34-byte blocks, then n_blocks x 128 little-endian f32 expected values.
// Checks, all bit-exact against the expected f32:
//   1. the CPU reference decoder (type_traits->to_float),
//   2. GET_ROWS on every available backend device (CPU, Vulkan, ...), which runs
//      each backend's own TQ2_T decoder,
// plus the CPU vec_dot against the decoded weights (tolerance: Q8_0 activations)
// and a placeholder-encoder round trip.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef TQ2T_REF_DEFAULT
#define TQ2T_REF_DEFAULT "tq2t_ref.bin"
#endif

static uint32_t hyb_index(uint32_t e) {
    const uint32_t x = ((e >> 1) & 0x7fffu) * 0x9e3779b1u; // 15-bit states
    return (e & 1) ? ((x >> 10) & 2047u) : (x >> 21);
}

static int compare(const char * what, const float * got, const float * want, size_t n) {
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        if (memcmp(&got[i], &want[i], sizeof(float)) != 0) {
            if (bad < 5) {
                fprintf(stderr, "  %s: mismatch at %zu: got %.9g want %.9g\n", what, i, got[i], want[i]);
            }
            bad++;
        }
    }
    printf("%-40s %s (%zu/%zu bit-exact)\n", what, bad ? "FAIL" : "OK", n - bad, n);
    return bad ? 1 : 0;
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : TQ2T_REF_DEFAULT;
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    uint32_t nb = 0;
    if (fread(&nb, 4, 1, f) != 1 || nb == 0) {
        fprintf(stderr, "bad header\n");
        return 1;
    }
    const size_t bs = ggml_type_size(GGML_TYPE_TQ2_T);
    const int64_t qk = ggml_blck_size(GGML_TYPE_TQ2_T);
    if (bs != 34 || qk != 128) {
        fprintf(stderr, "unexpected tq2_t block: %zu bytes / %lld weights\n", bs, (long long) qk);
        return 1;
    }
    std::vector<uint8_t> blocks(nb * bs);
    std::vector<float> want(nb * qk);
    if (fread(blocks.data(), 1, blocks.size(), f) != blocks.size() ||
        fread(want.data(), sizeof(float), want.size(), f) != want.size()) {
        fprintf(stderr, "short file\n");
        return 1;
    }
    fclose(f);

    int fails = 0;
    if (hyb_index(12345) != 1657 || hyb_index(12344) != 1035) {
        fprintf(stderr, "hyb_index(12345) = %u (want 1657), hyb_index(12344) = %u (want 1035)\n", hyb_index(12345), hyb_index(12344));
        fails++;
    }

    // 1. CPU reference decoder
    std::vector<float> got(nb * qk);
    ggml_get_type_traits(GGML_TYPE_TQ2_T)->to_float(blocks.data(), got.data(), nb * qk);
    fails += compare("cpu to_float", got.data(), want.data(), got.size());

    // 2. GET_ROWS on each backend device: a [128 x nb] tq2_t matrix, all rows gathered.
    ggml_backend_load_all();
    for (size_t di = 0; di < ggml_backend_dev_count(); ++di) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(di);
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (!be) {
            continue;
        }
        ggml_init_params ip = { 4 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ2_T, qk, nb);
        ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nb);
        ggml_tensor * out = ggml_get_rows(ctx, a, idx);
        std::string name = std::string("get_rows on ") + ggml_backend_dev_name(dev);
        if (!ggml_backend_supports_op(be, out)) {
            printf("%-40s skipped (unsupported)\n", name.c_str());
            ggml_free(ctx);
            ggml_backend_free(be);
            continue;
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_tensor_set(a, blocks.data(), 0, blocks.size());
        std::vector<int32_t> rows(nb);
        for (uint32_t i = 0; i < nb; ++i) {
            rows[i] = (int32_t) i;
        }
        ggml_backend_tensor_set(idx, rows.data(), 0, rows.size() * sizeof(int32_t));
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: compute failed\n", name.c_str());
            fails++;
        } else {
            std::vector<float> res(nb * qk);
            ggml_backend_tensor_get(out, res.data(), 0, res.size() * sizeof(float));
            fails += compare(name.c_str(), res.data(), want.data(), res.size());
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(be);
    }

    // 3. CPU vec_dot vs the decoded weights (activations go through Q8_0, so tolerance)
    {
        const auto * cpu = ggml_get_type_traits_cpu(GGML_TYPE_TQ2_T);
        const ggml_type vt = cpu->vec_dot_type;
        const int64_t n = nb * qk;
        std::vector<float> x(n);
        for (int64_t i = 0; i < n; ++i) {
            x[i] = 0.1f + std::cos(0.37f * (float) i);
        }
        std::vector<uint8_t> xq(ggml_row_size(vt, n));
        ggml_get_type_traits_cpu(vt)->from_float(x.data(), xq.data(), n);
        std::vector<float> xd(n);
        ggml_get_type_traits(vt)->to_float(xq.data(), xd.data(), n);
        double ref = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            ref += (double) want[i] * xd[i];
        }
        float res = 0.0f;
        cpu->vec_dot(n, &res, 0, blocks.data(), 0, xq.data(), 0, 1);
        const bool ok = std::fabs(res - ref) <= 1e-4 * (1.0 + std::fabs(ref));
        printf("%-40s %s (got %.6f ref %.6f)\n", "cpu vec_dot", ok ? "OK" : "FAIL", res, ref);
        fails += ok ? 0 : 1;
    }

    // 4. placeholder encoder: decodes, and beats the all-zero-codes baseline
    {
        const int64_t n = 8 * qk;
        std::vector<float> x(n), y(n);
        for (int64_t i = 0; i < n; ++i) {
            x[i] = std::sin(1.3f * (float) i) + 0.5f * std::cos(0.21f * (float) i);
        }
        std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_TQ2_T, n));
        ggml_quantize_chunk(GGML_TYPE_TQ2_T, x.data(), q.data(), 0, 1, n, nullptr);
        ggml_get_type_traits(GGML_TYPE_TQ2_T)->to_float(q.data(), y.data(), n);
        double se = 0.0, sx = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            se += (x[i] - y[i]) * (x[i] - y[i]);
            sx += x[i] * x[i];
        }
        const double rel = se / sx;
        const bool ok = rel < 0.5;
        printf("%-40s %s (relative MSE %.4f)\n", "placeholder encoder round trip", ok ? "OK" : "FAIL", rel);
        fails += ok ? 0 : 1;
    }

    printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
