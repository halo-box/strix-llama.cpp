// Prints a digest of each fork type: quantized bytes, dequantized values, validation and a CPU MUL_MAT.
// Built against the fork master and against the hooks branch: the outputs must be identical.
// Uses the type ids by number, so the same source builds on both.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint64_t fnv(const void * p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ b[i]) * 1099511628211ull;
    }
    return h;
}

int main() {
    const int ids[] = { 100, 101, 102, 103, 104, 107, 143, 144, 145, 146, 147, 148, 149 };
    const int64_t k = 512, m = 48, n = 7;

    std::vector<float> w(k * m), x(k * n);
    for (size_t i = 0; i < w.size(); ++i) {
        w[i] = std::sin(0.37f * i) * (1.0f + 0.5f * std::cos(0.011f * i));
    }
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::cos(0.13f * i);
    }

    // the registry API: the CPU backend can be a loaded module (GGML_BACKEND_DL)
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    set_n_threads(cpu, 4);

    for (int id : ids) {
        const ggml_type t = (ggml_type) id;
        const size_t row = ggml_row_size(t, k);
        std::vector<uint8_t> q(row * m);
        const size_t sz = ggml_quantize_chunk(t, w.data(), q.data(), 0, m, k, nullptr);

        std::vector<float> dq(k * m);
        ggml_get_type_traits(t)->to_float(q.data(), dq.data(), k * m);
        double err = 0.0;
        for (size_t i = 0; i < dq.size(); ++i) {
            err += (dq[i] - w[i]) * (dq[i] - w[i]);
        }
        const bool valid = ggml_validate_row_data(t, q.data(), q.size());

        // MUL_MAT on the CPU
        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * tw = ggml_new_tensor_2d(ctx, t, k, m);
        ggml_tensor * tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
        ggml_tensor * ty = ggml_mul_mat(ctx, tw, tx);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ty);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        ggml_backend_tensor_set(tw, q.data(), 0, q.size());
        ggml_backend_tensor_set(tx, x.data(), 0, ggml_nbytes(tx));
        const bool ok = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
        std::vector<float> y(m * n);
        ggml_backend_tensor_get(ty, y.data(), 0, ggml_nbytes(ty));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);

        printf("%-18s id %3d blck %3lld size %3zu | quant %zu bytes %016llx | dequant %016llx rmse %.6f | valid %d | mul_mat %d %016llx\n",
               ggml_type_name(t), id, (long long) ggml_blck_size(t), ggml_type_size(t), sz, (unsigned long long) fnv(q.data(), q.size()),
               (unsigned long long) fnv(dq.data(), dq.size() * sizeof(float)), std::sqrt(err / dq.size()), valid, ok,
               (unsigned long long) fnv(y.data(), y.size() * sizeof(float)));
    }
    ggml_backend_free(cpu);
    return 0;
}
