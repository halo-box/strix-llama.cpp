// hooks: tests/test-graph-src-type.cpp of the fork (25b1c175e, upstream 9048bad8d, new in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
#include "ggml.h"
#include "ggml-backend.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

static const int64_t K = 256, M = 512, N_MATS = 8, N = 32;

struct mmid_graph {
    ggml_context * ctx;
    ggml_tensor * a, * b, * ids, * out;
    ggml_cgraph * gf;
};

static mmid_graph build(ggml_type type) {
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    mmid_graph g;
    g.ctx = ggml_init(ip);
    g.a   = ggml_new_tensor_3d(g.ctx, type, K, M, N_MATS);
    g.b   = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, K, 1, N);
    g.ids = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, 1, N);
    g.out = ggml_mul_mat_id(g.ctx, g.a, g.b, g.ids);
    g.gf  = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    return g;
}

static void upload(const mmid_graph & g, const std::vector<float> & w, const std::vector<float> & x, const std::vector<int32_t> & ids) {
    std::vector<uint8_t> q(ggml_nbytes(g.a));
    ggml_quantize_chunk(g.a->type, w.data(), q.data(), 0, M * N_MATS, K, nullptr);
    ggml_backend_tensor_set(g.a, q.data(), 0, q.size());
    ggml_backend_tensor_set(g.b, x.data(), 0, ggml_nbytes(g.b));
    ggml_backend_tensor_set(g.ids, ids.data(), 0, ggml_nbytes(g.ids));
}

static std::vector<float> run(ggml_backend_t be, const mmid_graph & g, int times) {
    for (int i = 0; i < times; ++i) {
        if (ggml_backend_graph_compute(be, g.gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FAIL: compute\n");
            exit(1);
        }
    }
    std::vector<float> o(ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, o.data(), 0, ggml_nbytes(g.out));
    return o;
}

} // namespace ggml_fork
using namespace ggml_fork; // hooks: main is global
int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    }
    if (!dev) {
        printf("SKIP: no GPU device\n");
        return 0;
    }
    ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    const ggml_type first = GGML_TYPE_MXFP4, second = GGML_TYPE_Q4_0_ROCMFP4_FAST;
    if (ggml_blck_size(first) != ggml_blck_size(second) || ggml_type_size(first) != ggml_type_size(second)) {
        fprintf(stderr, "FAIL: %s and %s no longer share a block layout\n", ggml_type_name(first), ggml_type_name(second));
        return 1;
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> w(K * M * N_MATS), x(K * N);
    for (auto & v : w) v = dist(rng);
    for (auto & v : x) v = dist(rng);
    std::vector<int32_t> ids(N);
    for (int i = 0; i < N; ++i) ids[i] = i % N_MATS;

    mmid_graph g = build(first);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(g.ctx, gpu);
    if (!ggml_backend_supports_op(gpu, g.out)) {
        printf("SKIP: %s does not support MUL_MAT_ID on %s\n", ggml_backend_name(gpu), ggml_type_name(first));
        return 0;
    }
    upload(g, w, x, ids);
    run(gpu, g, 4);

    g.a->type = second;
    if (!ggml_backend_supports_op(gpu, g.out)) {
        printf("SKIP: %s does not support MUL_MAT_ID on %s\n", ggml_backend_name(gpu), ggml_type_name(second));
        return 0;
    }
    upload(g, w, x, ids);
    const std::vector<float> got = run(gpu, g, 4);

    mmid_graph r = build(second);
    ggml_backend_buffer_t rbuf = ggml_backend_alloc_ctx_tensors(r.ctx, cpu);
    upload(r, w, x, ids);
    const std::vector<float> ref = run(cpu, r, 1);

    double err = 0.0, norm = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) {
            fprintf(stderr, "FAIL: %s output %zu is %f after the weight type changed from %s to %s at the same address\n",
                    ggml_backend_name(gpu), i, got[i], ggml_type_name(first), ggml_type_name(second));
            return 1;
        }
        err  += (got[i] - ref[i]) * (got[i] - ref[i]);
        norm += ref[i] * ref[i];
    }
    const double nmse = err / norm;
    if (nmse > 5e-4) {
        fprintf(stderr, "FAIL: %s NMSE %g against the CPU after the weight type changed from %s to %s at the same address\n",
                ggml_backend_name(gpu), nmse, ggml_type_name(first), ggml_type_name(second));
        return 1;
    }
    printf("OK: %s NMSE %g\n", ggml_backend_name(gpu), nmse);

    ggml_backend_buffer_free(rbuf);
    ggml_backend_buffer_free(buf);
    ggml_free(r.ctx);
    ggml_free(g.ctx);
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    return 0;
}
