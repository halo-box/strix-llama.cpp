// Digests of the fork types on a GPU backend (ROCm0 by default): MUL_MAT at several batch sizes, MUL_MAT_ID and
// GET_ROWS for every type id >= 100. Built against the fork master and this branch; the outputs must be identical.
// usage: hip-equiv [backend name]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

static uint64_t fnv(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ ((const uint8_t *) p)[i]) * 1099511628211ull;
    }
    return h;
}

static uint32_t rng = 4242;
static float frand() {
    rng = rng * 1664525u + 1013904223u;
    return ((rng >> 8) % 20000) / 10000.0f - 1.0f;
}

static ggml_backend_t gpu;

struct in_t {
    ggml_tensor * t;
    int64_t       mod; // > 0: I32 indices in [0, mod)
};

static void fill(const in_t & in) {
    ggml_tensor * t = in.t;
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> v(n);
        // distinct values within a row (MUL_MAT_ID: the experts of a token)
        const int64_t row = t->ne[0];
        for (int64_t r = 0; r < n / row; ++r) {
            rng = rng * 1664525u + 1013904223u;
            const int64_t start = (rng >> 8) % in.mod;
            for (int64_t j = 0; j < row; ++j) {
                v[r*row + j] = (int32_t) ((start + j) % in.mod);
            }
        }
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        return;
    }
    std::vector<float> f(n);
    for (auto & x : f) x = frand();
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
        return;
    }
    std::vector<uint8_t> q(ggml_nbytes(t));
    ggml_quantize_chunk(t->type, f.data(), q.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

static void run(const std::string & name, const std::function<ggml_tensor *(ggml_context *, std::vector<in_t> &)> & build) {
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<in_t> in;
    ggml_tensor * out = build(ctx, in);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (!ggml_backend_supports_op(gpu, out)) {
        printf("%-48s not supported\n", name.c_str());
        ggml_free(ctx);
        return;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    for (auto & i : in) {
        fill(i);
    }
    const bool ok = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
    std::vector<uint8_t> o(ggml_nbytes(out));
    ggml_backend_tensor_get(out, o.data(), 0, o.size());
    double sum = 0;
    for (size_t k = 0; k < o.size() / 4; ++k) sum += std::fabs(((const float *) o.data())[k]);
    printf("%-48s %s %016llx  |sum| %.6g\n", name.c_str(), ok ? "ok  " : "FAIL", (unsigned long long) fnv(o.data(), o.size()), sum);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    gpu = ggml_backend_init_by_name(argc > 1 ? argv[1] : "ROCm0", nullptr);
    if (!gpu) {
        fprintf(stderr, "no backend\n");
        return 1;
    }
    const int64_t k = 512, n = 256;
    for (int id = 100; id < GGML_TYPE_COUNT; ++id) {
        const ggml_type t = (ggml_type) id;
        if (ggml_type_size(t) == 0 || ggml_blck_size(t) == 0) {
            continue;
        }
        const std::string tn = ggml_type_name(t);
        for (int64_t m : { 1, 2, 5, 8, 16, 64, 512 }) {
            run("mul_mat " + tn + " m=" + std::to_string(m), [&](ggml_context * ctx, std::vector<in_t> & in) {
                ggml_tensor * w = ggml_new_tensor_2d(ctx, t, k, n);
                ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
                in = { { w, 0 }, { x, 0 } };
                return ggml_mul_mat(ctx, w, x);
            });
        }
        for (int64_t tokens : { 1, 4, 64 }) {
            run("mul_mat_id " + tn + " tokens=" + std::to_string(tokens), [&](ggml_context * ctx, std::vector<in_t> & in) {
                const int64_t E = 8, used = 2;
                ggml_tensor * w   = ggml_new_tensor_3d(ctx, t, k, n, E);
                ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, used, tokens);
                ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);
                in = { { w, 0 }, { x, 0 }, { ids, E } };
                return ggml_mul_mat_id(ctx, w, x, ids);
            });
        }
        run("get_rows " + tn, [&](ggml_context * ctx, std::vector<in_t> & in) {
            ggml_tensor * w   = ggml_new_tensor_2d(ctx, t, k, 64);
            ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 9);
            in = { { w, 0 }, { ids, 64 } };
            return ggml_get_rows(ctx, w, ids);
        });
    }
    ggml_backend_free(gpu);
    return 0;
}
