// Digests of the EXL3 ops on the CPU (builds with GGML_EXL3=ON). Built against the fork master and this branch.
// The inputs are a fixed pseudo-random pattern: every trellis word decodes, so the digests compare the kernels.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

extern "C" {
struct ggml_tensor * ggml_exl3_mul_mat(struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *, struct ggml_tensor *, struct ggml_tensor *, int);
struct ggml_tensor * ggml_exl3_mul_mat_id(struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *, struct ggml_tensor *, struct ggml_tensor *, struct ggml_tensor *, int);
struct ggml_tensor * ggml_exl3_get_rows(struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *, struct ggml_tensor *, int64_t);
}

static uint64_t fnv(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ ((const uint8_t *) p)[i]) * 1099511628211ull;
    }
    return h;
}

static uint32_t rng = 12345;
static uint32_t next() {
    rng = rng * 1664525u + 1013904223u;
    return rng >> 8;
}

static ggml_backend_t cpu;

static void fill(ggml_tensor * t, int64_t mod) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_I16) {
        std::vector<int16_t> v(n);
        for (auto & x : v) x = (int16_t) next();
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    } else if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> v(n);
        for (auto & x : v) x = (int32_t) (next() % mod);
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> v(n);
        for (auto & x : v) x = ggml_fp32_to_fp16(((next() % 2000) / 1000.0f - 1.0f) * (mod ? 1.0f : 0.05f));
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    } else {
        std::vector<float> v(n);
        for (auto & x : v) x = (next() % 2000) / 1000.0f - 1.0f;
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    }
}

struct in_t {
    ggml_tensor * t;
    int64_t       mod;
};

static void run(const std::string & name, const std::function<ggml_tensor *(ggml_context *, std::vector<in_t> &)> & build) {
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<in_t> in;
    ggml_tensor * out = build(ctx, in);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (!ggml_backend_supports_op(cpu, out)) {
        printf("%-44s not supported\n", name.c_str());
        ggml_free(ctx);
        return;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    for (auto & i : in) {
        fill(i.t, i.mod);
    }
    const bool ok = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    std::vector<uint8_t> o(ggml_nbytes(out));
    ggml_backend_tensor_get(out, o.data(), 0, o.size());
    printf("%-44s %s %016llx\n", name.c_str(), ok ? "ok  " : "FAIL", (unsigned long long) fnv(o.data(), o.size()));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main() {
    ggml_backend_load_all();
    cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu)), "ggml_backend_set_n_threads");
    set_n_threads(cpu, 4);

    const int64_t k = 256, n = 384;
    for (int cb = 0; cb <= 2; ++cb) {
        for (int K : { 2, 4, 8 }) {
            for (int64_t m : { 1, 9 }) {
                run("mul_mat cb=" + std::to_string(cb) + " K=" + std::to_string(K) + " m=" + std::to_string(m),
                    [&](ggml_context * ctx, std::vector<in_t> & in) {
                        ggml_tensor * tr  = ggml_new_tensor_3d(ctx, GGML_TYPE_I16, 16 * K, n / 16, k / 16);
                        ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
                        ggml_tensor * suh = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, k);
                        ggml_tensor * svh = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n);
                        in = { { tr, 0 }, { x, 0 }, { suh, 1 }, { svh, 1 } };
                        return ggml_exl3_mul_mat(ctx, tr, x, suh, svh, cb);
                    });
                run("mul_mat_id cb=" + std::to_string(cb) + " K=" + std::to_string(K) + " tokens=" + std::to_string(m),
                    [&](ggml_context * ctx, std::vector<in_t> & in) {
                        const int64_t E = 4, n_used = 2;
                        ggml_tensor * tr  = ggml_new_tensor_4d(ctx, GGML_TYPE_I16, 16 * K, n / 16, k / 16, E);
                        ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, m);
                        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, m);
                        ggml_tensor * suh = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, E);
                        ggml_tensor * svh = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n, E);
                        in = { { tr, 0 }, { x, 0 }, { ids, E }, { suh, 1 }, { svh, 1 } };
                        return ggml_exl3_mul_mat_id(ctx, tr, x, ids, suh, svh, cb);
                    });
            }
        }
    }
    for (int K : { 1, 3, 8 }) {
        run("get_rows K=" + std::to_string(K), [&](ggml_context * ctx, std::vector<in_t> & in) {
            const int64_t groups = 3, rows = 50;
            ggml_tensor * table = ggml_new_tensor_2d(ctx, GGML_TYPE_I16, groups * (1 + 16 * K), rows);
            ggml_tensor * ids   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 7);
            ggml_tensor * signs = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, groups);
            in = { { table, 0 }, { ids, rows }, { signs, 1 } };
            return ggml_exl3_get_rows(ctx, table, ids, signs, 700);
        });
    }
    ggml_backend_free(cpu);
    return 0;
}
