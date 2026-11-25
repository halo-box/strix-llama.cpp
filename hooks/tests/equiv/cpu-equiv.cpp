// Digests of CPU ops that the fork changes: MUL_MAT with an F16 src1, MUL_MAT_ID, binary ops with f16 sources and an
// f32 result, ADD / OUT_PROD / GET_ROWS on the fork types. Built against the fork master and this branch.

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
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ b[i]) * 1099511628211ull;
    }
    return h;
}

static ggml_backend_t cpu;

// fills every input with a fixed pattern (quantized inputs through the quantizer), computes, prints a digest of the output
static void run(const std::string & name, const std::function<ggml_tensor *(ggml_context *, std::vector<ggml_tensor *> &)> & build) {
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> in;
    ggml_tensor * out = build(ctx, in);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (!ggml_backend_supports_op(cpu, out)) {
        printf("%-40s not supported\n", name.c_str());
        ggml_free(ctx);
        return;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    int seed = 1;
    for (ggml_tensor * t : in) {
        const int64_t n = ggml_nelements(t);
        std::vector<float> f(n);
        for (int64_t i = 0; i < n; ++i) {
            f[i] = std::sin(0.31f * (i + 17 * seed)) * (1.0f + 0.3f * std::cos(0.007f * i));
        }
        seed++;
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
        } else if (t->type == GGML_TYPE_I32) {
            std::vector<int32_t> v(n);
            const int64_t mod = t->op_params[0] > 0 ? t->op_params[0] : 1;
            for (int64_t i = 0; i < n; ++i) {
                v[i] = (int32_t) ((i * 7 + 3) % mod);
            }
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        } else {
            std::vector<uint8_t> q(ggml_nbytes(t));
            ggml_quantize_chunk(t->type, f.data(), q.data(), 0, n / t->ne[0], t->ne[0], nullptr);
            ggml_backend_tensor_set(t, q.data(), 0, q.size());
        }
    }
    const bool ok = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    std::vector<uint8_t> o(ggml_nbytes(out));
    ggml_backend_tensor_get(out, o.data(), 0, o.size());
    printf("%-40s %s %016llx\n", name.c_str(), ok ? "ok  " : "FAIL", (unsigned long long) fnv(o.data(), o.size()));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static ggml_tensor * input(ggml_context * ctx, std::vector<ggml_tensor *> & in, ggml_type t, int64_t a, int64_t b, int64_t c = 1) {
    ggml_tensor * x = ggml_new_tensor_3d(ctx, t, a, b, c);
    in.push_back(x);
    return x;
}

int main() {
    ggml_backend_load_all();
    cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu)), "ggml_backend_set_n_threads");
    set_n_threads(cpu, 4);

    const int k = 512;
    for (int ti : { (int) GGML_TYPE_Q4_0, (int) GGML_TYPE_Q8_0, (int) GGML_TYPE_Q4_K, (int) GGML_TYPE_Q6_K, (int) GGML_TYPE_IQ4_XS, 100, 144 }) {
        const ggml_type t = (ggml_type) ti;
        for (ggml_type t1 : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
            for (int n : { 1, 7, 64 }) {
                run(std::string("mul_mat ") + ggml_type_name(t) + " x " + ggml_type_name(t1) + " n=" + std::to_string(n),
                    [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                        ggml_tensor * w = input(ctx, in, t, k, 96);
                        ggml_tensor * x = input(ctx, in, t1 == GGML_TYPE_F16 ? GGML_TYPE_F32 : t1, k, n);
                        if (t1 == GGML_TYPE_F16) {
                            x = ggml_cast(ctx, x, GGML_TYPE_F16);
                        }
                        return ggml_mul_mat(ctx, w, x);
                    });
                run(std::string("mul_mat_id ") + ggml_type_name(t) + " x " + ggml_type_name(t1) + " n=" + std::to_string(n),
                    [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                        const int n_as = 6, n_used = 2;
                        ggml_tensor * w = input(ctx, in, t, k, 64, n_as);
                        ggml_tensor * x = input(ctx, in, GGML_TYPE_F32, k, n_used, n);
                        if (t1 == GGML_TYPE_F16) {
                            x = ggml_cast(ctx, x, GGML_TYPE_F16);
                        }
                        ggml_tensor * ids = input(ctx, in, GGML_TYPE_I32, n_used, n);
                        ids->op_params[0] = n_as;
                        return ggml_mul_mat_id(ctx, w, x, ids);
                    });
            }
        }
    }
    for (int op = 0; op < 4; ++op) {
        for (ggml_type a : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
            run(std::string("binary op ") + std::to_string(op) + " " + ggml_type_name(a) + " , f16 -> f32",
                [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                    ggml_tensor * x = input(ctx, in, GGML_TYPE_F32, 300, 5);
                    ggml_tensor * y = ggml_cast(ctx, input(ctx, in, GGML_TYPE_F32, 300, 5), GGML_TYPE_F16);
                    if (a == GGML_TYPE_F16) {
                        x = ggml_cast(ctx, x, GGML_TYPE_F16);
                        return ggml_add_cast(ctx, x, y, GGML_TYPE_F32);
                    }
                    switch (op) {
                        case 0:  return ggml_add(ctx, x, y);
                        case 1:  return ggml_sub(ctx, x, y);
                        case 2:  return ggml_mul(ctx, x, y);
                        default: return ggml_div(ctx, x, y);
                    }
                });
        }
    }
    for (int ti : { 100, 101, 102, 103, 104, 107, 143, 144, 145, 146, 147, 148, 149 }) {
        const ggml_type t = (ggml_type) ti;
        run(std::string("get_rows ") + ggml_type_name(t), [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            ggml_tensor * w = input(ctx, in, t, k, 40);
            ggml_tensor * r = input(ctx, in, GGML_TYPE_I32, 9, 1);
            r->op_params[0] = 40;
            return ggml_get_rows(ctx, w, r);
        });
        run(std::string("add ") + ggml_type_name(t), [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            ggml_tensor * w = input(ctx, in, t, k, 8);
            return ggml_add(ctx, w, input(ctx, in, GGML_TYPE_F32, k, 8));
        });
        run(std::string("out_prod ") + ggml_type_name(t), [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            ggml_tensor * w = input(ctx, in, t, k, 8);
            return ggml_out_prod(ctx, w, input(ctx, in, GGML_TYPE_F32, 4, 8));
        });
    }
    ggml_backend_free(cpu);
    return 0;
}
