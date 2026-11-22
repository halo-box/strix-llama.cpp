// Digests of the fork ops added to ggml core, on the CPU: flash attention with selected keys
// (ggml_flash_attn_ext_add_top_k) and the f16 forms of ggml_dsv4_hc_pre_gated_ext. Built against the fork master and
// this branch; the outputs must be identical.

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

static uint32_t rng = 777;
static uint32_t next() {
    rng = rng * 1664525u + 1013904223u;
    return rng >> 8;
}

static ggml_backend_t cpu;

// kind: 0 values in [-1, 1), 1 a mask (0 or -inf), 2 indices in [lo, lo + mod)
struct in_t {
    ggml_tensor * t;
    int           kind;
    int64_t       lo;
    int64_t       mod;
};

static void fill(const in_t & in) {
    ggml_tensor * t = in.t;
    const int64_t n = ggml_nelements(t);
    std::vector<float> f(n);
    for (auto & x : f) {
        x = in.kind == 1 ? ((next() % 5) == 0 ? -INFINITY : 0.0f) : (next() % 2000) / 1000.0f - 1.0f;
    }
    if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> v(n);
        for (auto & x : v) x = (int32_t) (in.lo + (int64_t) (next() % in.mod));
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> v(n);
        for (int64_t i = 0; i < n; ++i) v[i] = ggml_fp32_to_fp16(f[i]);
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    } else {
        ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
    }
}

static void run(const std::string & name, const std::function<ggml_tensor *(ggml_context *, std::vector<in_t> &)> & build) {
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<in_t> in;
    ggml_tensor * out = build(ctx, in);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (!ggml_backend_supports_op(cpu, out)) {
        printf("%-56s not supported\n", name.c_str());
        ggml_free(ctx);
        return;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    for (auto & i : in) {
        fill(i);
    }
    const bool ok = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    std::vector<uint8_t> o(ggml_nbytes(out));
    ggml_backend_tensor_get(out, o.data(), 0, o.size());
    printf("%-56s %s %016llx\n", name.c_str(), ok ? "ok  " : "FAIL", (unsigned long long) fnv(o.data(), o.size()));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main() {
    ggml_backend_load_all();
    cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu)), "ggml_backend_set_n_threads");
    set_n_threads(cpu, 4);

    const int64_t D = 64, H = 4, Hkv = 2, n_top = 16;
    for (int64_t nkv : { 256, 640 }) {
        for (int64_t nq : { 1, 7, 40 }) {
            for (bool mask : { false, true }) {
                for (ggml_type kvt : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
                    for (bool top_k : { false, true }) {
                        const std::string name = "fa nkv=" + std::to_string(nkv) + " nq=" + std::to_string(nq) +
                            (mask ? " mask" : "") + " " + ggml_type_name(kvt) + (top_k ? " top_k" : "");
                        run(name, [&](ggml_context * ctx, std::vector<in_t> & in) {
                            ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, nq, H, 1);
                            ggml_tensor * k = ggml_new_tensor_4d(ctx, kvt, D, nkv, Hkv, 1);
                            ggml_tensor * v = ggml_new_tensor_4d(ctx, kvt, D, nkv, Hkv, 1);
                            ggml_tensor * m = mask ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nkv, nq, 1, 1) : nullptr;
                            in = { { q, 0, 0, 0 }, { k, 0, 0, 0 }, { v, 0, 0, 0 } };
                            if (m) in.push_back({ m, 1, 0, 0 });
                            ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, m, 0.125f, 0.0f, 0.0f);
                            if (top_k) {
                                const int64_t n_kv_raw = nkv / 4;
                                ggml_tensor * ids = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, n_top, nq, 1, 1);
                                // some negative and some past the end: ignored
                                in.push_back({ ids, 2, -3, nkv - n_kv_raw + 6 });
                                ggml_flash_attn_ext_add_top_k(fa, ids, n_kv_raw);
                            }
                            return fa;
                        });
                    }
                }
            }
        }
    }

    const int64_t n_embd = 128, hc = 4, n_tokens = 5;
    for (ggml_type xt : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
        for (ggml_type gt : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
            for (ggml_type ot : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
                const std::string name = std::string("hc_pre_gated_ext x=") + ggml_type_name(xt) + " gate=" + ggml_type_name(gt) +
                    " out=" + ggml_type_name(ot);
                run(name, [&](ggml_context * ctx, std::vector<in_t> & in) {
                    ggml_tensor * x = ggml_new_tensor_3d(ctx, xt, n_embd, hc, n_tokens);
                    ggml_tensor * g = ggml_new_tensor_3d(ctx, gt, n_embd, hc, n_tokens);
                    in = { { x, 0, 0, 0 }, { g, 0, 0, 0 } };
                    return ggml_dsv4_hc_pre_gated_ext(ctx, x, g, 0.5f, ot);
                });
            }
        }
    }
    run("hc_pre", [&](ggml_context * ctx, std::vector<in_t> & in) {
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, n_tokens);
        in = { { x, 0, 0, 0 }, { w, 0, 0, 0 } };
        return ggml_dsv4_hc_pre(ctx, x, w);
    });
    ggml_backend_free(cpu);
    return 0;
}
