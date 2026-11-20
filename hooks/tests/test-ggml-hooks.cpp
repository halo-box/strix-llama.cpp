// End to end test of the backend hooks, with the example kernels:
// for each backend, the result must match the reference with the kernel on and off,
// and the kernel must run only when it is on.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hooks.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const float SCALE = 2.5f;
static const float BIAS  = 0.5f;

enum run_result { RUN_OK, RUN_FAILED, RUN_UNSUPPORTED };

// a and b are 1-element-offset views, so the Vulkan kernels see misaligned buffers
static run_result run(ggml_backend_t backend, bool fused, int64_t n, std::vector<float> & out, double & max_err) {
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a_buf = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n + 1);
    ggml_tensor * b_buf = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n + 1);
    ggml_tensor * a     = ggml_view_1d(ctx, a_buf, n, sizeof(float));
    ggml_tensor * b     = ggml_view_1d(ctx, b_buf, n, sizeof(float));

    ggml_tensor * t = ggml_scale_bias(ctx, a, SCALE, BIAS);
    if (fused) {
        // b first, so its VIEW node comes before SCALE and SCALE, ADD are consecutive in the graph
        t = ggml_add(ctx, b, t);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t);

    // like the scheduler: a backend only gets the nodes it supports (with a hook kernel, the supports hook says yes)
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(gf, i))) {
            ggml_free(ctx);
            return RUN_UNSUPPORTED;
        }
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> ha(n + 1), hb(n + 1);
    for (int64_t i = 0; i < n + 1; ++i) {
        ha[i] = std::sin(0.001f * i);
        hb[i] = std::cos(0.002f * i);
    }
    ggml_backend_tensor_set(a_buf, ha.data(), 0, ggml_nbytes(a_buf));
    ggml_backend_tensor_set(b_buf, hb.data(), 0, ggml_nbytes(b_buf));

    const bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;

    out.resize(n);
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    max_err = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double ref = ha[i + 1] * SCALE + BIAS + (fused ? hb[i + 1] : 0.0f);
        max_err = std::max(max_err, std::fabs(ref - out[i]));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok ? RUN_OK : RUN_FAILED;
}

// upstream ADD, hook SCALE, upstream ADD, hook SCALE, upstream ADD: segments before, between and after the hook kernels
static run_result run_mixed(ggml_backend_t backend, int64_t n, double & max_err) {
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * t = ggml_add(ctx, a, b);
    t = ggml_scale_bias(ctx, t, SCALE, BIAS);
    t = ggml_add(ctx, t, b);
    t = ggml_scale_bias(ctx, t, SCALE, BIAS);
    t = ggml_add(ctx, t, a);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t);
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(gf, i))) {
            ggml_free(ctx);
            return RUN_UNSUPPORTED;
        }
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::vector<float> ha(n), hb(n), out(n);
    for (int64_t i = 0; i < n; ++i) {
        ha[i] = std::sin(0.001f * i);
        hb[i] = std::cos(0.002f * i);
    }
    ggml_backend_tensor_set(a, ha.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, hb.data(), 0, ggml_nbytes(b));

    // twice: the second run must not see state left by the first one
    bool ok = true;
    for (int rep = 0; rep < 2 && ok; ++rep) {
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    }

    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    max_err = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double v = ha[i] + hb[i];
        v = v * SCALE + BIAS + hb[i];
        v = v * SCALE + BIAS + ha[i];
        max_err = std::max(max_err, std::fabs(v - out[i]) / (1.0 + std::fabs(v)));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok ? RUN_OK : RUN_FAILED;
}

static int n_fail = 0;

static void check(bool cond, const std::string & what) {
    printf("  %-70s %s\n", what.c_str(), cond ? "OK" : "FAIL");
    n_fail += !cond;
}

static void test_case(ggml_backend_t backend, const std::string & kernel, bool fused) {
    const int64_t n = 100003;
    for (bool on : { true, false }) {
        ggml_hook_kernel_set_enabled("example_*", on);
        ggml_hook_kernel_reset_hits();
        std::vector<float> out;
        double err = 0.0;
        const run_result res = run(backend, fused, n, out, err);
        const uint64_t hits = ggml_hook_kernel_hits(kernel.c_str());
        const std::string tag = kernel + (on ? " on" : " off");
        if (res == RUN_UNSUPPORTED) {
            // fine only when off: the backend has no native kernel, the scheduler would use another backend
            check(!on, tag + ": not supported by the backend without the hook kernel");
            continue;
        }
        check(res == RUN_OK && err < 1e-5, tag + ": result (max err " + std::to_string(err) + ")");
        check(on ? hits > 0 : hits == 0, tag + ": hits = " + std::to_string(hits));
    }
}

// MUL_MAT or GET_ROWS of a weight of type `type`, made of the q8_0 bytes `q`, on the CPU
static std::vector<float> run_weight_op(ggml_backend_t backend, ggml_type type, bool get_rows, const std::vector<uint8_t> & q, int64_t k, int64_t m,
                                        const std::vector<float> & x, int64_t n) {
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, type, k, m);
    ggml_tensor * t;
    ggml_tensor * in;
    if (get_rows) {
        in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        t  = ggml_get_rows(ctx, w, in);
    } else {
        in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
        t  = ggml_mul_mat(ctx, w, in);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t);
    std::vector<float> out;
    if (ggml_backend_supports_op(backend, t)) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(w, q.data(), 0, q.size());
        if (get_rows) {
            std::vector<int32_t> rows(n);
            for (int64_t i = 0; i < n; ++i) {
                rows[i] = (int32_t) ((i * 7) % m);
            }
            ggml_backend_tensor_set(in, rows.data(), 0, ggml_nbytes(in));
        } else {
            ggml_backend_tensor_set(in, x.data(), 0, ggml_nbytes(in));
        }
        ggml_backend_graph_compute(backend, gf);
        out.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    return out;
}

static void test_types() {
    printf("hook types\n");
    const ggml_type t = GGML_TYPE_Q8_0_HOOK_EXAMPLE;
    check(std::string(ggml_type_name(t)) == "q8_0_hook_example", "type name");
    check(ggml_blck_size(t) == ggml_blck_size(GGML_TYPE_Q8_0) && ggml_type_size(t) == ggml_type_size(GGML_TYPE_Q8_0) && ggml_is_quantized(t), "block and type size");
    check(std::string(ggml_type_name((ggml_type) 60)) == "unused", "ids between upstream and hook types are named");

    const int64_t k = 256, m = 64, n = 5;
    std::vector<float> wf(k * m), x(k * n);
    for (size_t i = 0; i < wf.size(); ++i) {
        wf[i] = std::sin(0.37f * i);
    }
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::cos(0.11f * i);
    }
    std::vector<uint8_t> q(ggml_row_size(t, k) * m), q_ref(q.size());
    const size_t sz     = ggml_quantize_chunk(t, wf.data(), q.data(), 0, m, k, nullptr);
    const size_t sz_ref = ggml_quantize_chunk(GGML_TYPE_Q8_0, wf.data(), q_ref.data(), 0, m, k, nullptr);
    check(sz == sz_ref && q == q_ref, "quantize (HOOK(type-quantize)) gives the q8_0 bytes");
    check(ggml_validate_row_data(t, q.data(), q.size()), "validate (HOOK(type-validate))");

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    for (bool get_rows : { false, true }) {
        const std::vector<float> a = run_weight_op(cpu, t, get_rows, q, k, m, x, n);
        const std::vector<float> b = run_weight_op(cpu, GGML_TYPE_Q8_0, get_rows, q, k, m, x, n);
        check(!a.empty() && a == b, std::string("CPU ") + (get_rows ? "GET_ROWS (hook kernel)" : "MUL_MAT (HOOK(type-traits-cpu))") + " equals q8_0");
    }
    ggml_backend_free(cpu);

    // a device without a kernel for the type must not take the node
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
        check(run_weight_op(gpu, t, false, q, k, m, x, n).empty(), std::string(ggml_backend_dev_name(dev)) + ": MUL_MAT with a hook type not supported");
        ggml_backend_free(gpu);
    }

    // gguf: the loader takes hook type ids
    {
        const char * path = "/tmp/test-ggml-hooks.gguf";
        ggml_init_params ip = { ggml_tensor_overhead() * 2 + q.size() + 64, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, t, k, m);
        ggml_set_name(w, "w");
        memcpy(w->data, q.data(), q.size());
        gguf_context * g = gguf_init_empty();
        gguf_add_tensor(g, w);
        gguf_write_to_file(g, path, false);
        gguf_free(g);
        ggml_free(ctx);

        ggml_context * ctx2 = nullptr;
        gguf_init_params gp = { false, &ctx2 };
        gguf_context * g2 = gguf_init_from_file(path, gp);
        ggml_tensor * w2 = g2 ? ggml_get_tensor(ctx2, "w") : nullptr;
        check(w2 && w2->type == t && memcmp(w2->data, q.data(), q.size()) == 0, "gguf write and read of a hook type tensor");
        if (g2) {
            gguf_free(g2);
            ggml_free(ctx2);
        }
        remove(path);
    }
}

// backend extension: counts the scheduler syncs before a re-reserve (HOOK(sched-realloc-sync)), and the backend life time
static int n_realloc_sync = 0, n_backend_init = 0;
GGML_HOOK_REGISTER_BACKEND_EXT([] {
    ggml_hook_backend_ext e = ggml_hook_ext("test_cpu_ext", GGML_HOOK_BACKEND_CPU);
    e.skip_realloc_sync = [](ggml_backend_t) { n_realloc_sync++; return false; };
    e.backend_init      = [](ggml_backend_t) { n_backend_init++; };
    return e;
}());

static void test_sched() {
    printf("scheduler\n");
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    // with one backend the allocator grows its buffer by itself: the scheduler re-reserves only with several backends
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev) {
        gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    }
    ggml_backend_t gpu = gpu_dev ? ggml_backend_dev_init(gpu_dev, nullptr) : nullptr;
    std::vector<ggml_backend_t> backends;
    if (gpu) {
        backends.push_back(gpu);
    }
    backends.push_back(cpu);
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), 1024, false, true);
    for (int64_t n : { 16, 4096 }) {
        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_set_input(a);
        ggml_tensor * t = ggml_scale(ctx, ggml_add(ctx, a, a), 2.0f);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t);
        ggml_backend_sched_alloc_graph(sched, gf);
        std::vector<float> h(n, 1.0f);
        ggml_backend_tensor_set(a, h.data(), 0, ggml_nbytes(a));
        ggml_backend_sched_graph_compute(sched, gf);
        ggml_backend_sched_reset(sched);
        ggml_free(ctx);
    }
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    if (gpu) {
        ggml_backend_free(gpu);
    }
    check(n_backend_init > 0, "backend extension: backend_init (" + std::to_string(n_backend_init) + ")");
    if (gpu) {
        check(n_realloc_sync > 0, "backend extension: skip_realloc_sync asked " + std::to_string(n_realloc_sync) + " times");
    } else {
        printf("  skip_realloc_sync: no GPU, not tested\n");
    }
}

int main() {
    ggml_backend_load_all();
    test_types();
    test_sched();

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const std::string reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        std::string prefix;
        if (reg == "CPU") {
            prefix = "example_cpu_";
        } else if (reg == "Vulkan") {
            prefix = "example_vk_";
        } else if (reg == "CUDA" || reg == "ROCm") {
            prefix = "example_cuda_";
        } else {
            continue;
        }

        printf("%s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);

        test_case(backend, prefix + "scale_f32", false);
        for (bool on : { true, false }) {
            ggml_hook_kernel_set_enabled("example_*", false);
            ggml_hook_kernel_set_enabled((prefix + "scale_f32").c_str(), on);
            ggml_hook_kernel_reset_hits();
            double err = 0.0;
            const run_result res = run_mixed(backend, 100003, err);
            const uint64_t hits = ggml_hook_kernel_hits((prefix + "scale_f32").c_str());
            const std::string tag = prefix + "scale_f32 mixed graph" + (on ? " on" : " off");
            check(res == RUN_OK && err < 1e-5, tag + ": result (max rel err " + std::to_string(err) + ")");
            check(on ? hits == 4 : hits == 0, tag + ": hits = " + std::to_string(hits) + " (2 nodes x 2 runs)");
        }
        // CPU op kernels run from its extra buffer type, the CPU has no fusion hook
        if (prefix != "example_cpu_") {
            test_case(backend, prefix + "scale_add_f32", true);
        }
        ggml_backend_free(backend);
    }

    ggml_hook_kernel_print();
    printf("%s\n", n_fail == 0 ? "all tests passed" : "some tests FAILED");
    return n_fail == 0 ? 0 : 1;
}
