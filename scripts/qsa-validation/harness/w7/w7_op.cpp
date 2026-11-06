// AT-W7PERF op level (PLAN-v3 §16.5): SPARSE_ATTN over all-visible rows (plan + attend) with forced G1 and forced G4, against
// the causal FLASH_ATTN_EXT of the same K / V pair, on n_q x n_kv, GQA 12, D 256, one stream, SYN data. The op is timed by the
// wall time of ggml_backend_graph_compute (it synchronizes) with HIP graphs off, after 3 warm-ups: per (arm, shape, type pair)
// the median of REPS computes on each of 12 data seeds ("layers"), and the mean of those medians.
// usage: w7_op <proc tag> <backend> <reps> <pairs: k:v,k:v> <n_q list> <n_kv list>   (GGML_CUDA_SA_FORCE is set by this process per arm)
// CPU arm: backend CPU with arms G1 only (reports the CPU attend per layer).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static ggml_type type_of(const std::string & s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && s == n) { return (ggml_type) t; }
    }
    fprintf(stderr, "bad type %s\n", s.c_str()); exit(2);
}

static std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> r; size_t p = 0;
    while (p <= s.size()) { size_t q = std::min(s.find(',', p), s.size()); r.push_back(s.substr(p, q - p)); p = q + 1; }
    return r;
}

// fill a [256, n_kv, n_head_kv] tensor of type t from unit-variance normal f32 rows
static void fill_tensor(ggml_tensor * t, int64_t n_kv, int64_t n_head_kv, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int64_t D = 256;
    std::vector<float> f(D * n_kv * n_head_kv);
    for (auto & x : f) { x = nd(rng) / 16.0f; }
    std::vector<uint8_t> b(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) b.data(), (int64_t) f.size());
    } else if (t->type == GGML_TYPE_BF16) {
        ggml_fp32_to_bf16_row(f.data(), (ggml_bf16_t *) b.data(), (int64_t) f.size());
    } else {
        ggml_quantize_chunk(t->type, f.data(), b.data(), 0, n_kv * n_head_kv, D, nullptr);
    }
    ggml_backend_tensor_set(t, b.data(), 0, b.size());
}

static double run_arm(ggml_backend_t be, const std::string & arm, ggml_type tk, ggml_type tv, int64_t n_q, int64_t n_kv, int reps, unsigned seed) {
    const int64_t D = 256, n_head_kv = 2, n_head = 24;
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_q, n_head);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, tk, D, n_kv, n_head_kv);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, tv, D, n_kv, n_head_kv);
    ggml_tensor * rows = nullptr, * mask = nullptr, * out = nullptr;
    ggml_cgraph * gf = ggml_new_graph(ctx);
    const float scale = 1.0f / 16.0f;
    if (arm == "dense") {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_q);
        out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(out, GGML_PREC_F32);
    } else {
        rows = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, n_kv, n_q, 1, 1);
        ggml_tensor * plan = ggml_sparse_attn_plan(ctx, rows, n_kv, arm == "G4" ? 4 : 1);
        ggml_build_forward_expand(gf, plan);
        out = ggml_sparse_attn(ctx, q, k, v, plan, scale, 0, 0);
    }
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (!ggml_backend_supports_op(be, ggml_graph_node(gf, i))) {
            ggml_free(ctx);
            return -1.0;
        }
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "alloc failed\n"); exit(1); }
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> qv(D * n_q * n_head);
    for (auto & x : qv) { x = nd(rng) / 16.0f; }
    ggml_backend_tensor_set(q, qv.data(), 0, qv.size() * 4);
    fill_tensor(k, n_kv, n_head_kv, rng);
    fill_tensor(v, n_kv, n_head_kv, rng);
    // causal: the n_q queries are the last positions of the n_kv cells
    if (mask) {
        std::vector<ggml_fp16_t> m(n_kv * n_q);
        const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        for (int64_t i = 0; i < n_q; ++i) {
            const int64_t p = n_kv - n_q + i;
            for (int64_t j = 0; j < n_kv; ++j) { m[i * n_kv + j] = j <= p ? z : ninf; }
        }
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * 2);
    }
    if (rows) {
        std::vector<int32_t> r(n_kv * n_q);
        for (int64_t i = 0; i < n_q; ++i) {
            const int64_t p = n_kv - n_q + i;
            for (int64_t j = 0; j < n_kv; ++j) { r[i * n_kv + j] = j <= p ? (int32_t) j : -1; }
        }
        ggml_backend_tensor_set(rows, r.data(), 0, r.size() * 4);
    }
    for (int w = 0; w < 3; ++w) {
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
    }
    std::vector<double> ms;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

int main(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage: w7_op tag backend reps pairs n_q_list n_kv_list\n"); return 2; }
    const std::string tag = argv[1], bname = argv[2];
    const int reps = atoi(argv[3]);
    ggml_backend_load_all();
    ggml_backend_t be = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (bname == ggml_backend_dev_name(d)) { be = ggml_backend_dev_init(d, nullptr); }
    }
    if (!be) { fprintf(stderr, "no backend %s\n", bname.c_str()); return 2; }
    if (bname == "CPU") { ggml_backend_cpu_set_n_threads(be, 8); }
    const bool cpu = bname == "CPU";
    for (const auto & pair : split(argv[4])) {
        const size_t c = pair.find(':');
        const ggml_type tk = type_of(pair.substr(0, c)), tv = type_of(pair.substr(c + 1));
        for (const auto & nq_s : split(argv[5])) {
            for (const auto & nk_s : split(argv[6])) {
                const int64_t n_q = atoll(nq_s.c_str()), n_kv = atoll(nk_s.c_str());
                if (n_q > n_kv) { continue; }
                for (const char * arm : { "G1", "G4", "dense" }) {
                    if (cpu && std::string(arm) != "G1") { continue; }
                    double sum = 0; std::string line;
                    bool ok = true;
                    for (unsigned l = 0; l < 12 && ok; ++l) {
                        const double m = run_arm(be, arm, tk, tv, n_q, n_kv, reps, 1000u * (unsigned) l + 17u);
                        ok = m >= 0;
                        sum += m;
                        char b[32]; snprintf(b, sizeof(b), " %.4f", m); line += b;
                    }
                    if (!ok) {
                        printf("R %s %s %s %s n_q=%lld n_kv=%lld arm=%s unsupported\n", tag.c_str(), bname.c_str(), ggml_type_name(tk), ggml_type_name(tv), (long long) n_q, (long long) n_kv, arm);
                        continue;
                    }
                    printf("R %s %s %s %s n_q=%lld n_kv=%lld arm=%s ms=%.4f layers:%s\n", tag.c_str(), bname.c_str(), ggml_type_name(tk), ggml_type_name(tv),
                           (long long) n_q, (long long) n_kv, arm, sum / 12.0, line.c_str());
                    fflush(stdout);
                }
            }
        }
    }
    return 0;
}
