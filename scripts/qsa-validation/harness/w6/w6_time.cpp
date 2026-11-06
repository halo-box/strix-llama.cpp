// W6 (PLAN-v3 §16.5 AT-B2, AT-B3): op time of GGML_OP_SPARSE_ATTN on Vulkan, in one build, against the old selected-key FLASH_ATTN_EXT
// (flash_attn_sel_cm.comp, its pre-pass included) on the same rows and data.
//
// Fixture: a C18 / C3 rows file (per QSA layer {char name[64]; int32 n_q, ns, nk, has_mask, pos0, kt;} then n_q * ns int32 cells), SYN data as the
// W3 / W5 harnesses: Q uniform +-1.7 [n_q][24][256], K and V f16 values uniform +-1.7 per (cell, kv head), the same generator and seed per layer;
// another type is the quantization (ggml_quantize_chunk) of those f16 values. Cache layout [256, 2, n_kv] viewed as [256, n_kv, 2], GQA 12, scale 1/16.
// Arms (argv): "sel" = the old op; a type "q8_0" = K and V of that type; "q8_0:f16" = K : V. The V variant is whatever GGML_VK_SA_FORCE selects.
// Per layer and arm: a graph holding the plan alone and a graph holding the plan and R attend nodes reading it (sel: R FLASH_ATTN_EXT nodes), each
// computed 3 times to warm up, then `reps` times (wall time of ggml_backend_graph_compute, which synchronizes).
//   attend = (median(plan + R attends) - median(plan)) / R; total = attend + plan (the old op has its pre-pass inside, so total is what it compares with).
// With GGML_VK_PERF_LOGGER=1 the backend's per-op timings go to stderr; the "MARK" lines let w6_time_an.py attribute them to (layer, arm, phase).
// usage: w6_time rows_file reps R arms(e.g. sel,f16,q8_0) [layers: all | list] [backend, default Vulkan0]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

struct rec { char name[64]; int32_t n_q, ns, nk, has_mask, pos0, kt; };

static double median(std::vector<double> x) {
    std::sort(x.begin(), x.end());
    return x.size() % 2 ? x[x.size() / 2] : 0.5 * (x[x.size() / 2 - 1] + x[x.size() / 2]);
}

static std::vector<std::string> split(const std::string & s, char sep = ',') {
    std::vector<std::string> r;
    std::string x;
    for (char c : s + sep) {
        if (c == sep) { if (!x.empty()) r.push_back(x); x.clear(); } else x += c;
    }
    return r;
}

static ggml_type type_of(const std::string & n) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * tn = ggml_type_name((ggml_type) t);
        if (tn && n == tn) return (ggml_type) t;
    }
    fprintf(stderr, "unknown type %s\n", n.c_str());
    exit(1);
}

static double run_graph(ggml_backend_t be, ggml_cgraph * gf, int reps) {
    for (int w = 0; w < 3; ++w) {
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
    }
    fprintf(stderr, "MARK measure\n");
    std::vector<double> ms;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    fprintf(stderr, "MARK end\n");
    return median(ms);
}

static void set_kv(ggml_tensor * t, const std::vector<float> & x, int nk4) {
    std::vector<uint8_t> qb(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(x.data(), (ggml_fp16_t *) qb.data(), x.size());
    } else {
        ggml_quantize_chunk(t->type, x.data(), qb.data(), 0, (int64_t) nk4 * 2, 256, nullptr);
    }
    ggml_backend_tensor_set(t, qb.data(), 0, qb.size());
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: w6_time rows_file reps R arms [layers] [backend]\n"); return 1; }
    const std::string rowsf = argv[1];
    const int reps = atoi(argv[2]), R = atoi(argv[3]);
    const std::vector<std::string> arms = split(argv[4]);
    std::set<int> layers;
    if (argc > 5 && std::string(argv[5]) != "all") for (auto & x : split(argv[5])) layers.insert(atoi(x.c_str()));
    const char * bname = argc > 6 ? argv[6] : "Vulkan0";
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_name(bname, nullptr);
    if (!be) { fprintf(stderr, "no backend %s\n", bname); return 1; }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(be);
    FILE * fr = fopen(rowsf.c_str(), "rb");
    if (!fr) { perror("rows"); return 1; }
    const char * force = getenv("GGML_VK_SA_FORCE");
    printf("# w6_time rows %s reps %d R %d backend %s GGML_VK_SA_FORCE=%s\n", rowsf.c_str(), reps, R, bname, force ? force : "(unset)");
    for (int L = 0;; ++L) {
        rec h;
        if (fread(&h, sizeof(h), 1, fr) != 1) break;
        std::vector<int32_t> rows((size_t) h.n_q * h.ns);
        if (fread(rows.data(), 4, rows.size(), fr) != rows.size()) { fprintf(stderr, "short rows\n"); return 1; }
        if (!layers.empty() && !layers.count(L)) continue;
        const int n_q = h.n_q, ns = h.ns, nk = h.nk, nk4 = 4 * ((nk + 3) / 4);
        std::mt19937_64 rg(0xC18ULL + (uint64_t) L * 7919ULL + (uint64_t) nk);
        auto u = [&]() { return (float) ((rg() >> 40) * (2.0 / 16777216.0) - 1.0) * 1.7f; };
        std::vector<float> Q((size_t) n_q * 24 * 256), K((size_t) nk4 * 512, 0.0f), V((size_t) nk4 * 512, 0.0f);
        for (auto & x : Q) x = u();
        for (size_t i = 0; i < (size_t) nk * 512; ++i) K[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(u()));
        for (size_t i = 0; i < (size_t) nk * 512; ++i) V[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(u()));
        for (const std::string & arm : arms) {
            const bool old_op = arm == "sel";
            const auto tt = split(arm, ':');
            const ggml_type tk = old_op ? GGML_TYPE_F16 : type_of(tt[0]);
            const ggml_type tv = old_op ? GGML_TYPE_F16 : type_of(tt.size() > 1 ? tt[1] : tt[0]);
            ggml_init_params ip = { ggml_tensor_overhead() * (R + 16) + 2 * ggml_graph_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * q_st = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 24, n_q, 1);
            ggml_tensor * k_st = ggml_new_tensor_4d(ctx, tk, 256, 2, nk4, 1);
            ggml_tensor * v_st = ggml_new_tensor_4d(ctx, tv, 256, 2, nk4, 1);
            ggml_tensor * sel  = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, ns, n_q, 1, 1);
            ggml_tensor * q = ggml_permute(ctx, q_st, 0, 2, 1, 3);
            ggml_tensor * k = ggml_permute(ctx, k_st, 0, 2, 1, 3);
            ggml_tensor * v = ggml_permute(ctx, v_st, 0, 2, 1, 3);
            ggml_tensor * plan = old_op ? nullptr : ggml_sparse_attn_plan(ctx, sel, nk, 1);
            ggml_cgraph * gp = ggml_new_graph(ctx), * ga = ggml_new_graph(ctx);
            if (plan) {
                ggml_set_output(plan);
                ggml_build_forward_expand(gp, plan);
            }
            size_t alloc_attn = 0, alloc_plan = plan ? ggml_backend_buft_get_alloc_size(buft, plan) : 0;
            for (int r = 0; r < R; ++r) {
                ggml_tensor * o;
                if (old_op) {
                    o = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / 16.0f, 0.0f, 0.0f);
                    ggml_flash_attn_ext_add_top_k(o, sel, 0);
                } else {
                    o = ggml_sparse_attn(ctx, q, k, v, plan, 1.0f / 16.0f, 0, 0);
                    alloc_attn = ggml_backend_buft_get_alloc_size(buft, o);
                }
                ggml_set_output(o);
                ggml_build_forward_expand(ga, o);
            }
            bool ok = true;
            for (int i = 0; i < ggml_graph_n_nodes(ga); ++i) ok = ok && ggml_backend_supports_op(be, ggml_graph_node(ga, i));
            if (!ok) { printf("L%d %s not supported\n", L, arm.c_str()); ggml_free(ctx); continue; }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }
            ggml_backend_tensor_set(q_st, Q.data(), 0, ggml_nbytes(q_st));
            ggml_backend_tensor_set(sel, rows.data(), 0, ggml_nbytes(sel));
            set_kv(k_st, K, nk4);
            set_kv(v_st, V, nk4);
            double tp = 0.0;
            if (plan) {
                fprintf(stderr, "MARK L%d %s plan\n", L, arm.c_str());
                tp = run_graph(be, gp, reps);
            }
            fprintf(stderr, "MARK L%d %s graph\n", L, arm.c_str());
            const double ta = run_graph(be, ga, reps);
            const double att = (ta - tp) / R;
            printf("L%d RES %-10s n_q %d nk %d plan_med %.4f graph_med %.4f attend %.4f total %.4f ms alloc_plan %zu alloc_attend %zu\n",
                   L, arm.c_str(), n_q, nk, tp, ta, att, att + tp, alloc_plan, alloc_attn);
            fflush(stdout);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }
    fclose(fr);
    ggml_backend_free(be);
    return 0;
}
