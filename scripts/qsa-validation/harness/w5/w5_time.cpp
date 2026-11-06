// W5 (PLAN-v3 §13 W5 "generic kernel time at C18's wiki d4k / d131k recorded against f16", P4 / P7): the time of the
// GGML_OP_SPARSE_ATTN op of the rewrite, per K = V type, on the C18 rows with SYN data. Recorded, not gated (U8).
//
// Fixture: a C18 / C3 rows file (per QSA layer {char name[64]; int32 n_q, ns, nk, has_mask, pos0, kt;} then n_q * ns
// int32 cells; ~/dev/wt_lab/c18/cap/wiki{S_p4096,L_p131072}.rows), SYN data as the W3 harness (w3_attn.cu, C18
// c18_attn.cu:193-199): Q uniform +-1.7 [n_q][24][256], K and V f16 values uniform +-1.7 per (cell, kv head), the same
// generator and seed per layer; a type other than f16 is the quantization (ggml_quantize_chunk) or the f32 / bf16
// value of those f16 values. Layout of the cache: [256, 2 kv heads, n_kv] viewed as [256, n_kv, 2] (the llama cache
// layout), GQA 12, scale 1/16.
// Per layer and type: one graph holding the plan (G = 4, the C18 prefill shape) and R attend nodes reading it, and a
// graph holding the plan alone; each computed 3 times to warm up, then `reps` times (wall time of
// ggml_backend_graph_compute, which synchronizes). attend = (median(plan + R attends) - median(plan)) / R. Summary: the
// mean over layers of the per-layer attend time, as C18 / W3 (mean of per-layer medians), and its ratio to f16.
// The tier is the backend's (S_fast types: fast; others: generic); run the process with GGML_CUDA_SA_FORCE=generic /
// portable / vdirect to time another tier or V loader (the switch is read once per process).
// usage: w5_time rows_file reps R types(e.g. f16,f32,q4_1) [layers: all | list] [backend, default ROCm0]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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

static std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> r;
    std::string x;
    for (char c : s + ",") {
        if (c == ',') { if (!x.empty()) r.push_back(x); x.clear(); } else x += c;
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

static double run_graph(ggml_backend_t be, ggml_cgraph * gf, int reps, std::vector<double> & ms) {
    for (int w = 0; w < 3; ++w) {
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
    }
    ms.clear();
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return median(ms);
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: w5_time rows_file reps R types [layers] [backend]\n"); return 1; }
    const std::string rowsf = argv[1];
    const int reps = atoi(argv[2]), R = atoi(argv[3]);
    const std::vector<std::string> types = split(argv[4]);
    std::set<int> layers;
    if (argc > 5 && std::string(argv[5]) != "all") for (auto & x : split(argv[5])) layers.insert(atoi(x.c_str()));
    const char * bname = argc > 6 ? argv[6] : "ROCm0";
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_name(bname, nullptr);
    if (!be) { fprintf(stderr, "no backend %s\n", bname); return 1; }
    FILE * fr = fopen(rowsf.c_str(), "rb");
    if (!fr) { perror("rows"); return 1; }
    const char * force = getenv("GGML_CUDA_SA_FORCE");
    printf("# w5_time rows %s reps %d R %d backend %s GGML_CUDA_SA_FORCE=%s\n", rowsf.c_str(), reps, R, bname, force ? force : "(unset)");
    std::map<std::string, double> sum; std::map<std::string, int> nl;
    for (int L = 0;; ++L) {
        rec h;
        if (fread(&h, sizeof(h), 1, fr) != 1) break;
        std::vector<int32_t> rows((size_t) h.n_q * h.ns);
        if (fread(rows.data(), 4, rows.size(), fr) != rows.size()) { fprintf(stderr, "short rows\n"); return 1; }
        if (!layers.empty() && !layers.count(L)) continue;
        const int n_q = h.n_q, ns = h.ns, nk = h.nk, nk4 = 4 * ((nk + 3) / 4);
        // SYN data (w3_attn.cu): uniform +-1.7, per layer
        std::mt19937_64 rg(0xC18ULL + (uint64_t) L * 7919ULL + (uint64_t) nk);
        auto u = [&]() { return (float) ((rg() >> 40) * (2.0 / 16777216.0) - 1.0) * 1.7f; };
        std::vector<float> Q((size_t) n_q * 24 * 256), K((size_t) nk4 * 512, 0.0f), V((size_t) nk4 * 512, 0.0f);
        for (auto & x : Q) x = u();
        for (size_t i = 0; i < (size_t) nk * 512; ++i) K[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(u()));
        for (size_t i = 0; i < (size_t) nk * 512; ++i) V[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(u()));
        for (const std::string & tn : types) {
            const ggml_type t = type_of(tn);
            ggml_init_params ip = { ggml_tensor_overhead() * (R + 16) + 2 * ggml_graph_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * q_st = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 24, n_q, 1);
            ggml_tensor * k_st = ggml_new_tensor_4d(ctx, t, 256, 2, nk4, 1);
            ggml_tensor * v_st = ggml_new_tensor_4d(ctx, t, 256, 2, nk4, 1);
            ggml_tensor * sel  = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, ns, n_q, 1, 1);
            ggml_tensor * q = ggml_permute(ctx, q_st, 0, 2, 1, 3);
            ggml_tensor * k = ggml_permute(ctx, k_st, 0, 2, 1, 3);
            ggml_tensor * v = ggml_permute(ctx, v_st, 0, 2, 1, 3);
            ggml_tensor * plan = ggml_sparse_attn_plan(ctx, sel, nk, 4);
            ggml_cgraph * gp = ggml_new_graph(ctx), * ga = ggml_new_graph(ctx);
            ggml_set_output(plan);
            ggml_build_forward_expand(gp, plan);
            for (int r = 0; r < R; ++r) {
                ggml_tensor * o = ggml_sparse_attn(ctx, q, k, v, plan, 1.0f / 16.0f, 0, 0);
                ggml_set_output(o);
                ggml_build_forward_expand(ga, o);
            }
            bool ok = true;
            for (int i = 0; i < ggml_graph_n_nodes(ga); ++i) ok = ok && ggml_backend_supports_op(be, ggml_graph_node(ga, i));
            if (!ok) { printf("L%d %s not supported\n", L, tn.c_str()); ggml_free(ctx); continue; }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }
            ggml_backend_tensor_set(q_st, Q.data(), 0, ggml_nbytes(q_st));
            ggml_backend_tensor_set(sel, rows.data(), 0, ggml_nbytes(sel));
            for (auto pr : { std::make_pair(k_st, &K), std::make_pair(v_st, &V) }) {
                std::vector<uint8_t> qb(ggml_nbytes(pr.first));
                ggml_quantize_chunk(t, pr.second->data(), qb.data(), 0, (int64_t) nk4 * 2, 256, nullptr);
                ggml_backend_tensor_set(pr.first, qb.data(), 0, qb.size());
            }
            std::vector<double> ms;
            const double tp = run_graph(be, gp, reps, ms);
            const double ta = run_graph(be, ga, reps, ms);
            const double att = (ta - tp) / R;
            printf("L%d RES %-8s n_q %d nk %d plan_med %.4f graph_med %.4f attend %.4f ms\n", L, tn.c_str(), n_q, nk, tp, ta, att);
            fflush(stdout);
            sum[tn] += att; nl[tn]++;
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }
    fclose(fr);
    printf("SUMMARY (mean over layers of the per-layer attend time, ms; ratio to f16 of the same run)\n");
    const double f16 = sum.count("f16") ? sum["f16"] / nl["f16"] : 0.0;
    for (auto & kv : sum) {
        const double m = kv.second / nl[kv.first];
        printf("SUM %-8s layers %d attend %.4f ratio_to_f16 %s\n", kv.first.c_str(), nl[kv.first], m,
               f16 > 0 ? std::to_string(m / f16).c_str() : "n/a");
    }
    ggml_backend_free(be);
    return 0;
}
