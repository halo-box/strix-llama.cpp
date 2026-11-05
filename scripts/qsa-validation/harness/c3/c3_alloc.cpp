// C3 (b): does an integer "plan" tensor with exactly one consumer survive gallocr memory reuse, scheduler splits and
// HIP graph capture/replay? Surrogate ops (no plan op exists yet): plan = ARGSORT (I32, the producer), attend = GET_ROWS
// reading the plan (the single consumer). All values are small integers so every op is exact; the output is compared
// bit for bit (memcmp) against a host reference on every iteration, with fresh input data each iteration.
// Topologies:  A (2 splits): CPU split [c = x*2] -> ROCm split [plan, pressure chain, attend(plan), + c, tail chain]
//              B (3 splits): ROCm [plan, pressure] -> CPU [c] -> ROCm [attend(plan) + c, tail]  (plan lives across a split)
// Modes: reuse (one cgraph, allocated once, recomputed 20x: HIP graph warmup, capture and replay) or rebuild (new
// context, sched reset + alloc every iteration, as llama.cpp does when the graph cannot be reused).
// usage: c3_alloc A|B reuse|rebuild iters
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>

static int g_capture = 0, g_reused = 0, g_reset = 0;
static void logcb(ggml_log_level level, const char * text, void *) {
    if (strstr(text, "warmup complete")) g_capture++;
    if (strstr(text, "warmup reset")) g_reset++;
    if (strstr(text, "reused")) g_reused++;
    if (level >= GGML_LOG_LEVEL_WARN || getenv("C3_VERBOSE")) fputs(text, stderr);
}

static const int NS = 1024, NQ = 256, D = 64, NK = 4096, NP = 6, NT = 6, NL = 12;

struct built { ggml_context * ctx; ggml_cgraph * gf; ggml_tensor *scores, *kv, *x, *plan, *attend, *out, *out2; std::vector<ggml_tensor*> tail; };

static built build(char topo, ggml_backend_sched_t sched, ggml_backend_t gpu, ggml_backend_t cpu) {
    built b;
    ggml_init_params ip = { ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true };
    b.ctx = ggml_init(ip);
    ggml_context * ctx = b.ctx;
    b.scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NS, NQ); ggml_set_name(b.scores, "scores"); ggml_set_input(b.scores);
    b.kv     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, NK);  ggml_set_name(b.kv, "kv");         ggml_set_input(b.kv);
    b.x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 1);   ggml_set_name(b.x, "x");           ggml_set_input(b.x);
    b.gf = ggml_new_graph_custom(ctx, 256, false);
    ggml_tensor * c = nullptr;
    auto cpu_split = [&]() {
        c = ggml_scale(ctx, b.x, 2.0f); ggml_set_name(c, "c_cpu");
        ggml_build_forward_expand(b.gf, c);
        ggml_backend_sched_set_tensor_backend(sched, c, cpu);
    };
    if (topo == 'A') cpu_split();
    b.plan = ggml_argsort(ctx, b.scores, GGML_SORT_ORDER_ASC); ggml_set_name(b.plan, "plan");
    ggml_build_forward_expand(b.gf, b.plan);
    ggml_backend_sched_set_tensor_backend(sched, b.plan, gpu);
    // pressure: NP intermediates of the plan's size class, each dead right after its use, allocated while the plan is live
    ggml_tensor * t = b.kv;
    for (int i = 0; i < NP; ++i) {
        ggml_tensor * u = ggml_add(ctx, t, b.kv);           // 1 MiB each
        u = ggml_sub(ctx, u, t);                           // == kv
        t = ggml_add(ctx, u, t);                           // t_{i+1} = t_i + kv
        char nm[32]; snprintf(nm, sizeof(nm), "press%d", i); ggml_set_name(t, nm);
        ggml_build_forward_expand(b.gf, t);
        ggml_backend_sched_set_tensor_backend(sched, t, gpu);
    }
    if (topo == 'B') cpu_split();
    b.attend = ggml_get_rows(ctx, t, ggml_reshape_1d(ctx, b.plan, NS * NQ));   // [D, NS*NQ], reads the plan once
    ggml_set_name(b.attend, "attend");
    ggml_backend_sched_set_tensor_backend(sched, b.attend, gpu);
    ggml_tensor * o = ggml_add(ctx, b.attend, c);       // broadcast the CPU-split row
    // tail: tensors allocated after the plan died, so gallocr may hand them the plan's memory
    for (int i = 0; i < NT; ++i) {
        ggml_tensor * s = ggml_scale(ctx, o, 1.0f); char nm[32]; snprintf(nm, sizeof(nm), "tail%d", i); ggml_set_name(s, nm);
        b.tail.push_back(s);
        o = ggml_sub(ctx, ggml_add(ctx, o, s), s);       // == o
    }
    b.out = o; ggml_set_name(b.out, "out"); ggml_set_output(b.out);
    ggml_build_forward_expand(b.gf, b.out);
    // late: NL plan-sized (1 MiB) tensors created after the plan's only consumer ran and all alive at once (summed at the
    // end), so the allocator must place one of them in the plan's freed bytes (checked below); their values are checked too
    std::vector<ggml_tensor *> late;
    for (int i = 0; i < NL; ++i) {
        ggml_tensor * l = ggml_scale(ctx, b.kv, (float) (i + 1)); char nm[32]; snprintf(nm, sizeof(nm), "late%d", i); ggml_set_name(l, nm);
        ggml_build_forward_expand(b.gf, l);
        ggml_backend_sched_set_tensor_backend(sched, l, gpu);
        late.push_back(l);
    }
    ggml_tensor * acc = late[0];
    for (int i = 1; i < NL; ++i) { acc = ggml_add(ctx, acc, late[i]); ggml_backend_sched_set_tensor_backend(sched, acc, gpu); }
    b.out2 = ggml_scale(ctx, acc, 1.0f); ggml_set_name(b.out2, "out2"); ggml_set_output(b.out2);
    ggml_build_forward_expand(b.gf, b.out2);
    return b;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: c3_alloc A|B reuse|rebuild iters\n"); return 1; }
    const char topo = argv[1][0];
    const bool rebuild = std::string(argv[2]) == "rebuild";
    const int iters = atoi(argv[3]);
    ggml_log_set(logcb, nullptr);
    ggml_backend_load_all();
    ggml_backend_dev_t gdev = ggml_backend_dev_by_name("ROCm0");
    if (!gdev) { fprintf(stderr, "no GPU\n"); return 1; }
    ggml_backend_t gpu = ggml_backend_dev_init(gdev, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t backends[2] = { gpu, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 1024, false, false);
    printf("C3B topo=%c mode=%s iters=%d gpu=%s graphs_env=%s\n", topo, rebuild ? "rebuild" : "reuse", iters, ggml_backend_dev_name(gdev),
           getenv("GGML_CUDA_DISABLE_GRAPHS") ? getenv("GGML_CUDA_DISABLE_GRAPHS") : "unset");
    std::mt19937 rng(1234);
    built b{};
    int fails = 0, n_splits = -1;
    bool reused_mem = false;
    for (int it = 0; it < iters; ++it) {
        if (it == 0 || rebuild) {
            if (it > 0) { ggml_free(b.ctx); }
            ggml_backend_sched_reset(sched);
            b = build(topo, sched, gpu, cpu);
            if (!ggml_backend_sched_alloc_graph(sched, b.gf)) { fprintf(stderr, "alloc failed\n"); return 1; }
            n_splits = ggml_backend_sched_get_n_splits(sched);
            // does a tensor allocated after the plan's last use overlap the plan's bytes?
            const char * p0 = (const char *) b.plan->data, * p1 = p0 + ggml_nbytes(b.plan);
            std::string ov;
            for (int i = 0; i < ggml_graph_n_nodes(b.gf); ++i) {
                ggml_tensor * n = ggml_graph_node(b.gf, i);
                if (n == b.plan || n->buffer != b.plan->buffer || n->view_src) continue;
                const char * q0 = (const char *) n->data, * q1 = q0 + ggml_nbytes(n);
                if (q0 < p1 && p0 < q1) { ov += " "; ov += n->name; }
            }
            reused_mem = !ov.empty();
            if (it == 0 || it == 1) printf("ALLOC it=%d splits=%d plan_backend=%s attend_backend=%s plan_bytes=%zu overlap_with_plan_bytes:[%s]\n", it, n_splits,
                   ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched, b.plan)), ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched, b.attend)),
                   ggml_nbytes(b.plan), ov.c_str());
        }
        // fresh inputs: distinct scores (a random permutation, so ARGSORT has no ties), small-integer kv and x
        std::vector<float> sc((size_t) NS * NQ), kv((size_t) D * NK), x(D);
        std::vector<int> perm(NS);
        for (int q = 0; q < NQ; ++q) { std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), rng); for (int j = 0; j < NS; ++j) sc[(size_t) q * NS + j] = (float) perm[j]; }
        for (auto & v : kv) v = (float) ((int) (rng() % 17) - 8);
        for (auto & v : x) v = (float) ((int) (rng() % 9) - 4);
        ggml_backend_tensor_set(b.scores, sc.data(), 0, sc.size() * 4);
        ggml_backend_tensor_set(b.kv, kv.data(), 0, kv.size() * 4);
        ggml_backend_tensor_set(b.x, x.data(), 0, x.size() * 4);
        if (ggml_backend_sched_graph_compute(sched, b.gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
        std::vector<float> out((size_t) D * NS * NQ);
        ggml_backend_tensor_get(b.out, out.data(), 0, out.size() * 4);
        // host reference: t = (NP+1)*kv, plan[q][r] = index of the r-th smallest score, out = t[plan] + 2x
        std::vector<float> ref(out.size());
        for (int q = 0; q < NQ; ++q) {
            std::vector<int> idx(NS); std::iota(idx.begin(), idx.end(), 0);
            const float * s = sc.data() + (size_t) q * NS;
            std::sort(idx.begin(), idx.end(), [&](int a, int c2) { return s[a] < s[c2]; });
            for (int r = 0; r < NS; ++r) for (int d = 0; d < D; ++d)
                ref[((size_t) q * NS + r) * D + d] = (float) (NP + 1) * kv[(size_t) idx[r] * D + d] + 2.0f * x[d];
        }
        std::vector<float> out2((size_t) D * NK), ref2((size_t) D * NK);
        ggml_backend_tensor_get(b.out2, out2.data(), 0, out2.size() * 4);
        for (size_t i = 0; i < ref2.size(); ++i) ref2[i] = (float) (NL * (NL + 1) / 2) * kv[i];
        const bool ok = memcmp(out.data(), ref.data(), out.size() * 4) == 0 && memcmp(out2.data(), ref2.data(), out2.size() * 4) == 0;
        fails += !ok;
        printf("ITER %d memcmp=%s splits=%d plan_mem_reused_later=%d captures_so_far=%d resets=%d\n", it, ok ? "equal" : "DIFF", n_splits, (int) reused_mem, g_capture, g_reset);
    }
    printf("C3B_SUMMARY topo=%c mode=%s iters=%d fails=%d splits=%d graph_captures=%d warmup_resets=%d\n", topo, rebuild ? "rebuild" : "reuse", iters, fails, n_splits, g_capture, g_reset);
    ggml_free(b.ctx);
    ggml_backend_sched_free(sched); ggml_backend_free(gpu); ggml_backend_free(cpu);
    return fails ? 2 : 0;
}
