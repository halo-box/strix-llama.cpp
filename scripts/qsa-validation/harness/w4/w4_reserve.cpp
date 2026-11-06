// AT-C5 at op level (PLAN-v3 §16.5 AT-C, §7.1 item 1, §5 "workspace size"), W4: the reserve of a SPARSE_ATTN_PLAN +
// SPARSE_ATTN graph through ggml_backend_sched covers every runtime ubatch shape and every G the HIP group policy can
// return, including a forced G flip between reserve and runtime, with 0 re-plans.
//
// Setup (the llama layout): ROCm0 + CPU in one scheduler; K / V caches [256, n_head_kv 2, n_ctx_pad] of type T in a
// ROCm0 buffer of their own, viewed as [256, n_kv_pad, 2] (n_kv padded to 256, llama-kv-cache.cpp:1347-1352); q
// [256, 24, n_q] F32 permuted to the FLASH_ATTN_EXT layout; sel I32 [2051, n_q] (top_k + r - 1); plan with G; attend; a
// MUL consumer. q and sel are graph inputs allocated by the scheduler (as llama's).
//   Placement: the plan and the attend are pinned to ROCm0 (ggml_backend_sched_set_tensor_backend), as llama pins them to
//   the backend of the layer's K buffer (PLAN-v3 §10). Unpinned, the scheduler puts the plan on the CPU next to its host
//   input sel and hands the HIP attend a copy of the plan dst as src[3], which the HIP backend rejects ("malformed
//   SPARSE_ATTN node": n_sel, n_kv and G live in the plan node, not in its bytes); finding F-W4-1 for W7's guard.
//   1. Reserve: ggml_backend_sched_reserve on the graph at (n_q = ub, n_kv = n_ctx) with G = the HIP group proc
//      (ggml_backend_sparse_attn_group, GGML_CUDA_SA_FORCE unset). Records the per-backend buffer sizes.
//   2. Formula check: the reserved plan node's alloc size >= max over G of plan_nbytes + scratch (§2.1); the reserved
//      attend node's alloc size >= round_up(nbytes, 256) + the largest G1 partial buffer over n_q' <= ub computed here
//      independently with PR #219's rule (qsa-decode.cu:134-145, clamp to the plan cap), and >= every runtime node's
//      alloc size (below).
//   3. Runtime: for every n_q' in 1..ub, n_kv' in {2052, n_ctx / 4, n_ctx / 2, n_ctx} (padded to 256, >= 2052), G in
//      {policy, forced g4, forced g1} (the forced G comes from the same proc with GGML_CUDA_SA_FORCE set, read at every
//      call): build the graph, ggml_backend_sched_alloc_graph, compute every 8th shape and the named ones (n_q' 1, 65,
//      66, 127, 128, 129, ub) with random rows / q, then require: no realloc callback (the scheduler's realloc counter),
//      buffer sizes unchanged (strict reserve), finite output. GGML_SCHED_DEBUG_REALLOC=1 also aborts on an
//      unexpected re-plan inside the scheduler (ggml-backend.cpp:1958-1966).
// usage: w4_reserve <type f16|q8_0|q4_0|bf16> <ub> <n_ctx> [compute_stride]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int64_t pad256(int64_t x) { return (x + 255) / 256 * 256; }

// PLAN-v3 §5 / R1-C: G1 split-K partial bytes with PR #219's decode rule, recomputed here (not the backend's function)
static int64_t pr_splits(int64_t nq, int64_t ns, int64_t nk) {
    const int64_t step = 64;
    int64_t chunk = step;
    if (nq > 8) {
        const int64_t target = std::max<int64_t>(1, (8 * ((ns + step - 1) / step) + nq - 1) / nq);
        chunk = ((ns + target - 1) / target + 15) / 16 * 16;
    }
    const int64_t s = (ns + chunk - 1) / chunk;
    const int64_t cap = ((std::min(ns, (nk + 3) / 4) + 3) / 4) * 4;
    return std::max<int64_t>(1, std::min(std::min(s, cap / 4), (int64_t) 65535));
}
static size_t g1_partials(int64_t nq, int64_t ns, int64_t nk, int64_t nh, int64_t nst) {
    const int64_t s = pr_splits(nq, ns, nk);
    return s > 1 ? (size_t) (nst * nq * nh * s * 258 * 4) : 0;
}

struct graph_t {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * q = nullptr, * sel = nullptr, * plan = nullptr, * att = nullptr, * out = nullptr;
};

static graph_t build(ggml_tensor * kc, ggml_tensor * vc, int64_t nq, int64_t nkv, int G, int64_t ns) {
    graph_t g;
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    ggml_tensor * qs = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 256, 24, nq);
    ggml_set_input(qs); ggml_set_name(qs, "q_in");
    g.q = qs;
    ggml_tensor * q = ggml_permute(g.ctx, qs, 0, 2, 1, 3);
    g.sel = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, ns, nq);
    ggml_set_input(g.sel); ggml_set_name(g.sel, "sel");
    ggml_tensor * k = ggml_view_3d(g.ctx, kc, 256, nkv, 2, kc->nb[2], kc->nb[1], 0);
    ggml_tensor * v = ggml_view_3d(g.ctx, vc, 256, nkv, 2, vc->nb[2], vc->nb[1], 0);
    g.plan = ggml_sparse_attn_plan(g.ctx, g.sel, nkv, G);
    g.att  = ggml_sparse_attn(g.ctx, q, k, v, g.plan, 0.0625f, 0, 0);
    g.out  = ggml_scale(g.ctx, g.att, 1.0f);
    ggml_set_output(g.out);
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    return g;
}

static int n_realloc = 0;
static void on_realloc(ggml_backend_sched_t, ggml_cgraph *, bool ids, bool size, void *) {
    ++n_realloc;
    fprintf(stderr, "REALLOC callback (backend_ids_changed %d graph_size_changed %d)\n", (int) ids, (int) size);
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: w4_reserve type ub n_ctx [compute_stride]\n"); return 1; }
    const std::string ts = argv[1];
    const int64_t ub = atoll(argv[2]), n_ctx = pad256(atoll(argv[3]));
    const int stride = argc > 4 ? atoi(argv[4]) : 8;
    const ggml_type T = ts == "f16" ? GGML_TYPE_F16 : ts == "q8_0" ? GGML_TYPE_Q8_0 : ts == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_BF16;
    const int64_t ns = 2051;
    unsetenv("GGML_CUDA_SA_FORCE");
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("ROCm0");
    if (!dev) { fprintf(stderr, "no ROCm0\n"); return 1; }
    ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr), cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    auto group = (ggml_backend_sparse_attn_group_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_sparse_attn_group");
    if (!group) { fprintf(stderr, "no ggml_backend_sparse_attn_group proc\n"); return 1; }

    // caches
    ggml_init_params kp = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * kctx = ggml_init(kp);
    ggml_tensor * kc = ggml_new_tensor_3d(kctx, T, 256, 2, n_ctx);
    ggml_tensor * vc = ggml_new_tensor_3d(kctx, T, 256, 2, n_ctx);
    ggml_backend_buffer_t kbuf = ggml_backend_alloc_ctx_tensors(kctx, gpu);
    {   // random cache contents (one row per cell and kv head, quantized from U(-1, 1))
        std::mt19937 rng(5); std::uniform_real_distribution<float> u(-1.f, 1.f);
        std::vector<float> f(256); const size_t rb = ggml_row_size(T, 256);
        std::vector<uint8_t> row(rb), all(rb * 2 * n_ctx);
        for (int64_t i = 0; i < 2 * n_ctx; ++i) { for (auto & x : f) x = u(rng); ggml_quantize_chunk(T, f.data(), all.data() + i * rb, 0, 1, 256, nullptr); }
        ggml_backend_tensor_set(kc, all.data(), 0, all.size());
        ggml_backend_tensor_set(vc, all.data(), 0, all.size());
    }
    ggml_backend_t backs[2] = { gpu, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backs, nullptr, 2, 4096, false, true);
    ggml_backend_sched_set_realloc_callback(sched, on_realloc, nullptr);

    const int G_res = group(dev, ub, n_ctx, T, T, false);
    graph_t R = build(kc, vc, ub, n_ctx, G_res, ns);
    ggml_backend_sched_set_tensor_backend(sched, R.plan, gpu);
    ggml_backend_sched_set_tensor_backend(sched, R.att, gpu);
    // the formula checks read the reserve graph's own nodes before the scheduler rewires any source
    const size_t a_plan = ggml_backend_buft_get_alloc_size(ggml_backend_dev_buffer_type(dev), R.plan);
    const size_t a_att  = ggml_backend_buft_get_alloc_size(ggml_backend_dev_buffer_type(dev), R.att);
    if (!ggml_backend_sched_reserve(sched, R.gf)) { fprintf(stderr, "reserve failed\n"); return 1; }
    const size_t res_gpu = ggml_backend_sched_get_buffer_size(sched, gpu), res_cpu = ggml_backend_sched_get_buffer_size(sched, cpu);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    size_t plan_formula = 0;
    for (int G : {1, 4}) plan_formula = std::max(plan_formula, ggml_sparse_attn_plan_nbytes(ns, ub, 1, n_ctx, G));
    plan_formula += ggml_sparse_attn_plan_scratch_nbytes(ns, ub, 1);
    size_t part_max = 0; int64_t part_arg = 0;
    for (int64_t nq = 1; nq <= ub; ++nq) {
        for (int64_t nk : {(int64_t) 2052, n_ctx}) {
            const size_t p = g1_partials(nq, ns, nk, 24, 1);
            if (p > part_max) { part_max = p; part_arg = nq; }
        }
    }
    const size_t att_formula = (size_t) pad256(ggml_nbytes(R.att)) + part_max;
    printf("RESERVE type %s ub %lld n_ctx %lld G_policy %d gpu %zu cpu %zu | plan alloc %zu >= formula %zu: %s | attend alloc %zu >= nbytes + G1 partial max %zu (n_q' %lld): %s\n",
           ts.c_str(), (long long) ub, (long long) n_ctx, G_res, res_gpu, res_cpu, a_plan, plan_formula, a_plan >= plan_formula ? "OK" : "FAIL",
           a_att, att_formula, (long long) part_arg, a_att >= att_formula ? "OK" : "FAIL");
    bool ok = a_plan >= plan_formula && a_att >= att_formula;
    ggml_free(R.ctx);

    std::mt19937 rng(9);
    const std::vector<int64_t> kvs = { 2052, std::max<int64_t>(2052, pad256(n_ctx / 4)), std::max<int64_t>(2052, pad256(n_ctx / 2)), n_ctx };
    const int64_t named[] = { 1, 65, 66, 127, 128, 129 };
    size_t n_shapes = 0, n_computed = 0, n_bad = 0, max_att_rt = 0, max_plan_rt = 0;
    for (const char * force : { "", "g4", "g1" }) {
        if (*force) setenv("GGML_CUDA_SA_FORCE", force, 1); else unsetenv("GGML_CUDA_SA_FORCE");
        for (int64_t nkv : kvs) {
            if (nkv > n_ctx) continue;
            for (int64_t nq = 1; nq <= ub; ++nq) {
                const int G = group(dev, nq, nkv, T, T, false);
                if (*force && G != (force[1] - '0')) { fprintf(stderr, "group proc ignored GGML_CUDA_SA_FORCE=%s (G %d)\n", force, G); return 1; }
                graph_t g = build(kc, vc, nq, nkv, G, ns);
                ggml_backend_sched_set_tensor_backend(sched, g.plan, gpu);
                ggml_backend_sched_set_tensor_backend(sched, g.att, gpu);
                max_att_rt  = std::max(max_att_rt, ggml_backend_buft_get_alloc_size(buft, g.att));
                max_plan_rt = std::max(max_plan_rt, ggml_backend_buft_get_alloc_size(buft, g.plan));
                if (!ggml_backend_sched_alloc_graph(sched, g.gf)) { fprintf(stderr, "alloc failed\n"); return 1; }
                ++n_shapes;
                bool named_shape = nq == ub;
                for (int64_t x : named) named_shape = named_shape || nq == x;
                if (named_shape || nq % stride == 0) {
                    std::vector<float> qv((size_t) 256 * 24 * nq); std::uniform_real_distribution<float> u(-1.f, 1.f);
                    for (auto & x : qv) x = u(rng);
                    ggml_backend_tensor_set(g.q, qv.data(), 0, qv.size() * 4);
                    std::vector<int32_t> rows((size_t) ns * nq);
                    for (auto & x : rows) x = (int32_t) (rng() % (uint64_t) nkv);
                    ggml_backend_tensor_set(g.sel, rows.data(), 0, rows.size() * 4);
                    if (ggml_backend_sched_graph_compute(sched, g.gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
                    std::vector<float> o(ggml_nelements(g.out)); ggml_backend_tensor_get(g.out, o.data(), 0, o.size() * 4);
                    for (float x : o) n_bad += !std::isfinite(x);
                    ++n_computed;
                }
                const size_t gs = ggml_backend_sched_get_buffer_size(sched, gpu), cs = ggml_backend_sched_get_buffer_size(sched, cpu);
                if (gs != res_gpu || cs != res_cpu) {
                    fprintf(stderr, "STRICT violation at n_q %lld n_kv %lld G %d force '%s': gpu %zu (reserved %zu) cpu %zu (reserved %zu)\n",
                            (long long) nq, (long long) nkv, G, force, gs, res_gpu, cs, res_cpu);
                    ok = false;
                }
                ggml_backend_sched_reset(sched);
                ggml_free(g.ctx);
            }
        }
    }
    unsetenv("GGML_CUDA_SA_FORCE");
    const bool cover = a_att >= max_att_rt && a_plan >= max_plan_rt;
    ok = ok && n_realloc == 0 && n_bad == 0 && cover;
    printf("RUNTIME type %s ub %lld n_ctx %lld shapes %zu computed %zu nonfinite %zu reallocs %d | max runtime attend alloc %zu <= reserved %zu, plan %zu <= %zu: %s | %s\n",
           ts.c_str(), (long long) ub, (long long) n_ctx, n_shapes, n_computed, n_bad, n_realloc, max_att_rt, a_att, max_plan_rt, a_plan,
           cover ? "OK" : "FAIL", ok ? "PASS" : "FAIL");
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(kbuf);
    ggml_free(kctx);
    ggml_backend_free(gpu); ggml_backend_free(cpu);
    return ok ? 0 : 2;
}
