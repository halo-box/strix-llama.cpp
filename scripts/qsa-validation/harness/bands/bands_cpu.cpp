// R1-M-Lab A, step 2: CPU-only bands on identical rows.
//   (a) CPU FA + src[5]: ggml_flash_attn_ext + ggml_flash_attn_ext_add_top_k(n_kv_raw 0) on the ggml CPU backend of the
//       PR tree (94e086847, ops.cpp:8946 ggml_compute_forward_flash_attn_ext_f16_one_chunk), prec F32, no mask, scale 1/16,
//       exactly the graph of test_qsa_prefill_maskless (tests/test-backend-ops.cpp:11531-11542).
//   (b) the §3 contract (bands_ref.h O_CON), (c) the f64 oracles O_PLAN and O_C18.
//   For every pair in {f32, f16, bf16, q8_0, q4_0}^2. Planted defects on the contract: d1 last tail cell -> -1, d2 the median
//   block -> -1 (R1-Q8 lab definitions), d3 V x (1 + 1e-3); each reported against the clean contract.
// g++ -O3 -march=native -ffp-contract=off -std=c++17 -I<tree>/ggml/include bands_cpu.cpp -L<tree>/build/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,<tree>/build/bin -lpthread -o bands_cpu
// usage: bands_cpu <rows|SYN:U:nk:seed> <qkv|SYN> <layers|all> <stride> <tag> [nthr] [defects 0|1] [pairs all|gpu]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "bands_ref.h"
#include <map>

using namespace bands;

static ggml_type gt(int T) { switch (T) { case BT_F16: return GGML_TYPE_F16; case BT_Q8: return GGML_TYPE_Q8_0; case BT_Q4: return GGML_TYPE_Q4_0; case BT_BF: return GGML_TYPE_BF16; default: return GGML_TYPE_F32; } }

// CPU FA + src[5] over all n_q queries; out [n_q][24][256]
static void cpu_fa(ggml_backend_t be, const fixture & F, const std::vector<int32_t> & rows, int KT, const std::vector<uint8_t> & KC,
                   int VT, const std::vector<uint8_t> & VC, std::vector<float> & out) {
    const int n_q = F.n_q, nk = F.nk, ns = F.ns;
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 24, n_q, 1);
    ggml_tensor * k0 = ggml_new_tensor_4d(ctx, gt(KT), 256, 2, nk, 1);
    ggml_tensor * v0 = ggml_new_tensor_4d(ctx, gt(VT), 256, 2, nk, 1);
    ggml_tensor * ids = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, ns, n_q, 1, 1);
    ggml_tensor * q = ggml_permute(ctx, q0, 0, 2, 1, 3), * k = ggml_permute(ctx, k0, 0, 2, 1, 3), * v = ggml_permute(ctx, v0, 0, 2, 1, 3);
    ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / 16.0f, 0.0f, 0.0f);
    ggml_flash_attn_ext_add_top_k(o, ids, 0);
    ggml_prec_set_acc(o, GGML_PREC_F32);
    ggml_cgraph * gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (ggml_nbytes(k0) != KC.size() || ggml_nbytes(v0) != VC.size()) { fprintf(stderr, "cache size mismatch\n"); exit(1); }
    ggml_backend_tensor_set(q0, F.Q.data(), 0, ggml_nbytes(q0));
    ggml_backend_tensor_set(k0, KC.data(), 0, KC.size());
    ggml_backend_tensor_set(v0, VC.data(), 0, VC.size());
    ggml_backend_tensor_set(ids, rows.data(), 0, ggml_nbytes(ids));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
    if (!(o->ne[0] == 256 && o->ne[1] == 24 && o->ne[2] == n_q)) { fprintf(stderr, "out shape\n"); exit(1); }
    out.resize((size_t) n_q * 24 * 256); ggml_backend_tensor_get(o, out.data(), 0, out.size() * 4);
    ggml_backend_buffer_free(buf); ggml_free(ctx);
}

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: rows qkv layers stride tag [nthr] [defects] [pairs]\n"); return 1; }
    const std::string rowsf = argv[1], qkvf = argv[2], ls = argv[3], tag = argv[5];
    const int stride = atoi(argv[4]), nthr = argc > 6 ? atoi(argv[6]) : 32, defects = argc > 7 ? atoi(argv[7]) : 1;
    const std::string pairsel = argc > 8 ? argv[8] : "all";
    std::set<int> layers; const bool lall = ls == "all";
    if (!lall) { std::string x; for (char c : ls + ",") { if (c == ',') { if (!x.empty()) layers.insert(atoi(x.c_str())); x.clear(); } else x += c; } }
    ggml_backend_t be = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(be, nthr);
    const int TT[5] = {BT_F32, BT_F16, BT_BF, BT_Q8, BT_Q4};
    std::vector<std::pair<int,int>> pairs;
    if (pairsel == "gpu") pairs = {{BT_F16,BT_F16},{BT_Q8,BT_Q8},{BT_Q4,BT_Q4},{BT_BF,BT_BF},{BT_Q8,BT_F16},{BT_Q4,BT_F16},{BT_BF,BT_F16},{BT_F16,BT_Q8},{BT_F16,BT_Q4},{BT_F16,BT_BF}};
    else for (int a : TT) for (int b : TT) pairs.push_back({a, b});
    fixture_reader R; if (!R.open(rowsf, qkvf)) return 1;
    printf("# bands_cpu tag %s rows %s data %s layers %s stride %d nthr %d defects %d pairs %zu\n", tag.c_str(), rowsf.c_str(), qkvf.c_str(), ls.c_str(), stride, nthr, defects, pairs.size());
    std::map<std::string, errstat> pooled; std::map<std::string, double> lmax;
    auto rec = [&](const std::string & k, int L, const errstat & es) {
        printf("CPUB %s L%d %s %s\n", tag.c_str(), L, k.c_str(), fmt_stat(es).c_str()); fflush(stdout);
        pooled[k].merge(es); lmax[k] = std::max(lmax[k], es.nmse());
    };
    fixture F;
    while (R.next(F, [&](int L) { return lall || layers.count(L); })) {
        const int L = F.L; if (!(lall || layers.count(L))) continue;
        std::vector<int> sel; for (int q = 0; q < F.n_q; q += stride) sel.push_back(q);
        std::vector<uint8_t> KC[6], VC[6]; deq KD[6], VD[6];
        for (int T : TT) { build_cache(T, F.K16, F.nk, KC[T]); build_cache(T, F.V16, F.nk, VC[T]); build_deq(T, KC[T], F.nk, KD[T]); build_deq(T, VC[T], F.nk, VD[T]); }
        for (int T : TT) printf("CACHE_HASH %s L%d K %s %016llx\nCACHE_HASH %s L%d V %s %016llx\n", tag.c_str(), L, tname(T), (unsigned long long) fnv1a(KC[T].data(), KC[T].size()), tag.c_str(), L, tname(T), (unsigned long long) fnv1a(VC[T].data(), VC[T].size()));
        printf("ROWS_HASH %s L%d %016llx Q_HASH %016llx\n", tag.c_str(), L, (unsigned long long) fnv1a(F.rows.data(), F.rows.size() * 4), (unsigned long long) fnv1a(F.Q.data(), F.Q.size() * 4));
        // the harness caches against ggml's own quantizers (ggml_quantize_chunk: q8_0/q4_0 _ref, bf16 RNE)
        for (int T : {BT_Q8, BT_Q4, BT_BF}) {
            std::vector<float> f32((size_t) F.nk * 512); for (size_t i = 0; i < f32.size(); ++i) f32[i] = h2f(F.K16[i]);
            std::vector<uint8_t> g(KC[T].size());
            const size_t nb = ggml_quantize_chunk(gt(T), f32.data(), g.data(), 0, (int64_t) F.nk * 2, 256, nullptr);
            printf("QUANT_CHECK %s L%d %s bytes %zu ggml %zu identical %d\n", tag.c_str(), L, tname(T), KC[T].size(), nb, (int) (nb == g.size() && memcmp(g.data(), KC[T].data(), nb) == 0));
        }
        std::vector<int32_t> rows_d[3] = {F.rows, F.rows, F.rows};
        for (int q = 0; q < F.n_q; ++q) { defect_row(1, rows_d[1].data() + (size_t) q * F.ns, F.ns); defect_row(2, rows_d[2].data() + (size_t) q * F.ns, F.ns); }
        { size_t c1 = 0, c2 = 0; for (size_t i = 0; i < F.rows.size(); ++i) { c1 += F.rows[i] >= 0 && rows_d[1][i] < 0; c2 += F.rows[i] >= 0 && rows_d[2][i] < 0; }
          printf("DEFECT_ROWS %s L%d d1_cells_dropped %zu d2_cells_dropped %zu n_q %d\n", tag.c_str(), L, c1, c2, F.n_q); }
        for (auto pr : pairs) {
            const int KT = pr.first, VT = pr.second; const std::string pn = std::string(tname(KT)) + "/" + tname(VT);
            std::vector<float> fa; cpu_fa(be, F, F.rows, KT, KC[KT], VT, VC[VT], fa);
            std::vector<double> ref[3];
            for (int m = 0; m < 3; ++m) oracle_rows(m, F, sel, F.rows, KD[KT], VD[VT], 1.0f / 16.0f, 1.0f, ref[m], nthr);
            auto cmp_fa = [&](int m) { errstat es; for (size_t si = 0; si < sel.size(); ++si) for (int hq = 0; hq < 24; ++hq) es.add_row(ref[m].data() + (si * 24 + hq) * 256, fa.data() + ((size_t) sel[si] * 24 + hq) * 256); return es; };
            auto cmp_rr = [&](const std::vector<double> & r, const std::vector<double> & t) { errstat es; for (size_t k = 0; k < sel.size() * 24; ++k) es.add_row(r.data() + k * 256, t.data() + k * 256); return es; };
            rec(pn + " cpufa_vs_contract", L, cmp_fa(O_CON));
            rec(pn + " cpufa_vs_f64_plan", L, cmp_fa(O_PLAN));
            rec(pn + " cpufa_vs_f64_c18", L, cmp_fa(O_C18));
            rec(pn + " contract_vs_f64_plan", L, cmp_rr(ref[O_PLAN], ref[O_CON]));
            rec(pn + " contract_vs_f64_c18", L, cmp_rr(ref[O_C18], ref[O_CON]));
            rec(pn + " f64_c18_vs_f64_plan", L, cmp_rr(ref[O_PLAN], ref[O_C18]));
            const bool gpu_pair = (KT != BT_F32 && VT != BT_F32) && (KT == VT || KT == BT_F16 || VT == BT_F16);
            if (defects && gpu_pair) {
                std::vector<double> d;
                oracle_rows(O_CON, F, sel, rows_d[1], KD[KT], VD[VT], 1.0f / 16.0f, 1.0f, d, nthr); rec(pn + " defect_d1_tailcell_vs_contract", L, cmp_rr(ref[O_CON], d));
                oracle_rows(O_CON, F, sel, rows_d[2], KD[KT], VD[VT], 1.0f / 16.0f, 1.0f, d, nthr); rec(pn + " defect_d2_block_vs_contract", L, cmp_rr(ref[O_CON], d));
                oracle_rows(O_CON, F, sel, F.rows, KD[KT], VD[VT], 1.0f / 16.0f, 1.001f, d, nthr);  rec(pn + " defect_d3_v1e-3_vs_contract", L, cmp_rr(ref[O_CON], d));
            }
        }
    }
    for (auto & kv : pooled) printf("CPUSUM %s %s nmse_layer_max %.4e %s\n", tag.c_str(), kv.first.c_str(), lmax[kv.first], fmt_stat(kv.second).c_str());
    ggml_backend_free(be);
    return 0;
}
