// W1 (PLAN-v3 §13 W1 done-when): the CPU GGML_OP_SPARSE_ATTN against R1-M-Lab A's oracles, on R1-M-Lab A's / R2-BOP's
// fixtures (bands_ref.h of harness/bandsop, unchanged: wiki d4k rows + .qkv captures, SYN:U:nk:seed, UNI:...):
//   (1) bit equality with bands_ref.h O_CON (oracle_group mode O_CON), every type pair of {f32, f16, bf16, q8_0, q4_0},
//       for plans with G = 1 and G = 4 (the CPU op's output does not depend on G: the same cells in the same order);
//   (2) contract vs f64: NMSE(CPU op, O_C18) and NMSE(CPU op, O_PLAN), normalised by the f64 oracle as R1-A did;
//   (3) coarse cross-check: NMSE(CPU FA + src[5], CPU op) normalised by the CPU op (R1-A "CPU-FA vs contract"), with
//       bands_cpu.cpp's FA graph (n_kv_raw 0: FA's src[5] cell = row[j]).
// Built with the flags of bands_cpu.cpp (bands_cpu.cpp:8), against the rewrite's libraries:
//   g++ -O3 -march=native -ffp-contract=off -std=c++17 -I<tree>/ggml/include -I<tree>/scripts/qsa-validation/harness/bandsop \
//       w1_ocon.cpp -L<build>/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,<build>/bin -lpthread -o w1_ocon
// usage: w1_ocon <rows|SYN:U:nk:seed|UNI:A:nk:seed:n_q:ns:blocks[:sort]> <qkv|SYN> <layers|all> <stride> <tag> [nthr] [pairs all|fast]
//        w1_ocon overflow [nthr]     the bf16 / q8_0 V overflow rule of the contract against bands_ref.h's dequant
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "bands_ref.h"
#include <map>

using namespace bands;

static ggml_type gt(int T) { switch (T) { case BT_F16: return GGML_TYPE_F16; case BT_Q8: return GGML_TYPE_Q8_0; case BT_Q4: return GGML_TYPE_Q4_0; case BT_BF: return GGML_TYPE_BF16; default: return GGML_TYPE_F32; } }

// the CPU SPARSE_ATTN over all n_q queries (plan with group G); out [n_q][24][256]
static void cpu_sparse(ggml_backend_t be, const fixture & F, const std::vector<int32_t> & rows, int KT, const std::vector<uint8_t> & KC,
                       int VT, const std::vector<uint8_t> & VC, int G, std::vector<float> & out) {
    const int n_q = F.n_q, nk = F.nk, ns = F.ns;
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 24, n_q, 1);
    ggml_tensor * k0 = ggml_new_tensor_4d(ctx, gt(KT), 256, 2, nk, 1);
    ggml_tensor * v0 = ggml_new_tensor_4d(ctx, gt(VT), 256, 2, nk, 1);
    ggml_tensor * ids = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, ns, n_q, 1, 1);
    ggml_tensor * q = ggml_permute(ctx, q0, 0, 2, 1, 3), * k = ggml_permute(ctx, k0, 0, 2, 1, 3), * v = ggml_permute(ctx, v0, 0, 2, 1, 3);
    ggml_tensor * plan = ggml_sparse_attn_plan(ctx, ids, nk, G);
    ggml_tensor * o = ggml_sparse_attn(ctx, q, k, v, plan, 1.0f / 16.0f, 0, 0);
    ggml_cgraph * gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (ggml_nbytes(k0) != KC.size() || ggml_nbytes(v0) != VC.size()) { fprintf(stderr, "cache size mismatch\n"); exit(1); }
    ggml_backend_tensor_set(q0, F.Q.data(), 0, ggml_nbytes(q0));
    ggml_backend_tensor_set(k0, KC.data(), 0, KC.size());
    ggml_backend_tensor_set(v0, VC.data(), 0, VC.size());
    ggml_backend_tensor_set(ids, rows.data(), 0, ggml_nbytes(ids));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
    // dst [256, 24, n_q, 1]: the [n_q][24][256] order of the fixture outputs
    out.resize((size_t) n_q * 24 * 256); ggml_backend_tensor_get(o, out.data(), 0, out.size() * 4);
    ggml_backend_buffer_free(buf); ggml_free(ctx);
}

// CPU FA + src[5] (bands_cpu.cpp cpu_fa, unchanged)
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
    ggml_backend_tensor_set(q0, F.Q.data(), 0, ggml_nbytes(q0));
    ggml_backend_tensor_set(k0, KC.data(), 0, KC.size());
    ggml_backend_tensor_set(v0, VC.data(), 0, VC.size());
    ggml_backend_tensor_set(ids, rows.data(), 0, ggml_nbytes(ids));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
    out.resize((size_t) n_q * 24 * 256); ggml_backend_tensor_get(o, out.data(), 0, out.size() * 4);
    ggml_backend_buffer_free(buf); ggml_free(ctx);
}

// bf16 and q8_0 V rows past the f16 range: the contract (ggml.h) gives +-inf, bands_ref.h's build_deq h values
// (bands_ref.h:311: bf16 saturated to +-65504 first, then rh) give what?
static int overflow_mode(int nthr) {
    ggml_backend_t be = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(be, nthr);
    fixture F; F.n_q = 1; F.ns = 1; F.nk = 4;
    F.Q.assign(24 * 256, 0.5f);
    F.rows = { 1 };
    int fails = 0;
    for (int VT : { BT_BF, BT_Q8 }) {
        std::vector<uint16_t> k16((size_t) F.nk * 512, f2h(0.25f));
        std::vector<uint8_t> KC; build_cache(BT_F16, k16, F.nk, KC);
        std::vector<uint8_t> VC((size_t) F.nk * 2 * hbytes(VT), 0);
        for (int kh = 0; kh < F.nk * 2; ++kh) {
            uint8_t * o = VC.data() + (size_t) kh * hbytes(VT);
            for (int d = 0; d < 256; ++d) {
                const float big = d % 2 ? -1.0e5f : 7.0e4f;
                if (VT == BT_BF) ((uint16_t *) o)[d] = f2bf(d % 5 == 0 ? big : 1.0f);
                else { bq8 * b = (bq8 *) o + d / 32; b->d = f2h(1000.0f); b->qs[d % 32] = (int8_t) (d % 5 == 0 ? (big > 0 ? 127 : -127) : 1); }
            }
        }
        deq KD, VD; build_deq(BT_F16, KC, F.nk, KD); build_deq(VT, VC, F.nk, VD);
        std::vector<float> op; cpu_sparse(be, F, F.rows, BT_F16, KC, VT, VC, 1, op);
        std::vector<double> ref; oracle_rows(O_CON, F, {0}, F.rows, KD, VD, 1.0f / 16.0f, 1.0f, ref, 1);
        int op_inf = 0, ref_inf = 0, op_inf_ok = 0;
        for (int d = 0; d < 256; ++d) {
            const bool want_inf = d % 5 == 0;   // 7e4 / -1e5 (bf16 RNE 70144 / -99840), 1000 * +-127
            op_inf += std::isinf(op[d]); ref_inf += std::isinf(ref[d]);
            op_inf_ok += want_inf == (bool) std::isinf(op[d]) && (!want_inf || std::signbit(op[d]) == (d % 2 == 1));
        }
        printf("OVERFLOW V=%s cpu_op inf=%d (expected %d, matching sign/position %d/256) bands_ref_O_CON inf=%d (first value %.9g, CPU op %.9g)\n",
               tname(VT), op_inf, 52, op_inf_ok, ref_inf, ref[0], op[0]);
        fails += op_inf_ok != 256;
    }
    ggml_backend_free(be);
    return fails ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (argc >= 2 && std::string(argv[1]) == "overflow") return overflow_mode(argc > 2 ? atoi(argv[2]) : 8);
    if (argc < 6) { fprintf(stderr, "usage: rows qkv layers stride tag [nthr] [pairs all|fast]\n"); return 1; }
    const std::string rowsf = argv[1], qkvf = argv[2], ls = argv[3], tag = argv[5];
    const int stride = atoi(argv[4]), nthr = argc > 6 ? atoi(argv[6]) : 32;
    const std::string pairsel = argc > 7 ? argv[7] : "all";
    std::set<int> layers; const bool lall = ls == "all";
    if (!lall) { std::string x; for (char c : ls + ",") { if (c == ',') { if (!x.empty()) layers.insert(atoi(x.c_str())); x.clear(); } else x += c; } }
    ggml_backend_t be = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(be, nthr);
    const int TT[5] = {BT_F32, BT_F16, BT_BF, BT_Q8, BT_Q4};
    std::vector<std::pair<int,int>> pairs;
    if (pairsel == "fast") { for (int a : {BT_F16, BT_BF, BT_Q8, BT_Q4}) for (int b : {BT_F16, BT_BF, BT_Q8, BT_Q4}) pairs.push_back({a, b}); }
    else for (int a : TT) for (int b : TT) pairs.push_back({a, b});
    fixture_reader R; if (!R.open(rowsf, qkvf)) return 1;
    printf("# w1_ocon tag %s rows %s data %s layers %s stride %d nthr %d pairs %zu\n", tag.c_str(), rowsf.c_str(), qkvf.c_str(), ls.c_str(), stride, nthr, pairs.size());
    std::map<std::string, errstat> pooled; std::map<std::string, double> lmax;
    auto rec = [&](const std::string & k, int L, const errstat & es) {
        printf("W1B %s L%d %s %s\n", tag.c_str(), L, k.c_str(), fmt_stat(es).c_str()); fflush(stdout);
        pooled[k].merge(es); lmax[k] = std::max(lmax[k], es.nmse());
    };
    size_t n_bits_total = 0, n_bits_diff = 0, n_g_diff = 0;
    fixture F;
    while (R.next(F, [&](int L) { return lall || layers.count(L); })) {
        const int L = F.L; if (!(lall || layers.count(L))) continue;
        std::vector<int> sel; for (int q = 0; q < F.n_q; q += stride) sel.push_back(q);
        std::vector<uint8_t> KC[6], VC[6]; deq KD[6], VD[6]; bool haveK[6] = {false}, haveV[6] = {false};
        for (int T : TT) {
            haveK[T] = build_cache_fx(T, F, true, KC[T]);  if (haveK[T]) build_deq(T, KC[T], F.nk, KD[T]);
            haveV[T] = build_cache_fx(T, F, false, VC[T]); if (haveV[T]) build_deq(T, VC[T], F.nk, VD[T]);
        }
        printf("ROWS_HASH %s L%d %016llx Q_HASH %016llx label %s\n", tag.c_str(), L, (unsigned long long) fnv1a(F.rows.data(), F.rows.size() * 4),
               (unsigned long long) fnv1a(F.Q.data(), F.Q.size() * 4), F.label.c_str());
        for (int T : TT) if (haveK[T]) printf("CACHE_HASH %s L%d K %s %016llx\n", tag.c_str(), L, tname(T), (unsigned long long) fnv1a(KC[T].data(), KC[T].size()));
        for (auto pr : pairs) {
            const int KT = pr.first, VT = pr.second;
            if (!haveK[KT] || !haveV[VT]) continue;
            const std::string pn = std::string(tname(KT)) + "/" + tname(VT);
            std::vector<float> op1, op4, fa;
            cpu_sparse(be, F, F.rows, KT, KC[KT], VT, VC[VT], 1, op1);
            cpu_sparse(be, F, F.rows, KT, KC[KT], VT, VC[VT], 4, op4);
            const bool g_same = memcmp(op1.data(), op4.data(), op1.size() * 4) == 0;
            n_g_diff += !g_same;
            std::vector<double> ref[3];
            for (int m = 0; m < 3; ++m) oracle_rows(m, F, sel, F.rows, KD[KT], VD[VT], 1.0f / 16.0f, 1.0f, ref[m], nthr);
            // (1) bits
            size_t nd = 0, nt = 0; double maxabs = 0;
            for (size_t si = 0; si < sel.size(); ++si) for (int i = 0; i < 24 * 256; ++i) {
                const float a = op1[(size_t) sel[si] * 24 * 256 + i], b = (float) ref[O_CON][si * 24 * 256 + i];
                ++nt; if (memcmp(&a, &b, 4) != 0) { ++nd; maxabs = std::max(maxabs, (double) std::fabs(a - b)); }
            }
            n_bits_total += nt; n_bits_diff += nd;
            printf("W1BITS %s L%d %s values %zu different %zu max_abs_diff %.3e G1_eq_G4 %d\n", tag.c_str(), L, pn.c_str(), nt, nd, maxabs, (int) g_same);
            // (2) contract vs f64, normalised by the f64 oracle
            auto cmp_op = [&](int m) { errstat es; for (size_t si = 0; si < sel.size(); ++si) for (int hq = 0; hq < 24; ++hq) es.add_row(ref[m].data() + (si * 24 + hq) * 256, op1.data() + ((size_t) sel[si] * 24 + hq) * 256); return es; };
            rec(pn + " cpuop_vs_f64_c18", L, cmp_op(O_C18));
            rec(pn + " cpuop_vs_f64_plan", L, cmp_op(O_PLAN));
            // (3) CPU FA vs the CPU op (= contract), normalised by the CPU op
            cpu_fa(be, F, F.rows, KT, KC[KT], VT, VC[VT], fa);
            errstat es; std::vector<double> r(256);
            for (size_t si = 0; si < sel.size(); ++si) for (int hq = 0; hq < 24; ++hq) {
                const float * o = op1.data() + ((size_t) sel[si] * 24 + hq) * 256;
                for (int d = 0; d < 256; ++d) r[d] = o[d];
                es.add_row(r.data(), fa.data() + ((size_t) sel[si] * 24 + hq) * 256);
            }
            rec(pn + " cpufa_vs_cpuop", L, es);
        }
    }
    for (auto & kv : pooled) printf("W1SUM %s %s nmse_layer_max %.4e %s\n", tag.c_str(), kv.first.c_str(), lmax[kv.first], fmt_stat(kv.second).c_str());
    printf("W1BITS_TOTAL %s values %zu different %zu G1_ne_G4_pairs %zu\n", tag.c_str(), n_bits_total, n_bits_diff, n_g_diff);
    ggml_backend_free(be);
    return n_bits_diff == 0 && n_g_diff == 0 ? 0 : 2;
}
