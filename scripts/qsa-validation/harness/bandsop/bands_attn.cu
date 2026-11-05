// R2-M-LAB-BANDS-OP: R1-M-Lab A's bands_attn.cu + UNI/TBO fixtures (bands_ref.h), BANDS_DEFECTS=1 planted defects, TBO bitwise check.
// R1-M-Lab A: C18 prefill G4 harness (c18_attn.cu) with an extended check mode instead of the timing loop.
// hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I../c18 bands_attn.cu -o bands_attn
// usage: bands_attn <rows|SYN:U:nk:seed> <qkv|SYN> <full_layers|all> <stride> <variants: ship|ship+direct|name,...> <tag> [nthr]
//   rows on layers in full_layers: every query; other layers: every stride-th query; all 24 heads.
// Kernels and plan are C18's, unchanged: c18_kernel.inc (c18_attn_kernel<KT,KL,VT,VL>), pr_attn.inc (PR qsa3_attn_kernel
// verbatim), pr_union_kernels.inc (PR rows+merge plan) from ~/dev/wt_lab/c18 (md5 as in MEASUREMENTS C18 Setup).
// Per (variant, layer, oracle): bands::errstat over the selected rows vs O_C18 / O_PLAN / O_CON (bands_ref.h), plus bitwise
// comparison with the first variant of the same (K type, V type) pair over the whole output.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <string>
#include <set>
#include <map>
#include <functional>

#define RDNA3
#define NO_DEVICE_CODE
#define QK8_0 32
typedef struct { uint16_t d; int8_t qs[QK8_0]; } block_q8_0;
static __device__ __forceinline__ uint16_t ggml_cuda_f32_to_act16(const float f, const bool) { return __builtin_bit_cast(uint16_t, (_Float16) f); }
#define QSA3_L2E 1.4426950408889634f
#define QSA3_G 4
#include "pr_union_kernels.inc"
#include "pr_attn.inc"
#include "c18_kernel.inc"
#include "bands_ref.h"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static const char * LN[5] = {"D", "Dd", "X", "S", "P"};
static const char * TN[5] = {"f16", "q8_0", "q4_0", "bf16", "bf16w"};
static int HBk(int T) { return bands::hbytes(T == TBW ? TBF : T); }
struct variant { std::string name, group; int KT, KL, VT, VL; bool pr; };
typedef void (*kfn_t)(const float *, const char *, const char *, const uint32_t *, const uint16_t *, const int *, float *, const qsa3_layout);
template<int KT, int KL, int VT, int VL> static kfn_t kf() { return c18_attn_kernel<KT, KL, VT, VL>; }


// R2 (op): planted defects on the contract (R1-Q8 definitions, bands_ref.h defect_row; d3 = V x 1.001 before the f16 rounding),
// each measured against the clean contract on the same rows: DEFECT <tag> L<l> pair <k>/<v> d<i> cells_dropped <n> <errstat>
static void run_defects(const std::string & tag, const bands::fixture & F, const std::vector<int> & sel, const bands::deq & KD, const bands::deq & VD,
                        const char * kt, const char * vt, const std::vector<double> & con, int nthr) {
    for (int def = 1; def <= 3; ++def) {
        std::vector<int32_t> rows = F.rows; size_t dropped = 0;
        if (def < 3) for (int q : sel) { int32_t * r = rows.data() + (size_t) q * F.ns; std::vector<int> c0, c1; bands::row_cells(r, F.ns, F.nk, c0); bands::defect_row(def, r, F.ns); bands::row_cells(r, F.ns, F.nk, c1); dropped += c0.size() - c1.size(); }
        std::vector<double> d; bands::oracle_rows(bands::O_CON, F, sel, rows, KD, VD, 1.0f / 16.0f, def == 3 ? 1.001f : 1.0f, d, nthr);
        bands::errstat es; for (size_t r = 0; r < sel.size() * 24; ++r) es.add_row(con.data() + r * 256, d.data() + r * 256);
        printf("DEFECT %s L%d pair %s/%s d%d cells_dropped %zu %s\n", tag.c_str(), F.L, kt, vt, def, dropped, bands::fmt_stat(es).c_str());
    }
}
// R2 (op): harness output vs the dumped test-backend-ops ROCm0 output (whole tensor), and test-backend-ops' CPU output vs the contract
static void tbo_compare(const std::string & tag, const bands::fixture & F, const char * name, const float * o, size_t n) {
    if (F.out_gpu.empty()) return;
    size_t neq = 0; double se = 0, sr = 0, mx = 0;
    for (size_t i = 0; i < n; ++i) { neq += memcmp(&o[i], &F.out_gpu[i], 4) != 0; const double e = (double) o[i] - F.out_gpu[i]; se += e * e; sr += (double) F.out_gpu[i] * F.out_gpu[i]; mx = std::max(mx, std::fabs(e)); }
    printf("TBO_BITWISE %s %s words %zu words_differ_vs_tbo_rocm0 %zu nmse %.4e maxabs %.4e\n", tag.c_str(), name, n, neq, sr > 0 ? se / sr : 0.0, mx);
}
int main(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage: rows qkv full_layers stride variants tag [nthr]\n"); return 1; }
    const std::string rowsf = argv[1], qkvf = argv[2], fulls = argv[3], vs = argv[5], tag = argv[6];
    const int stride = atoi(argv[4]), nthr = argc > 7 ? atoi(argv[7]) : 32; const bool defects = getenv("BANDS_DEFECTS") && atoi(getenv("BANDS_DEFECTS"));
    std::set<int> full; bool full_all = fulls == "all";
    if (!full_all) { std::string x; for (char c : fulls + ",") { if (c == ',') { if (!x.empty()) full.insert(atoi(x.c_str())); x.clear(); } else x += c; } }

    std::vector<variant> V; std::map<std::string, kfn_t> KF;
    auto add = [&](const std::string & grp, int KT, int KL, int VT, int VL, kfn_t f) {
        V.push_back({std::string("K") + TN[KT] + LN[KL] + "-V" + TN[VT] + LN[VL], grp, KT, KL, VT, VL, false}); KF[V.back().name] = f;
    };
    // the variants the plan ships (PLAN-v3 W3 / C18 decision 1), K sweep, V sweep
    add("ship", TF16, LD, TF16, LP, kf<TF16, LD, TF16, LP>());
    add("ship", TQ8,  LX, TQ8,  LP, kf<TQ8,  LX, TQ8,  LP>());
    add("ship", TQ4,  LX, TQ4,  LP, kf<TQ4,  LX, TQ4,  LP>());
    add("ship", TBF,  LX, TBF,  LP, kf<TBF,  LX, TBF,  LP>());
    add("ship", TQ8,  LX, TF16, LP, kf<TQ8,  LX, TF16, LP>());
    add("ship", TQ4,  LX, TF16, LP, kf<TQ4,  LX, TF16, LP>());
    add("ship", TBF,  LX, TF16, LP, kf<TBF,  LX, TF16, LP>());
    add("ship", TF16, LD, TQ8,  LP, kf<TF16, LD, TQ8,  LP>());
    add("ship", TF16, LD, TQ4,  LP, kf<TF16, LD, TQ4,  LP>());
    add("ship", TF16, LD, TBF,  LP, kf<TF16, LD, TBF,  LP>());
    // direct V (quantized types at large spread only, C18 decision 1)
    add("direct", TQ8, LX, TQ8, LX, kf<TQ8, LX, TQ8, LX>());
    add("direct", TQ4, LX, TQ4, LX, kf<TQ4, LX, TQ4, LX>());
    // PR verbatim references
    V.push_back({"PR-V0-f16", "pr", TF16, LD, TF16, LP, true}); V.push_back({"PR-V0-q8_0", "pr", TQ8, LD, TQ8, LP, true});
    V.push_back({"PR-VD-f16", "pr", TF16, LD, TF16, LD, true}); V.push_back({"PR-VD-q8_0", "pr", TQ8, LD, TQ8, LD, true});
    std::vector<int> vl;
    {
        std::set<std::string> sel; std::string x; for (char c : vs + ",") { if (c == ',') { sel.insert(x); x.clear(); } else x += c; }
        const bool ship = sel.count("ship") || sel.count("ship+direct"), direct = sel.count("ship+direct");
        // order: grouped by pair so that the oracles of a pair are computed once
        std::vector<std::pair<int,int>> pairs;
        for (int i = 0; i < (int) V.size(); ++i) {
            const bool on = sel.count(V[i].name) || (ship && (V[i].group == "ship" || V[i].group == "pr")) || (direct && V[i].group == "direct");
            if (!on) continue;
            const auto p = std::make_pair(V[i].KT, V[i].VT); if (std::find(pairs.begin(), pairs.end(), p) == pairs.end()) pairs.push_back(p);
        }
        for (auto p : pairs) for (int i = 0; i < (int) V.size(); ++i) {
            const bool on = sel.count(V[i].name) || (ship && (V[i].group == "ship" || V[i].group == "pr")) || (direct && V[i].group == "direct");
            if (on && V[i].KT == p.first && V[i].VT == p.second) vl.push_back(i);
        }
    }
    bands::fixture_reader R; if (!R.open(rowsf, qkvf)) return 1;
    printf("# bands_attn tag %s rows %s data %s full %s stride %d variants %zu nthr %d\n", tag.c_str(), rowsf.c_str(), qkvf.c_str(), fulls.c_str(), stride, vl.size(), nthr);
    std::map<std::string, bands::errstat> pooled; std::map<std::string, double> nmse_lmax; std::map<std::string, int> ord;
    bands::fixture F;
    while (R.next(F, [](int) { return true; })) {
        const int L = F.L, n_q = F.n_q, ns = F.ns, nk = F.nk;
        printf("FIXTURE %s L%d %s\n", tag.c_str(), L, F.label.c_str());
        printf("CASE %s n_q %d ns %d nk %d blocks %d direct_rule %d\n", tag.c_str(), n_q, ns, nk, F.blocks, (int) ((size_t) n_q * 64 < (size_t) nk));
        auto typeok = [&](int i) { return (F.rawKT < 0 || F.rawKT == V[i].KT) && (F.rawVT < 0 || F.rawVT == V[i].VT); };
        std::vector<int> sel; for (int q = 0; q < n_q; ++q) if (full_all || full.count(L) || q % stride == 0) sel.push_back(q);
        // caches
        std::vector<uint8_t> KC[4], VC[4]; bands::deq KD[4], VD[4];
        std::set<int> needK, needV; for (int i : vl) if (typeok(i)) { needK.insert(V[i].KT); needV.insert(V[i].VT); }
        char * dK[4] = {}, * dV[4] = {}; uint16_t * dPV[4] = {};
        for (int T : needK) { bands::build_cache_fx(T, F, true, KC[T]); bands::build_deq(T, KC[T], nk, KD[T]); CK(hipMalloc(&dK[T], KC[T].size())); CK(hipMemcpy(dK[T], KC[T].data(), KC[T].size(), hipMemcpyHostToDevice)); }
        for (int T : needV) { bands::build_cache_fx(T, F, false, VC[T]); bands::build_deq(T, VC[T], nk, VD[T]); CK(hipMalloc(&dV[T], VC[T].size())); CK(hipMemcpy(dV[T], VC[T].data(), VC[T].size(), hipMemcpyHostToDevice)); }
        for (int T : needK) printf("CACHE_HASH %s L%d K %s %016llx\n", tag.c_str(), L, TN[T], (unsigned long long) bands::fnv1a(KC[T].data(), KC[T].size()));
        for (int T : needV) printf("CACHE_HASH %s L%d V %s %016llx\n", tag.c_str(), L, TN[T], (unsigned long long) bands::fnv1a(VC[T].data(), VC[T].size()));
        printf("ROWS_HASH %s L%d %016llx Q_HASH %016llx\n", tag.c_str(), L, (unsigned long long) bands::fnv1a(F.rows.data(), F.rows.size() * 4), (unsigned long long) bands::fnv1a(F.Q.data(), F.Q.size() * 4));
        const size_t nel = (size_t) nk * 2 * 256;
        auto pack = [&](int T) {
            const size_t nb1 = 2 * (size_t) HBk(T), nb2 = HBk(T); const uint32_t count = nel / 4;
            const dim3 gr((count + 255) / 256), bl(256);
            switch (T) {
            case TF16: hipLaunchKernelGGL(c18_pack_v<TF16>, gr, bl, 0, 0, dV[T], dPV[T], nb1, nb2, nk, count); break;
            case TQ8:  hipLaunchKernelGGL(c18_pack_v<TQ8>,  gr, bl, 0, 0, dV[T], dPV[T], nb1, nb2, nk, count); break;
            case TQ4:  hipLaunchKernelGGL(c18_pack_v<TQ4>,  gr, bl, 0, 0, dV[T], dPV[T], nb1, nb2, nk, count); break;
            case TBF:  hipLaunchKernelGGL(c18_pack_v<TBF>,  gr, bl, 0, 0, dV[T], dPV[T], nb1, nb2, nk, count); break;
            }
            CK(hipGetLastError());
        };
        std::set<int> needP; for (int i : vl) if (typeok(i) && V[i].VL == LP) needP.insert(V[i].VT);
        for (int T : needP) { CK(hipMalloc(&dPV[T], nel * 2)); pack(T); }
        CK(hipDeviceSynchronize());
        // plan (PR rows + merge kernels, PR launch parameters; as c18_attn.cu:225-240)
        int32_t * d_ids; float * d_q, * d_out;
        CK(hipMalloc(&d_ids, F.rows.size() * 4)); CK(hipMemcpy(d_ids, F.rows.data(), F.rows.size() * 4, hipMemcpyHostToDevice));
        CK(hipMalloc(&d_q, F.Q.size() * 4)); CK(hipMemcpy(d_q, F.Q.data(), F.Q.size() * 4, hipMemcpyHostToDevice));
        const size_t outn = (size_t) n_q * 24 * 256, outb = outn * 4;
        CK(hipMalloc(&d_out, outb));
        const int ng4 = (n_q + 3) / 4, cap4 = (4 * ns + 3) & ~3;
        uint32_t * d_ublk; uint16_t * d_umask; int * d_ucount, * d_srow, * d_sflag;
        CK(hipMalloc(&d_ublk, (size_t) ng4 * cap4 * 4)); CK(hipMalloc(&d_umask, (size_t) ng4 * cap4 * 2)); CK(hipMalloc(&d_ucount, ng4 * 4));
        CK(hipMalloc(&d_srow, (size_t) n_q * ns * 4)); CK(hipMalloc(&d_sflag, n_q * 4)); CK(hipMemset(d_sflag, 0, n_q * 4));
        {
            int np2 = 1; while (np2 < ns) np2 <<= 1; if ((size_t) np2 * 4 > 60 * 1024) np2 = 0;
            hipLaunchKernelGGL(qsa3_rows_kernel, dim3(n_q), dim3(256), (size_t) std::max(np2 ? np2 : ns, 2 * ns) * 4, 0, (const int *) d_ids, (size_t) ns * 4, ns, nk, d_srow, d_sflag, (const char *) nullptr, (size_t) 0, np2);
            hipLaunchKernelGGL(qsa3_merge_kernel, dim3(ng4), dim3(QSA3_MERGE_LANES), (size_t) QSA3_G * ns * 4, 0, (const int *) d_ids, (size_t) ns * 4, n_q, ns, nk, (const int *) d_srow, (const int *) d_sflag, d_ublk, d_umask, d_ucount, cap4);
            CK(hipGetLastError()); CK(hipDeviceSynchronize());
        }
        auto lay = [&](int KT, int VT) {
            return qsa3_layout{(size_t) 24 * 1024, (size_t) 1024, (size_t) 1024, (size_t) 24 * 1024, 0,
                               2 * (size_t) HBk(KT), 2 * (size_t) HBk(VT), (size_t) HBk(KT), (size_t) HBk(VT), nk, n_q, 12, cap4, 1.0f / 16.0f, 24, 0};
        };
        auto launch = [&](int i, float * dst) {
            const variant & v = V[i];
            const dim3 gr(ng4, 2), bl(256);
            if (v.pr) {
                const bool q8 = v.KT == TQ8; const qsa3_layout s = lay(v.KT, v.VT);
                if (v.VL == LP) {
                    if (q8) hipLaunchKernelGGL((qsa3_attn_kernel<true, 8, false, true>), gr, bl, 0, 0, d_q, (const uint16_t *) dK[TQ8], dPV[TQ8], nullptr, d_ublk, d_umask, d_ucount, dst, s, nullptr, nullptr, false);
                    else    hipLaunchKernelGGL((qsa3_attn_kernel<true, 8, false, false>), gr, bl, 0, 0, d_q, (const uint16_t *) dK[TF16], dPV[TF16], nullptr, d_ublk, d_umask, d_ucount, dst, s, nullptr, nullptr, false);
                } else {
                    if (q8) hipLaunchKernelGGL((qsa3_attn_kernel<true, 8, true, true>), gr, bl, 0, 0, d_q, (const uint16_t *) dK[TQ8], (const uint16_t *) dV[TQ8], nullptr, d_ublk, d_umask, d_ucount, dst, s, nullptr, nullptr, false);
                    else    hipLaunchKernelGGL((qsa3_attn_kernel<true, 8, true, false>), gr, bl, 0, 0, d_q, (const uint16_t *) dK[TF16], (const uint16_t *) dV[TF16], nullptr, d_ublk, d_umask, d_ucount, dst, s, nullptr, nullptr, false);
                }
            } else {
                const char * vptr = v.VL == LP ? (const char *) dPV[v.VT] : dV[v.VT];
                hipLaunchKernelGGL(KF[v.name], gr, bl, 0, 0, d_q, dK[v.KT], vptr, d_ublk, d_umask, d_ucount, dst, lay(v.KT, v.VT));
            }
            CK(hipGetLastError());
        };
        std::vector<float> o(outn), first;
        std::pair<int,int> curpair(-1, -1);
        std::vector<double> ref[3];
        for (int i : vl) {
            if (!typeok(i)) continue;
            const variant & v = V[i];
            const auto key = std::make_pair(v.KT, v.VT);
            if (key != curpair) {
                curpair = key; first.clear();
                for (int m = 0; m < 3; ++m) bands::oracle_rows(m, F, sel, F.rows, KD[v.KT], VD[v.VT], 1.0f / 16.0f, 1.0f, ref[m], nthr);
                if (defects) run_defects(tag, F, sel, KD[v.KT], VD[v.VT], TN[v.KT], TN[v.VT], ref[bands::O_CON], nthr);
                if (!F.out_cpu.empty()) { bands::errstat es; for (size_t si = 0; si < sel.size(); ++si) for (int hq = 0; hq < 24; ++hq) es.add_row(ref[bands::O_CON].data() + (si * 24 + hq) * 256, F.out_cpu.data() + ((size_t) sel[si] * 24 + hq) * 256);
                    printf("TBO_CPU %s L%d pair %s/%s tbo_cpu_vs_contract %s\n", tag.c_str(), L, TN[v.KT], TN[v.VT], bands::fmt_stat(es).c_str()); }
                // reference vs reference on the same rows: contract vs each f64 oracle, and the two f64 oracles
                const int cmpv[3][2] = {{bands::O_CON, bands::O_C18}, {bands::O_CON, bands::O_PLAN}, {bands::O_C18, bands::O_PLAN}};   // {test, reference}
                for (const auto & cp : cmpv) {
                    const int tm = cp[0], rm = cp[1];
                    bands::errstat es; for (size_t r = 0; r < sel.size() * 24; ++r) es.add_row(ref[rm].data() + r * 256, ref[tm].data() + r * 256);
                    printf("REFCMP %s L%d pair %s/%s %s_vs_%s %s\n", tag.c_str(), L, TN[v.KT], TN[v.VT], bands::oname(tm), bands::oname(rm), bands::fmt_stat(es).c_str());
                    const std::string pk = std::string("REF ") + TN[v.KT] + "/" + TN[v.VT] + " " + bands::oname(tm) + "_vs_" + bands::oname(rm);
                    pooled[pk].merge(es); nmse_lmax[pk] = std::max(nmse_lmax[pk], es.nmse());
                }
            }
            CK(hipMemset(d_out, 0xff, outb)); launch(i, d_out); CK(hipDeviceSynchronize());
            CK(hipMemcpy(o.data(), d_out, outb, hipMemcpyDeviceToHost));
            tbo_compare(tag, F, v.name.c_str(), o.data(), o.size());
            if (first.empty()) { first = o; printf("BITWISE %s L%d %-16s pair %s/%s pair_reference\n", tag.c_str(), L, v.name.c_str(), TN[v.KT], TN[v.VT]); }
            else { size_t neq = 0; for (size_t k = 0; k < outn; ++k) neq += memcmp(&o[k], &first[k], 4) != 0;
                   printf("BITWISE %s L%d %-16s pair %s/%s words_differ_vs_pair_ref %zu\n", tag.c_str(), L, v.name.c_str(), TN[v.KT], TN[v.VT], neq); }
            for (int m = 0; m < 3; ++m) {
                bands::errstat es;
                for (size_t si = 0; si < sel.size(); ++si) for (int hq = 0; hq < 24; ++hq)
                    es.add_row(ref[m].data() + (si * 24 + hq) * 256, o.data() + ((size_t) sel[si] * 24 + hq) * 256);
                printf("BAND %s L%d %-16s pair %s/%s gpu_vs_%s %s\n", tag.c_str(), L, v.name.c_str(), TN[v.KT], TN[v.VT], bands::oname(m), bands::fmt_stat(es).c_str());
                const std::string pk = v.name + " " + TN[v.KT] + "/" + TN[v.VT] + " gpu_vs_" + bands::oname(m);
                if (!ord.count(pk)) ord[pk] = (int) ord.size();
                pooled[pk].merge(es); nmse_lmax[pk] = std::max(nmse_lmax[pk], es.nmse());
            }
            fflush(stdout);
        }
        for (int T = 0; T < 4; ++T) { if (dK[T]) hipFree(dK[T]); if (dV[T]) hipFree(dV[T]); if (dPV[T]) hipFree(dPV[T]); }
        hipFree(d_ids); hipFree(d_q); hipFree(d_out); hipFree(d_ublk); hipFree(d_umask); hipFree(d_ucount); hipFree(d_srow); hipFree(d_sflag);
    }
    for (auto & kv : pooled) printf("BANDSUM %s %s nmse_layer_max %.4e %s\n", tag.c_str(), kv.first.c_str(), nmse_lmax[kv.first], bands::fmt_stat(kv.second).c_str());
    return 0;
}
