// W4 harness (PLAN-v3 §13 W4): the new G1 attend (sa_attend_g1 + sa_merge_splits) and the new G4 attend (sa_attend_g4 +
// sa_attend_fixup, packed V with sa_pack_v or direct V), the sources of the qsa-rewrite worktree copied by build.sh (the
// only edit: the common.cuh include replaced by this harness's prelude), next to PR #219's verbatim decode kernel
// (qsa_decode_wmma_partial + qsa_decode_merge, C18 pr_dec.inc) and C18's decode kernel with the shipped loaders
// (c18_dec_kernel<T, K D|X, V S>), all in one process, on R2-BOP fixtures (bands_ref.h: C18 rows files + SYN data, the
// SYN plan, UNI, TBO dumps).
//
// The head of R2-BOP's bands_dec.cu (lines 1-281: prelude, c18_dec_kernel, the includes of c18_kernel.inc and pr_dec.inc)
// is included verbatim (build.sh extracts it), so the PR and C18 kernels are the ones C18 / R1-A / R2-BOP measured.
//
// The plans of the new kernels are sa_ref::plan (tests/sparse-attn-ref.h, the C3 oracle), which W1/W2 showed byte-equal
// to the CPU and HIP SPARSE_ATTN_PLAN ops (W1: 169/169 cases, 1080 synthetic, 288 captured; W2: 185/185 whole-dst).
//
// usage: w4_dec <rows|SYN:U:nk:seed|UNI:..|TBO:file> <qkv|SYN> <n_q> <variants> <tag> <mode> [reps] [qpos] [layers]
//   n_q      queries used per layer (0 = all of the fixture's); qpos first (default) | last: the first / last n_q queries
//   variants comma list of: PR-dec-<T>, PR-decM-<T> (T f16 | q8_0), C18-<T>, C18M-<T> (the C18 shipped loaders),
//            G1-<T>, G1M-<T>, G4P-<T>, G4D-<T>, PACK-<T>, or the groups "atA", "c18gate", "band"
//            (*M = with the split merge; G4P = pack + attend + fixup; G4D = attend + fixup; PACK = the pack alone)
//   mode     band: NMSE of every merged output against O_CON (bands_ref.h), BANDSUM lines as bands_dec, plus the new
//            kernels' bitwise determinism over two launches; BANDS_DEFECTS=1 adds the planted d1/d2/d3 (bands_dec rules)
//            time: C18's timing method: 3 warm-up rounds, then `reps` rounds, every variant once per round with one
//            hipEvent pair around all its launches; per layer the median, SUM = mean of the per-layer medians
//   env BANDS_SYN_LAYERS=N: a SYN rows fixture yields N layers (layer L: rows generator seeded with seed * 1000003 + nk +
//            L * 104729, layer 0 = the single SYN layer of bands_ref.h; data per layer as bands_ref.h)

#include "ggml.h"
#include "bands_dec_head.inc"          // EV/bandsop/src/bands_dec.cu lines 1-281 (prelude, c18_dec_kernel, PR decode)
#include "bands_ref.h"
#include "sparse-attn-ref.h"

#define GGML_USE_HIP 1
#include "sparse-attn-attend-g4.cuh"
#include "sparse-attn-attend-g1.cuh"
#include "sparse-attn-pack.inc"
#include "sparse-attn-merge.inc"

#include <functional>

static const char * TNAME[4] = {"f16", "q8_0", "q4_0", "bf16"};   // C18 ids TF16, TQ8, TQ4, TBF
static int sa_id(int T) { return T == TF16 ? SA_T_F16 : T == TQ8 ? SA_T_Q8_0 : T == TQ4 ? SA_T_Q4_0 : SA_T_BF16; }
static int c18_id(const std::string & t) { return t == "f16" ? TF16 : t == "q8_0" ? TQ8 : t == "q4_0" ? TQ4 : t == "bf16" ? TBF : -1; }

// the PR's qsa_decode_merge, verbatim (qsa-decode.cu:65-79 at 94e086847), as bands_dec.cu
static __global__ __launch_bounds__(256) void pr_merge(const float * partial, float * out, int splits) {
    const int row = blockIdx.x, d = threadIdx.x;
    const float * src = partial + size_t(row)*splits*258;
    float maximum = -INFINITY;
    for (int s = 0; s < splits; ++s) { maximum = fmaxf(maximum, src[s*258 + 256]); }
    float sum = 0.0f, value = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float count = src[s*258 + 257];
        const float factor = count > 0.0f ? expf(src[s*258 + 256] - maximum) : 0.0f;
        value += src[s*258 + d]*factor;
        sum += count*factor;
    }
    out[size_t(row)*256 + d] = sum > 0.0f ? value/sum : 0.0f;
}

template <int T, int VL> static sa_g4_kernel_t g4k() { return sa_attend_g4<T, T, VL, SA_EPI_PLAIN, 12>; }
template <int T> static sa_fixup_kernel_t fixk() { return sa_attend_fixup<T, T, SA_EPI_PLAIN>; }
template <int T> static sa_g1_kernel_t g1k() { return sa_attend_g1<T, T, 12>; }

static sa_g4_kernel_t g4_of(int sid, int vl) {
    switch (sid) {
        case SA_T_F16:  return vl == SA_V_PACKED ? g4k<SA_T_F16,  SA_V_PACKED>() : g4k<SA_T_F16,  SA_V_SPLITGATHER>();
        case SA_T_BF16: return vl == SA_V_PACKED ? g4k<SA_T_BF16, SA_V_PACKED>() : g4k<SA_T_BF16, SA_V_SPLITGATHER>();
        case SA_T_Q8_0: return vl == SA_V_PACKED ? g4k<SA_T_Q8_0, SA_V_PACKED>() : g4k<SA_T_Q8_0, SA_V_SPLITGATHER>();
        default:        return vl == SA_V_PACKED ? g4k<SA_T_Q4_0, SA_V_PACKED>() : g4k<SA_T_Q4_0, SA_V_SPLITGATHER>();
    }
}
static sa_fixup_kernel_t fix_of(int sid) {
    return sid == SA_T_F16 ? fixk<SA_T_F16>() : sid == SA_T_BF16 ? fixk<SA_T_BF16>() : sid == SA_T_Q8_0 ? fixk<SA_T_Q8_0>() : fixk<SA_T_Q4_0>();
}
static sa_g1_kernel_t g1_of(int sid) {
    return sid == SA_T_F16 ? g1k<SA_T_F16>() : sid == SA_T_BF16 ? g1k<SA_T_BF16>() : sid == SA_T_Q8_0 ? g1k<SA_T_Q8_0>() : g1k<SA_T_Q4_0>();
}
typedef void (*dfn_t)(const char *, const char *, const char *, const char *, size_t, size_t, size_t, size_t, size_t, size_t, size_t, int, int, int, int, int, float, float *);

static dfn_t c18_ship(int T) {   // C18 decision 2: K direct for f16 / q8_0, split for q4_0 / bf16; V LDS-staged
    return T == TF16 ? c18_dec_kernel<TF16, LD, LS> : T == TQ8 ? c18_dec_kernel<TQ8, LD, LS> : T == TQ4 ? c18_dec_kernel<TQ4, LX, LS> : c18_dec_kernel<TBF, LX, LS>;
}

enum vkind { K_PR, K_PRM, K_C18, K_C18M, K_G1, K_G1M, K_G4P, K_G4D, K_PACK };
struct wvar { std::string name; int kind; int T; };

int main(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage: rows qkv n_q variants tag mode [reps] [qpos] [layers]\n"); return 1; }
    const std::string rowsf = argv[1], qkvf = argv[2], vs = argv[4], tag = argv[5], mode = argv[6];
    const int NQarg = atoi(argv[3]), reps = argc > 7 ? atoi(argv[7]) : 10;
    const bool qlast = argc > 8 && std::string(argv[8]) == "last";
    std::set<int> layers;
    if (argc > 9 && std::string(argv[9]) != "all") { std::string x; for (char c : std::string(argv[9]) + ",") { if (c == ',') { layers.insert(atoi(x.c_str())); x.clear(); } else x += c; } }
    const bool band = mode == "band", defects = getenv("BANDS_DEFECTS") && atoi(getenv("BANDS_DEFECTS"));
    const int syn_layers = getenv("BANDS_SYN_LAYERS") ? atoi(getenv("BANDS_SYN_LAYERS")) : 1;

    std::vector<wvar> V;
    {
        std::vector<std::string> toks; std::string x; for (char c : vs + ",") { if (c == ',') { if (!x.empty()) toks.push_back(x); x.clear(); } else x += c; }
        auto add = [&](const std::string & n) {
            const size_t p = n.find('-');
            std::string k = n.substr(0, p), t = n.substr(p + 1);
            if (k == "PR") { k = n.substr(0, n.rfind('-')); t = n.substr(n.rfind('-') + 1); }   // PR-dec-<T>, PR-decM-<T>
            const int T = c18_id(t);
            int kind = -1;
            if (k == "PR-dec") kind = K_PR; else if (k == "PR-decM") kind = K_PRM; else if (k == "C18") kind = K_C18; else if (k == "C18M") kind = K_C18M;
            else if (k == "G1") kind = K_G1; else if (k == "G1M") kind = K_G1M; else if (k == "G4P") kind = K_G4P; else if (k == "G4D") kind = K_G4D; else if (k == "PACK") kind = K_PACK;
            if (kind < 0 || T < 0 || ((kind == K_PR || kind == K_PRM) && T != TF16 && T != TQ8)) { fprintf(stderr, "bad variant %s\n", n.c_str()); exit(1); }
            V.push_back({n, kind, T});
        };
        for (const std::string & tk : toks) {
            if (tk == "atA") { for (const char * t : {"f16", "bf16", "q8_0", "q4_0"}) { add(std::string("G1M-") + t); add(std::string("G4P-") + t); add(std::string("G4D-") + t); add(std::string("PACK-") + t); } }
            else if (tk == "c18gate") { for (const char * t : {"f16", "bf16", "q8_0", "q4_0"}) { add(std::string("C18-") + t); add(std::string("G1-") + t); add(std::string("C18M-") + t); add(std::string("G1M-") + t); } add("PR-dec-f16"); add("PR-dec-q8_0"); }
            else if (tk == "band") { add("PR-decM-f16"); add("PR-decM-q8_0"); for (const char * t : {"f16", "bf16", "q8_0", "q4_0"}) add(std::string("G1M-") + t); }
            else add(tk);
        }
    }
    std::set<int> types; for (auto & v : V) types.insert(v.T);

    bands::fixture_reader R; if (!R.open(rowsf, qkvf)) return 1;
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    printf("# w4_dec tag %s rows %s data %s n_q %d qpos %s variants %zu mode %s reps %d syn_layers %d defects %d\n", tag.c_str(), rowsf.c_str(),
           qkvf.c_str(), NQarg, qlast ? "last" : "first", V.size(), mode.c_str(), reps, syn_layers, (int) defects);
    std::map<std::string, double> sum_med; std::map<std::string, bands::errstat> pooled; std::map<std::string, double> nmse_lmax;
    int n_layers = 0;
    bands::fixture F;
    const bool synrows = rowsf.rfind("SYN:", 0) == 0;
    int syn_L = 0;
    auto want = [&](int L) { return layers.empty() || layers.count(L); };
    for (;;) {
        bool got;
        if (synrows && syn_L > 0) {
            if (syn_L >= syn_layers) break;
            // the next SYN layer: the bands_ref.h SYN rows generator seeded with seed * 1000003 + nk + L * 104729
            std::vector<int32_t> rows;
            {
                const int n_q = 4096, ns = 2051, nqd = 513, ng4 = n_q / 4, nk = R.synNk, nQ = nk / 4, uq = (int) std::lround(R.synU * nqd);
                std::mt19937_64 rng(R.synSeed * 1000003ULL + (uint64_t) nk + (uint64_t) syn_L * 104729ULL);
                std::vector<int> perm(nQ); for (int i = 0; i < nQ; ++i) perm[i] = i;
                rows.assign((size_t) n_q * ns, -1);
                std::vector<int> S(uq), sub(uq), pick;
                for (int g = 0; g < ng4; ++g) {
                    for (int i = 0; i < uq; ++i) { const int j = i + (int) (rng() % (uint64_t) (nQ - i)); std::swap(perm[i], perm[j]); S[i] = perm[i]; }
                    std::sort(S.begin(), S.end());
                    for (int qi = 0; qi < 4; ++qi) {
                        pick.clear();
                        if (qi == 0 || uq == nqd) for (int i = 0; i < nqd; ++i) pick.push_back(S[i]);
                        else if (qi == 1) for (int i = uq - nqd; i < uq; ++i) pick.push_back(S[i]);
                        else { for (int i = 0; i < uq; ++i) sub[i] = i; for (int i = 0; i < nqd; ++i) { const int j = i + (int) (rng() % (uint64_t) (uq - i)); std::swap(sub[i], sub[j]); }
                               std::sort(sub.begin(), sub.begin() + nqd); for (int i = 0; i < nqd; ++i) pick.push_back(S[sub[i]]); }
                        int32_t * rr = rows.data() + (size_t) (4 * g + qi) * ns; int o = 0;
                        for (int i = 0; i < nqd; ++i) { const int nc = i + 1 < nqd ? 4 : ns - 4 * (nqd - 1); for (int c = 0; c < nc; ++c) rr[o++] = 4 * pick[i] + c; }
                    }
                }
            }
            F.rows = rows; F.L = syn_L;
            // data of layer syn_L (bands_ref.h SYN data rule: 0xC18 + L * 7919 + nk)
            std::mt19937_64 rg(0xC18ULL + (uint64_t) syn_L * 7919ULL + (uint64_t) F.nk);
            auto u = [&]() { return (float) ((rg() >> 40) * (2.0 / 16777216.0) - 1.0) * 1.7f; };
            for (auto & x : F.Q) x = u();
            for (auto & x : F.K16) x = bands::f2h(u());
            for (auto & x : F.V16) x = bands::f2h(u());
            ++syn_L;
            got = true;
        } else {
            got = R.next(F, want);
            if (synrows) syn_L = 1;
        }
        if (!got) break;
        if (!want(F.L)) continue;
        const int L = F.L, nk = F.nk, ns = F.ns, NQ = NQarg > 0 ? std::min(NQarg, F.n_q) : F.n_q;
        const int q0 = qlast ? F.n_q - NQ : 0;
        // the selected queries' rows and Q
        std::vector<int32_t> rows((size_t) NQ * ns); std::copy(F.rows.begin() + (size_t) q0 * ns, F.rows.begin() + (size_t) (q0 + NQ) * ns, rows.begin());
        std::vector<float> Qs((size_t) NQ * 24 * 256); std::copy(F.Q.begin() + (size_t) q0 * 24 * 256, F.Q.begin() + (size_t) (q0 + NQ) * 24 * 256, Qs.begin());
        printf("FIXTURE %s L%d %s n_q_used %d q0 %d\n", tag.c_str(), L, F.label.c_str(), NQ, q0);
        printf("ROWS_HASH %s L%d %016llx\n", tag.c_str(), L, (unsigned long long) bands::fnv1a(rows.data(), rows.size() * 4));
        bands::fixture Fs = F; Fs.rows = rows; Fs.Q = Qs; Fs.n_q = NQ;   // the oracles read F.rows / F.Q by query index
        std::vector<int> sel; for (int q = 0; q < NQ; ++q) sel.push_back(q);

        std::vector<uint8_t> KC[4], VC[4]; bands::deq KD[4], VD[4]; char * dK[4] = {}, * dV[4] = {};
        std::set<int> need;
        for (int T : types) if ((F.rawKT < 0 || F.rawKT == T) && (F.rawVT < 0 || F.rawVT == T)) need.insert(T);
        for (int T : need) {
            bands::build_cache_fx(T, F, true, KC[T]); bands::build_cache_fx(T, F, false, VC[T]);
            if (band) { bands::build_deq(T, KC[T], nk, KD[T]); bands::build_deq(T, VC[T], nk, VD[T]); }
            CK(hipMalloc(&dK[T], KC[T].size())); CK(hipMemcpy(dK[T], KC[T].data(), KC[T].size(), hipMemcpyHostToDevice));
            CK(hipMalloc(&dV[T], VC[T].size())); CK(hipMemcpy(dV[T], VC[T].data(), VC[T].size(), hipMemcpyHostToDevice));
        }
        // PR launch (qsa-decode.cu:136-158): unique ids, step 64, chunk / splits rule
        int32_t * d_ids, * d_uids; float * d_q, * d_part, * d_out;
        CK(hipMalloc(&d_ids, rows.size() * 4)); CK(hipMemcpy(d_ids, rows.data(), rows.size() * 4, hipMemcpyHostToDevice));
        CK(hipMalloc(&d_uids, rows.size() * 4));
        hipLaunchKernelGGL(qsa_decode_unique_ids, dim3(NQ), dim3(256), 0, 0, (const char *) d_ids, (size_t) ns * 4, d_uids, ns, nk);
        CK(hipGetLastError()); CK(hipDeviceSynchronize());
        const int step = 64, nh = 24;
        int chunk = step;
        if (NQ > 8) { const int target = std::max(1, (8 * ((ns + step - 1) / step) + NQ - 1) / NQ); chunk = ((ns + target - 1) / target + 15) / 16 * 16; }
        const int pr_splits = (ns + chunk - 1) / chunk;
        const int g1_splits = (int) sa_g1_splits(NQ, ns, nk);
        CK(hipMalloc(&d_q, Qs.size() * 4)); CK(hipMemcpy(d_q, Qs.data(), Qs.size() * 4, hipMemcpyHostToDevice));
        const size_t outn = (size_t) NQ * nh * 256;
        const size_t pn = (size_t) NQ * nh * std::max(pr_splits, g1_splits) * 258;
        CK(hipMalloc(&d_part, pn * 4)); CK(hipMalloc(&d_out, outn * 4));
        // plans (sa_ref::plan = the C3 oracle = the op's bytes)
        uint8_t * d_plan[5] = {};
        ggml_sparse_attn_plan_layout PL[5];
        double ent[5] = {0, 0, 0, 0, 0};
        for (int G : {1, 4}) {
            const std::vector<uint8_t> p = sa_ref::plan(rows, ns, NQ, 1, nk, G);
            PL[G] = ggml_sparse_attn_plan_get_layout(ns, NQ, 1, nk, G);
            CK(hipMalloc(&d_plan[G], p.size())); CK(hipMemcpy(d_plan[G], p.data(), p.size(), hipMemcpyHostToDevice));
            const uint32_t * cnt = (const uint32_t *) (p.data() + PL[G].off_counts);
            for (int64_t g = 0; g < PL[G].n_groups; ++g) ent[G] += cnt[g];
        }
        // G4 packed V workspaces
        const uint32_t nquads = (uint32_t) ((nk + 3) / 4);
        uint16_t * dPV[4] = {}; uint8_t * dQF[4] = {}; uint32_t * dANY[4] = {}; uint8_t * d_flags;
        for (int T : need) { CK(hipMalloc(&dPV[T], (size_t) nquads * 2 * 1024 * 2)); CK(hipMalloc(&dQF[T], (size_t) nquads * 2)); CK(hipMalloc(&dANY[T], 4)); }
        CK(hipMalloc(&d_flags, (size_t) PL[4].n_groups * 2 * 8));
        const size_t q1 = 24 * 1024, q2 = 1024;
        auto pack_new = [&](int T) {
            const uint64_t count = (uint64_t) nquads * 256 * 2;
            const unsigned blocks = (unsigned) ((count + SA_PACK_THREADS - 1) / SA_PACK_THREADS);
            const size_t nb1 = 2 * (size_t) bands::hbytes(T), nb2 = bands::hbytes(T), nb3 = 0;
            CK(hipMemsetAsync(dANY[T], 0, 4, 0));
            switch (sa_id(T)) {
                case SA_T_F16:  sa_pack_v<SA_T_F16> <<<blocks, SA_PACK_THREADS>>>(dV[T], dPV[T], nb1, nb2, nb3, 2, nquads, count, dQF[T], dANY[T]); break;
                case SA_T_BF16: sa_pack_v<SA_T_BF16><<<blocks, SA_PACK_THREADS>>>(dV[T], dPV[T], nb1, nb2, nb3, 2, nquads, count, dQF[T], dANY[T]); break;
                case SA_T_Q8_0: sa_pack_v<SA_T_Q8_0><<<blocks, SA_PACK_THREADS>>>(dV[T], dPV[T], nb1, nb2, nb3, 2, nquads, count, dQF[T], dANY[T]); break;
                default:        sa_pack_v<SA_T_Q4_0><<<blocks, SA_PACK_THREADS>>>(dV[T], dPV[T], nb1, nb2, nb3, 2, nquads, count, dQF[T], dANY[T]); break;
            }
        };
        auto g4args = [&](int T, int vl) {
            const ggml_sparse_attn_plan_layout & P = PL[4];
            sa_g4_args a = {};
            a.q = (const char *) d_q; a.q_nb1 = q1; a.q_nb2 = q2; a.q_nb3 = 0;
            a.k = dK[T]; a.k_nb1 = 2 * (size_t) bands::hbytes(T); a.k_nb2 = bands::hbytes(T); a.k_nb3 = 0;
            a.v = dV[T]; a.v_nb1 = 2 * (size_t) bands::hbytes(T); a.v_nb2 = bands::hbytes(T); a.v_nb3 = 0;
            a.pv = dPV[T]; a.pv_nquads = nquads;
            a.counts = (const uint32_t *) (d_plan[4] + P.off_counts); a.quads = (const uint32_t *) (d_plan[4] + P.off_quads);
            a.masks = (const uint16_t *) (d_plan[4] + P.off_masks); a.n_groups = (int) P.n_groups; a.cap = (int) P.cap;
            a.n_q = NQ; a.n_head = 24; a.n_head_kv = 2; a.scale = 1.0f / 16.0f; a.out = d_out; a.flags = d_flags;
            if (vl == SA_V_PACKED) { a.qflags = dQF[T]; a.qany = dANY[T]; }
            return a;
        };
        auto g1args = [&](int T) {
            const ggml_sparse_attn_plan_layout & P = PL[1];
            sa_g1_args a = {};
            a.q = (const char *) d_q; a.q_nb1 = q1; a.q_nb2 = q2; a.q_nb3 = 0;
            a.k = dK[T]; a.k_nb1 = 2 * (size_t) bands::hbytes(T); a.k_nb2 = bands::hbytes(T); a.k_nb3 = 0;
            a.v = dV[T]; a.v_nb1 = 2 * (size_t) bands::hbytes(T); a.v_nb2 = bands::hbytes(T); a.v_nb3 = 0;
            a.counts = (const uint32_t *) (d_plan[1] + P.off_counts); a.quads = (const uint32_t *) (d_plan[1] + P.off_quads);
            a.masks = (const uint16_t *) (d_plan[1] + P.off_masks); a.n_groups = (int) P.n_groups; a.cap = (int) P.cap;
            a.n_q = NQ; a.n_head = 24; a.n_head_kv = 2; a.n_stream = 1; a.splits = g1_splits; a.scale = 1.0f / 16.0f;
            a.out = d_out; a.partial = d_part;
            return a;
        };
        auto launch = [&](const wvar & v) {
            const int T = v.T; const size_t k1 = 2 * (size_t) bands::hbytes(T), k2 = bands::hbytes(T);
            switch (v.kind) {
                case K_PR: case K_PRM: {
                    const dim3 gr(2, NQ, pr_splits);
                    if (T == TQ8) hipLaunchKernelGGL(qsa_decode_wmma_partial<true>,  gr, dim3(256), 0, 0, (const char *) d_q, dK[T], dV[T], (const char *) nullptr, (const char *) d_uids, q1, q2, k1, k2, k1, k2, (size_t) 0, (size_t) ns * 4, nk, ns, nh, pr_splits, chunk, 1.0f / 16.0f, d_part);
                    else          hipLaunchKernelGGL(qsa_decode_wmma_partial<false>, gr, dim3(256), 0, 0, (const char *) d_q, dK[T], dV[T], (const char *) nullptr, (const char *) d_uids, q1, q2, k1, k2, k1, k2, (size_t) 0, (size_t) ns * 4, nk, ns, nh, pr_splits, chunk, 1.0f / 16.0f, d_part);
                    if (v.kind == K_PRM) hipLaunchKernelGGL(pr_merge, dim3(NQ * nh), dim3(256), 0, 0, (const float *) d_part, d_out, pr_splits);
                } break;
                case K_C18: case K_C18M:
                    hipLaunchKernelGGL(c18_ship(T), dim3(2, NQ, pr_splits), dim3(256), 0, 0, (const char *) d_q, dK[T], dV[T], (const char *) d_uids, q1, q2, k1, k2, k1, k2, (size_t) ns * 4, nk, ns, nh, pr_splits, chunk, 1.0f / 16.0f, d_part);
                    if (v.kind == K_C18M) hipLaunchKernelGGL(pr_merge, dim3(NQ * nh), dim3(256), 0, 0, (const float *) d_part, d_out, pr_splits);
                    break;
                case K_G1: case K_G1M: {
                    const sa_g1_args a = g1args(T);
                    g1_of(sa_id(T))<<<dim3((unsigned) (NQ * 2), (unsigned) g1_splits), 256>>>(a);
                    if (v.kind == K_G1M && g1_splits > 1) sa_merge_splits<<<(unsigned) (NQ * nh), 256>>>(d_part, d_out, g1_splits);
                } break;
                case K_G4P: { pack_new(T); const sa_g4_args a = g4args(T, SA_V_PACKED); const dim3 gr((unsigned) PL[4].n_groups, 2, 1);
                              g4_of(sa_id(T), SA_V_PACKED)<<<gr, 256>>>(a); fix_of(sa_id(T))<<<gr, 256>>>(a, 12); } break;
                case K_G4D: { const sa_g4_args a = g4args(T, SA_V_SPLITGATHER); const dim3 gr((unsigned) PL[4].n_groups, 2, 1);
                              g4_of(sa_id(T), SA_V_SPLITGATHER)<<<gr, 256>>>(a); fix_of(sa_id(T))<<<gr, 256>>>(a, 12); } break;
                case K_PACK: pack_new(T); break;
            }
            CK(hipGetLastError());
        };
        printf("CASE %s L%d n_q %d ns %d nk %d pr_splits %d pr_chunk %d g1_splits %d g1_entries_per_q %.2f g4_entries_per_group %.2f\n", tag.c_str(), L, NQ, ns, nk,
               pr_splits, chunk, g1_splits, ent[1] / PL[1].n_groups, ent[4] / PL[4].n_groups);
        if (band) {
            int curT = -1; std::vector<double> ref[3]; std::vector<float> o(outn), o2(outn);
            for (const wvar & v : V) {
                if (!need.count(v.T) || v.kind == K_PACK) continue;
                if (v.kind == K_PR || v.kind == K_C18 || (v.kind == K_G1 && g1_splits > 1)) continue;   // band mode compares merged outputs
                if (v.T != curT) {
                    curT = v.T;
                    for (int m = 0; m < 3; ++m) bands::oracle_rows(m, Fs, sel, rows, KD[v.T], VD[v.T], 1.0f / 16.0f, 1.0f, ref[m], 32);
                    if (defects) {
                        for (int def = 1; def <= 3; ++def) {   // bands_dec run_defects
                            std::vector<int32_t> rd = rows; size_t dropped = 0;
                            if (def < 3) for (int q : sel) { int32_t * r = rd.data() + (size_t) q * ns; std::vector<int> c0, c1; bands::row_cells(r, ns, nk, c0); bands::defect_row(def, r, ns); bands::row_cells(r, ns, nk, c1); dropped += c0.size() - c1.size(); }
                            std::vector<double> d; bands::oracle_rows(bands::O_CON, Fs, sel, rd, KD[v.T], VD[v.T], 1.0f / 16.0f, def == 3 ? 1.001f : 1.0f, d, 32);
                            bands::errstat es; for (size_t r = 0; r < sel.size() * 24; ++r) es.add_row(ref[bands::O_CON].data() + r * 256, d.data() + r * 256);
                            printf("DEFECT %s L%d pair %s/%s d%d cells_dropped %zu %s\n", tag.c_str(), L, TNAME[v.T], TNAME[v.T], def, dropped, bands::fmt_stat(es).c_str());
                        }
                    }
                }
                CK(hipMemset(d_out, 0xff, outn * 4)); launch(v); CK(hipDeviceSynchronize()); CK(hipMemcpy(o.data(), d_out, outn * 4, hipMemcpyDeviceToHost));
                CK(hipMemset(d_out, 0xff, outn * 4)); launch(v); CK(hipDeviceSynchronize()); CK(hipMemcpy(o2.data(), d_out, outn * 4, hipMemcpyDeviceToHost));
                size_t ndet = 0; for (size_t k = 0; k < outn; ++k) ndet += memcmp(&o[k], &o2[k], 4) != 0;
                printf("DETERMINISM %s L%d %-14s words_differ_run1_run2 %zu\n", tag.c_str(), L, v.name.c_str(), ndet);
                if (!F.out_gpu.empty() && (v.kind == K_G1M)) {   // TBO: the test-backend-ops ROCm0 output (same kernel)
                    size_t neq = 0; for (size_t k = 0; k < outn; ++k) neq += memcmp(&o[k], &F.out_gpu[k], 4) != 0;
                    printf("TBO_BITWISE %s %s words %zu words_differ_vs_tbo_rocm0 %zu\n", tag.c_str(), v.name.c_str(), outn, neq);
                }
                for (int m = 0; m < 3; ++m) {
                    bands::errstat es;
                    for (int r = 0; r < NQ * nh; ++r) es.add_row(ref[m].data() + (size_t) r * 256, o.data() + (size_t) r * 256);
                    printf("BAND %s L%d %-14s pair %s/%s gpu_vs_%s %s\n", tag.c_str(), L, v.name.c_str(), TNAME[v.T], TNAME[v.T], bands::oname(m), bands::fmt_stat(es).c_str());
                    const std::string pk = v.name + " " + TNAME[v.T] + "/" + TNAME[v.T] + " gpu_vs_" + bands::oname(m);
                    pooled[pk].merge(es); nmse_lmax[pk] = std::max(nmse_lmax[pk], es.nmse());
                }
                fflush(stdout);
            }
        } else {
            std::vector<std::vector<float>> t(V.size());
            for (int it = 0; it < 3 + reps; ++it) {
                for (size_t i = 0; i < V.size(); ++i) {
                    if (!need.count(V[i].T)) continue;
                    CK(hipEventRecord(e0, 0)); launch(V[i]); CK(hipEventRecord(e1, 0)); CK(hipEventSynchronize(e1));
                    float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it >= 3) t[i].push_back(ms);
                }
            }
            for (size_t i = 0; i < V.size(); ++i) {
                if (t[i].empty()) continue;
                std::vector<float> x = t[i]; std::sort(x.begin(), x.end());
                const double med = x.size() % 2 ? x[x.size() / 2] : 0.5 * (x[x.size() / 2 - 1] + x[x.size() / 2]);
                printf("L%d RES %-14s n_q %d nk %d med %.4f min %.4f max %.4f\n", L, V[i].name.c_str(), NQ, nk, med, x.front(), x.back());
                sum_med[V[i].name] += med;
            }
            fflush(stdout);
        }
        ++n_layers;
        for (int T : need) { hipFree(dK[T]); hipFree(dV[T]); hipFree(dPV[T]); hipFree(dQF[T]); hipFree(dANY[T]); }
        hipFree(d_ids); hipFree(d_uids); hipFree(d_q); hipFree(d_part); hipFree(d_out); hipFree(d_flags); hipFree(d_plan[1]); hipFree(d_plan[4]);
    }
    if (band) {
        for (auto & kv : pooled) printf("BANDSUM %s %s nmse_layer_max %.4e %s\n", tag.c_str(), kv.first.c_str(), nmse_lmax[kv.first], bands::fmt_stat(kv.second).c_str());
    } else {
        printf("SUMMARY layers %d (mean of per-layer medians, ms)\n", n_layers);
        for (auto & kv : sum_med) printf("SUM %-14s %.4f\n", kv.first.c_str(), kv.second / n_layers);
    }
    return 0;
}
