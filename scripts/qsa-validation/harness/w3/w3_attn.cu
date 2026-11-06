// W3 kernel-time harness (PLAN-v3 §13 W3 done-when "kernel timing on C18 rows"): the new G4 attend (sa_attend_g4 +
// sa_pack_v + sa_attend_fixup, the sources of the qsa-rewrite worktree copied by build.sh, the only edit being the
// common.cuh include replaced by this harness's prelude) against the PR-verbatim qsa3_attn_kernel (C18 pr_attn.inc) and
// C18's own reimplementation c18_attn_kernel<f16 D, f16 P>, in one process, interleaved, C18's timing method:
// 3 warm-up rounds, then `reps` rounds; every variant launched once per round with one hipEvent pair; per layer the
// median; the summary is the mean of the per-layer medians.
// Fixtures: C18 rows files (~/dev/wt_lab/c18/cap/*.rows) with SYN data (uniform +-1.7, C18's generator, deterministic per
// layer), or the C6/C18 synthetic plan SYN:U:nk:seed. The plan is the PR's own rows + merge kernels (C18), converted on
// the host to the ggml.h layout the new kernel reads (counts, quads with the last real quad as padding, masks).
// usage: w3_attn <rows|SYN:U:nk:seed> <reps> <n_q (0 = all)> <types: f16,bf16,q8_0,q4_0> <vl: P|D|PD> [layers: all|list]
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
#define GGML_UNUSED(x) (void)(x)
#include "sparse-attn-attend-g4.cuh"
#include "sparse-attn-pack.inc"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

struct hdr { char name[64]; int32_t n_q, ns, nk, has_mask, pos0, kt; };
static float h2f(uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }
static uint16_t f2h(float f) { return __builtin_bit_cast(uint16_t, (_Float16) f); }
static uint16_t f2bf(float f) { uint32_t u = __builtin_bit_cast(uint32_t, f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t) (u >> 16); }

// C18 type ids (c18_kernel.inc) and bytes per 256-dim row
static const int HB[4] = {512, 272, 144, 512};
static const char * TN[4] = {"f16", "q8_0", "q4_0", "bf16"};
static int sa_id(int T) { return T == TF16 ? SA_T_F16 : T == TQ8 ? SA_T_Q8_0 : T == TQ4 ? SA_T_Q4_0 : SA_T_BF16; }

// C18 build_cache (c18_attn.cu:50-79), verbatim apart from the names
static void build_cache(int T, const std::vector<uint16_t> & f, int nk, std::vector<uint8_t> & out) {
    const size_t rowb = 2 * (size_t) HB[T];
    out.assign((size_t) nk * rowb, 0);
    for (size_t kh = 0; kh < (size_t) nk * 2; ++kh) {
        const uint16_t * x = f.data() + kh * 256;
        uint8_t * o = out.data() + kh * HB[T];
        if (T == TF16) { memcpy(o, x, 512); }
        else if (T == TBF) { uint16_t * ob = (uint16_t *) o; for (int d = 0; d < 256; ++d) ob[d] = f2bf(h2f(x[d])); }
        else if (T == TQ8) {
            block_q8_0 * bq = (block_q8_0 *) o;
            for (int b = 0; b < 8; ++b) {
                float am = 0; for (int i = 0; i < 32; ++i) am = std::max(am, std::fabs(h2f(x[32*b+i])));
                const float d = am / 127.f, id = d ? 1.f / d : 0.f;
                bq[b].d = f2h(d); for (int i = 0; i < 32; ++i) bq[b].qs[i] = (int8_t) roundf(h2f(x[32*b+i]) * id);
            }
        } else {
            block_q4_0 * bq = (block_q4_0 *) o;
            for (int b = 0; b < 8; ++b) {
                float am = 0, mx = 0; for (int i = 0; i < 32; ++i) { const float v = h2f(x[32*b+i]); if (am < std::fabs(v)) { am = std::fabs(v); mx = v; } }
                const float d = mx / -8.f, id = d ? 1.f / d : 0.f;
                bq[b].d = f2h(d);
                for (int j = 0; j < 16; ++j) {
                    const uint8_t a = (uint8_t) std::min(15, (int) (int8_t) (h2f(x[32*b+j]) * id + 8.5f));
                    const uint8_t c = (uint8_t) std::min(15, (int) (int8_t) (h2f(x[32*b+16+j]) * id + 8.5f));
                    bq[b].qs[j] = a | (c << 4);
                }
            }
        }
    }
}

typedef void (*g4k_t)(const sa_g4_args);
template <int T, int VL> static g4k_t g4k() { return sa_attend_g4<T, T, VL, SA_EPI_PLAIN, 12>; }
template <int T> static sa_fixup_kernel_t fixk() { return sa_attend_fixup<T, T, SA_EPI_PLAIN>; }

struct variant { std::string name; int kind; int T; int vl; };   // kind 0 PR-V0-f16, 1 C18 Kf16D-Vf16P, 2 new attend+fixup, 3 PR pack f16, 4 new pack

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: rows|SYN:U:nk:seed reps n_q types vl [layers]\n"); return 1; }
    const std::string rowsf = argv[1];
    const int reps = atoi(argv[2]), nq_use_arg = atoi(argv[3]);
    std::set<std::string> tsel; { std::string x; for (char c : std::string(argv[4]) + ",") { if (c == ',') { tsel.insert(x); x.clear(); } else x += c; } }
    const std::string vls = argv[5];
    std::set<int> layers;
    if (argc > 6 && std::string(argv[6]) != "all") { std::string x; for (char c : std::string(argv[6]) + ",") { if (c == ',') { layers.insert(atoi(x.c_str())); x.clear(); } else x += c; } }

    std::vector<variant> V;
    V.push_back({"PR-V0-f16", 0, TF16, 0});
    V.push_back({"PACKV-f16", 3, TF16, 0});
    V.push_back({"C18-Kf16D-Vf16P", 1, TF16, 0});
    std::map<std::string, g4k_t> KN; std::map<int, sa_fixup_kernel_t> FX;
    for (int T : {TF16, TBF, TQ8, TQ4}) {
        if (!tsel.count(TN[T])) continue;
        const int sid = sa_id(T);
        FX[T] = sid == SA_T_F16 ? fixk<SA_T_F16>() : sid == SA_T_BF16 ? fixk<SA_T_BF16>() : sid == SA_T_Q8_0 ? fixk<SA_T_Q8_0>() : fixk<SA_T_Q4_0>();
        if (vls.find('P') != std::string::npos) {
            const std::string n = std::string("NEW-") + TN[T] + "-P";
            V.push_back({n, 2, T, SA_V_PACKED});
            KN[n] = sid == SA_T_F16 ? g4k<SA_T_F16, SA_V_PACKED>() : sid == SA_T_BF16 ? g4k<SA_T_BF16, SA_V_PACKED>() : sid == SA_T_Q8_0 ? g4k<SA_T_Q8_0, SA_V_PACKED>() : g4k<SA_T_Q4_0, SA_V_PACKED>();
            V.push_back({std::string("NEWPACK-") + TN[T], 4, T, 0});
        }
        if (vls.find('D') != std::string::npos) {
            const std::string n = std::string("NEW-") + TN[T] + "-D";
            V.push_back({n, 2, T, SA_V_SPLITGATHER});
            KN[n] = sid == SA_T_F16 ? g4k<SA_T_F16, SA_V_SPLITGATHER>() : sid == SA_T_BF16 ? g4k<SA_T_BF16, SA_V_SPLITGATHER>() : sid == SA_T_Q8_0 ? g4k<SA_T_Q8_0, SA_V_SPLITGATHER>() : g4k<SA_T_Q4_0, SA_V_SPLITGATHER>();
        }
    }

    FILE * fr = nullptr; const bool synrows = rowsf.rfind("SYN:", 0) == 0;
    double synU = 0; int synNk = 0; unsigned long long synSeed = 1;
    if (synrows) { if (sscanf(rowsf.c_str(), "SYN:%lf:%d:%llu", &synU, &synNk, &synSeed) != 3) { fprintf(stderr, "bad SYN spec\n"); return 1; } }
    else { fr = fopen(rowsf.c_str(), "rb"); if (!fr) { perror("rows"); return 1; } }
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    printf("# w3_attn rows %s reps %d n_q %d variants %zu\n", rowsf.c_str(), reps, nq_use_arg, V.size());
    std::map<std::string, double> sum_med; int n_layers = 0; int L = -1;
    for (;;) {
        hdr h; std::vector<int32_t> rows;
        if (synrows) {
            if (L >= 0) break;
            memset(&h, 0, sizeof(h)); h.n_q = 4096; h.ns = 2051; h.nk = synNk;
            // C18 c18_attn.cu:155-173 (C6 uni mode), verbatim
            const int n_q = 4096, ns = 2051, nqd = 513, ng4 = n_q / 4, nQ = synNk / 4, uq = (int) std::lround(synU * nqd);
            if (uq > nQ) { fprintf(stderr, "union too large\n"); return 1; }
            std::mt19937_64 rng(synSeed * 1000003ULL + (uint64_t) synNk);
            std::vector<int> perm(nQ); for (int i = 0; i < nQ; ++i) perm[i] = i;
            rows.resize((size_t) n_q * ns);
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
        } else {
            if (fread(&h, sizeof(h), 1, fr) != 1) break;
            rows.resize((size_t) h.n_q * h.ns);
            if (fread(rows.data(), 4, rows.size(), fr) != rows.size()) { fprintf(stderr, "short rows\n"); return 1; }
        }
        ++L;
        if (!layers.empty() && !layers.count(L)) continue;
        const int ns = h.ns, nk = h.nk;
        const int n_q = nq_use_arg > 0 && nq_use_arg < h.n_q ? nq_use_arg : h.n_q;   // the first n_q queries
        // SYN data: uniform +-1.7 (C6 / C18), deterministic per layer (C18 c18_attn.cu:193-199)
        std::vector<float> Qh; std::vector<uint16_t> K16, V16;
        {
            std::mt19937_64 rg(0xC18ULL + (uint64_t) L * 7919ULL + (uint64_t) nk);
            auto u = [&]() { return (float) ((rg() >> 40) * (2.0 / 16777216.0) - 1.0) * 1.7f; };
            Qh.resize((size_t) h.n_q * 24 * 256); for (auto & x : Qh) x = u();
            K16.resize((size_t) nk * 512); V16.resize((size_t) nk * 512);
            for (auto & x : K16) x = f2h(u()); for (auto & x : V16) x = f2h(u());
        }
        std::set<int> types; types.insert(TF16); for (auto & v : V) if (v.kind == 2 || v.kind == 4) types.insert(v.T);
        char * dK[4] = {}, * dV[4] = {}; uint16_t * dPV = nullptr, * dPVn[4] = {}; uint8_t * dQF[4] = {}; uint32_t * dANY[4] = {};
        for (int T : types) {
            std::vector<uint8_t> C;
            build_cache(T, K16, nk, C); CK(hipMalloc(&dK[T], C.size())); CK(hipMemcpy(dK[T], C.data(), C.size(), hipMemcpyHostToDevice));
            build_cache(T, V16, nk, C); CK(hipMalloc(&dV[T], C.size())); CK(hipMemcpy(dV[T], C.data(), C.size(), hipMemcpyHostToDevice));
            CK(hipMalloc(&dPVn[T], (size_t) nk * 2 * 256 * 2)); CK(hipMalloc(&dQF[T], (size_t) nk / 4 * 2)); CK(hipMalloc(&dANY[T], 4));
        }
        const size_t nel = (size_t) nk * 2 * 256;
        CK(hipMalloc(&dPV, nel * 2));
        auto pack_pr = [&]() {   // C18 PACKV-f16 = the PR's qsa_pack_v indexing
            const uint32_t count = nel / 4;
            hipLaunchKernelGGL(c18_pack_v<TF16>, dim3((count + 255) / 256), dim3(256), 0, 0, dV[TF16], dPV, 2 * (size_t) HB[TF16], (size_t) HB[TF16], nk, count);
        };
        auto pack_new = [&](int T) {
            const uint64_t count = (uint64_t) (nk / 4) * 256 * 2;
            const unsigned blocks = (unsigned) ((count + SA_PACK_THREADS - 1) / SA_PACK_THREADS);
            const size_t nb1 = 2 * (size_t) HB[T], nb2 = HB[T], nb3 = 0;
            CK(hipMemsetAsync(dANY[T], 0, 4, 0));   // as ggml_cuda_sparse_attn_pack_v
            switch (sa_id(T)) {
                case SA_T_F16:  sa_pack_v<SA_T_F16> <<<blocks, SA_PACK_THREADS>>>(dV[T], dPVn[T], nb1, nb2, nb3, 2, (uint32_t) (nk / 4), count, dQF[T], dANY[T]); break;
                case SA_T_BF16: sa_pack_v<SA_T_BF16><<<blocks, SA_PACK_THREADS>>>(dV[T], dPVn[T], nb1, nb2, nb3, 2, (uint32_t) (nk / 4), count, dQF[T], dANY[T]); break;
                case SA_T_Q8_0: sa_pack_v<SA_T_Q8_0><<<blocks, SA_PACK_THREADS>>>(dV[T], dPVn[T], nb1, nb2, nb3, 2, (uint32_t) (nk / 4), count, dQF[T], dANY[T]); break;
                default:        sa_pack_v<SA_T_Q4_0><<<blocks, SA_PACK_THREADS>>>(dV[T], dPVn[T], nb1, nb2, nb3, 2, (uint32_t) (nk / 4), count, dQF[T], dANY[T]); break;
            }
        };
        for (int T : types) pack_new(T);
        pack_pr();
        // PR plan (C18: the PR's rows + merge kernels with the PR launch parameters), on the first n_q queries
        int32_t * d_ids; float * d_q, * d_out;
        CK(hipMalloc(&d_ids, (size_t) n_q * ns * 4)); CK(hipMemcpy(d_ids, rows.data(), (size_t) n_q * ns * 4, hipMemcpyHostToDevice));
        CK(hipMalloc(&d_q, Qh.size() * 4)); CK(hipMemcpy(d_q, Qh.data(), Qh.size() * 4, hipMemcpyHostToDevice));
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
        // the same plan in the ggml.h layout (padding: the last real quad with mask 0)
        std::vector<uint32_t> ub((size_t) ng4 * cap4), cnt(ng4); std::vector<uint16_t> um((size_t) ng4 * cap4); std::vector<int> uc(ng4);
        CK(hipMemcpy(ub.data(), d_ublk, ub.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(um.data(), d_umask, um.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(uc.data(), d_ucount, ng4 * 4, hipMemcpyDeviceToHost));
        double ent = 0;
        for (int g = 0; g < ng4; ++g) {
            cnt[g] = (uint32_t) uc[g]; ent += uc[g];
            uint32_t last = 0;
            for (int j = 0; j < cap4; ++j) {
                uint32_t & q = ub[(size_t) g * cap4 + j]; uint16_t & m = um[(size_t) g * cap4 + j];
                if (j < uc[g] && q != QSA3_BLK_SENT) { last = q; } else { q = last; m = 0; }
            }
        }
        uint32_t * d_cnt, * d_quads; uint16_t * d_masks; uint8_t * d_flags;
        CK(hipMalloc(&d_cnt, ng4 * 4)); CK(hipMalloc(&d_quads, ub.size() * 4)); CK(hipMalloc(&d_masks, um.size() * 2)); CK(hipMalloc(&d_flags, (size_t) ng4 * 2 * 8));
        CK(hipMemcpy(d_cnt, cnt.data(), ng4 * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(d_quads, ub.data(), ub.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_masks, um.data(), um.size() * 2, hipMemcpyHostToDevice));
        const qsa3_layout s = {(size_t) 24 * 1024, (size_t) 1024, (size_t) 1024, (size_t) 24 * 1024, 0, 2 * (size_t) HB[TF16], 2 * (size_t) HB[TF16], (size_t) HB[TF16], (size_t) HB[TF16], nk, n_q, 12, cap4, 1.0f / 16.0f, 24, 0};
        auto args = [&](const variant & v) {
            sa_g4_args a = {};
            a.q = (const char *) d_q; a.q_nb1 = 24 * 1024; a.q_nb2 = 1024; a.q_nb3 = 0;
            a.k = dK[v.T]; a.k_nb1 = 2 * (size_t) HB[v.T]; a.k_nb2 = HB[v.T]; a.k_nb3 = 0;
            a.v = dV[v.T]; a.v_nb1 = 2 * (size_t) HB[v.T]; a.v_nb2 = HB[v.T]; a.v_nb3 = 0;
            a.pv = dPVn[v.T]; a.pv_nquads = (size_t) nk / 4;
            a.counts = d_cnt; a.quads = d_quads; a.masks = d_masks; a.n_groups = ng4; a.cap = cap4;
            a.n_q = n_q; a.n_head = 24; a.n_head_kv = 2; a.scale = 1.0f / 16.0f; a.out = d_out; a.flags = d_flags;
            if (v.vl == SA_V_PACKED) { a.qflags = dQF[v.T]; a.qany = dANY[v.T]; }
            return a;
        };
        auto launch = [&](const variant & v) {
            switch (v.kind) {
                case 0: hipLaunchKernelGGL((qsa3_attn_kernel<true, 8, false, false>), dim3(ng4, 2), dim3(256), 0, 0, d_q, (const uint16_t *) dK[TF16], dPV, nullptr, d_ublk, d_umask, d_ucount, d_out, s, nullptr, nullptr, false); break;
                case 1: hipLaunchKernelGGL((c18_attn_kernel<TF16, LD, TF16, LP>), dim3(ng4, 2), dim3(256), 0, 0, d_q, dK[TF16], (const char *) dPV, d_ublk, d_umask, d_ucount, d_out, s); break;
                case 2: { const sa_g4_args a = args(v); KN[v.name]<<<dim3(ng4, 2, 1), 256>>>(a); FX[v.T]<<<dim3(ng4, 2, 1), 256>>>(a, 12); } break;
                case 3: pack_pr(); break;
                case 4: pack_new(v.T); break;
            }
            CK(hipGetLastError());
        };
        // correctness: every new variant vs the PR-verbatim output (f16), and P vs D of a type bitwise
        std::vector<float> ref(outn), o(outn); std::map<int, std::vector<float>> firstT;
        CK(hipMemset(d_out, 0xff, outb)); launch(V[0]); CK(hipDeviceSynchronize()); CK(hipMemcpy(ref.data(), d_out, outb, hipMemcpyDeviceToHost));
        for (auto & v : V) {
            if (v.kind != 2) continue;
            CK(hipMemset(d_out, 0xff, outb)); launch(v); CK(hipDeviceSynchronize()); CK(hipMemcpy(o.data(), d_out, outb, hipMemcpyDeviceToHost));
            size_t neq = 0, nnan = 0; double se = 0, sr = 0;
            for (size_t k = 0; k < outn; ++k) { neq += memcmp(&o[k], &ref[k], 4) != 0; nnan += std::isnan(o[k]); se += (double) (o[k] - ref[k]) * (o[k] - ref[k]); sr += (double) ref[k] * ref[k]; }
            size_t neqT = 0; if (firstT.count(v.T)) { for (size_t k = 0; k < outn; ++k) neqT += memcmp(&o[k], &firstT[v.T][k], 4) != 0; } else firstT[v.T] = o;
            printf("L%d CHK %-16s vs_PR-V0-f16 words_differ %zu nmse %.3e nan %zu | vs_same_type_first words_differ %zu\n", L, v.name.c_str(), neq, sr > 0 ? se / sr : 0.0, nnan, neqT);
        }
        // timing
        std::vector<std::vector<float>> t(V.size());
        for (int it = 0; it < 3 + reps; ++it) {
            for (size_t i = 0; i < V.size(); ++i) {
                CK(hipEventRecord(e0, 0)); launch(V[i]); CK(hipEventRecord(e1, 0)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it >= 3) t[i].push_back(ms);
            }
        }
        for (size_t i = 0; i < V.size(); ++i) {
            std::vector<float> x = t[i]; std::sort(x.begin(), x.end());
            const double med = x.size() % 2 ? x[x.size() / 2] : 0.5 * (x[x.size() / 2 - 1] + x[x.size() / 2]);
            printf("L%d RES %-16s n_q %d nk %d ch/q %.2f med %.4f min %.4f\n", L, V[i].name.c_str(), n_q, nk, ent / ng4 / 4 / 4, med, x[0]);
            sum_med[V[i].name] += med;
        }
        fflush(stdout);
        ++n_layers;
        for (int T : types) { hipFree(dK[T]); hipFree(dV[T]); hipFree(dPVn[T]); hipFree(dQF[T]); hipFree(dANY[T]); }
        hipFree(dPV); hipFree(d_ids); hipFree(d_q); hipFree(d_out); hipFree(d_ublk); hipFree(d_umask); hipFree(d_ucount); hipFree(d_srow); hipFree(d_sflag);
        hipFree(d_cnt); hipFree(d_quads); hipFree(d_masks); hipFree(d_flags);
    }
    printf("SUMMARY layers %d (mean of per-layer medians, ms)\n", n_layers);
    for (auto & kv : sum_med) printf("SUM %-16s %.4f\n", kv.first.c_str(), kv.second / n_layers);
    return 0;
}
