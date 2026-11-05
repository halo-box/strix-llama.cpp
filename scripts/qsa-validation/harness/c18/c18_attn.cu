// C18: loader microbench (direct-register vs LDS-staged dequant, per type, K and V separately) on captured QSA rows.
// hipcc --offload-arch=gfx1151 -O3 -std=c++17 c18_attn.cu -o c18_attn
// usage: c18_attn <rows_file | SYN:U:nk:seed> <qkv_file | SYN> reps layers(all|list) variants(all|ksweep|vsweep|pair|list)
//   rows  : C5 capture (FLASH_ATTN_EXT src[5] of every QSA layer, ub4096) or the C6 synthetic fixed-work plan
//           (513 quads/query, G4 union round(U*513) quads drawn uniformly over nk/4 quads)
//   data  : real f16 Q/K/V of the same calls (C5 .qkv) or SYN (uniform +-1.7, as C6). q8_0 / q4_0 / bf16 caches are
//           built on the host from the f16 K/V with the ggml reference quantizers (quantize_row_q8_0_ref / _q4_0_ref, RNE bf16)
// Plan: the PR's own rows+merge kernels (pr_union_kernels.inc, verbatim) with the PR launch parameters.
// Kernels: PR verbatim qsa3_attn_kernel (pr_attn.inc) for f16/q8_0 as references; c18_attn_kernel<KT,KL,VT,VL> (c18_kernel.inc).
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

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

struct hdr { char name[64]; int32_t n_q, ns, nk, has_mask, pos0, kt; };
struct qhdr { char name[64]; int32_t which, type; int64_t ne[4], nb[4]; uint64_t nbytes; };

static float h2f(uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }
static uint16_t f2h(float f) { return __builtin_bit_cast(uint16_t, (_Float16) f); }
static uint16_t f2bf(float f) { uint32_t u = __builtin_bit_cast(uint32_t, f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t) (u >> 16); }
static float bf2f(uint16_t b) { return __builtin_bit_cast(float, (uint32_t) b << 16); }

static const int HB[5] = {512, 272, 144, 512, 512};   // bytes per head row of 256 dims: f16, q8_0, q4_0, bf16, bf16
static const char * TN[5] = {"f16", "q8_0", "q4_0", "bf16", "bf16w"};
static const char * LN[5] = {"D", "Dd", "X", "S", "P"};

// host caches of one type from f16 rows [nk][2 heads][256]
static void build_cache(int T, const std::vector<uint16_t> & f, int nk, std::vector<uint8_t> & out) {
    const size_t rowb = 2 * (size_t) HB[T];
    out.assign((size_t) nk * rowb, 0);
    for (size_t kh = 0; kh < (size_t) nk * 2; ++kh) {
        const uint16_t * x = f.data() + kh * 256;
        uint8_t * o = out.data() + kh * HB[T];
        if (T == TF16) { memcpy(o, x, 512); }
        else if (T == TBF || T == TBW) { uint16_t * ob = (uint16_t *) o; for (int d = 0; d < 256; ++d) ob[d] = f2bf(h2f(x[d])); }
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
// value of the cache of type T at (key, head, dim) as the kernels see it (f16 after dequant, or the bf16 value for TBW)
static float cache_val(int T, const std::vector<uint8_t> & B, int key, int kvh, int d) {
    const uint8_t * o = B.data() + ((size_t) key * 2 + kvh) * HB[T];
    if (T == TF16) return h2f(((const uint16_t *) o)[d]);
    if (T == TBW) return bf2f(((const uint16_t *) o)[d]);
    if (T == TBF) { float v = bf2f(((const uint16_t *) o)[d]); v = std::min(65504.f, std::max(-65504.f, v)); return (float) (_Float16) v; }
    if (T == TQ8) { const block_q8_0 * bq = (const block_q8_0 *) o + d / 32; return (float) (_Float16) ((float) bq->qs[d % 32] * h2f(bq->d)); }
    const block_q4_0 * bq = (const block_q4_0 *) o + d / 32; const int j = d % 32;
    const int qv = j < 16 ? (bq->qs[j] & 15) : (bq->qs[j - 16] >> 4);
    return (float) (_Float16) ((float) (qv - 8) * h2f(bq->d));
}

struct variant { std::string name, group; int KT, KL, VT, VL; bool pr; };
typedef void (*kfn_t)(const float *, const char *, const char *, const uint32_t *, const uint16_t *, const int *, float *, const qsa3_layout);
template<int KT, int KL, int VT, int VL> static kfn_t kf() { return c18_attn_kernel<KT, KL, VT, VL>; }

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: rows|SYN:U:nk:seed qkv|SYN reps layers variants\n"); return 1; }
    const std::string rowsf = argv[1], qkvf = argv[2];
    const int reps = atoi(argv[3]);
    std::set<int> layers; std::set<std::string> vsel;
    { const std::string a = argv[4]; if (a != "all") { std::string x; for (char c : a + ",") { if (c == ',') { layers.insert(atoi(x.c_str())); x.clear(); } else x += c; } } }
    { const std::string a = argv[5]; std::string x; for (char c : a + ",") { if (c == ',') { vsel.insert(x); x.clear(); } else x += c; } }

    // ---- variant table ----
    std::vector<variant> V; std::map<std::string, kfn_t> KF;
    auto add = [&](const std::string & grp, int KT, int KL, int VT, int VL, kfn_t f) {
        const std::string n = std::string("K") + TN[KT] + LN[KL] + "-V" + TN[VT] + LN[VL];
        for (auto & v : V) if (v.name == n) { v.group += "," + grp; return; }
        V.push_back({n, grp, KT, KL, VT, VL, false}); KF[n] = f;
    };
#define KS_(T, L) add("ksweep", T, L, TF16, LP, kf<T, L, TF16, LP>())
    KS_(TF16, LD); KS_(TF16, LX); KS_(TF16, LS);
    KS_(TQ8, LD); KS_(TQ8, LDD); KS_(TQ8, LX); KS_(TQ8, LS);
    KS_(TQ4, LD); KS_(TQ4, LDD); KS_(TQ4, LX); KS_(TQ4, LS);
    KS_(TBF, LD); KS_(TBF, LDD); KS_(TBF, LX); KS_(TBF, LS);
    KS_(TBW, LD); KS_(TBW, LX); KS_(TBW, LS);
#define VS_(T, L) add("vsweep", TF16, LD, T, L, kf<TF16, LD, T, L>())
    VS_(TF16, LP); VS_(TBW, LP);
    VS_(TF16, LD); VS_(TF16, LX); VS_(TF16, LS);
    VS_(TQ8, LD); VS_(TQ8, LX); VS_(TQ8, LS);
    VS_(TQ4, LD); VS_(TQ4, LX); VS_(TQ4, LS);
    VS_(TBF, LD); VS_(TBF, LX); VS_(TBF, LS);
    VS_(TBW, LD); VS_(TBW, LX); VS_(TBW, LS);
#define PA_(T, KL, VL) add("pair", T, KL, T, VL, kf<T, KL, T, VL>())
    PA_(TF16, LD, LP); PA_(TF16, LD, LD); PA_(TF16, LX, LX); PA_(TF16, LX, LS); PA_(TF16, LS, LS);
    PA_(TQ8, LD, LP);  PA_(TQ8, LD, LD);  PA_(TQ8, LX, LX);  PA_(TQ8, LX, LS);  PA_(TQ8, LS, LS);
    PA_(TQ4, LD, LP);  PA_(TQ4, LD, LD);  PA_(TQ4, LX, LX);  PA_(TQ4, LX, LS);  PA_(TQ4, LS, LS);
    PA_(TBF, LD, LP);  PA_(TBF, LD, LD);  PA_(TBF, LX, LX);  PA_(TBF, LX, LS);  PA_(TBF, LS, LS);
    PA_(TBW, LD, LP);  PA_(TBW, LX, LS);
    // PR verbatim references (route V0 = K direct + V packed; VD = K and V direct)
    V.push_back({"PR-V0-f16", "pr", TF16, LD, TF16, LP, true}); V.push_back({"PR-V0-q8_0", "pr", TQ8, LD, TQ8, LP, true});
    V.push_back({"PR-VD-f16", "pr", TF16, LD, TF16, LD, true}); V.push_back({"PR-VD-q8_0", "pr", TQ8, LD, TQ8, LD, true});
    std::vector<int> vl;
    for (int i = 0; i < (int) V.size(); ++i) {
        bool on = vsel.count("all") || vsel.count(V[i].name) || V[i].pr;
        for (const auto & s : vsel) if (V[i].group.find(s) != std::string::npos && s.size() > 2) on = true;
        if (on) vl.push_back(i);
    }
    // ---- fixture ----
    FILE * fr = nullptr; bool synrows = rowsf.rfind("SYN:", 0) == 0;
    double synU = 0; int synNk = 0; uint64_t synSeed = 1;
    if (synrows) { if (sscanf(rowsf.c_str(), "SYN:%lf:%d:%llu", &synU, &synNk, (unsigned long long *) &synSeed) != 3) { fprintf(stderr, "bad SYN spec\n"); return 1; } }
    else { fr = fopen(rowsf.c_str(), "rb"); if (!fr) { perror("rows"); return 1; } }
    FILE * fq = qkvf == "SYN" ? nullptr : fopen(qkvf.c_str(), "rb");
    if (qkvf != "SYN" && !fq) { perror("qkv"); return 1; }
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    printf("# c18 rows %s data %s reps %d variants %zu\n", rowsf.c_str(), qkvf.c_str(), reps, vl.size());
    std::map<std::string, double> sum_med; std::map<std::string, double> sum_ch; int n_layers = 0;
    int L = -1;
    for (;;) {
        hdr h; std::vector<int32_t> rows;
        if (synrows) {
            if (L >= 0) break;
            memset(&h, 0, sizeof(h)); snprintf(h.name, 64, "syn"); h.n_q = 4096; h.ns = 2051; h.nk = synNk;
            const int n_q = 4096, ns = 2051, nqd = 513, ng4 = n_q / 4, nQ = synNk / 4, uq = (int) std::lround(synU * nqd);
            if (uq > nQ) { fprintf(stderr, "union too large\n"); return 1; }
            std::mt19937_64 rng(synSeed * 1000003ULL + (uint64_t) synNk);   // same generator as C6 c6_attn.cu (uni mode)
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
        const int n_q = h.n_q, ns = h.ns, nk = h.nk;
        std::vector<float> Qh; std::vector<uint16_t> K16, V16;
        if (fq) {
            for (int i = 0; i < 3; ++i) {
                qhdr qh; if (fread(&qh, sizeof(qh), 1, fq) != 1) { fprintf(stderr, "short qkv\n"); return 1; }
                std::vector<uint8_t> B(qh.nbytes); if (fread(B.data(), 1, qh.nbytes, fq) != qh.nbytes) { fprintf(stderr, "short qkv data\n"); return 1; }
                if (i == 0) { if (!(qh.type == 0 && qh.nb[1] == 24 * 1024 && qh.nb[2] == 1024 && qh.ne[1] == n_q && qh.ne[2] == 24)) { fprintf(stderr, "Q layout\n"); return 1; }
                              Qh.resize((size_t) n_q * 24 * 256); memcpy(Qh.data(), B.data(), Qh.size() * 4); }
                else { if (!(qh.type == 1 && qh.ne[1] == nk && qh.nb[1] == 1024 && qh.nb[2] == 512 && B.size() >= (size_t) nk * 1024)) { fprintf(stderr, "K/V layout\n"); return 1; }
                       std::vector<uint16_t> & D = i == 1 ? K16 : V16; D.resize((size_t) nk * 512); memcpy(D.data(), B.data(), D.size() * 2); }
            }
        }
        if (!layers.empty() && !layers.count(L)) continue;
        if (!fq) {   // SYN data: uniform +-1.7 (C6), deterministic per layer
            std::mt19937_64 rg(0xC18ULL + (uint64_t) L * 7919ULL + (uint64_t) nk);
            auto u = [&]() { return (float) ((rg() >> 40) * (2.0 / 16777216.0) - 1.0) * 1.7f; };
            Qh.resize((size_t) n_q * 24 * 256); for (auto & x : Qh) x = u();
            K16.resize((size_t) nk * 512); V16.resize((size_t) nk * 512);
            for (auto & x : K16) x = f2h(u()); for (auto & x : V16) x = f2h(u());
        }
        // caches per type (TBW shares the bf16 bytes)
        std::vector<uint8_t> KC[5], VC[5];
        std::set<int> needK, needV;
        for (int i : vl) { needK.insert(V[i].KT == TBW ? TBF : V[i].KT); needV.insert(V[i].VT == TBW ? TBF : V[i].VT); }
        char * dK[5] = {}, * dV[5] = {}; uint16_t * dPV[5] = {};
        for (int T : needK) { build_cache(T, K16, nk, KC[T]); CK(hipMalloc(&dK[T], KC[T].size())); CK(hipMemcpy(dK[T], KC[T].data(), KC[T].size(), hipMemcpyHostToDevice)); }
        for (int T : needV) { build_cache(T, V16, nk, VC[T]); CK(hipMalloc(&dV[T], VC[T].size())); CK(hipMemcpy(dV[T], VC[T].data(), VC[T].size(), hipMemcpyHostToDevice)); }
        dK[TBW] = dK[TBF]; dV[TBW] = dV[TBF]; KC[TBW] = KC[TBF]; VC[TBW] = VC[TBF];
        const size_t nel = (size_t) nk * 2 * 256;
        auto pack = [&](int T) {
            const int S = T == TBW ? TBF : T; const size_t nb1 = 2 * (size_t) HB[S], nb2 = HB[S]; const uint32_t count = nel / 4;
            const dim3 gr((count + 255) / 256), bl(256);
            switch (T) {
            case TF16: hipLaunchKernelGGL(c18_pack_v<TF16>, gr, bl, 0, 0, dV[S], dPV[T], nb1, nb2, nk, count); break;
            case TQ8:  hipLaunchKernelGGL(c18_pack_v<TQ8>,  gr, bl, 0, 0, dV[S], dPV[T], nb1, nb2, nk, count); break;
            case TQ4:  hipLaunchKernelGGL(c18_pack_v<TQ4>,  gr, bl, 0, 0, dV[S], dPV[T], nb1, nb2, nk, count); break;
            case TBF:  hipLaunchKernelGGL(c18_pack_v<TBF>,  gr, bl, 0, 0, dV[S], dPV[T], nb1, nb2, nk, count); break;
            case TBW:  hipLaunchKernelGGL(c18_pack_v<TBW>,  gr, bl, 0, 0, dV[S], dPV[T], nb1, nb2, nk, count); break;
            }
            CK(hipGetLastError());
        };
        std::set<int> needP; for (int i : vl) if (V[i].VL == LP) needP.insert(V[i].VT);
        for (int T : needP) { CK(hipMalloc(&dPV[T], nel * 2)); pack(T); }
        // PR's packed V for the PR references (f16 cache, and q8_0 cache)
        CK(hipDeviceSynchronize());
        // plan (PR verbatim rows + merge kernels)
        int32_t * d_ids; float * d_q, * d_out, * d_ref;
        CK(hipMalloc(&d_ids, rows.size() * 4)); CK(hipMemcpy(d_ids, rows.data(), rows.size() * 4, hipMemcpyHostToDevice));
        CK(hipMalloc(&d_q, Qh.size() * 4)); CK(hipMemcpy(d_q, Qh.data(), Qh.size() * 4, hipMemcpyHostToDevice));
        const size_t outn = (size_t) n_q * 24 * 256, outb = outn * 4;
        CK(hipMalloc(&d_out, outb)); CK(hipMalloc(&d_ref, outb));
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
        std::vector<int> uc(ng4); CK(hipMemcpy(uc.data(), d_ucount, ng4 * 4, hipMemcpyDeviceToHost));
        double ent = 0; for (int x : uc) ent += x;
        const double chunks = ent / 4.0 * 2.0;   // chunk-heads per call
        auto lay = [&](int KT, int VT) {
            const int SK = KT == TBW ? TBF : KT, SV = VT == TBW ? TBF : VT;
            return qsa3_layout{(size_t) 24 * 1024, (size_t) 1024, (size_t) 1024, (size_t) 24 * 1024, 0,
                               2 * (size_t) HB[SK], 2 * (size_t) HB[SV], (size_t) HB[SK], (size_t) HB[SV], nk, n_q, 12, cap4, 1.0f / 16.0f, 24, 0};
        };
        auto launch = [&](int i, float * dst) {
            const variant & v = V[i];
            const dim3 gr(ng4, 2), bl(256);
            if (v.pr) {
                const bool q8 = v.KT == TQ8;
                const qsa3_layout s = lay(v.KT, v.VT);
                if (v.VL == LP) {   // PR V0 packs V with its own pack: equal to c18_pack_v for f16 / q8_0 (checked below)
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
        // ---- correctness: the first variant of each (K type, V type) pair vs a CPU f64 reference (3 queries x 24 heads,
        //      first processed layer only); every other variant bitwise against the first of its pair, on every layer ----
        std::vector<float> o(outn);
        std::map<std::pair<int,int>, std::vector<float>> first;
        const int qs_s[3] = {0, 2050, n_q - 1};
        for (int i : vl) {
            CK(hipMemset(d_out, 0xff, outb)); launch(i, d_out); CK(hipDeviceSynchronize());
            CK(hipMemcpy(o.data(), d_out, outb, hipMemcpyDeviceToHost));
            const variant & v = V[i];
            size_t nnan = 0; for (float x : o) nnan += std::isnan(x);
            const auto key = std::make_pair(v.KT, v.VT);
            if (!first.count(key)) {
                first[key] = o;
                double maxrel = -1;
                if (n_layers == 0) {
                    maxrel = 0;
                    for (int qq : qs_s) {
                        const int32_t * rr = rows.data() + (size_t) qq * ns;
                        std::set<int> cells; for (int k = 0; k < ns; ++k) if (rr[k] >= 0 && rr[k] < nk) cells.insert(rr[k]);
                        for (int hq = 0; hq < 24; ++hq) {
                            const int kvh = hq / 12; const float * qv = Qh.data() + ((size_t) qq * 24 + hq) * 256;
                            std::vector<double> sc; double mx = -1e300;
                            for (int c : cells) { double s = 0; for (int d = 0; d < 256; ++d) s += (double) (float) ((_Float16) qv[d] * (_Float16) (1.0f / 16.0f)) * cache_val(v.KT, KC[v.KT], c, kvh, d); sc.push_back(s); mx = std::max(mx, s); }
                            double den = 0; for (double & s : sc) { s = std::exp(s - mx); den += s; }
                            std::vector<double> acc(256, 0); int ci = 0;
                            for (int c : cells) { for (int d = 0; d < 256; ++d) acc[d] += sc[ci] * cache_val(v.VT, VC[v.VT], c, kvh, d); ++ci; }
                            double amax = 0, emax = 0;
                            for (int d = 0; d < 256; ++d) { const double rv = acc[d] / den; amax = std::max(amax, std::fabs(rv)); emax = std::max(emax, std::fabs(rv - o[((size_t) qq * 24 + hq) * 256 + d])); }
                            maxrel = std::max(maxrel, emax / std::max(amax, 1e-30));
                        }
                    }
                }
                printf("L%d CHK %-22s pair %s/%s cpu_f64_maxrel %.3e nan %zu (pair reference)\n", L, v.name.c_str(), TN[v.KT], TN[v.VT], maxrel, nnan);
            } else {
                const auto & f = first[key]; size_t neq = 0; double md = 0;
                for (size_t k = 0; k < outn; ++k) { neq += memcmp(&o[k], &f[k], 4) != 0; md = std::max(md, (double) std::fabs(o[k] - f[k])); }
                printf("L%d CHK %-22s pair %s/%s words_differ_vs_pair_ref %zu max_abs %.3e nan %zu\n", L, v.name.c_str(), TN[v.KT], TN[v.VT], neq, md, nnan);
            }
        }
        // ---- timing: 3 warm-up rounds, then reps rounds; every variant once per round (interleaved), plus the V packs ----
        std::vector<std::vector<float>> t(V.size()); std::map<int, std::vector<float>> tp;
        for (int it = 0; it < 3 + reps; ++it) {
            for (int i : vl) {
                CK(hipEventRecord(e0, 0)); launch(i, d_out); CK(hipEventRecord(e1, 0)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it >= 3) t[i].push_back(ms);
            }
            for (int T : needP) {
                CK(hipEventRecord(e0, 0)); pack(T); CK(hipEventRecord(e1, 0)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it >= 3) tp[T].push_back(ms);
            }
        }
        auto stats = [](std::vector<float> x, double & med, double & mean, double & sd, double & mn) {
            std::sort(x.begin(), x.end()); mean = 0; sd = 0; for (float y : x) mean += y; mean /= x.size();
            for (float y : x) sd += (y - mean) * (y - mean); sd = std::sqrt(sd / std::max<size_t>(1, x.size() - 1));
            med = x.size() % 2 ? x[x.size() / 2] : 0.5 * (x[x.size() / 2 - 1] + x[x.size() / 2]); mn = x[0];
        };
        for (int i : vl) {
            double med, mean, sd, mn; stats(t[i], med, mean, sd, mn);
            printf("L%d RES %-22s nk %d ch/q %.2f med %.4f mean %.4f sd %.4f min %.4f n %zu ns/chunk %.2f\n", L, V[i].name.c_str(), nk, ent / ng4 / 4 / 4, med, mean, sd, mn, t[i].size(), med * 1e6 / chunks);
            sum_med[V[i].name] += med; sum_ch[V[i].name] += chunks;
        }
        for (auto & kv : tp) {
            double med, mean, sd, mn; stats(kv.second, med, mean, sd, mn);
            const std::string n = std::string("PACKV-") + TN[kv.first];
            printf("L%d RES %-22s nk %d med %.4f mean %.4f sd %.4f min %.4f n %zu\n", L, n.c_str(), nk, med, mean, sd, mn, kv.second.size());
            sum_med[n] += med;
        }
        fflush(stdout);
        ++n_layers;
        for (int T = 0; T < 5; ++T) { if (T != TBW) { if (dK[T]) hipFree(dK[T]); if (dV[T]) hipFree(dV[T]); } if (dPV[T]) hipFree(dPV[T]); }
        hipFree(d_ids); hipFree(d_q); hipFree(d_out); hipFree(d_ref); hipFree(d_ublk); hipFree(d_umask); hipFree(d_ucount); hipFree(d_srow); hipFree(d_sflag);
    }
    printf("SUMMARY layers %d (mean of per-layer medians, ms; ns/chunk over all layers)\n", n_layers);
    for (auto & kv : sum_med) printf("SUM %-22s %.4f ns/chunk %.2f\n", kv.first.c_str(), kv.second / n_layers, sum_ch.count(kv.first) ? kv.second * 1e6 / sum_ch[kv.first] : 0.0);
    return 0;
}
