// R1-M-Lab A (op-level correctness bands): host-side fixtures, caches and oracles shared by the GPU check harnesses
// (bands_attn.cu, bands_dec.cu, built with hipcc) and the CPU harness (bands_cpu.cpp, built with g++ against ggml).
// Everything here runs on the host.
//
// R2-M-LAB-BANDS-OP adds two fixture modes (everything else is R1-M-Lab A's file unchanged):
//   UNI:A:nk:seed:n_q:ns:blocks  test-backend-ops' maskless inputs: Q/K/V ~ std::uniform_real_distribution<float>(-A, A) on a
//       std::default_random_engine (init_tensor_uniform, tests/test-backend-ops.cpp:59-108 at 94e086847; its seed there is
//       std::random_device, :71, so values cannot be reproduced, only the generator and distribution), K/V converted from f32
//       (f16 RNE; q8_0 = quantize_row_q8_0_ref on the f32 values, as ggml_quantize_chunk does); rows = the PR cell formula of
//       test_qsa_prefill_maskless::initialize_tensors (:11545-11574): blocks = 0 plain formula, blocks > 0 complete-block rows;
//       blocks = -1 takes the C6/C18 SYN rows (U 1.8) instead, to separate the row effect from the data effect.
//   TBO:<file.bdmp>  a dump written by the patched test-backend-ops (tree_bandsop.diff): the case's exact Q, K/V cache bytes,
//       rows, and its ROCm0 and CPU outputs.
// Fixture   : C5/C18 rows file (+ .qkv) or the C6/C18 synthetic plan SYN:U:nk:seed with SYN data (uniform +-1.7);
//             the generator code is C18's c18_attn.cu:155-173 / :193-199, copied verbatim.
// Caches    : C18's build_cache (c18_attn.cu:50-79): q8_0 = quantize_row_q8_0_ref, q4_0 = quantize_row_q4_0_ref, bf16 RNE,
//             plus f32 (the f16 values widened) for the CPU harness. bands_cpu checks these bytes against ggml_quantize_chunk.
// Dequant   : st = the type's to_float value (ggml dequantize_row_*: q8 qs*d, q4 (q-8)*d, bf16 bits<<16, f16, f32);
//             h  = st rounded to f16 (bf16 saturated to +-65504 first, as the kernels' fmed3), i.e. what the kernels see.
// Oracles (per query row, the row's valid cells deduplicated and ascending):
//   O_C18  f64, q = f16(f16(q) * f16(scale)), K/V = h        (C18's check, c18_attn.cu:290)
//   O_PLAN f64, q = q * scale in f32, K/V = st (no f16 rounding)  (PLAN-v3 §3 "second oracle", PLAN-v3.md:155)
//   O_CON  the §3 contract in f32: q*scale -> f16, K/V to_float -> f16, f32 scores (sequential over d), f32 online
//          softmax per cell in ascending order (expf), P rounded to f16 for PV, l summed from the f32 p (as the HIP
//          kernel, c18_kernel.inc:562), f32 PV accumulation, out = acc / l.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <set>
#include <random>
#include <algorithm>
#include <thread>
#include <atomic>
#include <functional>

// host arithmetic without FMA contraction (ggml-base's quantize_row_*_ref is built for baseline x86-64, no FMA)
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace bands {

static inline uint64_t fnv1a(const void * p, size_t n) { const uint8_t * b = (const uint8_t *) p; uint64_t h = 1469598103934665603ULL; for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; } return h; }

enum { BT_F16 = 0, BT_Q8 = 1, BT_Q4 = 2, BT_BF = 3, BT_F32 = 5 };
static inline const char * tname(int T) { switch (T) { case BT_F16: return "f16"; case BT_Q8: return "q8_0"; case BT_Q4: return "q4_0"; case BT_BF: return "bf16"; case BT_F32: return "f32"; } return "?"; }
static inline int hbytes(int T) { switch (T) { case BT_F16: return 512; case BT_Q8: return 272; case BT_Q4: return 144; case BT_BF: return 512; case BT_F32: return 1024; } return 0; }

static inline float h2f(uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }
static inline uint16_t f2h(float f) { return __builtin_bit_cast(uint16_t, (_Float16) f); }
static inline uint16_t f2bf(float f) { uint32_t u = __builtin_bit_cast(uint32_t, f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t) (u >> 16); }
static inline float bf2f(uint16_t b) { return __builtin_bit_cast(float, (uint32_t) b << 16); }
static inline float rh(float x) { return (float) (_Float16) x; }

struct bq8 { uint16_t d; int8_t qs[32]; };
struct bq4 { uint16_t d; uint8_t qs[16]; };

struct hdr { char name[64]; int32_t n_q, ns, nk, has_mask, pos0, kt; };
struct qhdr { char name[64]; int32_t which, type; int64_t ne[4], nb[4]; uint64_t nbytes; };

// ---------------- fixture ----------------
struct fixture {
    int L = -1, n_q = 0, ns = 0, nk = 0;
    std::vector<int32_t> rows;      // [n_q][ns]
    std::vector<float> Q;           // [n_q][24][256]
    std::vector<uint16_t> K16, V16; // [nk][2][256] f16
    std::vector<float> K32, V32;    // UNI: f32 source values ([nk][2][256]); caches are built from these
    int rawKT = -1, rawVT = -1; std::vector<uint8_t> Kraw, Vraw;   // TBO: exact cache bytes of type rawKT / rawVT
    std::vector<float> out_gpu, out_cpu;                            // TBO: test-backend-ops ROCm0 / CPU outputs [n_q][24][256]
    int blocks = 0; std::string label;
};

struct fixture_reader {
    FILE * fr = nullptr, * fq = nullptr; bool synrows = false; double synU = 0; int synNk = 0; uint64_t synSeed = 1; int L = -1;
    bool uni = false, tbo = false; double uniA = 1.0; int uniNk = 0, uniNq = 0, uniNs = 0, uniBlocks = 0, uniSort = 0; uint64_t uniSeed = 1; std::string tbof;
    bool open(const std::string & rowsf, const std::string & qkvf) {
        if (rowsf.rfind("UNI:", 0) == 0) { uni = true; if (sscanf(rowsf.c_str(), "UNI:%lf:%d:%llu:%d:%d:%d:%d", &uniA, &uniNk, (unsigned long long *) &uniSeed, &uniNq, &uniNs, &uniBlocks, &uniSort) < 6) { fprintf(stderr, "bad UNI spec\n"); return false; } return true; }
        if (rowsf.rfind("TBO:", 0) == 0) { tbo = true; tbof = rowsf.substr(4); return true; }
        synrows = rowsf.rfind("SYN:", 0) == 0;
        if (synrows) { if (sscanf(rowsf.c_str(), "SYN:%lf:%d:%llu", &synU, &synNk, (unsigned long long *) &synSeed) != 3) { fprintf(stderr, "bad SYN spec\n"); return false; } }
        else { fr = fopen(rowsf.c_str(), "rb"); if (!fr) { perror("rows"); return false; } }
        if (qkvf != "SYN") { fq = fopen(qkvf.c_str(), "rb"); if (!fq) { perror("qkv"); return false; } }
        return true;
    }

    // C6/C18 SYN rows (same generator as the SYN branch of next(), C18 c18_attn.cu:155-173), 4096 x 2051
    static void syn_rows(double U, int nk, uint64_t seed, std::vector<int32_t> & rows) {
        const int n_q = 4096, ns = 2051, nqd = 513, ng4 = n_q / 4, nQ = nk / 4, uq = (int) std::lround(U * nqd);
        if (uq > nQ) { fprintf(stderr, "union too large\n"); exit(1); }
        std::mt19937_64 rng(seed * 1000003ULL + (uint64_t) nk);
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
    }
    // test_qsa_prefill_maskless::initialize_tensors rows, verbatim (tests/test-backend-ops.cpp:11545-11573 at 94e086847)
    static void tbo_rows(int queries, int keys, int selected, int blocks, std::vector<int32_t> & picks) {
        picks.assign((size_t) queries * selected, 0);
        if (blocks > 0) {
            if (!(selected > 3 && (selected-3)%4 == 0 && keys%7919 != 0)) { fprintf(stderr, "blocks assert\n"); exit(1); }
            const int budget=(selected-3)/4;
            const auto cell=[&](int logical) { return int((int64_t(logical)*7919)%keys); };
            for (int q=0;q<queries;++q) {
                int32_t * row=picks.data()+size_t(q)*selected;
                std::fill(row,row+selected,-1);
                if (q==0 && queries>1) continue;
                const int vis=std::min(blocks+q/4,keys/4-1);
                const int n=std::min(vis,budget);
                for (int i=0;i<n;++i) {
                    const int b=vis<=budget ? i : int((int64_t(i)*vis+(q*131)%vis)/budget);
                    for (int c=0;c<4;++c) row[4*i+c]=cell(4*b+c);
                }
                for (int c=0;c<q%4;++c) row[4*budget+c]=cell(4*vis+c);
            }
            return;
        }
        for (int q=0;q<queries;++q) {
            for (int j=0;j<selected;++j) {
                int key=1+(j*37+q*13)%(keys-4);
                if (j==0) key=selected==1 ? 2 : -1;
                if (j==1) key=2;
                if (j>selected-3 && selected>4) key=-1;
                picks[q*selected+j]=key;
            }
        }
    }
    bool next_uni(fixture & F) {
        if (L >= 0) return false;
        ++L; F.L = 0; F.n_q = uniNq; F.ns = uniNs; F.nk = uniNk; F.blocks = uniBlocks;
        if (uniBlocks == -1) {
            if (uniNs != 2051 || uniNq > 4096) { fprintf(stderr, "SYN rows need ns 2051, n_q <= 4096\n"); exit(1); }
            std::vector<int32_t> r; syn_rows(1.8, uniNk, uniSeed, r); F.rows.assign(r.begin(), r.begin() + (size_t) uniNq * uniNs);
        } else tbo_rows(uniNq, uniNk, uniNs, uniBlocks, F.rows);
        // optional 7th field 1: sort every row ascending (-1 first), to test the cell-order dependence (O_CON is ascending)
        if (uniSort) for (int q = 0; q < uniNq; ++q) std::sort(F.rows.begin() + (size_t) q * uniNs, F.rows.begin() + (size_t) (q + 1) * uniNs);
        // init_tensor_uniform's generator and distribution (test-backend-ops.cpp:71-72), one engine per tensor, fixed seed
        auto fill = [&](std::vector<float> & x, size_t n, uint64_t which) {
            std::default_random_engine gen((unsigned) (uniSeed * 1000003ULL + which * 7919ULL + (uint64_t) uniNk));
            std::uniform_real_distribution<float> dist((float) -uniA, (float) uniA);
            x.resize(n); for (auto & v : x) v = dist(gen);
        };
        fill(F.Q, (size_t) uniNq * 24 * 256, 1); fill(F.K32, (size_t) uniNk * 512, 2); fill(F.V32, (size_t) uniNk * 512, 3);
        F.K16.resize(F.K32.size()); F.V16.resize(F.V32.size());
        for (size_t i = 0; i < F.K32.size(); ++i) { F.K16[i] = f2h(F.K32[i]); F.V16[i] = f2h(F.V32[i]); }
        char b[160]; snprintf(b, sizeof b, "UNI A %g nk %d seed %llu n_q %d ns %d blocks %d sort %d", uniA, uniNk, (unsigned long long) uniSeed, uniNq, uniNs, uniBlocks, uniSort); F.label = b;
        return true;
    }
    bool next_tbo(fixture & F) {
        if (L >= 0) return false;
        FILE * f = fopen(tbof.c_str(), "rb"); if (!f) { perror("tbo"); exit(1); }
        int32_t h[8]; if (fread(h, 4, 8, f) != 8) { fprintf(stderr, "short tbo\n"); exit(1); }
        const int queries = h[0], keys = h[1], selected = h[2], ratio = h[3], tk = h[6], tv = h[7];
        if (ratio != 12) { fprintf(stderr, "tbo ratio %d\n", ratio); exit(1); }
        auto bt = [](int ggt) { return ggt == 1 ? BT_F16 : ggt == 8 ? BT_Q8 : -1; };   // GGML_TYPE_F16 = 1, GGML_TYPE_Q8_0 = 8
        ++L; F.L = 0; F.n_q = queries; F.ns = selected; F.nk = keys; F.blocks = h[5]; F.rawKT = bt(tk); F.rawVT = bt(tv);
        if (F.rawKT < 0 || F.rawVT < 0) { fprintf(stderr, "tbo types %d %d\n", tk, tv); exit(1); }
        auto rd = [&](void * p, size_t n) { if (fread(p, 1, n, f) != n) { fprintf(stderr, "short tbo data\n"); exit(1); } };
        F.Q.resize((size_t) queries * 24 * 256); rd(F.Q.data(), F.Q.size() * 4);
        F.Kraw.resize((size_t) keys * 2 * hbytes(F.rawKT)); rd(F.Kraw.data(), F.Kraw.size());
        F.Vraw.resize((size_t) keys * 2 * hbytes(F.rawVT)); rd(F.Vraw.data(), F.Vraw.size());
        F.rows.resize((size_t) queries * selected); rd(F.rows.data(), F.rows.size() * 4);
        F.out_gpu.resize((size_t) queries * 24 * 256); rd(F.out_gpu.data(), F.out_gpu.size() * 4);
        F.out_cpu.resize(F.out_gpu.size()); rd(F.out_cpu.data(), F.out_cpu.size() * 4);
        fclose(f);
        if (F.rawKT == BT_F16) { F.K16.resize((size_t) keys * 512); memcpy(F.K16.data(), F.Kraw.data(), F.Kraw.size()); }
        if (F.rawVT == BT_F16) { F.V16.resize((size_t) keys * 512); memcpy(F.V16.data(), F.Vraw.data(), F.Vraw.size()); }
        char b[600]; snprintf(b, sizeof b, "TBO %s q %d keys %d sel %d interleaved %d blocks %d kt %s vt %s", tbof.c_str(), queries, keys, selected, h[4], h[5], tname(F.rawKT), tname(F.rawVT)); F.label = b;
        return true;
    }
    // reads the next layer; data are materialised only when want(L) (the .qkv stream is skipped otherwise)
    bool next(fixture & F, const std::function<bool(int)> & want) {
        if (uni) return next_uni(F);
        if (tbo) return next_tbo(F);
        hdr h;
        if (synrows) {
            if (L >= 0) return false;
            memset(&h, 0, sizeof(h)); h.n_q = 4096; h.ns = 2051; h.nk = synNk;
            const int n_q = 4096, ns = 2051, nqd = 513, ng4 = n_q / 4, nQ = synNk / 4, uq = (int) std::lround(synU * nqd);
            if (uq > nQ) { fprintf(stderr, "union too large\n"); return false; }
            std::mt19937_64 rng(synSeed * 1000003ULL + (uint64_t) synNk);   // C18 c18_attn.cu:157 (= C6 uni mode)
            std::vector<int> perm(nQ); for (int i = 0; i < nQ; ++i) perm[i] = i;
            F.rows.resize((size_t) n_q * ns);
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
                    int32_t * rr = F.rows.data() + (size_t) (4 * g + qi) * ns; int o = 0;
                    for (int i = 0; i < nqd; ++i) { const int nc = i + 1 < nqd ? 4 : ns - 4 * (nqd - 1); for (int c = 0; c < nc; ++c) rr[o++] = 4 * pick[i] + c; }
                }
            }
        } else {
            if (fread(&h, sizeof(h), 1, fr) != 1) return false;
            F.rows.resize((size_t) h.n_q * h.ns);
            if (fread(F.rows.data(), 4, F.rows.size(), fr) != F.rows.size()) { fprintf(stderr, "short rows\n"); exit(1); }
        }
        ++L; F.L = L; F.n_q = h.n_q; F.ns = h.ns; F.nk = h.nk;
        const int n_q = h.n_q, nk = h.nk;
        const bool w = want(L);
        if (fq) {
            for (int i = 0; i < 3; ++i) {
                qhdr qh; if (fread(&qh, sizeof(qh), 1, fq) != 1) { fprintf(stderr, "short qkv\n"); exit(1); }
                if (!w) { fseeko(fq, (off_t) qh.nbytes, SEEK_CUR); continue; }
                std::vector<uint8_t> B(qh.nbytes); if (fread(B.data(), 1, qh.nbytes, fq) != qh.nbytes) { fprintf(stderr, "short qkv data\n"); exit(1); }
                if (i == 0) { if (!(qh.type == 0 && qh.nb[1] == 24 * 1024 && qh.nb[2] == 1024 && qh.ne[1] == n_q && qh.ne[2] == 24)) { fprintf(stderr, "Q layout\n"); exit(1); }
                              F.Q.resize((size_t) n_q * 24 * 256); memcpy(F.Q.data(), B.data(), F.Q.size() * 4); }
                else { if (!(qh.type == 1 && qh.ne[1] == nk && qh.nb[1] == 1024 && qh.nb[2] == 512 && B.size() >= (size_t) nk * 1024)) { fprintf(stderr, "K/V layout\n"); exit(1); }
                       std::vector<uint16_t> & D = i == 1 ? F.K16 : F.V16; D.resize((size_t) nk * 512); memcpy(D.data(), B.data(), D.size() * 2); }
            }
        } else if (w) {   // SYN data: uniform +-1.7 (C6), deterministic per layer (C18 c18_attn.cu:193-199)
            std::mt19937_64 rg(0xC18ULL + (uint64_t) L * 7919ULL + (uint64_t) nk);
            auto u = [&]() { return (float) ((rg() >> 40) * (2.0 / 16777216.0) - 1.0) * 1.7f; };
            F.Q.resize((size_t) n_q * 24 * 256); for (auto & x : F.Q) x = u();
            F.K16.resize((size_t) nk * 512); F.V16.resize((size_t) nk * 512);
            for (auto & x : F.K16) x = f2h(u()); for (auto & x : F.V16) x = f2h(u());
        }
        return true;
    }
};

// ---------------- caches (C18 build_cache + f32) ----------------
static void build_cache(int T, const std::vector<uint16_t> & f, int nk, std::vector<uint8_t> & out) {
    const int HB = hbytes(T);
    out.assign((size_t) nk * 2 * HB, 0);
    for (size_t kh = 0; kh < (size_t) nk * 2; ++kh) {
        const uint16_t * x = f.data() + kh * 256;
        uint8_t * o = out.data() + kh * HB;
        if (T == BT_F16) { memcpy(o, x, 512); }
        else if (T == BT_F32) { float * of = (float *) o; for (int d = 0; d < 256; ++d) of[d] = h2f(x[d]); }
        else if (T == BT_BF) { uint16_t * ob = (uint16_t *) o; for (int d = 0; d < 256; ++d) ob[d] = f2bf(h2f(x[d])); }
        else if (T == BT_Q8) {
            bq8 * bq = (bq8 *) o;
            for (int b = 0; b < 8; ++b) {
                float am = 0; for (int i = 0; i < 32; ++i) am = std::max(am, std::fabs(h2f(x[32*b+i])));
                const float d = am / 127.f, id = d ? 1.f / d : 0.f;
                bq[b].d = f2h(d); for (int i = 0; i < 32; ++i) bq[b].qs[i] = (int8_t) roundf(h2f(x[32*b+i]) * id);
            }
        } else {
            bq4 * bq = (bq4 *) o;
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

// R2 (op): the cache of fixture F for type T. TBO: the dumped bytes (T must be the dumped type). UNI: f16 = RNE of the f32
// source (= K16), q8_0 = quantize_row_q8_0_ref on the f32 source (ggml-quants.c:278-301, as test-backend-ops' ggml_quantize_chunk);
// other types from the f16 values as before.
static bool build_cache_fx(int T, const fixture & F, bool isK, std::vector<uint8_t> & out) {
    const int rawT = isK ? F.rawKT : F.rawVT;
    if (rawT >= 0) { if (rawT != T) return false; out = isK ? F.Kraw : F.Vraw; return true; }
    const std::vector<float> & s32 = isK ? F.K32 : F.V32;
    if (T == BT_Q8 && !s32.empty()) {
        out.assign((size_t) F.nk * 2 * hbytes(T), 0);
        for (size_t kh = 0; kh < (size_t) F.nk * 2; ++kh) {
            const float * x = s32.data() + kh * 256; bq8 * bq = (bq8 *) (out.data() + kh * hbytes(T));
            for (int b = 0; b < 8; ++b) {
                float am = 0; for (int i = 0; i < 32; ++i) am = std::max(am, std::fabs(x[32*b+i]));
                const float d = am / 127.f, id = d ? 1.f / d : 0.f;
                bq[b].d = f2h(d); for (int i = 0; i < 32; ++i) bq[b].qs[i] = (int8_t) roundf(x[32*b+i] * id);
            }
        }
        return true;
    }
    build_cache(T, isK ? F.K16 : F.V16, F.nk, out); return true;
}
struct deq { std::vector<float> st, h; };   // [nk][2][256]
static void build_deq(int T, const std::vector<uint8_t> & B, int nk, deq & D) {
    const int HB = hbytes(T); const size_t n = (size_t) nk * 512;
    D.st.resize(n); D.h.resize(n);
    for (size_t kh = 0; kh < (size_t) nk * 2; ++kh) {
        const uint8_t * o = B.data() + kh * HB; float * st = D.st.data() + kh * 256; float * hh = D.h.data() + kh * 256;
        for (int d = 0; d < 256; ++d) {
            float v;
            if (T == BT_F16) v = h2f(((const uint16_t *) o)[d]);
            else if (T == BT_F32) v = ((const float *) o)[d];
            else if (T == BT_BF) v = bf2f(((const uint16_t *) o)[d]);
            else if (T == BT_Q8) { const bq8 * b = (const bq8 *) o + d / 32; v = (float) b->qs[d % 32] * h2f(b->d); }
            else { const bq4 * b = (const bq4 *) o + d / 32; const int j = d % 32; const int qv = j < 16 ? (b->qs[j] & 15) : (b->qs[j - 16] >> 4); v = (float) (qv - 8) * h2f(b->d); }
            st[d] = v;
            hh[d] = T == BT_BF ? rh(std::min(65504.f, std::max(-65504.f, v))) : rh(v);
        }
    }
}

// valid cells of a row: deduplicated, ascending, in [0, nk)
static void row_cells(const int32_t * rr, int ns, int nk, std::vector<int> & cells) {
    cells.clear(); for (int k = 0; k < ns; ++k) if (rr[k] >= 0 && rr[k] < nk) cells.push_back(rr[k]);
    std::sort(cells.begin(), cells.end()); cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
}
// planted row defects (R1-Q8 lab, SP/final219/r1q8/r1q8_lab.diff:73-87): d1 = last valid tail slot (W-3..W-1) -> -1;
// d2 = the median valid block (4 cells) -> -1
static void defect_row(int def, int32_t * row, int W) {
    if (def == 1) { for (int j = W - 1; j >= W - 3; --j) if (row[j] >= 0) { row[j] = -1; break; } }
    else if (def == 2) {
        const int nb = (W - 3) / 4; int nv = 0; for (int b = 0; b < nb; ++b) nv += row[4*b] >= 0;
        if (nv == 0) return; int want = nv / 2, k = 0;
        for (int b = 0; b < nb; ++b) { if (row[4*b] < 0) continue; if (k++ == want) { for (int c = 0; c < 4; ++c) row[4*b + c] = -1; break; } }
    }
}

enum { O_C18 = 0, O_PLAN = 1, O_CON = 2 };
static const char * oname(int o) { return o == O_C18 ? "f64_c18" : o == O_PLAN ? "f64_plan" : "contract"; }

// one (query, kv head): the 12 query heads kvh*12 .. kvh*12+11; q12 = Q + (query*24 + kvh*12)*256; out = [12][256]
static void oracle_group(int mode, const float * q12, const std::vector<int> & cells, const deq & K, const deq & V, int kvh,
                         float scale, float vscale, double * out) {
    const int nc = (int) cells.size();
    if (nc == 0) { for (int i = 0; i < 12 * 256; ++i) out[i] = 0; return; }
    if (mode != O_CON) {
        alignas(64) double qT[256][12];
        for (int h = 0; h < 12; ++h) for (int d = 0; d < 256; ++d) {
            const float x = q12[h * 256 + d];
            qT[d][h] = mode == O_C18 ? (double) (float) ((_Float16) x * (_Float16) scale) : (double) (x * scale);
        }
        const float * Ka = mode == O_C18 ? K.h.data() : K.st.data();
        const float * Va = mode == O_C18 ? V.h.data() : V.st.data();
        std::vector<double> S((size_t) nc * 12);
        double mx[12]; for (int h = 0; h < 12; ++h) mx[h] = -1e300;
        for (int ci = 0; ci < nc; ++ci) {
            const float * kr = Ka + ((size_t) cells[ci] * 2 + kvh) * 256;
            double acc[12] = {0};
            for (int d = 0; d < 256; ++d) { const double kd = kr[d]; for (int h = 0; h < 12; ++h) acc[h] += qT[d][h] * kd; }
            for (int h = 0; h < 12; ++h) { S[(size_t) ci * 12 + h] = acc[h]; mx[h] = std::max(mx[h], acc[h]); }
        }
        double den[12] = {0};
        for (int ci = 0; ci < nc; ++ci) for (int h = 0; h < 12; ++h) { double & s = S[(size_t) ci * 12 + h]; s = std::exp(s - mx[h]); den[h] += s; }
        for (int i = 0; i < 12 * 256; ++i) out[i] = 0;
        for (int ci = 0; ci < nc; ++ci) {
            const float * vr = Va + ((size_t) cells[ci] * 2 + kvh) * 256;
            for (int h = 0; h < 12; ++h) { const double w = S[(size_t) ci * 12 + h]; double * o = out + h * 256; for (int d = 0; d < 256; ++d) o[d] += w * (double) vr[d]; }
        }
        for (int h = 0; h < 12; ++h) for (int d = 0; d < 256; ++d) out[h * 256 + d] /= den[h];
        return;
    }
    // contract (f32)
    alignas(64) float qT[256][12];
    for (int h = 0; h < 12; ++h) for (int d = 0; d < 256; ++d) qT[d][h] = rh(q12[h * 256 + d] * scale);
    float m[12], l[12]; for (int h = 0; h < 12; ++h) { m[h] = -INFINITY; l[h] = 0.f; }
    alignas(64) float acc[12][256]; for (int h = 0; h < 12; ++h) for (int d = 0; d < 256; ++d) acc[h][d] = 0.f;
    alignas(64) float vtmp[256];
    for (int ci = 0; ci < nc; ++ci) {
        const float * kr = K.h.data() + ((size_t) cells[ci] * 2 + kvh) * 256;
        float s[12] = {0};
        for (int d = 0; d < 256; ++d) { const float kd = kr[d]; for (int h = 0; h < 12; ++h) s[h] += qT[d][h] * kd; }
        const float * vr;
        if (vscale == 1.f) vr = V.h.data() + ((size_t) cells[ci] * 2 + kvh) * 256;
        else { const float * vs = V.st.data() + ((size_t) cells[ci] * 2 + kvh) * 256; for (int d = 0; d < 256; ++d) vtmp[d] = rh(vs[d] * vscale); vr = vtmp; }
        for (int h = 0; h < 12; ++h) {
            if (s[h] > m[h]) {
                const float alpha = m[h] == -INFINITY ? 0.f : expf(m[h] - s[h]);
                for (int d = 0; d < 256; ++d) acc[h][d] *= alpha;
                l[h] *= alpha; m[h] = s[h];
            }
            const float p = expf(s[h] - m[h]);
            const float ph = rh(p);
            l[h] += p;
            for (int d = 0; d < 256; ++d) acc[h][d] += ph * vr[d];
        }
    }
    for (int h = 0; h < 12; ++h) for (int d = 0; d < 256; ++d) out[h * 256 + d] = l[h] > 0.f ? (double) (acc[h][d] / l[h]) : 0.0;
}

static void parallel_for(int n, int nthr, const std::function<void(int)> & f) {
    std::atomic<int> next(0); std::vector<std::thread> th;
    for (int t = 0; t < nthr; ++t) th.emplace_back([&]() { for (;;) { const int i = next.fetch_add(1); if (i >= n) break; f(i); } });
    for (auto & t : th) t.join();
}

// oracle outputs for the selected queries (all 24 heads): out[(si*24 + hq)*256 + d]
static void oracle_rows(int mode, const fixture & F, const std::vector<int> & sel, const std::vector<int32_t> & rows,
                        const deq & K, const deq & V, float scale, float vscale, std::vector<double> & out, int nthr) {
    out.assign(sel.size() * 24 * 256, 0.0);
    parallel_for((int) sel.size() * 2, nthr, [&](int t) {
        const int si = t >> 1, kvh = t & 1, q = sel[si];
        std::vector<int> cells; row_cells(rows.data() + (size_t) q * F.ns, F.ns, F.nk, cells);
        oracle_group(mode, F.Q.data() + ((size_t) q * 24 + kvh * 12) * 256, cells, K, V, kvh, scale, vscale, out.data() + ((size_t) si * 24 + kvh * 12) * 256);
    });
}

// ---------------- error statistics ----------------
// per row (query, head): maxrel = max|e| / max|ref|, rl2 = ||e|| / ||ref||; per op: nmse = sum e^2 / sum ref^2 (as test-backend-ops nmse, normalised by the reference)
struct errstat {
    double se = 0, sr = 0; std::vector<float> maxrel, rl2; size_t nan = 0;
    template<typename T> void add_row(const double * ref, const T * x) {
        double am = 0, em = 0, s2 = 0, r2 = 0;
        for (int d = 0; d < 256; ++d) { const double e = (double) x[d] - ref[d]; if (std::isnan(e)) ++nan; am = std::max(am, std::fabs(ref[d])); em = std::max(em, std::fabs(e)); s2 += e * e; r2 += ref[d] * ref[d]; }
        se += s2; sr += r2; maxrel.push_back((float) (am > 0 ? em / am : (em > 0 ? INFINITY : 0)));
        rl2.push_back((float) (r2 > 0 ? std::sqrt(s2 / r2) : (s2 > 0 ? INFINITY : 0)));
    }
    void merge(const errstat & o) { se += o.se; sr += o.sr; nan += o.nan; maxrel.insert(maxrel.end(), o.maxrel.begin(), o.maxrel.end()); rl2.insert(rl2.end(), o.rl2.begin(), o.rl2.end()); }
    double nmse() const { return sr > 0 ? se / sr : 0; }
};
static double pctl(std::vector<float> v, double p) {   // nearest-rank percentile
    if (v.empty()) return 0; std::sort(v.begin(), v.end());
    size_t k = (size_t) std::ceil(p / 100.0 * v.size()); if (k < 1) k = 1; if (k > v.size()) k = v.size(); return v[k - 1];
}
static std::string fmt_stat(const errstat & s) {
    char b[512];
    snprintf(b, sizeof(b), "rows %zu nmse %.4e maxrel_p50 %.4e maxrel_p99.9 %.4e maxrel_p100 %.4e rl2_p50 %.4e rl2_p99.9 %.4e rl2_p100 %.4e nan %zu",
             s.maxrel.size(), s.nmse(), pctl(s.maxrel, 50), pctl(s.maxrel, 99.9), pctl(s.maxrel, 100), pctl(s.rl2, 50), pctl(s.rl2, 99.9), pctl(s.rl2, 100), s.nan);
    return b;
}

} // namespace bands
