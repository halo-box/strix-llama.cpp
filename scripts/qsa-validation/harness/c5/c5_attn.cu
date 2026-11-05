// C5 (second half): time attend schedules on replayed QSA rows. hipcc --offload-arch=gfx1151 -O3
// usage: c5_attn rows_file f16|q8_0 reps qkv_file|SYN [layer_list|all] [variant_list|all]
// V0 = the PR's qsa3_attn_kernel<true,8,false,Q8> (verbatim, pr_attn.inc) on the plan built by the PR's own rows+merge
// kernels (verbatim, pr_union_kernels.inc); the in-situ route at ub4096 for n_kv < 64*n_q (PR qsa-prefill.cu:684-686).
// V1..V8 = c5_attn_kernel (c5_variant.inc) on CPU-built plans; see the variant table in main.
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

#define RDNA3
#define NO_DEVICE_CODE
#define QK8_0 32
typedef struct { uint16_t d; int8_t qs[QK8_0]; } block_q8_0;
static __device__ __forceinline__ uint16_t ggml_cuda_f32_to_act16(const float f, const bool) { return __builtin_bit_cast(uint16_t, (_Float16) f); }
#define QSA3_L2E 1.4426950408889634f
#define QSA3_G 4
#include "pr_union_kernels.inc"
#include "pr_attn.inc"
#include "c5_variant.inc"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

struct hdr { char name[64]; int32_t n_q, ns, nk, has_mask, pos0, kt; };
struct qhdr { char name[64]; int32_t which, type; int64_t ne[4], nb[4]; uint64_t nbytes; };

// CPU plan: groups of G consecutive queries; per entry a quad id and a (4*G)-bit mask (bit 4*j + cell); ASC or BUCKET order
template <typename MT>
static void build_plan(const std::vector<int32_t> & rows, int n_q, int ns, int nk, int G, bool bucket, int cap,
                       std::vector<uint32_t> & ublk, std::vector<MT> & umask, std::vector<int> & ucount, double & mean_entries) {
    const int ng = (n_q + G - 1) / G;
    ublk.assign((size_t) ng * cap, QSA3_BLK_SENT); umask.assign((size_t) ng * cap, 0); ucount.assign(ng, 0);
    std::vector<std::pair<uint32_t, uint32_t>> u;
    double tot = 0;
    for (int g = 0; g < ng; ++g) {
        u.clear();
        for (int j = 0; j < G && G * g + j < n_q; ++j) {
            const int32_t * r = rows.data() + (size_t) (G * g + j) * ns;
            for (int i = 0; i < ns; ++i) { const int c = r[i]; if (c >= 0 && c < nk) u.push_back({(uint32_t) c >> 2, 1u << (4 * j + (c & 3))}); }
        }
        std::sort(u.begin(), u.end());
        size_t w = 0;
        for (size_t i = 0; i < u.size(); ++i) { if (w && u[w - 1].first == u[i].first) u[w - 1].second |= u[i].second; else u[w++] = u[i]; }
        u.resize(w);
        size_t o = 0;
        auto put = [&](uint32_t b, uint32_t m) { if (o >= (size_t) cap) { fprintf(stderr, "cap overflow\n"); exit(1); } ublk[(size_t) g * cap + o] = b; umask[(size_t) g * cap + o] = (MT) m; ++o; };
        if (!bucket) {
            for (auto & e : u) put(e.first, e.second);
            while (o & 3) put(QSA3_BLK_SENT, 0);
        } else {
            auto pat = [&](uint32_t m) { uint32_t p = 0; for (int j = 0; j < G; ++j) if ((m >> (4 * j)) & 0xF) p |= 1u << j; return p; };
            std::stable_sort(u.begin(), u.end(), [&](const auto & a, const auto & b) { return pat(a.second) < pat(b.second); });
            for (size_t i = 0; i < u.size(); ) {
                const uint32_t p = pat(u[i].second);
                while (i < u.size() && pat(u[i].second) == p) { put(u[i].first, u[i].second); ++i; }
                while (o & 3) put(QSA3_BLK_SENT, 0);
            }
        }
        ucount[g] = (int) o; tot += (double) o;
    }
    mean_entries = tot / ng;
}

static float h2f(uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }
static uint16_t f2h(float f) { return __builtin_bit_cast(uint16_t, (_Float16) f); }

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: rows f16|q8_0 reps qkv|SYN [layers|all] [variants|all]\n"); return 1; }
    const bool q8 = std::string(argv[2]) == "q8_0";
    const int reps = atoi(argv[3]);
    const std::string qkvf = argv[4];
    std::set<int> layers, vars;
    auto parse = [](const char * s, std::set<int> & out) { if (std::string(s) == "all") return; std::string x; for (const char * p = s;; ++p) { if (*p == ',' || !*p) { out.insert(atoi(x.c_str())); x.clear(); if (!*p) break; } else x += *p; } };
    if (argc > 5) parse(argv[5], layers);
    if (argc > 6) parse(argv[6], vars);
    const int NV = 20;
    const char * vname[NV] = {"V0 PR-G4-ASC", "V1 c5-G4-ASC", "V2 G4-ASC-skip", "V3 PR-kern-BUCKET", "V4 G4-BUCKET-r12", "V5 G4-BUCKET-r16",
                              "V6 G8-ASC", "V7 G8-ASC-skip", "V8 G4-ASC-r16skip", "V9 G4-ASC-spec", "V10 G4-BUCKET-spec", "V11 G4-BUCKET-r16spec", "V12 G8-ASC-spec", "V13 ORACLE-G4-2tiles", "V14 ORACLE-G4-1tile", "V15 G4-BUCKET-2body", "V16 G4-BUCKET-r16-2body", "V17 G8-ASC-2body", "V18 G4-BUCKET-2loop", "V19 G4-BUCKET-r16-2loop"};
    FILE * fr = fopen(argv[1], "rb"); if (!fr) { perror("rows"); return 1; }
    FILE * fq = qkvf == "SYN" ? nullptr : fopen(qkvf.c_str(), "rb");
    if (qkvf != "SYN" && !fq) { perror("qkv"); return 1; }
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    std::vector<double> sum_med(NV, 0); int n_layers = 0;
    std::mt19937 rng(12345);
    std::vector<float> synQ; std::vector<uint8_t> synK, synV; int syn_nk = -1;
    hdr h; int L = -1;
    printf("# rows %s type %s reps %d data %s\n", argv[1], argv[2], reps, qkvf.c_str());
    while (fread(&h, sizeof(h), 1, fr) == 1) {
        ++L;
        std::vector<int32_t> rows((size_t) h.n_q * h.ns);
        if (fread(rows.data(), 4, rows.size(), fr) != rows.size()) { fprintf(stderr, "short rows\n"); return 1; }
        std::vector<uint8_t> Qb, Kb, Vb; size_t knb1 = 0, knb2 = 0;
        if (fq) {
            for (int i = 0; i < 3; ++i) {
                qhdr qh; if (fread(&qh, sizeof(qh), 1, fq) != 1) { fprintf(stderr, "short qkv\n"); return 1; }
                std::vector<uint8_t> & B = i == 0 ? Qb : i == 1 ? Kb : Vb;
                B.resize(qh.nbytes); if (fread(B.data(), 1, qh.nbytes, fq) != qh.nbytes) { fprintf(stderr, "short qkv data\n"); return 1; }
                if (i == 0 && !(qh.type == 0 && qh.nb[1] == 24 * 1024 && qh.nb[2] == 1024 && qh.ne[1] == h.n_q && qh.ne[2] == 24)) { fprintf(stderr, "Q layout\n"); return 1; }
                if (i == 1) { knb1 = qh.nb[1]; knb2 = qh.nb[2]; if ((qh.type == 8) != q8 || qh.ne[1] != h.nk) { fprintf(stderr, "K type/nk mismatch\n"); return 1; } }
            }
        }
        if (!layers.empty() && !layers.count(L)) continue;
        const int n_q = h.n_q, ns = h.ns, nk = h.nk;
        const size_t rowb = q8 ? 2 * 8 * sizeof(block_q8_0) : 2 * 256 * 2;
        if (!fq) {
            if (synQ.empty()) { std::normal_distribution<float> nd(0.f, 1.f); synQ.resize((size_t) n_q * 24 * 256); for (auto & x : synQ) x = nd(rng); }
            if (syn_nk != nk) {
                std::normal_distribution<float> nd(0.f, 1.f);
                synK.assign((size_t) nk * rowb, 0); synV.assign((size_t) nk * rowb, 0);
                for (int t = 0; t < 2; ++t) {
                    std::vector<uint8_t> & B = t == 0 ? synK : synV;
                    for (size_t key = 0; key < (size_t) nk; ++key) {
                        if (!q8) { uint16_t * p = (uint16_t *) (B.data() + key * rowb); for (int d = 0; d < 512; ++d) p[d] = f2h(nd(rng)); }
                        else {
                            block_q8_0 * bq = (block_q8_0 *) (B.data() + key * rowb);
                            for (int b = 0; b < 16; ++b) { float x[32], am = 0; for (int i = 0; i < 32; ++i) { x[i] = nd(rng); am = std::max(am, std::fabs(x[i])); }
                                const float d = am / 127.f; bq[b].d = f2h(d); for (int i = 0; i < 32; ++i) bq[b].qs[i] = (int8_t) std::lround(x[i] / d); }
                        }
                    }
                }
                syn_nk = nk;
            }
            Qb.assign((const uint8_t *) synQ.data(), (const uint8_t *) (synQ.data() + synQ.size()));
            Kb = synK; Vb = synV; knb1 = rowb; knb2 = rowb / 2;
        }
        if (Kb.size() != (size_t) nk * rowb || knb1 != rowb || knb2 != rowb / 2) { fprintf(stderr, "K bytes %zu nb %zu %zu\n", Kb.size(), knb1, knb2); return 1; }
        // plans
        const int cap4 = (4 * ns + 3) & ~3, cap4b = cap4 + 4 * 15, cap8 = (8 * ns + 3) & ~3;
        std::vector<uint32_t> b4a, b4b, b8a; std::vector<uint16_t> m4a, m4b; std::vector<uint32_t> m8a; std::vector<int> c4a, c4b, c8a;
        double e4a, e4b, e8a;
        build_plan<uint16_t>(rows, n_q, ns, nk, 4, false, cap4, b4a, m4a, c4a, e4a);
        build_plan<uint16_t>(rows, n_q, ns, nk, 4, true, cap4b, b4b, m4b, c4b, e4b);
        build_plan<uint32_t>(rows, n_q, ns, nk, 8, false, cap8, b8a, m8a, c8a, e8a);
        const int ng4 = (n_q + 3) / 4, ng8 = (n_q + 7) / 8;
        // device buffers
        int32_t * d_ids; float * d_q, * d_out, * d_ref; uint8_t * d_k, * d_v; uint16_t * d_pv;
        CK(hipMalloc(&d_ids, rows.size() * 4)); CK(hipMemcpy(d_ids, rows.data(), rows.size() * 4, hipMemcpyHostToDevice));
        CK(hipMalloc(&d_q, Qb.size())); CK(hipMemcpy(d_q, Qb.data(), Qb.size(), hipMemcpyHostToDevice));
        CK(hipMalloc(&d_k, Kb.size())); CK(hipMemcpy(d_k, Kb.data(), Kb.size(), hipMemcpyHostToDevice));
        CK(hipMalloc(&d_v, Vb.size())); CK(hipMemcpy(d_v, Vb.data(), Vb.size(), hipMemcpyHostToDevice));
        const size_t nel_v = (size_t) nk * 2 * 256;
        CK(hipMalloc(&d_pv, nel_v * 2));
        { const uint32_t count = nel_v / 4; if (q8) hipLaunchKernelGGL(qsa_pack_v<true>, dim3((count + 255) / 256), dim3(256), 0, 0, (const char *) d_v, d_pv, knb1, knb2, nk, count);
          else hipLaunchKernelGGL(qsa_pack_v<false>, dim3((count + 255) / 256), dim3(256), 0, 0, (const char *) d_v, d_pv, knb1, knb2, nk, count); CK(hipGetLastError()); }
        const size_t outb = (size_t) n_q * 24 * 256 * 4;
        CK(hipMalloc(&d_out, outb)); CK(hipMalloc(&d_ref, outb));
        // PR union (verbatim kernels, PR launch parameters qsa-prefill.cu:701-718)
        uint32_t * d_ublk_pr; uint16_t * d_umask_pr; int * d_ucount_pr, * d_srow, * d_sflag;
        CK(hipMalloc(&d_ublk_pr, (size_t) ng4 * cap4 * 4)); CK(hipMalloc(&d_umask_pr, (size_t) ng4 * cap4 * 2)); CK(hipMalloc(&d_ucount_pr, ng4 * 4));
        CK(hipMalloc(&d_srow, (size_t) n_q * ns * 4)); CK(hipMalloc(&d_sflag, n_q * 4)); CK(hipMemset(d_sflag, 0, n_q * 4));
        {
            int np2 = 1; while (np2 < ns) np2 <<= 1; if ((size_t) np2 * 4 > 60 * 1024) np2 = 0;
            hipLaunchKernelGGL(qsa3_rows_kernel, dim3(n_q), dim3(256), (size_t) std::max(np2 ? np2 : ns, 2 * ns) * 4, 0, (const int *) d_ids, (size_t) ns * 4, ns, nk, d_srow, d_sflag, (const char *) nullptr, (size_t) 0, np2);
            hipLaunchKernelGGL(qsa3_merge_kernel, dim3(ng4), dim3(QSA3_MERGE_LANES), (size_t) QSA3_G * ns * 4, 0, (const int *) d_ids, (size_t) ns * 4, n_q, ns, nk, (const int *) d_srow, (const int *) d_sflag, d_ublk_pr, d_umask_pr, d_ucount_pr, cap4);
            CK(hipGetLastError()); CK(hipDeviceSynchronize());
            std::vector<uint32_t> pb((size_t) ng4 * cap4); std::vector<uint16_t> pm((size_t) ng4 * cap4); std::vector<int> pc(ng4);
            CK(hipMemcpy(pb.data(), d_ublk_pr, pb.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(pm.data(), d_umask_pr, pm.size() * 2, hipMemcpyDeviceToHost));
            CK(hipMemcpy(pc.data(), d_ucount_pr, ng4 * 4, hipMemcpyDeviceToHost));
            long bad = 0;
            for (int g = 0; g < ng4; ++g) { if (pc[g] != c4a[g]) { ++bad; continue; } for (int e = 0; e < pc[g]; ++e) { const size_t i = (size_t) g * cap4 + e; if (pb[i] != b4a[i] || pm[i] != m4a[i]) { ++bad; break; } } }
            printf("L%d %s nk %d n_q %d ns %d | PR merge vs CPU G4-ASC plan: %ld/%d groups differ | entries/group: G4-ASC %.1f G4-BUCKET %.1f G8-ASC %.1f\n",
                   L, h.name, nk, n_q, ns, bad, ng4, e4a, e4b, e8a);
        }
        auto up = [](const void * p, size_t n) { void * d; CK(hipMalloc(&d, n)); CK(hipMemcpy(d, p, n, hipMemcpyHostToDevice)); return d; };
        uint32_t * d_b4a = (uint32_t *) up(b4a.data(), b4a.size() * 4); uint16_t * d_m4a = (uint16_t *) up(m4a.data(), m4a.size() * 2); int * d_c4a = (int *) up(c4a.data(), c4a.size() * 4);
        uint32_t * d_b4b = (uint32_t *) up(b4b.data(), b4b.size() * 4); uint16_t * d_m4b = (uint16_t *) up(m4b.data(), m4b.size() * 2); int * d_c4b = (int *) up(c4b.data(), c4b.size() * 4);
        std::vector<uint16_t> m8s((size_t) ng8 * 2 * cap8);
        for (int g = 0; g < ng8; ++g) for (int hh = 0; hh < 2; ++hh) for (int e = 0; e < cap8; ++e) m8s[((size_t) g * 2 + hh) * cap8 + e] = (uint16_t) (m8a[(size_t) g * cap8 + e] >> (16 * hh));
        uint32_t * d_b8a = (uint32_t *) up(b8a.data(), b8a.size() * 4); uint16_t * d_m8a = (uint16_t *) up(m8s.data(), m8s.size() * 2); int * d_c8a = (int *) up(c8a.data(), c8a.size() * 4);
        auto lay = [&](int cap) { return qsa3_layout{(size_t) 24 * 1024, (size_t) 1024, (size_t) 1024, (size_t) 24 * 1024, 0, knb1, (size_t) 0, knb2, (size_t) 0, nk, n_q, 12, cap, 1.0f / 16.0f, 24, 0}; };
        const uint16_t * dk = (const uint16_t *) d_k;
        auto launch = [&](int v, float * dst) {
            const dim3 g4(ng4, 2), g8(ng8, 2);
            switch (v) {
            case 0: hipLaunchKernelGGL((q8 ? qsa3_attn_kernel<true, 8, false, true> : qsa3_attn_kernel<true, 8, false, false>), g4, dim3(256), 0, 0, d_q, dk, d_pv, nullptr, d_ublk_pr, d_umask_pr, d_ucount_pr, dst, lay(cap4), nullptr, nullptr, false); break;
            case 1: hipLaunchKernelGGL((q8 ? c5_attn_kernel<1, 12, 0, true, uint16_t> : c5_attn_kernel<1, 12, 0, false, uint16_t>), g4, dim3(256), 0, 0, d_q, dk, d_pv, d_b4a, d_m4a, d_c4a, dst, lay(cap4)); break;
            case 2: hipLaunchKernelGGL((q8 ? c5_attn_kernel<1, 12, 1, true, uint16_t> : c5_attn_kernel<1, 12, 1, false, uint16_t>), g4, dim3(256), 0, 0, d_q, dk, d_pv, d_b4a, d_m4a, d_c4a, dst, lay(cap4)); break;
            case 3: hipLaunchKernelGGL((q8 ? qsa3_attn_kernel<true, 8, false, true> : qsa3_attn_kernel<true, 8, false, false>), g4, dim3(256), 0, 0, d_q, dk, d_pv, nullptr, d_b4b, d_m4b, d_c4b, dst, lay(cap4b), nullptr, nullptr, false); break;
            case 4: hipLaunchKernelGGL((q8 ? c5_attn_kernel<1, 12, 1, true, uint16_t> : c5_attn_kernel<1, 12, 1, false, uint16_t>), g4, dim3(256), 0, 0, d_q, dk, d_pv, d_b4b, d_m4b, d_c4b, dst, lay(cap4b)); break;
            case 5: hipLaunchKernelGGL((q8 ? c5_attn_kernel<1, 16, 1, true, uint16_t> : c5_attn_kernel<1, 16, 1, false, uint16_t>), g4, dim3(256), 0, 0, d_q, dk, d_pv, d_b4b, d_m4b, d_c4b, dst, lay(cap4b)); break;
            case 6: hipLaunchKernelGGL((q8 ? c5_attn_kernel<2, 12, 0, true, uint16_t> : c5_attn_kernel<2, 12, 0, false, uint16_t>), g8, dim3(512), 0, 0, d_q, dk, d_pv, d_b8a, d_m8a, d_c8a, dst, lay(cap8)); break;
            case 7: hipLaunchKernelGGL((q8 ? c5_attn_kernel<2, 12, 1, true, uint16_t> : c5_attn_kernel<2, 12, 1, false, uint16_t>), g8, dim3(512), 0, 0, d_q, dk, d_pv, d_b8a, d_m8a, d_c8a, dst, lay(cap8)); break;
            case 8: hipLaunchKernelGGL((q8 ? c5_attn_kernel<1, 16, 1, true, uint16_t> : c5_attn_kernel<1, 16, 1, false, uint16_t>), g4, dim3(256), 0, 0, d_q, dk, d_pv, d_b4a, d_m4a, d_c4a, dst, lay(cap4)); break;
            case 9:  hipLaunchKernelGGL((q8 ? c5_attn_kernel<1, 12, 2, true, uint16_t> : c5_attn_kernel<1, 12, 2, false, uint16_t>), g4, dim3(256), 0, 0, d_q, dk, d_pv, d_b4a, d_m4a, d_c4a, dst, lay(cap4)); break;
            case 10: hipLaunchKernelGGL((q8 ? c5_attn_kernel<1, 12, 2, true, uint16_t> : c5_attn_kernel<1, 12, 2, false, uint16_t>), g4, dim3(256), 0, 0, d_q, dk, d_pv, d_b4b, d_m4b, d_c4b, dst, lay(cap4b)); break;
            case 11: hipLaunchKernelGGL((q8 ? c5_attn_kernel<1, 16, 2, true, uint16_t> : c5_attn_kernel<1, 16, 2, false, uint16_t>), g4, dim3(256), 0, 0, d_q, dk, d_pv, d_b4b, d_m4b, d_c4b, dst, lay(cap4b)); break;
            case 12: hipLaunchKernelGGL((q8 ? c5_attn_kernel<2, 12, 2, true, uint16_t> : c5_attn_kernel<2, 12, 2, false, uint16_t>), g8, dim3(512), 0, 0, d_q, dk, d_pv, d_b8a, d_m8a, d_c8a, dst, lay(cap8)); break;
            case 13: hipLaunchKernelGGL((q8 ? c5_attn_kernel<1, 12, 16 + 3, true, uint16_t> : c5_attn_kernel<1, 12, 16 + 3, false, uint16_t>), g4, dim3(256), 0, 0, d_q, dk, d_pv, d_b4a, d_m4a, d_c4a, dst, lay(cap4)); break;
            case 14: hipLaunchKernelGGL((q8 ? c5_attn_kernel<1, 12, 16 + 1, true, uint16_t> : c5_attn_kernel<1, 12, 16 + 1, false, uint16_t>), g4, dim3(256), 0, 0, d_q, dk, d_pv, d_b4a, d_m4a, d_c4a, dst, lay(cap4)); break;
            case 15: hipLaunchKernelGGL((q8 ? c5_attn_kernel<1, 12, 3, true, uint16_t> : c5_attn_kernel<1, 12, 3, false, uint16_t>), g4, dim3(256), 0, 0, d_q, dk, d_pv, d_b4b, d_m4b, d_c4b, dst, lay(cap4b)); break;
            case 16: hipLaunchKernelGGL((q8 ? c5_attn_kernel<1, 16, 3, true, uint16_t> : c5_attn_kernel<1, 16, 3, false, uint16_t>), g4, dim3(256), 0, 0, d_q, dk, d_pv, d_b4b, d_m4b, d_c4b, dst, lay(cap4b)); break;
            case 17: hipLaunchKernelGGL((q8 ? c5_attn_kernel<2, 12, 3, true, uint16_t> : c5_attn_kernel<2, 12, 3, false, uint16_t>), g8, dim3(512), 0, 0, d_q, dk, d_pv, d_b8a, d_m8a, d_c8a, dst, lay(cap8)); break;
            case 18: hipLaunchKernelGGL((q8 ? c5_attn_kernel<1, 12, 4, true, uint16_t> : c5_attn_kernel<1, 12, 4, false, uint16_t>), g4, dim3(256), 0, 0, d_q, dk, d_pv, d_b4b, d_m4b, d_c4b, dst, lay(cap4b)); break;
            case 19: hipLaunchKernelGGL((q8 ? c5_attn_kernel<1, 16, 4, true, uint16_t> : c5_attn_kernel<1, 16, 4, false, uint16_t>), g4, dim3(256), 0, 0, d_q, dk, d_pv, d_b4b, d_m4b, d_c4b, dst, lay(cap4b)); break;
            }
            CK(hipGetLastError());
        };
        std::vector<int> vl; for (int v = 0; v < NV; ++v) if (vars.empty() || vars.count(v) || v == 0) vl.push_back(v);
        // correctness: every variant vs V0, and V0 vs a CPU reference on sampled queries
        launch(0, d_ref); CK(hipDeviceSynchronize());
        std::vector<float> ref(outb / 4), o(outb / 4);
        CK(hipMemcpy(ref.data(), d_ref, outb, hipMemcpyDeviceToHost));
        {
            const int qs_s[8] = {0, 1, 5, 1023, 2050, 3001, 4094, 4095};
            double maxrel = 0;
            for (int qi = 0; qi < 8; ++qi) {
                const int qq = qs_s[qi] % n_q; const int32_t * r = rows.data() + (size_t) qq * ns;
                std::set<int> cells; for (int i = 0; i < ns; ++i) if (r[i] >= 0 && r[i] < nk) cells.insert(r[i]);
                for (int hq = 0; hq < 24; ++hq) {
                    const int kvh = hq / 12; const float * qv = (const float *) Qb.data() + ((size_t) qq * 24 + hq) * 256;
                    auto kval = [&](const std::vector<uint8_t> & B, int key, int d) -> float {
                        if (!q8) return h2f(((const uint16_t *) (B.data() + (size_t) key * rowb + kvh * rowb / 2))[d]);
                        const block_q8_0 * bq = (const block_q8_0 *) (B.data() + (size_t) key * rowb + kvh * rowb / 2) + d / 32;
                        return (float) (_Float16) ((float) bq->qs[d % 32] * h2f(bq->d));
                    };
                    std::vector<double> sc; double mx = -1e300;
                    for (int c : cells) { double s = 0; for (int d = 0; d < 256; ++d) s += (double) (float) ((_Float16) qv[d] * (_Float16) (1.0f / 16.0f)) * kval(Kb, c, d); sc.push_back(s); mx = std::max(mx, s); }
                    double den = 0; for (double & s : sc) { s = std::exp(s - mx); den += s; }
                    double amax = 0, emax = 0; int ci = 0;
                    std::vector<double> acc(256, 0);
                    for (int c : cells) { for (int d = 0; d < 256; ++d) acc[d] += sc[ci] * kval(Vb, c, d); ++ci; }
                    for (int d = 0; d < 256; ++d) { const double rv = acc[d] / den; amax = std::max(amax, std::fabs(rv)); emax = std::max(emax, std::fabs(rv - ref[((size_t) qq * 24 + hq) * 256 + d])); }
                    maxrel = std::max(maxrel, emax / std::max(amax, 1e-30));
                }
            }
            printf("L%d V0 vs CPU f64 reference (8 queries x 24 heads): max |err|/max|ref| per head = %.3e\n", L, maxrel);
        }
        for (int v : vl) {
            if (v == 0 || v == 13 || v == 14) continue;
            CK(hipMemset(d_out, 0xff, outb)); launch(v, d_out); CK(hipDeviceSynchronize());
            CK(hipMemcpy(o.data(), d_out, outb, hipMemcpyDeviceToHost));
            double md = 0, s2 = 0, r2 = 0; size_t neq = 0, nnan = 0;
            for (size_t i = 0; i < o.size(); ++i) { if (std::isnan(o[i])) { ++nnan; continue; } const double d = std::fabs((double) o[i] - ref[i]); md = std::max(md, d); s2 += d * d; r2 += (double) ref[i] * ref[i]; neq += memcmp(&o[i], &ref[i], 4) != 0; }
            printf("L%d %-18s vs V0: max_abs %.3e rel_l2 %.3e words_differ %zu nan %zu\n", L, vname[v], md, std::sqrt(s2 / std::max(r2, 1e-30)), neq, nnan);
        }
        // timing: 3 warm-up rounds, then reps rounds; each round launches every variant once (interleaved), one event pair each
        std::vector<std::vector<float>> t(NV);
        for (int it = 0; it < 3 + reps; ++it) {
            for (int v : vl) {
                CK(hipEventRecord(e0, 0)); launch(v, d_out); CK(hipEventRecord(e1, 0)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it >= 3) t[v].push_back(ms);
            }
        }
        printf("L%d timing (ms; median mean sd min, n=%d):", L, reps);
        for (int v : vl) {
            std::vector<float> x = t[v]; std::sort(x.begin(), x.end());
            double mean = 0, sd = 0; for (float y : x) mean += y; mean /= x.size(); for (float y : x) sd += (y - mean) * (y - mean); sd = std::sqrt(sd / std::max<size_t>(1, x.size() - 1));
            const double med = x.size() % 2 ? x[x.size() / 2] : 0.5 * (x[x.size() / 2 - 1] + x[x.size() / 2]);
            printf(" | %s %.3f %.3f %.3f %.3f", vname[v], med, mean, sd, x[0]);
            sum_med[v] += med;
        }
        printf("\n"); fflush(stdout);
        ++n_layers;
        hipFree(d_ids); hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_pv); hipFree(d_out); hipFree(d_ref);
        hipFree(d_ublk_pr); hipFree(d_umask_pr); hipFree(d_ucount_pr); hipFree(d_srow); hipFree(d_sflag);
        hipFree(d_b4a); hipFree(d_m4a); hipFree(d_c4a); hipFree(d_b4b); hipFree(d_m4b); hipFree(d_c4b); hipFree(d_b8a); hipFree(d_m8a); hipFree(d_c8a);
    }
    printf("SUMMARY layers=%d mean of per-layer medians (ms) and ratio to V0:", n_layers);
    for (int v = 0; v < NV; ++v) if (sum_med[v] > 0) printf(" | %s %.3f %.4f", vname[v], sum_med[v] / n_layers, sum_med[v] / sum_med[0]);
    printf("\n");
    return 0;
}
