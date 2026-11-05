// C3 (a): plan-stage cost on captured QSA rows. T0 = the PR's in-op union build (qsa3_rows_kernel + qsa3_merge_kernel,
// copied verbatim from PR 94e086847 ggml/src/ggml-cuda/qsa-prefill.cu into pr_union_kernels.inc). P = prototype of the
// standalone plan stage of plans W/Y/Z: validate, sort, dedup (sa_rows) + G=4 k-way merge into quads with 16-bit cell masks,
// ASC or BUCKET order (sa_merge). Every output is checked bit-exact against a CPU reference.
// usage: c3_plan rows_file reps [shuffle_seed]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <string>

#define QSA3_G 4
#include "pr_union_kernels.inc"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define SA_SENT 0x7FFFFFFF
#define SA_BLK_SENT 0xFFFFFFFFu

// ---------------- P: sa_rows (one 256-lane workgroup per query) ----------------
// stage[q][0..ns): sorted unique valid cells, SA_SENT after; cnt[q] = number of cells
template <bool FORCE_SORT>
__global__ __launch_bounds__(256) void sa_rows(const int * __restrict__ ids, const int ld, const int ns, const int nk, const int np2,
                                               int * __restrict__ stage, int * __restrict__ cnt_out) {
    extern __shared__ int ent[];          // np2 entries (row, padded), then ns entries (compacted)
    __shared__ int scan_s[256];
    __shared__ int notstrict;
    const int q = blockIdx.x, tid = threadIdx.x;
    const int * row = ids + (size_t) q * ld;
    int * cmp = ent + np2;
    int * dst = stage + (size_t) q * ns;
    for (int j = tid; j < ns; j += 256) {
        const int key = row[j];
        ent[j] = (key >= 0 && key < nk) ? key : SA_SENT;     // validate
    }
    for (int j = ns + tid; j < np2; j += 256) { ent[j] = SA_SENT; }
    if (tid == 0) { notstrict = 0; }
    __syncthreads();
    const int per = (ns + 255) / 256, j0 = min(ns, tid * per), j1 = min(ns, j0 + per);
    if (!FORCE_SORT) {
        // fast path: compact the valid cells in row order; if strictly ascending they are already sorted and unique
        int c = 0;
        for (int j = j0; j < j1; ++j) { c += ent[j] != SA_SENT; }
        scan_s[tid] = c;
        __syncthreads();
        for (int off = 1; off < 256; off <<= 1) {
            const int v = tid >= off ? scan_s[tid - off] : 0;
            __syncthreads();
            scan_s[tid] += v;
            __syncthreads();
        }
        const int total = scan_s[255];
        int o = scan_s[tid] - c;
        for (int j = j0; j < j1; ++j) { if (ent[j] != SA_SENT) { cmp[o++] = ent[j]; } }
        __syncthreads();
        for (int j = tid; j + 1 < total; j += 256) { if (cmp[j] >= cmp[j+1]) { atomicOr(&notstrict, 1); } }
        __syncthreads();
        if (!notstrict) {
            for (int j = tid; j < ns; j += 256) { dst[j] = j < total ? cmp[j] : SA_SENT; }
            if (tid == 0) { cnt_out[q] = total; }
            return;
        }
        __syncthreads();
    }
    // general path: bitonic sort over np2 (sentinels sort last), then dedup-compaction
    for (int k = 2; k <= np2; k <<= 1) {
        for (int jj = k >> 1; jj > 0; jj >>= 1) {
            for (int i = tid; i < np2; i += 256) {
                const int ixj = i ^ jj;
                if (ixj > i) {
                    const int a = ent[i], b = ent[ixj];
                    const bool asc = (i & k) == 0;
                    if ((a > b) == asc) { ent[i] = b; ent[ixj] = a; }
                }
            }
            __syncthreads();
        }
    }
    int c = 0;
    for (int j = j0; j < j1; ++j) { const int e = ent[j]; c += e != SA_SENT && (j == 0 || ent[j-1] != e); }
    scan_s[tid] = c;
    __syncthreads();
    for (int off = 1; off < 256; off <<= 1) {
        const int v = tid >= off ? scan_s[tid - off] : 0;
        __syncthreads();
        scan_s[tid] += v;
        __syncthreads();
    }
    const int total = scan_s[255];
    int o = scan_s[tid] - c;
    for (int j = j0; j < j1; ++j) { const int e = ent[j]; if (e != SA_SENT && (j == 0 || ent[j-1] != e)) { cmp[o++] = e; } }
    __syncthreads();
    for (int j = tid; j < ns; j += 256) { dst[j] = j < total ? cmp[j] : SA_SENT; }
    if (tid == 0) { cnt_out[q] = total; }
}

// ---------------- P: sa_merge<BUCKET, LANES> (one workgroup per group of 4 queries) ----------------
// Reads the rows as the PR merge does (raw ids with the validity transform when sflag[q] == 0, else the sorted copy
// srow[q] written by qsa3_rows_kernel). Duplicate cells collapse in the 4-bit cell mask, so the output is the deduped
// union. out per group: ublk[cap] quad ids, umask[cap] (bit 4*qi + (cell&3)), ucount = padded entry count.
// ASC: strictly ascending quads, padded to a multiple of 4 at the end (== qsa3_merge_kernel output).
// BUCKET: stable by query-presence pattern (4 bits, ascending), quad ascending inside a bucket, each bucket padded to a
// multiple of 4 (pads: BLK_SENT, mask 0).
template <bool BUCKET, int LANES>
__global__ __launch_bounds__(LANES) void sa_merge(const int * __restrict__ ids, const size_t i1, const int n_q, const int ns, const int nk,
        const int * __restrict__ srow, const int * __restrict__ sflag,
        uint32_t * __restrict__ ublk, uint16_t * __restrict__ umask, int * __restrict__ ucount, const int cap) {
    extern __shared__ int rows_s[];                  // 4 * ns
    __shared__ int scan_s[LANES];
    __shared__ unsigned short pc[16][LANES + 2];     // BUCKET: per-pattern per-lane counts, then exclusive offsets (< cap <= 65535)
    __shared__ int pbase[16], ptot[16];
    const int lane = threadIdx.x, g = blockIdx.x;
#pragma unroll
    for (int qi = 0; qi < 4; ++qi) {
        const int q = 4 * g + qi;
        int * d = rows_s + qi * ns;
        if (q < n_q) {
            const bool rw = sflag[q] == 0;
            const int * s = rw ? reinterpret_cast<const int *>(reinterpret_cast<const char *>(ids) + (size_t) q * i1) : srow + (size_t) q * ns;
            for (int j = lane; j < ns; j += LANES) { d[j] = qsa3_get(s, j, nk, rw); }
        } else {
            for (int j = lane; j < ns; j += LANES) { d[j] = SA_SENT; }
        }
    }
    if (BUCKET) { for (int p = 0; p < 16; ++p) { pc[p][lane] = 0; } }
    __syncthreads();
    const int * rp[4]; int cnt[4];
#pragma unroll
    for (int qi = 0; qi < 4; ++qi) {
        rp[qi] = rows_s + qi * ns;
        int a = 0, b = ns;
        while (a < b) { const int mid = (a + b) >> 1; if (rp[qi][mid] < SA_SENT) { a = mid + 1; } else { b = mid; } }
        cnt[qi] = a;
    }
    int kmax = -1, ref = -1;
#pragma unroll
    for (int qi = 0; qi < 4; ++qi) {
        if (cnt[qi] > 0) { kmax = max(kmax, rp[qi][cnt[qi] - 1]); if (ref < 0) { ref = qi; } }
    }
    uint32_t * ob = ublk + (size_t) g * cap; uint16_t * om = umask + (size_t) g * cap;
    if (kmax < 0) { if (lane == 0) { ucount[g] = 0; } return; }
    const int rc = cnt[ref];
    const int lo = lane == 0 ? 0 : (rp[ref][(int) (((long long) rc * lane) / LANES)] & ~3);
    const int hi = lane == LANES - 1 ? ((kmax + 4) & ~3) : (rp[ref][(int) (((long long) rc * (lane + 1)) / LANES)] & ~3);
    int p0[4], pe[4];
#pragma unroll
    for (int qi = 0; qi < 4; ++qi) {
        int a = 0, b = cnt[qi];
        while (a < b) { const int mid = (a + b) >> 1; if (rp[qi][mid] < lo) { a = mid + 1; } else { b = mid; } }
        p0[qi] = a; b = cnt[qi];
        while (a < b) { const int mid = (a + b) >> 1; if (rp[qi][mid] < hi) { a = mid + 1; } else { b = mid; } }
        pe[qi] = a;
    }
    // pass 1: count (per pattern if BUCKET)
    int n = 0;
    {
        int pp[4], head[4];
#pragma unroll
        for (int qi = 0; qi < 4; ++qi) { pp[qi] = p0[qi]; head[qi] = pp[qi] < pe[qi] ? rp[qi][pp[qi]] : SA_SENT; }
        while (true) {
            int b = SA_SENT;
#pragma unroll
            for (int qi = 0; qi < 4; ++qi) { b = min(b, head[qi]); }
            if (b == SA_SENT) { break; }
            b >>= 2;
            int pat = 0;
#pragma unroll
            for (int qi = 0; qi < 4; ++qi) {
                if (head[qi] != SA_SENT && (head[qi] >> 2) == b) {
                    pat |= 1 << qi;
                    const int * rr = rp[qi];
                    int pn = pp[qi];
                    if ((head[qi] & 3) == 0 && pn + 3 < pe[qi] && rr[pn+3] == head[qi] + 3 && rr[pn+1] == head[qi] + 1 && rr[pn+2] == head[qi] + 2) { pn += 4; }
                    else { do { ++pn; } while (pn < pe[qi] && (rr[pn] >> 2) == b); }
                    pp[qi] = pn; head[qi] = pn < pe[qi] ? rr[pn] : SA_SENT;
                }
            }
            if (BUCKET) { pc[pat][lane]++; }
            ++n;
        }
    }
    int o = 0;
    if (!BUCKET) {
        scan_s[lane] = n;
        __syncthreads();
        for (int off = 1; off < LANES; off <<= 1) {
            const int v = lane >= off ? scan_s[lane - off] : 0;
            __syncthreads();
            scan_s[lane] += v;
            __syncthreads();
        }
        o = scan_s[lane] - n;
        if (lane == LANES - 1) {
            int t = scan_s[LANES - 1];
            while (t & 3) { ob[t] = SA_BLK_SENT; om[t] = 0; ++t; }
            ucount[g] = t;
        }
    } else {
        __syncthreads();
        {   // exclusive scan over lanes, per pattern: LANES/16 threads per pattern, 16 lanes each
            constexpr int P = LANES / 16;
            const int p = lane / P, s = lane % P;
            int part = 0;
            for (int i = 0; i < 16; ++i) { part += pc[p][s * 16 + i]; }
            scan_s[lane] = part;
            __syncthreads();
            int base = 0;
            for (int k = 0; k < s; ++k) { base += scan_s[p * P + k]; }
            for (int i = 0; i < 16; ++i) { const int v = pc[p][s * 16 + i]; pc[p][s * 16 + i] = base; base += v; }
            if (s == P - 1) { ptot[p] = base; }
        }
        __syncthreads();
        if (lane == 0) {
            int b = 0;
            for (int p = 0; p < 16; ++p) { pbase[p] = b; b += (ptot[p] + 3) & ~3; }
            ucount[g] = b;
        }
        __syncthreads();
        if (lane < 16) {
            for (int t = pbase[lane] + ptot[lane]; t < pbase[lane] + ((ptot[lane] + 3) & ~3); ++t) { ob[t] = SA_BLK_SENT; om[t] = 0; }
        }
    }
    // pass 2: write
    {
        int pp[4], head[4];
#pragma unroll
        for (int qi = 0; qi < 4; ++qi) { pp[qi] = p0[qi]; head[qi] = pp[qi] < pe[qi] ? rp[qi][pp[qi]] : SA_SENT; }
        while (true) {
            int b = SA_SENT;
#pragma unroll
            for (int qi = 0; qi < 4; ++qi) { b = min(b, head[qi]); }
            if (b == SA_SENT) { break; }
            b >>= 2;
            uint32_t mk = 0; int pat = 0;
#pragma unroll
            for (int qi = 0; qi < 4; ++qi) {
                if (head[qi] != SA_SENT && (head[qi] >> 2) == b) {
                    pat |= 1 << qi;
                    const int * rr = rp[qi];
                    int pn = pp[qi];
                    if ((head[qi] & 3) == 0 && pn + 3 < pe[qi] && rr[pn+3] == head[qi] + 3 && rr[pn+1] == head[qi] + 1 && rr[pn+2] == head[qi] + 2) { mk |= 0xFu << (4*qi); pn += 4; }
                    else { do { mk |= 1u << (4 * qi + (rr[pn] & 3)); ++pn; } while (pn < pe[qi] && (rr[pn] >> 2) == b); }
                    pp[qi] = pn; head[qi] = pn < pe[qi] ? rr[pn] : SA_SENT;
                }
            }
            int dsti;
            if (BUCKET) { dsti = pbase[pat] + pc[pat][lane]++; } else { dsti = o++; }
            ob[dsti] = (uint32_t) b; om[dsti] = (uint16_t) mk;
        }
    }
}

// ---------------- CPU reference ----------------
struct plan_ref { std::vector<int> count; std::vector<std::vector<std::pair<uint32_t, uint16_t>>> ent; };
static plan_ref cpu_plan(const std::vector<int> & rows, int n_q, int ns, int nk, bool bucket) {
    const int ng = (n_q + 3) / 4;
    plan_ref r; r.count.resize(ng); r.ent.resize(ng);
    for (int g = 0; g < ng; ++g) {
        std::map<uint32_t, uint16_t> m;
        for (int qi = 0; qi < 4; ++qi) {
            const int q = 4 * g + qi; if (q >= n_q) break;
            for (int j = 0; j < ns; ++j) { const int k = rows[(size_t) q * ns + j]; if (k >= 0 && k < nk) m[(uint32_t) k >> 2] |= (uint16_t) (1u << (4 * qi + (k & 3))); }
        }
        auto & e = r.ent[g];
        if (!bucket) {
            for (auto & kv : m) e.push_back(kv);
            while (e.size() & 3) e.push_back({SA_BLK_SENT, 0});
        } else {
            for (int p = 1; p < 16; ++p) {
                size_t c = 0;
                for (auto & kv : m) {
                    int pat = 0; for (int qi = 0; qi < 4; ++qi) if ((kv.second >> (4 * qi)) & 0xF) pat |= 1 << qi;
                    if (pat == p) { e.push_back(kv); ++c; }
                }
                while (c & 3) { e.push_back({SA_BLK_SENT, 0}); ++c; }
            }
        }
        r.count[g] = (int) e.size();
    }
    return r;
}

static int verify(const plan_ref & ref, const std::vector<uint32_t> & b, const std::vector<uint16_t> & m, const std::vector<int> & c, int cap) {
    int bad = 0;
    for (size_t g = 0; g < ref.count.size(); ++g) {
        if (c[g] != ref.count[g]) { ++bad; continue; }
        for (int i = 0; i < c[g]; ++i) {
            if (b[g * cap + i] != ref.ent[g][i].first || m[g * cap + i] != ref.ent[g][i].second) { ++bad; break; }
        }
    }
    return bad;
}

struct rec { char name[64]; int32_t n_q, ns, nk, has_mask, pos0, kt; };

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: c3_plan rows_file reps [shuffle_seed]\n"); return 1; }
    const int reps = atoi(argv[2]);
    const int shuffle_seed = argc > 3 ? atoi(argv[3]) : 0;
    FILE * f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 1; }
    hipEvent_t e0, e1, e2;
    CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1)); CK(hipEventCreate(&e2));
    int layer = 0;
    rec h;
    while (fread(&h, sizeof(h), 1, f) == 1) {
        const int n_q = h.n_q, ns = h.ns, nk = h.nk;
        std::vector<int> rows((size_t) n_q * ns);
        if (fread(rows.data(), 4, rows.size(), f) != rows.size()) { fprintf(stderr, "short read\n"); return 1; }
        if (shuffle_seed) {
            std::mt19937 rng(shuffle_seed + layer);
            for (int q = 0; q < n_q; ++q) std::shuffle(rows.begin() + (size_t) q * ns, rows.begin() + (size_t) (q + 1) * ns, rng);
        }
        // row statistics
        long long nvalid = 0; int nsorted = 0;
        for (int q = 0; q < n_q; ++q) {
            int prev = -1; bool asc = true;
            for (int j = 0; j < ns; ++j) { const int k = rows[(size_t) q * ns + j]; if (k >= 0 && k < nk) { ++nvalid; if (k <= prev) asc = false; prev = k; } }
            nsorted += asc;
        }
        const int ng = (n_q + 3) / 4;
        const int cap = (4 * ns + 3) & ~3;   // as the PR (qsa-prefill.cu:702); also >= the BUCKET bound (union <= 4*ns cells => <= ns... see log)
        int np2 = 1; while (np2 < ns) np2 <<= 1;
        int *d_ids, *d_srow, *d_sflag, *d_stage, *d_cnt, *d_ucount;
        uint32_t * d_ublk; uint16_t * d_umask;
        CK(hipMalloc(&d_ids, rows.size() * 4)); CK(hipMemcpy(d_ids, rows.data(), rows.size() * 4, hipMemcpyHostToDevice));
        CK(hipMalloc(&d_srow, rows.size() * 4)); CK(hipMalloc(&d_sflag, n_q * 4));
        CK(hipMalloc(&d_stage, rows.size() * 4)); CK(hipMalloc(&d_cnt, n_q * 4));
        int * d_ones; { std::vector<int> ones(n_q, 1); CK(hipMalloc(&d_ones, n_q * 4)); CK(hipMemcpy(d_ones, ones.data(), n_q * 4, hipMemcpyHostToDevice)); }
        CK(hipMalloc(&d_ublk, (size_t) ng * (cap + 64) * 4)); CK(hipMalloc(&d_umask, (size_t) ng * (cap + 64) * 2)); CK(hipMalloc(&d_ucount, ng * 4));
        const plan_ref ref_asc = cpu_plan(rows, n_q, ns, nk, false);
        const plan_ref ref_bkt = cpu_plan(rows, n_q, ns, nk, true);
        double mean_asc = 0, mean_bkt = 0;
        for (int g = 0; g < ng; ++g) { mean_asc += ref_asc.count[g]; mean_bkt += ref_bkt.count[g]; }
        int maxb = 0; for (int g = 0; g < ng; ++g) maxb = std::max(maxb, ref_bkt.count[g]);
        printf("LAYER %d name=%s pos0=%d nq=%d ns=%d nk=%d valid_per_q=%.1f sorted_rows=%d/%d union_asc=%.1f union_bkt=%.1f max_bkt=%d cap=%d shuffle=%d\n",
               layer, h.name, h.pos0, n_q, ns, nk, (double) nvalid / n_q, nsorted, n_q, mean_asc / ng, mean_bkt / ng, maxb, cap, shuffle_seed);
        if (maxb > cap + 64) { fprintf(stderr, "cap overflow\n"); return 1; }
        const size_t sh_rows_pr = (size_t) std::max(np2, 2 * ns) * 4, sh_merge = (size_t) 4 * ns * 4, sh_rows_p = (size_t) (np2 + ns) * 4;
        // variants: 0 T0 (PR rows+merge), 1 P_asc (PR rows + sa_merge ASC, 128 lanes), 2 P_bucket (128 lanes),
        // 3 P_asc256 / 4 P_bucket256 (256 lanes), 5 P_staged_bucket (sa_rows: always writes the sorted, deduped staging rows; then BUCKET merge)
        const char * vname[8] = { "T0_pr_rows+merge", "P_asc", "P_bucket", "P_asc_l256", "P_bucket_l256", "P_staged_bucket", "P_asc_l512", "P_bucket_l512" };
        for (int v = 0; v < 8; ++v) {
            auto launch = [&]() {
                CK(hipEventRecord(e0, 0));
                if (v == 5) {
                    hipLaunchKernelGGL(sa_rows<false>, dim3(n_q), dim3(256), sh_rows_p, 0, d_ids, ns, ns, nk, np2, d_stage, d_cnt);
                    CK(hipEventRecord(e1, 0));
                    // staged rows are already validated: pass them as the "sorted copy" (sflag = 1 for every query)
                    hipLaunchKernelGGL((sa_merge<true, 128>), dim3(ng), dim3(128), sh_merge, 0, d_ids, (size_t) ns * 4, n_q, ns, nk, d_stage, d_ones, d_ublk, d_umask, d_ucount, cap);
                } else {
                    hipLaunchKernelGGL(qsa3_rows_kernel, dim3(n_q), dim3(256), sh_rows_pr, 0, d_ids, (size_t) ns * 4, ns, nk, d_srow, d_sflag, (const char *) nullptr, (size_t) 0, np2);
                    CK(hipEventRecord(e1, 0));
                    if (v == 0) hipLaunchKernelGGL(qsa3_merge_kernel, dim3(ng), dim3(QSA3_MERGE_LANES), sh_merge, 0, d_ids, (size_t) ns * 4, n_q, ns, nk, d_srow, d_sflag, d_ublk, d_umask, d_ucount, cap);
                    if (v == 1) hipLaunchKernelGGL((sa_merge<false, 128>), dim3(ng), dim3(128), sh_merge, 0, d_ids, (size_t) ns * 4, n_q, ns, nk, d_srow, d_sflag, d_ublk, d_umask, d_ucount, cap);
                    if (v == 2) hipLaunchKernelGGL((sa_merge<true, 128>),  dim3(ng), dim3(128), sh_merge, 0, d_ids, (size_t) ns * 4, n_q, ns, nk, d_srow, d_sflag, d_ublk, d_umask, d_ucount, cap);
                    if (v == 3) hipLaunchKernelGGL((sa_merge<false, 256>), dim3(ng), dim3(256), sh_merge, 0, d_ids, (size_t) ns * 4, n_q, ns, nk, d_srow, d_sflag, d_ublk, d_umask, d_ucount, cap);
                    if (v == 4) hipLaunchKernelGGL((sa_merge<true, 256>),  dim3(ng), dim3(256), sh_merge, 0, d_ids, (size_t) ns * 4, n_q, ns, nk, d_srow, d_sflag, d_ublk, d_umask, d_ucount, cap);
                    if (v == 6) hipLaunchKernelGGL((sa_merge<false, 512>), dim3(ng), dim3(512), sh_merge, 0, d_ids, (size_t) ns * 4, n_q, ns, nk, d_srow, d_sflag, d_ublk, d_umask, d_ucount, cap);
                    if (v == 7) hipLaunchKernelGGL((sa_merge<true, 512>),  dim3(ng), dim3(512), sh_merge, 0, d_ids, (size_t) ns * 4, n_q, ns, nk, d_srow, d_sflag, d_ublk, d_umask, d_ucount, cap);
                }
                CK(hipGetLastError());
                CK(hipEventRecord(e2, 0));
                CK(hipEventSynchronize(e2));
            };
            for (int w = 0; w < 3; ++w) launch();
            std::vector<float> tr, tm, tt;
            for (int r = 0; r < reps; ++r) {
                CK(hipMemset(d_ucount, 0xff, ng * 4));
                launch();
                float a, b; CK(hipEventElapsedTime(&a, e0, e1)); CK(hipEventElapsedTime(&b, e1, e2));
                tr.push_back(a); tm.push_back(b); tt.push_back(a + b);
            }
            std::vector<uint32_t> hb((size_t) ng * cap); std::vector<uint16_t> hm((size_t) ng * cap); std::vector<int> hc(ng);
            CK(hipMemcpy(hc.data(), d_ucount, ng * 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy2D(hb.data(), cap * 4, d_ublk, cap * 4, cap * 4, ng, hipMemcpyDeviceToHost));
            CK(hipMemcpy2D(hm.data(), cap * 2, d_umask, cap * 2, cap * 2, ng, hipMemcpyDeviceToHost));
            const bool bkt = v == 2 || v == 4 || v == 5 || v == 7;
            const int bad = verify(bkt ? ref_bkt : ref_asc, hb, hm, hc, cap);
            auto med = [](std::vector<float> x) { std::sort(x.begin(), x.end()); return x[x.size() / 2]; };
            auto mean = [](const std::vector<float> & x) { double s = 0; for (float y : x) s += y; return s / x.size(); };
            auto sd = [&](const std::vector<float> & x) { const double m = mean(x); double s = 0; for (float y : x) s += (y - m) * (y - m); return sqrt(s / (x.size() - 1)); };
            const float mn = *std::min_element(tt.begin(), tt.end()), mx = *std::max_element(tt.begin(), tt.end());
            printf("TIME layer=%d variant=%s n=%d total_med=%.4f total_mean=%.4f total_sd=%.4f min=%.4f max=%.4f rows_med=%.4f merge_med=%.4f mismatch_groups=%d\n",
                   layer, vname[v], reps, med(tt), mean(tt), sd(tt), mn, mx, med(tr), med(tm), bad);
            fflush(stdout);
        }
        CK(hipFree(d_ids)); CK(hipFree(d_srow)); CK(hipFree(d_sflag)); CK(hipFree(d_stage)); CK(hipFree(d_cnt));
        CK(hipFree(d_ublk)); CK(hipFree(d_umask)); CK(hipFree(d_ucount)); CK(hipFree(d_ones));
        ++layer;
    }
    return 0;
}
