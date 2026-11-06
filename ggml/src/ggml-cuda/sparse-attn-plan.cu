// GGML_OP_SPARSE_ATTN_PLAN on HIP (PLAN-v3 §2.1, §6.2): canonicalise per-query selection rows into the grouped plan
// whose whole dst is byte-identical to the CPU op (ggml-cpu/sparse-attn.cpp) and to the C3 oracle (tests/sparse-attn-ref.h).
//
// Three kernels on the op's stream, no pool allocation (D12): the sort/dedup scratch lives in the get_alloc_size
// over-allocation of dst, right after ggml_nbytes(dst).
//   sa_plan_init       header (n_bad = 0), the zero bytes up to off_counts and the two section gaps
//   sa_plan_rows       one block per query row: validate (cells outside [0, n_kv) become a sentinel, values other than
//                      -1 are counted into hdr.n_bad), and when the valid cells are not already strictly ascending with
//                      every invalid entry after them, sort and dedup them into the row's scratch. A row that is already
//                      in that form is not copied: its scratch row only holds the marker SA_PLAN_RAW and the merge reads
//                      sel itself (every row of the 36 recaptured C3 layers at ub4096 d131k takes this path,
//                      scripts/qsa-validation/summaries/w2_plan_c3.md).
//   sa_plan_merge<G>   one block per group of G consecutive queries of a stream: G-way merge of the sorted rows by
//                      physical quad (cell >> 2) into strictly ascending entries with the per-query cell bits, then the
//                      padding of ggml.h up to cap. Each lane owns a quad-aligned key range (quantiles of the group's
//                      longest row) and runs a count pass, a block scan and a write pass. 512 lanes for G = 4 (C3 "P_asc
//                      512", the fastest variant measured: 0.733 / 0.883 ms per layer on text / random rows at ub4096
//                      d131k), 256 for G = 1.
// Ported from the PR #219 tree's union builder (qsa3_rows_kernel + qsa3_merge_kernel, qsa-prefill.cu:28-240, as the C3
// harness copied them) with C3's sa_merge<ASC, 512> (scripts/qsa-validation/harness/c3/c3_plan.cu). Changes: the plan
// is a standalone op (streams, G in {1, 4}, the ggml.h layout and padding rule, n_bad); rows with repeated cells are
// deduplicated before the merge (by code reading, the PR merge's complete-quad shortcut would emit a quad twice for a
// non-decreasing row such as 4,5,6,7,7, which its rows kernel passes through unsorted); the bitonic sort runs in place
// on the row length (no power-of-two padding); the row loads are unrolled so that several are in flight per lane; rows
// or groups too long for LDS run the same algorithms on the scratch / sel in global memory.
//
// Determinism: the output depends only on the set of valid cells of each query (every step is a pure function of the
// sorted unique rows); n_bad is an integer atomicAdd, whose sum does not depend on the order.

#include "sparse-attn.cuh"
#include "sparse-attn-census.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdlib>

#if defined(GGML_USE_HIP)

#define SA_PLAN_SENT 0x7FFFFFFF   // invalid entry / end of a sorted row (cells are < 2^24, so it sorts after every cell)
#define SA_PLAN_RAW  (-2)         // scratch[row][0]: the row is read from sel itself (already strictly ascending)
#define SA_PLAN_DUP  0x40000000   // repeat mark of the global-memory rows path (not a cell bit: cells are < 2^24)

static constexpr int SA_PLAN_INIT_THREADS = 256;
static constexpr int SA_PLAN_ROWS_THREADS = 256;
static constexpr int SA_PLAN_LOAD_UNROLL  = 4;   // row loads in flight per thread

template <int G> struct sa_plan_merge_threads;
template <> struct sa_plan_merge_threads<1> { static constexpr int value = 256; };
template <> struct sa_plan_merge_threads<4> { static constexpr int value = 512; };

// LDS left for the dynamic rows of a block: the static arrays (scan, flags) use at most 2 KiB
static constexpr size_t SA_PLAN_STATIC_LDS_MARGIN = 4096;

struct sa_plan_args {
    const char * sel;
    size_t       sel_nb1;
    size_t       sel_nb3;
    int          n_sel;
    int          n_q;
    int          n_kv;
    int          stride;      // entries per scratch row: round4(n_sel)
    int          n_groups;    // per stream
    int          cap;
    int32_t    * scratch;     // n_stream * n_q rows of stride entries, right after ggml_nbytes(dst)
    uint8_t    * dst;
    size_t       off_counts;
    size_t       off_quads;
    size_t       off_masks;
};

static __device__ __forceinline__ int sa_plan_valid(const int key, const int n_kv) {
    return key >= 0 && key < n_kv ? key : SA_PLAN_SENT;
}

// exclusive prefix sum of v over the NT threads of the block; total = the sum over the block
template <int NT>
static __device__ __forceinline__ int sa_plan_block_scan(int * scan_s, const int v, int & total) {
    const int tid = threadIdx.x;
    scan_s[tid] = v;
    __syncthreads();
#pragma unroll 1
    for (int off = 1; off < NT; off <<= 1) {
        const int t = tid >= off ? scan_s[tid - off] : 0;
        __syncthreads();
        scan_s[tid] += t;
        __syncthreads();
    }
    total = scan_s[NT - 1];
    const int excl = scan_s[tid] - v;
    __syncthreads();
    return excl;
}

// ascending in-place sort of x[0, n) by the block (n block-uniform): bitonic network in its "flip" form, in which every
// comparator puts the minimum at the lower index. Padding the array to the next power of two with +inf leaves the
// padding where it is, so the comparators that touch it (partner >= n) are skipped and no padding is stored.
template <int NT>
static __device__ __forceinline__ void sa_plan_sort(int * x, const int n) {
    if (n < 2) {
        return;
    }
    int lg = 0;
    while ((1 << lg) < n) {
        ++lg;
    }
    const int slots = 1 << (lg - 1);   // comparators per step: half of the padded length
#pragma unroll 1
    for (int kl = 1; kl <= lg; ++kl) {
        {   // flip: pairs (b + r, b + 2^kl - 1 - r) inside each block b of 2^kl
            const int hl = kl - 1;
#pragma unroll 1
            for (int c = threadIdx.x; c < slots; c += NT) {
                const int b = (c >> hl) << kl;
                const int r = c & ((1 << hl) - 1);
                const int i = b + r;
                const int p = b + (1 << kl) - 1 - r;
                if (p < n) {
                    const int u = x[i], w = x[p];
                    if (u > w) {
                        x[i] = w;
                        x[p] = u;
                    }
                }
            }
            __syncthreads();
        }
#pragma unroll 1
        for (int jl = kl - 2; jl >= 0; --jl) {   // half-cleaners: pairs (i, i + 2^jl)
#pragma unroll 1
            for (int c = threadIdx.x; c < slots; c += NT) {
                const int i = ((c >> jl) << (jl + 1)) + (c & ((1 << jl) - 1));
                const int p = i + (1 << jl);
                if (p < n) {
                    const int u = x[i], w = x[p];
                    if (u > w) {
                        x[i] = w;
                        x[p] = u;
                    }
                }
            }
            __syncthreads();
        }
    }
}

static __global__ void __launch_bounds__(SA_PLAN_INIT_THREADS) sa_plan_init(
        uint8_t * __restrict__ dst, const ggml_sparse_attn_plan_hdr hdr,
        const size_t end_counts, const size_t end_quads) {
    const uint8_t * h = reinterpret_cast<const uint8_t *>(&hdr);
    for (size_t b = threadIdx.x; b < hdr.off_counts; b += SA_PLAN_INIT_THREADS) {
        dst[b] = b < sizeof(hdr) ? h[b] : 0;
    }
    for (size_t b = end_counts + threadIdx.x; b < hdr.off_quads; b += SA_PLAN_INIT_THREADS) {
        dst[b] = 0;
    }
    for (size_t b = end_quads + threadIdx.x; b < hdr.off_masks; b += SA_PLAN_INIT_THREADS) {
        dst[b] = 0;
    }
}

// LDS: ent[n_sel] (the validated row) and cmp[n_sel] (its valid cells in row order, then sorted) in dynamic LDS.
// Global (rows too long for LDS): the validated row is written to the scratch row and sorted there; repeats are marked,
// replaced by the sentinel and the row is sorted again.
template <bool LDS>
static __global__ void __launch_bounds__(SA_PLAN_ROWS_THREADS) sa_plan_rows(const sa_plan_args a) {
    constexpr int NT = SA_PLAN_ROWS_THREADS;
    extern __shared__ int sa_plan_rows_lds[];
    __shared__ int scan_s[NT];
    __shared__ int flag;
    __shared__ unsigned int bad_s;

    const int tid = threadIdx.x;
    const int r   = blockIdx.x;   // s * n_q + i
    const int s   = r / a.n_q;
    const int i   = r - s * a.n_q;
    const int ns  = a.n_sel;
    const int * row = reinterpret_cast<const int *>(a.sel + (size_t) i * a.sel_nb1 + (size_t) s * a.sel_nb3);
    int * out = a.scratch + (size_t) r * a.stride;
    int * ent = LDS ? sa_plan_rows_lds : out;

    if (tid == 0) {
        flag  = 0;
        bad_s = 0;
    }
    __syncthreads();

    // SA_PLAN_LOAD_UNROLL loads in flight per thread (the loop is latency-bound on the row reads)
    unsigned int bad = 0;
    for (int j0 = tid; j0 < ns; j0 += SA_PLAN_LOAD_UNROLL * NT) {
        int key[SA_PLAN_LOAD_UNROLL];
#pragma unroll
        for (int u = 0; u < SA_PLAN_LOAD_UNROLL; ++u) {
            const int j = j0 + u * NT;
            key[u] = j < ns ? row[j] : -1;
        }
#pragma unroll
        for (int u = 0; u < SA_PLAN_LOAD_UNROLL; ++u) {
            const int j = j0 + u * NT;
            if (j < ns) {
                bad += key[u] != -1 && (key[u] < 0 || key[u] >= a.n_kv);
                ent[j] = sa_plan_valid(key[u], a.n_kv);
            }
        }
    }
    if (bad) {
        atomicAdd(&bad_s, bad);
    }
    __syncthreads();
    if (tid == 0 && bad_s) {
        atomicAdd(reinterpret_cast<unsigned int *>(a.dst + offsetof(ggml_sparse_attn_plan_hdr, n_bad)), bad_s);
    }

    // usable as it is: valid cells strictly ascending, every invalid entry after them
    int f = 0;
    for (int j = tid; j + 1 < ns; j += NT) {
        const int y = ent[j + 1];
        f |= y != SA_PLAN_SENT && ent[j] >= y;
    }
    if (f) {
        atomicOr(&flag, 1);
    }
    __syncthreads();
    if (!flag) {
        if (LDS && tid == 0) {
            out[0] = SA_PLAN_RAW;
        }
        return;   // global: the scratch row already holds the validated row, which is sorted and unique
    }

    if constexpr (LDS) {
        int * cmp = ent + ns;
        // the valid cells in row order (rows of holes around ascending cells need nothing else)
        const int per = (ns + NT - 1) / NT;
        const int j0  = min(ns, tid * per);
        const int j1  = min(ns, j0 + per);
        int c = 0;
        for (int j = j0; j < j1; ++j) {
            c += ent[j] != SA_PLAN_SENT;
        }
        int total;
        int o = sa_plan_block_scan<NT>(scan_s, c, total);
        for (int j = j0; j < j1; ++j) {
            if (ent[j] != SA_PLAN_SENT) {
                cmp[o++] = ent[j];
            }
        }
        if (tid == 0) {
            flag = 0;
        }
        __syncthreads();
        f = 0;
        for (int j = tid; j + 1 < total; j += NT) {
            f |= cmp[j] >= cmp[j + 1];
        }
        if (f) {
            atomicOr(&flag, 1);
        }
        __syncthreads();

        int n = total;
        if (flag) {
            sa_plan_sort<NT>(cmp, total);
            const int per2 = (total + NT - 1) / NT;
            const int q0   = min(total, tid * per2);
            const int q1   = min(total, q0 + per2);
            int u = 0;
            for (int j = q0; j < q1; ++j) {
                u += j == 0 || cmp[j] != cmp[j - 1];
            }
            int ou = sa_plan_block_scan<NT>(scan_s, u, n);
            for (int j = q0; j < q1; ++j) {
                if (j == 0 || cmp[j] != cmp[j - 1]) {
                    out[ou++] = cmp[j];
                }
            }
        } else {
            for (int j = tid; j < total; j += NT) {
                out[j] = cmp[j];
            }
        }
        for (int j = n + tid; j < ns; j += NT) {
            out[j] = SA_PLAN_SENT;
        }
    } else {
        sa_plan_sort<NT>(ent, ns);
        // mark every repeat of its predecessor (the mark keeps the cell bits, so a neighbour reading a marked entry
        // concurrently still compares the right cell)
        for (int j = tid + 1; j < ns; j += NT) {
            const int x = ent[j];
            if (x != SA_PLAN_SENT && ((x ^ ent[j - 1]) & ~SA_PLAN_DUP) == 0) {
                ent[j] = x | SA_PLAN_DUP;
            }
        }
        __syncthreads();
        for (int j = tid; j < ns; j += NT) {
            if (ent[j] & SA_PLAN_DUP) {
                ent[j] = SA_PLAN_SENT;   // the sentinel has the bit set too: unchanged
            }
        }
        __syncthreads();
        sa_plan_sort<NT>(ent, ns);
    }
}

// the padding of ggml.h for entries [t0, cap) of a group: quad = last, mask = 0. Entries up to the next multiple of 4 one by
// one, then 16-byte quad stores and 8-byte mask stores (quads are 16-byte aligned at multiples of 4 entries: off_quads is a
// multiple of 256, cap of 4 and the dst of 128; masks are 8-byte aligned there)
template <int NT>
static __device__ __forceinline__ void sa_plan_fill(uint32_t * oq, uint16_t * om, const int t0, const int cap, const uint32_t last) {
    const int t4 = min(cap, (t0 + 3) & ~3);
    for (int t = t0 + (int) threadIdx.x; t < t4; t += NT) {
        oq[t] = last;
        om[t] = 0;
    }
    uint4 * oq4 = reinterpret_cast<uint4 *>(oq + t4);
    uint2 * om4 = reinterpret_cast<uint2 *>(om + t4);
    const uint4 q4 = make_uint4(last, last, last, last);
    const uint2 m4 = make_uint2(0, 0);
    for (int v = threadIdx.x; v < (cap - t4) / 4; v += NT) {
        oq4[v] = q4;
        om4[v] = m4;
    }
}

// entry j of query qi of the group: rows in LDS are already validated; in global memory a raw sel row is validated on read
template <bool LDS>
static __device__ __forceinline__ int sa_plan_get(const int * p, const bool raw, const int j, const int n_kv) {
    const int v = p[j];
    return !LDS && raw ? sa_plan_valid(v, n_kv) : v;
}

template <int G, bool LDS>
static __global__ void __launch_bounds__(sa_plan_merge_threads<G>::value) sa_plan_merge(const sa_plan_args a) {
    constexpr int NT = sa_plan_merge_threads<G>::value;
    extern __shared__ int sa_plan_merge_lds[];   // LDS: G rows of n_sel entries
    __shared__ int scan_s[NT];

    const int lane = threadIdx.x;
    const int e    = blockIdx.x;   // s * n_groups + g
    const int s    = e / a.n_groups;
    const int g    = e - s * a.n_groups;
    const int ns   = a.n_sel;
    const int n_kv = a.n_kv;

    const int * rp[G];   // row qi of the group (nullptr past n_q)
    bool raw[G];         // rp[qi] is a raw sel row, validated on read (global-memory variant only)
    int cnt[G];
    const int * src[G];
#pragma unroll
    for (int qi = 0; qi < G; ++qi) {   // the G marker reads are issued together
        const int q = g * G + qi;
        cnt[qi] = 0;
        raw[qi] = q < a.n_q && a.scratch[(size_t) (s * a.n_q + q) * a.stride] == SA_PLAN_RAW;
    }
#pragma unroll
    for (int qi = 0; qi < G; ++qi) {
        const int q = g * G + qi;
        src[qi] = q >= a.n_q ? nullptr
                : raw[qi]    ? reinterpret_cast<const int *>(a.sel + (size_t) q * a.sel_nb1 + (size_t) s * a.sel_nb3)
                :              a.scratch + (size_t) (s * a.n_q + q) * a.stride;
        rp[qi] = src[qi];
    }
    if constexpr (LDS) {
        // copy the rows into LDS, validating raw rows: G x SA_PLAN_LOAD_UNROLL / 2 loads in flight per lane
        constexpr int U = SA_PLAN_LOAD_UNROLL / 2;
        for (int j0 = lane; j0 < ns; j0 += U * NT) {
            int v[G][U];
#pragma unroll
            for (int qi = 0; qi < G; ++qi) {
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const int j = j0 + u * NT;
                    v[qi][u] = src[qi] != nullptr && j < ns ? src[qi][j] : SA_PLAN_SENT;
                }
            }
#pragma unroll
            for (int qi = 0; qi < G; ++qi) {
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const int j = j0 + u * NT;
                    if (src[qi] != nullptr && j < ns) {
                        sa_plan_merge_lds[qi * ns + j] = raw[qi] ? sa_plan_valid(v[qi][u], n_kv) : v[qi][u];
                    }
                }
            }
        }
#pragma unroll
        for (int qi = 0; qi < G; ++qi) {
            rp[qi]  = src[qi] != nullptr ? sa_plan_merge_lds + qi * ns : nullptr;
            raw[qi] = false;
        }
        __syncthreads();
    }

    // number of valid cells of each row (the sentinels are at the end)
#pragma unroll
    for (int qi = 0; qi < G; ++qi) {
        if (rp[qi] != nullptr) {
            int lo = 0, hi = ns;
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                if (sa_plan_get<LDS>(rp[qi], raw[qi], mid, n_kv) != SA_PLAN_SENT) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            cnt[qi] = lo;
        }
    }
    int kmax = -1;
    int rc   = 0;
#pragma unroll
    for (int qi = 0; qi < G; ++qi) {
        if (cnt[qi] > 0) {
            kmax = max(kmax, sa_plan_get<LDS>(rp[qi], raw[qi], cnt[qi] - 1, n_kv));
            rc   = max(rc, cnt[qi]);
        }
    }

    uint32_t * oq = reinterpret_cast<uint32_t *>(a.dst + a.off_quads) + (size_t) e * a.cap;
    uint16_t * om = reinterpret_cast<uint16_t *>(a.dst + a.off_masks) + (size_t) e * a.cap;
    uint32_t * oc = reinterpret_cast<uint32_t *>(a.dst + a.off_counts);
    if (kmax < 0) {   // empty group (block-uniform)
        sa_plan_fill<NT>(oq, om, 0, a.cap, 0);
        if (lane == 0) {
            oc[e] = 0;
        }
        return;
    }

    // the lane's quad-aligned key range [lo, hi): quantiles of the longest row (the first one of that length)
    int lo = 0;
    int hi = (kmax + 4) & ~3;
    {
        bool done = false;
#pragma unroll
        for (int qi = 0; qi < G; ++qi) {
            if (!done && cnt[qi] == rc) {
                done = true;
                if (lane > 0) {
                    lo = sa_plan_get<LDS>(rp[qi], raw[qi], (int) (((long long) rc * lane) / NT), n_kv) & ~3;
                }
                if (lane < NT - 1) {
                    hi = sa_plan_get<LDS>(rp[qi], raw[qi], (int) (((long long) rc * (lane + 1)) / NT), n_kv) & ~3;
                }
            }
        }
    }
    int p0[G], pe[G];
#pragma unroll
    for (int qi = 0; qi < G; ++qi) {
        int l = 0, h = cnt[qi];
        while (l < h) {
            const int mid = (l + h) >> 1;
            if (sa_plan_get<LDS>(rp[qi], raw[qi], mid, n_kv) < lo) {
                l = mid + 1;
            } else {
                h = mid;
            }
        }
        p0[qi] = l;
        h = cnt[qi];
        while (l < h) {
            const int mid = (l + h) >> 1;
            if (sa_plan_get<LDS>(rp[qi], raw[qi], mid, n_kv) < hi) {
                l = mid + 1;
            } else {
                h = mid;
            }
        }
        pe[qi] = l;
    }

    // pass 1: number of distinct quads in the lane's range
    int n = 0;
    {
        int pp[G], head[G];
#pragma unroll
        for (int qi = 0; qi < G; ++qi) {
            pp[qi]   = p0[qi];
            head[qi] = pp[qi] < pe[qi] ? sa_plan_get<LDS>(rp[qi], raw[qi], pp[qi], n_kv) : SA_PLAN_SENT;
        }
        while (true) {
            int b = SA_PLAN_SENT;
#pragma unroll
            for (int qi = 0; qi < G; ++qi) {
                b = min(b, head[qi]);
            }
            if (b == SA_PLAN_SENT) {
                break;
            }
            b >>= 2;
#pragma unroll
            for (int qi = 0; qi < G; ++qi) {
                if (head[qi] != SA_PLAN_SENT && (head[qi] >> 2) == b) {
                    int pn = pp[qi];
                    if ((head[qi] & 3) == 0 && pn + 3 < pe[qi] && sa_plan_get<LDS>(rp[qi], raw[qi], pn + 3, n_kv) == head[qi] + 3) {
                        pn += 4;   // a complete quad (rows are strictly ascending, so cells +1 and +2 are in between)
                    } else {
                        do {
                            ++pn;
                        } while (pn < pe[qi] && (sa_plan_get<LDS>(rp[qi], raw[qi], pn, n_kv) >> 2) == b);
                    }
                    pp[qi]   = pn;
                    head[qi] = pn < pe[qi] ? sa_plan_get<LDS>(rp[qi], raw[qi], pn, n_kv) : SA_PLAN_SENT;
                }
            }
            ++n;
        }
    }
    int total;
    int o = sa_plan_block_scan<NT>(scan_s, n, total);

    // pass 2: write the lane's entries
    {
        int pp[G], head[G];
#pragma unroll
        for (int qi = 0; qi < G; ++qi) {
            pp[qi]   = p0[qi];
            head[qi] = pp[qi] < pe[qi] ? sa_plan_get<LDS>(rp[qi], raw[qi], pp[qi], n_kv) : SA_PLAN_SENT;
        }
        while (true) {
            int b = SA_PLAN_SENT;
#pragma unroll
            for (int qi = 0; qi < G; ++qi) {
                b = min(b, head[qi]);
            }
            if (b == SA_PLAN_SENT) {
                break;
            }
            b >>= 2;
            uint32_t mk = 0;
#pragma unroll
            for (int qi = 0; qi < G; ++qi) {
                if (head[qi] != SA_PLAN_SENT && (head[qi] >> 2) == b) {
                    int pn = pp[qi];
                    if ((head[qi] & 3) == 0 && pn + 3 < pe[qi] && sa_plan_get<LDS>(rp[qi], raw[qi], pn + 3, n_kv) == head[qi] + 3) {
                        mk |= 0xFu << (4 * qi);
                        pn += 4;
                    } else {
                        int x = head[qi];
                        do {
                            mk |= 1u << (4 * qi + (x & 3));
                            ++pn;
                        } while (pn < pe[qi] && ((x = sa_plan_get<LDS>(rp[qi], raw[qi], pn, n_kv)) >> 2) == b);
                    }
                    pp[qi]   = pn;
                    head[qi] = pn < pe[qi] ? sa_plan_get<LDS>(rp[qi], raw[qi], pn, n_kv) : SA_PLAN_SENT;
                }
            }
            oq[o] = (uint32_t) b;
            om[o] = (uint16_t) mk;
            ++o;
        }
    }

    // padding of ggml.h: up to cap, the last real quad (the largest, kmax >> 2) with an empty mask
    sa_plan_fill<NT>(oq, om, total, a.cap, (uint32_t) (kmax >> 2));
    if (lane == 0) {
        oc[e] = (uint32_t) ((total + 3) & ~3);
    }
}

static bool sa_plan_force_global() {
    // test switch: run the rows and the merge on global memory whatever their length (both kernel variants are tested)
    static const bool force = [] {
        const char * e = getenv("GGML_CUDA_SA_PLAN_GLOBAL");
        return e != nullptr && atoi(e) != 0;
    }();
    return force;
}

template <int G>
static void sa_plan_launch_merge(const sa_plan_args & a, const int n_blocks, const size_t lds_budget, cudaStream_t stream) {
    constexpr int NT = sa_plan_merge_threads<G>::value;
    const size_t lds = (size_t) G * (size_t) a.n_sel * sizeof(int);
    if (!sa_plan_force_global() && lds <= lds_budget) {
        sa_plan_merge<G, true><<<n_blocks, NT, lds, stream>>>(a);
    } else {
        sa_plan_merge<G, false><<<n_blocks, NT, 0, stream>>>(a);
    }
}

static bool sa_plan_group_compiled(const int group) {
#define SA_PLAN_GROUP_IS(g) if (group == g) { return true; }
    GGML_CUDA_SA_PLAN_GROUPS(SA_PLAN_GROUP_IS)
#undef SA_PLAN_GROUP_IS
    return false;
}

bool ggml_cuda_sparse_attn_plan_supported(int device, const ggml_tensor * op) {
    GGML_UNUSED(device);   // integer kernels without arch-specific instructions: every HIP device
    const ggml_tensor * sel = op->src[0];
    if (sel == nullptr || sel->type != GGML_TYPE_I32 || op->type != GGML_TYPE_I8 || !ggml_is_contiguous(op)) {
        return false;
    }
    if (sel->nb[0] != sizeof(int32_t) || sel->nb[1] % sizeof(int32_t) != 0 || sel->nb[3] % sizeof(int32_t) != 0 || sel->ne[2] != 1) {
        return false;
    }
    if (!sa_plan_group_compiled(ggml_get_op_params_i32(op, 1))) {
        return false;
    }
    const int64_t n_sel = sel->ne[0], n_q = sel->ne[1], n_stream = sel->ne[3];
    // int indexing of the kernels (rows and groups on grid.x, cells and entries in int) and the u32 n_bad counter
    return n_sel >= 1 && n_sel <= INT_MAX / 4 && n_q * n_stream <= INT_MAX && n_sel * n_q * n_stream <= (int64_t) UINT32_MAX;
}

size_t ggml_cuda_sparse_attn_plan_get_alloc_size(const ggml_tensor * op) {
    const ggml_tensor * sel = op->src[0];
    const int64_t n_sel = sel->ne[0], n_q = sel->ne[1], n_stream = sel->ne[3];
    const int64_t n_kv = ggml_get_op_params_i32(op, 0);
    size_t dst_max = ggml_nbytes(op);
#define SA_PLAN_DST_MAX(g) dst_max = std::max(dst_max, ggml_sparse_attn_plan_nbytes(n_sel, n_q, n_stream, n_kv, g));
    GGML_CUDA_SA_PLAN_GROUPS(SA_PLAN_DST_MAX)
#undef SA_PLAN_DST_MAX
    return dst_max + ggml_sparse_attn_plan_scratch_nbytes(n_sel, n_q, n_stream);
}

void ggml_cuda_sparse_attn_plan(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * sel = dst->src[0];
    const int64_t n_sel    = sel->ne[0];
    const int64_t n_q      = sel->ne[1];
    const int64_t n_stream = sel->ne[3];
    const int64_t n_kv     = ggml_get_op_params_i32(dst, 0);
    const int32_t G        = ggml_get_op_params_i32(dst, 1);

    const ggml_sparse_attn_plan_layout L = ggml_sparse_attn_plan_get_layout(n_sel, n_q, n_stream, n_kv, G);
    const size_t scratch_bytes = ggml_sparse_attn_plan_scratch_nbytes(n_sel, n_q, n_stream);
    GGML_ASSERT(ggml_nbytes(dst) == L.nbytes && ggml_is_contiguous(dst));
    GGML_ASSERT(L.nbytes % sizeof(int32_t) == 0);
    // the scratch is the over-allocation of ggml_cuda_sparse_attn_plan_get_alloc_size: dst must own its allocation
    GGML_ASSERT(dst->view_src == nullptr && dst->buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_get_alloc_size(dst->buffer, dst) >= L.nbytes + scratch_bytes &&
                "SPARSE_ATTN_PLAN dst is not over-allocated by the HIP buffer type (get_alloc_size)");

    sa_plan_args a;
    a.sel        = (const char *) sel->data;
    a.sel_nb1    = sel->nb[1];
    a.sel_nb3    = sel->nb[3];
    a.n_sel      = (int) n_sel;
    a.n_q        = (int) n_q;
    a.n_kv       = (int) n_kv;
    a.stride     = (int) GGML_PAD(n_sel, 4);
    a.n_groups   = (int) L.n_groups;
    a.cap        = (int) L.cap;
    a.dst        = (uint8_t *) dst->data;
    a.scratch    = (int32_t *) (a.dst + L.nbytes);
    a.off_counts = L.off_counts;
    a.off_quads  = L.off_quads;
    a.off_masks  = L.off_masks;

    ggml_sparse_attn_plan_hdr hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic      = GGML_SPARSE_ATTN_PLAN_MAGIC;
    hdr.version    = GGML_SPARSE_ATTN_PLAN_VERSION;
    hdr.group      = (uint32_t) G;
    hdr.n_q        = (uint32_t) n_q;
    hdr.n_stream   = (uint32_t) n_stream;
    hdr.n_groups   = (uint32_t) L.n_groups;
    hdr.cap        = (uint32_t) L.cap;
    hdr.n_bad      = 0;   // the rows kernel adds its counts
    hdr.off_counts = L.off_counts;
    hdr.off_quads  = L.off_quads;
    hdr.off_masks  = L.off_masks;
    hdr.reserved   = 0;
    const size_t n_ent      = (size_t) n_stream * (size_t) L.n_groups;
    const size_t end_counts = L.off_counts + n_ent * sizeof(uint32_t);
    const size_t end_quads  = L.off_quads + n_ent * (size_t) L.cap * sizeof(uint32_t);

    const size_t smpb       = ggml_cuda_info().devices[ctx.device].smpb;
    const size_t lds_budget = smpb > SA_PLAN_STATIC_LDS_MARGIN ? smpb - SA_PLAN_STATIC_LDS_MARGIN : 0;
    cudaStream_t stream     = ctx.stream();

    sa_plan_init<<<1, SA_PLAN_INIT_THREADS, 0, stream>>>(a.dst, hdr, end_counts, end_quads);

    const int    n_rows   = (int) (n_stream * n_q);
    const size_t rows_lds = 2 * (size_t) n_sel * sizeof(int);
    if (!sa_plan_force_global() && rows_lds <= lds_budget) {
        sa_plan_rows<true><<<n_rows, SA_PLAN_ROWS_THREADS, rows_lds, stream>>>(a);
    } else {
        sa_plan_rows<false><<<n_rows, SA_PLAN_ROWS_THREADS, 0, stream>>>(a);
    }

    const int n_groups_total = (int) n_ent;
    switch (G) {
#define SA_PLAN_CASE(g)                                                       \
        case g:                                                               \
            sa_plan_launch_merge<g>(a, n_groups_total, lds_budget, stream);   \
            ggml_cuda_sa_census_add(GGML_CUDA_SA_CENSUS_PLAN_G##g);           \
            break;
        GGML_CUDA_SA_PLAN_GROUPS(SA_PLAN_CASE)
#undef SA_PLAN_CASE
        default:
            GGML_ABORT("SPARSE_ATTN_PLAN: group %d has no HIP kernel", G);
    }
}

#else // !defined(GGML_USE_HIP)

// CUDA builds: no sparse-attention kernels (PLAN-v3 §6.2, D22); supports_op is false, so these are never dispatched
bool ggml_cuda_sparse_attn_plan_supported(int device, const ggml_tensor * op) {
    GGML_UNUSED(device);
    GGML_UNUSED(op);
    return false;
}

size_t ggml_cuda_sparse_attn_plan_get_alloc_size(const ggml_tensor * op) {
    return ggml_nbytes(op);
}

void ggml_cuda_sparse_attn_plan(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("SPARSE_ATTN_PLAN is not supported by this CUDA build");
}

#endif // defined(GGML_USE_HIP)
