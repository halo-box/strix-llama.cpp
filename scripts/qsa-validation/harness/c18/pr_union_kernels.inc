#define QSA3_SENT 0x7FFFFFFF

// padding sentinel of the union's block-id list (uint32, so the id space does not limit the depth)
#define QSA3_BLK_SENT 0xFFFFFFFFu
__global__ __launch_bounds__(256) void qsa3_rows_kernel(
        const int * __restrict__ ids, const size_t i1, const int ns, const int nk, int * __restrict__ srow, int * __restrict__ sflag,
        const char * mask, size_t m1, const int np2) {
    extern __shared__ int ent[];   // max(np2, 2 ns) entries: the row, then its compacted copy
    __shared__ int unsorted, kdesc;
    __shared__ int scan_s[256];
    const int q = blockIdx.x, tid = threadIdx.x;
    const int * row = reinterpret_cast<const int *>(reinterpret_cast<const char *>(ids) + (size_t) q * i1);
    if (tid == 0) { unsorted = 0; }
    __syncthreads();
    // without a mask the row itself is the visibility record: the graph only lists cells the query may see
    // and writes -1 elsewhere (complete-block selection with tails)
    const auto * mq = mask ? reinterpret_cast<const uint16_t *>(mask + (size_t) q * m1) : nullptr;
    for (int j = tid; j < ns; j += 256) {
        const int key = row[j];
        const bool valid = key >= 0 && key < nk;
        const bool visible = valid && (!mq || mq[key] != 0xfc00u);
        ent[j] = visible ? key : QSA3_SENT;
        if (valid && !visible) { atomicOr(&unsorted, 1); }
    }
    __syncthreads();
    for (int j = tid; j + 1 < ns; j += 256) {
        if (ent[j] > ent[j+1]) { atomicOr(&unsorted, 1); }
    }
    __syncthreads();
    if (unsorted) {
        // holes (-1 / hidden cells -> sentinel) around ascending keys, the usual complete-block row: compact the keys in
        // row order with a prefix sum; if they are ascending that is exactly the sorted row (sentinels sort last)
        int * cmp = ent + ns;
        const int per = (ns + 255) / 256, j0 = min(ns, tid * per), j1 = min(ns, j0 + per);
        int cnt = 0;
        for (int j = j0; j < j1; ++j) { cnt += ent[j] != QSA3_SENT; }
        scan_s[tid] = cnt;
        if (tid == 0) { kdesc = 0; }
        __syncthreads();
        for (int off = 1; off < 256; off <<= 1) {
            const int v = tid >= off ? scan_s[tid - off] : 0;
            __syncthreads();
            scan_s[tid] += v;
            __syncthreads();
        }
        const int total = scan_s[255];
        int o = scan_s[tid] - cnt;
        for (int j = j0; j < j1; ++j) { if (ent[j] != QSA3_SENT) { cmp[o++] = ent[j]; } }
        __syncthreads();
        for (int j = tid; j + 1 < total; j += 256) {
            if (cmp[j] > cmp[j+1]) { atomicOr(&kdesc, 1); }
        }
        __syncthreads();
        if (!kdesc) {
            int * dst = srow + (size_t) q * ns;
            for (int j = tid; j < ns; j += 256) { dst[j] = j < total ? cmp[j] : QSA3_SENT; }
            if (tid == 0) { sflag[q] = 1; }
            return;
        }
    }
    if (unsorted) {
        int * dst = srow + (size_t) q * ns;
        if (np2 == 0) {
            for (int j = tid; j < ns; j += 256) {
                const int e = ent[j];
                int rank = 0;
                for (int k = 0; k < ns; ++k) { const int f = ent[k]; rank += (f < e) || (f == e && k < j); }
                dst[rank] = e;
            }
        } else {
            // bitonic sort of the row padded to np2 with the sentinel (sorts last): O(n log^2 n) instead of the O(n^2)
            // rank sort; equal entries are equal values, so the sorted row is the same as the rank sort's
            for (int j = ns + tid; j < np2; j += 256) { ent[j] = QSA3_SENT; }
            __syncthreads();
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
            for (int j = tid; j < ns; j += 256) { dst[j] = ent[j]; }
        }
    }
    if (tid == 0) { sflag[q] = unsorted; }
}

// kernel B: one workgroup (128 lanes) per group of 4 queries. Lane = key sub-range; the range boundaries are the quantiles of the first non-empty row (block aligned), so the work per lane is balanced for the real (sorted, similar) rows. Each lane merges the 4 sorted rows inside its range: pass 1 counts, block-wide prefix sum, pass 2 writes.
#define QSA3_MERGE_LANES 128
static __device__ __forceinline__ int qsa3_get(const int * __restrict__ p, const int j, const int nk, const bool raw) {
    const int key = p[j];
    return raw ? ((key >= 0 && key < nk) ? key : QSA3_SENT) : key;
}
__global__ __launch_bounds__(QSA3_MERGE_LANES) void qsa3_merge_kernel(
        const int * __restrict__ ids, const size_t i1, const int n_q, const int ns, const int nk,
        const int * __restrict__ srow, const int * __restrict__ sflag,
        uint32_t * __restrict__ ublk, uint16_t * __restrict__ umask, int * __restrict__ ucount, const int cap) {
    extern __shared__ int rows_s[];
    __shared__ int scan_s[QSA3_MERGE_LANES];
    const int lane = threadIdx.x;
    const int g = blockIdx.x;
#pragma unroll
    for (int qi = 0; qi < QSA3_G; ++qi) {
        const int q = QSA3_G * g + qi;
        int * dst = rows_s + qi * ns;
        if (q < n_q) {
            const bool rw = sflag[q] == 0;
            const int * src = rw ? reinterpret_cast<const int *>(reinterpret_cast<const char *>(ids) + (size_t) q * i1) : srow + (size_t) q * ns;
            for (int j = lane; j < ns; j += QSA3_MERGE_LANES) { dst[j] = qsa3_get(src, j, nk, rw); }
        } else {
            for (int j = lane; j < ns; j += QSA3_MERGE_LANES) { dst[j] = QSA3_SENT; }
        }
    }
    __syncthreads();
    const int * rp[QSA3_G]; bool raw[QSA3_G]; int cnt[QSA3_G];
#pragma unroll
    for (int qi = 0; qi < QSA3_G; ++qi) {
        raw[qi] = false; rp[qi] = rows_s + qi * ns;
        int a = 0, b = ns;
        while (a < b) { const int mid = (a + b) >> 1; if (rp[qi][mid] < QSA3_SENT) { a = mid + 1; } else { b = mid; } }
        cnt[qi] = a;
    }
    int kmax = -1, ref = -1;
#pragma unroll
    for (int qi = 0; qi < QSA3_G; ++qi) {
        if (cnt[qi] > 0) { kmax = max(kmax, qsa3_get(rp[qi], cnt[qi] - 1, nk, raw[qi])); if (ref < 0) { ref = qi; } }
    }
    uint32_t * ob = ublk + (size_t) g * cap; uint16_t * om = umask + (size_t) g * cap;
    if (kmax < 0) { if (lane == 0) { ucount[g] = 0; } return; }
    const int rc = cnt[ref];
    const int lo = lane == 0 ? 0 : (qsa3_get(rp[ref], (int) (((long long) rc * lane) / QSA3_MERGE_LANES), nk, raw[ref]) & ~3);
    const int hi = lane == QSA3_MERGE_LANES - 1 ? ((kmax + 4) & ~3)
                 : (qsa3_get(rp[ref], (int) (((long long) rc * (lane + 1)) / QSA3_MERGE_LANES), nk, raw[ref]) & ~3);
    int p[QSA3_G], pe[QSA3_G];
#pragma unroll
    for (int qi = 0; qi < QSA3_G; ++qi) {
        int a = 0, b = cnt[qi];
        while (a < b) { const int mid = (a + b) >> 1; if (qsa3_get(rp[qi], mid, nk, raw[qi]) < lo) { a = mid + 1; } else { b = mid; } }
        p[qi] = a; b = cnt[qi];
        while (a < b) { const int mid = (a + b) >> 1; if (qsa3_get(rp[qi], mid, nk, raw[qi]) < hi) { a = mid + 1; } else { b = mid; } }
        pe[qi] = a;
    }
    int n = 0;
    {
        int pp[QSA3_G]; int head[QSA3_G];
#pragma unroll
        for (int qi = 0; qi < QSA3_G; ++qi) { pp[qi] = p[qi]; head[qi] = pp[qi] < pe[qi] ? qsa3_get(rp[qi], pp[qi], nk, raw[qi]) : QSA3_SENT; }
        while (true) {
            int b = QSA3_SENT;
#pragma unroll
            for (int qi = 0; qi < QSA3_G; ++qi) { b = min(b, head[qi]); }
            if (b == QSA3_SENT) { break; }
            b >>= 2;
#pragma unroll
            for (int qi = 0; qi < QSA3_G; ++qi) {
                if (head[qi] != QSA3_SENT && (head[qi] >> 2) == b) {
                    const int * rr = rp[qi];
                    int pn = pp[qi];
                    if ((head[qi] & 3) == 0 && pn + 3 < pe[qi] && rr[pn+3] == head[qi] + 3 && rr[pn+1] == head[qi] + 1 && rr[pn+2] == head[qi] + 2) { pn += 4; }
                    else { do { ++pn; } while (pn < pe[qi] && (rr[pn] >> 2) == b); }
                    pp[qi] = pn; head[qi] = pn < pe[qi] ? rr[pn] : QSA3_SENT;
                }
            }
            ++n;
        }
    }
    scan_s[lane] = n;
    __syncthreads();
    for (int off = 1; off < QSA3_MERGE_LANES; off <<= 1) {
        const int v = lane >= off ? scan_s[lane - off] : 0;
        __syncthreads();
        scan_s[lane] += v;
        __syncthreads();
    }
    const int total = scan_s[QSA3_MERGE_LANES - 1];
    int o = scan_s[lane] - n;
    {
        int head[QSA3_G];
#pragma unroll
        for (int qi = 0; qi < QSA3_G; ++qi) { head[qi] = p[qi] < pe[qi] ? qsa3_get(rp[qi], p[qi], nk, raw[qi]) : QSA3_SENT; }
        while (true) {
            int b = QSA3_SENT;
#pragma unroll
            for (int qi = 0; qi < QSA3_G; ++qi) { b = min(b, head[qi]); }
            if (b == QSA3_SENT) { break; }
            b >>= 2;
            uint32_t mk = 0;
#pragma unroll
            for (int qi = 0; qi < QSA3_G; ++qi) {
                if (head[qi] != QSA3_SENT && (head[qi] >> 2) == b) {
                    const int * rr = rp[qi];
                    int pn = p[qi];
                    if ((head[qi] & 3) == 0 && pn + 3 < pe[qi] && rr[pn+3] == head[qi] + 3 && rr[pn+1] == head[qi] + 1 && rr[pn+2] == head[qi] + 2) { mk |= 0xFu << (4*qi); pn += 4; }
                    else { do { mk |= 1u << (4*qi + (rr[pn] & 3)); ++pn; } while (pn < pe[qi] && (rr[pn] >> 2) == b); }
                    p[qi] = pn; head[qi] = pn < pe[qi] ? rr[pn] : QSA3_SENT;
                }
            }
            ob[o] = (uint32_t) b; om[o] = (uint16_t) mk; ++o;
        }
    }
    if (lane == QSA3_MERGE_LANES - 1) {
        int t = total;
        while (t & 3) { ob[t] = QSA3_BLK_SENT; om[t] = 0; ++t; }
        ucount[g] = t;
    }
}

