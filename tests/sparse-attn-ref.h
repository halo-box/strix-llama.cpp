// Host references for GGML_OP_SPARSE_ATTN_PLAN and GGML_OP_SPARSE_ATTN (tests only).
//
// plan(): the C3 oracle. The algorithm is cpu_plan() of the C3 experiment (scripts/qsa-validation/harness/c3/c3_plan.cu,
//   "CPU reference": one std::map<quad, mask> per group, filled with every valid cell of the group's queries, read in
//   ascending quad order, padded to a multiple of 4), generalised from G = 4 to G in {1, 4} and to n_stream streams,
//   and encoded in the dst layout of ggml.h (header, counts, quads, masks; the padding rule of ggml.h replaces C3's
//   0xFFFFFFFF padding quad by the last real quad of the group). It shares no code with the ggml CPU op
//   (ggml-cpu/sparse-attn.cpp: per-row sort/unique, then a k-way merge).
//
// attend(): O_CON, the numeric contract of ggml_sparse_attn (ggml.h), written as R1-M-Lab A's oracle_group(O_CON)
//   (scripts/qsa-validation/harness/bandsop/bands_ref.h) is, generalised to any D, GQA ratio, stream count, K/V type
//   (the type's to_float, rounded to f16) and the H_q / H_o rotations. The cells of a query come straight from its
//   selection row (valid, sorted, unique), not from a plan. Any translation unit that includes this header and wants
//   bit equality with the CPU op must be compiled with -ffp-contract=off (tests/CMakeLists.txt).
#pragma once

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <thread>
#include <vector>

namespace sa_ref {

// rows[((s * n_q) + i) * n_sel + j] = sel[j, i, 0, s]
static std::vector<uint8_t> plan(const std::vector<int32_t> & rows, int64_t n_sel, int64_t n_q, int64_t n_stream,
                                 int64_t n_kv, int32_t G) {
    const ggml_sparse_attn_plan_layout L = ggml_sparse_attn_plan_get_layout(n_sel, n_q, n_stream, n_kv, G);
    std::vector<uint8_t> out(L.nbytes, 0);
    uint32_t * counts = (uint32_t *) (out.data() + L.off_counts);
    uint32_t * quads  = (uint32_t *) (out.data() + L.off_quads);
    uint16_t * masks  = (uint16_t *) (out.data() + L.off_masks);

    uint64_t n_bad = 0;
    for (size_t k = 0; k < rows.size(); ++k) {
        n_bad += rows[k] != -1 && (rows[k] < 0 || rows[k] >= n_kv);
    }

    for (int64_t s = 0; s < n_stream; ++s) {
        for (int64_t g = 0; g < L.n_groups; ++g) {
            std::map<uint32_t, uint16_t> m;
            for (int qi = 0; qi < G; ++qi) {
                const int64_t q = G * g + qi;
                if (q >= n_q) {
                    break;
                }
                for (int64_t j = 0; j < n_sel; ++j) {
                    const int32_t k = rows[(size_t) ((s * n_q + q) * n_sel + j)];
                    if (k >= 0 && k < n_kv) {
                        m[(uint32_t) k >> 2] |= (uint16_t) (1u << (4 * qi + (k & 3)));
                    }
                }
            }
            const int64_t e = s * L.n_groups + g;
            uint32_t * oq = quads + e * L.cap;
            uint16_t * om = masks + e * L.cap;
            int64_t n = 0;
            for (const auto & kv : m) {
                oq[n] = kv.first;
                om[n] = kv.second;
                ++n;
            }
            const uint32_t last = n > 0 ? oq[n - 1] : 0;
            counts[e] = (uint32_t) ((n + 3) & ~(int64_t) 3);
            for (int64_t k = n; k < L.cap; ++k) {
                oq[k] = last;
                om[k] = 0;
            }
        }
    }

    ggml_sparse_attn_plan_hdr hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic      = GGML_SPARSE_ATTN_PLAN_MAGIC;
    hdr.version    = GGML_SPARSE_ATTN_PLAN_VERSION;
    hdr.group      = (uint32_t) G;
    hdr.n_q        = (uint32_t) n_q;
    hdr.n_stream   = (uint32_t) n_stream;
    hdr.n_groups   = (uint32_t) L.n_groups;
    hdr.cap        = (uint32_t) L.cap;
    hdr.n_bad      = (uint32_t) n_bad;
    hdr.off_counts = L.off_counts;
    hdr.off_quads  = L.off_quads;
    hdr.off_masks  = L.off_masks;
    memcpy(out.data(), &hdr, sizeof(hdr));
    return out;
}

static inline float rh(float x) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
}

// the H_q / H_o transform: orthonormal Sylvester-Hadamard over consecutive blocks of n values (n = 0: identity),
// scale by 1/sqrt(n) first, then the butterflies (ggml-cpu ggml_compute_forward_fwht order)
static void fwht(float * x, int64_t D, int32_t n) {
    if (n <= 0) {
        return;
    }
    const float sc = 1.0f / sqrtf((float) n);
    for (int64_t b = 0; b < D; b += n) {
        float * y = x + b;
        for (int32_t j = 0; j < n; ++j) {
            y[j] = y[j] * sc;
        }
        for (int32_t len = 1; len < n; len <<= 1) {
            for (int32_t i = 0; i < n; i += 2 * len) {
                for (int32_t j = 0; j < len; ++j) {
                    const float u = y[i + j];
                    const float v = y[i + len + j];
                    y[i + j]       = u + v;
                    y[i + len + j] = u - v;
                }
            }
        }
    }
}

// a host copy of a (possibly strided) tensor: the bytes from t->data over ggml_nbytes(t), indexed with t->nb
struct view {
    ggml_type type;
    int64_t   ne[4];
    size_t    nb[4];
    std::vector<uint8_t> bytes;

    const uint8_t * row(int64_t i1, int64_t i2, int64_t i3) const {
        return bytes.data() + i1 * nb[1] + i2 * nb[2] + i3 * nb[3];
    }
};

// contract value of one K/V row: to_float, rounded to f16
static void load_row(const view & t, int64_t i1, int64_t i2, int64_t i3, float * out) {
    const int64_t D = t.ne[0];
    const uint8_t * r = t.row(i1, i2, i3);
    if (t.type == GGML_TYPE_F32) {
        memcpy(out, r, D * sizeof(float));
    } else {
        ggml_get_type_traits(t.type)->to_float(r, out, D);
    }
    for (int64_t d = 0; d < D; ++d) {
        out[d] = rh(out[d]);
    }
}

// valid cells of a row, sorted and unique
static void row_cells(const int32_t * row, int64_t n_sel, int64_t n_kv, std::vector<int32_t> & cells) {
    cells.clear();
    for (int64_t j = 0; j < n_sel; ++j) {
        if (row[j] >= 0 && row[j] < n_kv) {
            cells.push_back(row[j]);
        }
    }
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
}

// O_CON for every (s, i, h); q: F32 view [D, n_q, n_head, n_stream]; out [D, n_head, n_q, n_stream] contiguous
static std::vector<float> attend(const view & q, const view & k, const view & v, const std::vector<int32_t> & rows,
                                 int64_t n_sel, int64_t n_kv, float scale, int32_t rot_q, int32_t rot_o, int n_threads) {
    const int64_t D = q.ne[0], n_q = q.ne[1], n_head = q.ne[2], n_stream = q.ne[3];
    const int64_t n_head_kv = k.ne[2];
    const int64_t r = n_head / n_head_kv;
    std::vector<float> out((size_t) (D * n_head * n_q * n_stream), 0.0f);

    const int64_t n_items = n_stream * n_q * n_head_kv;
    auto work = [&](int ith, int nth) {
        std::vector<float> qT(D * r), acc(r * D), tmp(D), kr(D), vr(D), sc(r), m(r), l(r);
        std::vector<int32_t> cells;
        for (int64_t t = ith; t < n_items; t += nth) {
            const int64_t kvh = t % n_head_kv;
            const int64_t i   = (t / n_head_kv) % n_q;
            const int64_t s   = t / (n_head_kv * n_q);
            for (int64_t hr = 0; hr < r; ++hr) {
                memcpy(tmp.data(), q.row(i, kvh * r + hr, s), D * sizeof(float));
                fwht(tmp.data(), D, rot_q);
                for (int64_t d = 0; d < D; ++d) {
                    qT[d * r + hr] = rh(tmp[d] * scale);
                }
                m[hr] = -INFINITY;
                l[hr] = 0.0f;
            }
            std::fill(acc.begin(), acc.end(), 0.0f);
            row_cells(rows.data() + (size_t) ((s * n_q + i) * n_sel), n_sel, n_kv, cells);
            for (int32_t c : cells) {
                load_row(k, c, kvh, s, kr.data());
                load_row(v, c, kvh, s, vr.data());
                for (int64_t hr = 0; hr < r; ++hr) {
                    sc[hr] = 0.0f;
                }
                for (int64_t d = 0; d < D; ++d) {
                    const float kd = kr[d];
                    for (int64_t hr = 0; hr < r; ++hr) {
                        sc[hr] += qT[d * r + hr] * kd;
                    }
                }
                for (int64_t hr = 0; hr < r; ++hr) {
                    float * a = acc.data() + hr * D;
                    if (sc[hr] > m[hr]) {
                        const float alpha = m[hr] == -INFINITY ? 0.0f : expf(m[hr] - sc[hr]);
                        for (int64_t d = 0; d < D; ++d) {
                            a[d] *= alpha;
                        }
                        l[hr] *= alpha;
                        m[hr] = sc[hr];
                    }
                    const float p  = expf(sc[hr] - m[hr]);
                    const float ph = rh(p);
                    l[hr] += p;
                    for (int64_t d = 0; d < D; ++d) {
                        a[d] += ph * vr[d];
                    }
                }
            }
            for (int64_t hr = 0; hr < r; ++hr) {
                const float * a = acc.data() + hr * D;
                for (int64_t d = 0; d < D; ++d) {
                    tmp[d] = l[hr] > 0.0f ? a[d] / l[hr] : 0.0f;
                }
                fwht(tmp.data(), D, rot_o);
                memcpy(out.data() + (size_t) (((s * n_q + i) * n_head + kvh * r + hr) * D), tmp.data(), D * sizeof(float));
            }
        }
    };
    std::vector<std::thread> th;
    for (int t = 0; t < n_threads; ++t) {
        th.emplace_back(work, t, n_threads);
    }
    for (auto & x : th) {
        x.join();
    }
    return out;
}

// f64 oracles of PLAN-v3 §3 (R1-M-Lab A, bands_ref.h), reported only (never gated):
//   O_C18  q = f16(f16(q) * f16(scale)), K/V = to_float rounded to f16, bf16 saturated to +-65504 first (c18_attn.cu:290)
//   O_PLAN q * scale in f32, K/V = to_float (no f16 rounding)
// both with an exact f64 softmax over the row's valid cells. Computed for the queries listed in `queries` (all heads and
// streams); returns max |gpu - ref| / max |ref| over them, against `got` ([D, n_head, n_q, n_stream] contiguous).
// Infinite or NaN reference values are skipped (the overflow case).
enum f64_oracle { O_C18 = 0, O_PLAN = 1 };

static double f64_maxrel(const view & q, const view & k, const view & v, const std::vector<int32_t> & rows, int64_t n_sel,
                         int64_t n_kv, float scale, int32_t rot_q, int32_t rot_o, const std::vector<int64_t> & queries,
                         const float * got, f64_oracle mode) {
    const int64_t D = q.ne[0], n_q = q.ne[1], n_head = q.ne[2], n_stream = q.ne[3];
    const int64_t r = n_head / k.ne[2];
    auto row_f = [&](const view & t, int64_t c, int64_t g, int64_t s, std::vector<float> & out) {
        out.resize(D);
        const uint8_t * p = t.row(c, g, s);
        if (t.type == GGML_TYPE_F32) {
            memcpy(out.data(), p, D * sizeof(float));
        } else {
            ggml_get_type_traits(t.type)->to_float(p, out.data(), D);
        }
        if (mode == O_C18) {
            for (int64_t d = 0; d < D; ++d) {
                float x = out[d];
                if (t.type == GGML_TYPE_BF16) {
                    x = std::min(65504.0f, std::max(-65504.0f, x));
                }
                out[d] = rh(x);
            }
        }
    };
    double emax = 0.0, amax = 0.0;
    std::vector<int32_t> cells;
    std::vector<float> tmp(D), kr, vr;
    std::vector<double> qd(D), sc, acc(D);
    for (int64_t s = 0; s < n_stream; ++s) {
        for (int64_t i : queries) {
            if (i < 0 || i >= n_q) {
                continue;
            }
            row_cells(rows.data() + (size_t) ((s * n_q + i) * n_sel), n_sel, n_kv, cells);
            for (int64_t h = 0; h < n_head; ++h) {
                const int64_t kvh = h / r;
                memcpy(tmp.data(), q.row(i, h, s), D * sizeof(float));
                fwht(tmp.data(), D, rot_q);
                for (int64_t d = 0; d < D; ++d) {
                    qd[d] = mode == O_C18 ? (double) rh(rh(tmp[d]) * rh(scale)) : (double) (tmp[d] * scale);
                }
                sc.assign(cells.size(), 0.0);
                double mx = -INFINITY;
                for (size_t ci = 0; ci < cells.size(); ++ci) {
                    row_f(k, cells[ci], kvh, s, kr);
                    double x = 0.0;
                    for (int64_t d = 0; d < D; ++d) {
                        x += qd[d] * (double) kr[d];
                    }
                    sc[ci] = x;
                    mx = std::max(mx, x);
                }
                double den = 0.0;
                std::fill(acc.begin(), acc.end(), 0.0);
                for (size_t ci = 0; ci < cells.size(); ++ci) {
                    const double p = std::exp(sc[ci] - mx);
                    den += p;
                    row_f(v, cells[ci], kvh, s, vr);
                    for (int64_t d = 0; d < D; ++d) {
                        acc[d] += p * (double) vr[d];
                    }
                }
                for (int64_t d = 0; d < D; ++d) {
                    tmp[d] = den > 0.0 ? (float) (acc[d] / den) : 0.0f;
                }
                fwht(tmp.data(), D, rot_o);
                const float * g = got + (size_t) (((s * n_q + i) * n_head + h) * D);
                for (int64_t d = 0; d < D; ++d) {
                    if (!std::isfinite(tmp[d]) || !std::isfinite(g[d])) {
                        continue;
                    }
                    amax = std::max(amax, (double) std::fabs(tmp[d]));
                    emax = std::max(emax, (double) std::fabs((double) g[d] - (double) tmp[d]));
                }
            }
        }
    }
    return amax > 0.0 ? emax / amax : emax;
}

} // namespace sa_ref
