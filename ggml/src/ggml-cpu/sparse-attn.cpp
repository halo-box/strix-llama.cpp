// CPU forwards of GGML_OP_SPARSE_ATTN_PLAN and GGML_OP_SPARSE_ATTN (the reference of every backend, ggml.h).
//
// This translation unit is compiled with -ffp-contract=off (ggml-cpu/CMakeLists.txt): the attend implements the
// numeric contract of ggml_sparse_attn operation by operation, and a fused multiply-add of `s += q * k` or
// `acc += p * v` would round differently. Vectorisation is allowed: every vector lane performs the same scalar
// operation in the same order (the score loop runs the r query heads of a kv head in the lanes, "vertical" SIMD).

#include "ops.h"

#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "simd-mappings.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------------------------------------------------
// plan
// ---------------------------------------------------------------------------------------------------------------------

static const int32_t SA_ROW_END = INT32_MAX;   // end marker of a sorted row in the scratch (cells are < 2^24)

size_t ggml_cpu_sparse_attn_plan_work_size(const struct ggml_tensor * node, int n_tasks) {
    const struct ggml_tensor * sel = node->src[0];
    return ggml_sparse_attn_plan_scratch_nbytes(sel->ne[0], sel->ne[1], sel->ne[3]) + (size_t) n_tasks * CACHE_LINE_SIZE;
}

void ggml_compute_forward_sparse_attn_plan(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_sparse_attn_plan_shape_why(dst) == nullptr);

    const struct ggml_tensor * sel = dst->src[0];

    const int64_t n_sel    = sel->ne[0];
    const int64_t n_q      = sel->ne[1];
    const int64_t n_stream = sel->ne[3];
    const int64_t n_kv     = ggml_get_op_params_i32(dst, 0);
    const int32_t G        = ggml_get_op_params_i32(dst, 1);

    const ggml_sparse_attn_plan_layout L = ggml_sparse_attn_plan_get_layout(n_sel, n_q, n_stream, n_kv, G);
    GGML_ASSERT((size_t) ggml_nbytes(dst) == L.nbytes);
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t stride   = GGML_PAD(n_sel, 4);
    const size_t  scratch  = ggml_sparse_attn_plan_scratch_nbytes(n_sel, n_q, n_stream);
    GGML_ASSERT(params->wsize >= scratch + (size_t) nth * CACHE_LINE_SIZE);
    int32_t  * rows = (int32_t *) params->wdata;
    uint64_t * bad  = (uint64_t *) ((char *) params->wdata + scratch);   // one cache line per thread

    // phase 1: each query row -> its valid cells, ascending and unique, in the scratch (SA_ROW_END after the last one)
    {
        const int64_t n_rows = n_stream * n_q;
        const int64_t r0 = n_rows * ith / nth;
        const int64_t r1 = n_rows * (ith + 1) / nth;
        uint64_t n_bad = 0;
        for (int64_t r = r0; r < r1; ++r) {
            const int64_t s = r / n_q;
            const int64_t i = r % n_q;
            const char * src = (const char *) sel->data + i * sel->nb[1] + s * sel->nb[3];
            int32_t * out = rows + r * stride;
            int64_t n = 0;
            for (int64_t j = 0; j < n_sel; ++j) {
                const int32_t c = *(const int32_t *) (src + j * sizeof(int32_t));
                if (c >= 0 && c < n_kv) {
                    out[n++] = c;
                } else if (c != -1) {
                    ++n_bad;
                }
            }
            bool ascending = true;
            for (int64_t j = 1; j < n && ascending; ++j) {
                ascending = out[j - 1] < out[j];
            }
            if (!ascending) {
                std::sort(out, out + n);
                n = std::unique(out, out + n) - out;
            }
            for (int64_t j = n; j < stride; ++j) {
                out[j] = SA_ROW_END;
            }
        }
        bad[ith * (CACHE_LINE_SIZE / sizeof(uint64_t))] = n_bad;
    }

    ggml_barrier(params->threadpool);

    uint8_t  * base   = (uint8_t *) dst->data;
    uint32_t * counts = (uint32_t *) (base + L.off_counts);
    uint32_t * quads  = (uint32_t *) (base + L.off_quads);
    uint16_t * masks  = (uint16_t *) (base + L.off_masks);
    const int64_t n_ent = n_stream * L.n_groups;
    const int64_t cap   = L.cap;

    if (ith == 0) {
        uint64_t n_bad = 0;
        for (int t = 0; t < nth; ++t) {
            n_bad += bad[t * (CACHE_LINE_SIZE / sizeof(uint64_t))];
        }
        GGML_ASSERT(n_bad <= UINT32_MAX);
        ggml_sparse_attn_plan_hdr hdr;
        memset(&hdr, 0, sizeof(hdr));
        hdr.magic      = GGML_SPARSE_ATTN_PLAN_MAGIC;
        hdr.version    = GGML_SPARSE_ATTN_PLAN_VERSION;
        hdr.group      = (uint32_t) G;
        hdr.n_q        = (uint32_t) n_q;
        hdr.n_stream   = (uint32_t) n_stream;
        hdr.n_groups   = (uint32_t) L.n_groups;
        hdr.cap        = (uint32_t) cap;
        hdr.n_bad      = (uint32_t) n_bad;
        hdr.off_counts = L.off_counts;
        hdr.off_quads  = L.off_quads;
        hdr.off_masks  = L.off_masks;
        hdr.reserved   = 0;
        memset(base, 0, L.off_counts);
        memcpy(base, &hdr, sizeof(hdr));
        // section alignment gaps
        const size_t end_counts = L.off_counts + (size_t) n_ent * sizeof(uint32_t);
        const size_t end_quads  = L.off_quads  + (size_t) n_ent * (size_t) cap * sizeof(uint32_t);
        memset(base + end_counts, 0, L.off_quads - end_counts);
        memset(base + end_quads,  0, L.off_masks - end_quads);
    }

    // phase 2: each group -> the union of its queries' cells by ascending quad, with the per-query cell bits
    {
        const int64_t e0 = n_ent * ith / nth;
        const int64_t e1 = n_ent * (ith + 1) / nth;
        for (int64_t e = e0; e < e1; ++e) {
            const int64_t s = e / L.n_groups;
            const int64_t g = e % L.n_groups;
            const int32_t * rp[4];
            int64_t pos[4];
            int nr = 0;
            for (int j = 0; j < G; ++j) {
                const int64_t i = g * G + j;
                rp[j]  = i < n_q ? rows + (s * n_q + i) * stride : nullptr;
                pos[j] = 0;
                nr     = j + 1;
            }
            uint32_t * oq = quads + e * cap;
            uint16_t * om = masks + e * cap;
            int64_t n = 0;
            for (;;) {
                int32_t head = SA_ROW_END;
                for (int j = 0; j < nr; ++j) {
                    if (rp[j] && pos[j] < stride) {
                        head = std::min(head, rp[j][pos[j]]);
                    }
                }
                if (head == SA_ROW_END) {
                    break;
                }
                const int32_t quad = head >> 2;
                uint32_t mask = 0;
                for (int j = 0; j < nr; ++j) {
                    if (!rp[j]) {
                        continue;
                    }
                    while (pos[j] < stride && rp[j][pos[j]] != SA_ROW_END && (rp[j][pos[j]] >> 2) == quad) {
                        mask |= 1u << (4 * j + (rp[j][pos[j]] & 3));
                        ++pos[j];
                    }
                }
                GGML_ASSERT(n < cap);
                oq[n] = (uint32_t) quad;
                om[n] = (uint16_t) mask;
                ++n;
            }
            const uint32_t last = n > 0 ? oq[n - 1] : 0;
            const int64_t padded = GGML_PAD(n, 4);
            GGML_ASSERT(padded <= cap);
            for (int64_t k = n; k < cap; ++k) {
                oq[k] = last;
                om[k] = 0;
            }
            counts[e] = (uint32_t) padded;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// attend
// ---------------------------------------------------------------------------------------------------------------------

static inline float sa_round_f16(float x) {
    return GGML_CPU_FP16_TO_FP32(GGML_CPU_FP32_TO_FP16(x));
}

// orthonormal Sylvester-Hadamard over consecutive blocks of n values, in the operation order of ggml-cpu's
// GGML_HINT_SRC0_IS_HADAMARD mul_mat (ggml_compute_forward_fwht: scale by 1/sqrt(n) first, then the butterflies
// from len 1 up; its SIMD passes compute u - v as fma(v, -1, u), which is exact, so the values are identical)
static void sa_fwht(float * x, int64_t D, int32_t n) {
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

// K/V row -> its contract values: to_float, rounded to f16 (f16 rows are exact already)
static inline void sa_load_row(const ggml_tensor * t, const char * row, float * out, int64_t D) {
    switch (t->type) {
        case GGML_TYPE_F16:
            for (int64_t d = 0; d < D; ++d) {
                out[d] = GGML_CPU_FP16_TO_FP32(((const ggml_fp16_t *) row)[d]);
            }
            return;
        case GGML_TYPE_F32:
            for (int64_t d = 0; d < D; ++d) {
                out[d] = sa_round_f16(((const float *) row)[d]);
            }
            return;
        default:
            ggml_get_type_traits(t->type)->to_float(row, out, D);
            for (int64_t d = 0; d < D; ++d) {
                out[d] = sa_round_f16(out[d]);
            }
            return;
    }
}

// s[h] = sum_d qT[d][h] * k[d], summed sequentially over d, the R heads in the vector lanes
template <int R>
static inline void sa_scores(const float * qT, const float * k, int64_t D, int /*r*/, float * s) {
    float acc[R];
    for (int h = 0; h < R; ++h) {
        acc[h] = 0.0f;
    }
    for (int64_t d = 0; d < D; ++d) {
        const float kd = k[d];
        const float * qd = qT + d * R;
        for (int h = 0; h < R; ++h) {
            acc[h] += qd[h] * kd;
        }
    }
    for (int h = 0; h < R; ++h) {
        s[h] = acc[h];
    }
}

static inline void sa_scores_any(const float * qT, const float * k, int64_t D, int r, float * s) {
    for (int h = 0; h < r; ++h) {
        s[h] = 0.0f;
    }
    for (int64_t d = 0; d < D; ++d) {
        const float kd = k[d];
        const float * qd = qT + d * r;
        for (int h = 0; h < r; ++h) {
            s[h] += qd[h] * kd;
        }
    }
}

typedef void (*sa_scores_t)(const float * qT, const float * k, int64_t D, int r, float * s);

static sa_scores_t sa_scores_for(int r) {
    switch (r) {
        case 1:  return sa_scores<1>;
        case 2:  return sa_scores<2>;
        case 4:  return sa_scores<4>;
        case 8:  return sa_scores<8>;
        case 12: return sa_scores<12>;
        case 16: return sa_scores<16>;
        default: return sa_scores_any;
    }
}

static size_t sa_attend_thread_bytes(int64_t D, int64_t r) {
    // qT [D][r], acc [r][D], tmp, krow, vrow [D], s, m, l [r]
    return GGML_PAD((size_t) (2 * D * r + 3 * D + 3 * r) * sizeof(float), CACHE_LINE_SIZE);
}

size_t ggml_cpu_sparse_attn_work_size(const struct ggml_tensor * node, int n_tasks) {
    const int64_t D = node->src[0]->ne[0];
    const int64_t r = node->src[0]->ne[2] / node->src[1]->ne[2];
    return (size_t) n_tasks * sa_attend_thread_bytes(D, r) + CACHE_LINE_SIZE;
}

void ggml_compute_forward_sparse_attn(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    // the invariants of ggml_sparse_attn: a node an RPC server deserialized was never built by it
    GGML_ASSERT(ggml_sparse_attn_shape_why(dst) == nullptr);

    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * plan = dst->src[3];

    const int64_t D         = q->ne[0];
    const int64_t n_q       = q->ne[1];
    const int64_t n_head    = q->ne[2];
    const int64_t n_stream  = q->ne[3];
    const int64_t n_head_kv = k->ne[2];
    const int     r         = (int) (n_head / n_head_kv);

    const float   scale = ggml_get_op_params_f32(dst, 0);
    const int32_t rot_q = ggml_get_op_params_i32(dst, 1);
    const int32_t rot_o = ggml_get_op_params_i32(dst, 2);

    GGML_ASSERT(dst->type == GGML_TYPE_F32 && dst->nb[0] == sizeof(float));
    GGML_ASSERT(k->type == GGML_TYPE_F32 || ggml_get_type_traits(k->type)->to_float != nullptr);
    GGML_ASSERT(v->type == GGML_TYPE_F32 || ggml_get_type_traits(v->type)->to_float != nullptr);

    const ggml_sparse_attn_plan_hdr * hdr = (const ggml_sparse_attn_plan_hdr *) plan->data;
    GGML_ASSERT(hdr->magic == GGML_SPARSE_ATTN_PLAN_MAGIC && hdr->version == GGML_SPARSE_ATTN_PLAN_VERSION);
    GGML_ASSERT(hdr->n_q == (uint32_t) n_q && hdr->n_stream == (uint32_t) n_stream);
    const int32_t G = (int32_t) hdr->group;
    GGML_ASSERT(G == ggml_get_op_params_i32(plan, 1));
    const int64_t n_groups = hdr->n_groups;
    const int64_t cap      = hdr->cap;
    const uint32_t * counts = (const uint32_t *) ((const char *) plan->data + hdr->off_counts);
    const uint32_t * quads  = (const uint32_t *) ((const char *) plan->data + hdr->off_quads);
    const uint16_t * masks  = (const uint16_t *) ((const char *) plan->data + hdr->off_masks);

    const int ith = params->ith;
    const int nth = params->nth;

    const size_t per_thread = sa_attend_thread_bytes(D, r);
    GGML_ASSERT(params->wsize >= (size_t) nth * per_thread);
    float * w    = (float *) ((char *) params->wdata + (size_t) ith * per_thread);
    float * qT   = w;                 // [D][r]
    float * acc  = qT + D * r;        // [r][D]
    float * tmp  = acc + D * r;       // [D]
    float * krow = tmp + D;           // [D]
    float * vrow = krow + D;          // [D]
    float * sc   = vrow + D;          // [r]
    float * mx   = sc + r;            // [r]
    float * l    = mx + r;            // [r]

    const sa_scores_t scores = sa_scores_for(r);

    // work items (s, i, kv head), the r query heads of one kv head in one item
    const int64_t n_items = n_stream * n_q * n_head_kv;
    const int64_t t0 = n_items * ith / nth;
    const int64_t t1 = n_items * (ith + 1) / nth;

    for (int64_t t = t0; t < t1; ++t) {
        const int64_t kvh = t % n_head_kv;
        const int64_t i   = (t / n_head_kv) % n_q;
        const int64_t s   = t / (n_head_kv * n_q);

        // 1. q: H_q in f32, x scale in f32, rounded to f16
        for (int hr = 0; hr < r; ++hr) {
            const int64_t h = kvh * r + hr;
            const float * qrow = (const float *) ((const char *) q->data + i * q->nb[1] + h * q->nb[2] + s * q->nb[3]);
            memcpy(tmp, qrow, D * sizeof(float));
            sa_fwht(tmp, D, rot_q);
            for (int64_t d = 0; d < D; ++d) {
                qT[d * r + hr] = sa_round_f16(tmp[d] * scale);
            }
            mx[hr] = -INFINITY;
            l[hr]  = 0.0f;
        }
        memset(acc, 0, (size_t) D * r * sizeof(float));

        // 2. the group's entries that hold cells of this query, in the plan's ascending quad order
        const int64_t g  = i / G;
        const int     jq = (int) (i % G);
        const int64_t e  = s * n_groups + g;
        const uint32_t   n_e = counts[e];
        const uint32_t * eq  = quads + e * cap;
        const uint16_t * em  = masks + e * cap;
        const char * kbase = (const char *) k->data + kvh * k->nb[2] + s * k->nb[3];
        const char * vbase = (const char *) v->data + kvh * v->nb[2] + s * v->nb[3];

        for (uint32_t ei = 0; ei < n_e; ++ei) {
            const uint32_t bits = (em[ei] >> (4 * jq)) & 0xF;
            if (bits == 0) {
                continue;
            }
            for (int c = 0; c < 4; ++c) {
                if (!(bits & (1u << c))) {
                    continue;
                }
                const int64_t cell = 4 * (int64_t) eq[ei] + c;

                // 3. K and V of the cell, dequantized once for the r heads of this kv head
                sa_load_row(k, kbase + cell * k->nb[1], krow, D);
                sa_load_row(v, vbase + cell * v->nb[1], vrow, D);

                // 4. f32 scores, sequential over d
                scores(qT, krow, D, r, sc);

                // online softmax per cell; P rounded to f16 for PV against the running max, l sums the f32 p
                for (int hr = 0; hr < r; ++hr) {
                    float * a = acc + hr * D;
                    if (sc[hr] > mx[hr]) {
                        const float alpha = mx[hr] == -INFINITY ? 0.0f : expf(mx[hr] - sc[hr]);
                        for (int64_t d = 0; d < D; ++d) {
                            a[d] *= alpha;
                        }
                        l[hr] *= alpha;
                        mx[hr] = sc[hr];
                    }
                    const float p  = expf(sc[hr] - mx[hr]);
                    const float ph = sa_round_f16(p);
                    l[hr] += p;
                    // 5. f32 PV
                    for (int64_t d = 0; d < D; ++d) {
                        a[d] += ph * vrow[d];
                    }
                }
            }
        }

        // 5. acc / l, H_o in f32
        for (int hr = 0; hr < r; ++hr) {
            const int64_t h = kvh * r + hr;
            const float * a = acc + hr * D;
            for (int64_t d = 0; d < D; ++d) {
                tmp[d] = l[hr] > 0.0f ? a[d] / l[hr] : 0.0f;
            }
            sa_fwht(tmp, D, rot_o);
            float * out = (float *) ((char *) dst->data + h * dst->nb[1] + i * dst->nb[2] + s * dst->nb[3]);
            memcpy(out, tmp, D * sizeof(float));
        }
    }
}
