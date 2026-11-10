// TE rows of a non-compact QSA ubatch (PLAN-v3 §4.1 row 4, W8 "done when"): the rows src/models/qwen4exp-te.h builds
// must name, for every (query, stream), exactly the cells PR219's masked route attended: top_k(expanded, width) ∩ {mask == 0}.
//
// The layouts are a C++ port of the R1-CB oracle (r1/r1_oracle.py, itself a port of PR219 94e086847's
// llama_memory_hybrid_idx::set_input_qsa_scan grouping and blk_bias rules and of build_qsa_top_k): a unified two-sequence
// cache with a seq_cp prefix of L ∈ {8192, 8193, 8194, 8195} cells and 40 own cells per sequence, and two image layouts
// (sequence 1 holds an M-RoPE image next to text, sequence 0 holds 2000 text cells; image at position 28930 = 2 mod 4 and
// 32779 = 3 mod 4). Split streams (n_stream 2) and several score strips per ubatch are covered too.
//
// Per backend of the build, one graph computes, from the same inputs:
//   new:   per strip, qwen4exp_te_expand + qwen4exp_te_select (the production ops), the strips joined along ne[1];
//   PR219: the masked route's mask from the same top_k (PR219 qwen4exp.cpp build_attn_qsa, kept verbatim here:
//          fill(-inf), set_rows of zeros at the top-k cells, + kq_mask), whose cells at 0 are the attended set.
// Pass: for every row the set {rows >= 0} equals {j : PR219 mask[j] == 0}, rows lists no cell twice, every listed cell is
// one top_k picked, and (host check) PR219's set equals {top_k[i] : kq_mask[top_k[i]] == 0}. Also, per strip, the use
// counts of the graph's expanded chain are the ones the Vulkan TOPK_QSA fusion requires (ggml_vk_can_fuse_topk_qsa:
// GET_ROWS, PERMUTE, CONT, RESHAPE, ADD one use each, the mask CPY two, its own dst included), so the production ops never
// give expanded a second consumer (fusion_uses below).
// usage: test-qsa-te-rows [backend name substring]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "../src/models/qwen4exp-te.h"
#include "../ggml/src/ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

static constexpr int64_t R     = 4;
static constexpr int64_t TOPK  = 2048;
static constexpr int64_t WIDTH = TOPK + R - 1;   // 2051, PR219 build_qsa_top_k

static int g_fail = 0;

// one stream's cells and its queries, with the scan's arrays (r1_oracle.py Cache)
struct layout {
    std::string name;
    std::vector<int64_t>  pos, x, y;
    std::vector<uint32_t> seqs;                   // bit mask of sequence ids
    struct query { int64_t cell; int seq; bool is2d; };
    std::vector<query> queries;

    // filled by finish()
    int64_t n_kv = 0, n_blocks = 0, n_bid = 0, dead = 0;
    std::vector<int32_t>  cell_blk;
    std::vector<int64_t>  bid_idx;
    std::vector<uint32_t> bid_set;

    int64_t add(int64_t p, uint32_t s, int64_t xx = 0, int64_t yy = 0) {
        pos.push_back(p); seqs.push_back(s); x.push_back(xx); y.push_back(yy);
        return (int64_t) pos.size() - 1;
    }

    // groups keyed by (position block, exact sequence set), complete groups only get a block id, in block order; every
    // other cell maps to the spare ("dead") block (set_input_qsa_scan group_cells with two sequences present)
    void finish(int64_t n_kv_view) {
        const int64_t n = (int64_t) pos.size();
        n_kv = n_kv_view;
        GGML_ASSERT(n <= n_kv);
        n_blocks = n_kv / R + 1;
        std::map<std::pair<int64_t, uint32_t>, std::vector<int64_t>> groups;
        for (int64_t j = 0; j < n; ++j) {
            groups[{ pos[j] / R, seqs[j] }].push_back(j);
        }
        cell_blk.assign(n_kv, -1);
        for (const auto & g : groups) {   // std::map: ascending position block
            std::set<int64_t> slots;
            for (int64_t j : g.second) {
                slots.insert(pos[j] % R);
            }
            if ((int64_t) slots.size() != R) {
                continue;
            }
            const int32_t b = (int32_t) bid_idx.size();
            bid_idx.push_back(g.first.first * R);
            bid_set.push_back(g.first.second);
            for (int64_t j : g.second) {
                cell_blk[j] = b;
            }
        }
        n_bid = (int64_t) bid_idx.size();
        GGML_ASSERT(n_bid < n_blocks);
        dead = n_bid;
        for (auto & c : cell_blk) {
            if (c < 0) {
                c = (int32_t) dead;
            }
        }
    }

    // llama_kv_cache::set_input_kq_mask for one query: same sequence, earlier position, or the same position and, for an
    // image query, not is_2d_gt
    bool visible(const query & q, int64_t j) const {
        if (j >= (int64_t) pos.size() || !(seqs[j] & (1u << q.seq))) {
            return false;
        }
        const int64_t p = pos[q.cell];
        if (pos[j] != p) {
            return pos[j] < p;
        }
        if (!q.is2d) {
            return true;
        }
        const bool gt = y[j] > y[q.cell] || (y[j] == y[q.cell] && x[j] > x[q.cell]);
        return !gt;
    }

    // the blk_bias branch of set_input_qsa_scan: -inf for a block of another sequence set (and past the complete blocks),
    // 1e9 at and after the query's tail start and for the dead block, 0 otherwise
    float bias(const query & q, int64_t b) const {
        if (b == dead) {
            return 1e9f;
        }
        if (b >= n_bid || !(bid_set[b] & (1u << q.seq))) {
            return -INFINITY;
        }
        const int64_t tail_start = (pos[q.cell] + 1) / R * R;
        return bid_idx[b] >= tail_start ? 1e9f : 0.0f;
    }
};

static int64_t pad256(int64_t n) { return (n + 255) / 256 * 256; }

// r1_oracle.py text_case: a seq_cp prefix of L cells shared by sequences 0 and 1, then 40 own cells each; queries = own cells
static layout text_case(int64_t L) {
    layout c;
    c.name = "text_L" + std::to_string(L);
    for (int64_t p = 0; p < L; ++p) {
        c.add(p, 3u);
    }
    for (int s = 0; s < 2; ++s) {
        for (int64_t p = L; p < L + 40; ++p) {
            c.queries.push_back({ c.add(p, 1u << s), s, false });
        }
    }
    return c;
}

// r1_oracle.py image_case: sequence 0 holds n0 text cells; sequence 1 holds n_before text cells, an image of 1036 cells at
// position n_before (grid gh x gw, ext x/y as llama_kv_cache::apply_ubatch stores them), then n_after text cells.
// Queries: every 3rd image cell, every 3rd text cell after the image, and the last 3 cells of sequence 0 (2000 visible
// cells < 2051: top_k must fill the row with hidden cells, which the rows must drop)
static layout image_case(int64_t n_before, int64_t n_after, int64_t gh = 28, int64_t gw = 37, int64_t n0 = 2000) {
    layout c;
    c.name = "image_pos" + std::to_string(n_before);
    std::vector<int64_t> seq0;
    for (int64_t p = 0; p < n0; ++p) {
        seq0.push_back(c.add(p, 1u));
    }
    for (int64_t p = 0; p < n_before; ++p) {
        c.add(p, 2u);
    }
    const int64_t t = n_before;
    std::vector<int64_t> img;
    for (int64_t yy = 0; yy < gh && (int64_t) img.size() < 1036; ++yy) {
        for (int64_t xx = 0; xx < gw && (int64_t) img.size() < 1036; ++xx) {
            img.push_back(c.add(t, 2u, t + xx, t + yy));
        }
    }
    const int64_t p0 = t + std::max(gh, gw);
    std::vector<int64_t> txt;
    for (int64_t p = p0; p < p0 + n_after; ++p) {
        txt.push_back(c.add(p, 2u));
    }
    for (size_t i = 0; i < img.size(); i += 3) {
        c.queries.push_back({ img[i], 1, true });
    }
    for (size_t i = 0; i < txt.size(); i += 3) {
        c.queries.push_back({ txt[i], 1, false });
    }
    for (int64_t k = 3; k >= 1; --k) {
        c.queries.push_back({ seq0[n0 - k], 0, false });
    }
    return c;
}

struct case_t {
    std::string name;
    std::vector<layout> streams;   // one layout per stream, all with the same query count
    int64_t strip;                 // queries per score strip
};

struct stats { int64_t rows = 0, mism = 0, dup = 0, not_picked = 0, host_mism = 0, underfull = 0, dropped = 0, bias_ninf_vis = 0; };

// The use counts ggml_vk_can_fuse_topk_qsa requires of the nodes it replaces (ggml-vulkan.cpp: every node before the TOP_K
// single-use, the CPY two uses because it counts its own dst): GET_ROWS -> PERMUTE -> CONT -> ADD <- RESHAPE <- CPY, the
// ADD feeding the TOP_K. "" when every strip's chain has them, else the first offender.
static std::string fusion_uses(ggml_cgraph * gf, const std::vector<ggml_tensor *> & expanded) {
    std::map<const ggml_tensor *, int> idx;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        idx[ggml_graph_node(gf, i)] = i;
    }
    for (ggml_tensor * add : expanded) {
        ggml_tensor * cont    = add->src[0];
        ggml_tensor * permute = cont ? cont->src[0] : nullptr;
        ggml_tensor * gr      = permute ? permute->src[0] : nullptr;
        ggml_tensor * reshape = add->src[1];
        ggml_tensor * cpy     = reshape ? reshape->src[0] : nullptr;
        const struct { ggml_tensor * t; ggml_op op; int32_t want; const char * what; } chain[] = {
            { gr, GGML_OP_GET_ROWS, 1, "GET_ROWS" }, { permute, GGML_OP_PERMUTE, 1, "PERMUTE" }, { cont, GGML_OP_CONT, 1, "CONT" },
            { cpy, GGML_OP_CPY, 2, "CPY (mask cast)" }, { reshape, GGML_OP_RESHAPE, 1, "RESHAPE" }, { add, GGML_OP_ADD, 1, "ADD (expanded)" },
        };
        for (const auto & e : chain) {
            if (e.t == nullptr || e.t->op != e.op || idx.count(e.t) == 0) {
                return std::string(e.what) + " is not in the graph where the pattern expects it";
            }
            const int32_t n = ggml_node_get_use_count(gf, idx[e.t]);
            if (n != e.want) {
                return std::string(e.what) + " has " + std::to_string(n) + " uses, the fusion needs " + std::to_string(e.want);
            }
        }
    }
    return "";
}

static void run_case(ggml_backend_t be, const char * bname, case_t & c) {
    const int64_t n_stream = (int64_t) c.streams.size();
    const int64_t n_tps = (int64_t) c.streams[0].queries.size();
    int64_t n_kv = 0;
    for (auto & l : c.streams) {
        GGML_ASSERT((int64_t) l.queries.size() == n_tps);
        n_kv = std::max(n_kv, pad256((int64_t) l.pos.size()));
    }
    for (auto & l : c.streams) {
        l.finish(n_kv);
    }
    const int64_t n_blocks = c.streams[0].n_blocks;
    const int64_t width = std::min<int64_t>(n_kv, WIDTH);

    // host inputs: block scores (relu'd index scores, so >= 0), per-block bias, KQ mask (f16, as flash attention keeps it),
    // cell_blk
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> uni(0.0f, 50.0f);
    std::vector<float> score((size_t) n_blocks * n_tps * n_stream), bias(score.size());
    std::vector<ggml_fp16_t> mask((size_t) n_kv * n_tps * n_stream);
    std::vector<int32_t> cell_blk((size_t) n_kv * n_stream);
    stats st;
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int64_t s = 0; s < n_stream; ++s) {
        const layout & l = c.streams[s];
        std::copy(l.cell_blk.begin(), l.cell_blk.end(), cell_blk.begin() + s * n_kv);
        for (int64_t q = 0; q < n_tps; ++q) {
            const auto & qq = l.queries[q];
            int64_t n_vis = 0;
            for (int64_t b = 0; b < n_blocks; ++b) {
                const size_t e = (size_t) ((s * n_tps + q) * n_blocks + b);
                score[e] = uni(rng);
                bias[e]  = l.bias(qq, b);
            }
            for (int64_t j = 0; j < n_kv; ++j) {
                const bool v = l.visible(qq, j);
                mask[(size_t) ((s * n_tps + q) * n_kv + j)] = v ? z : ninf;
                n_vis += v;
                // the precondition of qwen4exp-te.h: a visible cell's block bias is finite
                st.bias_ninf_vis += v && std::isinf(bias[(size_t) ((s * n_tps + q) * n_blocks + l.cell_blk[j])]);
            }
            st.underfull += n_vis < width;
        }
    }

    ggml_init_params ip = { 4096 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8192, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t_score = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_blocks, n_tps, n_stream);
    ggml_tensor * t_bias  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_blocks, n_tps, n_stream);
    ggml_tensor * t_mask  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_tps, 1, n_stream);
    ggml_tensor * t_cblk  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, n_stream);
    for (ggml_tensor * t : { t_score, t_bias, t_mask, t_cblk }) {
        ggml_set_input(t);
    }

    // the strip loop of build_qsa_top_k (non-compact, blk_bias): strip views of the bias and the mask, the score biased per
    // block, then the production ops of qwen4exp-te.h
    ggml_tensor * biased = ggml_add(ctx, t_score, t_bias);
    std::vector<ggml_tensor *> rows_s, topk_s, expanded_s;
    for (int64_t first = 0; first < n_tps; first += c.strip) {
        const int64_t n_query = std::min(c.strip, n_tps - first);
        ggml_tensor * sc = ggml_view_3d(ctx, biased, n_blocks, n_query, n_stream, biased->nb[1], biased->nb[2], first * biased->nb[1]);
        sc = ggml_cont(ctx, sc);
        ggml_tensor * qm = ggml_view_4d(ctx, t_mask, n_kv, n_query, 1, n_stream, t_mask->nb[1], t_mask->nb[2], t_mask->nb[3],
                first * t_mask->nb[1]);
        ggml_tensor * add = ggml_reshape_3d(ctx, ggml_cast(ctx, qm, GGML_TYPE_F32), n_kv, n_query, n_stream);
        ggml_tensor * expanded = qwen4exp_te_expand(ctx, sc, t_cblk, add);
        const qwen4exp_te_sel sel = qwen4exp_te_select(ctx, expanded, qm, width);
        expanded_s.push_back(expanded);
        rows_s.push_back(sel.rows);
        topk_s.push_back(sel.top_k);
    }
    ggml_tensor * rows = rows_s[0];
    ggml_tensor * top_k = topk_s[0];
    for (size_t i = 1; i < rows_s.size(); ++i) {
        rows  = ggml_concat(ctx, rows, rows_s[i], 1);
        top_k = ggml_concat(ctx, top_k, topk_s[i], 1);
    }

    // PR219 94e086847 qwen4exp.cpp build_attn_qsa, the masked half (deleted from the model in W8), on the same top_k
    ggml_tensor * kq_mask_all = ggml_fill(ctx, t_mask, -INFINITY);
    kq_mask_all = ggml_view_4d(ctx, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
            kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);
    ggml_tensor * top_k_3d = ggml_view_4d(ctx, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2],
            top_k->ne[3] * top_k->nb[3], 0);
    ggml_tensor * zeros = ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]), 0.0f);
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx, kq_mask_all, zeros, top_k_3d);
    kq_mask_top_k = ggml_view_4d(ctx, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3],
            kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);
    kq_mask_top_k = ggml_add(ctx, kq_mask_top_k, t_mask);
    ggml_tensor * pr219 = ggml_cast(ctx, kq_mask_top_k, GGML_TYPE_F32);
    ggml_tensor * top_k_out = ggml_cont(ctx, top_k);

    ggml_set_output(rows);
    ggml_set_output(pr219);
    ggml_set_output(top_k_out);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, rows);
    ggml_build_forward_expand(gf, pr219);
    ggml_build_forward_expand(gf, top_k_out);
    const std::string uses = fusion_uses(gf, expanded_s);
    if (!uses.empty()) {
        printf("  [%s %s] the expanded chain cannot fuse: %s: FAIL\n", bname, c.name.c_str(), uses.c_str());
        g_fail++;
    }
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(be, node)) {
            printf("  [%s %s] unsupported op %s (%s): FAIL\n", bname, c.name.c_str(), ggml_op_desc(node), ggml_type_name(node->type));
            g_fail++;
            ggml_free(ctx);
            return;
        }
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(ga, gf)) {
        printf("  [%s %s] graph allocation failed: FAIL\n", bname, c.name.c_str());
        g_fail++;
        ggml_gallocr_free(ga);
        ggml_free(ctx);
        return;
    }
    ggml_backend_tensor_set(t_score, score.data(), 0, score.size() * sizeof(float));
    ggml_backend_tensor_set(t_bias,  bias.data(),  0, bias.size()  * sizeof(float));
    ggml_backend_tensor_set(t_mask,  mask.data(),  0, mask.size()  * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(t_cblk,  cell_blk.data(), 0, cell_blk.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        printf("  [%s %s] compute failed: FAIL\n", bname, c.name.c_str());
        g_fail++;
        ggml_gallocr_free(ga);
        ggml_free(ctx);
        return;
    }

    std::vector<int32_t> h_rows(ggml_nelements(rows)), h_topk(ggml_nelements(top_k_out));
    std::vector<float>   h_pr(ggml_nelements(pr219));
    ggml_backend_tensor_get(rows, h_rows.data(), 0, h_rows.size() * sizeof(int32_t));
    ggml_backend_tensor_get(top_k_out, h_topk.data(), 0, h_topk.size() * sizeof(int32_t));
    ggml_backend_tensor_get(pr219, h_pr.data(), 0, h_pr.size() * sizeof(float));

    const bool shape_ok = rows->type == GGML_TYPE_I32 && rows->ne[0] == width && rows->ne[1] == n_tps && rows->ne[2] == 1 &&
        rows->ne[3] == n_stream;
    for (int64_t s = 0; s < n_stream; ++s) {
        for (int64_t q = 0; q < n_tps; ++q) {
            const size_t r0 = (size_t) ((s * n_tps + q) * width);
            std::set<int32_t> picked(h_topk.begin() + r0, h_topk.begin() + r0 + width);
            std::set<int32_t> got, want, host;
            for (int64_t i = 0; i < width; ++i) {
                const int32_t v = h_rows[r0 + i];
                if (v < 0) {
                    st.dropped += 1;
                    continue;
                }
                st.dup += !got.insert(v).second;
                st.not_picked += picked.count(v) == 0;
            }
            for (int64_t j = 0; j < n_kv; ++j) {
                if (h_pr[(size_t) ((s * n_tps + q) * n_kv + j)] == 0.0f) {
                    want.insert((int32_t) j);
                }
            }
            for (int32_t j : picked) {
                if (j >= 0 && j < n_kv && mask[(size_t) ((s * n_tps + q) * n_kv + j)] == z) {
                    host.insert(j);
                }
            }
            st.rows += 1;
            st.mism += got != want;
            st.host_mism += want != host;
        }
    }
    printf("  [%s %s] n_kv=%lld n_tps=%lld n_stream=%lld strips=%zu ne=[%lld,%lld,%lld,%lld] rows=%lld mismatch=%lld host_mismatch=%lld "
           "dup=%lld not_picked=%lld underfull_rows=%lld dropped=%lld visible_with_ninf_bias=%lld shape=%d\n",
           bname, c.name.c_str(), (long long) n_kv, (long long) n_tps, (long long) n_stream, rows_s.size(),
           (long long) rows->ne[0], (long long) rows->ne[1], (long long) rows->ne[2], (long long) rows->ne[3],
           (long long) st.rows, (long long) st.mism, (long long) st.host_mism, (long long) st.dup, (long long) st.not_picked,
           (long long) st.underfull, (long long) st.dropped, (long long) st.bias_ninf_vis, (int) shape_ok);
    if (st.mism || st.host_mism || st.dup || st.not_picked || st.bias_ninf_vis || !shape_ok || st.rows != n_tps * n_stream) {
        g_fail++;
    }
    ggml_gallocr_free(ga);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    const char * only = argc > 1 ? argv[1] : nullptr;
    ggml_backend_load_all();
    struct be_t { ggml_backend_t be; std::string name; };
    std::vector<be_t> bes;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
        if (only && strstr(ggml_backend_dev_name(d), only) == nullptr) {
            continue;
        }
        ggml_backend_t b = ggml_backend_dev_init(d, nullptr);
        if (b != nullptr) {
            bes.push_back({ b, ggml_backend_dev_name(d) });
        }
    }
    if (bes.empty()) {
        printf("no backend\n");
        return 2;
    }
    for (const auto & b : bes) {
        printf("backend %s\n", b.name.c_str());
        std::vector<case_t> cases;
        for (int64_t L : { 8192, 8193, 8194, 8195 }) {
            cases.push_back({ "text_L" + std::to_string(L), { text_case(L) }, 80 });
            cases.push_back({ "text_L" + std::to_string(L) + "_strips32", { text_case(L) }, 32 });
        }
        cases.push_back({ "image_pos28930", { image_case(28930, 1328) }, 1024 });
        cases.push_back({ "image_pos32779_strips256", { image_case(32779, 1573) }, 256 });
        cases.push_back({ "split2_text_L8193_L8195_strips32", { text_case(8193), text_case(8195) }, 32 });
        for (auto & c : cases) {
            run_case(b.be, b.name.c_str(), c);
        }
        ggml_backend_free(b.be);
    }
    printf("RESULT %s (failures %d)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
