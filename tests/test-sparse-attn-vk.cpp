// Vulkan checks of GGML_OP_SPARSE_ATTN_PLAN / GGML_OP_SPARSE_ATTN that test-backend-ops cannot express (PLAN-v3 §13 W6):
//
//   limits      per tensor, round_up(nbytes) + workspace < the buffer type's maximum and each binding <= the device's binding
//               limit, at the reserve of -c 1048576 (n_kv 2^20), ub 512 / 4096 / 8192, 12 x 2051 keys;
//   capability  supports_op is false, before any dispatch, when the pipeline does not exist (GGML_VK_SA_CAP_MASK), and the tier order
//               follows what is left; a forced tier the device cannot create is unsupported, not a null-pipeline dispatch;
//   rot         H_q in every kernel (the rot_dump hook) and H_o bitwise equal to fwht.comp (a GGML_HINT_SRC0_IS_HADAMARD mul_mat on
//               the same backend) on the same input, rot 64 / 128 / 256, for every tier the device has;
//   stress      gallocr places the workspace of a sparse node over a freed tensor that the preceding dispatch still reads, and only
//               the workspace overlaps it (dst does not, so the barrier tracker sees nothing; the sync rule is all that orders them).
//               The test asserts that layout from tensor->data and get_alloc_size and FAILS when it does not hold; then the outputs
//               must be memcmp-stable over 20 iterations and bitwise equal to the same graph with the tensor kept alive.
//
// Without a Vulkan backend it exits 0 ("skipped"). usage: test-sparse-attn-vk [limits|capability|rot|stress ...]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

static ggml_backend_reg_t g_reg = nullptr;
static ggml_backend_dev_t g_dev = nullptr;
static ggml_backend_t     g_backend = nullptr;
static ggml_backend_buffer_type_t g_buft = nullptr;
static int g_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); ++g_fail; } } while (0)

static void set_force(const char * force, const char * mask = nullptr) {
    if (force) { setenv("GGML_VK_SA_FORCE", force, 1); } else { unsetenv("GGML_VK_SA_FORCE"); }
    if (mask)  { setenv("GGML_VK_SA_CAP_MASK", mask, 1); } else { unsetenv("GGML_VK_SA_CAP_MASK"); }
    auto reload = (void (*)(void)) ggml_backend_reg_get_proc_address(g_reg, "ggml_backend_debug_env_reload");
    GGML_ASSERT(reload);
    reload();
}

struct sa_shape {
    ggml_type tk = GGML_TYPE_F16, tv = GGML_TYPE_F16;
    int64_t n_q = 65, n_kv = 4096, n_sel = 257, kv_head = 2, gqa = 12, n_stream = 1;
    int rot_q = 0, rot_o = 0;
    float scale = 0.0625f;
};

struct sa_graph {
    ggml_tensor * q_st, * q, * k_st, * k, * v_st, * v, * sel, * plan, * out;
};

static sa_graph sa_build(ggml_context * ctx, const sa_shape & s) {
    sa_graph g;
    g.q_st = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, s.kv_head * s.gqa, s.n_q, s.n_stream);
    g.q    = ggml_permute(ctx, g.q_st, 0, 2, 1, 3);
    g.k_st = ggml_new_tensor_4d(ctx, s.tk, 256, s.kv_head, s.n_kv, s.n_stream);
    g.k    = ggml_permute(ctx, g.k_st, 0, 2, 1, 3);
    g.v_st = ggml_new_tensor_4d(ctx, s.tv, 256, s.kv_head, s.n_kv, s.n_stream);
    g.v    = ggml_permute(ctx, g.v_st, 0, 2, 1, 3);
    g.sel  = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, s.n_sel, s.n_q, 1, s.n_stream);
    g.plan = ggml_sparse_attn_plan(ctx, g.sel, s.n_kv, 1);
    g.out  = ggml_sparse_attn(ctx, g.q, g.k, g.v, g.plan, s.scale, s.rot_q, s.rot_o);
    ggml_set_input(g.q_st);
    ggml_set_input(g.k_st);
    ggml_set_input(g.v_st);
    ggml_set_input(g.sel);
    ggml_set_output(g.out);
    // ggml-alloc frees a leaf after its last consumer unless it is an output: the inputs must survive for the next iteration
    ggml_set_output(g.q_st);
    ggml_set_output(g.k_st);
    ggml_set_output(g.v_st);
    ggml_set_output(g.sel);
    return g;
}

static void fill_kv(ggml_tensor * t, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const size_t rows = ggml_nelements(t) / 256;
    std::vector<float> f(rows * 256);
    for (auto & x : f) x = dist(rng);
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) bytes.data(), f.size());
    } else {
        ggml_quantize_chunk(t->type, f.data(), bytes.data(), 0, rows, 256, nullptr);
    }
    ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
}

static void fill_inputs(const sa_graph & g, const sa_shape & s, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> q(ggml_nelements(g.q_st));
    for (auto & x : q) x = dist(rng);
    ggml_backend_tensor_set(g.q_st, q.data(), 0, ggml_nbytes(g.q_st));
    fill_kv(g.k_st, rng);
    fill_kv(g.v_st, rng);
    std::vector<int32_t> rows(ggml_nelements(g.sel), -1);
    for (int64_t r = 0; r < s.n_stream * s.n_q; ++r) {
        std::set<int32_t> cells;
        while ((int64_t) cells.size() < std::min(s.n_sel, s.n_kv)) {
            cells.insert((int32_t) (rng() % (uint32_t) s.n_kv));
        }
        int64_t j = 0;
        for (int32_t c : cells) {
            rows[(size_t) (r * s.n_sel + j++)] = c;
        }
        std::shuffle(rows.begin() + (size_t) (r * s.n_sel), rows.begin() + (size_t) ((r + 1) * s.n_sel), rng);
    }
    ggml_backend_tensor_set(g.sel, rows.data(), 0, ggml_nbytes(g.sel));
}

static std::vector<uint8_t> get_bytes(const ggml_tensor * t) {
    std::vector<uint8_t> b(ggml_nbytes(t));
    ggml_backend_tensor_get(t, b.data(), 0, b.size());
    return b;
}

static ggml_context * new_ctx(size_t mem = 16 << 20) {
    ggml_init_params p = { mem, nullptr, true };
    return ggml_init(p);
}

// ---- limits -----------------------------------------------------------------------------------------------------------

static void test_limits() {
    printf("limits: the -c 1048576 reserve\n");
    set_force(nullptr);
    auto limit_fn = (size_t (*)(ggml_backend_t)) ggml_backend_reg_get_proc_address(g_reg, "ggml_backend_debug_vk_sa_binding_limit");
    GGML_ASSERT(limit_fn);
    const size_t limit = limit_fn(g_backend);
    const size_t max_size = ggml_backend_buft_get_max_size(g_buft);
    const size_t align = ggml_backend_buft_get_alignment(g_buft);
    printf("  binding limit %zu, buffer type maximum %zu, alignment %zu\n", limit, max_size, align);
    for (ggml_type t : { GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 }) {
        for (int64_t ub : { 512, 4096, 8192 }) {
            ggml_context * ctx = new_ctx();
            sa_shape s;
            s.tk = s.tv = t;
            s.n_q = ub;
            s.n_kv = 1048576;
            s.n_sel = 2051;
            const sa_graph g = sa_build(ctx, s);
            const size_t k_bytes = ggml_nbytes(g.k), v_bytes = ggml_nbytes(g.v);
            const bool binds = k_bytes <= limit && v_bytes <= limit && ggml_nbytes(g.out) <= limit && ggml_nbytes(g.q) <= limit;
            const bool sup_plan = ggml_backend_dev_supports_op(g_dev, g.plan);
            const bool sup_attn = ggml_backend_dev_supports_op(g_dev, g.out);
            const size_t a_plan = ggml_backend_buft_get_alloc_size(g_buft, g.plan);
            const size_t a_attn = ggml_backend_buft_get_alloc_size(g_buft, g.out);
            printf("  %-5s ub %-5" PRId64 ": K/V view %zu B; plan dst %zu alloc %zu; attend dst %zu alloc %zu; supports %d %d\n",
                   ggml_type_name(t), ub, k_bytes, ggml_nbytes(g.plan), a_plan, ggml_nbytes(g.out), a_attn, sup_plan, sup_attn);
            CHECK(a_plan < max_size, "plan alloc %zu >= block %zu", a_plan, max_size);
            CHECK(a_attn < max_size, "attend alloc %zu >= block %zu", a_attn, max_size);
            CHECK(a_plan - ggml_nbytes(g.plan) <= limit && ggml_nbytes(g.plan) <= limit, "plan binding past %zu", limit);
            CHECK(sup_plan, "plan not supported");
            // a device with a smaller binding limit (lavapipe 128 MiB) must reject the shape, never bind past it
            CHECK(sup_attn == binds, "attend supported %d, its bindings fit %d", sup_attn, binds);
            CHECK(!sup_attn || (a_attn - ggml_nbytes(g.out) <= limit && ggml_nbytes(g.out) <= limit), "attend binding past %zu", limit);
            ggml_free(ctx);
        }
    }
}

// ---- capability -------------------------------------------------------------------------------------------------------

static void test_capability() {
    printf("capability: no pipeline, no support\n");
    ggml_context * ctx = new_ctx();
    sa_shape s;
    const sa_graph g = sa_build(ctx, s);

    set_force(nullptr);
    const bool sup_plan = ggml_backend_dev_supports_op(g_dev, g.plan), sup_attn = ggml_backend_dev_supports_op(g_dev, g.out);
    printf("  default: plan %d attend %d\n", sup_plan, sup_attn);
    CHECK(sup_plan && sup_attn, "the device must run the default tiers");

    set_force(nullptr, "15");
    CHECK(!ggml_backend_dev_supports_op(g_dev, g.plan), "plan supported with no plan pipeline");
    CHECK(!ggml_backend_dev_supports_op(g_dev, g.out), "attend supported with no pipeline at all");
    printf("  mask 15 (no pipeline): plan %d attend %d\n", ggml_backend_dev_supports_op(g_dev, g.plan), ggml_backend_dev_supports_op(g_dev, g.out));

    set_force(nullptr, "8");
    CHECK(!ggml_backend_dev_supports_op(g_dev, g.plan), "plan supported with no plan pipeline");
    CHECK(!ggml_backend_dev_supports_op(g_dev, g.out), "attend supported with no plan pipeline");

    set_force(nullptr, "7");
    CHECK(ggml_backend_dev_supports_op(g_dev, g.plan), "the plan has its own gate");
    CHECK(!ggml_backend_dev_supports_op(g_dev, g.out), "attend supported with no attend pipeline");

    set_force(nullptr, "3");
    CHECK(ggml_backend_dev_supports_op(g_dev, g.out), "the portable tier is left");

    // a forced tier whose pipeline is masked away
    set_force("portable", "4");
    CHECK(!ggml_backend_dev_supports_op(g_dev, g.out), "forced portable with no portable pipeline");
    set_force("scalar", "2");
    CHECK(!ggml_backend_dev_supports_op(g_dev, g.out), "forced scalar with no scalar pipeline");

    // shapes no tier takes
    set_force(nullptr);
    {
        ggml_context * ctx2 = new_ctx();
        sa_shape s2;
        s2.gqa = 3;   // GQA not in {4, 8, 12}
        const sa_graph g2 = sa_build(ctx2, s2);
        CHECK(!ggml_backend_dev_supports_op(g_dev, g2.out), "GQA 3 supported");
        ggml_tensor * sel = ggml_new_tensor_4d(ctx2, GGML_TYPE_I32, 257, 8, 1, 1);
        ggml_tensor * p4 = ggml_sparse_attn_plan(ctx2, sel, 4096, 4);
        CHECK(!ggml_backend_dev_supports_op(g_dev, p4), "a G = 4 plan supported");
        ggml_free(ctx2);
    }
    ggml_free(ctx);
    set_force(nullptr);
}

// ---- graph helper -----------------------------------------------------------------------------------------------------

struct run_ctx {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_cgraph * gf = nullptr;
    run_ctx() {
        ctx = new_ctx(64 << 20);
        galloc = ggml_gallocr_new(g_buft);
        gf = ggml_new_graph_custom(ctx, 256, false);
    }
    ~run_ctx() {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
    }
};

// ---- rot --------------------------------------------------------------------------------------------------------------

static ggml_tensor * hadamard(ggml_context * ctx, ggml_tensor * x, int r, ggml_tensor ** h_out) {
    ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, r);
    ggml_set_input(h);
    *h_out = h;
    ggml_tensor * rows = ggml_is_contiguous(x) ? ggml_reshape_2d(ctx, x, r, ggml_nelements(x) / r) : ggml_cont_2d(ctx, x, r, ggml_nelements(x) / r);
    ggml_tensor * res = ggml_mul_mat(ctx, h, rows);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    ggml_set_output(res);
    return res;
}

static void set_hadamard(ggml_tensor * h, int r) {
    std::vector<float> m((size_t) r * r);
    const float sc = 1.0f / std::sqrt((float) r);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < r; ++j) {
            m[(size_t) i * r + j] = (__builtin_popcount(i & j) & 1 ? -1.0f : 1.0f) * sc;
        }
    }
    ggml_backend_tensor_set(h, m.data(), 0, ggml_nbytes(h));
}

static void test_rot() {
    printf("rot: H_q and H_o against fwht.comp\n");
    struct tier { const char * name; const char * force; };
    const std::vector<tier> tiers = { {"default", nullptr}, {"scalar", "scalar"}, {"portable", "portable"} };
    for (const tier & t : tiers) {
        set_force(t.force);
        {
            ggml_context * c = new_ctx();
            sa_shape s;
            const sa_graph g = sa_build(c, s);
            const bool sup = ggml_backend_dev_supports_op(g_dev, g.out);
            ggml_free(c);
            if (!sup) {
                printf("  tier %s: not on this device\n", t.name);
                continue;
            }
        }
        for (int r : { 64, 128, 256 }) {
            for (int64_t n_q : { 3, 65 }) {
                for (ggml_type ty : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
                    sa_shape s;
                    s.tk = s.tv = ty;
                    s.n_q = n_q;
                    s.n_sel = 257;
                    // H_q: the kernel's q transform, dumped
                    set_force((std::string(t.force ? std::string(t.force) + "," : std::string()) + "rot_dump").c_str());
                    std::vector<uint8_t> dump, ref;
                    {
                        run_ctx rc;
                        s.rot_q = r;
                        s.rot_o = 0;
                        const sa_graph g = sa_build(rc.ctx, s);
                        ggml_build_forward_expand(rc.gf, g.out);
                        GGML_ASSERT(ggml_gallocr_alloc_graph(rc.galloc, rc.gf));
                        fill_inputs(g, s, 7);
                        GGML_ASSERT(ggml_backend_graph_compute(g_backend, rc.gf) == GGML_STATUS_SUCCESS);
                        dump = get_bytes(g.out);
                    }
                    set_force(t.force);
                    {
                        run_ctx rc;
                        s.rot_q = 0;
                        const sa_graph g = sa_build(rc.ctx, s);
                        ggml_tensor * h = nullptr;
                        ggml_tensor * res = hadamard(rc.ctx, g.q_st, r, &h);
                        ggml_build_forward_expand(rc.gf, g.out);   // keeps K, V and sel allocated
                        ggml_build_forward_expand(rc.gf, res);
                        GGML_ASSERT(ggml_gallocr_alloc_graph(rc.galloc, rc.gf));
                        fill_inputs(g, s, 7);
                        set_hadamard(h, r);
                        GGML_ASSERT(ggml_backend_graph_compute(g_backend, rc.gf) == GGML_STATUS_SUCCESS);
                        ref = get_bytes(res);
                    }
                    CHECK(dump == ref, "tier %s: H_q rot %d n_q %d %s differs from fwht.comp", t.name, r, (int) n_q, ggml_type_name(ty));
                    // H_o: the op against its rot 0 output through fwht.comp
                    std::vector<uint8_t> a, b;
                    {
                        run_ctx rc;
                        s.rot_o = r;
                        const sa_graph g = sa_build(rc.ctx, s);
                        ggml_build_forward_expand(rc.gf, g.out);
                        GGML_ASSERT(ggml_gallocr_alloc_graph(rc.galloc, rc.gf));
                        fill_inputs(g, s, 11);
                        GGML_ASSERT(ggml_backend_graph_compute(g_backend, rc.gf) == GGML_STATUS_SUCCESS);
                        a = get_bytes(g.out);
                    }
                    {
                        run_ctx rc;
                        s.rot_o = 0;
                        const sa_graph g = sa_build(rc.ctx, s);
                        ggml_tensor * h = nullptr;
                        ggml_tensor * res = hadamard(rc.ctx, g.out, r, &h);
                        ggml_build_forward_expand(rc.gf, res);
                        GGML_ASSERT(ggml_gallocr_alloc_graph(rc.galloc, rc.gf));
                        fill_inputs(g, s, 11);
                        set_hadamard(h, r);
                        GGML_ASSERT(ggml_backend_graph_compute(g_backend, rc.gf) == GGML_STATUS_SUCCESS);
                        b = get_bytes(res);
                    }
                    CHECK(a == b, "tier %s: H_o rot %d n_q %d %s differs from fwht.comp", t.name, r, (int) n_q, ggml_type_name(ty));
                    printf("  tier %-8s rot %-3d n_q %-2d %-4s: H_q %s, H_o %s\n", t.name, r, (int) n_q, ggml_type_name(ty), dump == ref ? "bitwise" : "DIFFERS", a == b ? "bitwise" : "DIFFERS");
                }
            }
        }
    }
    set_force(nullptr);
}

// ---- stress -----------------------------------------------------------------------------------------------------------

// Node order: [plan] S F T S2(S) F2(F) F3(F2) R(T) <node under test> ... so that (best fit, ggml-alloc.c): S is a small block that
//   attracts the small outputs; S, F and T are neighbours, freed by S2, F2 and R; F3 reads F2, so the tracker places a barrier there and the
//   reads of S and F are forgotten; R reads T right before the node. The node's dst lands on what is left of S and F, and ends before T: its
//   workspace lies inside T and no range the tracker knows overlaps dst. F holds dst's bytes (rounded up) so that this is possible.
struct stress_out { std::vector<uint8_t> attend, r; bool layout = false; };

static stress_out stress_run(const sa_shape & s, const bool under_test_is_plan, const bool keep_t, const int iters, int & n_unstable) {
    stress_out o;
    run_ctx rc;
    ggml_context * ctx = rc.ctx;
    sa_graph g = sa_build(ctx, s);
    const size_t align = ggml_backend_buft_get_alignment(g_buft);
    ggml_tensor * node = under_test_is_plan ? g.plan : g.out;
    const size_t node_alloc = ggml_backend_buft_get_alloc_size(g_buft, node);
    const size_t node_bytes = ggml_nbytes(node);
    const size_t f_bytes = (node_bytes + align - 1) / align * align;
    const size_t ws = node_alloc - f_bytes;
    GGML_ASSERT(node_alloc > f_bytes);
    const size_t t_bytes = std::max<size_t>(8 << 20, ws + (1 << 20)) / 4 * 4;

    auto input = [&](size_t bytes) { ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) (bytes / 4)); ggml_set_input(t); ggml_set_output(t); return t; };
    ggml_tensor * s_in = input(1024), * f_in = input(f_bytes), * t_in = input(t_bytes);
    ggml_tensor * S = ggml_cont(ctx, s_in), * F = ggml_cont(ctx, f_in), * T = ggml_cont(ctx, t_in);
    ggml_tensor * S2 = ggml_sum(ctx, S), * F2 = ggml_sum(ctx, F), * F3 = ggml_sum(ctx, F2), * R = ggml_sum(ctx, T);
    ggml_set_output(F3);
    ggml_set_output(S2);
    ggml_set_output(R);
    if (keep_t) {
        ggml_set_output(T);
    }

    if (!under_test_is_plan) {
        ggml_build_forward_expand(rc.gf, g.plan);
    }
    for (ggml_tensor * n : { S, F, T, S2, F2, F3, R }) {
        ggml_build_forward_expand(rc.gf, n);
    }
    const int i_r = ggml_graph_n_nodes(rc.gf) - 1;
    ggml_build_forward_expand(rc.gf, node);
    ggml_build_forward_expand(rc.gf, g.out);
    GGML_ASSERT(ggml_graph_node(rc.gf, i_r) == R);
    // the views between R and the node (none of them dispatches) are skipped by the assert below
    int i_node = i_r + 1;
    while (ggml_graph_node(rc.gf, i_node) != node) {
        const ggml_op op = ggml_graph_node(rc.gf, i_node)->op;
        CHECK(op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE, "a dispatch between R and the sparse node");
        ++i_node;
    }

    GGML_ASSERT(ggml_gallocr_alloc_graph(rc.galloc, rc.gf));

    const uint8_t * nd = (const uint8_t *) node->data;
    const uint8_t * td = (const uint8_t *) T->data;
    const bool same = node->buffer == T->buffer;
    const bool ws_overlaps = same && nd + f_bytes < td + ggml_nbytes(T) && td < nd + node_alloc;
    const bool dst_clear = !same || nd + node_bytes <= td || td + ggml_nbytes(T) <= nd;
    // nothing the barrier tracker still holds overlaps dst: R reads T (dst is clear of it) and writes R; F3 is the barrier before it. So the
    // tracker asks for no barrier between R and the node, and only the sync rule orders the node's workspace after R's read of T.
    const auto clear_of = [&](const ggml_tensor * x) {
        const uint8_t * xd = (const uint8_t *) x->data;
        return node->buffer != x->buffer || nd + node_bytes <= xd || xd + ggml_nbytes(x) <= nd;
    };
    const bool tracker_blind = clear_of(R) && clear_of(F3);
    o.layout = ws_overlaps && dst_clear && tracker_blind;
    printf("  %s %s: dst %p + %zu, alloc %zu, workspace %zu; T %p + %zu: workspace over T %d, dst clear of T %d, dst clear of the unsynced writes %d\n",
           under_test_is_plan ? "plan  " : "attend", keep_t ? "ref  " : "stress", node->data, node_bytes, node_alloc, ws, T->data, ggml_nbytes(T), ws_overlaps, dst_clear, tracker_blind);

    fill_inputs(g, s, 23);
    std::mt19937 rng(29);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (ggml_tensor * in : { s_in, f_in, t_in }) {
        std::vector<float> x(ggml_nelements(in));
        for (auto & v : x) v = dist(rng);
        ggml_backend_tensor_set(in, x.data(), 0, ggml_nbytes(in));
    }
    n_unstable = 0;
    for (int it = 0; it < iters; ++it) {
        GGML_ASSERT(ggml_backend_graph_compute(g_backend, rc.gf) == GGML_STATUS_SUCCESS);
        std::vector<uint8_t> a = get_bytes(g.out), r = get_bytes(R);
        if (it == 0) {
            o.attend = a;
            o.r = r;
        } else if (a != o.attend || r != o.r) {
            ++n_unstable;
        }
    }
    return o;
}

static void test_stress() {
    printf("stress: the workspace over a tensor the preceding dispatch reads\n");
    set_force(nullptr);
    sa_shape s;
    s.n_q = 3;          // n_wg 6: split-K on the 64-lane tiers
    s.n_kv = 8192;
    s.n_sel = 2051;
    struct variant { const char * name; bool plan; };
    for (const variant & v : { variant{"attend workspace", false}, variant{"plan scratch", true} }) {
        {
            // the portable tier has no attend workspace
            ggml_context * c = new_ctx();
            const sa_graph g = sa_build(c, s);
            const bool has_ws = ggml_backend_buft_get_alloc_size(g_buft, v.plan ? g.plan : g.out) > ggml_nbytes(v.plan ? g.plan : g.out);
            ggml_free(c);
            if (!has_ws) {
                printf("  %s: this tier has none, skipped\n", v.name);
                continue;
            }
        }
        int unstable = 0, unused = 0;
        const stress_out st = stress_run(s, v.plan, false, 20, unstable);
        const stress_out rf = stress_run(s, v.plan, true, 1, unused);
        CHECK(st.layout, "%s: the layout does not overlap as the test requires: it would pass trivially", v.name);
        CHECK(unstable == 0, "%s: %d of 19 iterations differ from the first", v.name, unstable);
        CHECK(st.attend == rf.attend, "%s: attend output differs from the graph with T kept alive", v.name);
        CHECK(st.r == rf.r, "%s: the reader of T differs from the graph with T kept alive", v.name);
        printf("  %s: layout %d, unstable iterations %d of 19, attend bitwise to the reference %d, reader of T bitwise %d\n",
               v.name, st.layout, unstable, st.attend == rf.attend, st.r == rf.r);
    }
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    g_reg = ggml_backend_reg_by_name("Vulkan");
    if (g_reg == nullptr || ggml_backend_reg_dev_count(g_reg) == 0) {
        printf("no Vulkan backend: skipped\n");
        return 0;
    }
    g_dev = ggml_backend_reg_dev_get(g_reg, 0);
    g_backend = ggml_backend_dev_init(g_dev, nullptr);
    g_buft = ggml_backend_dev_buffer_type(g_dev);
    printf("device: %s\n", ggml_backend_dev_description(g_dev));

    std::set<std::string> want;
    for (int i = 1; i < argc; ++i) {
        want.insert(argv[i]);
    }
    const auto on = [&](const char * n) { return want.empty() || want.count(n) != 0; };
    if (on("limits"))     test_limits();
    if (on("capability")) test_capability();
    if (on("rot"))        test_rot();
    if (on("stress"))     test_stress();

    ggml_backend_free(g_backend);
    printf("%s: %d failed checks\n", g_fail == 0 ? "OK" : "FAILED", g_fail);
    return g_fail == 0 ? 0 : 1;
}
