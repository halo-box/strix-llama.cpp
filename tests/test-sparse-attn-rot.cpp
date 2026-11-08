// HIP checks of the in-op rotations of GGML_OP_SPARSE_ATTN (PLAN-v3 §8, §13 W10) that test-backend-ops cannot express:
// the transforms are compared bit for bit with fwht.cu (a GGML_HINT_SRC0_IS_HADAMARD mul_mat on the same backend), not
// within a band.
//
//   H_q    the kernel's q transform (GGML_CUDA_SA_ROT_DUMP=1: an op with rot_q != 0 writes H_q q, f32, into dst and
//          stops) against fwht.cu of the same q;
//   H_o    the op with rot_o against fwht.cu of the same op's rot 0 output;
//   graph  the op with (rot_q, rot_o) against the graph-side form it replaces in qwen4exp (fwht.cu of q, the op at rot 0,
//          fwht.cu of the output): equal bit for bit, because everything but the transforms is the rot 0 arithmetic;
//   gated  (reported, not gated: fused and unfused gates are not claimed bit-equal, PLAN-v3 §2.2 item 4) the gate fused
//          into the rotated epilogue against the unfused pre-W10 graph (fwht.cu between the attend and the gate).
//
// Shapes: G4 with packed V (n_q 65, n_kv 4096 < 64 * 65) and direct V (n_kv 16384), G1 with one split (n_q 3, n_sel 17)
// and with the split merge (n_sel 257: 5 splits); K = V in f16, q8_0 (the fast tier) and q4_1 (the generic tier);
// rot 64 / 128 / 256 on each side. The tier is the process's: run it once per GGML_CUDA_SA_FORCE value
// ("", generic, portable, fixup). HIP graphs are disabled (the dump hook is read at dispatch time).
//
// Without a ROCm backend it exits 0 ("skipped"). usage: test-sparse-attn-rot [hq|ho|graph|gated ...]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

static ggml_backend_reg_t         g_reg     = nullptr;
static ggml_backend_dev_t         g_dev     = nullptr;
static ggml_backend_t             g_backend = nullptr;
static ggml_backend_buffer_type_t g_buft    = nullptr;
static int g_fail = 0, g_checks = 0;

#define CHECK(cond, ...) do { ++g_checks; if (!(cond)) { printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); ++g_fail; } } while (0)

struct sa_shape {
    const char * name = "";
    ggml_type tk = GGML_TYPE_F16, tv = GGML_TYPE_F16;
    int64_t n_q = 65, n_kv = 4096, n_sel = 257, kv_head = 2, gqa = 12;
    int G = 4;
    int rot_q = 0, rot_o = 0;
    float scale = 0.0625f;
};

struct sa_graph {
    ggml_tensor * q_st, * q, * k_st, * k, * v_st, * v, * sel, * plan, * out, * gate;
};

static sa_graph sa_build(ggml_context * ctx, const sa_shape & s, ggml_tensor * q_override = nullptr) {
    sa_graph g = {};
    g.q_st = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, s.kv_head * s.gqa, s.n_q, 1);
    ggml_tensor * qs = q_override ? q_override : g.q_st;
    g.q    = ggml_permute(ctx, qs, 0, 2, 1, 3);
    g.k_st = ggml_new_tensor_4d(ctx, s.tk, 256, s.kv_head, s.n_kv, 1);
    g.k    = ggml_permute(ctx, g.k_st, 0, 2, 1, 3);
    g.v_st = ggml_new_tensor_4d(ctx, s.tv, 256, s.kv_head, s.n_kv, 1);
    g.v    = ggml_permute(ctx, g.v_st, 0, 2, 1, 3);
    g.sel  = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, s.n_sel, s.n_q, 1, 1);
    g.plan = ggml_sparse_attn_plan(ctx, g.sel, s.n_kv, s.G);
    g.out  = ggml_sparse_attn(ctx, g.q, g.k, g.v, g.plan, s.scale, s.rot_q, s.rot_o);
    g.gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, s.kv_head * s.gqa, s.n_q);
    for (ggml_tensor * t : { g.q_st, g.k_st, g.v_st, g.sel, g.gate }) {
        ggml_set_input(t);
        ggml_set_output(t);   // keep the inputs allocated for the whole graph
    }
    return g;
}

// the values are drawn whether or not the graph allocated t (a graph may not use every input), so every graph of a check
// sees the same data
static void set_if_allocated(ggml_tensor * t, const void * data, size_t n) {
    if (t->buffer != nullptr) {
        ggml_backend_tensor_set(t, data, 0, n);
    }
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
    set_if_allocated(t, bytes.data(), bytes.size());
}

static void fill_inputs(const sa_graph & g, const sa_shape & s, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> q(ggml_nelements(g.q_st));
    for (auto & x : q) x = 4.0f * dist(rng);   // scores of a few units: a non-trivial softmax
    set_if_allocated(g.q_st, q.data(), ggml_nbytes(g.q_st));
    std::vector<float> gate(ggml_nelements(g.gate));
    for (auto & x : gate) x = 3.0f * dist(rng);
    set_if_allocated(g.gate, gate.data(), ggml_nbytes(g.gate));
    fill_kv(g.k_st, rng);
    fill_kv(g.v_st, rng);
    std::vector<int32_t> rows(ggml_nelements(g.sel), -1);
    for (int64_t r = 0; r < s.n_q; ++r) {
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
    set_if_allocated(g.sel, rows.data(), ggml_nbytes(g.sel));
}

static std::vector<uint8_t> get_bytes(const ggml_tensor * t) {
    std::vector<uint8_t> b(ggml_nbytes(t));
    ggml_backend_tensor_get(t, b.data(), 0, b.size());
    return b;
}

struct run_ctx {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_cgraph * gf = nullptr;
    run_ctx() {
        ggml_init_params p = { 64 << 20, nullptr, true };
        ctx = ggml_init(p);
        galloc = ggml_gallocr_new(g_buft);
        gf = ggml_new_graph_custom(ctx, 256, false);
    }
    ~run_ctx() {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
    }
    void run() {
        GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    }
    void compute() {
        GGML_ASSERT(ggml_backend_graph_compute(g_backend, gf) == GGML_STATUS_SUCCESS);
    }
};

// fwht.cu: the hinted Hadamard mul_mat over rows of r (llama_mul_mat_hadamard's graph)
static ggml_tensor * hadamard(ggml_context * ctx, ggml_tensor * x, int r, std::vector<ggml_tensor *> & hs) {
    ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, r);
    ggml_set_input(h);
    hs.push_back(h);
    ggml_tensor * rows = ggml_is_contiguous(x) ? ggml_reshape_2d(ctx, x, r, ggml_nelements(x) / r) : ggml_cont_2d(ctx, x, r, ggml_nelements(x) / r);
    ggml_tensor * res = ggml_mul_mat(ctx, h, rows);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    return ggml_reshape_4d(ctx, res, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
}

static void set_hadamards(const std::vector<ggml_tensor *> & hs) {
    for (ggml_tensor * h : hs) {
        const int r = (int) h->ne[0];
        std::vector<float> m((size_t) r * r);
        const float sc = 1.0f / std::sqrt((float) r);
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < r; ++j) {
                m[(size_t) i * r + j] = (__builtin_popcount(i & j) & 1 ? -1.0f : 1.0f) * sc;
            }
        }
        ggml_backend_tensor_set(h, m.data(), 0, ggml_nbytes(h));
    }
}

static size_t n_diff_words(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    GGML_ASSERT(a.size() == b.size() && a.size() % 4 == 0);
    size_t n = 0;
    for (size_t i = 0; i < a.size(); i += 4) {
        n += memcmp(a.data() + i, b.data() + i, 4) != 0;
    }
    return n;
}

static double nmse(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    const float * x = (const float *) a.data(), * y = (const float *) b.data();
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size() / 4; ++i) {
        num += ((double) x[i] - y[i]) * ((double) x[i] - y[i]);
        den += (double) y[i] * y[i];
    }
    return den > 0.0 ? num / den : num;
}

static bool supported(const sa_shape & s) {
    ggml_init_params p = { 16 << 20, nullptr, true };
    ggml_context * c = ggml_init(p);
    const sa_graph g = sa_build(c, s);
    const bool ok = ggml_backend_dev_supports_op(g_dev, g.out) && ggml_backend_dev_supports_op(g_dev, g.plan);
    ggml_free(c);
    return ok;
}

static std::vector<sa_shape> shapes() {
    std::vector<sa_shape> v;
    for (ggml_type t : { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1 }) {
        sa_shape s;
        s.tk = s.tv = t;
        s.name = "G4 packed V"; s.G = 4; s.n_q = 65; s.n_kv = 4096;  s.n_sel = 257; v.push_back(s);
        s.name = "G4 direct V"; s.G = 4; s.n_q = 65; s.n_kv = 16384; s.n_sel = 257; v.push_back(s);
        s.name = "G1 1 split";  s.G = 1; s.n_q = 3;  s.n_kv = 4096;  s.n_sel = 17;  v.push_back(s);
        s.name = "G1 merge";    s.G = 1; s.n_q = 3;  s.n_kv = 4096;  s.n_sel = 257; v.push_back(s);
    }
    return v;
}

static void test_hq() {
    printf("hq: H_q in the kernel against fwht.cu\n");
    for (sa_shape s : shapes()) {
        for (int r : { 64, 128, 256 }) {
            s.rot_q = r;
            s.rot_o = 0;
            if (!supported(s)) {
                printf("  %-12s %-5s rot %3d: not supported\n", s.name, ggml_type_name(s.tk), r);
                CHECK(false, "%s %s rot_q %d not supported", s.name, ggml_type_name(s.tk), r);
                continue;
            }
            std::vector<uint8_t> dump, ref;
            {
                run_ctx rc;
                const sa_graph g = sa_build(rc.ctx, s);
                ggml_set_output(g.out);
                ggml_build_forward_expand(rc.gf, g.out);
                rc.run();
                fill_inputs(g, s, 7);
                setenv("GGML_CUDA_SA_ROT_DUMP", "1", 1);
                rc.compute();
                unsetenv("GGML_CUDA_SA_ROT_DUMP");
                dump = get_bytes(g.out);
            }
            {
                run_ctx rc;
                const sa_graph g = sa_build(rc.ctx, s);
                std::vector<ggml_tensor *> hs;
                ggml_tensor * res = hadamard(rc.ctx, g.q_st, r, hs);
                ggml_set_output(res);
                ggml_build_forward_expand(rc.gf, res);
                rc.run();
                fill_inputs(g, s, 7);
                set_hadamards(hs);
                rc.compute();
                ref = get_bytes(res);
            }
            const size_t nd = n_diff_words(dump, ref);
            CHECK(nd == 0, "%s %s: H_q rot %d differs from fwht.cu in %zu of %zu values", s.name, ggml_type_name(s.tk), r, nd, dump.size() / 4);
            printf("  %-12s %-5s rot %3d: H_q %s (%zu values)\n", s.name, ggml_type_name(s.tk), r, nd == 0 ? "bitwise" : "DIFFERS", dump.size() / 4);
        }
    }
}

static void test_ho() {
    printf("ho: H_o in the kernel against fwht.cu of the rot 0 output\n");
    for (sa_shape s : shapes()) {
        for (int r : { 64, 128, 256 }) {
            s.rot_q = 0;
            s.rot_o = r;
            if (!supported(s)) {
                CHECK(false, "%s %s rot_o %d not supported", s.name, ggml_type_name(s.tk), r);
                continue;
            }
            std::vector<uint8_t> a, b;
            {
                run_ctx rc;
                const sa_graph g = sa_build(rc.ctx, s);
                ggml_set_output(g.out);
                ggml_build_forward_expand(rc.gf, g.out);
                rc.run();
                fill_inputs(g, s, 11);
                rc.compute();
                a = get_bytes(g.out);
            }
            {
                run_ctx rc;
                s.rot_o = 0;
                const sa_graph g = sa_build(rc.ctx, s);
                std::vector<ggml_tensor *> hs;
                ggml_tensor * res = hadamard(rc.ctx, g.out, r, hs);
                ggml_set_output(res);
                ggml_build_forward_expand(rc.gf, res);
                rc.run();
                fill_inputs(g, s, 11);
                set_hadamards(hs);
                rc.compute();
                b = get_bytes(res);
                s.rot_o = r;
            }
            const size_t nd = n_diff_words(a, b);
            CHECK(nd == 0, "%s %s: H_o rot %d differs from fwht.cu in %zu of %zu values", s.name, ggml_type_name(s.tk), r, nd, a.size() / 4);
            printf("  %-12s %-5s rot %3d: H_o %s (%zu values)\n", s.name, ggml_type_name(s.tk), r, nd == 0 ? "bitwise" : "DIFFERS", a.size() / 4);
        }
    }
}

// the graph-side form W10 replaces: fwht.cu(q) -> op(rot 0) -> fwht.cu(out)
static void test_graph() {
    printf("graph: the op with (rot_q, rot_o) against fwht.cu(q) -> op(rot 0) -> fwht.cu(out)\n");
    for (sa_shape s : shapes()) {
        for (auto rr : std::vector<std::pair<int, int>>{ {256, 64}, {64, 256}, {128, 128}, {256, 0}, {0, 64} }) {
            s.rot_q = rr.first;
            s.rot_o = rr.second;
            if (!supported(s)) {
                CHECK(false, "%s %s rot (%d, %d) not supported", s.name, ggml_type_name(s.tk), rr.first, rr.second);
                continue;
            }
            std::vector<uint8_t> a, b;
            {
                run_ctx rc;
                const sa_graph g = sa_build(rc.ctx, s);
                ggml_set_output(g.out);
                ggml_build_forward_expand(rc.gf, g.out);
                rc.run();
                fill_inputs(g, s, 23);
                rc.compute();
                a = get_bytes(g.out);
            }
            {
                run_ctx rc;
                sa_shape s0 = s;
                s0.rot_q = s0.rot_o = 0;
                std::vector<ggml_tensor *> hs;
                // q_st is created by sa_build; the rotated q is a node over it
                ggml_tensor * q_st = ggml_new_tensor_4d(rc.ctx, GGML_TYPE_F32, 256, s.kv_head * s.gqa, s.n_q, 1);
                ggml_tensor * q_rot = s.rot_q ? hadamard(rc.ctx, q_st, s.rot_q, hs) : q_st;
                sa_graph g = sa_build(rc.ctx, s0, q_rot);
                // sa_build made its own q_st: feed ours instead (same name in fill_inputs)
                ggml_set_input(q_st);
                ggml_set_output(q_st);
                g.q_st = q_st;
                ggml_tensor * res = s.rot_o ? hadamard(rc.ctx, g.out, s.rot_o, hs) : g.out;
                ggml_set_output(res);
                ggml_build_forward_expand(rc.gf, res);
                rc.run();
                fill_inputs(g, s, 23);
                set_hadamards(hs);
                rc.compute();
                b = get_bytes(res);
            }
            const size_t nd = n_diff_words(a, b);
            CHECK(nd == 0, "%s %s: rot (%d, %d) differs from the graph-side form in %zu of %zu values (nmse %.3e)",
                  s.name, ggml_type_name(s.tk), rr.first, rr.second, nd, a.size() / 4, nmse(a, b));
            printf("  %-12s %-5s rot (%3d, %3d): %s\n", s.name, ggml_type_name(s.tk), rr.first, rr.second, nd == 0 ? "bitwise" : "DIFFERS");
        }
    }
}

// the HIP dispatch census (ggml-cuda.cu): times exactly `key` ran since the reset
static int64_t census(const char * key) {
    auto count = (uint64_t (*)(const char *)) ggml_backend_reg_get_proc_address(g_reg, "ggml_backend_debug_dispatch_census_count");
    GGML_ASSERT(count != nullptr);
    return (int64_t) count(key);
}

static void census_reset() {
    auto reset = (void (*)(void)) ggml_backend_reg_get_proc_address(g_reg, "ggml_backend_debug_dispatch_census_reset");
    GGML_ASSERT(reset != nullptr);
    reset();
}

// reported only: the gate fused into the rotated epilogue against the unfused pre-W10 graph (the census checks that the
// first graph ran the EPI = gated instance and the second the plain one)
static void test_gated() {
    printf("gated (report): fused gate in the rotated epilogue against fwht.cu between the attend and an unfused gate\n");
    for (sa_shape s : shapes()) {
        if (s.G != 4 || s.tk == GGML_TYPE_Q4_1) {
            continue;   // the gated instances are fast G4 only
        }
        s.rot_q = 256;
        s.rot_o = 64;
        const char * force = getenv("GGML_CUDA_SA_FORCE");
        const bool fast = force == nullptr || (strstr(force, "generic") == nullptr && strstr(force, "portable") == nullptr);
        const std::string tn = ggml_type_name(s.tk);
        const std::string vl = s.n_kv < 64 * s.n_q ? "packed" : "direct";
        const std::string kg = "sa_attend/G4/" + tn + "/" + tn + "/" + vl + "/gated/gqa12";
        const std::string kp = "sa_attend/G4/" + tn + "/" + tn + "/" + vl + "/plain/gqa12";
        int64_t fused_g = 0, fused_p = 0, unf_g = 0, unf_p = 0;
        std::vector<uint8_t> a, b;
        {
            run_ctx rc;
            const sa_graph g = sa_build(rc.ctx, s);
            ggml_tensor * cur = ggml_reshape_3d(rc.ctx, g.out, 256, s.kv_head * s.gqa, s.n_q);
            ggml_tensor * mul = ggml_mul(rc.ctx, cur, ggml_sigmoid(rc.ctx, ggml_cont(rc.ctx, g.gate)));
            ggml_set_output(mul);
            ggml_build_forward_expand(rc.gf, mul);
            rc.run();
            fill_inputs(g, s, 31);
            census_reset();
            rc.compute();
            fused_g = census(kg.c_str());
            fused_p = census(kp.c_str());
            a = get_bytes(mul);
        }
        {
            run_ctx rc;
            sa_shape s0 = s;
            s0.rot_o = 0;
            const sa_graph g = sa_build(rc.ctx, s0);
            std::vector<ggml_tensor *> hs;
            ggml_tensor * cur = ggml_reshape_3d(rc.ctx, hadamard(rc.ctx, g.out, s.rot_o, hs), 256, s.kv_head * s.gqa, s.n_q);
            ggml_tensor * mul = ggml_mul(rc.ctx, cur, ggml_sigmoid(rc.ctx, ggml_cont(rc.ctx, g.gate)));
            ggml_set_output(mul);
            ggml_build_forward_expand(rc.gf, mul);
            rc.run();
            fill_inputs(g, s, 31);
            set_hadamards(hs);
            census_reset();
            rc.compute();
            unf_g = census(kg.c_str());
            unf_p = census(kp.c_str());
            b = get_bytes(mul);
        }
        if (fast) {
            CHECK(fused_g == 1 && fused_p == 0, "%s %s: the rotated graph did not fuse the gate (%s %lld, %s %lld)", s.name,
                  tn.c_str(), kg.c_str(), (long long) fused_g, kp.c_str(), (long long) fused_p);
            CHECK(unf_g == 0 && unf_p == 1, "%s %s: the graph-side form fused the gate (%s %lld, %s %lld)", s.name,
                  tn.c_str(), kg.c_str(), (long long) unf_g, kp.c_str(), (long long) unf_p);
        }
        printf("  %-12s %-5s census: in-op rotation gated %lld plain %lld; graph-side rotation gated %lld plain %lld\n", s.name,
               tn.c_str(), (long long) fused_g, (long long) fused_p, (long long) unf_g, (long long) unf_p);
        const size_t nd = n_diff_words(a, b);
        printf("  %-12s %-5s rot (256, 64): fused vs unfused %s, %zu of %zu values differ, nmse %.3e\n", s.name,
               ggml_type_name(s.tk), nd == 0 ? "bitwise" : "differ", nd, a.size() / 4, nmse(a, b));
        CHECK(nmse(a, b) < 1e-12, "%s %s: gated rotated epilogue far from the unfused graph (nmse %.3e)", s.name, ggml_type_name(s.tk), nmse(a, b));
    }
}

int main(int argc, char ** argv) {
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
    unsetenv("GGML_CUDA_SA_ROT_DUMP");
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("ROCm");
    if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) {
        printf("no ROCm backend: skipped\n");
        return 0;
    }
    g_reg = reg;
    g_dev = ggml_backend_reg_dev_get(reg, 0);
    g_backend = ggml_backend_dev_init(g_dev, nullptr);
    g_buft = ggml_backend_dev_buffer_type(g_dev);
    const char * force = getenv("GGML_CUDA_SA_FORCE");
    printf("device: %s, GGML_CUDA_SA_FORCE=%s\n", ggml_backend_dev_description(g_dev), force ? force : "");

    std::set<std::string> want;
    for (int i = 1; i < argc; ++i) {
        want.insert(argv[i]);
    }
    const auto on = [&](const char * n) { return want.empty() || want.count(n) != 0; };
    if (on("hq"))    test_hq();
    if (on("ho"))    test_ho();
    if (on("graph")) test_graph();
    if (on("gated")) test_gated();

    ggml_backend_free(g_backend);
    printf("%s: %d of %d checks failed\n", g_fail == 0 ? "OK" : "FAILED", g_fail, g_checks);
    return g_fail == 0 ? 0 : 1;
}
