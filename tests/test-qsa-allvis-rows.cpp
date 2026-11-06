// All-visible rows of the QSA dense-shortcut range (PLAN-v3 §4.1, §14.1; the R1-AV lab test test_allvis_rows.cpp): rows built
// from the KQ mask with the ops of llama_model_qwen4exp::graph::build_qsa_rows_av must equal {j : mask = 0} per (query row,
// stream), with the 4-D shape [n_kv, n_tps, 1, n_stream] the plan asserts, on each backend of the build.
// usage: test-qsa-allvis-rows [mask.bin n_kv n_q]...   (recorded f16 masks; the synthetic split-stream cases always run)
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

static int g_fail = 0;

static ggml_tensor * build_rows(ggml_context * ctx, ggml_tensor * m) {
    ggml_tensor * mf  = m->type == GGML_TYPE_F32 ? m : ggml_cast(ctx, m, GGML_TYPE_F32);
    ggml_tensor * vis = ggml_step(ctx, ggml_scale_bias(ctx, mf, 1.0f, 1.0f));
    ggml_tensor * ar  = ggml_scale_bias(ctx, ggml_arange(ctx, 0.0f, (float) m->ne[0], 1.0f), 1.0f, 1.0f);
    ggml_tensor * idf = ggml_scale_bias(ctx, ggml_mul(ctx, vis, ar), 1.0f, -1.0f);
    return ggml_cast(ctx, idf, GGML_TYPE_I32);
}

static void run_case(ggml_backend_t be, const char * bname, const char * cname, const std::vector<ggml_fp16_t> & mask,
                     int64_t n_kv, int64_t n_tps, int64_t n_stream) {
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_tps, 1, n_stream);
    ggml_tensor * rows = build_rows(ctx, m);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, rows);
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (!ggml_backend_supports_op(be, ggml_graph_node(gf, i))) {
            printf("  [%s %s] unsupported op %s: FAIL\n", bname, cname, ggml_op_desc(ggml_graph_node(gf, i)));
            g_fail++;
            ggml_free(ctx);
            return;
        }
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(m, mask.data(), 0, mask.size()*sizeof(ggml_fp16_t));
    ggml_backend_graph_compute(be, gf);
    std::vector<int32_t> out(ggml_nelements(rows));
    ggml_backend_tensor_get(rows, out.data(), 0, out.size()*sizeof(int32_t));
    // row (q, s) lists j iff mask[s][q][j] == 0, at slot j; -1 elsewhere
    int64_t bad = 0, vis = 0;
    std::set<uint16_t> other;
    for (int64_t s = 0; s < n_stream; ++s) {
        for (int64_t q = 0; q < n_tps; ++q) {
            for (int64_t j = 0; j < n_kv; ++j) {
                const size_t e = (size_t) ((s*n_tps + q)*n_kv + j);
                const float mv = ggml_fp16_to_fp32(mask[e]);
                if (mv != 0.0f && !(std::isinf(mv) && mv < 0)) {
                    other.insert(mask[e]);
                }
                const int32_t want = mv == 0.0f ? (int32_t) j : -1;
                vis += want >= 0;
                bad += out[e] != want;
            }
        }
    }
    const bool shape_ok = rows->ne[0] == n_kv && rows->ne[1] == n_tps && rows->ne[2] == 1 && rows->ne[3] == n_stream;
    printf("  [%s %s] ne=[%lld,%lld,%lld,%lld] mismatch=%lld/%zu visible=%lld other_mask_values=%zu shape=%d\n", bname, cname,
           (long long) rows->ne[0], (long long) rows->ne[1], (long long) rows->ne[2], (long long) rows->ne[3], (long long) bad,
           out.size(), (long long) vis, other.size(), (int) shape_ok);
    if (bad != 0 || !shape_ok) {
        g_fail++;
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// split-stream causal mask: stream s holds len[s] cells (positions 0..len-1, its own sequence); the ubatch's n_tps queries of
// stream s are its last n_tps positions; cells past len (padding of the n_kv view) are -inf
static std::vector<ggml_fp16_t> synth(int64_t n_kv, int64_t n_tps, const std::vector<int64_t> & len) {
    std::vector<ggml_fp16_t> m(n_kv*n_tps*len.size());
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (size_t s = 0; s < len.size(); ++s) {
        for (int64_t q = 0; q < n_tps; ++q) {
            const int64_t p = len[s] - n_tps + q;
            for (int64_t j = 0; j < n_kv; ++j) {
                m[(s*n_tps + q)*n_kv + j] = j <= p ? z : ninf;
            }
        }
    }
    return m;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    struct be_t { ggml_backend_t be; std::string name; };
    std::vector<be_t> bes;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
        ggml_backend_t b = ggml_backend_dev_init(d, nullptr);
        if (b != nullptr) {
            bes.push_back({ b, ggml_backend_dev_name(d) });
        }
    }
    struct case_t { std::string name; int64_t n_kv, n_tps; std::vector<int64_t> len; };
    const std::vector<case_t> cases = {
        { "np2_ntps1_kv2048",   2048, 1,   { 2048, 1203 } },
        { "np2_ntps128_kv2048", 2048, 128, { 1977, 640 } },
        { "np2_ntps128_kv512",  512,  128, { 512, 128 } },
        { "np2_ntps1_kv256",    256,  1,   { 17, 256 } },
        { "np1_ntps512_kv2048", 2048, 512, { 2048 } },
    };
    for (const auto & b : bes) {
        printf("backend %s\n", b.name.c_str());
        for (const auto & c : cases) {
            run_case(b.be, b.name.c_str(), c.name.c_str(), synth(c.n_kv, c.n_tps, c.len), c.n_kv, c.n_tps, (int64_t) c.len.size());
        }
        for (int a = 1; a + 2 < argc; a += 3) {
            const int64_t n_kv = atoll(argv[a + 1]), n_q = atoll(argv[a + 2]);
            std::vector<ggml_fp16_t> m(n_kv*n_q);
            FILE * f = fopen(argv[a], "rb");
            if (f == nullptr || fread(m.data(), 2, m.size(), f) != m.size()) {
                printf("cannot read %s\n", argv[a]);
                return 2;
            }
            fclose(f);
            run_case(b.be, b.name.c_str(), argv[a], m, n_kv, n_q, 1);
        }
    }
    printf("RESULT %s (failures %d)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
