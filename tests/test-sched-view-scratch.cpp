// ggml_backend_sched with a scratch tensor reached only through views (qwen4exp's padded destinations): its VIEW node
// must place the source, else the reserve asserts buffer_id >= 0 in ggml_gallocr_allocate_node. One CPU backend suffices.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>

int main() {
    const int64_t ne0 = 64, pad = 16, nt = 8;

    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, nt);
    ggml_set_name(x, "x");
    ggml_set_input(x);

    // the scratch: [ne0 + pad, nt] f16, reached only through views
    ggml_tensor * buf = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ne0 + pad, nt);
    ggml_set_name(buf, "pad_buf");
    ggml_tensor * dst = ggml_view_2d(ctx, buf, ne0, nt, buf->nb[1], 0);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);   // the VIEW node first, as the model emits it
    ggml_tensor * y   = ggml_cpy(ctx, ggml_scale(ctx, x, 2.0f), dst);
    ggml_tensor * rd  = ggml_view_2d(ctx, y, ne0, nt, y->nb[1], 0);
    ggml_tensor * out = ggml_scale(ctx, ggml_cast(ctx, rd, GGML_TYPE_F32), 0.5f);
    ggml_set_name(out, "out");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    // through the registry: with GGML_BACKEND_DL the CPU backend is a loadable module, not linked in
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        fprintf(stderr, "FAIL: no CPU backend\n");
        return 1;
    }
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        fprintf(stderr, "FAIL: ggml_backend_sched_alloc_graph\n");
        return 1;
    }
    std::vector<float> xv(ne0 * nt);
    for (size_t i = 0; i < xv.size(); ++i) {
        xv[i] = 0.25f * (float) ((int) (i % 37) - 18);   // exact in f16
    }
    ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: compute\n");
        return 1;
    }
    std::vector<float> ov(ne0 * nt);
    ggml_backend_tensor_get(out, ov.data(), 0, ggml_nbytes(out));
    for (size_t i = 0; i < ov.size(); ++i) {
        if (ov[i] != xv[i]) {
            fprintf(stderr, "FAIL: out[%zu] = %f, want %f\n", i, ov[i], xv[i]);
            return 1;
        }
    }

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    printf("test-sched-view-scratch: OK\n");
    return 0;
}
