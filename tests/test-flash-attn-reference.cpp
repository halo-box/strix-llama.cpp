#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <vector>

// Uniform logits have an analytical mean; alternating exact logits also exercise
// online max rescaling against an independent double-precision two-pass oracle.
int main() {
    bool ok = true;
    for (const int nk : {2049, 34815, 34816, 34817}) {
        for (const bool varying : {false, true}) {
            constexpr int d = 64;
            ggml_context * ctx = ggml_init({32*1024*1024, nullptr, false});
            ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, 1, 1, 1);
            ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, nk, 1, 1);
            ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, nk, 1, 1);
            ggml_set_zero(q);
            ggml_set_zero(k);
            static_cast<float *>(q->data)[0] = 1.0f;
            double denominator = 0.0;
            double numerator[d] = {};
            for (int i = 0; i < nk; ++i) {
                const float score = varying ? float(i % 9 - 4)*0.5f : 0.0f;
                static_cast<ggml_fp16_t *>(k->data)[i*d] = ggml_fp32_to_fp16(score);
                const double weight = std::exp(double(score) - (varying ? 2.0 : 0.0));
                denominator += weight;
                for (int j = 0; j < d; ++j) {
                    const float value = j % 2 == 0 ? 1.0f : float((i + j) % 17 - 8)*0.125f;
                    static_cast<ggml_fp16_t *>(v->data)[i*d + j] = ggml_fp32_to_fp16(value);
                    numerator[j] += weight*value;
                }
            }
            ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f, 0.0f, 0.0f);
            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, out);
            ggml_cplan plan = ggml_graph_plan(graph, 1, nullptr);
            plan.use_ref = true;
            std::vector<uint8_t> work(plan.work_size);
            plan.work_data = work.data();
            ok &= ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS;
            double max_error = 0.0;
            for (int j = 0; j < d; ++j) {
                const double actual = static_cast<float *>(out->data)[j];
                const double error = std::abs(actual - numerator[j]/denominator);
                ok &= std::isfinite(actual) && error < 2e-5;
                max_error = std::fmax(max_error, error);
            }
            std::printf("CPU reference nk=%d varying=%d max_abs=%.9g %s\n", nk, varying, max_error, ok ? "OK" : "FAIL");
            ggml_free(ctx);
        }
    }
    return ok ? 0 : 1;
}
