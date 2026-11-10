// The "supported here" list of the QSA init error (PLAN-v3 §10, llama_qsa_supported_types in src/llama-context.cpp): it
// probes the nine K / V types of the sparse op on the node's device. It ran out of its ggml context (16 tensor objects for
// 9 x 2) and aborted in ggml_new_object / GGML_ASSERT(obj_new) instead of returning the list, so the designed init error
// turned into a process abort. CPU only, no model file: the node is a no_alloc SPARSE_ATTN of the qwen4exp shape.
#include "../src/llama-context.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdio>
#include <string>

int main() {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu == nullptr) {
        printf("test-qsa-supported-types: no CPU device\n");
        return 1;
    }
    ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx { ggml_init(ip) };
    // q [256, n_q, n_head, n_stream], K / V views of 2052 cells and 2 KV heads (GQA 12), a G = 1 plan over 2051 cells
    ggml_tensor * q    = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 256, 9, 24, 1);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_Q8_0, 256, 2052, 2, 1);
    ggml_tensor * v    = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_Q8_0, 256, 2052, 2, 1);
    ggml_tensor * sel  = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_I32, 2051, 9, 1, 1);
    ggml_tensor * plan = ggml_sparse_attn_plan(ctx.get(), sel, 2051, 1);
    ggml_tensor * attn = ggml_sparse_attn(ctx.get(), q, k, v, plan, 0.0625f, 0, 0);

    const std::string list = llama_qsa_supported_types(cpu, attn);
    printf("supported here (CPU): '%s'\n", list.c_str());
    // the CPU op takes every type of S: the list names all nine
    const char * want[] = { "f16", "bf16", "q8_0", "q4_0", "f32", "q4_1", "q5_0", "q5_1", "iq4_nl" };
    int missing = 0;
    for (const char * t : want) {
        if (list.find(t) == std::string::npos) {
            printf("  missing %s\n", t);
            missing++;
        }
    }
    printf("test-qsa-supported-types: %s\n", missing == 0 ? "OK" : "FAILED");
    return missing == 0 ? 0 : 1;
}
