// The "supported here" list of the QSA init error (PLAN-v3 §10, llama_qsa_supported_types in src/llama-context.cpp): it
// probes the nine K / V types of the sparse op on the node's device. It ran out of its ggml context (16 tensor objects for
// 9 x 2) and aborted in ggml_new_object / GGML_ASSERT(obj_new) instead of returning the list, so the designed init error
// turned into a process abort. CPU only, no model file: the node is a no_alloc SPARSE_ATTN of the qwen4exp shape.
#include "../src/llama-context.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdio>
#include <iterator>
#include <string>
#include <vector>

// the comma-separated list as its entries: a type is in it only as a whole entry ("f16" is not in "bf16, q8_0": a
// substring search found it inside "bf16", so a list without f16 passed the f16 check)
static std::vector<std::string> split_list(const std::string & list) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (pos <= list.size() && !list.empty()) {
        const size_t end = list.find(',', pos);
        std::string item = list.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        const size_t a = item.find_first_not_of(' '), b = item.find_last_not_of(' ');
        out.push_back(a == std::string::npos ? "" : item.substr(a, b - a + 1));
        if (end == std::string::npos) {
            break;
        }
        pos = end + 1;
    }
    return out;
}

static bool has_entry(const std::vector<std::string> & items, const std::string & t) {
    for (const auto & i : items) {
        if (i == t) {
            return true;
        }
    }
    return false;
}

int main() {
    // the checker itself: a list with bf16 but not f16 misses f16
    {
        const auto items = split_list("bf16, q8_0, q4_0, f32, q4_1, q5_0, q5_1, iq4_nl");
        if (has_entry(items, "f16") || !has_entry(items, "bf16") || !has_entry(items, "iq4_nl") || items.size() != 8) {
            printf("test-qsa-supported-types: the list checker is wrong (f16 found in a list without it)\n");
            return 1;
        }
    }
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
    const std::vector<std::string> items = split_list(list);
    int missing = 0;
    for (const char * t : want) {
        if (!has_entry(items, t)) {
            printf("  missing %s\n", t);
            missing++;
        }
    }
    if (items.size() != std::size(want)) {
        printf("  %zu entries, want %zu\n", items.size(), std::size(want));
        missing++;
    }
    printf("test-qsa-supported-types: %s\n", missing == 0 ? "OK" : "FAILED");
    return missing == 0 ? 0 : 1;
}
