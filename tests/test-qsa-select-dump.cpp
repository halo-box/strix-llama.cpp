// Census rows generated from the shipped policy (PLAN-v3 §14.3): reads shape lines
//   <n_q> <n_kv> <n_stream> <k> <v> <shortcut 0|1> [<n_sel> <gqa>]
// on stdin and prints SA_SELECT lines for scripts/qsa-validation/census_check.py, from the backend's own group() policy and
// its sa_select logic (the test-only proc ggml_backend_sparse_attn_select_dump). The gate fusion applies to every fusible
// attend, whatever the cache types: since W10 the output un-rotation of a quantized V cache runs inside the op (rot_o), so
// no node sits between the attend and the gate (the backend's select dump names the gated key only where an EPI = gated
// instance exists: the fast G4 tier).
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <iostream>

static ggml_type type_of(const std::string & s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && s == n) {
            return (ggml_type) t;
        }
    }
    return GGML_TYPE_COUNT;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    ggml_backend_sparse_attn_select_dump_t dump = nullptr;
    ggml_backend_sparse_attn_group_t group = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count() && dump == nullptr; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(d);
        dump = (ggml_backend_sparse_attn_select_dump_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_sparse_attn_select_dump");
        group = (ggml_backend_sparse_attn_group_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_sparse_attn_group");
        dev = d;
    }
    if (dump == nullptr || group == nullptr) {
        fprintf(stderr, "no backend with ggml_backend_sparse_attn_select_dump\n");
        return 2;
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        long long n_q, n_kv, n_stream, n_sel = -1, gqa = 12;
        int shortcut;
        std::string ks, vs;
        if (!(is >> n_q >> n_kv >> n_stream >> ks >> vs >> shortcut)) {
            continue;
        }
        const bool extended = (bool) (is >> n_sel >> gqa);
        if (!extended) {
            n_sel = shortcut ? n_kv : 2051;
            gqa   = 12;
        }
        const ggml_type k = type_of(ks), v = type_of(vs);
        const int32_t G = group(dev, n_q, n_kv, k, v, shortcut != 0);
        const int64_t n_head_kv = 2;
        char buf[1024];
        dump(dev, n_q, n_kv, n_stream, n_sel, k, v, gqa * n_head_kv, n_head_kv, G, true, buf, sizeof(buf));
        printf("SA_SELECT %s -> %s\n", line.c_str(), buf);
    }
    return 0;
}
