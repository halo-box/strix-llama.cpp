#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static float random_value(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (int(state & 2047) - 1024)/1024.0f;
}

// Smaller query views select the unchanged Vulkan path. Query rows are
// independent, and both paths use the same CM1 tile and unsplit KV traversal.
static bool check(ggml_backend_t backend, int64_t nk, int64_t nq) {
    constexpr int64_t d = 256, hq = 24, hk = 2, chunk = 512;
    const size_t context_size = 128*ggml_tensor_overhead() + 6*ggml_graph_overhead();
    ggml_context * ctx = ggml_init({context_size, nullptr, true});
    GGML_ASSERT(ctx);
    auto qb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, hq, nq, 1);
    auto kb = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, hk, nk, 1);
    auto vb = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, hk, nk, 1);
    auto q = ggml_permute(ctx, qb, 0, 2, 1, 3);
    auto k = ggml_permute(ctx, kb, 0, 2, 1, 3);
    auto v = ggml_permute(ctx, vb, 0, 2, 1, 3);
    auto mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nk, nq, 1, 1);
    auto full = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.0625f, 0, 0);
    ggml_flash_attn_ext_set_prec(full, GGML_PREC_F32);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, full);
    std::vector<ggml_tensor *> parts;
    std::vector<ggml_cgraph *> graphs;
    for (int64_t start = 0; start < nq; start += chunk) {
        const int64_t rows = std::min(chunk, nq - start);
        auto qs = ggml_view_4d(ctx, q, d, rows, hq, 1, q->nb[1], q->nb[2], q->nb[3], start*q->nb[1]);
        auto ms = ggml_view_4d(ctx, mask, nk, rows, 1, 1, mask->nb[1], mask->nb[2], mask->nb[3], start*mask->nb[1]);
        auto part = ggml_flash_attn_ext(ctx, qs, k, v, ms, 0.0625f, 0, 0);
        ggml_flash_attn_ext_set_prec(part, GGML_PREC_F32);
        auto part_graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(part_graph, part);
        parts.push_back(part);
        graphs.push_back(part_graph);
    }
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer && ggml_backend_supports_op(backend, full));
    uint32_t seed = 0x13579bdf;
    std::vector<float> qdata(ggml_nelements(qb));
    std::vector<ggml_fp16_t> kdata(ggml_nelements(kb)), vdata(ggml_nelements(vb));
    for (auto & x : qdata) { x = random_value(seed); }
    for (auto & x : kdata) { x = ggml_fp32_to_fp16(random_value(seed)); }
    for (auto & x : vdata) { x = ggml_fp32_to_fp16(random_value(seed)); }
    std::vector<ggml_fp16_t> mdata(size_t(nk)*nq, 0xfc00);
    for (int64_t row = 0; row < nq; ++row) {
        std::fill_n(mdata.data() + row*nk, nk - nq + row + 1, ggml_fp16_t(0));
    }
    ggml_backend_tensor_set(qb, qdata.data(), 0, qdata.size()*sizeof(float));
    ggml_backend_tensor_set(kb, kdata.data(), 0, kdata.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(vb, vdata.data(), 0, vdata.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(mask, mdata.data(), 0, mdata.size()*sizeof(ggml_fp16_t));
    std::vector<float> reference(ggml_nelements(full)), actual(reference.size());
    size_t offset = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
        ggml_backend_tensor_memset(parts[i], 0xa5, 0, ggml_nbytes(parts[i]));
        GGML_ASSERT(ggml_backend_graph_compute(backend, graphs[i]) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(parts[i], reference.data() + offset, 0, ggml_nbytes(parts[i]));
        offset += ggml_nelements(parts[i]);
    }
    GGML_ASSERT(offset == reference.size());
    ggml_backend_tensor_memset(full, 0x7f, 0, ggml_nbytes(full));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(full, actual.data(), 0, ggml_nbytes(full));
    bool valid = memcmp(reference.data(), actual.data(), ggml_nbytes(full)) == 0;
    for (float x : actual) { valid &= std::isfinite(x); }
    printf("%s N=%lld Q=%lld words=%zu reference_query_chunk=512\n",
        valid ? "PASS" : "FAIL", (long long) nk, (long long) nq, actual.size());
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return valid;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    auto dev = ggml_backend_dev_by_name("Vulkan0");
    if (!dev || std::string(ggml_backend_dev_description(dev)).find("RADV STRIX_HALO") == std::string::npos) {
        puts("SKIP: test targets RADV STRIX_HALO");
        return 77;
    }
    auto backend = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend);
    bool valid = true;
    if (argc == 2) {
        const int64_t n = std::strtoll(argv[1], nullptr, 10);
        GGML_ASSERT(n >= 2048 && n <= 130048);
        valid = check(backend, n, 2048);
    } else {
        for (int64_t n : {32512, 32768, 33024}) { valid &= check(backend, n, 2048); }
        valid &= check(backend, 32768, 2047);
    }
    ggml_backend_free(backend);
    return valid ? 0 : 1;
}
