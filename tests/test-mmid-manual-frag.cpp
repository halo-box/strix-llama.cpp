// MUL_MAT_ID manual coopmat fragments (GGML_VK_MMID_MANUAL_FRAG): per case, the output bytes of one MUL_MAT_ID graph on Vulkan0 as an
// FNV-1a hash plus the dispatch census paths MUL_MAT_ID_ALL / MUL_MAT_ID_PF (coopmat register-prefetch and REG_A pipelines) /
// MUL_MAT_ID_MANUAL_FRAG (the "_mf" builds). The manual-fragment choice is made once per device, so the comparison is across
// processes: run it with the default (manual fragments where the RDNA3 layout probe passed), with GGML_VK_MMID_MANUAL_FRAG=0 and
// with GGML_VK_TEST_MF_SKEW=1, then compare the hashes (tests/test-mmid-manual-frag.py):
//   default == off   on every case: the manual fragment reads give coopMatLoad's elements, so the WMMA sequence is the same
//   skew != off      on every case that ran a "_mf" pipeline: the comparison can fail (positive control)
//   mf / pf == 1     on the default run, 0 on the off run (engagement); rega_mf / rega == 1 likewise for the REG_A twins, and the
//                    REG_A shapes (320 experts top-10 at n 2048 = 64 rows per expert) must run REG_A at all (rega > 0)
// Inputs are seeded, so a process sees the same bytes every run. The routed f16-B coopmat path is forced (GGML_VK_MMID_F16COOPMAT=all)
// so every type here reaches the pipelines that have an "_mf" build. Skips (exit 0) without a Vulkan device.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// the Vulkan dispatch census test hooks (off until the first reset)
typedef uint64_t (*fusion_count_t)(const char *);
typedef void     (*fusion_reset_t)(void);

static fusion_count_t g_count = nullptr;
static fusion_reset_t g_reset = nullptr;

static uint64_t fnv1a(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= ((const uint8_t *) p)[i]; h *= 1099511628211ull; }
    return h;
}

static void set_vals(ggml_tensor * t, const std::vector<float> & v) {
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(v.size());
        for (size_t i = 0; i < v.size(); ++i) h[i] = ggml_fp32_to_fp16(v[i]);
        ggml_backend_tensor_set(t, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
    } else {
        std::vector<uint8_t> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, v.data(), q.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    }
}

static std::vector<float> randv(size_t n, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto & x : v) x = u(rng);
    return v;
}

enum epi { EPI_F32, EPI_CPY16, EPI_MUL_CPY16 };

// a: [k, m, n_exp] experts; b: [k, n_used or 1 (bcast), n]; ids: [n_used, n] distinct experts per token
static void run_case(ggml_backend_t be, ggml_type ta, ggml_type tb, int64_t m, int64_t k, int n_exp, int n_used, int64_t n,
                     bool bcast, epi e, int64_t pad_b) {
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int64_t nb_rows = bcast ? 1 : n_used;
    ggml_tensor * a = ggml_new_tensor_3d(ctx, ta, k, m, n_exp);
    // padded B rows: B is a view of a [k + pad_b, ...] tensor (the row stride a padded activation buffer gives)
    ggml_tensor * bb = ggml_new_tensor_3d(ctx, tb, k + pad_b, nb_rows, n);
    ggml_tensor * b = pad_b ? ggml_view_3d(ctx, bb, k, nb_rows, n, bb->nb[1], bb->nb[2], 0) : bb;
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n);
    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, n);
    ggml_tensor * out = ggml_mul_mat_id(ctx, a, b, ids);
    if (e == EPI_MUL_CPY16) {
        out = ggml_mul(ctx, out, w);
    }
    if (e != EPI_F32) {
        out = ggml_cast(ctx, out, GGML_TYPE_F16);
    }
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    std::mt19937 rng(1234u + (uint32_t) ta * 7u + (uint32_t) m + (uint32_t) k + (uint32_t) n);
    set_vals(a, randv(ggml_nelements(a), rng));
    set_vals(bb, randv(ggml_nelements(bb), rng));
    set_vals(w, randv(ggml_nelements(w), rng));
    std::vector<int32_t> id((size_t) n_used * n);
    std::vector<int> perm(n_exp);
    for (int64_t t = 0; t < n; ++t) {
        for (int i = 0; i < n_exp; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int u = 0; u < n_used; ++u) id[(size_t) t * n_used + u] = perm[u];
    }
    ggml_backend_tensor_set(ids, id.data(), 0, id.size() * sizeof(int32_t));

    g_reset();
    ggml_backend_graph_compute(be, gf);
    ggml_backend_synchronize(be);
    std::vector<uint8_t> bytes(ggml_nbytes(out));
    ggml_backend_tensor_get(out, bytes.data(), 0, bytes.size());
    const char * en = e == EPI_F32 ? "f32" : e == EPI_CPY16 ? "cpy16" : "mul_cpy16";
    printf("case %s_x_%s m%lld k%lld E%d u%d n%lld %s %s pad%lld hash=%016llx all=%llu pf=%llu mf=%llu rega=%llu rega_mf=%llu\n",
           ggml_type_name(ta), ggml_type_name(tb), (long long) m, (long long) k, n_exp, n_used, (long long) n, bcast ? "bcast" : "rows", en,
           (long long) pad_b, (unsigned long long) fnv1a(bytes.data(), bytes.size()),
           (unsigned long long) g_count("path:MUL_MAT_ID_ALL"), (unsigned long long) g_count("path:MUL_MAT_ID_PF"),
           (unsigned long long) g_count("path:MUL_MAT_ID_MANUAL_FRAG"),
           (unsigned long long) g_count("path:MUL_MAT_ID_REGA"), (unsigned long long) g_count("path:MUL_MAT_ID_REGA_MF"));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main() {
    setenv("GGML_VK_MMID_F16COOPMAT", "all", 0);
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("Vulkan0");
    if (!dev) { printf("test-mmid-manual-frag: no Vulkan0 device, skipped\n"); return 0; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    g_count = (fusion_count_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_dispatch_census_count");
    g_reset = (fusion_reset_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_debug_dispatch_census_reset");
    if (!g_count || !g_reset) { printf("FAIL: no dispatch census counters\n"); return 1; }
    printf("test-mmid-manual-frag: GGML_VK_MMID_F16COOPMAT=%s GGML_VK_MMID_MANUAL_FRAG=%s GGML_VK_TEST_MF_SKEW=%s\n",
           getenv("GGML_VK_MMID_F16COOPMAT"), getenv("GGML_VK_MMID_MANUAL_FRAG") ? getenv("GGML_VK_MMID_MANUAL_FRAG") : "(default)",
           getenv("GGML_VK_TEST_MF_SKEW") ? getenv("GGML_VK_TEST_MF_SKEW") : "0");

    // Flash-Next experts: gate/up iq3_xxs [2560 -> 640], down iq4_nl [640 -> 2560], 320 of them top-10 at ub 2048 (64 rows per
    // expert: the REG_A medium tile), B broadcast for gate/up; q8_0 and the K-quants as the other pf types
    for (ggml_type tb : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
        run_case(be, GGML_TYPE_IQ3_XXS, tb, 640, 2560, 320, 10, 2048, true,  EPI_F32,       0);
        run_case(be, GGML_TYPE_IQ3_XXS, tb, 640, 2560, 320, 10, 2048, true,  EPI_CPY16,     0);
        run_case(be, GGML_TYPE_IQ4_NL,  tb, 2560, 640, 320, 10, 2048, false, EPI_MUL_CPY16, 0);
        for (int64_t n : { 2048, 130 }) {
            run_case(be, GGML_TYPE_Q8_0,    tb, 640, 2560, 32, 10, n, true,  EPI_F32,       0);
            run_case(be, GGML_TYPE_IQ3_XXS, tb, 640, 2560, 32, 10, n, false, EPI_F32,       0);
            run_case(be, GGML_TYPE_IQ4_NL,  tb, 2560, 640, 32, 10, n, false, EPI_MUL_CPY16, 0);
            run_case(be, GGML_TYPE_Q4_K,    tb, 768, 2560, 32, 10, n, true,  EPI_F32,       0);
            run_case(be, GGML_TYPE_Q6_K,    tb, 2560, 768, 32, 10, n, false, EPI_MUL_CPY16, 0);
        }
        // padded B rows, a k off the 64 alignment (unaligned pipeline: no _mf build, must be unchanged too)
        run_case(be, GGML_TYPE_IQ3_XXS, tb, 640, 2560, 32, 10, 512, false, EPI_F32, 64);
        run_case(be, GGML_TYPE_Q8_0,    tb, 640, 2592, 32, 10, 512, false, EPI_F32, 0);
    }
    ggml_backend_free(be);
    printf("test-mmid-manual-frag: done\n");
    return 0;
}
