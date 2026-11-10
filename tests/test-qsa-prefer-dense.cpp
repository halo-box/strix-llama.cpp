// The QSA shortcut's dense pick (ggml_backend_sparse_attn_prefer_dense, ggml-backend.h) must be an op the device runs: qwen4exp
// builds the dense FLASH_ATTN_EXT for the n_kv <= top_k + r - 1 shortcut whenever the policy returns true, and the QSA guard
// fails the context ("no dense attention kernel for this pair on this device") when the device's FLASH_ATTN_EXT does not take
// the pair. HIP's table ignored its device: a build with -DGGML_CUDA_FA=OFF (no FLASH_ATTN_EXT at all) picked dense for f16 /
// f16 in every n_q band and for most quantized pairs at n_q <= 127, so every first decode step failed there.
//
// For every device whose backend exposes the policy (HIP, Vulkan): every (K, V, n_q band) cell the policy calls dense must be a
// FLASH_ATTN_EXT of the shortcut's shape (D 256, 24 heads, 2 KV heads, causal f16 mask) that the device's supports_op takes.
// A backend without the policy is skipped. Exit status 0 = every dense pick is runnable.
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <string>

static bool fa_supported(ggml_backend_dev_t dev, ggml_type tk, ggml_type tv, int64_t n_q, int64_t n_kv) {
    ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, n_q, 24, 1);
    ggml_tensor * k  = ggml_new_tensor_4d(ctx, tk, 256, n_kv, 2, 1);
    ggml_tensor * v  = ggml_new_tensor_4d(ctx, tv, 256, n_kv, 2, 1);
    ggml_tensor * m  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q, 1, 1);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, m, 0.0625f, 0.0f, 0.0f);
    const bool ok = ggml_backend_dev_supports_op(dev, fa);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_backend_load_all();
    int n_fail = 0, n_dev = 0;
    const ggml_type types[] = { GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL };
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto prefer = reg ? (ggml_backend_sparse_attn_prefer_dense_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_sparse_attn_prefer_dense") : nullptr;
        if (prefer == nullptr) {
            printf("%s (%s): no sparse-attention dense policy, skipped\n", ggml_backend_dev_name(dev), ggml_backend_reg_name(reg));
            continue;
        }
        n_dev++;
        int n_dense = 0, n_cells = 0, n_bad = 0;
        for (ggml_type tk : types) {
            for (ggml_type tv : types) {
                for (int64_t n_q : { 1, 8, 32, 127, 128, 511, 1024 }) {
                    for (int64_t n_kv : { 256, 2048 }) {
                        n_cells++;
                        if (!prefer(dev, tk, tv, n_q, n_kv)) {
                            continue;
                        }
                        n_dense++;
                        if (!fa_supported(dev, tk, tv, n_q, n_kv)) {
                            if (n_bad++ < 10) {
                                printf("  FAIL %s: dense picked for %s/%s n_q %d n_kv %d, but this device's FLASH_ATTN_EXT rejects it\n",
                                       ggml_backend_dev_name(dev), ggml_type_name(tk), ggml_type_name(tv), (int) n_q, (int) n_kv);
                            }
                        }
                    }
                }
            }
        }
        printf("%s (%s): dense %d of %d cells, %d not runnable; FLASH_ATTN_EXT f16/f16 %s, q8_0/q8_0 %s\n", ggml_backend_dev_name(dev),
               ggml_backend_reg_name(reg), n_dense, n_cells, n_bad, fa_supported(dev, GGML_TYPE_F16, GGML_TYPE_F16, 1, 2048) ? "supported" : "unsupported",
               fa_supported(dev, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 1, 2048) ? "supported" : "unsupported");
        n_fail += n_bad;
    }
    printf("test-qsa-prefer-dense: %s (%d devices with the policy, %d bad dense picks)\n", n_fail == 0 ? "OK" : "FAILED", n_dev, n_fail);
    return n_fail == 0 ? 0 : 1;
}
