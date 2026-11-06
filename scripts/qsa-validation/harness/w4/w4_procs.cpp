// W4: the HIP policy procs as llama resolves them (ggml_backend_reg_get_proc_address of the device's reg): prints the
// group policy over the AT-A grid (n_q x n_kv x V type, past the shortcut and in it) and the prefer_dense table, and
// checks GGML_CUDA_SA_FORCE=g1|g4 (read at every call). usage: w4_procs
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("ROCm0");
    if (!dev) { fprintf(stderr, "no ROCm0\n"); return 1; }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto group = (ggml_backend_sparse_attn_group_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_sparse_attn_group");
    auto dense = (ggml_backend_sparse_attn_prefer_dense_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_sparse_attn_prefer_dense");
    if (!group || !dense) { fprintf(stderr, "procs missing\n"); return 1; }
    const ggml_type T[4] = { GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 };
    const int64_t nqs[] = { 1, 2, 4, 8, 16, 32, 64, 96, 127, 128, 255, 256, 511, 512, 1024, 4096 };
    const int64_t kvs[] = { 2304, 8192, 16384, 32768, 49152, 65536, 98304, 135168, 262144, 1048576 };
    int bad = 0;
    for (int shortcut = 0; shortcut < 2; ++shortcut) {
        for (int t = 0; t < 4; ++t) {
            for (int64_t kv : kvs) {
                printf("GROUP shortcut %d V %-5s n_kv %8lld:", shortcut, ggml_type_name(T[t]), (long long) kv);
                for (int64_t nq : nqs) printf(" %lld:%d", (long long) nq, group(dev, nq, kv, T[t], T[t], shortcut));
                printf("\n");
            }
        }
    }
    for (const char * f : { "g1", "g4" }) {
        setenv("GGML_CUDA_SA_FORCE", f, 1);
        for (int64_t nq : nqs) for (int64_t kv : kvs) bad += group(dev, nq, kv, T[2], T[2], false) != f[1] - '0';
    }
    unsetenv("GGML_CUDA_SA_FORCE");
    for (int k = 0; k < 4; ++k) for (int v = 0; v < 4; ++v) {
        printf("DENSE K %-5s V %-5s:", ggml_type_name(T[k]), ggml_type_name(T[v]));
        for (int64_t nq : { 1, 4, 16, 64, 256, 768, 2048 }) printf(" %d", (int) dense(dev, T[k], T[v], nq, 2048));
        printf("\n");
    }
    printf("FORCE checks wrong %d -> %s\n", bad, bad ? "FAIL" : "OK");
    return bad ? 1 : 0;
}
