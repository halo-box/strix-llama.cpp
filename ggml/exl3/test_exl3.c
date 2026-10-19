// EXL3 decode self-test.
//
// The golden hashes come from exllamav3 tests/testlib/trellis.py (a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a) on the
// same xorshift32 inputs: decode() for the codebook tables, dequant_rotated() for W_hat. The W golden is the f32 rounding
// of diag(suh) H W_hat H diag(svh) evaluated exactly (integers) on that W_hat; the float64 dequant() of the oracle
// differs from it only by its own rounding (160 of the 884736 values, by one f32 ulp or near zero).
//
// The op tests compare ggml_exl3_mul_mat with ggml_mul_mat on the f32 W of ggml_exl3_dequantize. ggml_mul_mat runs on
// the CPU backend; the op runs there too, or on the backend given with --backend. A case passes when the op is within
// NMSE 1e-10 of ggml_mul_mat (f32 grade), or when its NMSE against a double sum on the first rows is at most twice that
// of the backend's own ggml_mul_mat on the same f32 W, x and shape (the GPU prefill GEMMs of this tree round both
// operands to f16; the backend is opted in to them as llama opts in its contexts).
//
// The row-ring goldens come from exllamav3 modules/quant/exl3_lib/ngram_codec.py dequant_rows (same pin) on the same
// xorshift32 inputs: n-gram rows of 160 with a bias, and token embedding rows of 4 rings of 256 with +/-1 signs, cropped.
//
// test-exl3                -> codebook tables, every rate x codebook on synthetic tiles, shape checks, op on synthetic tiles,
//                             row rings for K = 1..8, EXL3_GET_ROWS against the row decoder, EXL3_MUL_MAT_ID against
//                             EXL3_MUL_MAT per expert (bit for bit) and ggml_mul_mat_id on the f32 W of the experts
// test-exl3 --gguf FILE    -> decode every EXL3 linear of a GGUF container, one JSON line per tensor with sha256
// test-exl3 --op-gguf FILE -> op on the first linear of each (k, n, K) in a GGUF container, at the decode and prompt row counts
// test-exl3 --exact-gguf FILE -> op on every linear of a container on one-hot rows, where it is exact: equal to the CPU op
//                             bit for bit iff the backend decodes W_hat as the CPU does (see exact_gguf)
// test-exl3 --rows-gguf FILE -> EXL3_GET_ROWS on every row of each row table of a container, against the row decoder
// test-exl3 --dequantize IN OUT -> plain GGUF reference of a container: each EXL3 linear becomes "<base>.weight" F32 from
//                             ggml_exl3_dequantize, the exl3.* keys are dropped, all other keys and tensors are copied as they are
// test-exl3 --backend NAME [--op-gguf FILE] -> the op tests with the op on backend NAME (e.g. ROCm0)

#include "exl3.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "hash/sha256/sha256.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#    define exl3_fseek(f, off) _fseeki64(f, (__int64) (off), SEEK_SET)
#else
#    include <sys/types.h>
#    define exl3_fseek(f, off) fseeko(f, (off_t) (off), SEEK_SET)
#endif

static uint64_t fnv1a64(const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

static uint32_t xorshift32(uint32_t * s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

// normal fp16 with magnitude in [2^-5, 2^2)
static ggml_fp16_t rand_f16(uint32_t * s) {
    const uint32_t r = xorshift32(s);
    const uint16_t bits = (uint16_t) (((r >> 31) << 15) | ((10 + (r >> 8) % 7) << 10) | (r & 0x3ff));
    ggml_fp16_t h;
    memcpy(&h, &bits, sizeof(h));
    return h;
}

static const struct { int cb; uint64_t hash; } codebook_goldens[] = {
    { 0, 0xfad431e7e9266898ull }, // 3inst
    { 1, 0x7bb36b05a3b885a7ull }, // mcg
    { 2, 0x55e1c748ed9f0cd7ull }, // mul1
};

static const struct { int tile_words; int cb; uint64_t w_hat; uint64_t w; } tile_goldens[] = {
    {  16, 0, 0x74e599a34f95e431ull, 0x6cc7a0e65db6ddbdull },
    {  16, 1, 0xe4fe1f740825c2bfull, 0xc8a833ae9e6d2e03ull },
    {  16, 2, 0x87e6960dc9eeacd5ull, 0xd5ce1a6ba594d3acull },
    {  24, 2, 0x440f66937a6e7d8cull, 0xad41919a4971d0c7ull },
    {  32, 0, 0xd93db145e8ac28dbull, 0x8af112811c15655eull },
    {  32, 1, 0x373667cf0a675e92ull, 0x79f9d2696e16af78ull },
    {  32, 2, 0xed50e4d2c8f993dbull, 0xb73f4d3bebbd4606ull },
    {  40, 2, 0x844b9f574394ed7cull, 0x0182743d4a3081adull },
    {  48, 0, 0xf0016aa51bcbb576ull, 0x148462040b4a30d3ull },
    {  48, 1, 0xdd4f427e2c2a6c6dull, 0x3653cd2ed1650618ull },
    {  48, 2, 0x968f15287bed92e6ull, 0xb5926e8935fc0033ull },
    {  56, 2, 0x4981dd0f8cde68a4ull, 0x99f9eabef86c6a0aull },
    {  64, 0, 0x0522a294340a1828ull, 0xba4288e2fa48bccbull },
    {  64, 1, 0x33aa888f7a3390d7ull, 0x7d4776b7b2095931ull },
    {  64, 2, 0xb43dfb8577e89638ull, 0xb0d422c4cffefd73ull },
    {  80, 0, 0x1ea2f7218ebac349ull, 0x9d704cfba626fe3eull },
    {  80, 1, 0x0b279a2f23f01490ull, 0x860a49203c03ba58ull },
    {  80, 2, 0xcbf9350492d69652ull, 0x15abe1e1fad48785ull },
    {  96, 0, 0x75633b219d9c9dadull, 0xd9e875175a1cdc23ull },
    {  96, 1, 0x215a9dcf441b3103ull, 0x244fb85fe437c023ull },
    {  96, 2, 0x20e48230b3c615adull, 0xd70d6ea444373454ull },
    { 112, 0, 0x5cc35dbd721343ebull, 0x1c153687c1ec1320ull },
    { 112, 1, 0xdb4ab36b0cb0454cull, 0x0769f5ac9078d700ull },
    { 112, 2, 0x1609d106d33f95f2ull, 0x73bc3ca6ff1afa49ull },
    { 128, 0, 0xb1c9efecd1e72ad1ull, 0xe56e8502e9afb882ull },
    { 128, 1, 0x783e7a28d589bc43ull, 0xa2226052a3c22aa1ull },
    { 128, 2, 0xc1ceac9d5293bbffull, 0x914b948352a7c99dull },
};

static int test_codebooks(void) {
    int n_fail = 0;
    ggml_fp16_t * table = (ggml_fp16_t *) malloc(65536*sizeof(ggml_fp16_t));
    for (size_t i = 0; i < sizeof(codebook_goldens)/sizeof(codebook_goldens[0]); i++) {
        const enum ggml_exl3_codebook cb = (enum ggml_exl3_codebook) codebook_goldens[i].cb;
        for (uint32_t s = 0; s < 65536; s++) {
            table[s] = ggml_exl3_codebook_value((uint16_t) s, cb);
        }
        const uint64_t h = fnv1a64(table, 65536*sizeof(ggml_fp16_t));
        if (h != codebook_goldens[i].hash) {
            fprintf(stderr, "codebook %s: hash %016llx, expected %016llx\n", ggml_exl3_codebook_name(cb),
                    (unsigned long long) h, (unsigned long long) codebook_goldens[i].hash);
            n_fail++;
        }
    }
    free(table);
    return n_fail;
}

static int test_tiles(void) {
    const int64_t tk = 8;
    const int64_t tn = 16;
    const int64_t k  = tk*16;
    const int64_t n  = tn*16;

    int n_fail = 0;
    ggml_fp16_t * w_hat = (ggml_fp16_t *) malloc(k*n*sizeof(ggml_fp16_t));
    float       * w     = (float *)       malloc(k*n*sizeof(float));
    int16_t     * trellis = (int16_t *)   malloc(tk*tn*128*sizeof(int16_t));
    ggml_fp16_t suh[128];
    ggml_fp16_t svh[256];

    for (size_t i = 0; i < sizeof(tile_goldens)/sizeof(tile_goldens[0]); i++) {
        const int tile_words = tile_goldens[i].tile_words;
        const enum ggml_exl3_codebook cb = (enum ggml_exl3_codebook) tile_goldens[i].cb;

        uint32_t s = 0x12345678u + 0x9E3779B9u*(uint32_t) (tile_words*4 + cb);
        if (s == 0) {
            s = 1;
        }
        for (int64_t j = 0; j < tk*tn*tile_words; j++) {
            trellis[j] = (int16_t) (uint16_t) (xorshift32(&s) >> 16);
        }
        for (int64_t j = 0; j < k; j++) {
            suh[j] = rand_f16(&s);
        }
        for (int64_t j = 0; j < n; j++) {
            svh[j] = rand_f16(&s);
        }

        ggml_exl3_decode_rotated(trellis, k, n, tile_words, cb, w_hat);
        ggml_exl3_dequantize(trellis, suh, svh, k, n, tile_words, cb, w);

        const uint64_t h_hat = fnv1a64(w_hat, k*n*sizeof(ggml_fp16_t));
        const uint64_t h_w   = fnv1a64(w, k*n*sizeof(float));
        if (h_hat != tile_goldens[i].w_hat || h_w != tile_goldens[i].w) {
            fprintf(stderr, "K = %.1f %s: W_hat %016llx (expected %016llx), W %016llx (expected %016llx)\n",
                    tile_words/16.0, ggml_exl3_codebook_name(cb),
                    (unsigned long long) h_hat, (unsigned long long) tile_goldens[i].w_hat,
                    (unsigned long long) h_w, (unsigned long long) tile_goldens[i].w);
            n_fail++;
        }
    }

    free(trellis);
    free(w);
    free(w_hat);
    return n_fail;
}

// K, n-gram rows (8 x 160 with bias), token embedding rows (8 x 4 groups of 256, cropped to 1000)
static const struct { int K; uint64_t ngram; uint64_t embd; } row_goldens[] = {
    { 1, 0x04d0e311747c42acull, 0xc56102632334f75dull },
    { 2, 0x0c2b81394354d778ull, 0xe604fba14b513850ull },
    { 3, 0xda80b62104bc4600ull, 0xed6c73a527107c7cull },
    { 4, 0xc5881597b3e8cff1ull, 0x93bcd19561debb0cull },
    { 5, 0x56e10bb0b45f04e9ull, 0xd78ef9acd5b346daull },
    { 6, 0x125f6e56569f7d4cull, 0xec14ef4163ff2e39ull },
    { 7, 0x28e8389972f2d894ull, 0x209508df938773e2ull },
    { 8, 0x82d5b1b4868cd200ull, 0x537e53ff18bdea0aull },
};

static int test_rows(void) {
    enum { R = 8, GROUPS = 4, HIDDEN = 1000 };
    int n_fail = 0;
    for (size_t i = 0; i < sizeof(row_goldens)/sizeof(row_goldens[0]); i++) {
        const int K = row_goldens[i].K;
        uint32_t s = 0x2468ACE0u + 0x9E3779B9u*(uint32_t) K;

        const int64_t words = ggml_exl3_ring_words(GGML_EXL3_ROW_DIM, K);
        int16_t     * ng    = (int16_t *)     malloc(R*words*sizeof(int16_t));
        ggml_fp16_t * bias  = (ggml_fp16_t *) malloc(R*GGML_EXL3_ROW_DIM*sizeof(ggml_fp16_t));
        float       * o_ng  = (float *)       malloc(R*GGML_EXL3_ROW_DIM*sizeof(float));
        for (int r = 0; r < R; r++) {
            const ggml_fp16_t sc = rand_f16(&s);
            memcpy(&ng[r*words], &sc, sizeof(sc));
            for (int64_t j = 1; j < words; j++) {
                ng[r*words + j] = (int16_t) (uint16_t) (xorshift32(&s) >> 16);
            }
        }
        for (int j = 0; j < R*GGML_EXL3_ROW_DIM; j++) {
            bias[j] = rand_f16(&s);
        }
        for (int r = 0; r < R; r++) {
            ggml_exl3_decode_ngram_row(&ng[r*words], K, &bias[r*GGML_EXL3_ROW_DIM], &o_ng[r*GGML_EXL3_ROW_DIM]);
        }

        const int64_t gw    = ggml_exl3_ring_words(GGML_EXL3_GROUP_DIM, K);
        ggml_fp16_t * signs = (ggml_fp16_t *) malloc(GROUPS*GGML_EXL3_GROUP_DIM*sizeof(ggml_fp16_t));
        int16_t     * em    = (int16_t *)     malloc(R*GROUPS*gw*sizeof(int16_t));
        float       * o_em  = (float *)       malloc(R*HIDDEN*sizeof(float));
        for (int j = 0; j < GROUPS*GGML_EXL3_GROUP_DIM; j++) {
            const uint16_t b = (xorshift32(&s) >> 31) ? 0xBC00 : 0x3C00;
            memcpy(&signs[j], &b, sizeof(b));
        }
        for (int r = 0; r < R*GROUPS; r++) {
            const ggml_fp16_t sc = rand_f16(&s);
            memcpy(&em[r*gw], &sc, sizeof(sc));
            for (int64_t j = 1; j < gw; j++) {
                em[r*gw + j] = (int16_t) (uint16_t) (xorshift32(&s) >> 16);
            }
        }
        n_fail += ggml_exl3_embd_k(GROUPS*gw, GROUPS) != K;
        for (int r = 0; r < R; r++) {
            ggml_exl3_decode_embd_row(&em[r*GROUPS*gw], GROUPS, K, signs, HIDDEN, &o_em[r*HIDDEN]);
        }

        const uint64_t h_ng = fnv1a64(o_ng, R*GGML_EXL3_ROW_DIM*sizeof(float));
        const uint64_t h_em = fnv1a64(o_em, R*HIDDEN*sizeof(float));
        if (h_ng != row_goldens[i].ngram || h_em != row_goldens[i].embd) {
            fprintf(stderr, "rows K = %d: n-gram %016llx (want %016llx), embd %016llx (want %016llx)\n", K,
                    (unsigned long long) h_ng, (unsigned long long) row_goldens[i].ngram,
                    (unsigned long long) h_em, (unsigned long long) row_goldens[i].embd);
            n_fail++;
        }
        free(ng);
        free(bias);
        free(o_ng);
        free(signs);
        free(em);
        free(o_em);
    }

    // head of an n-gram row: the last offset at or below it
    const int64_t offs[3] = { 0, 10, 25 };
    n_fail += ggml_exl3_ngram_head(offs, 3, 0) != 0 || ggml_exl3_ngram_head(offs, 3, 9) != 0 ||
              ggml_exl3_ngram_head(offs, 3, 10) != 1 || ggml_exl3_ngram_head(offs, 3, 24) != 1 ||
              ggml_exl3_ngram_head(offs, 3, 25) != 2 || ggml_exl3_ngram_head(offs, 3, 1000) != 2;
    // embedding widths: groups of 1 + 16 K words
    n_fail += ggml_exl3_embd_k(260, 4) != 4 || ggml_exl3_embd_k(261, 4) != -1 || ggml_exl3_embd_k(4*(1 + 16*9), 4) != -1;
    return n_fail;
}

static int test_checks(void) {
    int n_fail = 0;

    n_fail += !ggml_exl3_rate_supported(64, GGML_EXL3_CODEBOOK_MCG);
    n_fail +=  ggml_exl3_rate_supported(40, GGML_EXL3_CODEBOOK_MCG);  // half rates need mul1
    n_fail += !ggml_exl3_rate_supported(40, GGML_EXL3_CODEBOOK_MUL1);
    n_fail +=  ggml_exl3_rate_supported(72, GGML_EXL3_CODEBOOK_MUL1); // K = 4.5
    n_fail +=  ggml_exl3_rate_supported(144, GGML_EXL3_CODEBOOK_MUL1);

    struct ggml_init_params params = { 16*ggml_tensor_overhead(), NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * trellis = ggml_new_tensor_3d(ctx, GGML_TYPE_I16, 64, 512, 160); // k = 2560, n = 8192, K = 4
    struct ggml_tensor * suh     = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 2560);
    struct ggml_tensor * svh     = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 8192);
    struct ggml_tensor * bad     = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 8192 - 128);
    struct ggml_tensor * odd     = ggml_new_tensor_3d(ctx, GGML_TYPE_I16, 64, 512, 161);

    n_fail += !ggml_exl3_check_linear(trellis, suh, svh, GGML_EXL3_CODEBOOK_MCG);
    n_fail +=  ggml_exl3_check_linear(trellis, suh, bad, GGML_EXL3_CODEBOOK_MCG);
    n_fail +=  ggml_exl3_check_linear(trellis, svh, suh, GGML_EXL3_CODEBOOK_MCG);
    n_fail +=  ggml_exl3_check_linear(odd,     suh, svh, GGML_EXL3_CODEBOOK_MCG);
    ggml_free(ctx);

    if (n_fail) {
        fprintf(stderr, "shape/rate checks: %d failures\n", n_fail);
    }
    return n_fail;
}

#define OP_NMSE_MAX   1e-10
#define OP_NMSE_CALIB 2.0 // allowed NMSE against the double sum, relative to the backend's own ggml_mul_mat

static int op_n_threads(void) {
    const char * env = getenv("EXL3_TEST_THREADS");
    return env ? atoi(env) : 16;
}

// backend of the op: CPU unless --backend NAME
static const char * op_backend_name = NULL;

static ggml_backend_t op_cpu_init(void) {
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, op_n_threads());
    return be;
}

static ggml_backend_t op_backend_init(void) {
    if (op_backend_name == NULL) {
        return op_cpu_init();
    }
    ggml_backend_t be = ggml_backend_init_by_name(op_backend_name, NULL);
    if (be == NULL) {
        fprintf(stderr, "no backend named %s\n", op_backend_name);
        return NULL;
    }
    // the BF16/F16 WMMA GEMMs (mmb) from 512 rows, as llama sets every context it creates for a model other than qwen4exp
    ggml_backend_dev_t dev = ggml_backend_get_device(be);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : NULL;
    static const char * const names[2] = { "ggml_backend_cuda_set_mmb_enabled", "ggml_backend_cuda_set_mmb_small_batch" };
    for (int i = 0; i < 2 && reg != NULL; i++) {
        void * p = ggml_backend_reg_get_proc_address(reg, names[i]);
        if (p != NULL) {
            void (*set)(ggml_backend_t, bool);
            memcpy(&set, &p, sizeof(set));
            set(be, i == 0);
        }
    }
    return be;
}

// activation in [-1, 1), one channel in 128 is 32 times larger
static float rand_act(uint32_t * s, int64_t channel) {
    const float u = (float) (xorshift32(s) >> 8)*(1.0f/8388608.0f) - 1.0f;
    return channel % 128 == 77 ? 32.0f*u : u;
}

static double nmse(const float * a, const float * b, int64_t n) {
    double num = 0.0;
    double den = 0.0;
    for (int64_t i = 0; i < n; i++) {
        const double d = (double) a[i] - (double) b[i];
        num += d*d;
        den += (double) b[i]*(double) b[i];
    }
    return den > 0.0 ? num/den : (num > 0.0 ? 1.0 : 0.0);
}

// one EXL3 linear in a buffer of the op backend, with its f32 W as a ggml [k, n] tensor for ggml_mul_mat in a CPU buffer
// and, when the op backend is not the CPU, a copy in a buffer of the op backend
struct op_weights {
    struct ggml_context * ctx;
    ggml_backend_buffer_t buf;
    struct ggml_context * ctx_w;
    ggml_backend_buffer_t buf_w;
    struct ggml_context * ctx_wb;
    ggml_backend_buffer_t buf_wb;
    struct ggml_tensor  * trellis;
    struct ggml_tensor  * suh;
    struct ggml_tensor  * svh;
    struct ggml_tensor  * w;
    struct ggml_tensor  * w_be;
    float               * w_t; // [n][k] host copy for the double sums
    int64_t               k;
    int64_t               n;
    int                   cb;
};

static bool op_weights_init(struct op_weights * ow, ggml_backend_t be, ggml_backend_t be_ref, const int16_t * trellis,
                            const ggml_fp16_t * suh, const ggml_fp16_t * svh, int64_t k, int64_t n, int64_t tile_words, int cb) {
    memset(ow, 0, sizeof(*ow));
    ow->k  = k;
    ow->n  = n;
    ow->cb = cb;

    float * w = (float *) malloc(k*n*sizeof(float));
    ow->w_t   = (float *) malloc(k*n*sizeof(float));
    if (w == NULL || ow->w_t == NULL) {
        free(w);
        return false;
    }
    ggml_exl3_dequantize(trellis, suh, svh, k, n, tile_words, (enum ggml_exl3_codebook) cb, w);
    for (int64_t i = 0; i < k; i++) {
        for (int64_t j = 0; j < n; j++) {
            ow->w_t[j*k + i] = w[i*n + j];
        }
    }
    free(w);

    struct ggml_init_params params = { 8*ggml_tensor_overhead(), NULL, true };
    ow->ctx     = ggml_init(params);
    ow->ctx_w   = ggml_init(params);
    ow->trellis = ggml_new_tensor_3d(ow->ctx, GGML_TYPE_I16, tile_words, n/GGML_EXL3_TILE, k/GGML_EXL3_TILE);
    ow->suh     = ggml_new_tensor_1d(ow->ctx, GGML_TYPE_F16, k);
    ow->svh     = ggml_new_tensor_1d(ow->ctx, GGML_TYPE_F16, n);
    ow->w       = ggml_new_tensor_2d(ow->ctx_w, GGML_TYPE_F32, k, n);
    ow->buf     = ggml_backend_alloc_ctx_tensors(ow->ctx, be);
    ow->buf_w   = ggml_backend_alloc_ctx_tensors(ow->ctx_w, be_ref);
    if (ow->buf == NULL || ow->buf_w == NULL) {
        return false;
    }
    ggml_backend_tensor_set(ow->trellis, trellis, 0, ggml_nbytes(ow->trellis));
    ggml_backend_tensor_set(ow->suh, suh, 0, ggml_nbytes(ow->suh));
    ggml_backend_tensor_set(ow->svh, svh, 0, ggml_nbytes(ow->svh));
    ggml_backend_tensor_set(ow->w, ow->w_t, 0, ggml_nbytes(ow->w));
    if (be != be_ref) {
        ow->ctx_wb = ggml_init(params);
        ow->w_be   = ggml_new_tensor_2d(ow->ctx_wb, GGML_TYPE_F32, k, n);
        ow->buf_wb = ggml_backend_alloc_ctx_tensors(ow->ctx_wb, be);
        if (ow->buf_wb == NULL) {
            return false;
        }
        ggml_backend_tensor_set(ow->w_be, ow->w_t, 0, ggml_nbytes(ow->w_be));
    }
    return true;
}

static void op_weights_free(struct op_weights * ow) {
    ggml_backend_buffer_free(ow->buf);
    ggml_backend_buffer_free(ow->buf_w);
    ggml_backend_buffer_free(ow->buf_wb);
    ggml_free(ow->ctx);
    ggml_free(ow->ctx_w);
    ggml_free(ow->ctx_wb);
    free(ow->w_t);
}

// nr rows of x; x_pad > 0 makes x a strided view whose padding is NaN
// the op runs on be, ggml_mul_mat on be_ref and, when ow->w_be is set, on be too; prints one JSON line, returns 1 on
// failure
static int op_check_rows(ggml_backend_t be, ggml_backend_t be_ref, const struct op_weights * ow, int64_t nr, int64_t x_pad,
                         uint32_t seed, int64_t n_true, const char * label) {
    const int64_t k  = ow->k;
    const int64_t n  = ow->n;
    const int64_t kx = k + x_pad;

    struct ggml_init_params params = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true };
    struct ggml_context * ctx     = ggml_init(params);
    struct ggml_context * ctx_ref = ggml_init(params);
    struct ggml_tensor * x_full = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kx, nr);
    struct ggml_tensor * x      = x_pad > 0 ? ggml_view_2d(ctx, x_full, k, nr, x_full->nb[1], 0) : x_full;
    struct ggml_tensor * y_op   = ggml_exl3_mul_mat(ctx, ow->trellis, x, ow->suh, ow->svh, ow->cb);
    struct ggml_tensor * y_be   = ow->w_be ? ggml_mul_mat(ctx, ow->w_be, x) : NULL;
    struct ggml_cgraph * gf     = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y_op);
    if (y_be) {
        ggml_build_forward_expand(gf, y_be);
    }
    struct ggml_tensor * x_rfull = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_F32, kx, nr);
    struct ggml_tensor * x_r     = x_pad > 0 ? ggml_view_2d(ctx_ref, x_rfull, k, nr, x_rfull->nb[1], 0) : x_rfull;
    struct ggml_tensor * y_ref   = ggml_mul_mat(ctx_ref, ow->w, x_r);
    struct ggml_cgraph * gf_ref  = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(gf_ref, y_ref);

    int fail = 1;
    float * xd   = (float *) malloc(kx*nr*sizeof(float));
    float * yo   = (float *) malloc(n*nr*sizeof(float));
    float * yr   = (float *) malloc(n*nr*sizeof(float));
    float * yb   = (float *) malloc(n*nr*sizeof(float));
    float * yt   = (float *) malloc(n*(n_true > 0 ? n_true : 1)*sizeof(float));
    ggml_backend_buffer_t buf     = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_buffer_t buf_ref = ggml_backend_alloc_ctx_tensors(ctx_ref, be_ref);
    if (!ggml_backend_supports_op(be, y_op)) {
        fprintf(stderr, "%s: EXL3_MUL_MAT is not supported by the %s backend\n", label, ggml_backend_name(be));
    } else if (buf != NULL && buf_ref != NULL && xd != NULL && yo != NULL && yr != NULL && yb != NULL && yt != NULL) {
        for (int64_t r = 0; r < nr; r++) {
            for (int64_t i = 0; i < kx; i++) {
                xd[r*kx + i] = i < k ? rand_act(&seed, i) : NAN;
            }
        }
        ggml_backend_tensor_set(x_full, xd, 0, ggml_nbytes(x_full));
        ggml_backend_tensor_set(x_rfull, xd, 0, ggml_nbytes(x_rfull));
        if (ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS &&
            ggml_backend_graph_compute(be_ref, gf_ref) == GGML_STATUS_SUCCESS) {
            ggml_backend_tensor_get(y_op, yo, 0, ggml_nbytes(y_op));
            ggml_backend_tensor_get(y_ref, yr, 0, ggml_nbytes(y_ref));
            ggml_backend_tensor_get(y_be ? y_be : y_ref, yb, 0, ggml_nbytes(y_ref));

            n_true = n_true < nr ? n_true : nr;
            for (int64_t r = 0; r < n_true; r++) {
                for (int64_t j = 0; j < n; j++) {
                    double s = 0.0;
                    const float * wj = ow->w_t + j*k;
                    const float * xr = xd + r*kx;
                    for (int64_t i = 0; i < k; i++) {
                        s += (double) xr[i]*(double) wj[i];
                    }
                    yt[r*n + j] = (float) s;
                }
            }
            const double e  = nmse(yo, yr, n*nr);
            const double eo = n_true > 0 ? nmse(yo, yt, n*n_true) : -1.0;
            const double er = n_true > 0 ? nmse(yr, yt, n*n_true) : -1.0;
            const double eb = n_true > 0 ? nmse(yb, yt, n*n_true) : -1.0;
            const bool   f32_grade  = e <= OP_NMSE_MAX;
            const bool   calibrated = n_true > 0 && eo <= OP_NMSE_CALIB*eb;
            fail = !(f32_grade || calibrated);
            printf("{\"case\": \"%s\", \"backend\": \"%s\", \"k\": %lld, \"n\": %lld, \"rows\": %lld, \"x_pad\": %lld, "
                   "\"nmse_op_vs_mul_mat\": %.3e, \"nmse_op_vs_f64\": %.3e, \"nmse_mul_mat_vs_f64\": %.3e, "
                   "\"nmse_be_mul_mat_vs_f64\": %.3e, \"op_over_be\": %.3f, \"f64_rows\": %lld, \"gate\": \"%s\", "
                   "\"pass\": %s}\n",
                   label, ggml_backend_name(be), (long long) k, (long long) n, (long long) nr, (long long) x_pad, e, eo, er,
                   eb, eb > 0.0 ? eo/eb : -1.0, (long long) n_true, f32_grade ? "f32" : calibrated ? "calibrated" : "none",
                   fail ? "false" : "true");
            fflush(stdout);
        }
    }

    free(xd);
    free(yo);
    free(yr);
    free(yb);
    free(yt);
    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(buf_ref);
    ggml_free(ctx);
    ggml_free(ctx_ref);
    return fail;
}

// every rate x codebook at k = 256, n = 384, row counts on both sides of the HIP GEMM threshold (32) and two strided x
static int test_op(void) {
    const int64_t tk = 16;
    const int64_t tn = 24;
    const int64_t k  = tk*16;
    const int64_t n  = tn*16;
    static const int64_t rows[] = { 1, 2, 3, 5, 8, 13, 32, 33, 130 };

    ggml_backend_t be     = op_backend_init();
    ggml_backend_t be_ref = op_cpu_init();
    if (be == NULL) {
        ggml_backend_free(be_ref);
        return 1;
    }

    int n_fail = 0;
    int n_run  = 0;
    int16_t     * trellis = (int16_t *) malloc(tk*tn*128*sizeof(int16_t));
    ggml_fp16_t * suh     = (ggml_fp16_t *) malloc(k*sizeof(ggml_fp16_t));
    ggml_fp16_t * svh     = (ggml_fp16_t *) malloc(n*sizeof(ggml_fp16_t));

    for (size_t i = 0; i < sizeof(tile_goldens)/sizeof(tile_goldens[0]); i++) {
        const int tile_words = tile_goldens[i].tile_words;
        const int cb         = tile_goldens[i].cb;

        uint32_t s = 0x2545F491u + 0x9E3779B9u*(uint32_t) (tile_words*4 + cb);
        for (int64_t j = 0; j < tk*tn*tile_words; j++) {
            trellis[j] = (int16_t) (uint16_t) (xorshift32(&s) >> 16);
        }
        for (int64_t j = 0; j < k; j++) {
            suh[j] = rand_f16(&s);
        }
        for (int64_t j = 0; j < n; j++) {
            svh[j] = rand_f16(&s);
        }

        struct op_weights ow;
        if (!op_weights_init(&ow, be, be_ref, trellis, suh, svh, k, n, tile_words, cb)) {
            fprintf(stderr, "op: weight setup failed\n");
            n_fail++;
            continue;
        }
        char label[64];
        snprintf(label, sizeof(label), "synthetic K=%.1f %s", tile_words/16.0, ggml_exl3_codebook_name((enum ggml_exl3_codebook) cb));
        for (size_t j = 0; j < sizeof(rows)/sizeof(rows[0]); j++) {
            n_fail += op_check_rows(be, be_ref, &ow, rows[j], 0, s + (uint32_t) j, rows[j], label);
            n_run++;
        }
        n_fail += op_check_rows(be, be_ref, &ow, 7, 160, s, 7, label);
        n_fail += op_check_rows(be, be_ref, &ow, 37, 160, s, 7, label);
        n_run += 2;
        op_weights_free(&ow);
    }

    free(trellis);
    free(suh);
    free(svh);
    ggml_backend_free(be);
    ggml_backend_free(be_ref);
    if (n_fail) {
        fprintf(stderr, "op: %d of %d cases failed\n", n_fail, n_run);
    }
    return n_fail;
}

static void sha256_hex(const void * data, size_t n, char * out) {
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_hash(digest, (const unsigned char *) data, n);
    for (int i = 0; i < SHA256_DIGEST_SIZE; i++) {
        snprintf(out + 2*i, 3, "%02x", digest[i]);
    }
}

static void * read_tensor(FILE * f, const struct gguf_context * gctx, const struct ggml_tensor * t) {
    const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
    const size_t  nb  = ggml_nbytes(t);
    void * buf = malloc(nb);
    if (tid < 0 || buf == NULL || exl3_fseek(f, gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, tid)) != 0 ||
            fread(buf, 1, nb, f) != nb) {
        free(buf);
        return NULL;
    }
    return buf;
}

// reads the metadata of an EXL3 container, returns its codebook or -1
static int open_container(const char * path, struct gguf_context ** gctx_out, struct ggml_context ** meta_out) {
    struct ggml_context * meta = NULL;
    struct gguf_init_params params = { true, &meta };
    struct gguf_context * gctx = gguf_init_from_file(path, params);
    if (gctx == NULL) {
        fprintf(stderr, "%s: cannot read GGUF\n", path);
        return -1;
    }

    const int64_t kid_ver = gguf_find_key(gctx, "exl3.version");
    const int64_t kid_cb  = gguf_find_key(gctx, "exl3.codebook");
    int cb = -1;
    if (kid_ver < 0 || gguf_get_kv_type(gctx, kid_ver) != GGUF_TYPE_UINT32 || gguf_get_val_u32(gctx, kid_ver) != 1 ||
            kid_cb < 0 || gguf_get_kv_type(gctx, kid_cb) != GGUF_TYPE_STRING) {
        fprintf(stderr, "%s: missing or unsupported exl3.version / exl3.codebook\n", path);
    } else if ((cb = ggml_exl3_codebook_from_name(gguf_get_val_str(gctx, kid_cb))) < 0) {
        fprintf(stderr, "%s: unknown exl3.codebook %s\n", path, gguf_get_val_str(gctx, kid_cb));
    }
    if (cb < 0) {
        gguf_free(gctx);
        ggml_free(meta);
        return -1;
    }
    *gctx_out = gctx;
    *meta_out = meta;
    return cb;
}

// for a "<base>.exl3_trellis" tensor: 1 and its base name and scales, -1 if the linear is malformed; 0 for other tensors
static int find_linear(struct ggml_context * meta, const struct ggml_tensor * t, int cb, char * base,
                       struct ggml_tensor ** su, struct ggml_tensor ** sv) {
    const char * suffix = ".exl3_trellis";
    const char * name   = ggml_get_name(t);
    const size_t len    = strlen(name);
    if (len <= strlen(suffix) || strcmp(name + len - strlen(suffix), suffix) != 0) {
        return 0;
    }
    char name_su[GGML_MAX_NAME + 16];
    char name_sv[GGML_MAX_NAME + 16];
    snprintf(base, GGML_MAX_NAME, "%.*s", (int) (len - strlen(suffix)), name);
    snprintf(name_su, sizeof(name_su), "%s.exl3_suh", base);
    snprintf(name_sv, sizeof(name_sv), "%s.exl3_svh", base);
    *su = ggml_get_tensor(meta, name_su);
    *sv = ggml_get_tensor(meta, name_sv);
    // one linear, or a stack of experts [tile_words, n/16, k/16, E] with suh [k, E] and svh [n, E]
    const int64_t n_exp = t->ne[3];
    const bool ok = *su != NULL && *sv != NULL && (n_exp == 1 ? ggml_exl3_check_linear(t, *su, *sv, (enum ggml_exl3_codebook) cb) :
        t->type == GGML_TYPE_I16 && (*su)->type == GGML_TYPE_F16 && (*sv)->type == GGML_TYPE_F16 &&
        ggml_exl3_rate_supported(t->ne[0], (enum ggml_exl3_codebook) cb) && t->ne[2] % 8 == 0 && t->ne[1] % 8 == 0 &&
        (*su)->ne[0] == t->ne[2]*GGML_EXL3_TILE && (*su)->ne[1] == n_exp && ggml_nelements(*su) == (*su)->ne[0]*n_exp &&
        (*sv)->ne[0] == t->ne[1]*GGML_EXL3_TILE && (*sv)->ne[1] == n_exp && ggml_nelements(*sv) == (*sv)->ne[0]*n_exp);
    if (!ok) {
        fprintf(stderr, "%s: bad EXL3 linear\n", base);
        return -1;
    }
    return 1;
}

static int check_gguf(const char * path) {
    struct gguf_context * gctx = NULL;
    struct ggml_context * meta = NULL;
    const int cb = open_container(path, &gctx, &meta);
    if (cb < 0) {
        return 1;
    }

    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        return 1;
    }

    int n_mod = 0;
    int n_err = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(meta); t != NULL; t = ggml_get_next_tensor(meta, t)) {
        char base[GGML_MAX_NAME];
        struct ggml_tensor * su = NULL;
        struct ggml_tensor * sv = NULL;
        const int found = find_linear(meta, t, cb, base, &su, &sv);
        if (found <= 0) {
            n_err += found < 0;
            continue;
        }

        const int64_t k     = t->ne[2]*GGML_EXL3_TILE;
        const int64_t n     = t->ne[1]*GGML_EXL3_TILE;
        const int64_t n_exp = t->ne[3];
        int16_t     * trellis = (int16_t *)     read_tensor(f, gctx, t);
        ggml_fp16_t * suh     = (ggml_fp16_t *) read_tensor(f, gctx, su);
        ggml_fp16_t * svh     = (ggml_fp16_t *) read_tensor(f, gctx, sv);
        ggml_fp16_t * w_hat   = (ggml_fp16_t *) malloc(n_exp*k*n*sizeof(ggml_fp16_t));
        float       * w       = (float *)       malloc(n_exp*k*n*sizeof(float));
        if (trellis == NULL || suh == NULL || svh == NULL || w_hat == NULL || w == NULL) {
            fprintf(stderr, "%s: read or alloc failed\n", base);
            n_err++;
        } else {
            char h_trellis[65], h_hat[65], h_w[65];
            // a stack: the experts one after the other
            for (int64_t e = 0; e < n_exp; e++) {
                const int16_t * te = trellis + e*(ggml_nelements(t)/n_exp);
                ggml_exl3_decode_rotated(te, k, n, t->ne[0], (enum ggml_exl3_codebook) cb, w_hat + e*k*n);
                ggml_exl3_dequantize(te, suh + e*k, svh + e*n, k, n, t->ne[0], (enum ggml_exl3_codebook) cb, w + e*k*n);
            }
            sha256_hex(trellis, ggml_nbytes(t), h_trellis);
            sha256_hex(w_hat, n_exp*k*n*sizeof(ggml_fp16_t), h_hat);
            sha256_hex(w, n_exp*k*n*sizeof(float), h_w);
            printf("{\"tensor\": \"%s\", \"k\": %lld, \"n\": %lld, \"experts\": %lld, \"tile_words\": %lld, \"codebook\": \"%s\", "
                   "\"trellis_sha256\": \"%s\", \"w_hat_f16_sha256\": \"%s\", \"w_f32_sha256\": \"%s\"}\n",
                   base, (long long) k, (long long) n, (long long) n_exp, (long long) t->ne[0],
                   ggml_exl3_codebook_name((enum ggml_exl3_codebook) cb), h_trellis, h_hat, h_w);
            fflush(stdout);
            n_mod++;
        }
        free(trellis);
        free(suh);
        free(svh);
        free(w_hat);
        free(w);
    }

    fclose(f);
    gguf_free(gctx);
    ggml_free(meta);
    fprintf(stderr, "%s: %d EXL3 linears decoded, %d errors\n", path, n_mod, n_err);
    return n_err != 0 || n_mod == 0;
}

// the op on the first linear of each (k, n, K): decode rows 1 to 8, both sides of the HIP GEMM threshold (32), a
// partial and a whole 128-row tile, ubatch 512 and past it
static int test_mul_mat_id_run(ggml_backend_t be, ggml_backend_t be_ref, const int16_t * trellis, const ggml_fp16_t * suh,
                               const ggml_fp16_t * svh, int64_t k, int64_t n, int64_t n_exp, int64_t tile_words, int cb,
                               int64_t n_used, int64_t ne11, int64_t n_tok, bool skew, uint32_t seed, const char * label);

static int op_gguf(const char * path) {
    static const int64_t rows[] = { 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 33, 64, 127, 128, 511, 512, 513 };

    struct gguf_context * gctx = NULL;
    struct ggml_context * meta = NULL;
    const int cb = open_container(path, &gctx, &meta);
    if (cb < 0) {
        return 1;
    }
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        return 1;
    }

    ggml_backend_t be     = op_backend_init();
    ggml_backend_t be_ref = op_cpu_init();
    if (be == NULL) {
        ggml_backend_free(be_ref);
        fclose(f);
        return 1;
    }

    int64_t seen[64][3];
    int n_seen = 0;
    int n_run  = 0;
    int n_fail = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(meta); t != NULL; t = ggml_get_next_tensor(meta, t)) {
        char base[GGML_MAX_NAME];
        struct ggml_tensor * su = NULL;
        struct ggml_tensor * sv = NULL;
        const int found = find_linear(meta, t, cb, base, &su, &sv);
        if (found <= 0) {
            n_fail += found < 0;
            continue;
        }
        const int64_t k     = t->ne[2]*GGML_EXL3_TILE;
        const int64_t n     = t->ne[1]*GGML_EXL3_TILE;
        const int64_t n_exp = t->ne[3];
        bool is_new = true;
        for (int i = 0; i < n_seen; i++) {
            is_new = is_new && !(seen[i][0] == k && seen[i][1] == n && seen[i][2] == t->ne[0]*1000 + n_exp);
        }
        if (!is_new || n_seen == 64) {
            continue;
        }
        seen[n_seen][0] = k;
        seen[n_seen][1] = n;
        seen[n_seen][2] = t->ne[0]*1000 + n_exp;
        n_seen++;

        int16_t     * trellis = (int16_t *)     read_tensor(f, gctx, t);
        ggml_fp16_t * suh     = (ggml_fp16_t *) read_tensor(f, gctx, su);
        ggml_fp16_t * svh     = (ggml_fp16_t *) read_tensor(f, gctx, sv);
        struct op_weights ow;
        if (n_exp > 1) {
            // a stack of experts: EXL3_MUL_MAT_ID at decode, draft-verification and prompt token counts (with 32 experts
            // and top-4: both sides of the HIP GEMM threshold, 12 rows per expert = 96 tokens), x broadcast and one per
            // slot; uniform routing, then skewed routing from 96 tokens (one expert in every token: one product from
            // MMID_ONE_ROWS tokens, others with many and few rows)
            const int64_t n_used = n_exp < 4 ? n_exp : 4;
            static const int64_t toks[] = { 1, 2, 4, 7, 32, 95, 96, 256, 257, 512, 96, 256, 512, 513 };
            static const size_t  n_uniform = 10;
            for (size_t j = 0; trellis != NULL && suh != NULL && svh != NULL && j < sizeof(toks)/sizeof(toks[0]); j++) {
                for (int64_t ne11 = 1; ne11 <= n_used; ne11 += n_used - 1 > 0 ? n_used - 1 : 1) {
                    n_fail += test_mul_mat_id_run(be, be_ref, trellis, suh, svh, k, n, n_exp, t->ne[0], cb, n_used, ne11,
                                                  toks[j], j >= n_uniform, 0x6A09E667u ^ (uint32_t) (n_seen*16 + j), base);
                    n_run++;
                }
            }
            n_fail += trellis == NULL || suh == NULL || svh == NULL;
        } else if (trellis == NULL || suh == NULL || svh == NULL ||
            !op_weights_init(&ow, be, be_ref, trellis, suh, svh, k, n, t->ne[0], cb)) {
            fprintf(stderr, "%s: read or setup failed\n", base);
            n_fail++;
        } else {
            uint32_t seed = 0x6A09E667u ^ (uint32_t) n_seen;
            for (size_t j = 0; j < sizeof(rows)/sizeof(rows[0]); j++) {
                n_fail += op_check_rows(be, be_ref, &ow, rows[j], 0, seed + (uint32_t) j, 4, base);
                n_run++;
            }
            n_fail += op_check_rows(be, be_ref, &ow, 5, 96, seed, 4, base);
            n_fail += op_check_rows(be, be_ref, &ow, 40, 96, seed, 4, base);
            n_run += 2;
            op_weights_free(&ow);
        }
        free(trellis);
        free(suh);
        free(svh);
    }

    ggml_backend_free(be);
    ggml_backend_free(be_ref);
    fclose(f);
    gguf_free(gctx);
    ggml_free(meta);
    fprintf(stderr, "%s: %d shapes, %d op cases, %d failures\n", path, n_seen, n_run, n_fail);
    return n_fail != 0 || n_run == 0;
}

// --exact-gguf: the op on the backend against the op on the CPU on inputs where every product is exact. With suh = 1 and
// x row r = row r%128 of the unnormalized 128-point Hadamard in k block r/128 (zeros elsewhere), H(x*suh) = 128 e_r, so
// each output is 128 W_hat[r, j] (one nonzero product, exact in f32) followed by the output rotation and svh, whose
// order and roundings the CPU op and the HIP dot kernel share: rows run in chunks of 32 (the dot kernel on HIP; for a
// stack, 8 tokens routed to all E experts) must equal the CPU op bit for bit, which holds iff the backend decodes
// every value of W_hat as the CPU decoder does (bit-exact against exllamav3's trellis.py, README). The WMMA paths
// (all k rows in one call; chunks of 511 rows, or 16 tokens for a stack) cannot be exact here: on gfx1151 the f16 WMMA
// returns about half of the negative single products one f32 ulp toward -inf (C = +0). They must stay within 2^-18 of
// the largest |value| of each output row; one fp16 ulp of a typical decoded value moves a row by ~3e-5 of its maximum.
// EXL3_EXACT_STRIDE=S checks every S-th linear of the container.

// one-hot rows r0 .. r0 + nr - 1 (k wide)
static void exact_rows(float * x, int64_t k, int64_t r0, int64_t nr) {
    memset(x, 0, k*nr*sizeof(float));
    for (int64_t r = 0; r < nr; r++) {
        const int64_t i = (r0 + r) % 128;
        float * xb = x + r*k + ((r0 + r) / 128)*128;
        for (int64_t j = 0; j < 128; j++) {
            int pc = 0;
            for (int64_t v = i & j; v != 0; v >>= 1) {
                pc += (int) (v & 1);
            }
            xb[j] = pc % 2 ? -1.0f : 1.0f;
        }
    }
}

// the op on be for nr rows of x: EXL3_MUL_MAT, or EXL3_MUL_MAT_ID with every expert per token (x broadcast); out is
// [nr][E][n]
static bool exact_run(ggml_backend_t be, const int16_t * trellis, const ggml_fp16_t * suh, const ggml_fp16_t * svh,
                      int64_t k, int64_t n, int64_t n_exp, int64_t tile_words, int cb, const float * x, int64_t nr,
                      float * out) {
    struct ggml_init_params params = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true };
    struct ggml_context * c   = ggml_init(params);
    struct ggml_tensor  * tr  = ggml_new_tensor_4d(c, GGML_TYPE_I16, tile_words, n/GGML_EXL3_TILE, k/GGML_EXL3_TILE, n_exp);
    struct ggml_tensor  * su  = n_exp == 1 ? ggml_new_tensor_1d(c, GGML_TYPE_F16, k) : ggml_new_tensor_2d(c, GGML_TYPE_F16, k, n_exp);
    struct ggml_tensor  * sv  = n_exp == 1 ? ggml_new_tensor_1d(c, GGML_TYPE_F16, n) : ggml_new_tensor_2d(c, GGML_TYPE_F16, n, n_exp);
    struct ggml_tensor  * xx  = ggml_new_tensor_3d(c, GGML_TYPE_F32, k, 1, nr);
    struct ggml_tensor  * ids = n_exp == 1 ? NULL : ggml_new_tensor_2d(c, GGML_TYPE_I32, n_exp, nr);
    struct ggml_tensor  * y   = n_exp == 1 ? ggml_exl3_mul_mat(c, tr, ggml_reshape_2d(c, xx, k, nr), su, sv, cb) :
                                             ggml_exl3_mul_mat_id(c, tr, xx, ids, su, sv, cb);
    struct ggml_cgraph  * g   = ggml_new_graph(c);
    ggml_build_forward_expand(g, y);
    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(c, be);
    bool ok = b != NULL && ggml_backend_supports_op(be, y);
    if (ok) {
        ggml_backend_tensor_set(tr, trellis, 0, ggml_nbytes(tr));
        ggml_backend_tensor_set(su, suh, 0, ggml_nbytes(su));
        ggml_backend_tensor_set(sv, svh, 0, ggml_nbytes(sv));
        ggml_backend_tensor_set(xx, x, 0, ggml_nbytes(xx));
        if (ids != NULL) {
            int32_t * id = (int32_t *) malloc(ggml_nbytes(ids));
            ok = id != NULL;
            for (int64_t i = 0; ok && i < n_exp*nr; i++) {
                id[i] = (int32_t) (i % n_exp);
            }
            if (ok) {
                ggml_backend_tensor_set(ids, id, 0, ggml_nbytes(ids));
            }
            free(id);
        }
        ok = ok && ggml_backend_graph_compute(be, g) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        ggml_backend_tensor_get(y, out, 0, ggml_nbytes(y));
    }
    ggml_backend_buffer_free(b);
    ggml_free(c);
    return ok;
}

static int exact_gguf(const char * path) {
    struct gguf_context * gctx = NULL;
    struct ggml_context * meta = NULL;
    const int cb = open_container(path, &gctx, &meta);
    if (cb < 0) {
        return 1;
    }
    FILE * f = fopen(path, "rb");
    ggml_backend_t be     = op_backend_init();
    ggml_backend_t be_cpu = op_cpu_init();
    if (f == NULL || be == NULL) {
        return 1;
    }
    const int stride = getenv("EXL3_EXACT_STRIDE") ? atoi(getenv("EXL3_EXACT_STRIDE")) : 1;

    int     n_lin  = 0;
    int     n_fail = 0;
    int64_t n_vals = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(meta); t != NULL; t = ggml_get_next_tensor(meta, t)) {
        char base[GGML_MAX_NAME];
        struct ggml_tensor * su = NULL;
        struct ggml_tensor * sv = NULL;
        const int found = find_linear(meta, t, cb, base, &su, &sv);
        if (found <= 0) {
            n_fail += found < 0;
            continue;
        }
        if (n_lin++ % (stride > 0 ? stride : 1) != 0) {
            continue;
        }
        const int64_t k     = t->ne[2]*GGML_EXL3_TILE;
        const int64_t n     = t->ne[1]*GGML_EXL3_TILE;
        const int64_t n_exp = t->ne[3];
        const int64_t width = n*n_exp; // output values per row
        int16_t     * trellis = (int16_t *)     read_tensor(f, gctx, t);
        ggml_fp16_t * svh     = (ggml_fp16_t *) read_tensor(f, gctx, sv);
        ggml_fp16_t * ones    = (ggml_fp16_t *) malloc(k*n_exp*sizeof(ggml_fp16_t));
        float       * x       = (float *)       malloc(k*k*sizeof(float));
        float       * ref     = (float *)       malloc(width*k*sizeof(float));
        float       * got     = (float *)       malloc(width*k*sizeof(float));
        bool ok = trellis != NULL && svh != NULL && ones != NULL && x != NULL && ref != NULL && got != NULL;
        for (int64_t i = 0; ok && i < k*n_exp; i++) {
            ones[i] = ggml_fp32_to_fp16(1.0f);
        }
        if (ok && getenv("EXL3_EXACT_SVH1") != NULL) {
            for (int64_t i = 0; i < n*n_exp; i++) {
                svh[i] = ggml_fp32_to_fp16(1.0f);
            }
        }
        if (ok) {
            exact_rows(x, k, 0, k);
            ok = exact_run(be_cpu, trellis, ones, svh, k, n, n_exp, t->ne[0], cb, x, k, ref);
        }
        // rows per call: all rows in one call (WMMA GEMM: one product for a dense linear, and for a stack, whose experts
        // then have k >= MMID_ONE_ROWS rows each); 511 (dense: the hi + lo GEMM) or 16 tokens (stack: 32-row groups);
        // 32 rows (dense) or 8 tokens (stack): the dot kernel, exact. A stack token goes to all E experts, so its rows
        // per expert are its token count
        const int64_t chunk[3] = { k, n_exp > 1 ? 16 : 511, n_exp > 1 ? 8 : 32 };
        int64_t diff[3] = { -1, -1, -1 };
        double  rel[3]  = { -1.0, -1.0, -1.0 };
        int64_t first_row = -1, first_col = -1;
        float   first_ref = 0.0f, first_got = 0.0f;
        for (int ps = 0; ok && ps < 3; ps++) {
            diff[ps] = 0;
            rel[ps]  = 0.0;
            for (int64_t r0 = 0; ok && r0 < k; r0 += chunk[ps]) {
                const int64_t nr = k - r0 < chunk[ps] ? k - r0 : chunk[ps];
                ok = exact_run(be, trellis, ones, svh, k, n, n_exp, t->ne[0], cb, x + r0*k, nr, got + r0*width);
            }
            for (int64_t r = 0; ok && r < k; r++) {
                for (int64_t e = 0; e < n_exp; e++) {
                    // one output row: the n values of row r for expert e
                    const float * rr = ref + r*width + e*n;
                    const float * gg = got + r*width + e*n;
                    double amax = 0.0;
                    for (int64_t j = 0; j < n; j++) {
                        const double av = fabs((double) rr[j]);
                        amax = av > amax ? av : amax;
                    }
                    for (int64_t j = 0; j < n; j++) {
                        if (memcmp(rr + j, gg + j, sizeof(float)) != 0) {
                            if (ps == 2 && first_row < 0) {
                                first_row = r;
                                first_col = e*n + j;
                                first_ref = rr[j];
                                first_got = gg[j];
                            }
                            diff[ps]++;
                            const double d = fabs((double) rr[j] - (double) gg[j])/(amax > 0.0 ? amax : 1.0);
                            rel[ps] = d > rel[ps] ? d : rel[ps];
                        }
                    }
                }
            }
            n_vals += k*width;
        }
        const double tol  = 1.0/262144.0;
        const bool   pass = ok && diff[2] == 0 && rel[0] <= tol && rel[1] <= tol;
        // the stack passes rely on the HIP routing thresholds (12 and 32 rows per expert, EXL3_MMID_*_ROWS in exl3.cu)
        n_fail += !pass;
        printf("{\"case\": \"exact %s\", \"backend\": \"%s\", \"k\": %lld, \"n\": %lld, \"experts\": %lld, \"K\": %.1f, "
               "\"dot_diff\": %lld, \"gemm_all_diff\": %lld, \"gemm_all_rel\": %.3e, \"gemm_chunk_diff\": %lld, \"gemm_chunk_rel\": %.3e, "
               "\"dot_first_row\": %lld, \"dot_first_col\": %lld, \"dot_first_ref\": %.9g, \"dot_first_got\": %.9g, \"pass\": %s}\n",
               base, ggml_backend_name(be), (long long) k, (long long) n, (long long) n_exp, t->ne[0]/16.0,
               (long long) diff[2], (long long) diff[0], rel[0], (long long) diff[1], rel[1], (long long) first_row,
               (long long) first_col, (double) first_ref, (double) first_got, pass ? "true" : "false");
        fflush(stdout);
        free(trellis);
        free(svh);
        free(ones);
        free(x);
        free(ref);
        free(got);
    }
    ggml_backend_free(be);
    ggml_backend_free(be_cpu);
    fclose(f);
    gguf_free(gctx);
    ggml_free(meta);
    fprintf(stderr, "%s: %d linears (every %d-th checked), %lld values compared, %d failures\n", path, n_lin, stride,
            (long long) n_vals, n_fail);
    return n_fail != 0;
}

// --rows-gguf: EXL3_GET_ROWS on the backend for every row of each row table ("<base>.exl3r_trellis" with its signs) of a
// container, against ggml_exl3_decode_embd_row on the CPU, bit for bit; the rows are decoded to n_embd (the
// architecture's embedding_length), in batches of 4096 ids
static int rows_gguf(const char * path) {
    struct ggml_context * meta = NULL;
    struct gguf_init_params params = { true, &meta };
    struct gguf_context * gctx = gguf_init_from_file(path, params);
    FILE * f = fopen(path, "rb");
    ggml_backend_t be = op_backend_init();
    if (gctx == NULL || f == NULL || be == NULL) {
        return 1;
    }
    const int64_t karch = gguf_find_key(gctx, "general.architecture");
    char key[256];
    snprintf(key, sizeof(key), "%s.embedding_length", karch >= 0 ? gguf_get_val_str(gctx, karch) : "");
    const int64_t kembd  = gguf_find_key(gctx, key);
    const int64_t n_embd = kembd >= 0 ? (int64_t) gguf_get_val_u32(gctx, kembd) : 0;

    int     n_tables = 0;
    int     n_fail   = 0;
    int64_t n_vals   = 0;
    const char * suffix = ".exl3r_trellis";
    for (struct ggml_tensor * t = ggml_get_first_tensor(meta); t != NULL; t = ggml_get_next_tensor(meta, t)) {
        const char * name = ggml_get_name(t);
        const size_t len  = strlen(name);
        if (len <= strlen(suffix) || strcmp(name + len - strlen(suffix), suffix) != 0) {
            continue;
        }
        char name_sg[GGML_MAX_NAME + 16];
        snprintf(name_sg, sizeof(name_sg), "%.*s.exl3r_signs", (int) (len - strlen(suffix)), name);
        struct ggml_tensor * sg = ggml_get_tensor(meta, name_sg);
        const int K = sg != NULL ? ggml_exl3_embd_k(t->ne[0], sg->ne[1]) : -1;
        n_tables++;
        if (K <= 0 || n_embd <= 0 || n_embd > sg->ne[1]*GGML_EXL3_GROUP_DIM) {
            fprintf(stderr, "%s: bad row table\n", name);
            n_fail++;
            continue;
        }
        int16_t     * table = (int16_t *)     read_tensor(f, gctx, t);
        ggml_fp16_t * signs = (ggml_fp16_t *) read_tensor(f, gctx, sg);
        const int64_t n_rows = t->ne[1];
        const int64_t batch  = 4096;
        float   * y     = (float *) malloc(batch*n_embd*sizeof(float));
        float   * y_ref = (float *) malloc(n_embd*sizeof(float));
        int32_t * ids   = (int32_t *) malloc(batch*sizeof(int32_t));
        bool    ok      = table != NULL && signs != NULL && y != NULL && y_ref != NULL && ids != NULL;
        int64_t n_diff  = 0;
        int64_t first   = -1;

        struct ggml_init_params p = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true };
        struct ggml_context * ctx = ggml_init(p);
        struct ggml_tensor  * t_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I16, t->ne[0], n_rows);
        struct ggml_tensor  * t_signs = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, GGML_EXL3_GROUP_DIM, sg->ne[1]);
        struct ggml_tensor  * t_ids   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch);
        struct ggml_tensor  * out     = ggml_exl3_get_rows(ctx, t_table, t_ids, t_signs, n_embd);
        struct ggml_cgraph  * gf      = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ok = ok && buf != NULL && ggml_backend_supports_op(be, out);
        if (ok) {
            ggml_backend_tensor_set(t_table, table, 0, ggml_nbytes(t_table));
            ggml_backend_tensor_set(t_signs, signs, 0, ggml_nbytes(t_signs));
        }
        for (int64_t r0 = 0; ok && r0 < n_rows; r0 += batch) {
            // the last batch repeats the last row
            for (int64_t i = 0; i < batch; i++) {
                ids[i] = (int32_t) (r0 + i < n_rows ? r0 + i : n_rows - 1);
            }
            ggml_backend_tensor_set(t_ids, ids, 0, ggml_nbytes(t_ids));
            ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
            if (ok) {
                ggml_backend_tensor_get(out, y, 0, ggml_nbytes(out));
            }
            for (int64_t i = 0; ok && i < batch && r0 + i < n_rows; i++) {
                ggml_exl3_decode_embd_row(table + (r0 + i)*t->ne[0], sg->ne[1], K, signs, n_embd, y_ref);
                for (int64_t j = 0; j < n_embd; j++) {
                    if (memcmp(y_ref + j, y + i*n_embd + j, sizeof(float)) != 0) {
                        first = first < 0 ? r0 + i : first;
                        n_diff++;
                    }
                }
                n_vals += n_embd;
            }
        }
        const bool pass = ok && n_diff == 0;
        n_fail += !pass;
        printf("{\"case\": \"rows %s\", \"backend\": \"%s\", \"rows\": %lld, \"groups\": %lld, \"K\": %d, \"n_out\": %lld, "
               "\"values_differing\": %lld, \"first_row\": %lld, \"pass\": %s}\n",
               name, ggml_backend_name(be), (long long) n_rows, (long long) sg->ne[1], K, (long long) n_embd,
               (long long) n_diff, (long long) first, pass ? "true" : "false");
        fflush(stdout);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        free(table);
        free(signs);
        free(y);
        free(y_ref);
        free(ids);
    }
    ggml_backend_free(be);
    fclose(f);
    gguf_free(gctx);
    ggml_free(meta);
    fprintf(stderr, "%s: %d row tables, %lld values compared, %d failures\n", path, n_tables, (long long) n_vals, n_fail);
    return n_fail != 0 || n_tables == 0;
}

static bool write_zeros(FILE * f, size_t n) {
    static const char zeros[64] = { 0 };
    while (n > 0) {
        const size_t m = n < sizeof(zeros) ? n : sizeof(zeros);
        if (fwrite(zeros, 1, m, f) != m) {
            return false;
        }
        n -= m;
    }
    return true;
}

static bool ends_with(const char * s, const char * suffix) {
    const size_t ls = strlen(s);
    const size_t lx = strlen(suffix);
    return ls >= lx && strcmp(s + ls - lx, suffix) == 0;
}

static int dequantize_gguf(const char * path_in, const char * path_out) {
    struct gguf_context * gctx = NULL;
    struct ggml_context * meta = NULL;
    const int cb = open_container(path_in, &gctx, &meta);
    if (cb < 0) {
        return 1;
    }

    struct gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, gctx);
    for (int64_t i = 0; i < gguf_get_n_kv(gctx); i++) {
        if (strncmp(gguf_get_key(gctx, i), "exl3.", 5) == 0) {
            gguf_remove_key(out, gguf_get_key(gctx, i));
        }
    }

    // tensor order of the container; the weight of a linear takes the place of its trellis
    struct ggml_init_params params = { (size_t) gguf_get_n_tensors(gctx)*ggml_tensor_overhead(), NULL, true };
    struct ggml_context * ctx_w = ggml_init(params);
    int n_lin    = 0;
    int n_scales = 0;
    int n_err    = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(meta); t != NULL; t = ggml_get_next_tensor(meta, t)) {
        char base[GGML_MAX_NAME];
        char wname[GGML_MAX_NAME + 8];
        struct ggml_tensor * su = NULL;
        struct ggml_tensor * sv = NULL;
        const int found = find_linear(meta, t, cb, base, &su, &sv);
        if (found < 0) {
            n_err++;
        } else if (found > 0) {
            snprintf(wname, sizeof(wname), "%s.weight", base);
            if (gguf_find_tensor(gctx, wname) >= 0 || strlen(wname) >= GGML_MAX_NAME) {
                fprintf(stderr, "%s: cannot add %s\n", path_in, wname);
                n_err++;
                continue;
            }
            struct ggml_tensor * w = ggml_new_tensor_3d(ctx_w, GGML_TYPE_F32, t->ne[2]*GGML_EXL3_TILE, t->ne[1]*GGML_EXL3_TILE, t->ne[3]);
            ggml_set_name(w, wname);
            gguf_add_tensor(out, w);
            n_lin++;
        } else if (ends_with(ggml_get_name(t), ".exl3_suh") || ends_with(ggml_get_name(t), ".exl3_svh")) {
            n_scales++;
        } else {
            gguf_add_tensor(out, t);
        }
    }
    if (n_err != 0 || n_lin == 0 || n_scales != 2*n_lin) {
        fprintf(stderr, "%s: %d errors, %d linears, %d scale vectors\n", path_in, n_err, n_lin, n_scales);
        return 1;
    }

    FILE * fi = fopen(path_in, "rb");
    FILE * fo = fopen(path_out, "wb");
    if (fi == NULL || fo == NULL) {
        fprintf(stderr, "cannot open %s or %s\n", path_in, path_out);
        return 1;
    }
    const size_t meta_size = gguf_get_meta_size(out);
    void * md = malloc(meta_size);
    gguf_get_meta_data(out, md);
    bool ok = fwrite(md, 1, meta_size, fo) == meta_size;
    free(md);

    const size_t align = gguf_get_alignment(out);
    size_t expected = meta_size;
    for (int64_t i = 0; ok && i < gguf_get_n_tensors(out); i++) {
        const char * name = gguf_get_tensor_name(out, i);
        const size_t size = gguf_get_tensor_size(out, i);
        char tname[GGML_MAX_NAME + 16];
        snprintf(tname, sizeof(tname), "%.*s.exl3_trellis", (int) (strlen(name) - (ends_with(name, ".weight") ? strlen(".weight") : 0)), name);
        struct ggml_tensor * t = ends_with(name, ".weight") ? ggml_get_tensor(meta, tname) : NULL;
        if (t != NULL) {
            char base[GGML_MAX_NAME];
            struct ggml_tensor * su = NULL;
            struct ggml_tensor * sv = NULL;
            find_linear(meta, t, cb, base, &su, &sv);
            const int64_t k     = t->ne[2]*GGML_EXL3_TILE;
            const int64_t n     = t->ne[1]*GGML_EXL3_TILE;
            const int64_t n_exp = t->ne[3];
            int16_t     * trellis = (int16_t *)     read_tensor(fi, gctx, t);
            ggml_fp16_t * suh     = (ggml_fp16_t *) read_tensor(fi, gctx, su);
            ggml_fp16_t * svh     = (ggml_fp16_t *) read_tensor(fi, gctx, sv);
            float       * w       = (float *)       malloc(k*n*sizeof(float));
            float       * w_t     = (float *)       malloc(k*n*sizeof(float));
            ok = trellis != NULL && suh != NULL && svh != NULL && w != NULL && w_t != NULL && size == (size_t) (n_exp*k*n)*sizeof(float);
            for (int64_t e = 0; ok && e < n_exp; e++) {
                // ggml rows are the n output columns of W, one expert after the other
                ggml_exl3_dequantize(trellis + e*(ggml_nelements(t)/n_exp), suh + e*k, svh + e*n, k, n, t->ne[0], (enum ggml_exl3_codebook) cb, w);
                for (int64_t r = 0; r < k; r++) {
                    for (int64_t c = 0; c < n; c++) {
                        w_t[c*k + r] = w[r*n + c];
                    }
                }
                ok = fwrite(w_t, 1, k*n*sizeof(float), fo) == k*n*sizeof(float);
            }
            free(trellis);
            free(suh);
            free(svh);
            free(w);
            free(w_t);
            fprintf(stderr, "%s: %s F32 [%lld, %lld, %lld]\n", path_out, name, (long long) k, (long long) n, (long long) n_exp);
        } else {
            struct ggml_tensor * src = ggml_get_tensor(meta, name);
            void * data = src != NULL && ggml_nbytes(src) == size ? read_tensor(fi, gctx, src) : NULL;
            ok = data != NULL && fwrite(data, 1, size, fo) == size;
            free(data);
        }
        ok = ok && write_zeros(fo, GGML_PAD(size, align) - size);
        expected += GGML_PAD(size, align);
    }
    ok = ok && (size_t) ftell(fo) == expected;
    ok = fclose(fo) == 0 && ok;
    fclose(fi);

    if (ok) {
        // the written file must read back with the same tensor list
        struct ggml_context * meta_out = NULL;
        struct gguf_init_params rp = { true, &meta_out };
        struct gguf_context * check = gguf_init_from_file(path_out, rp);
        ok = check != NULL && gguf_get_n_tensors(check) == gguf_get_n_tensors(out) && gguf_get_data_offset(check) == meta_size;
        if (check != NULL) {
            gguf_free(check);
            ggml_free(meta_out);
        }
    }
    fprintf(stderr, "%s: %s, %d EXL3 linears dequantized, %lld tensors, %zu bytes\n", path_out, ok ? "OK" : "FAILED",
            n_lin, (long long) gguf_get_n_tensors(out), expected);

    gguf_free(out);
    ggml_free(ctx_w);
    gguf_free(gctx);
    ggml_free(meta);
    return ok ? 0 : 1;
}

// EXL3_MUL_MAT_ID on the op backend on a synthetic stack, see test_mul_mat_id_run
static int test_mul_mat_id_case(ggml_backend_t be, ggml_backend_t be_ref, int64_t k, int64_t n, int64_t n_exp,
                                int64_t tile_words, int cb, int64_t n_used, int64_t ne11, int64_t n_tok, bool skew,
                                uint32_t seed) {
    uint32_t s = seed;
    const int64_t per = k/GGML_EXL3_TILE*n/GGML_EXL3_TILE*tile_words;
    int16_t     * trellis = (int16_t *)     malloc(n_exp*per*sizeof(int16_t));
    ggml_fp16_t * suh     = (ggml_fp16_t *) malloc(n_exp*k*sizeof(ggml_fp16_t));
    ggml_fp16_t * svh     = (ggml_fp16_t *) malloc(n_exp*n*sizeof(ggml_fp16_t));
    for (int64_t j = 0; j < n_exp*per; j++) {
        trellis[j] = (int16_t) (uint16_t) (xorshift32(&s) >> 16);
    }
    for (int64_t j = 0; j < n_exp*k; j++) {
        suh[j] = rand_f16(&s);
    }
    for (int64_t j = 0; j < n_exp*n; j++) {
        svh[j] = rand_f16(&s);
    }
    const int fail = test_mul_mat_id_run(be, be_ref, trellis, suh, svh, k, n, n_exp, tile_words, cb, n_used, ne11, n_tok, skew, s, "synthetic");
    free(trellis);
    free(suh);
    free(svh);
    return fail;
}

// EXL3_MUL_MAT on be of the nr rows xs of one expert, in chunks of up to `chunk` rows, each padded to at least `pad` rows
// with copies of its last row; output rows 0 .. nr - 1 to ys
static bool expert_rows(ggml_backend_t be, const int16_t * trellis, const ggml_fp16_t * suh, const ggml_fp16_t * svh,
                        int64_t k, int64_t n, int64_t tile_words, int cb, const float * xs, int64_t nr, int64_t chunk,
                        int64_t pad, float * ys) {
    struct ggml_init_params params = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true };
    bool ok = true;
    for (int64_t r0 = 0; ok && r0 < nr; r0 += chunk) {
        const int64_t nc = nr - r0 < chunk ? nr - r0 : chunk;
        const int64_t np = nc < pad ? pad : nc;
        struct ggml_context * c1 = ggml_init(params);
        struct ggml_tensor  * a  = ggml_new_tensor_3d(c1, GGML_TYPE_I16, tile_words, n/GGML_EXL3_TILE, k/GGML_EXL3_TILE);
        struct ggml_tensor  * su = ggml_new_tensor_1d(c1, GGML_TYPE_F16, k);
        struct ggml_tensor  * sv = ggml_new_tensor_1d(c1, GGML_TYPE_F16, n);
        struct ggml_tensor  * xx = ggml_new_tensor_2d(c1, GGML_TYPE_F32, k, np);
        struct ggml_tensor  * yy = ggml_exl3_mul_mat(c1, a, xx, su, sv, cb);
        struct ggml_cgraph  * g1 = ggml_new_graph(c1);
        ggml_build_forward_expand(g1, yy);
        ggml_backend_buffer_t b1 = ggml_backend_alloc_ctx_tensors(c1, be);
        float * xp = (float *) malloc(k*np*sizeof(float));
        float * yp = (float *) malloc(n*np*sizeof(float));
        ok = b1 != NULL && xp != NULL && yp != NULL;
        if (ok) {
            for (int64_t r = 0; r < np; r++) {
                memcpy(xp + r*k, xs + (r0 + (r < nc ? r : nc - 1))*k, k*sizeof(float));
            }
            ggml_backend_tensor_set(a, trellis, 0, ggml_nbytes(a));
            ggml_backend_tensor_set(su, suh, 0, ggml_nbytes(su));
            ggml_backend_tensor_set(sv, svh, 0, ggml_nbytes(sv));
            ggml_backend_tensor_set(xx, xp, 0, ggml_nbytes(xx));
            ok = ggml_backend_graph_compute(be, g1) == GGML_STATUS_SUCCESS;
        }
        if (ok) {
            ggml_backend_tensor_get(yy, yp, 0, ggml_nbytes(yy));
            memcpy(ys + r0*n, yp, nc*n*sizeof(float));
        }
        free(xp);
        free(yp);
        ggml_backend_buffer_free(b1);
        ggml_free(c1);
    }
    return ok;
}

// rows of one expert from which the HIP EXL3_MUL_MAT_ID runs one f16 product for them: where the backend's F32 control
// (ggml_mul_mat_id on an F32 stack, an MMB GEMM per expert from 512 rows of a view source) does
#define MMID_ONE_ROWS 512

// the experts of a stack: x of n_tok tokens (ne11 = 1: broadcast over the slots), n_used distinct experts per token,
// uniform, or skewed (slot 0 always the first expert, the other slots favouring low expert numbers).
// Exactness: the rows of each expert equal EXL3_MUL_MAT on that expert's slice of the stack for the same rows, bit for
// bit, in the path the op took: the expert's rows in one call (dot kernel on a GPU), or on a GPU backend the GEMM path
// (chunks of up to 511 rows padded to at least 33, so that EXL3_MUL_MAT runs its hi + lo GEMM; an expert with at least
// MMID_ONE_ROWS rows in one call, its one-product GEMM). NMSE: against ggml_mul_mat_id on the f32 W of the experts on the
// CPU (1e-10), or against a double sum on 8 pairs at most twice that of the op backend's own ggml_mul_mat_id on the same
// f32 W, and so for up to 8 pairs of the experts with at least MMID_ONE_ROWS rows and up to 8 of the others
static int test_mul_mat_id_run(ggml_backend_t be, ggml_backend_t be_ref, const int16_t * trellis, const ggml_fp16_t * suh,
                               const ggml_fp16_t * svh, int64_t k, int64_t n, int64_t n_exp, int64_t tile_words, int cb,
                               int64_t n_used, int64_t ne11, int64_t n_tok, bool skew, uint32_t seed, const char * label) {
    uint32_t s = seed;
    const int64_t tk = k/GGML_EXL3_TILE;
    const int64_t tn = n/GGML_EXL3_TILE;
    const int64_t per = tk*tn*tile_words;
    const int64_t n_pairs = n_used*n_tok;
    float       * w       = (float *)       malloc(n_exp*k*n*sizeof(float));
    float       * w_t     = (float *)       malloc(n_exp*k*n*sizeof(float));
    float       * xd      = (float *)       malloc(k*ne11*n_tok*sizeof(float));
    int32_t     * ids     = (int32_t *)     malloc(n_pairs*sizeof(int32_t));
    float       * y_id    = (float *)       malloc(n*n_pairs*sizeof(float));
    float       * y_ref   = (float *)       malloc(n*n_pairs*sizeof(float));
    float       * y_be    = (float *)       malloc(n*n_pairs*sizeof(float));
    for (int64_t e = 0; e < n_exp; e++) {
        ggml_exl3_dequantize(trellis + e*per, suh + e*k, svh + e*n, k, n, tile_words, (enum ggml_exl3_codebook) cb, w + e*k*n);
        for (int64_t i = 0; i < k; i++) {
            for (int64_t j = 0; j < n; j++) {
                w_t[e*k*n + j*k + i] = w[e*k*n + i*n + j];
            }
        }
    }
    free(w);
    for (int64_t j = 0; j < k*ne11*n_tok; j++) {
        xd[j] = rand_act(&s, j % k);
    }
    // distinct experts per token, the last expert never used when there are more than enough
    const uint32_t n_pick = (uint32_t) (n_exp > n_used ? n_exp - 1 : n_exp);
    for (int64_t t = 0; t < n_tok; t++) {
        for (int64_t i = 0; i < n_used; i++) {
            int32_t e;
            bool dup;
            do {
                if (skew && n_pick > (uint32_t) n_used) {
                    const double u = (double) (xorshift32(&s) >> 8)*(1.0/16777216.0);
                    e = i == 0 ? 0 : 1 + (int32_t) ((double) (n_pick - 1)*u*u*u);
                } else {
                    e = (int32_t) (xorshift32(&s) % n_pick);
                }
                dup = false;
                for (int64_t j = 0; j < i; j++) {
                    dup = dup || ids[t*n_used + j] == e;
                }
            } while (dup);
            ids[t*n_used + i] = e;
        }
    }
    int64_t * cnt = (int64_t *) calloc(n_exp, sizeof(int64_t));
    int64_t max_rows = 0;
    int64_t rows_one = 0;
    for (int64_t p = 0; cnt != NULL && p < n_pairs; p++) {
        cnt[ids[p]]++;
    }
    for (int64_t e = 0; cnt != NULL && e < n_exp; e++) {
        max_rows  = cnt[e] > max_rows ? cnt[e] : max_rows;
        rows_one += cnt[e] >= MMID_ONE_ROWS ? cnt[e] : 0;
    }

    struct ggml_init_params params = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true };
    struct ggml_context * ctx  = ggml_init(params);
    struct ggml_context * ctxr = ggml_init(params);
    struct ggml_tensor  * t_tr  = ggml_new_tensor_4d(ctx, GGML_TYPE_I16, tile_words, tn, tk, n_exp);
    struct ggml_tensor  * t_su  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, n_exp);
    struct ggml_tensor  * t_sv  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n, n_exp);
    struct ggml_tensor  * t_x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, ne11, n_tok);
    struct ggml_tensor  * t_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
    struct ggml_tensor  * t_w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n, n_exp);
    struct ggml_tensor  * out   = ggml_exl3_mul_mat_id(ctx, t_tr, t_x, t_ids, t_su, t_sv, cb);
    struct ggml_tensor  * out_b = ggml_mul_mat_id(ctx, t_w, t_x, t_ids);
    struct ggml_cgraph  * gf    = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_build_forward_expand(gf, out_b);
    struct ggml_tensor  * r_w   = ggml_new_tensor_3d(ctxr, GGML_TYPE_F32, k, n, n_exp);
    struct ggml_tensor  * r_x   = ggml_new_tensor_3d(ctxr, GGML_TYPE_F32, k, ne11, n_tok);
    struct ggml_tensor  * r_ids = ggml_new_tensor_2d(ctxr, GGML_TYPE_I32, n_used, n_tok);
    struct ggml_tensor  * r_out = ggml_mul_mat_id(ctxr, r_w, r_x, r_ids);
    struct ggml_cgraph  * gfr   = ggml_new_graph(ctxr);
    ggml_build_forward_expand(gfr, r_out);
    ggml_backend_buffer_t buf  = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_buffer_t bufr = ggml_backend_alloc_ctx_tensors(ctxr, be_ref);

    int fail = 1;
    const char * rows_equal = "no";
    double err = 1.0;
    double e_op = -1.0;
    double e_be = -1.0;
    double e_cls[2][2] = { { -1.0, -1.0 }, { -1.0, -1.0 } }; // [one product, two][op, backend]
    if (buf == NULL || bufr == NULL || w_t == NULL || y_be == NULL || cnt == NULL || !ggml_backend_supports_op(be, out)) {
        fprintf(stderr, "EXL3_MUL_MAT_ID is not supported by the %s backend\n", ggml_backend_name(be));
    } else {
        ggml_backend_tensor_set(t_tr, trellis, 0, ggml_nbytes(t_tr));
        ggml_backend_tensor_set(t_su, suh, 0, ggml_nbytes(t_su));
        ggml_backend_tensor_set(t_sv, svh, 0, ggml_nbytes(t_sv));
        ggml_backend_tensor_set(t_x, xd, 0, ggml_nbytes(t_x));
        ggml_backend_tensor_set(t_ids, ids, 0, ggml_nbytes(t_ids));
        ggml_backend_tensor_set(t_w, w_t, 0, ggml_nbytes(t_w));
        ggml_backend_tensor_set(r_w, w_t, 0, ggml_nbytes(r_w));
        ggml_backend_tensor_set(r_x, xd, 0, ggml_nbytes(r_x));
        ggml_backend_tensor_set(r_ids, ids, 0, ggml_nbytes(r_ids));
        if (ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS && ggml_backend_graph_compute(be_ref, gfr) == GGML_STATUS_SUCCESS) {
            ggml_backend_tensor_get(out, y_id, 0, ggml_nbytes(out));
            ggml_backend_tensor_get(out_b, y_be, 0, ggml_nbytes(out_b));
            ggml_backend_tensor_get(r_out, y_ref, 0, ggml_nbytes(r_out));
            err = nmse(y_id, y_ref, n*n_pairs);

            // double sums on the first and the last 4 pairs
            const int64_t n_f64 = n_pairs < 8 ? n_pairs : 8;
            double num_o = 0.0, num_b = 0.0, den = 0.0;
            for (int64_t q = 0; q < n_f64; q++) {
                const int64_t p  = q < 4 ? q : n_pairs - n_f64 + q;
                const float * xr = xd + ((p / n_used)*ne11 + (p % n_used) % ne11)*k;
                const float * we = w_t + ids[p]*k*n;
                for (int64_t j = 0; j < n; j++) {
                    double acc = 0.0;
                    for (int64_t i = 0; i < k; i++) {
                        acc += (double) xr[i]*(double) we[j*k + i];
                    }
                    const double t  = (double) (float) acc;
                    const double dop = (double) y_id[p*n + j] - t;
                    const double dbe = (double) y_be[p*n + j] - t;
                    num_o += dop*dop;
                    num_b += dbe*dbe;
                    den   += t*t;
                }
            }
            e_op = den > 0.0 ? num_o/den : 0.0;
            e_be = den > 0.0 ? num_b/den : 0.0;

            // and on the first 8 pairs of the experts with at least MMID_ONE_ROWS rows (cls 0) and of the others (cls 1)
            for (int cls = 0; cls < 2; cls++) {
                double co = 0.0, cbe = 0.0, cd = 0.0;
                int64_t taken = 0;
                for (int64_t p = 0; p < n_pairs && taken < 8; p++) {
                    if ((cnt[ids[p]] >= MMID_ONE_ROWS) != (cls == 0)) {
                        continue;
                    }
                    taken++;
                    const float * xr = xd + ((p / n_used)*ne11 + (p % n_used) % ne11)*k;
                    const float * we = w_t + ids[p]*k*n;
                    for (int64_t j = 0; j < n; j++) {
                        double acc = 0.0;
                        for (int64_t i = 0; i < k; i++) {
                            acc += (double) xr[i]*(double) we[j*k + i];
                        }
                        const double t  = (double) (float) acc;
                        const double dop = (double) y_id[p*n + j] - t;
                        const double dbe = (double) y_be[p*n + j] - t;
                        co  += dop*dop;
                        cbe += dbe*dbe;
                        cd  += t*t;
                    }
                }
                if (taken > 0 && cd > 0.0) {
                    e_cls[cls][0] = co/cd;
                    e_cls[cls][1] = cbe/cd;
                }
            }

            // the rows of each expert, in pair order, against EXL3_MUL_MAT on the expert's slice
            // eq_dot: the expert's rows in one EXL3_MUL_MAT call; eq_gemm: chunks that run the GEMM on a GPU
            bool eq_dot  = true;
            bool eq_gemm = op_backend_name != NULL;
            float * xs  = (float *) malloc(k*n_pairs*sizeof(float));
            float * ys  = (float *) malloc(n*n_pairs*sizeof(float));
            int64_t * pr = (int64_t *) malloc(n_pairs*sizeof(int64_t));
            for (int64_t e = 0; xs != NULL && ys != NULL && pr != NULL && (eq_dot || eq_gemm) && e < n_exp; e++) {
                int64_t nr = 0;
                for (int64_t p = 0; p < n_pairs; p++) {
                    if (ids[p] == e) {
                        memcpy(xs + nr*k, xd + ((p / n_used)*ne11 + (p % n_used) % ne11)*k, k*sizeof(float));
                        pr[nr++] = p;
                    }
                }
                if (nr == 0) {
                    continue;
                }
                for (int pass = 0; pass < 2; pass++) {
                    bool * eq = pass == 0 ? &eq_dot : &eq_gemm;
                    if (!*eq) {
                        continue;
                    }
                    *eq = pass == 0 || nr >= MMID_ONE_ROWS ?
                          expert_rows(be, trellis + e*per, suh + e*k, svh + e*n, k, n, tile_words, cb, xs, nr, nr, 0, ys) :
                          expert_rows(be, trellis + e*per, suh + e*k, svh + e*n, k, n, tile_words, cb, xs, nr, 511, 33, ys);
                    for (int64_t r = 0; *eq && r < nr; r++) {
                        *eq = memcmp(ys + r*n, y_id + pr[r]*n, n*sizeof(float)) == 0;
                    }
                }
            }
            if (xs == NULL || ys == NULL || pr == NULL) {
                eq_dot  = false;
                eq_gemm = false;
            }
            free(xs);
            free(ys);
            free(pr);
            rows_equal = eq_dot ? "one call" : eq_gemm ? "gemm chunks" : "no";
            const bool f32_grade  = err <= OP_NMSE_MAX;
            bool calibrated = e_be > 0.0 && e_op <= OP_NMSE_CALIB*e_be;
            for (int cls = 0; cls < 2; cls++) {
                calibrated = calibrated && (e_cls[cls][1] < 0.0 || (e_cls[cls][1] > 0.0 && e_cls[cls][0] <= OP_NMSE_CALIB*e_cls[cls][1]));
            }
            fail = !((eq_dot || eq_gemm) && (f32_grade || calibrated));
        }
    }
    printf("{\"case\": \"mul_mat_id %s\", \"backend\": \"%s\", \"k\": %lld, \"n\": %lld, \"experts\": %lld, \"K\": %.1f, \"cb\": %d, "
           "\"n_used\": %lld, \"ne11\": %lld, \"tokens\": %lld, \"routing\": \"%s\", \"max_rows\": %lld, \"rows_one\": %lld, "
           "\"rows_equal_mul_mat_per_expert\": \"%s\", \"nmse_vs_mul_mat_id\": %.3e, "
           "\"nmse_op_vs_f64\": %.3e, \"nmse_be_mul_mat_id_vs_f64\": %.3e, \"op_over_be\": %.3f, "
           "\"nmse_op_vs_f64_one\": %.3e, \"nmse_be_vs_f64_one\": %.3e, \"nmse_op_vs_f64_two\": %.3e, \"nmse_be_vs_f64_two\": %.3e, \"pass\": %s}\n",
           label, ggml_backend_name(be), (long long) k, (long long) n, (long long) n_exp, tile_words/16.0, cb, (long long) n_used,
           (long long) ne11, (long long) n_tok, skew ? "skewed" : "uniform", (long long) max_rows, (long long) rows_one, rows_equal,
           err, e_op, e_be, e_be > 0.0 ? e_op/e_be : -1.0, e_cls[0][0], e_cls[0][1], e_cls[1][0], e_cls[1][1], fail ? "false" : "true");
    fflush(stdout);
    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(bufr);
    ggml_free(ctx);
    ggml_free(ctxr);
    free(w_t);
    free(xd);
    free(ids);
    free(y_id);
    free(y_ref);
    free(y_be);
    free(cnt);
    return fail;
}

static int test_mul_mat_id_op(void) {
    ggml_backend_t be     = op_backend_init();
    ggml_backend_t be_ref = op_cpu_init();
    if (be == NULL || be_ref == NULL) {
        return 1;
    }
    int n_fail = 0;
    // x broadcast over the slots (gate, up) and one x per slot (down); decode and prompt token counts; K 2, 3.5, 4, 6
    n_fail += test_mul_mat_id_case(be, be_ref, 256, 384, 8, 32, 2, 2, 1, 13, false, 0x1001u);
    n_fail += test_mul_mat_id_case(be, be_ref, 256, 384, 8, 32, 2, 2, 2, 13, false, 0x1002u);
    n_fail += test_mul_mat_id_case(be, be_ref, 384, 256, 8, 56, 2, 3, 3, 1, false, 0x1003u);
    n_fail += test_mul_mat_id_case(be, be_ref, 256, 256, 6, 64, 1, 4, 1, 37, false, 0x1004u);
    n_fail += test_mul_mat_id_case(be, be_ref, 256, 128, 5, 96, 0, 1, 1, 9, false, 0x1005u);
    n_fail += test_mul_mat_id_case(be, be_ref, 256, 256, 4, 32, 2, 2, 2, 80, false, 0x1006u);
    // skewed routing: the first expert takes every token (one product on HIP from MMID_ONE_ROWS tokens)
    n_fail += test_mul_mat_id_case(be, be_ref, 256, 256, 8, 32, 2, 3, 1, 160, true, 0x1007u);
    n_fail += test_mul_mat_id_case(be, be_ref, 256, 384, 8, 64, 1, 3, 3, 520, true, 0x1008u);
    if (be != be_ref) {
        ggml_backend_free(be);
    }
    ggml_backend_free(be_ref);
    return n_fail;
}

// EXL3_GET_ROWS on the op backend against ggml_exl3_decode_embd_row, bit for bit
static int test_get_rows_op(void) {
    ggml_backend_t be = op_backend_init();
    if (be == NULL) {
        return 1;
    }
    int n_fail = 0;
    const int64_t groups = 4;
    const int64_t n_rows = 300;
    const int64_t n_ids  = 37;
    for (int K = 1; K <= 8; K++) {
        for (int64_t n_out = 1000; n_out <= 1024; n_out += 24) {
            uint32_t s = 0x13579BDFu + 0x9E3779B9u*(uint32_t) K;
            const int64_t gw = ggml_exl3_ring_words(GGML_EXL3_GROUP_DIM, K);
            int16_t     * table = (int16_t *)     malloc(n_rows*groups*gw*sizeof(int16_t));
            ggml_fp16_t * signs = (ggml_fp16_t *) malloc(groups*GGML_EXL3_GROUP_DIM*sizeof(ggml_fp16_t));
            int32_t     * ids   = (int32_t *)     malloc(n_ids*sizeof(int32_t));
            float       * y     = (float *)       malloc(n_ids*n_out*sizeof(float));
            float       * y_ref = (float *)       malloc(n_out*sizeof(float));
            for (int64_t r = 0; r < n_rows*groups; r++) {
                const ggml_fp16_t sc = rand_f16(&s);
                memcpy(&table[r*gw], &sc, sizeof(sc));
                for (int64_t j = 1; j < gw; j++) {
                    table[r*gw + j] = (int16_t) (uint16_t) (xorshift32(&s) >> 16);
                }
            }
            for (int64_t j = 0; j < groups*GGML_EXL3_GROUP_DIM; j++) {
                signs[j] = rand_f16(&s);
            }
            for (int64_t i = 0; i < n_ids; i++) {
                ids[i] = i == 0 ? (int32_t) (n_rows - 1) : (int32_t) (xorshift32(&s) % n_rows);
            }

            struct ggml_init_params params = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true };
            struct ggml_context * ctx = ggml_init(params);
            struct ggml_tensor  * t_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I16, groups*gw, n_rows);
            struct ggml_tensor  * t_signs = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, GGML_EXL3_GROUP_DIM, groups);
            struct ggml_tensor  * t_ids   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_ids);
            struct ggml_tensor  * out     = ggml_exl3_get_rows(ctx, t_table, t_ids, t_signs, n_out);
            struct ggml_cgraph  * gf      = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            bool ok = buf != NULL && ggml_backend_supports_op(be, out);
            if (ok) {
                ggml_backend_tensor_set(t_table, table, 0, ggml_nbytes(t_table));
                ggml_backend_tensor_set(t_signs, signs, 0, ggml_nbytes(t_signs));
                ggml_backend_tensor_set(t_ids, ids, 0, ggml_nbytes(t_ids));
                ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
            }
            if (ok) {
                ggml_backend_tensor_get(out, y, 0, ggml_nbytes(out));
                for (int64_t i = 0; ok && i < n_ids; i++) {
                    ggml_exl3_decode_embd_row(&table[ids[i]*groups*gw], groups, K, signs, n_out, y_ref);
                    ok = memcmp(y_ref, &y[i*n_out], n_out*sizeof(float)) == 0;
                }
            }
            if (!ok) {
                fprintf(stderr, "EXL3_GET_ROWS on %s, K = %d, n_out = %lld: differs from ggml_exl3_decode_embd_row\n",
                        ggml_backend_name(be), K, (long long) n_out);
                n_fail++;
            }
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            free(table);
            free(signs);
            free(ids);
            free(y);
            free(y_ref);
        }
    }
    ggml_backend_free(be);
    return n_fail;
}

int main(int argc, char ** argv) {
    if (argc >= 3 && strcmp(argv[1], "--backend") == 0) {
        op_backend_name = argv[2];
        argv[2] = argv[0];
        argv += 2;
        argc -= 2;
    }
    if (argc == 3 && strcmp(argv[1], "--gguf") == 0) {
        return check_gguf(argv[2]);
    }
    if (argc == 3 && strcmp(argv[1], "--op-gguf") == 0) {
        return op_gguf(argv[2]);
    }
    if (argc == 3 && strcmp(argv[1], "--exact-gguf") == 0) {
        return exact_gguf(argv[2]);
    }
    if (argc == 3 && strcmp(argv[1], "--rows-gguf") == 0) {
        return rows_gguf(argv[2]);
    }
    if (argc == 4 && strcmp(argv[1], "--dequantize") == 0) {
        return dequantize_gguf(argv[2], argv[3]);
    }
    if (argc != 1) {
        fprintf(stderr, "usage: %s [--backend NAME] [--gguf FILE | --op-gguf FILE | --exact-gguf FILE | --rows-gguf FILE | --dequantize IN OUT]\n", argv[0]);
        return 1;
    }

    int n_fail = 0;
    n_fail += test_codebooks();
    n_fail += test_tiles();
    n_fail += test_checks();
    n_fail += test_rows();
    n_fail += test_op();
    n_fail += test_get_rows_op();
    n_fail += test_mul_mat_id_op();
    if (n_fail) {
        fprintf(stderr, "test-exl3: %d failures\n", n_fail);
        return 1;
    }
    printf("test-exl3: OK (3 codebooks, %d rate x codebook cases, op on all of them, row rings and EXL3_GET_ROWS for K = 1..8, EXL3_MUL_MAT_ID)\n", (int) (sizeof(tile_goldens)/sizeof(tile_goldens[0])));
    return 0;
}
