// TQK6 / TQK7 reference-vector test.
//   test-tqk [dir containing tqk6_ref.bin and tqk7_ref.bin]
// Each file (from agention-infer examples/tq2t_vectors.rs) is: u32 n_blocks, then
// n_blocks x (2 + 4*K)-byte blocks, then n_blocks x 128 little-endian f32 expected values.
// Checks, per type:
//   1. the CPU reference decoder (type_traits->to_float), bit-exact,
//   2. a C++ transcription of the Vulkan state extraction (tqk.glsl: the byte view used
//      by dequant/get_rows/matmul, and the per-lane u16-word view used by the mat-vec)
//      against the literal format definition, on the reference blocks plus random ones,
//   3. GET_ROWS on every available backend device (CPU, Vulkan, ...), bit-exact,
//   4. CPU vec_dot against the decoded weights (tolerance: Q8_0 activations),
//   5. a placeholder-encoder round trip.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifndef TQK_REF_DIR
#define TQK_REF_DIR "."
#endif

static int compare(const char * what, const float * got, const float * want, size_t n) {
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        if (memcmp(&got[i], &want[i], sizeof(float)) != 0) {
            if (bad < 5) {
                fprintf(stderr, "  %s: mismatch at %zu: got %.9g want %.9g\n", what, i, got[i], want[i]);
            }
            bad++;
        }
    }
    printf("%-44s %s (%zu/%zu bit-exact)\n", what, bad ? "FAIL" : "OK", n - bad, n);
    return bad ? 1 : 0;
}

// Literal format definition: 16 stream bits from (31 - t)*K, LSB-first, circular.
static uint32_t state_spec(const uint8_t * qs, uint32_t K, uint32_t t) {
    const uint32_t nbits = 32*K;
    const uint32_t off = (31 - t)*K;
    uint32_t s = 0;
    for (uint32_t b = 0; b < 16; ++b) {
        const uint32_t i = (off + b) % nbits;
        s |= (uint32_t) ((qs[i >> 3] >> (i & 7)) & 1) << b;
    }
    return s;
}

// ---- transcription of tqk.glsl (keep in sync) ----
struct tqk_glsl {
    uint32_t K;
    uint32_t qs_bytes() const { return 4*K; }
    uint32_t qs_words() const { return 2*K; }
    uint32_t off(uint32_t t) const { return (31u - t) * K; }
    uint32_t byte_wrap(uint32_t i) const { return i >= qs_bytes() ? i - qs_bytes() : i; }
    uint32_t word_wrap(uint32_t i) const { return i >= qs_words() ? i - qs_words() : i; }
    static uint32_t state_bytes(uint32_t b0, uint32_t b1, uint32_t b2, uint32_t off) {
        const uint32_t w = b0 | (b1 << 8u) | (b2 << 16u);
        return (w >> (off & 7u)) & 0xFFFFu;
    }
    static uint32_t state_words(uint32_t lo, uint32_t mid, uint32_t hi, uint32_t rel) {
        const uint32_t wi = rel >> 4u;
        const uint32_t pair = wi == 0u ? lo : (wi == 1u ? mid : hi);
        return (pair >> (rel & 15u)) & 0xFFFFu;
    }
    // dequant_funcs.glsl tqk_state / dequant_tqk.glsl / mul_mm_funcs.glsl
    uint32_t state_byte_view(const uint8_t * qs, uint32_t t) const {
        const uint32_t o = off(t);
        const uint32_t i0 = o >> 3u;
        return state_bytes(qs[i0], qs[byte_wrap(i0 + 1u)], qs[byte_wrap(i0 + 2u)], o);
    }
    // mul_mat_vec.comp accumulate_tqk: the four states of one lane, steps 4*lane + i
    void lane_states(const uint8_t * qs, uint32_t lane, uint32_t st[4]) const {
        auto word = [&](uint32_t i) { return (uint32_t) qs[2*i] | ((uint32_t) qs[2*i + 1] << 8); }; // packed16 view, LE
        const uint32_t off3 = off(4u*lane + 3u);
        const uint32_t wb = off3 >> 4u;
        const uint32_t r = off3 & 15u;
        const uint32_t w0 = word(wb);
        const uint32_t w1 = word(word_wrap(wb + 1u));
        const uint32_t w2 = word(word_wrap(wb + 2u));
        const uint32_t w3 = word(word_wrap(wb + 3u));
        const uint32_t lo  = w0 | (w1 << 16u);
        const uint32_t mid = w1 | (w2 << 16u);
        const uint32_t hi  = w2 | (w3 << 16u);
        st[0] = state_words(lo, mid, hi, r + 3u*K);
        st[1] = state_words(lo, mid, hi, r + 2u*K);
        st[2] = state_words(lo, mid, hi, r + K);
        st[3] = state_words(lo, mid, hi, r);
    }
};

static int check_shader_extraction(uint32_t K, const std::vector<uint8_t> & blocks, size_t nb, size_t bs) {
    const tqk_glsl g = { K };
    std::vector<uint8_t> all = blocks;
    std::mt19937 rng(1234 + K);
    for (int i = 0; i < 4096; ++i) {
        for (size_t b = 0; b < bs; ++b) {
            all.push_back((uint8_t) rng());
        }
    }
    const size_t n_all = nb + 4096;
    size_t bad_b = 0, bad_w = 0;
    for (size_t ib = 0; ib < n_all; ++ib) {
        const uint8_t * qs = all.data() + ib*bs + 2;
        for (uint32_t t = 0; t < 32; ++t) {
            bad_b += g.state_byte_view(qs, t) != state_spec(qs, K, t);
        }
        for (uint32_t lane = 0; lane < 8; ++lane) {
            uint32_t st[4];
            g.lane_states(qs, lane, st);
            for (uint32_t i = 0; i < 4; ++i) {
                bad_w += st[i] != state_spec(qs, K, 4*lane + i);
            }
        }
    }
    const size_t n = n_all * 32;
    printf("%-44s %s (%zu/%zu states)\n", "shader byte-view states (tqk.glsl)", bad_b ? "FAIL" : "OK", n - bad_b, n);
    printf("%-44s %s (%zu/%zu states)\n", "shader mat-vec word-view states (tqk.glsl)", bad_w ? "FAIL" : "OK", n - bad_w, n);
    return (bad_b ? 1 : 0) + (bad_w ? 1 : 0);
}

static int run_type(ggml_type type, uint32_t K, const std::string & path) {
    printf("== %s (K=%u) from %s\n", ggml_type_name(type), K, path.c_str());
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        return 1;
    }
    uint32_t nb = 0;
    if (fread(&nb, 4, 1, f) != 1 || nb == 0) {
        fprintf(stderr, "bad header\n");
        fclose(f);
        return 1;
    }
    const size_t bs = ggml_type_size(type);
    const int64_t qk = ggml_blck_size(type);
    if (bs != 2 + 4*K || qk != 128) {
        fprintf(stderr, "unexpected %s block: %zu bytes / %lld weights\n", ggml_type_name(type), bs, (long long) qk);
        fclose(f);
        return 1;
    }
    std::vector<uint8_t> blocks(nb * bs);
    std::vector<float> want(nb * qk);
    if (fread(blocks.data(), 1, blocks.size(), f) != blocks.size() ||
        fread(want.data(), sizeof(float), want.size(), f) != want.size()) {
        fprintf(stderr, "short file\n");
        fclose(f);
        return 1;
    }
    fclose(f);

    int fails = 0;

    // 1. CPU reference decoder
    std::vector<float> got(nb * qk);
    ggml_get_type_traits(type)->to_float(blocks.data(), got.data(), nb * qk);
    fails += compare("cpu to_float", got.data(), want.data(), got.size());

    // 2. shader state extraction, transcribed
    fails += check_shader_extraction(K, blocks, nb, bs);

    // 3. GET_ROWS on each backend device: a [128 x nb] matrix, all rows gathered.
    for (size_t di = 0; di < ggml_backend_dev_count(); ++di) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(di);
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (!be) {
            continue;
        }
        ggml_init_params ip = { 4 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * a   = ggml_new_tensor_2d(ctx, type, qk, nb);
        ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nb);
        ggml_tensor * out = ggml_get_rows(ctx, a, idx);
        std::string name = std::string("get_rows on ") + ggml_backend_dev_name(dev);
        if (!ggml_backend_supports_op(be, out)) {
            printf("%-44s skipped (unsupported)\n", name.c_str());
            ggml_free(ctx);
            ggml_backend_free(be);
            continue;
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_tensor_set(a, blocks.data(), 0, blocks.size());
        std::vector<int32_t> rows(nb);
        for (uint32_t i = 0; i < nb; ++i) {
            rows[i] = (int32_t) i;
        }
        ggml_backend_tensor_set(idx, rows.data(), 0, rows.size() * sizeof(int32_t));
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: compute failed\n", name.c_str());
            fails++;
        } else {
            std::vector<float> res(nb * qk);
            ggml_backend_tensor_get(out, res.data(), 0, res.size() * sizeof(float));
            fails += compare(name.c_str(), res.data(), want.data(), res.size());
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(be);
    }

    // 4. CPU vec_dot vs the decoded weights (activations go through Q8_0, so tolerance)
    {
        const auto * cpu = ggml_get_type_traits_cpu(type);
        const ggml_type vt = cpu->vec_dot_type;
        const int64_t n = nb * qk;
        std::vector<float> x(n);
        for (int64_t i = 0; i < n; ++i) {
            x[i] = 0.1f + std::cos(0.37f * (float) i);
        }
        std::vector<uint8_t> xq(ggml_row_size(vt, n));
        ggml_get_type_traits_cpu(vt)->from_float(x.data(), xq.data(), n);
        std::vector<float> xd(n);
        ggml_get_type_traits(vt)->to_float(xq.data(), xd.data(), n);
        double ref = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            ref += (double) want[i] * xd[i];
        }
        float res = 0.0f;
        cpu->vec_dot(n, &res, 0, blocks.data(), 0, xq.data(), 0, 1);
        const bool ok = std::fabs(res - ref) <= 1e-4 * (1.0 + std::fabs(ref));
        printf("%-44s %s (got %.6f ref %.6f)\n", "cpu vec_dot", ok ? "OK" : "FAIL", res, ref);
        fails += ok ? 0 : 1;
    }

    // 5. placeholder encoder: valid rows that decode, and beat the all-zero-codes baseline
    {
        const int64_t n = 8 * qk;
        std::vector<float> x(n), y(n);
        for (int64_t i = 0; i < n; ++i) {
            x[i] = std::sin(1.3f * (float) i) + 0.5f * std::cos(0.21f * (float) i);
        }
        std::vector<uint8_t> q(ggml_row_size(type, n));
        ggml_quantize_chunk(type, x.data(), q.data(), 0, 1, n, nullptr);
        const bool valid = ggml_validate_row_data(type, q.data(), q.size());
        ggml_get_type_traits(type)->to_float(q.data(), y.data(), n);
        double se = 0.0, sx = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            se += (x[i] - y[i]) * (x[i] - y[i]);
            sx += x[i] * x[i];
        }
        const double rel = se / sx;
        const bool ok = valid && rel < 0.5;
        printf("%-44s %s (valid %d, relative MSE %.4f)\n", "placeholder encoder round trip", ok ? "OK" : "FAIL", valid, rel);
        fails += ok ? 0 : 1;
    }

    return fails;
}

int main(int argc, char ** argv) {
    const std::string dir = argc > 1 ? argv[1] : TQK_REF_DIR;
    ggml_backend_load_all();
    int fails = 0;
    fails += run_type(GGML_TYPE_TQK6, 6, dir + "/tqk6_ref.bin");
    fails += run_type(GGML_TYPE_TQK7, 7, dir + "/tqk7_ref.bin");
    printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
