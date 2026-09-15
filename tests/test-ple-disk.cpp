// llama_ple_disk must return, in every configuration, exactly the rows the CPU dequantizer
// produces for the same bytes: duplicates, cache hits, the serial paths and the pool alike.
#include "../src/llama-ple-disk.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
int main() {
    printf("llama_ple_disk is not available on Windows, skipping\n");
    return 0;
}
#else

#include <unistd.h>

struct table {
    ggml_type            type;
    int64_t              ne0;
    int64_t              nrows;
    size_t               offs;
    size_t               rs;
    std::string          fname;
    std::vector<uint8_t> bytes; // the rows as they follow the `offs` prefix bytes in the file
};

static table make_table(ggml_type type, int64_t ne0, int64_t nrows, std::mt19937 & rng) {
    table t;
    t.type  = type;
    t.ne0   = ne0;
    t.nrows = nrows;
    t.rs    = ggml_row_size(type, ne0);
    t.offs  = 3 * 4096 + 77; // rows deliberately start off any I/O alignment
    t.bytes.resize((size_t) nrows * t.rs);
    for (auto & b : t.bytes) {
        b = (uint8_t) rng();
    }

    const char * dir = getenv("TMPDIR");
    std::string path = std::string(dir ? dir : "/tmp") + "/test-ple-disk-XXXXXX";
    std::vector<char> buf(path.begin(), path.end());
    buf.push_back(0);
    const int fd = mkstemp(buf.data());
    GGML_ASSERT(fd >= 0);
    std::vector<uint8_t> prefix(t.offs);
    for (auto & b : prefix) {
        b = (uint8_t) rng();
    }
    GGML_ASSERT(write(fd, prefix.data(), prefix.size()) == (ssize_t) prefix.size());
    size_t done = 0;
    while (done < t.bytes.size()) {
        const ssize_t w = write(fd, t.bytes.data() + done, t.bytes.size() - done);
        GGML_ASSERT(w > 0);
        done += (size_t) w;
    }
    close(fd);
    t.fname = buf.data();
    return t;
}

static void reference(const table & t, const int32_t * idx, size_t n, float * dst) {
    const auto * traits = ggml_get_type_traits(t.type);
    for (size_t k = 0; k < n; ++k) {
        const uint8_t * src = t.bytes.data() + (size_t) idx[k] * t.rs;
        float *         out = dst + k * (size_t) t.ne0;
        if (t.type == GGML_TYPE_F32) {
            memcpy(out, src, t.rs);
        } else {
            traits->to_float(src, out, t.ne0);
        }
    }
}

int main() {
    {
        struct ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
        ggml_free(ggml_init(ip)); // conversion tables
    }

    std::mt19937 rng(20260915);
    const int64_t nrows = 50000;
    const struct { ggml_type type; int64_t ne0; } types[] = {
        { GGML_TYPE_IQ4_NL, 160 }, // the qwen4exp n-gram table layout
        { GGML_TYPE_Q8_0,    64 },
        { GGML_TYPE_F16,     48 },
        { GGML_TYPE_F32,     40 }, // no dequantizer: rows are copied
    };

    size_t n_gathers = 0, n_rows = 0;
    for (const auto & ty : types) {
        const table t = make_table(ty.type, ty.ne0, nrows, rng);

        std::vector<std::vector<int32_t>> patterns;
        patterns.push_back({});                                     // nothing to do
        patterns.push_back({ 7 });
        patterns.push_back({ 7, 7 });                               // one distinct row, serial read
        patterns.push_back({ 3, 1, 2 });                            // three distinct rows, still serial
        patterns.push_back({ 0, (int32_t) nrows - 1, 0, (int32_t) nrows - 1, 0 });
        {
            std::vector<int32_t> v(257);                            // just past the serial dequantization limit
            std::iota(v.begin(), v.end(), 100);
            patterns.push_back(v);
        }
        {
            std::vector<int32_t> v(5000);                           // descending, so sorting reorders everything
            std::iota(v.rbegin(), v.rend(), 20000);
            patterns.push_back(v);
        }
        {
            std::vector<int32_t> v(20000);                          // random over the whole table
            for (auto & r : v) {
                r = (int32_t) (rng() % nrows);
            }
            patterns.push_back(v);
        }
        {
            std::vector<int32_t> v(200000);                         // 500 distinct rows asked for 400 times each
            for (auto & r : v) {
                r = 1000 + (int32_t) (rng() % 500);
            }
            patterns.push_back(v);
        }
        {
            std::vector<int32_t> v = patterns[7];                   // half repeats a previous gather, half is new
            for (size_t k = 0; k < v.size(); k += 2) {
                v[k] = (int32_t) (rng() % nrows);
            }
            patterns.push_back(v);
        }

        for (const int32_t threads : { 1, 3, 64 }) {
            for (const size_t cache_mb : { (size_t) 0, (size_t) 1 }) {
                for (const bool direct : { false, true }) {
                    llama_ple_disk::params p;
                    p.n_threads   = threads;
                    p.cache_bytes = cache_mb << 20;
                    p.direct_io   = direct;
                    llama_ple_disk disk(t.fname, t.offs, t.type, t.ne0, t.nrows, p);
                    GGML_ASSERT(disk.n_rows() == nrows && disk.ne0() == t.ne0 && disk.row_size() == t.rs);

                    for (int pass = 0; pass < 2; ++pass) { // the second pass finds the first one's rows in the cache
                        for (const auto & idx : patterns) {
                            std::vector<float> got(idx.size() * (size_t) t.ne0 + 1), want(got.size());
                            memset(got.data(), 0xff, got.size() * sizeof(float));
                            memset(want.data(), 0xff, want.size() * sizeof(float));
                            reference(t, idx.data(), idx.size(), want.data());
                            disk.gather(idx.data(), idx.size(), got.data());
                            if (memcmp(got.data(), want.data(), got.size() * sizeof(float)) != 0) {
                                fprintf(stderr, "FAIL: %s, %zu rows, threads=%d cache=%zuMiB direct=%d pass=%d\n",
                                        ggml_type_name(t.type), idx.size(), threads, cache_mb, (int) direct, pass);
                                unlink(t.fname.c_str());
                                return 1;
                            }
                            n_gathers += 1;
                            n_rows    += idx.size();
                        }
                    }
                }
            }
        }
        unlink(t.fname.c_str());
    }
    printf("PASS: %zu gathers, %zu rows, byte-identical to the CPU dequantizer across 4 types x 3 thread counts x cache on/off x direct/buffered\n",
           n_gathers, n_rows);
    return 0;
}
#endif
