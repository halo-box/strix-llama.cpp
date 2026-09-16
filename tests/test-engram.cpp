#include "../src/llama-bounded-file.h"
#include "../src/llama-engram.h"
#include "../src/llama-ple-disk.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

static void check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

static void expect_invalid(const std::function<void()> & fn, const char * message) {
    try {
        fn();
    } catch (const std::invalid_argument &) {
        return;
    }
    check(false, message);
}

static void expect_domain(const std::function<void()> & fn, const char * message) {
    try {
        fn();
    } catch (const std::domain_error &) {
        return;
    }
    check(false, message);
}

static void expect_runtime(const std::function<void()> & fn, const char * message) {
    try {
        fn();
    } catch (const std::runtime_error &) {
        return;
    }
    check(false, message);
}

static llama_engram_layout make_layout() {
    llama_engram_layout layout;
    layout.encoding = "e4m3_e8m0_32_row264";
    layout.layer_ids = { 1, 14 };
    layout.token_map.resize(256);
    for (size_t i = 0; i < layout.token_map.size(); ++i) {
        layout.token_map[i] = (uint32_t) i / 2;
    }
    layout.compressed_vocab_size = 128;
    layout.pad_id = 1;
    for (size_t layer = 0; layer < LLAMA_ENGRAM_LAYERS; ++layer) {
        for (size_t depth = 0; depth < LLAMA_ENGRAM_NGRAM; ++depth) {
            layout.multipliers[layer][depth] = 35184372088831ull - 2 * (depth + 4 * layer);
        }
        for (size_t col = 0; col < LLAMA_ENGRAM_COLS; ++col) {
            layout.primes[layer][col] = 16000057;
            layout.rows[layer] += layout.primes[layer][col];
        }
    }
    return layout;
}

static void reference_hash(
        const llama_engram_layout & layout,
        const int32_t * tokens,
        const uint8_t * mask,
        size_t count,
        uint32_t * output) {
    for (size_t pos = 0; pos < count; ++pos) {
        for (size_t layer = 0; layer < LLAMA_ENGRAM_LAYERS; ++layer) {
            uint64_t offset = 0;
            for (size_t col = 0; col < LLAMA_ENGRAM_COLS; ++col) {
                uint64_t hash = 0;
                bool blocked = false;
                for (size_t shift = 0; shift < col / LLAMA_ENGRAM_HEADS + 2; ++shift) {
                    const bool before_start = shift > pos;
                    const size_t source = before_start ? 0 : pos - shift;
                    blocked = blocked || before_start || (mask != nullptr && mask[source] == 0);
                    const uint32_t id = blocked ? layout.pad_id : layout.token_map[tokens[source]];
                    hash ^= (uint64_t) id * layout.multipliers[layer][shift];
                }
                *output++ = (uint32_t) (hash % layout.primes[layer][col] + offset);
                offset += layout.primes[layer][col];
            }
        }
    }
}

static void test_layout_validation() {
    llama_engram_layout layout = make_layout();
    llama_engram_hasher valid(layout);
    check(valid.layout().rows == layout.rows, "valid Engram layout changed");

    llama_engram_layout bad = layout;
    bad.encoding = "e4m3";
    expect_invalid([&] { llama_engram_hasher hasher(bad); }, "bad encoding was accepted");
    bad = layout;
    bad.layer_ids[1] = bad.layer_ids[0];
    expect_invalid([&] { llama_engram_hasher hasher(bad); }, "duplicate layer IDs were accepted");
    bad = layout;
    bad.token_map.clear();
    expect_invalid([&] { llama_engram_hasher hasher(bad); }, "empty token map was accepted");
    bad = layout;
    bad.compressed_vocab_size = 0;
    expect_invalid([&] { llama_engram_hasher hasher(bad); }, "empty compressed vocabulary was accepted");
    bad = layout;
    bad.pad_id = bad.compressed_vocab_size;
    expect_invalid([&] { llama_engram_hasher hasher(bad); }, "bad pad ID was accepted");
    bad = layout;
    bad.token_map[10] = bad.compressed_vocab_size;
    expect_invalid([&] { llama_engram_hasher hasher(bad); }, "bad token map entry was accepted");
    bad = layout;
    bad.multipliers[0][0]--;
    expect_invalid([&] { llama_engram_hasher hasher(bad); }, "even multiplier was accepted");
    bad = layout;
    bad.multipliers[0][0] = UINT64_MAX;
    expect_invalid([&] { llama_engram_hasher hasher(bad); }, "overflowing multiplier was accepted");
    bad = layout;
    bad.primes[0][0] = 1;
    expect_invalid([&] { llama_engram_hasher hasher(bad); }, "bad prime was accepted");
    bad = layout;
    bad.rows[0]--;
    expect_invalid([&] { llama_engram_hasher hasher(bad); }, "bad row extent was accepted");
    bad = layout;
    bad.primes[0].fill(UINT32_MAX);
    expect_invalid([&] { llama_engram_hasher hasher(bad); }, "overflowing row extent was accepted");
}

static void test_hash() {
    const llama_engram_layout layout = make_layout();
    const llama_engram_hasher hasher(layout);
    constexpr size_t count = 513;
    constexpr size_t width = LLAMA_ENGRAM_LAYERS * LLAMA_ENGRAM_COLS;
    std::vector<int32_t> tokens(count);
    std::vector<uint8_t> mask(count);
    std::vector<uint32_t> expected(count * width);
    std::vector<uint32_t> actual(count * width);
    for (size_t i = 0; i < count; ++i) {
        tokens[i] = (int32_t) ((i * 97 + i / 3) % layout.token_map.size());
        mask[i] = i % 17 != 0 && (i < 125 || i > 131);
    }

    for (int masked = 0; masked < 2; ++masked) {
        const uint8_t * current_mask = masked != 0 ? mask.data() : nullptr;
        reference_hash(layout, tokens.data(), current_mask, count, expected.data());
        llama_engram_history expected_history;
        expected_history.reset();
        hasher.hash(expected_history, tokens.data(), current_mask, count, actual.data());
        check(actual == expected, "single-chunk Engram hash differs from full-history reference");
        for (size_t chunk = 1; chunk <= count; ++chunk) {
            llama_engram_history history;
            history.reset();
            for (size_t pos = 0; pos < count; pos += chunk) {
                const size_t size = std::min(chunk, count - pos);
                hasher.hash(history, tokens.data() + pos,
                            current_mask != nullptr ? current_mask + pos : nullptr,
                            size, actual.data() + pos * width);
            }
            check(actual == expected, "rolling Engram hash differs from full-history reference");
            check(history.tail == expected_history.tail, "chunking changed final Engram history");
        }
    }

    llama_engram_history history;
    history.reset();
    const llama_engram_history before = history;
    uint32_t output[2 * width];
    std::fill(output, output + 2 * width, UINT32_MAX);
    const int32_t invalid_high[] = { 0, (int32_t) layout.token_map.size() };
    expect_invalid([&] { hasher.hash(history, invalid_high, nullptr, 2, output); },
                   "high invalid token was accepted");
    check(history.tail == before.tail, "invalid token mutated Engram history");
    check(output[0] == UINT32_MAX, "invalid token changed Engram output");
    const int32_t invalid_low[] = { 0, -1 };
    expect_invalid([&] { hasher.hash(history, invalid_low, nullptr, 2, output); },
                   "negative token was accepted");
    check(history.tail == before.tail, "negative token mutated Engram history");
    hasher.hash(history, nullptr, nullptr, 0, nullptr);
    expect_invalid([&] { hasher.hash(history, invalid_high, nullptr, SIZE_MAX, output); },
                   "overflowing hash count was accepted");
    history.tail[0] = (int32_t) layout.compressed_vocab_size;
    expect_invalid([&] { hasher.hash(history, invalid_high, nullptr, 1, output); },
                   "invalid history was accepted");
}

static float decode_reference(uint8_t code, uint8_t scale) {
    const int exponent = (code >> 3) & 15;
    double value = exponent != 0 ?
            (1.0 + (code & 7) / 8.0) * std::pow(2.0, exponent - 7) :
            (code & 7) / 512.0;
    if ((code & 128) != 0) {
        value = -value;
    }
    float result = (float) (value * std::pow(2.0, (int) scale - 127));
    uint32_t bits;
    memcpy(&bits, &result, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void test_decode() {
    uint8_t row[LLAMA_ENGRAM_ROW_BYTES];
    float output[LLAMA_ENGRAM_DIM];
    for (uint32_t code = 0; code < 256; ++code) {
        memset(row, (int) code, LLAMA_ENGRAM_DIM);
        for (uint32_t scale = 0; scale < 256; ++scale) {
            memset(row + LLAMA_ENGRAM_DIM, (int) scale, LLAMA_ENGRAM_ROW_BYTES - LLAMA_ENGRAM_DIM);
            const float expected = decode_reference((uint8_t) code, (uint8_t) scale);
            const bool valid = (code & 127) != 127 && scale != 255 && std::isfinite(expected);
            if (valid) {
                llama_engram_decode_row(row, output);
                for (float value : output) {
                    check(memcmp(&value, &expected, sizeof(value)) == 0, "Engram decode differs from BF16 reference");
                }
            } else {
                output[0] = 123456.0f;
                expect_domain([&] { llama_engram_decode_row(row, output); }, "invalid Engram value was accepted");
                check(output[0] == 123456.0f, "invalid Engram value changed output");
            }
        }
    }

    memset(row, 0, sizeof(row));
    memset(row + LLAMA_ENGRAM_DIM, 127, LLAMA_ENGRAM_ROW_BYTES - LLAMA_ENGRAM_DIM);
    row[0] = 128;
    llama_engram_decode_row(row, output);
    check(output[0] == 0.0f && std::signbit(output[0]), "negative zero was not preserved");
    check(output[1] == 0.0f && !std::signbit(output[1]), "positive zero was not preserved");
}

#if !defined(_WIN32)
static void write_full(int fd, const void * data, size_t size, uint64_t offset) {
    const ssize_t written = pwrite(fd, data, size, (off_t) offset);
    check(written == (ssize_t) size, "failed to write Engram test row");
}

static void test_disk_rows() {
    char path[] = "/tmp/llama-engram-XXXXXX";
    const int fd = mkstemp(path);
    check(fd >= 0, "failed to create Engram test file");

    const uint64_t offset = (1ull << 33) + 32;
    uint8_t raw[3][LLAMA_ENGRAM_ROW_BYTES];
    for (size_t row = 0; row < 3; ++row) {
        for (size_t i = 0; i < LLAMA_ENGRAM_DIM; ++i) {
            raw[row][i] = (uint8_t) i;
        }
        raw[row][127] = 0;
        raw[row][255] = 128;
        for (size_t i = 0; i < LLAMA_ENGRAM_ROW_BYTES - LLAMA_ENGRAM_DIM; ++i) {
            raw[row][LLAMA_ENGRAM_DIM + i] = (uint8_t) (126 + row);
        }
    }
    write_full(fd, raw, sizeof(raw), offset);

    llama_engram_table table(path, offset, 3);
    check(table.n_rows() == 3, "Engram table row count changed");
    const uint32_t rows[] = { 2, 0, 2, 1 };
    float output[4 * LLAMA_ENGRAM_DIM];
    table.read(rows, 4, output);
    for (size_t row = 0; row < 4; ++row) {
        float expected[LLAMA_ENGRAM_DIM];
        llama_engram_decode_row(raw[rows[row]], expected);
        check(memcmp(output + row * LLAMA_ENGRAM_DIM, expected, sizeof(expected)) == 0,
              "Engram row order was not preserved");
    }

    constexpr size_t tokens = 2051;
    constexpr size_t stride = LLAMA_ENGRAM_LAYERS * LLAMA_ENGRAM_COLS;
    constexpr size_t width = LLAMA_ENGRAM_COLS * LLAMA_ENGRAM_DIM;
    std::vector<uint32_t> batch_rows(tokens * stride, UINT32_MAX);
    std::vector<float> batch(tokens * width + 1);
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t col = 0; col < LLAMA_ENGRAM_COLS; ++col) {
            batch_rows[token * stride + col] = (uint32_t) ((token * 7 + col * 11) % 3);
        }
    }
    const size_t sizes[] = { 1, 2, 31, 65, 257, 2047, 2048, 2049, tokens };
    float decoded[3][LLAMA_ENGRAM_DIM];
    for (size_t row = 0; row < 3; ++row) {
        llama_engram_decode_row(raw[row], decoded[row]);
    }
    for (size_t count : sizes) {
        batch[count * width] = 123456.0f;
        table.read_batch(batch_rows.data(), count, stride, batch.data());
        for (size_t token = 0; token < count; ++token) {
            for (size_t col = 0; col < LLAMA_ENGRAM_COLS; ++col) {
                const uint32_t row = batch_rows[token * stride + col];
                check(memcmp(batch.data() + token * width + col * LLAMA_ENGRAM_DIM,
                             decoded[row], sizeof(decoded[row])) == 0,
                      "batched Engram row differs from direct decode");
            }
        }
        check(batch[count * width] == 123456.0f, "batched Engram read overflowed output");
    }

    table.read(nullptr, 0, nullptr);
    table.read_batch(nullptr, 0, 0, nullptr);
    expect_invalid([&] { table.read_batch(batch_rows.data(), 1, LLAMA_ENGRAM_COLS - 1, batch.data()); },
                   "short Engram batch stride was accepted");
    expect_invalid([&] { table.read_batch(batch_rows.data(), 2, SIZE_MAX, batch.data()); },
                   "overflowing Engram batch stride was accepted");
    expect_invalid([&] { table.read_batch(batch_rows.data(), SIZE_MAX, stride, batch.data()); },
                   "overflowing Engram batch count was accepted");

    const uint32_t bad_row = 3;
    output[0] = 123456.0f;
    expect_invalid([&] { table.read(&bad_row, 1, output); }, "invalid Engram row was accepted");
    check(output[0] == 123456.0f, "invalid Engram row changed output");
    batch_rows[(tokens - 1) * stride] = bad_row;
    batch[0] = 123456.0f;
    expect_invalid([&] { table.read_batch(batch_rows.data(), tokens, stride, batch.data()); },
                   "invalid batched Engram row was accepted");
    check(batch[0] == 123456.0f, "invalid batched Engram row changed output");
    batch_rows[(tokens - 1) * stride] = 0;

    uint8_t invalid = 127;
    write_full(fd, &invalid, 1, offset);
    const uint32_t first_row = 0;
    expect_domain([&] { table.read(&first_row, 1, output); }, "E4M3 NaN row was accepted");
    write_full(fd, raw, sizeof(raw), offset);
    invalid = 255;
    write_full(fd, &invalid, 1, offset + LLAMA_ENGRAM_DIM);
    expect_domain([&] { table.read(&first_row, 1, output); }, "E8M0 scale 255 row was accepted");
    write_full(fd, raw, sizeof(raw), offset);

    check(ftruncate(fd, (off_t) (offset + 260)) == 0, "failed to truncate Engram test file");
    expect_runtime([&] { table.read(&first_row, 1, output); }, "short Engram read was accepted");
    close(fd);

    expect_invalid([&] { llama_engram_table invalid_table(path, offset, 1); },
                   "truncated Engram extent was accepted");
    expect_invalid([&] { llama_engram_table invalid_table(path, UINT64_MAX - 1, 3); },
                   "overflowing Engram extent was accepted");
    expect_invalid([&] { llama_engram_table invalid_table(path, offset, 0); },
                   "empty Engram table was accepted");
    check(unlink(path) == 0, "failed to remove Engram test file");
}

#if defined(__linux__)
static void test_direct_tail_read() {
    char path[] = "/tmp/llama-direct-tail-XXXXXX";
    const int fd = mkstemp(path);
    check(fd >= 0, "failed to create direct tail test file");

    constexpr uint64_t offset = 4096;
    const uint8_t expected[] = { 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2 };
    write_full(fd, expected, sizeof(expected), offset);
    close(fd);

    llama_bounded_file::params params;
    params.direct_io = true;
    params.direct_io_required = true;
    llama_bounded_file file(path, params);
    check(file.direct_io(), "direct tail test did not use O_DIRECT");
    llama_bounded_file::buffer scratch = file.make_buffer(sizeof(expected));
    uint8_t actual[sizeof(expected)] = {};
    file.read(offset, actual, sizeof(actual), scratch);
    check(memcmp(actual, expected, sizeof(actual)) == 0, "valid direct tail read failed");
    check(unlink(path) == 0, "failed to remove direct tail test file");
}
#endif

static void test_ple_disk_reader() {
    char path[] = "/tmp/llama-ple-reader-XXXXXX";
    const int fd = mkstemp(path);
    check(fd >= 0, "failed to create PLE test file");

    constexpr uint64_t offset = 32;
    constexpr size_t columns = 4;
    constexpr size_t rows = 3;
    const float table[rows][columns] = {
        { 1.0f, 2.0f, 3.0f, 4.0f },
        { -1.0f, -2.0f, -3.0f, -4.0f },
        { 0.5f, 0.25f, 0.125f, 0.0625f },
    };
    write_full(fd, table, sizeof(table), offset);
    close(fd);

    for (int direct = 0; direct < 2; ++direct) {
        llama_ple_disk::params params;
        params.n_threads = 2;
        params.cache_bytes = sizeof(table);
        params.direct_io = direct != 0;
        llama_ple_disk disk(path, offset, GGML_TYPE_F32, columns, rows, params);
        const int32_t ids[] = { 2, 0, 2, 1 };
        float output[4][columns];
        disk.gather(ids, 4, output[0]);
        for (size_t i = 0; i < 4; ++i) {
            check(memcmp(output[i], table[ids[i]], sizeof(output[i])) == 0,
                  "shared bounded reader changed PLE row output");
        }
    }

    llama_ple_disk::params params;
    params.n_threads = 4;
    params.cache_bytes = 0;
    params.direct_io = false;
    llama_ple_disk disk(path, offset, GGML_TYPE_F32, columns, rows, params);
    check(truncate(path, (off_t) (offset + sizeof(float))) == 0, "failed to truncate PLE test file");
    const int32_t ids[] = { 0, 1, 2 };
    float output[3][columns];
    expect_runtime([&] { disk.gather(ids, 3, output[0]); }, "threaded PLE read failure did not propagate");

    const int repair_fd = open(path, O_WRONLY);
    check(repair_fd >= 0, "failed to reopen PLE test file");
    write_full(repair_fd, table, sizeof(table), offset);
    close(repair_fd);
    disk.gather(ids, 3, output[0]);
    check(memcmp(output, table, sizeof(table)) == 0, "PLE reader did not recover after worker failure");

    check(unlink(path) == 0, "failed to remove PLE test file");
}
#endif

int main() {
    test_layout_validation();
    test_hash();
    test_decode();
#if !defined(_WIN32)
    test_disk_rows();
#if defined(__linux__)
    test_direct_tail_read();
#endif
    test_ple_disk_reader();
#endif
    std::puts("Engram layout, hash and bounded disk rows: PASS");
    return 0;
}
