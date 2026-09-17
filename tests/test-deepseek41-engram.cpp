#include "../src/llama-dsv41-engram.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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
    layout.token_map.resize(32);
    for (size_t i = 0; i < layout.token_map.size(); ++i) {
        layout.token_map[i] = (uint32_t) i;
    }
    layout.compressed_vocab_size = 32;
    layout.pad_id = 2;
    for (size_t layer = 0; layer < LLAMA_ENGRAM_LAYERS; ++layer) {
        for (size_t i = 0; i < LLAMA_ENGRAM_NGRAM; ++i) {
            layout.multipliers[layer][i] = 101 + 8*layer + 2*i;
        }
        for (size_t col = 0; col < LLAMA_ENGRAM_COLS; ++col) {
            layout.primes[layer][col] = 2;
            layout.rows[layer] += 2;
        }
    }
    return layout;
}

static void fill_row(uint8_t row[LLAMA_ENGRAM_ROW_BYTES], uint32_t id) {
    const uint8_t code = (uint8_t) (8 + id%100);
    std::fill(row, row + LLAMA_ENGRAM_DIM, code);
    std::fill(row + LLAMA_ENGRAM_DIM, row + LLAMA_ENGRAM_ROW_BYTES, 127);
}

#if !defined(_WIN32)
static void write_full(int fd, const void * data, size_t size, uint64_t offset) {
    const ssize_t written = pwrite(fd, data, size, (off_t) offset);
    check(written == (ssize_t) size, "failed to write DeepSeek V4.1 Engram test data");
}

struct test_file {
    std::string path;
    int fd = -1;
    std::array<llama_dsv41_engram_extent, LLAMA_ENGRAM_LAYERS> extents;

    test_file(const llama_engram_layout & layout) {
        char name[] = "/tmp/llama-dsv41-engram-XXXXXX";
        fd = mkstemp(name);
        check(fd >= 0, "failed to create DeepSeek V4.1 Engram test file");
        path = name;

        uint64_t offset = 4096;
        for (size_t layer = 0; layer < LLAMA_ENGRAM_LAYERS; ++layer) {
            extents[layer] = {
                path,
                offset,
                layout.rows[layer],
                LLAMA_ENGRAM_ROW_BYTES,
                layout.rows[layer],
                GGML_TYPE_I8,
            };
            for (uint32_t row_id = 0; row_id < layout.rows[layer]; ++row_id) {
                uint8_t row[LLAMA_ENGRAM_ROW_BYTES];
                fill_row(row, row_id + 7*layer);
                write_full(fd, row, sizeof(row), offset + (uint64_t) row_id*sizeof(row));
            }
            offset += (uint64_t) layout.rows[layer]*LLAMA_ENGRAM_ROW_BYTES + 4096;
        }
    }

    ~test_file() {
        if (fd >= 0) {
            close(fd);
        }
        if (!path.empty()) {
            unlink(path.c_str());
        }
    }
};

static void test_extent_validation() {
    llama_dsv41_engram_extent extent = {
        "/tmp/model.gguf", 4096, 48, LLAMA_ENGRAM_ROW_BYTES, 48, GGML_TYPE_I8,
    };
    llama_dsv41_validate_engram_extent(extent);
    extent.type = GGML_TYPE_F32;
    expect_invalid([&] { llama_dsv41_validate_engram_extent(extent); }, "non-I8 Engram extent was accepted");
    extent.type = GGML_TYPE_I8;
    extent.columns = LLAMA_ENGRAM_ROW_BYTES - 1;
    expect_invalid([&] { llama_dsv41_validate_engram_extent(extent); }, "short Engram row was accepted");
}

static void test_transactions_and_sequences() {
    const llama_engram_layout layout = make_layout();
    test_file file(layout);
    llama_dsv41_engram_runtime runtime(layout, file.extents, 8);

    std::vector<llama_dsv41_engram_token> tokens = {
        { 3, 0, { 7, 9 }, 1 },
        { 5, 1, { 7, 9 }, 0 },
        { 11, 0, { 12 }, 1 },
    };
    llama_dsv41_engram_transaction transaction = runtime.prepare(tokens);
    check(transaction.token_count() == tokens.size(), "Engram transaction token count mismatch");
    check(transaction.row_ids(1) == transaction.row_ids(0) + LLAMA_ENGRAM_COLS,
          "Engram layer row selection is not the second 24-ID half");
    llama_engram_hasher hasher(layout);
    llama_engram_history expected_history_7;
    llama_engram_history expected_history_12;
    expected_history_7.reset();
    expected_history_12.reset();
    uint32_t expected[3*LLAMA_ENGRAM_LAYERS*LLAMA_ENGRAM_COLS];
    const int32_t coupled_tokens[] = { 3, 5 };
    const uint8_t coupled_mask[] = { 1, 0 };
    hasher.hash(expected_history_7, coupled_tokens, coupled_mask, 2, expected);
    hasher.hash(
            expected_history_12,
            &tokens[2].token,
            &tokens[2].text,
            1,
            expected + 2*LLAMA_ENGRAM_LAYERS*LLAMA_ENGRAM_COLS);
    for (size_t token = 0; token < tokens.size(); ++token) {
        for (size_t layer = 0; layer < LLAMA_ENGRAM_LAYERS; ++layer) {
            check(std::memcmp(
                    transaction.row_ids(layer) + token*LLAMA_ENGRAM_LAYERS*LLAMA_ENGRAM_COLS,
                    expected + (token*LLAMA_ENGRAM_LAYERS + layer)*LLAMA_ENGRAM_COLS,
                    LLAMA_ENGRAM_COLS*sizeof(uint32_t)) == 0,
                  "DeepSeek V4.1 Engram IDs or 48-ID token stride differ from the core hasher");
        }
    }
    check(transaction.text_mask()[0] == 1 && transaction.text_mask()[1] == 0,
          "Engram text mask changed");
    check(runtime.sequence(7).pos == -1, "Engram prepare committed sequence state early");

    runtime.commit(transaction);
    check(runtime.sequence(7).pos == 1 && runtime.sequence(9).pos == 1,
          "Engram commit did not advance coupled sequences");
    check(runtime.sequence(7).history.tail[0] == LLAMA_ENGRAM_DEAD,
          "masked Engram token did not break sequence history");

    runtime.seq_copy(7, 13);
    check(runtime.sequence(13).history.tail == runtime.sequence(7).history.tail,
          "Engram sequence copy changed history");
    const llama_dsv41_engram_snapshot snapshot = runtime.checkpoint();
    runtime.seq_reset(7);
    check(runtime.sequence(7).pos == -1, "Engram sequence reset did not clear position");
    runtime.restore(snapshot);
    check(runtime.sequence(7).pos == 1, "Engram checkpoint restore lost position");
    runtime.seq_remove(13);
    check(runtime.sequence(13).pos == -1, "Engram sequence remove retained state");

    llama_dsv41_engram_transaction stale = runtime.prepare({ { 7, 2, { 7 }, 1 } });
    runtime.seq_copy(7, 14);
    expect_runtime([&] { runtime.commit(stale); }, "stale Engram transaction was committed");
    runtime.rollback(stale);

    llama_dsv41_engram_snapshot invalid = runtime.checkpoint();
    invalid.sequences[7].history.tail[0] = (int32_t) layout.compressed_vocab_size;
    expect_invalid([&] { runtime.restore(invalid); }, "invalid Engram snapshot was restored");
    expect_invalid(
            [&] { runtime.prepare({ { 8, 2, { 7, 7 }, 1 } }); },
            "duplicate Engram sequence ID was accepted");
}

static void test_scheduler_upload() {
    const llama_engram_layout layout = make_layout();
    test_file file(layout);
    llama_dsv41_engram_runtime runtime(layout, file.extents, 8);
    llama_dsv41_engram_transaction transaction = runtime.prepare({
        { 3, 0, { 0 }, 1 },
        { 5, 1, { 0 }, 0 },
        { 7, 2, { 0 }, 1 },
    });

    ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    check(ctx != nullptr, "failed to create DeepSeek V4.1 Engram upload context");
    ggml_tensor * rows = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, LLAMA_ENGRAM_COLS*LLAMA_ENGRAM_DIM, transaction.token_count());
    ggml_tensor * select = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, transaction.token_count());

    ggml_backend_t backend = ggml_backend_cpu_init();
    check(backend != nullptr, "failed to create DeepSeek V4.1 Engram upload backend");
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "failed to allocate DeepSeek V4.1 Engram upload tensors");

    std::vector<float> actual_rows(ggml_nelements(rows));
    std::vector<int32_t> actual_select(ggml_nelements(select));
    for (uint32_t layer = 0; layer < LLAMA_ENGRAM_LAYERS; ++layer) {
        transaction.upload_layer(layer, 0, transaction.token_count(), rows, select);
        ggml_backend_tensor_get(rows, actual_rows.data(), 0, ggml_nbytes(rows));
        ggml_backend_tensor_get(select, actual_select.data(), 0, ggml_nbytes(select));
        check(std::memcmp(
                actual_rows.data(),
                transaction.rows(layer),
                ggml_nbytes(rows)) == 0,
              "scheduler-backed Engram row upload changed the bounded pack");
        check(actual_select == std::vector<int32_t>({ 3, 1, 5 }),
              "scheduler-backed Engram text selection upload changed");
    }

    runtime.commit(transaction);
    expect_invalid(
            [&] { transaction.upload_layer(0, 0, 3, rows, select); },
            "committed Engram transaction uploaded stale input");

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

static void test_chunked_prefill() {
    const llama_engram_layout layout = make_layout();
    test_file file(layout);
    llama_dsv41_engram_runtime whole_runtime(layout, file.extents, 8);
    llama_dsv41_engram_runtime chunked_runtime(layout, file.extents, 8);
    const std::vector<llama_dsv41_engram_token> tokens = {
        { 3, 0, { 0 }, 1 },
        { 5, 1, { 0 }, 1 },
        { 7, 2, { 0 }, 0 },
        { 9, 3, { 0 }, 1 },
    };

    llama_dsv41_engram_transaction whole = whole_runtime.prepare(tokens);
    llama_dsv41_engram_transaction first = chunked_runtime.prepare({ tokens[0], tokens[1] });
    for (size_t token = 0; token < 2; ++token) {
        for (uint32_t layer = 0; layer < LLAMA_ENGRAM_LAYERS; ++layer) {
            check(std::memcmp(
                    whole.row_ids(layer) + token*LLAMA_ENGRAM_LAYERS*LLAMA_ENGRAM_COLS,
                    first.row_ids(layer) + token*LLAMA_ENGRAM_LAYERS*LLAMA_ENGRAM_COLS,
                    LLAMA_ENGRAM_COLS*sizeof(uint32_t)) == 0,
                  "first Engram prefill chunk differs from whole-chunk hashing");
        }
    }
    chunked_runtime.commit(first);

    llama_dsv41_engram_transaction second = chunked_runtime.prepare({ tokens[2], tokens[3] });
    for (size_t token = 0; token < 2; ++token) {
        for (uint32_t layer = 0; layer < LLAMA_ENGRAM_LAYERS; ++layer) {
            check(std::memcmp(
                    whole.row_ids(layer) + (token + 2)*LLAMA_ENGRAM_LAYERS*LLAMA_ENGRAM_COLS,
                    second.row_ids(layer) + token*LLAMA_ENGRAM_LAYERS*LLAMA_ENGRAM_COLS,
                    LLAMA_ENGRAM_COLS*sizeof(uint32_t)) == 0,
                  "later Engram prefill chunk lost committed history");
        }
    }
}

static void test_transactional_read_failure() {
    const llama_engram_layout layout = make_layout();
    test_file file(layout);
    llama_dsv41_engram_runtime runtime(layout, file.extents, 8);

    llama_dsv41_engram_transaction first = runtime.prepare({ { 1, 0, { 0 }, 1 } });
    runtime.commit(first);
    const llama_dsv41_engram_sequence_state before = runtime.sequence(0);

    const uint64_t truncated = file.extents[1].offset + LLAMA_ENGRAM_ROW_BYTES;
    check(ftruncate(file.fd, (off_t) truncated) == 0, "failed to truncate Engram transaction test file");
    expect_runtime(
            [&] { runtime.prepare({ { 2, 1, { 0 }, 1 } }); },
            "Engram read failure was not surfaced");
    const llama_dsv41_engram_sequence_state after = runtime.sequence(0);
    check(after.pos == before.pos && after.history.tail == before.history.tail,
          "failed Engram transaction advanced sequence state");
}
#endif

static float bf16(float value) {
    return ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
}

static void test_graph_gate() {
    constexpr int64_t width = 8;
    constexpr int64_t streams = 4;
    constexpr int64_t tokens = 2;

    ggml_init_params params = {
        /*.mem_size   =*/ 8*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    check(ctx != nullptr, "failed to create DeepSeek V4.1 Engram graph context");

    ggml_tensor * residual = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, streams, tokens);
    ggml_tensor * rows = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, LLAMA_ENGRAM_COLS*LLAMA_ENGRAM_DIM, tokens);
    ggml_tensor * engram_kv = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, LLAMA_ENGRAM_COLS*LLAMA_ENGRAM_DIM, 5*width);
    ggml_tensor * q_norm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, streams);
    ggml_tensor * k_norm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, streams);
    ggml_tensor * select = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);

    ggml_set_input(residual);
    ggml_set_input(rows);
    ggml_set_input(engram_kv);
    ggml_set_input(q_norm);
    ggml_set_input(k_norm);
    ggml_set_input(select);

    std::vector<float> residual_data(ggml_nelements(residual));
    std::vector<float> rows_data(ggml_nelements(rows));
    std::vector<float> engram_kv_data(ggml_nelements(engram_kv));
    std::vector<float> q_data(ggml_nelements(q_norm));
    std::vector<float> k_data(ggml_nelements(k_norm));
    for (int64_t i = 0; i < width*streams*tokens; ++i) {
        residual_data[i] = i < width*streams ?
                bf16(0.125f + (float) (i%13)/16.0f) :
                0.1234567f + (float) (i%13)/17.0f;
    }
    residual_data[width*streams] = -0.0f;
    residual_data[width*streams + 1] = 0.0f;
    std::vector<float> projected_data(5*width*tokens);
    for (int64_t token = 0; token < tokens; ++token) {
        rows_data[token*rows->ne[0]] = 1.0f;
        for (int64_t i = 0; i < 5*width; ++i) {
            projected_data[token*5*width + i] = 0.0625f + (float) ((token*5*width + i)%11)/32.0f;
            engram_kv_data[i*engram_kv->ne[0] + token] = projected_data[token*5*width + i];
        }
    }
    for (int64_t i = 0; i < width*streams; ++i) {
        q_data[i] = 0.5f + (float) (i%5)/8.0f;
        k_data[i] = 0.75f - (float) (i%3)/16.0f;
    }
    const int32_t select_data[] = { (int32_t) tokens, 1 };

    const std::vector<float> original = residual_data;
    ggml_backend_t backend = ggml_backend_cpu_init();
    check(backend != nullptr, "failed to create DeepSeek V4.1 Engram graph backend");
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, &buft, 1, 512, false, true);
    check(sched != nullptr, "failed to create DeepSeek V4.1 Engram graph scheduler");
    ggml_tensor * output = llama_dsv41_build_engram(
            ctx, residual, rows, engram_kv, q_norm, k_norm, select, 1.0e-20f, sched, backend);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    check(ggml_backend_sched_alloc_graph(sched, graph), "failed to allocate DeepSeek V4.1 Engram graph");
    ggml_backend_tensor_set(residual, residual_data.data(), 0, ggml_nbytes(residual));
    ggml_backend_tensor_set(rows, rows_data.data(), 0, ggml_nbytes(rows));
    ggml_backend_tensor_set(engram_kv, engram_kv_data.data(), 0, ggml_nbytes(engram_kv));
    ggml_backend_tensor_set(q_norm, q_data.data(), 0, ggml_nbytes(q_norm));
    ggml_backend_tensor_set(k_norm, k_data.data(), 0, ggml_nbytes(k_norm));
    ggml_backend_tensor_set(select, select_data, 0, sizeof(select_data));
    check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
          "DeepSeek V4.1 Engram graph execution failed");

    std::vector<float> actual(ggml_nelements(output));
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (int64_t stream = 0; stream < streams; ++stream) {
        double hidden_sq = 0.0;
        double key_sq = 0.0;
        double dot = 0.0;
        for (int64_t i = 0; i < width; ++i) {
            const float hidden = original[stream*width + i];
            const float key = bf16(projected_data[stream*width + i]);
            hidden_sq += hidden*hidden;
            key_sq += key*key;
            dot += hidden*q_data[stream*width + i]*k_data[stream*width + i]*key;
        }
        dot /= std::sqrt(hidden_sq/width + 1.0e-20);
        dot /= std::sqrt(key_sq/width + 1.0e-20);
        dot /= std::sqrt((double) width);
        const double gate = 1.0/(1.0 + std::exp(-std::copysign(std::sqrt(std::max(std::abs(dot), 1.0e-6)), dot)));
        for (int64_t i = 0; i < width; ++i) {
            const float value = bf16(projected_data[4*width + i]);
            const float expected = bf16(original[stream*width + i] + (float) gate*value);
            check(std::abs(actual[stream*width + i] - expected) <= std::max(1.0e-6f, std::abs(expected)/128.0f),
                  "DeepSeek V4.1 Engram gate differs from scalar reference");
            const size_t masked = width*streams + stream*width + i;
            check(std::memcmp(actual.data() + masked, original.data() + masked, sizeof(float)) == 0,
                  "masked DeepSeek V4.1 Engram row changed");
        }
    }

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

static void test_signed_zero_gate() {
    constexpr int64_t matrix_size = 32;

    ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    check(ctx != nullptr, "failed to create DeepSeek V4.1 signed-zero graph context");

    ggml_tensor * dot = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_tensor * matrix_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, matrix_size, matrix_size);
    ggml_tensor * matrix_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, matrix_size, matrix_size);
    ggml_set_input(dot);
    ggml_set_input(matrix_a);
    ggml_set_input(matrix_b);
    ggml_tensor * accelerated = ggml_mul_mat(ctx, matrix_a, matrix_b);

    ggml_backend_load_all();
    ggml_backend_t backend_accel = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_ACCEL &&
                ggml_backend_dev_supports_op(device, accelerated)) {
            backend_accel = ggml_backend_dev_init(device, nullptr);
            if (backend_accel != nullptr) {
                break;
            }
        }
    }

    ggml_backend_t backend_cpu = ggml_backend_cpu_init();
    check(backend_cpu != nullptr, "failed to create DeepSeek V4.1 signed-zero CPU backend");
    ggml_backend_t backends[] = {
        backend_accel != nullptr ? backend_accel : backend_cpu,
        backend_cpu,
    };
    ggml_backend_buffer_type_t bufts[] = {
        ggml_backend_get_default_buffer_type(backends[0]),
        ggml_backend_cpu_buffer_type(),
    };
    const int n_backends = backend_accel != nullptr ? 2 : 1;
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, n_backends, 32, false, true);
    check(sched != nullptr, "failed to create DeepSeek V4.1 signed-zero scheduler");
    if (backend_accel != nullptr) {
        ggml_backend_sched_set_tensor_backend(sched, accelerated, backend_accel);
        expect_invalid(
                [&] { llama_dsv41_build_engram_gate(ctx, dot, sched, backend_accel); },
                "DeepSeek V4.1 gate accepted a non-CPU backend");
    }

    ggml_tensor * gate = llama_dsv41_build_engram_gate(ctx, dot, sched, backend_cpu);
    ggml_tensor * output = ggml_add(ctx, gate, ggml_repeat(ctx, ggml_sum(ctx, accelerated), gate));
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    check(ggml_backend_sched_alloc_graph(sched, graph), "failed to allocate DeepSeek V4.1 signed-zero graph");
    if (backend_accel != nullptr) {
        check(ggml_backend_sched_get_tensor_backend(sched, accelerated) == backend_accel,
              "DeepSeek V4.1 gate moved surrounding work off its accelerator");
    }
    check(ggml_backend_sched_get_tensor_backend(sched, gate) == backend_cpu,
          "DeepSeek V4.1 gate was not assigned to the local CPU backend");

    const float input[] = { 0.0f, -0.0f };
    std::vector<float> zeros(matrix_size*matrix_size);
    ggml_backend_tensor_set(dot, input, 0, sizeof(input));
    ggml_backend_tensor_set(matrix_a, zeros.data(), 0, ggml_nbytes(matrix_a));
    ggml_backend_tensor_set(matrix_b, zeros.data(), 0, ggml_nbytes(matrix_b));
    check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
          "DeepSeek V4.1 signed-zero scheduler execution failed");
    const float positive = 1.0f/(1.0f + std::exp(-0.001f));
    const float negative = 1.0f/(1.0f + std::exp(0.001f));
    float actual[2];
    ggml_backend_tensor_get(output, actual, 0, sizeof(actual));
    check(std::abs(actual[0] - positive) < 1.0e-7f && actual[0] > 0.5f,
          "DeepSeek V4.1 positive-zero gate lost copysign semantics");
    check(std::abs(actual[1] - negative) < 1.0e-7f && actual[1] < 0.5f,
          "DeepSeek V4.1 negative-zero gate lost copysign semantics");

    ggml_backend_sched_free(sched);
    if (backend_accel != nullptr) {
        ggml_backend_free(backend_accel);
    }
    ggml_backend_free(backend_cpu);
    ggml_free(ctx);
}

int main() {
#if !defined(_WIN32)
    test_extent_validation();
    test_transactions_and_sequences();
    test_scheduler_upload();
    test_chunked_prefill();
    test_transactional_read_failure();
#endif
    test_graph_gate();
    test_signed_zero_gate();
    std::puts("DeepSeek V4.1 Engram runtime and graph: PASS");
    return 0;
}
