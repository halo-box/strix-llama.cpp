#include "../src/llama-expert-store.h"
#include "../src/llama-model-loader.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define REQUIRE(cond) do { if (!(cond)) { throw std::runtime_error("requirement failed: " #cond); } } while (0)

namespace {

struct temp_file {
    std::filesystem::path path;

    explicit temp_file(const char * suffix) {
        static uint64_t sequence = 0;
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
                ("llama-expert-store-" + std::to_string(stamp) + "-" + std::to_string(++sequence) + suffix);
    }

    ~temp_file() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

struct fixture {
    static constexpr int64_t n_embd = 512;
    static constexpr int64_t n_ff = 256;
    static constexpr int64_t n_expert = 4;

    temp_file file { ".gguf" };
    std::vector<llama_expert_store_tensor> tensors;

    fixture() {
        const size_t gate_plane = ggml_row_size(GGML_TYPE_IQ2_XXS, n_embd) * n_ff;
        const size_t down_plane = ggml_row_size(GGML_TYPE_Q2_K, n_ff) * n_embd;
        const size_t data_size = 2 * gate_plane * n_expert + down_plane * n_expert;

        ggml_init_params ggml_params = {
            /*.mem_size   =*/ data_size + 8 * ggml_tensor_overhead() + 4096,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        ggml_context * ctx = ggml_init(ggml_params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ2_XXS, n_embd, n_ff, n_expert);
        ggml_tensor * up   = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ2_XXS, n_embd, n_ff, n_expert);
        ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_K, n_ff, n_embd, n_expert);
        ggml_set_name(gate, "blk.0.ffn_gate_exps.weight");
        ggml_set_name(up,   "blk.0.ffn_up_exps.weight");
        ggml_set_name(down, "blk.0.ffn_down_exps.weight");

        fill_tensor(gate, 0x10);
        fill_tensor(up,   0x20);
        fill_tensor(down, 0x30);

        gguf_context * gguf = gguf_init_empty();
        REQUIRE(gguf != nullptr);
        gguf_set_val_str(gguf, "general.architecture", "deepseek41");
        gguf_add_tensor(gguf, gate);
        gguf_add_tensor(gguf, up);
        gguf_add_tensor(gguf, down);
        REQUIRE(gguf_write_to_file(gguf, file.path.string().c_str(), false));
        gguf_free(gguf);
        ggml_free(ctx);

        std::vector<std::string> splits;
        llama_model_loader loader(
                nullptr, nullptr, nullptr, file.path.string(), splits, nullptr,
                LLAMA_LOAD_MODE_MMAP, false, false, false, nullptr, nullptr);

        REQUIRE(loader.ctx_map.empty());
        tensors.push_back(loader.register_external_tensor(
                "blk.0.ffn_gate_exps.weight", 0, LLAMA_EXPERT_PROJECTION_GATE, { n_embd, n_ff, n_expert }));
        tensors.push_back(loader.register_external_tensor(
                "blk.0.ffn_up_exps.weight", 0, LLAMA_EXPERT_PROJECTION_UP, { n_embd, n_ff, n_expert }));
        tensors.push_back(loader.register_external_tensor(
                "blk.0.ffn_down_exps.weight", 0, LLAMA_EXPERT_PROJECTION_DOWN, { n_ff, n_embd, n_expert }));
        loader.done_getting_tensors();

        REQUIRE(loader.external.any());
        REQUIRE(loader.ctx_map.empty());
        for (const auto & tensor : tensors) {
            REQUIRE(tensor.file_index == 0);
            REQUIRE(loader.external.has(loader.require_tensor_meta(tensor.name)));
        }
        loader.init_mappings(true);
        REQUIRE(loader.mappings.size() == 1);
        REQUIRE(loader.ctx_map.empty());
        const auto & ranges = loader.external.for_file(0);
        REQUIRE(ranges.size() == 3);
        for (size_t i = 0; i < ranges.size(); ++i) {
            REQUIRE(ranges[i].first == tensors[i].file_offset);
            REQUIRE(ranges[i].second == tensors[i].file_offset + tensors[i].nb[2] * tensors[i].ne[2]);
        }
    }

    static void fill_tensor(ggml_tensor * tensor, uint8_t tag) {
        memset(tensor->data, 0, ggml_nbytes(tensor));
        for (int64_t expert = 0; expert < tensor->ne[2]; ++expert) {
            uint8_t * plane = static_cast<uint8_t *>(tensor->data) + expert * tensor->nb[2];
            plane[tensor->nb[2] - 1] = tag + expert;
        }
    }

    llama_expert_store make_store(size_t slots, size_t bytes) const {
        llama_expert_store_params params;
        params.cache_slots = slots;
        params.cache_bytes = bytes;
        params.direct_io = false;
        return llama_expert_store(tensors, params);
    }

    size_t max_plane_size() const {
        size_t result = 0;
        for (const auto & tensor : tensors) {
            result = std::max(result, tensor.nb[2]);
        }
        return result;
    }

    size_t all_projection_bytes() const {
        size_t result = 0;
        for (const auto & tensor : tensors) {
            result += tensor.nb[2];
        }
        return result;
    }
};

template<typename F>
void require_throws(F && fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception &) {
        threw = true;
    }
    REQUIRE(threw);
}

const llama_expert_store::payload & find_payload(
        const std::vector<llama_expert_store::payload> & payloads,
        llama_expert_projection projection,
        int32_t expert_id) {
    for (const auto & payload : payloads) {
        if (payload.projection == projection && payload.expert_id == expert_id) {
            return payload;
        }
    }
    throw std::runtime_error("payload not found");
}

void test_layout_and_offsets(const fixture & f) {
    const auto & gate = f.tensors[0];
    const auto & up = f.tensors[1];
    const auto & down = f.tensors[2];

    REQUIRE(gate.type == GGML_TYPE_IQ2_XXS);
    REQUIRE(up.type == GGML_TYPE_IQ2_XXS);
    REQUIRE(down.type == GGML_TYPE_Q2_K);
    REQUIRE(gate.nb[2] == ggml_row_size(GGML_TYPE_IQ2_XXS, fixture::n_embd) * fixture::n_ff);
    REQUIRE(down.nb[2] == ggml_row_size(GGML_TYPE_Q2_K, fixture::n_ff) * fixture::n_embd);
    REQUIRE(gate.file_offset + 3 * gate.nb[2] > gate.file_offset);

    llama_expert_store store = f.make_store(3, f.all_projection_bytes());
    REQUIRE(store.resident_entries() == 0);
    REQUIRE(store.resident_bytes() == 0);
    auto lease = store.acquire({
        { 0, LLAMA_EXPERT_PROJECTION_GATE, { 2 } },
        { 0, LLAMA_EXPERT_PROJECTION_UP,   { 2 } },
        { 0, LLAMA_EXPERT_PROJECTION_DOWN, { 2 } },
    });
    const auto payloads = lease.payloads();
    REQUIRE(payloads.size() == 3);
    REQUIRE(find_payload(payloads, LLAMA_EXPERT_PROJECTION_GATE, 2).data[gate.nb[2] - 1] == 0x12);
    REQUIRE(find_payload(payloads, LLAMA_EXPERT_PROJECTION_UP,   2).data[up.nb[2] - 1] == 0x22);
    REQUIRE(find_payload(payloads, LLAMA_EXPERT_PROJECTION_DOWN, 2).data[down.nb[2] - 1] == 0x32);
}

void test_alignment_and_large_offsets() {
    const auto aligned = llama_expert_store_align_read(4097, 5000, 4096, 20000);
    REQUIRE(aligned.offset == 4096);
    REQUIRE(aligned.prefix == 1);
    REQUIRE(aligned.size == 8192);

    const uint64_t large_offset = (uint64_t(1) << 32) + 123;
    const auto large = llama_expert_store_align_read(large_offset, 777, 4096, large_offset + 777);
    REQUIRE(large.offset > std::numeric_limits<uint32_t>::max());
    REQUIRE(large.prefix == 123);
    REQUIRE(large.size == 4096);

    require_throws([] {
        llama_expert_store_align_read(0, 1, 3000, 1);
    });
    require_throws([] {
        llama_expert_store_align_read(UINT64_MAX - 4, 8, 4096, UINT64_MAX);
    });

    size_t granularity_first = 1;
    size_t granularity_last = 1;
    llama_mlock::align_range(&granularity_first, &granularity_last);
    const size_t lock_granularity = granularity_last;
    REQUIRE(granularity_first == 0);
    REQUIRE(lock_granularity > 1);

    size_t lock_first = lock_granularity + 1;
    size_t lock_last = 2 * lock_granularity;
    llama_mlock::align_range(&lock_first, &lock_last);
    REQUIRE(lock_first == lock_granularity);
    REQUIRE(lock_last == 2 * lock_granularity);
}

void test_external_mapping_access_policy() {
    const size_t page = 4096;
    const llama_mmap::ranges external = {
        { 0, page },
        { 2 * page + 1, 4 * page - 1 },
        { 5 * page, 6 * page },
        { 7 * page, 8 * page },
        { 9 * page, 10 * page },
    };

    REQUIRE(llama_mmap::use_sequential_file_advice(false));
    REQUIRE(!llama_mmap::use_sequential_file_advice(true));
    REQUIRE(llama_mmap::planned_prefetch_ranges(10 * page, 10 * page, external, true).empty());

    const auto lazy_ranges = llama_mmap::planned_prefetch_ranges(10 * page, 10 * page, external, false);
    REQUIRE(lazy_ranges.size() == 4);
    REQUIRE(lazy_ranges.front() == std::make_pair(page, 2 * page + 1));
    REQUIRE(lazy_ranges.back() == std::make_pair(8 * page, 9 * page));
}

#if defined(__linux__)
int file_advice_calls = 0;

int fail_file_advice(int, int) {
    ++file_advice_calls;
    return EIO;
}

void test_external_mapping_advice_failure() {
    temp_file file { ".bin" };
    {
        std::ofstream out(file.path, std::ios::binary);
        REQUIRE(out.good());
        out.seekp(8191);
        out.put('\0');
    }

    llama_file input(file.path.string(), "rb");
    file_advice_calls = 0;
    bool continued_after_advice = false;
    require_throws([&] {
        llama_mmap mapping(&input, 0, false, { { 4096, 8192 } }, true, fail_file_advice);
        continued_after_advice = true;
    });
    REQUIRE(file_advice_calls == 1);
    REQUIRE(!continued_after_advice);

    llama_mmap legacy_mapping(&input, 0, false, {}, false, fail_file_advice);
    REQUIRE(file_advice_calls == 2);
    REQUIRE(legacy_mapping.addr() != nullptr);
}
#endif

void test_published_layout_accounting() {
    const int64_t n_embd = 7680;
    const int64_t n_ff = 1536;
    const int64_t n_expert = 384;
    const int64_t n_layer = 40;

    const uint64_t gate_plane = ggml_row_size(GGML_TYPE_IQ2_XXS, n_embd) * n_ff;
    const uint64_t up_plane = ggml_row_size(GGML_TYPE_IQ2_XXS, n_embd) * n_ff;
    const uint64_t down_plane = ggml_row_size(GGML_TYPE_Q2_K, n_ff) * n_embd;
    const uint64_t slot_bytes = (gate_plane + up_plane + down_plane) * n_layer;
    const uint64_t routed_bytes = slot_bytes * n_expert;
    const uint64_t dense_bytes = 10067427328;
    const uint64_t engram_bytes = 202758045696;

    REQUIRE(gate_plane == 3041280);
    REQUIRE(up_plane == 3041280);
    REQUIRE(down_plane == 3870720);
    REQUIRE(gate_plane + up_plane + down_plane == 9953280);
    REQUIRE(slot_bytes == 398131200);
    REQUIRE(routed_bytes == 152882380800);
    REQUIRE(224 * slot_bytes == 89181388800);
    REQUIRE(256 * slot_bytes == 101921587200);
    REQUIRE(dense_bytes + 224 * slot_bytes == 99248816128);
    REQUIRE(dense_bytes + 256 * slot_bytes == 111989014528);
    REQUIRE(engram_bytes > routed_bytes);
}

void test_large_offset_read() {
    temp_file sparse { ".bin" };
    const int64_t ne0 = 256;
    const int64_t ne1 = 256;
    const int64_t n_expert = 1;
    const size_t gate_plane = ggml_row_size(GGML_TYPE_IQ2_XXS, ne0) * ne1;
    const size_t down_plane = ggml_row_size(GGML_TYPE_Q2_K, ne0) * ne1;
    const uint64_t base = (uint64_t(1) << 32) + 4096;
    const uint64_t file_size = base + 2 * gate_plane + down_plane;

    {
        std::ofstream out(sparse.path, std::ios::binary | std::ios::trunc);
        REQUIRE(out.good());
        out.seekp(static_cast<std::streamoff>(file_size - 1));
        out.put('\0');
        const std::vector<std::pair<uint64_t, uint8_t>> markers = {
            { base + gate_plane - 1, 0x41 },
            { base + 2 * gate_plane - 1, 0x42 },
            { file_size - 1, 0x43 },
        };
        for (const auto & marker : markers) {
            out.seekp(static_cast<std::streamoff>(marker.first));
            out.put(static_cast<char>(marker.second));
        }
    }

    auto make_tensor = [&](const char * name, llama_expert_projection projection, ggml_type type, uint64_t offset) {
        llama_expert_store_tensor tensor;
        tensor.name = name;
        tensor.fname = sparse.path.string();
        tensor.layer = 0;
        tensor.projection = projection;
        tensor.type = type;
        tensor.ne[0] = ne0;
        tensor.ne[1] = ne1;
        tensor.ne[2] = n_expert;
        tensor.nb[0] = ggml_type_size(type);
        tensor.nb[1] = ggml_row_size(type, ne0);
        tensor.nb[2] = tensor.nb[1] * ne1;
        tensor.file_offset = offset;
        tensor.file_size = file_size;
        return tensor;
    };

    std::vector<llama_expert_store_tensor> tensors;
    tensors.push_back(make_tensor("gate", LLAMA_EXPERT_PROJECTION_GATE, GGML_TYPE_IQ2_XXS, base));
    tensors.push_back(make_tensor("up", LLAMA_EXPERT_PROJECTION_UP, GGML_TYPE_IQ2_XXS, base + gate_plane));
    tensors.push_back(make_tensor("down", LLAMA_EXPERT_PROJECTION_DOWN, GGML_TYPE_Q2_K, base + 2 * gate_plane));

    llama_expert_store_params params { 2 * gate_plane + down_plane, 3, 4096, false };
    llama_expert_store store(std::move(tensors), params);
    auto lease = store.acquire({
        { 0, LLAMA_EXPERT_PROJECTION_GATE, { 0 } },
        { 0, LLAMA_EXPERT_PROJECTION_UP,   { 0 } },
        { 0, LLAMA_EXPERT_PROJECTION_DOWN, { 0 } },
    });
    const auto payloads = lease.payloads();
    REQUIRE(find_payload(payloads, LLAMA_EXPERT_PROJECTION_GATE, 0).data[gate_plane - 1] == 0x41);
    REQUIRE(find_payload(payloads, LLAMA_EXPERT_PROJECTION_UP,   0).data[gate_plane - 1] == 0x42);
    REQUIRE(find_payload(payloads, LLAMA_EXPERT_PROJECTION_DOWN, 0).data[down_plane - 1] == 0x43);
}

void test_cache_and_remapping(const fixture & f) {
    const size_t gate_plane = f.tensors[0].nb[2];
    llama_expert_store store = f.make_store(2, 2 * f.max_plane_size());

    {
        auto lease = store.acquire({
            { 0, LLAMA_EXPERT_PROJECTION_GATE, { 0, 0 } },
            { 0, LLAMA_EXPERT_PROJECTION_GATE, { 1, 0 } },
        });
        REQUIRE(lease.slot_ids().size() == 2);
        REQUIRE(lease.slot_ids()[0][0] == lease.slot_ids()[0][1]);
        REQUIRE(lease.slot_ids()[1][1] == lease.slot_ids()[0][0]);
        REQUIRE(lease.slot_ids()[1][0] != lease.slot_ids()[1][1]);
        REQUIRE(store.resident_entries() == 2);
        REQUIRE(store.resident_bytes() == 2 * gate_plane);
    }

    {
        auto hit = store.acquire({ { 0, LLAMA_EXPERT_PROJECTION_GATE, { 0 } } });
        REQUIRE(hit.slot_ids()[0][0] == 0);
    }
    {
        auto miss = store.acquire({ { 0, LLAMA_EXPERT_PROJECTION_GATE, { 2 } } });
        REQUIRE(miss.slot_ids()[0][0] == 1);
    }

    const llama_expert_store_stats stats = store.stats();
    REQUIRE(stats.hits == 1);
    REQUIRE(stats.misses == 3);
    REQUIRE(stats.evictions == 1);
    REQUIRE(stats.bytes_read == 3 * gate_plane);

    require_throws([&] {
        store.acquire({ { 0, LLAMA_EXPERT_PROJECTION_GATE, { -1 } } });
    });
    require_throws([&] {
        store.acquire({ { 0, LLAMA_EXPERT_PROJECTION_GATE, { fixture::n_expert } } });
    });
}

void test_direct_io(const fixture & f) {
    llama_expert_store_params params;
    params.cache_slots = 1;
    params.cache_bytes = f.max_plane_size();
    params.io_alignment = 4096;
    params.direct_io = true;
    params.allow_buffered_io = true;

    llama_expert_store store(f.tensors, params);
    auto lease = store.acquire({ { 0, LLAMA_EXPERT_PROJECTION_GATE, { 3 } } });
    const auto payloads = lease.payloads();
    REQUIRE(payloads.size() == 1);
    REQUIRE(payloads[0].data[payloads[0].size - 1] == 0x13);
    REQUIRE(store.stats().bytes_read >= payloads[0].size);
    REQUIRE(store.stats().bytes_read <= payloads[0].size + 2 * params.io_alignment);
}

#if defined(__linux__)
void test_direct_io_file_tail(const fixture & f) {
    const auto & down = f.tensors[2];
    REQUIRE(down.file_offset + down.nb[2] * down.ne[2] == down.file_size);

    llama_expert_store_params params;
    params.cache_slots = 1;
    params.cache_bytes = f.max_plane_size();
    params.io_alignment = 4096;
    params.direct_io = true;
    params.allow_buffered_io = false;

    llama_expert_store store(f.tensors, params);
    REQUIRE(store.direct_io_active());
    auto lease = store.acquire({ { 0, LLAMA_EXPERT_PROJECTION_DOWN, { fixture::n_expert - 1 } } });
    const auto payloads = lease.payloads();
    REQUIRE(payloads.size() == 1);
    REQUIRE(payloads[0].data[payloads[0].size - 1] == 0x33);
    REQUIRE(store.direct_io_active());
}
#endif

#if defined(_WIN32)
void test_windows_direct_io_policy(const fixture & f) {
    llama_expert_store_params params;
    params.cache_slots = 1;
    params.cache_bytes = f.max_plane_size();
    params.direct_io = true;
    params.allow_buffered_io = false;

    require_throws([&] {
        llama_expert_store store(f.tensors, params);
    });

    params.allow_buffered_io = true;
    llama_expert_store store(f.tensors, params);
    REQUIRE(!store.direct_io_active());
    auto lease = store.acquire({ { 0, LLAMA_EXPERT_PROJECTION_GATE, { 0 } } });
    REQUIRE(lease.payloads().size() == 1);
}
#endif

void test_pins_and_atomic_failure(const fixture & f) {
    llama_expert_store store = f.make_store(1, f.max_plane_size());
    auto pinned = store.acquire({ { 0, LLAMA_EXPERT_PROJECTION_GATE, { 0 } } });
    const llama_expert_store_stats before = store.stats();

    require_throws([&] {
        store.acquire({ { 0, LLAMA_EXPERT_PROJECTION_GATE, { 1 } } });
    });
    REQUIRE(store.resident_entries() == 1);
    REQUIRE(store.stats().hits == before.hits);
    REQUIRE(store.stats().misses == before.misses);
    REQUIRE(pinned.payloads()[0].expert_id == 0);

    pinned = {};
    auto replacement = store.acquire({ { 0, LLAMA_EXPERT_PROJECTION_GATE, { 1 } } });
    REQUIRE(replacement.payloads()[0].expert_id == 1);
    REQUIRE(store.stats().evictions == 1);

    llama_expert_store::lease surviving;
    {
        llama_expert_store short_lived = f.make_store(1, f.max_plane_size());
        surviving = short_lived.acquire({ { 0, LLAMA_EXPERT_PROJECTION_GATE, { 2 } } });
    }
    REQUIRE(surviving.payloads()[0].expert_id == 2);
}

void test_limits_and_validation(const fixture & f) {
    require_throws([&] {
        f.make_store(0, f.max_plane_size());
    });
    require_throws([&] {
        f.make_store(1, f.tensors[2].nb[2] - 1);
    });
    {
        llama_expert_store_params params { f.max_plane_size(), 1, 1, false };
        llama_expert_store store(f.tensors, params);
        auto lease = store.acquire({ { 0, LLAMA_EXPERT_PROJECTION_GATE, { 0 } } });
        REQUIRE(lease.payloads().size() == 1);
    }
    {
        llama_expert_store store = f.make_store(3, f.max_plane_size());
        require_throws([&] {
            store.acquire({
                { 0, LLAMA_EXPERT_PROJECTION_GATE, { 0 } },
                { 0, LLAMA_EXPERT_PROJECTION_UP,   { 0 } },
            });
        });
        REQUIRE(store.resident_entries() == 0);
        REQUIRE(store.stats().misses == 0);
    }
    {
        llama_expert_store store = f.make_store(1, 2 * f.tensors[0].nb[2]);
        require_throws([&] {
            store.acquire({ { 0, LLAMA_EXPERT_PROJECTION_GATE, { 0, 1 } } });
        });
        REQUIRE(store.resident_entries() == 0);
        REQUIRE(store.stats().misses == 0);
    }

    auto bad_type = f.tensors;
    bad_type[0].type = GGML_TYPE_Q2_K;
    require_throws([&] {
        llama_expert_store_params params { f.all_projection_bytes(), 3, 4096, false };
        llama_expert_store store(std::move(bad_type), params);
    });

    auto bad_stride = f.tensors;
    bad_stride[1].nb[2]++;
    require_throws([&] {
        llama_expert_store_params params { f.all_projection_bytes(), 3, 4096, false };
        llama_expert_store store(std::move(bad_stride), params);
    });

    auto bad_bounds = f.tensors;
    bad_bounds[2].file_size = bad_bounds[2].file_offset + bad_bounds[2].nb[2] - 1;
    require_throws([&] {
        llama_expert_store_params params { f.all_projection_bytes(), 3, 4096, false };
        llama_expert_store store(std::move(bad_bounds), params);
    });
}

void test_payload_validation(const fixture & f) {
    temp_file copy { ".gguf" };
    std::filesystem::copy_file(f.file.path, copy.path);
    auto tensors = f.tensors;
    for (auto & tensor : tensors) {
        tensor.fname = copy.path.string();
        tensor.file_size = std::filesystem::file_size(copy.path);
    }

    {
        std::fstream io(copy.path, std::ios::binary | std::ios::in | std::ios::out);
        REQUIRE(io.good());
        io.seekp(static_cast<std::streamoff>(tensors[0].file_offset));
        const uint8_t invalid_scale[2] = { 0x00, 0x7c };
        io.write(reinterpret_cast<const char *>(invalid_scale), sizeof(invalid_scale));
    }

    llama_expert_store_params params { f.all_projection_bytes(), 3, 4096, false };
    llama_expert_store store(std::move(tensors), params);
    require_throws([&] {
        store.acquire({ { 0, LLAMA_EXPERT_PROJECTION_GATE, { 0 } } });
    });
    REQUIRE(store.resident_entries() == 0);
}

void test_truncated_file(const fixture & f) {
    temp_file copy { ".gguf" };
    std::filesystem::copy_file(f.file.path, copy.path);
    auto tensors = f.tensors;
    for (auto & tensor : tensors) {
        tensor.fname = copy.path.string();
        tensor.file_size = std::filesystem::file_size(copy.path);
    }

    llama_expert_store_params params { f.all_projection_bytes(), 3, 4096, false };
    llama_expert_store store(std::move(tensors), params);
    std::filesystem::resize_file(copy.path, f.tensors[0].file_offset + f.tensors[0].nb[2] - 1);
    require_throws([&] {
        store.acquire({ { 0, LLAMA_EXPERT_PROJECTION_GATE, { 0 } } });
    });
    REQUIRE(store.resident_entries() == 0);
    REQUIRE(store.stats().misses == 0);
}

void test_failed_replacement_keeps_resident_entry(const fixture & f) {
    temp_file copy { ".gguf" };
    std::filesystem::copy_file(f.file.path, copy.path);
    auto tensors = f.tensors;
    for (auto & tensor : tensors) {
        tensor.fname = copy.path.string();
        tensor.file_size = std::filesystem::file_size(copy.path);
    }

    llama_expert_store_params params { f.max_plane_size(), 1, 4096, false };
    llama_expert_store store(std::move(tensors), params);
    {
        auto resident = store.acquire({ { 0, LLAMA_EXPERT_PROJECTION_GATE, { 0 } } });
        REQUIRE(resident.payloads()[0].expert_id == 0);
    }
    const llama_expert_store_stats before = store.stats();
    std::filesystem::resize_file(copy.path, f.tensors[0].file_offset + 2*f.tensors[0].nb[2] - 1);
    require_throws([&] {
        store.acquire({ { 0, LLAMA_EXPERT_PROJECTION_GATE, { 1 } } });
    });
    REQUIRE(store.resident_entries() == 1);
    REQUIRE(store.stats().evictions == before.evictions);
    auto hit = store.acquire({ { 0, LLAMA_EXPERT_PROJECTION_GATE, { 0 } } });
    REQUIRE(hit.payloads()[0].expert_id == 0);
}

}

int main() {
    try {
        fixture f;
        test_layout_and_offsets(f);
        test_alignment_and_large_offsets();
        test_external_mapping_access_policy();
#if defined(__linux__)
        test_external_mapping_advice_failure();
#endif
        test_published_layout_accounting();
        test_large_offset_read();
        test_cache_and_remapping(f);
        test_direct_io(f);
#if defined(__linux__)
        test_direct_io_file_tail(f);
#endif
#if defined(_WIN32)
        test_windows_direct_io_policy(f);
#endif
        test_pins_and_atomic_failure(f);
        test_limits_and_validation(f);
        test_payload_validation(f);
        test_truncated_file(f);
        test_failed_replacement_keeps_resident_entry(f);
    } catch (const std::exception & e) {
        fprintf(stderr, "test-expert-store: %s\n", e.what());
        return 1;
    }
    return 0;
}
