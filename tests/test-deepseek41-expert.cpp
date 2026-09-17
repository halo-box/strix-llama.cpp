#include "../src/llama-dsv41-expert.h"
#include "../src/llama-dsv41.h"
#include "../src/llama-graph.h"
#include "../src/llama-model-loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#define REQUIRE(cond) do { if (!(cond)) { throw std::runtime_error("requirement failed: " #cond); } } while (0)

namespace {

struct temp_file {
    std::filesystem::path path;

    temp_file() {
        static uint64_t sequence = 0;
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
                ("llama-dsv41-expert-" + std::to_string(stamp) + "-" + std::to_string(++sequence) + ".gguf");
    }

    ~temp_file() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
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

struct fixture {
    static constexpr int64_t n_embd = 256;
    static constexpr int64_t n_ff = 256;
    static constexpr int64_t n_expert = LLAMA_DSV41_N_EXPERT;

    temp_file file;
    std::vector<llama_expert_store_tensor> tensors;

    fixture() {
        const size_t gate_plane = ggml_row_size(GGML_TYPE_IQ2_XXS, n_embd)*n_ff;
        const size_t down_plane = ggml_row_size(GGML_TYPE_Q2_K, n_ff)*n_embd;
        const size_t data_size = (2*gate_plane + down_plane)*n_expert;
        ggml_init_params params = {
            /*.mem_size   =*/ data_size + 8*ggml_tensor_overhead() + 4096,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        ggml_context * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ2_XXS, n_embd, n_ff, n_expert);
        ggml_tensor * up = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ2_XXS, n_embd, n_ff, n_expert);
        ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_K, n_ff, n_embd, n_expert);
        ggml_set_name(gate, "blk.0.ffn_gate_exps.weight");
        ggml_set_name(up, "blk.0.ffn_up_exps.weight");
        ggml_set_name(down, "blk.0.ffn_down_exps.weight");
        fill(gate, 0x10);
        fill(up, 0x20);
        fill(down, 0x30);

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
        const auto gate_extent = loader.register_external_tensor(
                "blk.0.ffn_gate_exps.weight", 0, LLAMA_EXPERT_PROJECTION_GATE, { n_embd, n_ff, n_expert });
        const auto up_extent = loader.register_external_tensor(
                "blk.0.ffn_up_exps.weight", 0, LLAMA_EXPERT_PROJECTION_UP, { n_embd, n_ff, n_expert });
        const auto down_extent = loader.register_external_tensor(
                "blk.0.ffn_down_exps.weight", 0, LLAMA_EXPERT_PROJECTION_DOWN, { n_ff, n_embd, n_expert });
        for (int32_t il = 0; il < (int32_t) LLAMA_DSV41_N_LAYER; ++il) {
            for (auto extent : { gate_extent, up_extent, down_extent }) {
                extent.layer = il;
                extent.name = "blk." + std::to_string(il) + extent.name.substr(5);
                tensors.push_back(std::move(extent));
            }
        }
    }

    static void fill(ggml_tensor * tensor, uint8_t tag) {
        memset(tensor->data, 0, ggml_nbytes(tensor));
        for (int64_t expert = 0; expert < tensor->ne[2]; ++expert) {
            uint8_t * plane = static_cast<uint8_t *>(tensor->data) + expert*tensor->nb[2];
            plane[tensor->nb[2] - 1] = tag + expert%32;
        }
    }

    size_t cache_bytes(size_t slots) const {
        size_t result = 0;
        for (const auto & tensor : tensors) {
            result += tensor.nb[2]*slots;
        }
        return result;
    }

    llama_dsv41_expert_runtime make_runtime(
            size_t slots,
            llama_dsv41_expert_runtime::upload_fn upload = {},
            llama_dsv41_expert_runtime::publish_fn before_publish = {}) const {
        llama_dsv41_expert_runtime_params params;
        params.cache_slots = slots;
        params.cache_bytes = cache_bytes(slots);
        params.direct_io = false;
        return llama_dsv41_expert_runtime(
                tensors,
                params,
                [](const llama_expert_store_tensor &) { return ggml_backend_cpu_buffer_type(); },
                std::move(upload),
                std::move(before_publish));
    }
};

void test_registration() {
    std::vector<std::string> names;
    const auto tensors = llama_dsv41_register_expert_tensors(
            [&](const std::string & name,
                int32_t layer,
                llama_expert_projection projection,
                const std::initializer_list<int64_t> & ne) {
                names.push_back(name);
                llama_expert_store_tensor tensor;
                tensor.name = name;
                tensor.fname = "unused";
                tensor.file_index = layer % 3;
                tensor.layer = layer;
                tensor.projection = projection;
                tensor.type = projection == LLAMA_EXPERT_PROJECTION_DOWN ? GGML_TYPE_Q2_K : GGML_TYPE_IQ2_XXS;
                std::copy(ne.begin(), ne.end(), tensor.ne);
                tensor.nb[0] = ggml_type_size(tensor.type);
                tensor.nb[1] = ggml_row_size(tensor.type, tensor.ne[0]);
                tensor.nb[2] = tensor.nb[1]*tensor.ne[1];
                tensor.file_offset = 4096 + (size_t) layer*3*1024 + (size_t) projection*1024;
                tensor.file_size = tensor.nb[2]*tensor.ne[2];
                return tensor;
            });
    REQUIRE(tensors.size() == LLAMA_DSV41_N_LAYER*3);
    REQUIRE(names.front() == "blk.0.ffn_gate_exps.weight");
    REQUIRE(names.back() == "blk.39.ffn_down_exps.weight");
    for (int32_t il = 0; il < (int32_t) LLAMA_DSV41_N_LAYER; ++il) {
        REQUIRE(tensors[3*il + 0].layer == il);
        REQUIRE(tensors[3*il + 1].layer == il);
        REQUIRE(tensors[3*il + 2].layer == il);
        REQUIRE(tensors[3*il + 0].file_index == (size_t) il % 3);
        REQUIRE(tensors[3*il + 0].file_offset == 4096 + (size_t) il*3*1024);
        REQUIRE(tensors[3*il + 0].ne[2] == LLAMA_DSV41_N_EXPERT);
        REQUIRE(tensors[3*il + 0].nb[2] == 3041280);
        REQUIRE(tensors[3*il + 1].nb[2] == 3041280);
        REQUIRE(tensors[3*il + 2].nb[2] == 3870720);
    }
    REQUIRE(std::none_of(names.begin(), names.end(), [](const std::string & name) {
        return name.find("shexp") != std::string::npos;
    }));
    size_t one_slot_bytes = 0;
    for (const auto & tensor : tensors) {
        one_slot_bytes += tensor.nb[2];
    }
    REQUIRE(one_slot_bytes == 398131200);
}

void test_configuration(const fixture & f) {
    llama_dsv41_expert_runtime_params params;
    params.direct_io = false;
    require_throws([&] {
        llama_dsv41_expert_runtime runtime(
                f.tensors, params, [](const llama_expert_store_tensor &) { return ggml_backend_cpu_buffer_type(); });
    });

    params.cache_slots = 1;
    params.cache_bytes = f.cache_bytes(1) - 1;
    require_throws([&] {
        llama_dsv41_expert_runtime runtime(
                f.tensors, params, [](const llama_expert_store_tensor &) { return ggml_backend_cpu_buffer_type(); });
    });

    auto runtime = f.make_runtime(1);
    runtime.acquire_context();
    require_throws([&] { runtime.acquire_context(); });
    runtime.release_context();
    runtime.acquire_context();
    runtime.release_context();
}

void test_remap_upload_and_eviction(const fixture & f) {
    struct upload_record {
        std::string name;
        size_t offset;
        std::vector<uint8_t> bytes;
    };
    std::vector<upload_record> uploads;
    auto runtime = f.make_runtime(3, [&](ggml_tensor * tensor, size_t offset, const void * data, size_t size) {
        uploads.push_back({ tensor->name, offset, std::vector<uint8_t>(
                static_cast<const uint8_t *>(data), static_cast<const uint8_t *>(data) + size) });
        ggml_backend_tensor_set(tensor, data, offset, size);
    });

    const auto remapped = runtime.remap(0, { 3, 3, 1, 2 });
    REQUIRE(remapped == std::vector<int32_t>({ 2, 2, 0, 1 }));
    REQUIRE(uploads.size() == 9);
    REQUIRE(uploads[0].bytes.back() == 0x11);
    REQUIRE(uploads[1].bytes.back() == 0x21);
    REQUIRE(uploads[2].bytes.back() == 0x31);
    require_throws([&] { runtime.remap(0, { 0 }); });
    runtime.release(0);

    uploads.clear();
    REQUIRE(runtime.remap(0, { 3 }).front() == 2);
    REQUIRE(uploads.empty());
    runtime.release(0);

    REQUIRE(runtime.remap(0, { 4 }).front() == 0);
    runtime.release(0);
}

void test_capacity_and_upload_failure(const fixture & f) {
    auto runtime = f.make_runtime(2);
    require_throws([&] { runtime.remap(0, { 0, 1, 2 }); });
    REQUIRE(runtime.remap(0, { 0, 0, 1 }).size() == 3);
    runtime.release(0);

    size_t calls = 0;
    auto failing = f.make_runtime(1, [&](ggml_tensor * tensor, size_t offset, const void * data, size_t size) {
        if (++calls == 5) {
            throw std::runtime_error("synthetic upload failure");
        }
        ggml_backend_tensor_set(tensor, data, offset, size);
    });
    REQUIRE(failing.remap(0, { 0 }).front() == 0);
    failing.release(0);
    require_throws([&] { failing.remap(0, { 1 }); });
    const size_t after_failure = calls;
    REQUIRE(failing.remap(0, { 0 }).front() == 0);
    REQUIRE(calls == after_failure + 3);
    failing.release(0);
}

void test_publication_allocation_failure(const fixture & f) {
    size_t attempts = 0;
    auto runtime = f.make_runtime(1, {}, [&] {
        if (attempts++ == 0) {
            throw std::bad_alloc();
        }
    });

    require_throws([&] { runtime.remap(0, { 0 }); });
    REQUIRE(runtime.remap(0, { 1 }).front() == 0);
    runtime.release(0);
}

void test_graph_callbacks(const fixture & f) {
    auto runtime = f.make_runtime(2);
    ggml_init_params params = {
        /*.mem_size   =*/ 2*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, &buft, 1, 64, false, true);
    REQUIRE(sched != nullptr);

    ggml_tensor * selected = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, 1);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, fixture::n_embd, 3, 1);
    ggml_set_input(selected);
    ggml_set_input(input);
    ggml_tensor * remapped = llama_dsv41_build_expert_remap(ctx, selected, runtime, 0, sched, backend);
    ggml_tensor * expert_values = ggml_mul_mat_id(
            ctx, runtime.cache_tensor(0, LLAMA_EXPERT_PROJECTION_GATE), input, remapped);
    ggml_tensor * release = llama_dsv41_build_expert_release(ctx, expert_values, runtime, 0, sched, backend);
    ggml_set_output(remapped);
    ggml_set_output(release);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, release);
    REQUIRE(ggml_backend_sched_alloc_graph(sched, graph));
    const int32_t original[] = { 3, 1, 1 };
    std::vector<float> input_data(fixture::n_embd*3, 1.0f);
    ggml_backend_tensor_set(selected, original, 0, sizeof(original));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size()*sizeof(float));
    REQUIRE(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    const std::string error = runtime.consume_error();
    if (!error.empty()) {
        throw std::runtime_error(error);
    }
    REQUIRE(ggml_backend_sched_get_tensor_backend(sched, remapped) == backend);
    REQUIRE(ggml_backend_sched_get_tensor_backend(sched, release) == backend);
    REQUIRE(runtime.remap(0, { 0 }).front() >= 0);
    runtime.release(0);
    runtime.release_all();

    const int32_t over_capacity[] = { 3, 1, 2 };
    ggml_backend_tensor_set(selected, over_capacity, 0, sizeof(over_capacity));
    REQUIRE(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    REQUIRE(!runtime.consume_error().empty());
    int32_t safe_ids[3] = { -1, -1, -1 };
    ggml_backend_tensor_get(remapped, safe_ids, 0, sizeof(safe_ids));
    REQUIRE(safe_ids[0] == 0 && safe_ids[1] == 0 && safe_ids[2] == 0);
    REQUIRE(runtime.remap(0, { 0 }).front() >= 0);
    runtime.release(0);

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

void test_graph_upload_failure_sentinel(const fixture & f) {
    size_t calls = 0;
    auto runtime = f.make_runtime(1, [&](ggml_tensor * tensor, size_t offset, const void * data, size_t size) {
        if (++calls == 5) {
            throw std::runtime_error("synthetic graph upload failure");
        }
        ggml_backend_tensor_set(tensor, data, offset, size);
    });
    ggml_init_params params = {
        /*.mem_size   =*/ 2*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, &buft, 1, 64, false, true);
    REQUIRE(sched != nullptr);

    ggml_tensor * selected = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, fixture::n_embd, 1, 1);
    ggml_set_input(selected);
    ggml_set_input(input);
    ggml_tensor * remapped = llama_dsv41_build_expert_remap(ctx, selected, runtime, 0, sched, backend);
    ggml_tensor * expert_values = ggml_mul_mat_id(
            ctx, runtime.cache_tensor(0, LLAMA_EXPERT_PROJECTION_GATE), input, remapped);
    ggml_tensor * release = llama_dsv41_build_expert_release(ctx, expert_values, runtime, 0, sched, backend);
    ggml_set_output(remapped);
    ggml_set_output(release);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, release);
    REQUIRE(ggml_backend_sched_alloc_graph(sched, graph));

    std::vector<float> input_data(fixture::n_embd, 1.0f);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size()*sizeof(float));
    const int32_t first[] = { 0 };
    ggml_backend_tensor_set(selected, first, 0, sizeof(first));
    REQUIRE(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    REQUIRE(runtime.consume_error().empty());

    const int32_t failed[] = { 1 };
    ggml_backend_tensor_set(selected, failed, 0, sizeof(failed));
    REQUIRE(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    REQUIRE(runtime.consume_error().find("synthetic graph upload failure") != std::string::npos);
    int32_t safe_id = -1;
    ggml_backend_tensor_get(remapped, &safe_id, 0, sizeof(safe_id));
    REQUIRE(safe_id == 0);

    REQUIRE(runtime.remap(0, { 2 }).front() == 0);
    runtime.release(0);
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

void test_grovemoe_lookup_ids() {
    ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, &buft, 1, 64, false, true);
    REQUIRE(sched != nullptr);

    ggml_tensor * selected = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    ggml_tensor * explicit_slots = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    ggml_set_input(selected);
    ggml_set_input(explicit_slots);
    const llm_moe_expert_ids grovemoe = llm_build_moe_expert_ids(
            ctx, LLM_ARCH_GROVEMOE, selected, nullptr, 2, 8, 4);
    const llm_moe_expert_ids deepseek = llm_build_moe_expert_ids(
            ctx, LLM_ARCH_DEEPSEEK41, selected, explicit_slots, 8, 8, 0);
    REQUIRE(grovemoe.routing == selected);
    REQUIRE(grovemoe.lookup != selected);
    REQUIRE(deepseek.routing == selected);
    REQUIRE(deepseek.lookup == explicit_slots);
    ggml_set_output(grovemoe.routing);
    ggml_set_output(grovemoe.lookup);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, grovemoe.routing);
    ggml_build_forward_expand(graph, grovemoe.lookup);
    REQUIRE(ggml_backend_sched_alloc_graph(sched, graph));

    const int32_t original[] = { 7, 1 };
    ggml_backend_tensor_set(selected, original, 0, sizeof(original));
    REQUIRE(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    int32_t routing_ids[2] = {};
    int32_t chunk_ids[2] = {};
    ggml_backend_tensor_get(grovemoe.routing, routing_ids, 0, sizeof(routing_ids));
    ggml_backend_tensor_get(grovemoe.lookup, chunk_ids, 0, sizeof(chunk_ids));
    REQUIRE(routing_ids[0] == 7 && routing_ids[1] == 1);
    REQUIRE(chunk_ids[0] == 1 && chunk_ids[1] == 0);

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

struct sync_test_context {
    llama_dsv41_expert_runtime * runtime = nullptr;
    int synchronize_count = 0;
    bool saw_pinned = false;
};

const char * sync_test_backend_name(ggml_backend_t) {
    return "dsv41-sync-test";
}

void sync_test_backend_synchronize(ggml_backend_t backend) {
    auto * state = static_cast<sync_test_context *>(backend->context);
    state->synchronize_count++;
    try {
        state->runtime->remap(0, { 2 });
    } catch (const std::exception &) {
        state->saw_pinned = true;
    }
}

const char * sync_test_device_name(ggml_backend_dev_t) {
    return "dsv41-sync-test";
}

enum ggml_backend_dev_type sync_test_device_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;
}

bool sync_test_device_supports_op(ggml_backend_dev_t, const ggml_tensor *) {
    return true;
}

bool sync_test_device_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return buft == ggml_backend_cpu_buffer_type();
}

void test_release_after_sync(const fixture & f) {
    auto runtime = f.make_runtime(1);
    REQUIRE(runtime.remap(0, { 1 }).front() == 0);

    sync_test_context state = { &runtime };
    ggml_backend_device device = {};
    device.iface.get_name = sync_test_device_name;
    device.iface.get_type = sync_test_device_type;
    device.iface.supports_op = sync_test_device_supports_op;
    device.iface.supports_buft = sync_test_device_supports_buft;
    ggml_backend backend = {};
    backend.iface.get_name = sync_test_backend_name;
    backend.iface.synchronize = sync_test_backend_synchronize;
    backend.device = &device;
    backend.context = &state;
    ggml_backend_t backends[] = { &backend };
    ggml_backend_buffer_type_t bufts[] = { ggml_backend_cpu_buffer_type() };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 1, 16, false, false);
    REQUIRE(sched != nullptr);

    runtime.release_all_after_sync(sched);
    REQUIRE(state.synchronize_count == 1);
    REQUIRE(state.saw_pinned);
    REQUIRE(runtime.remap(0, { 2 }).front() == 0);
    runtime.release(0);
    ggml_backend_sched_free(sched);
}

void test_original_and_slot_ids() {
    ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, &buft, 1, 64, false, true);
    REQUIRE(sched != nullptr);

    ggml_tensor * cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, 2);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 2, 1);
    ggml_tensor * slots = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    ggml_tensor * probs = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 4, 1);
    ggml_tensor * original = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    for (ggml_tensor * tensor : { cache, input, slots, probs, original }) {
        ggml_set_input(tensor);
    }

    ggml_tensor * expert_values = ggml_mul_mat_id(ctx, cache, input, slots);
    ggml_tensor * routing_weights = ggml_get_rows(ctx, probs, original);
    ggml_set_output(expert_values);
    ggml_set_output(routing_weights);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, expert_values);
    ggml_build_forward_expand(graph, routing_weights);
    REQUIRE(ggml_backend_sched_alloc_graph(sched, graph));

    const float cache_data[] = { 10.0f, 20.0f };
    const float input_data[] = { 1.0f, 1.0f };
    const int32_t slot_data[] = { 1, 0 };
    const float prob_data[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const int32_t original_data[] = { 3, 1 };
    ggml_backend_tensor_set(cache, cache_data, 0, sizeof(cache_data));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    ggml_backend_tensor_set(slots, slot_data, 0, sizeof(slot_data));
    ggml_backend_tensor_set(probs, prob_data, 0, sizeof(prob_data));
    ggml_backend_tensor_set(original, original_data, 0, sizeof(original_data));
    REQUIRE(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);

    float expert_result[2] = {};
    float weight_result[2] = {};
    ggml_backend_tensor_get(expert_values, expert_result, 0, sizeof(expert_result));
    ggml_backend_tensor_get(routing_weights, weight_result, 0, sizeof(weight_result));
    REQUIRE(expert_result[0] == 20.0f && expert_result[1] == 10.0f);
    REQUIRE(weight_result[0] == 4.0f && weight_result[1] == 2.0f);

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

}

int main() {
    try {
        test_registration();
        fixture f;
        test_configuration(f);
        test_remap_upload_and_eviction(f);
        test_capacity_and_upload_failure(f);
        test_publication_allocation_failure(f);
        test_graph_callbacks(f);
        test_graph_upload_failure_sentinel(f);
        test_grovemoe_lookup_ids();
        test_release_after_sync(f);
        test_original_and_slot_ids();
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-deepseek41-expert: %s\n", error.what());
        return 1;
    }
    return 0;
}
