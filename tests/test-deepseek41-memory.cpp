#include "../src/llama-batch.h"
#include "../src/llama-io.h"
#include "../src/llama-memory-dsv41.h"

#include "ggml-backend.h"

#include <algorithm>
#include <array>
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

static llama_dsv41_memory_config small_config(
        uint32_t n_ctx = 256,
        uint32_t n_seq = 3,
        uint32_t n_ubatch = 128) {
    llama_dsv41_memory_config config;
    config.n_ctx = n_ctx;
    config.n_seq = n_seq;
    config.n_ubatch = n_ubatch;
    config.kv_width = 8;
    config.index_width = 4;
    config.candidate_topk_blocks = 4;
    config.candidate_block_size = 8;
    config.ratios.resize(LLAMA_DSV41_N_LAYER);
    for (uint32_t il = 0; il < LLAMA_DSV41_N_LAYER; ++il) {
        config.ratios[il] = llama_dsv41_compress_ratio(il);
    }
    config.buft_for_layer = [](int32_t) { return ggml_backend_cpu_buffer_type(); };
    return config;
}

static llama_ubatch make_ubatch(llama_pos start, uint32_t count, llama_seq_id seq_id) {
    llama_batch_allocr allocator(1);
    llama_ubatch ubatch = allocator.ubatch_reserve(count, 1);
    ubatch.data->seq_id_data.resize(count);
    ubatch.data->seq_id_unq = { seq_id };
    ubatch.seq_id_unq = ubatch.data->seq_id_unq.data();
    ubatch.seq_idx[seq_id] = 0;
    for (uint32_t i = 0; i < count; ++i) {
        ubatch.token[i] = 10 + i;
        ubatch.pos[i] = start + i;
        ubatch.n_seq_id[i] = 1;
        ubatch.data->seq_id_data[i] = seq_id;
        ubatch.seq_id[i] = &ubatch.data->seq_id_data[i];
        ubatch.output[i] = 1;
    }
    return ubatch;
}

static llama_ubatch make_coupled_ubatch(
        llama_pos start,
        uint32_t count,
        llama_seq_id first,
        llama_seq_id second) {
    llama_ubatch ubatch = make_ubatch(start, count, first);
    ubatch.data->seq_id_data.resize((size_t) count*2);
    ubatch.data->seq_id_unq = { first, second };
    ubatch.seq_id_unq = ubatch.data->seq_id_unq.data();
    ubatch.n_seqs_unq = 2;
    ubatch.seq_idx[first] = 0;
    ubatch.seq_idx[second] = 1;
    for (uint32_t i = 0; i < count; ++i) {
        ubatch.n_seq_id[i] = 2;
        ubatch.data->seq_id_data[2*i] = first;
        ubatch.data->seq_id_data[2*i + 1] = second;
        ubatch.seq_id[i] = ubatch.data->seq_id_data.data() + 2*i;
    }
    return ubatch;
}

static size_t state_sequence_tensor_bytes(const llama_memory_dsv41 & memory) {
    size_t result = 0;
    const auto add = [&](const ggml_tensor * tensor) {
        result += sizeof(uint64_t);
        result += tensor->ne[2] == (int64_t) memory.config().n_seq ?
            tensor->nb[2] : tensor->nb[1];
    };
    for (uint32_t il = 0; il < memory.config().n_layer; ++il) {
        add(memory.raw_k(il));
    }
    for (uint32_t source : memory.config().kv_sources) {
        add(memory.compressed_kv(source));
        add(memory.index_keys(source));
        if (memory.config().ratios[source] == 2) {
            add(memory.compressor_carry_kv(source));
            add(memory.compressor_carry_score(source));
        }
    }
    add(memory.committed_candidate_ids());
    add(memory.position_state());
    return result;
}

class vector_writer : public llama_io_write_i {
public:
    void write(const void * src, size_t size) override {
        const uint8_t * bytes = static_cast<const uint8_t *>(src);
        data.insert(data.end(), bytes, bytes + size);
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        const size_t old_size = data.size();
        data.resize(old_size + size);
        ggml_backend_tensor_get(tensor, data.data() + old_size, offset, size);
    }

    size_t n_bytes() override {
        return data.size();
    }

    std::vector<uint8_t> data;
};

class vector_reader : public llama_io_read_i {
public:
    explicit vector_reader(const std::vector<uint8_t> & data) : data(data) {}

    void read(void * dst, size_t size) override {
        if (offset > data.size() || size > data.size() - offset) {
            throw std::runtime_error("test state buffer is truncated");
        }
        std::memcpy(dst, data.data() + offset, size);
        offset += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t tensor_offset, size_t size) override {
        (void) tensor;
        (void) tensor_offset;
        (void) size;
        throw std::runtime_error("DeepSeek V4.1 state restore must read tensor bytes before publication");
    }

    size_t n_bytes() override {
        return offset;
    }

private:
    const std::vector<uint8_t> & data;
    size_t offset = 0;
};

class device_writer : public llama_io_write_i {
public:
    void write(const void * src, size_t size) override {
        const uint8_t * bytes = static_cast<const uint8_t *>(src);
        metadata.insert(metadata.end(), bytes, bytes + size);
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        tensors.emplace_back(size);
        ggml_backend_tensor_get(tensor, tensors.back().data(), offset, size);
        tensor_bytes += size;
    }

    size_t n_bytes() override {
        return metadata.size();
    }

    std::vector<uint8_t> metadata;
    std::vector<std::vector<uint8_t>> tensors;
    size_t tensor_bytes = 0;
};

class device_reader : public llama_io_read_i {
public:
    explicit device_reader(const device_writer & writer) :
        metadata(writer.metadata),
        tensors(writer.tensors) {
    }

    void read(void * dst, size_t size) override {
        if (offset > metadata.size() || size > metadata.size() - offset) {
            throw std::runtime_error("test on-device metadata is truncated");
        }
        std::memcpy(dst, metadata.data() + offset, size);
        offset += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t tensor_offset, size_t size) override {
        if (i_tensor >= tensors.size() || tensors[i_tensor].size() != size) {
            throw std::runtime_error("test on-device tensor layout differs");
        }
        ggml_backend_tensor_set(tensor, tensors[i_tensor].data(), tensor_offset, size);
        ++i_tensor;
    }

    size_t n_bytes() override {
        return offset;
    }

    size_t tensor_reads() const {
        return i_tensor;
    }

private:
    const std::vector<uint8_t> & metadata;
    const std::vector<std::vector<uint8_t>> & tensors;
    size_t offset = 0;
    size_t i_tensor = 0;
};

static void test_transaction_commit_rollback() {
    llama_memory_dsv41 memory(small_config());
    llama_ubatch ubatch = make_ubatch(0, 1, 0);

    auto full_base = memory.init_full();
    auto * full = dynamic_cast<llama_memory_dsv41_context *>(full_base.get());
    check(full != nullptr && full->get_ubatch().n_tokens == memory.config().n_ubatch,
          "full memory context did not expose a bounded reserve plan");
    check(full->plan().sources.size() == memory.config().kv_sources.size(),
          "full memory context source plan mismatch");
    llama_ubatch reserve_decode = make_ubatch(0, 1, 0);
    check(full->topology(reserve_decode, 1).n_tokens == 1 &&
          full->plan().positions.size() == 1,
          "full memory context did not resize its synthetic decode plan");
    llama_ubatch reserve_prefill = make_ubatch(0, 7, 0);
    check(full->graph_plan(reserve_prefill).positions.size() == 7 &&
          full->topology(reserve_prefill, 3).n_outputs == 3,
          "full memory context did not resize its synthetic prefill plan");

    const size_t row_bytes = memory.raw_k(0)->nb[1];
    std::vector<uint8_t> original(row_bytes, 0x31);
    std::vector<uint8_t> changed(row_bytes, 0x72);
    std::vector<uint8_t> actual(row_bytes);
    ggml_backend_tensor_set(memory.raw_k(0), original.data(), 0, row_bytes);

    llama_memory_dsv41_context rollback_context(&memory, std::vector<llama_ubatch> { ubatch });
    check(rollback_context.apply(), "transaction prepare failed");
    check(memory.seq_pos_max(0) == -1, "prepare published the position early");
    ggml_backend_tensor_set(memory.raw_k(0), changed.data(), 0, row_bytes);
    rollback_context.stage_candidate_ids(0, { 7 });
    rollback_context.rollback();
    ggml_backend_tensor_get(memory.raw_k(0), actual.data(), 0, row_bytes);
    check(actual == original, "rollback did not restore a graph-written raw row");
    check(memory.seq_pos_max(0) == -1, "rollback changed the committed position");
    check(memory.sequence_candidate_ids(0).empty(), "rollback published candidate IDs");

    llama_memory_dsv41_context commit_context(&memory, std::vector<llama_ubatch> { ubatch });
    check(commit_context.apply(), "commit prepare failed");
    commit_context.stage_candidate_ids(0, { 3 });
    const auto topology = commit_context.topology(ubatch, 1);
    check(topology.n_tokens == 1 && topology.n_seqs == 1 && topology.n_outputs == 1,
          "graph topology batch identity mismatch");
    check(topology.start_positions == std::vector<llama_pos>({ 0 }),
          "graph topology start position mismatch");
    auto reusable_topology = topology;
    reusable_topology.seq_ids = { 2 };
    reusable_topology.start_positions = { 17 };
    reusable_topology.transaction_generation++;
    check(topology.same_topology(reusable_topology),
          "graph reuse rejected refreshed sequence and position inputs");
    reusable_topology.candidate_width++;
    check(!topology.same_topology(reusable_topology),
          "graph reuse accepted a changed candidate workspace shape");
    commit_context.commit();
    check(memory.seq_pos_max(0) == 0, "commit did not publish the position");
    check(memory.sequence_candidate_ids(0) == std::vector<int32_t>({ 3 }),
          "commit did not publish candidate IDs");

    llama_memory_dsv41_context candidate_rollback(
            &memory, std::vector<llama_ubatch> { make_ubatch(1, 1, 0) });
    check(candidate_rollback.apply(), "candidate rollback prepare failed");
    candidate_rollback.stage_candidate_ids(0, { 9 });
    candidate_rollback.rollback();
    check(memory.sequence_candidate_ids(0) == std::vector<int32_t>({ 3 }),
          "candidate rollback changed committed IDs");

    llama_memory_dsv41_context over_capacity(
            &memory,
            std::vector<llama_ubatch> {
                make_ubatch(memory.config().n_ctx, 1, 1),
            });
    check(!over_capacity.apply(), "over-capacity transaction was accepted");
    check(memory.seq_pos_max(1) == -1, "over-capacity prepare mutated sequence state");

    llama_memory_dsv41_context coupled(
            &memory,
            std::vector<llama_ubatch> {
                make_coupled_ubatch(0, 1, 1, 2),
            });
    check(!coupled.apply(), "coupled sequence evaluation was accepted");
    check(memory.seq_pos_max(1) == -1 && memory.seq_pos_max(2) == -1,
          "rejected coupled sequence evaluation mutated state");
}

static void test_window_compression_and_sequences() {
    llama_memory_dsv41 memory(small_config());
    llama_ubatch prefill = make_ubatch(0, 128, 0);
    llama_memory_dsv41_context prefill_context(
            &memory, std::vector<llama_ubatch> { prefill });
    check(prefill_context.apply(), "128-token prefill prepare failed");
    const auto & prefill_plan = prefill_context.plan();
    check(prefill_plan.raw.n_visible.back() == 128, "raw window width at 127 mismatch");
    check(prefill_plan.raw.read_idxs[127*128] == 0 &&
          prefill_plan.raw.read_idxs[127*128 + 127] == 127,
          "raw window order at 127 mismatch");
    check(prefill_plan.sources[0].ratio == 2 &&
          prefill_plan.sources[0].compression.n_visible.back() == 64,
          "ratio-2 visibility at 127 mismatch");
    prefill_context.stage_candidate_ids(127, { 4, 3, 2, 1 });
    prefill_context.commit();
    check(memory.seq_pos_max(0) == 127, "prefill position mismatch");

    const size_t raw_row_bytes = memory.raw_k(0)->nb[1];
    const size_t carry_plane_bytes = memory.compressor_carry_kv(2)->nb[2];
    std::vector<uint8_t> raw_before(raw_row_bytes, 0x21);
    std::vector<uint8_t> raw_after(raw_row_bytes, 0x72);
    std::vector<uint8_t> carry_before(carry_plane_bytes, 0x32);
    std::vector<uint8_t> carry_after(carry_plane_bytes, 0x83);
    std::vector<uint8_t> actual;
    ggml_backend_tensor_set(memory.raw_k(0), raw_before.data(), 0, raw_row_bytes);
    ggml_backend_tensor_set(memory.compressor_carry_kv(2), carry_before.data(), 0, carry_plane_bytes);

    llama_ubatch decode = make_ubatch(128, 1, 0);
    llama_memory_dsv41_context decode_context(
            &memory, std::vector<llama_ubatch> { decode });
    check(decode_context.apply(), "boundary decode prepare failed");
    const auto & decode_plan = decode_context.plan();
    check(decode_plan.raw.write_idxs == std::vector<int64_t>({ 0 }),
          "raw ring write did not wrap at 128");
    check(decode_plan.raw.read_idxs.front() == 1 &&
          decode_plan.raw.read_idxs[127] == 0,
          "raw ring order at 128 mismatch");
    check(decode_plan.sources[0].compression.n_visible == std::vector<int32_t>({ 64 }),
          "ratio-2 carry visibility at 128 mismatch");
    check(decode_plan.sources[0].compression.state_persist_dst_idxs == std::vector<int32_t>({ 0 }),
          "ratio-2 carry publication row mismatch");
    ggml_backend_tensor_set(memory.raw_k(0), raw_after.data(), 0, raw_row_bytes);
    ggml_backend_tensor_set(memory.compressor_carry_kv(2), carry_after.data(), 0, carry_plane_bytes);
    decode_context.stage_candidate_ids(0, { 8, 7, 6, 5 });
    decode_context.commit();
    check(memory.seq_pos_max(0) == 128, "decode position mismatch");
    check(memory.seq_rm(0, 128, -1), "immediate rollback failed");
    check(memory.seq_pos_max(0) == 127, "immediate rollback position mismatch");
    check(memory.sequence_candidate_ids(0) == std::vector<int32_t>({ 4, 3, 2, 1 }),
          "immediate rollback did not restore candidate state");
    actual.resize(raw_row_bytes);
    ggml_backend_tensor_get(memory.raw_k(0), actual.data(), 0, raw_row_bytes);
    check(actual == raw_before, "immediate rollback did not restore the raw ring row");
    actual.resize(carry_plane_bytes);
    ggml_backend_tensor_get(memory.compressor_carry_kv(2), actual.data(), 0, carry_plane_bytes);
    check(actual == carry_before, "immediate rollback did not restore compressor carry");

    llama_memory_dsv41_context recommit_context(
            &memory, std::vector<llama_ubatch> { decode });
    check(recommit_context.apply(), "decode after rollback prepare failed");
    recommit_context.stage_candidate_ids(0, { 8, 7, 6, 5 });
    recommit_context.commit();
    memory.seq_cp(0, 1, -1, -1);
    check(memory.seq_pos_max(1) == 128, "full sequence copy lost the position");
    check(memory.sequence_candidate_ids(1) == std::vector<int32_t>({ 8, 7, 6, 5 }),
          "full sequence copy lost candidate state");
    memory.seq_keep(1);
    check(memory.seq_pos_max(0) == -1 && memory.seq_pos_max(1) == 128,
          "sequence keep retained another sequence");
    check(!memory.seq_rm(1, 128, -1), "copied sequence rollback was accepted without a retained snapshot");
    check(memory.seq_pos_max(1) == 128, "rejected copied-sequence rollback mutated the position");
    check(!memory.seq_rm(1, 64, -1), "non-immediate suffix rollback was accepted");
    check(memory.seq_pos_max(1) == 128, "rejected suffix rollback mutated the position");
    check(!memory.seq_rm(1, 64, 96), "interior range removal was accepted");
    check(!memory.seq_rm(-1, 0, -1), "partial wildcard removal was accepted");
    check(memory.seq_pos_max(1) == 128, "rejected wildcard removal mutated the position");
    expect_invalid(
            [&] { memory.seq_cp(1, 2, 64, -1); },
            "partial sequence copy was accepted");
    check(memory.seq_rm(-1, -1, -1), "wildcard full-memory removal failed");
    check(memory.seq_pos_max(0) == -1 &&
          memory.seq_pos_max(1) == -1 &&
          memory.seq_pos_max(2) == -1,
          "wildcard full-memory removal retained sequence state");
    llama_memory_dsv41_context negative_wildcard_context(
            &memory, std::vector<llama_ubatch> { make_ubatch(0, 1, 2) });
    check(negative_wildcard_context.apply(), "negative wildcard setup failed");
    negative_wildcard_context.commit();
    check(memory.seq_rm(-2, -1, -1), "negative wildcard full-memory removal failed");
    check(memory.seq_pos_max(2) == -1, "negative wildcard removal retained sequence state");
}

static void test_long_prefill_raw_publication() {
    llama_memory_dsv41 memory(small_config(256, 1, 256));
    llama_memory_dsv41_context context(
            &memory,
            std::vector<llama_ubatch> { make_ubatch(0, 256, 0) });
    check(context.apply(), "long prefill prepare failed");
    const auto & raw = context.plan().raw;
    check(raw.persist_src_idxs.size() == LLAMA_DSV41_N_SWA,
          "long prefill retained duplicate raw ring writes");
    check(raw.persist_src_idxs.front() == 128 &&
          raw.persist_src_idxs.back() == 255,
          "long prefill did not retain the final raw window");
    std::vector<int64_t> unique = raw.write_idxs;
    std::sort(unique.begin(), unique.end());
    check(std::adjacent_find(unique.begin(), unique.end()) == unique.end(),
          "long prefill raw ring write indexes are not unique");
    context.rollback();
}

static void test_state_save_load() {
    llama_memory_dsv41 memory(small_config());
    llama_ubatch ubatch = make_ubatch(0, 3, 0);
    llama_memory_dsv41_context context(&memory, std::vector<llama_ubatch> { ubatch });
    check(context.apply(), "state test prepare failed");
    context.stage_candidate_ids(2, { 11 });
    context.commit();

    const size_t row_bytes = memory.raw_k(0)->nb[1];
    std::vector<uint8_t> expected(row_bytes, 0x5a);
    std::vector<uint8_t> actual(row_bytes);
    ggml_backend_tensor_set(memory.raw_k(0), expected.data(), 2*row_bytes, row_bytes);

    vector_writer writer;
    memory.state_write(writer, 0);
    expect_runtime(
            [&] {
                vector_reader nonempty_reader(writer.data);
                memory.state_read(nonempty_reader, 0);
            },
            "state restore overwrote a non-empty destination");
    memory.clear(true);
    vector_reader reader(writer.data);
    memory.state_read(reader, 2);
    check(memory.seq_pos_max(2) == 2, "state restore lost the position");
    check(memory.sequence_candidate_ids(2) == std::vector<int32_t>({ 11 }),
          "state restore lost candidate IDs");
    ggml_backend_tensor_get(
            memory.raw_k(0),
            actual.data(),
            ((size_t) 2*memory.config().raw_window + 2)*row_bytes,
            row_bytes);
    check(actual == expected, "state restore lost raw cache data");

    std::vector<uint8_t> truncated = writer.data;
    truncated.pop_back();
    expect_runtime(
            [&] {
                vector_reader truncated_reader(truncated);
                memory.state_read(truncated_reader, 1);
            },
            "truncated state restore was accepted");
    check(memory.seq_pos_max(1) == -1 && memory.seq_pos_max(2) == 2,
          "failed state restore changed committed sequence state");

    vector_writer full_writer;
    memory.state_write(full_writer, -1);
    memory.clear(true);
    std::vector<uint8_t> incomplete = full_writer.data;
    const uint32_t incomplete_count = memory.config().n_seq - 1;
    std::memcpy(
            incomplete.data() + sizeof(uint64_t) + sizeof(uint32_t),
            &incomplete_count,
            sizeof(incomplete_count));
    expect_runtime(
            [&] {
                vector_reader incomplete_reader(incomplete);
                memory.state_read(incomplete_reader, -1);
            },
            "incomplete full state restore was accepted");
    for (uint32_t seq = 0; seq < memory.config().n_seq; ++seq) {
        check(memory.seq_pos_max(seq) == -1, "incomplete full restore changed memory");
    }

    std::vector<uint8_t> duplicate = full_writer.data;
    const size_t header_size =
        sizeof(uint64_t) + 3*sizeof(uint32_t);
    const size_t first_record_size =
        sizeof(llama_seq_id) + sizeof(llama_pos) + sizeof(uint32_t) +
        sizeof(uint8_t) + state_sequence_tensor_bytes(memory);
    const llama_seq_id duplicate_id = 0;
    std::memcpy(
            duplicate.data() + header_size + first_record_size,
            &duplicate_id,
            sizeof(duplicate_id));
    expect_runtime(
            [&] {
                vector_reader duplicate_reader(duplicate);
                memory.state_read(duplicate_reader, -1);
            },
            "duplicate full state sequence was accepted");
    for (uint32_t seq = 0; seq < memory.config().n_seq; ++seq) {
        check(memory.seq_pos_max(seq) == -1, "duplicate full restore changed memory");
    }

    vector_reader full_reader(full_writer.data);
    memory.state_read(full_reader, -1);
    check(memory.seq_pos_max(2) == 2, "full state restore lost sequence state");

    device_writer on_device_writer;
    memory.state_write(on_device_writer, 2, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    vector_writer host_writer;
    memory.state_write(host_writer, 2);
    check(on_device_writer.metadata.size() + on_device_writer.tensor_bytes == host_writer.data.size(),
          "on-device state metadata includes tensor payload bytes");
    memory.clear(true);
    device_reader on_device_reader(on_device_writer);
    memory.state_read(on_device_reader, 1, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    check(on_device_reader.tensor_reads() == on_device_writer.tensors.size(),
          "on-device state restore did not consume every tensor");
    check(memory.seq_pos_max(1) == 2 &&
          memory.sequence_candidate_ids(1) == std::vector<int32_t>({ 11 }),
          "on-device state restore lost sequence metadata");
    ggml_backend_tensor_get(
            memory.raw_k(0),
            actual.data(),
            ((size_t) memory.config().raw_window + 2)*row_bytes,
            row_bytes);
    check(actual == expected, "on-device state restore lost raw cache data");
}

static void test_accounting() {
    const auto config = small_config();
    llama_memory_dsv41 memory(small_config());
    memory.set_graph_workspace_size(1234);
    const auto bytes = memory.accounting();
    const uint64_t raw_expected =
        (uint64_t) config.n_layer*config.kv_width*config.raw_window*config.n_seq*sizeof(uint16_t);
    const uint64_t compressed_rows =
        (uint64_t) 3*((config.n_ctx + 1)/2) + config.n_ctx;
    const uint64_t compressed_expected =
        compressed_rows*config.kv_width*config.n_seq*sizeof(uint16_t);
    const uint64_t index_expected =
        compressed_rows*config.index_width*config.n_seq*sizeof(uint16_t);
    const uint64_t carry_expected =
        (uint64_t) 3*2*config.kv_width*config.n_seq*sizeof(uint16_t)*2;
    const uint64_t candidate_scores_expected =
        (uint64_t) (config.n_ctx/config.candidate_block_size)*config.n_ubatch*sizeof(float);
    const uint64_t candidate_ids_expected =
        (uint64_t) config.candidate_topk_blocks*(config.n_ubatch + config.n_seq)*sizeof(int32_t);
    const uint64_t position_rows =
        1 + config.raw_window + 3*((config.n_ctx + 1)/2 + 2) + config.n_ctx;
    const uint64_t position_expected = position_rows*config.n_seq*sizeof(int32_t);

    check(bytes.raw_kv == raw_expected, "raw cache accounting mismatch");
    check(bytes.compressed_kv == compressed_expected, "compressed cache accounting mismatch");
    check(bytes.index_keys == index_expected, "index key accounting mismatch");
    check(bytes.compressor_carry == carry_expected, "compressor carry accounting mismatch");
    check(bytes.candidate_scores == candidate_scores_expected, "candidate score accounting mismatch");
    check(bytes.candidate_ids == candidate_ids_expected, "candidate ID accounting mismatch");
    check(bytes.position_state == position_expected, "position accounting mismatch");
    check(bytes.graph_workspace == 1234, "graph workspace accounting mismatch");
    const auto breakdown = memory.memory_breakdown();
    check(breakdown.size() == 1 &&
          breakdown.begin()->second >= bytes.total() - bytes.graph_workspace,
          "memory breakdown does not include all allocated state");

    auto no_alloc_config = small_config();
    no_alloc_config.no_alloc = true;
    llama_memory_dsv41 no_alloc_memory(std::move(no_alloc_config));
    check(no_alloc_memory.raw_k(0)->buffer == nullptr,
          "no-allocation memory probe allocated cache storage");
    const auto no_alloc_bytes = no_alloc_memory.accounting();
    check(no_alloc_bytes.total() == bytes.total() - bytes.graph_workspace,
          "no-allocation memory accounting differs from allocated state");
}

#if !defined(_WIN32)
static llama_engram_layout make_engram_layout() {
    llama_engram_layout layout;
    layout.encoding = LLAMA_DSV41_ENGRAM_ENCODING;
    layout.layer_ids = { 1, 14 };
    layout.token_map.resize(32);
    for (size_t i = 0; i < layout.token_map.size(); ++i) {
        layout.token_map[i] = i;
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

struct engram_test_file {
    std::string path = "test-dsv41-memory-engram.bin";
    int fd = -1;
    std::array<llama_dsv41_engram_extent, LLAMA_ENGRAM_LAYERS> extents;

    explicit engram_test_file(const llama_engram_layout & layout) {
        fd = open(path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0600);
        check(fd >= 0, "failed to create Engram memory test file");
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
            for (uint32_t row = 0; row < layout.rows[layer]; ++row) {
                uint8_t data[LLAMA_ENGRAM_ROW_BYTES];
                std::fill(data, data + LLAMA_ENGRAM_ROW_BYTES, (uint8_t) (8 + row));
                check(
                        pwrite(fd, data, sizeof(data), offset + (uint64_t) row*sizeof(data)) ==
                            (ssize_t) sizeof(data),
                        "failed to write Engram memory test row");
            }
            offset += (uint64_t) layout.rows[layer]*LLAMA_ENGRAM_ROW_BYTES + 4096;
        }
    }

    ~engram_test_file() {
        if (fd >= 0) {
            close(fd);
        }
        unlink(path.c_str());
    }
};

static void test_engram_transaction() {
    const llama_engram_layout layout = make_engram_layout();
    engram_test_file file(layout);
    auto config = small_config(2048, 1, 4);
    auto runtime = std::make_unique<llama_dsv41_engram_runtime>(
            layout, file.extents, config.n_ubatch);
    llama_dsv41_engram_runtime * runtime_ptr = runtime.get();
    config.engram = std::move(runtime);
    llama_memory_dsv41 memory(std::move(config));

    const size_t row_bytes = memory.raw_k(0)->nb[1];
    std::vector<uint8_t> changed(row_bytes, 0x6b);
    std::vector<uint8_t> actual(row_bytes);
    llama_memory_dsv41_context stale(
            &memory, std::vector<llama_ubatch> { make_ubatch(0, 1, 0) });
    check(stale.apply(), "stale Engram transaction prepare failed");
    ggml_backend_tensor_set(memory.raw_k(0), changed.data(), 0, row_bytes);
    stale.stage_candidate_ids(0, { 9 });
    runtime_ptr->seq_remove(0);
    expect_runtime([&] { stale.commit(); }, "stale Engram transaction commit was accepted");
    stale.rollback();
    ggml_backend_tensor_get(memory.raw_k(0), actual.data(), 0, row_bytes);
    check(std::all_of(actual.begin(), actual.end(), [](uint8_t value) { return value == 0; }),
          "failed commit did not restore graph-written state");
    check(memory.seq_pos_max(0) == -1 && memory.sequence_candidate_ids(0).empty(),
          "failed commit published sequence state");

    llama_memory_dsv41_context first(
            &memory, std::vector<llama_ubatch> { make_ubatch(0, 2, 0) });
    check(first.apply() && first.engram_transaction() != nullptr,
          "Engram transaction was not exposed");
    for (uint32_t layer = 0; layer < LLAMA_ENGRAM_LAYERS; ++layer) {
        const auto packed = first.engram_row_ids(layer);
        const uint32_t * strided = first.engram_transaction()->row_ids(layer);
        check(packed.size() == 2*LLAMA_ENGRAM_COLS, "packed Engram row ID shape mismatch");
        for (uint32_t token = 0; token < 2; ++token) {
            check(std::memcmp(
                    packed.data() + token*LLAMA_ENGRAM_COLS,
                    strided + token*LLAMA_ENGRAM_LAYERS*LLAMA_ENGRAM_COLS,
                    LLAMA_ENGRAM_COLS*sizeof(int32_t)) == 0,
                  "packed Engram row IDs changed the token stride");
        }
    }
    first.stage_candidate_ids(1, { 1 });
    first.commit();

    llama_memory_dsv41_context rolled_back(
            &memory, std::vector<llama_ubatch> { make_ubatch(2, 1, 0) });
    check(rolled_back.apply(), "Engram rollback prepare failed");
    rolled_back.rollback();

    llama_memory_dsv41_context second(
            &memory, std::vector<llama_ubatch> { make_ubatch(2, 1, 0) });
    check(second.apply(), "Engram state advanced during rollback");
    second.stage_candidate_ids(0, { 2 });
    second.commit();
    check(memory.seq_pos_max(0) == 2, "Engram commit position mismatch");
    check(memory.seq_rm(0, 2, -1), "Engram suffix rollback at transaction boundary failed");
    check(memory.seq_pos_max(0) == 1, "Engram suffix rollback did not restore position");

    for (llama_pos pos = 2; pos < 1024; ++pos) {
        llama_memory_dsv41_context decode(
                &memory, std::vector<llama_ubatch> { make_ubatch(pos, 1, 0) });
        check(decode.apply(), "long Engram decode prepare failed");
        decode.commit();
        check(memory.retained_rollback_count() == 1,
              "Engram decode retained more than the immediate rollback state");
    }
    check(memory.seq_rm(0, 1023, -1), "long Engram decode immediate rollback failed");
    check(memory.seq_pos_max(0) == 1022, "long Engram decode rollback restored the wrong position");
    check(memory.retained_rollback_count() == 0,
          "Engram rollback retained an obsolete rollback state");
}
#endif

int main() {
    test_transaction_commit_rollback();
    test_window_compression_and_sequences();
    test_long_prefill_raw_publication();
    test_state_save_load();
    test_accounting();
#if !defined(_WIN32)
    test_engram_transaction();
#endif
    return 0;
}
