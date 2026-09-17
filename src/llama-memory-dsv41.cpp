#include "llama-memory-dsv41.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace {

constexpr uint64_t DSV41_STATE_MAGIC = 0x314d454d31345644ULL;
constexpr uint32_t DSV41_STATE_VERSION = 1;

uint64_t hash_mix(uint64_t hash, uint64_t value) {
    hash ^= value;
    return hash*1099511628211ULL;
}

uint64_t hash_string(uint64_t hash, const char * value) {
    while (*value != '\0') {
        hash = hash_mix(hash, (uint8_t) *value++);
    }
    return hash;
}

uint32_t source_capacity(const llama_dsv41_memory_config & config, uint32_t source) {
    const uint32_t ratio = config.ratios.at(source);
    return (config.n_ctx + ratio - 1)/ratio;
}

llama_dsv41_memory_config make_model_config(
        const llama_model & model,
        ggml_type type_k,
        bool offload,
        uint32_t n_ctx,
        uint32_t n_seq,
        uint32_t n_ubatch,
        std::unique_ptr<llama_dsv41_engram_runtime> engram) {
    llama_dsv41_memory_config config;
    config.n_ctx = n_ctx;
    config.n_seq = n_seq;
    config.n_ubatch = n_ubatch;
    config.n_layer = model.hparams.n_layer();
    config.raw_window = model.hparams.n_swa;
    config.kv_width = model.hparams.n_embd_head_k();
    config.index_width = model.hparams.indexer_head_size;
    config.candidate_topk_blocks = model.hparams.dsv41_candidate_topk_blocks;
    config.candidate_block_size = model.hparams.dsv41_candidate_block_size;
    config.candidate_source_layer = model.hparams.dsv41_candidate_source_layer;
    config.type_k = type_k;
    config.type_index = type_k;
    config.no_alloc = model.hparams.no_alloc;
    config.attach_no_alloc_buffers = config.no_alloc;
    config.engram_enabled = model.hparams.dsv41_engram_layers.any();
    config.expert_enabled = model.requires_synchronous_graph();
    config.ratios.resize(config.n_layer);
    config.kv_sources.clear();
    config.index_sources.clear();
    for (uint32_t il = 0; il < config.n_layer; ++il) {
        config.ratios[il] = model.hparams.dsv4_compress_ratios[il];
        if (model.hparams.dsv41_is_kv_source(il)) {
            config.kv_sources.push_back(il);
        }
        if (model.hparams.dsv41_is_index_source(il)) {
            config.index_sources.push_back(il);
        }
    }
    config.buft_for_layer = [&model, offload](int32_t il) {
        return offload ? model.select_buft(il) : ggml_backend_cpu_buffer_type();
    };
    config.engram = std::move(engram);
    return config;
}

}

bool llama_dsv41_graph_topology::same_topology(const llama_dsv41_graph_topology & other) const {
    return n_tokens == other.n_tokens &&
        n_seqs == other.n_seqs &&
        n_outputs == other.n_outputs &&
        visible_raw_widths == other.visible_raw_widths &&
        visible_compressed_widths == other.visible_compressed_widths &&
        source_ratios == other.source_ratios &&
        source_carry_counts == other.source_carry_counts &&
        candidate_width == other.candidate_width &&
        backend_layout == other.backend_layout &&
        engram_enabled == other.engram_enabled &&
        expert_enabled == other.expert_enabled;
}

struct llama_memory_dsv41::impl {
    struct sequence_state {
        llama_pos pos = -1;
        std::vector<int32_t> candidates;
    };

    struct tensor_snapshot {
        ggml_tensor * tensor = nullptr;
        size_t offset = 0;
        std::vector<uint8_t> data;
    };

    struct rollback_state {
        llama_pos start_pos = -1;
        sequence_state sequence;
        std::vector<tensor_snapshot> snapshots;
        llama_dsv41_engram_sequence_state engram;
        bool has_engram = false;
    };

    struct source_storage {
        uint32_t layer = 0;
        uint32_t ratio = 0;
        uint32_t capacity = 0;
        ggml_tensor * kv = nullptr;
        ggml_tensor * index = nullptr;
        ggml_tensor * carry_kv = nullptr;
        ggml_tensor * carry_score = nullptr;
    };

    struct buffer_group {
        ggml_backend_buffer_type_t buft = nullptr;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buffer;
    };

    llama_dsv41_memory_config config;
    std::vector<sequence_state> sequences;
    std::vector<ggml_tensor *> raw;
    std::map<uint32_t, source_storage> sources;
    std::vector<buffer_group> groups;
    ggml_tensor * candidate_scores = nullptr;
    ggml_tensor * candidate_ids = nullptr;
    ggml_tensor * committed_candidate_ids = nullptr;
    ggml_tensor * positions = nullptr;
    uint32_t position_rows = 0;
    uint32_t candidate_blocks = 0;
    uint32_t candidate_width = 0;
    uint64_t generation = 0;
    uint64_t backend_layout = 1469598103934665603ULL;
    uint64_t graph_workspace = 0;
    bool transaction_active = false;
    std::map<llama_seq_id, rollback_state> rollback_states;

    explicit impl(llama_dsv41_memory_config config) : config(std::move(config)) {
        this->config.engram_enabled =
            this->config.engram_enabled || this->config.engram != nullptr;
        if (this->config.n_ctx == 0 || this->config.n_seq == 0 || this->config.n_ubatch == 0 ||
                this->config.n_layer == 0 || this->config.raw_window == 0 ||
                this->config.kv_width == 0 || this->config.index_width == 0 ||
                this->config.candidate_block_size == 0 || this->config.candidate_topk_blocks == 0) {
            throw std::invalid_argument("DeepSeek V4.1 memory dimensions must be non-zero");
        }
        if (this->config.ratios.empty()) {
            this->config.ratios.resize(this->config.n_layer);
            for (uint32_t il = 0; il < this->config.n_layer; ++il) {
                this->config.ratios[il] = llama_dsv41_compress_ratio(il);
            }
        }
        if (this->config.ratios.size() != this->config.n_layer) {
            throw std::invalid_argument("DeepSeek V4.1 memory ratio map has the wrong size");
        }
        if (this->config.candidate_source_layer >= this->config.n_layer) {
            throw std::invalid_argument("DeepSeek V4.1 candidate source layer is out of range");
        }
        if (!this->config.buft_for_layer) {
            this->config.buft_for_layer = [](int32_t) { return ggml_backend_cpu_buffer_type(); };
        }
        for (uint32_t source : this->config.kv_sources) {
            if (source >= this->config.n_layer || (this->config.ratios[source] != 1 && this->config.ratios[source] != 2)) {
                throw std::invalid_argument("DeepSeek V4.1 memory KV source is invalid");
            }
        }
        for (uint32_t source : this->config.index_sources) {
            if (source >= this->config.n_layer) {
                throw std::invalid_argument("DeepSeek V4.1 memory index source is invalid");
            }
        }
        if (this->config.engram && this->config.engram->max_tokens() < this->config.n_ubatch) {
            throw std::invalid_argument("DeepSeek V4.1 Engram runtime is smaller than n_ubatch");
        }

        sequences.resize(this->config.n_seq);
        raw.resize(this->config.n_layer);
        candidate_blocks = (this->config.n_ctx + this->config.candidate_block_size - 1)/
            this->config.candidate_block_size;
        candidate_width = std::min(candidate_blocks, this->config.candidate_topk_blocks);

        std::map<ggml_backend_buffer_type_t, size_t> tensor_counts;
        for (uint32_t il = 0; il < this->config.n_layer; ++il) {
            tensor_counts[this->config.buft_for_layer(il)]++;
        }
        for (uint32_t source : this->config.kv_sources) {
            tensor_counts[this->config.buft_for_layer(source)] += this->config.ratios[source] == 2 ? 4 : 2;
        }
        ggml_backend_buffer_type_t candidate_buft =
            this->config.buft_for_layer(this->config.candidate_source_layer);
        tensor_counts[candidate_buft] += 3;
        tensor_counts[ggml_backend_cpu_buffer_type()]++;

        for (const auto & entry : tensor_counts) {
            ggml_init_params params = {
                /*.mem_size   =*/ (entry.second + 4)*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx = ggml_init(params);
            if (ctx == nullptr) {
                throw std::runtime_error("failed to create DeepSeek V4.1 memory tensor context");
            }
            groups.push_back({ entry.first, ggml_context_ptr(ctx), nullptr });
        }

        auto context_for = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
            for (auto & group : groups) {
                if (group.buft == buft) {
                    return group.ctx.get();
                }
            }
            throw std::runtime_error("DeepSeek V4.1 memory buffer type is missing");
        };

        for (uint32_t il = 0; il < this->config.n_layer; ++il) {
            ggml_context * ctx = context_for(this->config.buft_for_layer(il));
            raw[il] = ggml_new_tensor_3d(
                    ctx, this->config.type_k, this->config.kv_width, this->config.raw_window, this->config.n_seq);
            ggml_format_name(raw[il], "dsv41_raw_k_l%u", il);
        }

        for (uint32_t source : this->config.kv_sources) {
            source_storage storage;
            storage.layer = source;
            storage.ratio = this->config.ratios[source];
            storage.capacity = source_capacity(this->config, source);
            ggml_context * ctx = context_for(this->config.buft_for_layer(source));
            storage.kv = ggml_new_tensor_3d(
                    ctx, this->config.type_k, this->config.kv_width, storage.capacity, this->config.n_seq);
            storage.index = ggml_new_tensor_3d(
                    ctx, this->config.type_index, this->config.index_width, storage.capacity, this->config.n_seq);
            ggml_format_name(storage.kv, "dsv41_comp_kv_l%u", source);
            ggml_format_name(storage.index, "dsv41_index_k_l%u", source);
            if (storage.ratio == 2) {
                storage.carry_kv = ggml_new_tensor_3d(
                        ctx, this->config.type_k, this->config.kv_width, storage.ratio, this->config.n_seq);
                storage.carry_score = ggml_new_tensor_3d(
                        ctx, this->config.type_k, this->config.kv_width, storage.ratio, this->config.n_seq);
                ggml_format_name(storage.carry_kv, "dsv41_carry_kv_l%u", source);
                ggml_format_name(storage.carry_score, "dsv41_carry_score_l%u", source);
            }
            sources.emplace(source, storage);
        }

        {
            ggml_context * ctx = context_for(candidate_buft);
            candidate_scores = ggml_new_tensor_2d(
                    ctx, GGML_TYPE_F32, candidate_blocks, this->config.n_ubatch);
            candidate_ids = ggml_new_tensor_2d(
                    ctx, GGML_TYPE_I32, candidate_width, this->config.n_ubatch);
            committed_candidate_ids = ggml_new_tensor_2d(
                    ctx, GGML_TYPE_I32, candidate_width, this->config.n_seq);
            ggml_set_name(candidate_scores, "dsv41_candidate_scores");
            ggml_set_name(candidate_ids, "dsv41_candidate_ids");
            ggml_set_name(committed_candidate_ids, "dsv41_committed_candidate_ids");
        }

        position_rows = 1 + this->config.raw_window;
        for (uint32_t source : this->config.kv_sources) {
            position_rows += source_capacity(this->config, source);
            if (this->config.ratios[source] == 2) {
                position_rows += 2;
            }
        }
        {
            ggml_context * ctx = context_for(ggml_backend_cpu_buffer_type());
            positions = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, position_rows, this->config.n_seq);
            ggml_set_name(positions, "dsv41_position_state");
        }

        for (auto & group : groups) {
            const size_t buffer_size =
                ggml_backend_alloc_ctx_tensors_from_buft_size(group.ctx.get(), group.buft);
            ggml_backend_buffer_t buffer = nullptr;
            if (this->config.no_alloc && this->config.attach_no_alloc_buffers) {
                buffer = ggml_backend_buft_alloc_buffer(group.buft, 0);
                for (ggml_tensor * tensor = ggml_get_first_tensor(group.ctx.get());
                        tensor != nullptr;
                        tensor = ggml_get_next_tensor(group.ctx.get(), tensor)) {
                    tensor->buffer = buffer;
                }
            } else if (!this->config.no_alloc) {
                buffer =
                    ggml_backend_alloc_ctx_tensors_from_buft(group.ctx.get(), group.buft);
            }
            if (buffer == nullptr && (!this->config.no_alloc || this->config.attach_no_alloc_buffers)) {
                throw std::runtime_error("failed to allocate DeepSeek V4.1 memory buffer");
            }
            if (buffer != nullptr) {
                group.buffer.reset(buffer);
            }
            if (buffer != nullptr && !this->config.no_alloc) {
                ggml_backend_buffer_clear(buffer, 0);
            }
            backend_layout = hash_string(backend_layout, ggml_backend_buft_name(group.buft));
            backend_layout = hash_mix(backend_layout, buffer_size);
        }
        for (ggml_tensor * tensor : all_tensors()) {
            backend_layout = hash_mix(backend_layout, tensor->type);
            backend_layout = hash_mix(backend_layout, tensor->ne[0]);
            backend_layout = hash_mix(backend_layout, tensor->ne[1]);
            backend_layout = hash_mix(backend_layout, tensor->ne[2]);
        }
        for (uint32_t seq = 0; seq < this->config.n_seq; ++seq) {
            update_position_state(seq);
        }
    }

    std::vector<ggml_tensor *> all_tensors() const {
        std::vector<ggml_tensor *> result;
        result.insert(result.end(), raw.begin(), raw.end());
        for (const auto & entry : sources) {
            result.push_back(entry.second.kv);
            result.push_back(entry.second.index);
            if (entry.second.carry_kv != nullptr) {
                result.push_back(entry.second.carry_kv);
                result.push_back(entry.second.carry_score);
            }
        }
        result.push_back(candidate_scores);
        result.push_back(candidate_ids);
        result.push_back(committed_candidate_ids);
        result.push_back(positions);
        return result;
    }

    std::vector<ggml_tensor *> sequence_tensors() const {
        std::vector<ggml_tensor *> result;
        result.insert(result.end(), raw.begin(), raw.end());
        for (const auto & entry : sources) {
            result.push_back(entry.second.kv);
            result.push_back(entry.second.index);
            if (entry.second.carry_kv != nullptr) {
                result.push_back(entry.second.carry_kv);
                result.push_back(entry.second.carry_score);
            }
        }
        result.push_back(committed_candidate_ids);
        result.push_back(positions);
        return result;
    }

    size_t sequence_plane(const ggml_tensor * tensor) const {
        if (tensor->ne[2] == (int64_t) config.n_seq) {
            return tensor->nb[2];
        }
        if (tensor->ne[1] == (int64_t) config.n_seq) {
            return tensor->nb[1];
        }
        throw std::runtime_error("DeepSeek V4.1 tensor has no sequence plane");
    }

    bool valid_seq(llama_seq_id seq_id) const {
        return seq_id >= 0 && (uint32_t) seq_id < config.n_seq;
    }

    void require_idle() const {
        if (transaction_active) {
            throw std::runtime_error("DeepSeek V4.1 memory operation is not allowed during a transaction");
        }
    }

    void update_position_state(llama_seq_id seq_id) {
        const std::vector<int32_t> values = position_state_values(sequences[seq_id]);
        if (!config.no_alloc) {
            ggml_backend_tensor_set(
                    positions, values.data(), (size_t) seq_id*positions->nb[1], values.size()*sizeof(int32_t));
        }
    }

    std::vector<int32_t> position_state_values(const sequence_state & sequence) const {
        std::vector<int32_t> values(position_rows, -1);
        const llama_pos pos = sequence.pos;
        values[0] = pos;
        if (pos >= 0) {
            const llama_pos first_raw = std::max<llama_pos>(0, pos + 1 - config.raw_window);
            for (llama_pos current = first_raw; current <= pos; ++current) {
                values[1 + current%config.raw_window] = current;
            }
        }
        uint32_t offset = 1 + config.raw_window;
        for (uint32_t source : config.kv_sources) {
            const uint32_t ratio = config.ratios[source];
            const uint32_t capacity = source_capacity(config, source);
            const uint32_t visible = pos < 0 ? 0 : (uint32_t) (pos + 1)/ratio;
            for (uint32_t row = 0; row < visible; ++row) {
                values[offset + row] = row*ratio;
            }
            offset += capacity;
            if (ratio == 2) {
                for (uint32_t row = 0; row < 2; ++row) {
                    if (pos >= (llama_pos) row) {
                        values[offset + row] = pos - ((pos - row)%2);
                    }
                }
                offset += 2;
            }
        }
        return values;
    }

    void update_candidate_state(llama_seq_id seq_id) {
        const std::vector<int32_t> values = candidate_state_values(sequences[seq_id]);
        if (!config.no_alloc) {
            ggml_backend_tensor_set(
                    committed_candidate_ids,
                    values.data(),
                    (size_t) seq_id*committed_candidate_ids->nb[1],
                    values.size()*sizeof(int32_t));
        }
    }

    std::vector<int32_t> candidate_state_values(const sequence_state & sequence) const {
        std::vector<int32_t> values(candidate_width);
        const auto & candidates = sequence.candidates;
        std::copy_n(candidates.begin(), std::min(candidates.size(), values.size()), values.begin());
        return values;
    }

    void clear_sequence_data(llama_seq_id seq_id) {
        if (config.no_alloc) {
            return;
        }
        for (ggml_tensor * tensor : sequence_tensors()) {
            const size_t plane = sequence_plane(tensor);
            std::vector<uint8_t> zeros(plane);
            ggml_backend_tensor_set(tensor, zeros.data(), (size_t) seq_id*plane, plane);
        }
    }

    void copy_sequence_data(llama_seq_id src, llama_seq_id dst) {
        if (config.no_alloc || src == dst) {
            return;
        }
        for (ggml_tensor * tensor : sequence_tensors()) {
            const size_t plane = sequence_plane(tensor);
            std::vector<uint8_t> data(plane);
            ggml_backend_tensor_get(tensor, data.data(), (size_t) src*plane, plane);
            ggml_backend_tensor_set(tensor, data.data(), (size_t) dst*plane, plane);
        }
    }
};

struct llama_memory_dsv41_context::transaction_state {
    llama_dsv41_memory_plan plan;
    std::vector<llama_memory_dsv41::impl::sequence_state> next_sequences;
    std::vector<llama_memory_dsv41::impl::tensor_snapshot> snapshots;
    std::unique_ptr<llama_dsv41_engram_transaction> engram;
    std::map<llama_seq_id, llama_dsv41_engram_sequence_state> engram_before;
    std::vector<llama_seq_id> seq_ids;
    std::vector<llama_pos> start_positions;
};

llama_memory_dsv41::llama_memory_dsv41(llama_dsv41_memory_config config) :
    pimpl(std::make_unique<impl>(std::move(config))) {
}

llama_memory_dsv41::llama_memory_dsv41(
        const llama_model & model,
        ggml_type type_k,
        bool offload,
        uint32_t n_ctx,
        uint32_t n_seq,
        uint32_t n_ubatch,
        std::unique_ptr<llama_dsv41_engram_runtime> engram) :
    llama_memory_dsv41(make_model_config(
            model, type_k, offload, n_ctx, n_seq, n_ubatch, std::move(engram))) {
}

llama_memory_dsv41::~llama_memory_dsv41() = default;

llama_memory_context_ptr llama_memory_dsv41::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    GGML_UNUSED(embd_all);
    if (n_ubatch == 0 || n_ubatch > pimpl->config.n_ubatch || pimpl->transaction_active) {
        return std::make_unique<llama_memory_dsv41_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    balloc.split_reset();
    std::vector<llama_ubatch> ubatches;
    while (true) {
        llama_ubatch ubatch = balloc.split_seq(n_ubatch);
        if (ubatch.n_tokens == 0) {
            break;
        }
        ubatches.push_back(std::move(ubatch));
    }
    if (balloc.get_n_used() != balloc.get_n_tokens() || ubatches.empty()) {
        return std::make_unique<llama_memory_dsv41_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }
    return std::make_unique<llama_memory_dsv41_context>(this, std::move(ubatches));
}

llama_memory_context_ptr llama_memory_dsv41::init_full() {
    return std::make_unique<llama_memory_dsv41_context>(this, true);
}

llama_memory_context_ptr llama_memory_dsv41::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);
    return std::make_unique<llama_memory_dsv41_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_memory_dsv41::get_can_shift() const {
    return false;
}

void llama_memory_dsv41::clear(bool data) {
    pimpl->require_idle();
    for (auto & sequence : pimpl->sequences) {
        sequence = {};
    }
    pimpl->rollback_states.clear();
    if (pimpl->config.engram) {
        for (uint32_t seq = 0; seq < pimpl->config.n_seq; ++seq) {
            pimpl->config.engram->seq_remove(seq);
        }
    }
    if (data && !pimpl->config.no_alloc) {
        for (auto & group : pimpl->groups) {
            ggml_backend_buffer_clear(group.buffer.get(), 0);
        }
    }
    for (uint32_t seq = 0; seq < pimpl->config.n_seq; ++seq) {
        pimpl->update_position_state(seq);
        pimpl->update_candidate_state(seq);
    }
    ++pimpl->generation;
}

bool llama_memory_dsv41::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    try {
        pimpl->require_idle();
    } catch (const std::exception & error) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, error.what());
        return false;
    }
    if (seq_id < 0) {
        if (p0 == -1 && p1 == -1) {
            clear(true);
            return true;
        }
        LLAMA_LOG_ERROR("%s: DeepSeek V4.1 only supports wildcard removal for the full memory\n", __func__);
        return false;
    }
    if (!pimpl->valid_seq(seq_id)) {
        LLAMA_LOG_ERROR("%s: DeepSeek V4.1 sequence ID %d is out of range\n", __func__, seq_id);
        return false;
    }

    const llama_pos max_pos = pimpl->sequences[seq_id].pos;
    const llama_pos begin = p0 < 0 ? 0 : p0;
    const llama_pos end = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
    if (begin >= end || begin > max_pos) {
        return true;
    }
    if (end <= max_pos) {
        LLAMA_LOG_ERROR("%s: DeepSeek V4.1 only supports full removal or suffix rollback\n", __func__);
        return false;
    }

    auto rollback = pimpl->rollback_states.end();
    if (begin > 0) {
        rollback = pimpl->rollback_states.find(seq_id);
        if (rollback == pimpl->rollback_states.end() || rollback->second.start_pos != begin) {
            LLAMA_LOG_ERROR("%s: DeepSeek V4.1 only supports rollback of the most recent committed ubatch\n", __func__);
            return false;
        }
    }

    if (begin == 0) {
        pimpl->sequences[seq_id] = {};
        pimpl->clear_sequence_data(seq_id);
        pimpl->update_position_state(seq_id);
        pimpl->update_candidate_state(seq_id);
        if (pimpl->config.engram) {
            pimpl->config.engram->seq_remove(seq_id);
        }
        pimpl->rollback_states.erase(seq_id);
    } else {
        for (const auto & snapshot : rollback->second.snapshots) {
            ggml_backend_tensor_set(
                    snapshot.tensor, snapshot.data.data(),
                    snapshot.offset, snapshot.data.size());
        }
        pimpl->sequences[seq_id] = rollback->second.sequence;
        if (rollback->second.has_engram) {
            llama_dsv41_engram_snapshot snapshot = pimpl->config.engram->checkpoint();
            snapshot.sequences[seq_id] = rollback->second.engram;
            pimpl->config.engram->restore(snapshot);
        }
        pimpl->update_position_state(seq_id);
        pimpl->update_candidate_state(seq_id);
        pimpl->rollback_states.erase(rollback);
    }
    ++pimpl->generation;
    return true;
}

void llama_memory_dsv41::seq_cp(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos p0,
        llama_pos p1) {
    pimpl->require_idle();
    if (!pimpl->valid_seq(seq_id_src) || !pimpl->valid_seq(seq_id_dst)) {
        throw std::invalid_argument("DeepSeek V4.1 sequence copy ID is out of range");
    }
    const llama_pos max_pos = pimpl->sequences[seq_id_src].pos;
    if ((p0 > 0) || (p1 >= 0 && p1 <= max_pos)) {
        throw std::invalid_argument("DeepSeek V4.1 only supports full sequence copy");
    }
    pimpl->copy_sequence_data(seq_id_src, seq_id_dst);
    pimpl->sequences[seq_id_dst] = pimpl->sequences[seq_id_src];
    pimpl->rollback_states.erase(seq_id_dst);
    if (pimpl->config.engram) {
        pimpl->config.engram->seq_copy(seq_id_src, seq_id_dst);
    }
    ++pimpl->generation;
}

void llama_memory_dsv41::seq_keep(llama_seq_id seq_id) {
    pimpl->require_idle();
    if (!pimpl->valid_seq(seq_id)) {
        throw std::invalid_argument("DeepSeek V4.1 sequence keep ID is out of range");
    }
    for (uint32_t current = 0; current < pimpl->config.n_seq; ++current) {
        if ((llama_seq_id) current == seq_id) {
            continue;
        }
        pimpl->sequences[current] = {};
        pimpl->clear_sequence_data(current);
        pimpl->update_position_state(current);
        pimpl->update_candidate_state(current);
        if (pimpl->config.engram) {
            pimpl->config.engram->seq_remove(current);
        }
        pimpl->rollback_states.erase(current);
    }
    ++pimpl->generation;
}

[[noreturn]] void llama_memory_dsv41::seq_add(
        llama_seq_id,
        llama_pos,
        llama_pos,
        llama_pos) {
    throw std::invalid_argument("DeepSeek V4.1 memory does not support position shifts");
}

[[noreturn]] void llama_memory_dsv41::seq_div(
        llama_seq_id,
        llama_pos,
        llama_pos,
        int) {
    throw std::invalid_argument("DeepSeek V4.1 memory does not support position division");
}

llama_pos llama_memory_dsv41::seq_pos_min(llama_seq_id seq_id) const {
    if (!pimpl->valid_seq(seq_id)) {
        return -1;
    }
    return pimpl->sequences[seq_id].pos < 0 ? -1 : 0;
}

llama_pos llama_memory_dsv41::seq_pos_max(llama_seq_id seq_id) const {
    if (!pimpl->valid_seq(seq_id)) {
        return -1;
    }
    return pimpl->sequences[seq_id].pos;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_dsv41::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    for (const auto & group : pimpl->groups) {
        const size_t size = pimpl->config.no_alloc ?
            ggml_backend_alloc_ctx_tensors_from_buft_size(group.ctx.get(), group.buft) :
            ggml_backend_buffer_get_size(group.buffer.get());
        result[group.buft] += size;
    }
    return result;
}

void llama_memory_dsv41::set_graph_workspace_size(size_t size) {
    pimpl->graph_workspace = size;
}

void llama_memory_dsv41::state_write(
        llama_io_write_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);
    pimpl->require_idle();
    if (seq_id != -1 && !pimpl->valid_seq(seq_id)) {
        throw std::invalid_argument("DeepSeek V4.1 state sequence ID is out of range");
    }

    io.write(&DSV41_STATE_MAGIC, sizeof(DSV41_STATE_MAGIC));
    io.write(&DSV41_STATE_VERSION, sizeof(DSV41_STATE_VERSION));
    const uint32_t count = seq_id == -1 ? pimpl->config.n_seq : 1;
    io.write(&count, sizeof(count));
    const auto tensors = pimpl->sequence_tensors();
    const uint32_t tensor_count = tensors.size();
    io.write(&tensor_count, sizeof(tensor_count));

    const llama_dsv41_engram_snapshot engram_snapshot =
        pimpl->config.engram ? pimpl->config.engram->checkpoint() : llama_dsv41_engram_snapshot {};
    for (uint32_t i = 0; i < count; ++i) {
        const llama_seq_id current = seq_id == -1 ? (llama_seq_id) i : seq_id;
        io.write(&current, sizeof(current));
        const auto & state = pimpl->sequences[current];
        io.write(&state.pos, sizeof(state.pos));
        const uint32_t n_candidates = state.candidates.size();
        io.write(&n_candidates, sizeof(n_candidates));
        if (n_candidates > 0) {
            io.write(state.candidates.data(), n_candidates*sizeof(int32_t));
        }
        const uint8_t has_engram = pimpl->config.engram ? 1 : 0;
        io.write(&has_engram, sizeof(has_engram));
        if (has_engram) {
            const auto found = engram_snapshot.sequences.find(current);
            llama_dsv41_engram_sequence_state engram_state;
            engram_state.history.reset();
            if (found != engram_snapshot.sequences.end()) {
                engram_state = found->second;
            }
            io.write(&engram_state.pos, sizeof(engram_state.pos));
            io.write(engram_state.history.tail.data(), sizeof(engram_state.history.tail));
        }
        for (ggml_tensor * tensor : tensors) {
            const uint64_t plane = pimpl->sequence_plane(tensor);
            io.write(&plane, sizeof(plane));
            io.write_tensor(tensor, (size_t) current*plane, plane);
        }
    }
}

void llama_memory_dsv41::state_read(
        llama_io_read_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    GGML_UNUSED(flags);
    pimpl->require_idle();
    if (seq_id != -1 && !pimpl->valid_seq(seq_id)) {
        throw std::invalid_argument("DeepSeek V4.1 state destination sequence ID is out of range");
    }

    uint64_t magic = 0;
    uint32_t version = 0;
    uint32_t count = 0;
    uint32_t tensor_count = 0;
    io.read(&magic, sizeof(magic));
    io.read(&version, sizeof(version));
    io.read(&count, sizeof(count));
    io.read(&tensor_count, sizeof(tensor_count));
    const auto tensors = pimpl->sequence_tensors();
    if (magic != DSV41_STATE_MAGIC || version != DSV41_STATE_VERSION ||
            tensor_count != tensors.size() || count == 0 ||
            (seq_id != -1 && count != 1) || (seq_id == -1 && count != pimpl->config.n_seq)) {
        throw std::runtime_error("DeepSeek V4.1 state header is incompatible");
    }

    const auto sequence_is_empty = [&](llama_seq_id current) {
        if (pimpl->sequences[current].pos >= 0 ||
                !pimpl->sequences[current].candidates.empty() ||
                pimpl->rollback_states.count(current) != 0) {
            return false;
        }
        return !pimpl->config.engram || pimpl->config.engram->sequence(current).pos < 0;
    };
    if (seq_id == -1) {
        for (uint32_t current = 0; current < pimpl->config.n_seq; ++current) {
            if (!sequence_is_empty(current)) {
                throw std::runtime_error("DeepSeek V4.1 full state restore requires empty memory");
            }
        }
    } else if (!sequence_is_empty(seq_id)) {
        throw std::runtime_error("DeepSeek V4.1 sequence state restore requires an empty destination");
    }

    llama_dsv41_engram_snapshot engram_snapshot =
        pimpl->config.engram && seq_id != -1 ?
            pimpl->config.engram->checkpoint() : llama_dsv41_engram_snapshot {};
    std::set<llama_seq_id> restored;
    std::vector<uint8_t> tensor_data;
    struct staged_sequence {
        llama_seq_id target = -1;
        llama_pos pos = -1;
        std::vector<int32_t> candidates;
        std::vector<uint64_t> planes;
    };
    std::vector<staged_sequence> staged;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        staged.reserve(count);
    }

    try {
        for (uint32_t i = 0; i < count; ++i) {
            llama_seq_id stored = -1;
            llama_pos pos = -1;
            uint32_t n_candidates = 0;
            uint8_t has_engram = 0;
            io.read(&stored, sizeof(stored));
            io.read(&pos, sizeof(pos));
            io.read(&n_candidates, sizeof(n_candidates));
            const llama_seq_id target = seq_id == -1 ? stored : seq_id;
            if (!pimpl->valid_seq(stored) || !pimpl->valid_seq(target) ||
                    (seq_id == -1 && !restored.insert(target).second) ||
                    pos < -1 || pos >= (llama_pos) pimpl->config.n_ctx ||
                    n_candidates > pimpl->candidate_width) {
                throw std::runtime_error("DeepSeek V4.1 state metadata is invalid");
            }
            std::vector<int32_t> candidates(n_candidates);
            if (n_candidates > 0) {
                io.read(candidates.data(), n_candidates*sizeof(int32_t));
            }
            io.read(&has_engram, sizeof(has_engram));
            if ((has_engram != 0) != (pimpl->config.engram != nullptr)) {
                throw std::runtime_error("DeepSeek V4.1 Engram state availability differs");
            }
            if (has_engram) {
                llama_dsv41_engram_sequence_state engram_state;
                io.read(&engram_state.pos, sizeof(engram_state.pos));
                io.read(engram_state.history.tail.data(), sizeof(engram_state.history.tail));
                engram_snapshot.sequences[target] = engram_state;
            }
            std::vector<uint64_t> planes;
            if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
                planes.reserve(tensors.size());
            }
            for (ggml_tensor * tensor : tensors) {
                uint64_t plane = 0;
                io.read(&plane, sizeof(plane));
                if (plane != pimpl->sequence_plane(tensor)) {
                    throw std::runtime_error("DeepSeek V4.1 state tensor layout differs");
                }
                if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
                    planes.push_back(plane);
                } else {
                    tensor_data.resize(plane);
                    io.read(tensor_data.data(), plane);
                    ggml_backend_tensor_set(tensor, tensor_data.data(), (size_t) target*plane, plane);
                }
            }
            if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
                staged.push_back({ target, pos, std::move(candidates), std::move(planes) });
            } else {
                pimpl->sequences[target].pos = pos;
                pimpl->sequences[target].candidates = std::move(candidates);
                pimpl->update_position_state(target);
                pimpl->update_candidate_state(target);
            }
        }
        if (pimpl->config.engram) {
            pimpl->config.engram->restore(engram_snapshot);
        }
        for (auto & sequence : staged) {
            for (size_t i = 0; i < tensors.size(); ++i) {
                io.read_tensor(
                        tensors[i],
                        (size_t) sequence.target*sequence.planes[i],
                        sequence.planes[i]);
            }
            pimpl->sequences[sequence.target].pos = sequence.pos;
            pimpl->sequences[sequence.target].candidates = std::move(sequence.candidates);
            pimpl->update_position_state(sequence.target);
            pimpl->update_candidate_state(sequence.target);
        }
        if (seq_id == -1) {
            pimpl->rollback_states.clear();
        } else {
            pimpl->rollback_states.erase(seq_id);
        }
        ++pimpl->generation;
    } catch (...) {
        if (seq_id == -1) {
            clear(true);
        } else {
            seq_rm(seq_id, -1, -1);
        }
        throw;
    }
}

ggml_tensor * llama_memory_dsv41::raw_k(uint32_t layer) const {
    return layer < pimpl->raw.size() ? pimpl->raw[layer] : nullptr;
}

ggml_tensor * llama_memory_dsv41::compressed_kv(uint32_t source_layer) const {
    const auto found = pimpl->sources.find(source_layer);
    return found == pimpl->sources.end() ? nullptr : found->second.kv;
}

ggml_tensor * llama_memory_dsv41::index_keys(uint32_t source_layer) const {
    const auto found = pimpl->sources.find(source_layer);
    return found == pimpl->sources.end() ? nullptr : found->second.index;
}

ggml_tensor * llama_memory_dsv41::compressor_carry_kv(uint32_t source_layer) const {
    const auto found = pimpl->sources.find(source_layer);
    return found == pimpl->sources.end() ? nullptr : found->second.carry_kv;
}

ggml_tensor * llama_memory_dsv41::compressor_carry_score(uint32_t source_layer) const {
    const auto found = pimpl->sources.find(source_layer);
    return found == pimpl->sources.end() ? nullptr : found->second.carry_score;
}

ggml_tensor * llama_memory_dsv41::candidate_scores() const {
    return pimpl->candidate_scores;
}

ggml_tensor * llama_memory_dsv41::candidate_ids() const {
    return pimpl->candidate_ids;
}

ggml_tensor * llama_memory_dsv41::committed_candidate_ids() const {
    return pimpl->committed_candidate_ids;
}

ggml_tensor * llama_memory_dsv41::position_state() const {
    return pimpl->positions;
}

const llama_dsv41_memory_config & llama_memory_dsv41::config() const {
    return pimpl->config;
}

llama_dsv41_memory_accounting llama_memory_dsv41::accounting() const {
    llama_dsv41_memory_accounting result;
    for (ggml_tensor * tensor : pimpl->raw) {
        result.raw_kv += ggml_nbytes(tensor);
    }
    for (const auto & entry : pimpl->sources) {
        result.compressed_kv += ggml_nbytes(entry.second.kv);
        result.index_keys += ggml_nbytes(entry.second.index);
        if (entry.second.carry_kv != nullptr) {
            result.compressor_carry += ggml_nbytes(entry.second.carry_kv);
            result.compressor_carry += ggml_nbytes(entry.second.carry_score);
        }
    }
    result.candidate_scores = ggml_nbytes(pimpl->candidate_scores);
    result.candidate_ids = ggml_nbytes(pimpl->candidate_ids) + ggml_nbytes(pimpl->committed_candidate_ids);
    result.position_state = ggml_nbytes(pimpl->positions);
    result.graph_workspace = pimpl->graph_workspace;
    return result;
}

std::vector<int32_t> llama_memory_dsv41::sequence_candidate_ids(llama_seq_id seq_id) const {
    if (!pimpl->valid_seq(seq_id)) {
        throw std::invalid_argument("DeepSeek V4.1 candidate sequence ID is out of range");
    }
    return pimpl->sequences[seq_id].candidates;
}

size_t llama_memory_dsv41::retained_rollback_count() const {
    return pimpl->rollback_states.size();
}

bool llama_memory_dsv41::engram_enabled() const {
    return pimpl->config.engram_enabled;
}

llama_memory_dsv41_context::llama_memory_dsv41_context(llama_memory_status status) :
    status(status) {
}

llama_memory_dsv41_context::llama_memory_dsv41_context(
        llama_memory_dsv41 * memory,
        bool full) :
    status(LLAMA_MEMORY_STATUS_SUCCESS),
    mem(memory),
    full(full) {
    if (!memory || !full) {
        return;
    }

    llama_batch_allocr allocator(1);
    llama_ubatch ubatch = allocator.ubatch_reserve(memory->pimpl->config.n_ubatch, 1);
    const uint32_t n_tokens = ubatch.n_tokens;
    const uint32_t n_streams = 1;
    ubatch.data->seq_id_data.resize((size_t) n_tokens*n_streams);
    ubatch.data->seq_id_unq.resize(n_streams);
    ubatch.seq_id_unq = ubatch.data->seq_id_unq.data();
    ubatch.n_seqs_unq = n_streams;
    for (uint32_t seq = 0; seq < n_streams; ++seq) {
        ubatch.data->seq_id_unq[seq] = seq;
        ubatch.seq_idx[seq] = seq;
    }
    for (uint32_t token = 0; token < n_tokens; ++token) {
        ubatch.token[token] = 0;
        ubatch.pos[token] = token;
        ubatch.n_seq_id[token] = n_streams;
        ubatch.seq_id[token] = ubatch.data->seq_id_data.data() + (size_t) token*n_streams;
        for (uint32_t seq = 0; seq < n_streams; ++seq) {
            ubatch.seq_id[token][seq] = seq;
        }
    }
    ubatches.push_back(std::move(ubatch));

    transaction = std::make_unique<transaction_state>();
    transaction->plan.generation = memory->pimpl->generation;
    transaction->next_sequences = memory->pimpl->sequences;
    for (uint32_t seq = 0; seq < n_streams; ++seq) {
        transaction->seq_ids.push_back(seq);
        transaction->start_positions.push_back(0);
        transaction->next_sequences[seq].pos = n_tokens - 1;
    }
    const uint32_t raw_width = memory->pimpl->config.raw_window;
    const uint32_t persist_first = n_tokens > raw_width ? n_tokens - raw_width : 0;
    for (uint32_t token = 0; token < n_tokens; ++token) {
        std::vector<llama_seq_id> token_sequences(n_streams);
        for (uint32_t seq = 0; seq < n_streams; ++seq) {
            token_sequences[seq] = seq;
            if (token >= persist_first) {
                transaction->plan.raw.persist_src_idxs.push_back(token);
                transaction->plan.raw.write_idxs.push_back(
                        (int64_t) seq*raw_width + token%raw_width);
            }
        }
        transaction->plan.token_seq_ids.push_back(std::move(token_sequences));
        transaction->plan.positions.push_back(token);
        const uint32_t visible = std::min(raw_width, token + 1);
        transaction->plan.raw.n_visible.push_back(visible);
        const auto order = llama_dsv41_raw_ring_order(token, raw_width);
        for (uint32_t row = 0; row < raw_width; ++row) {
            transaction->plan.raw.read_idxs.push_back(row < visible ? order[row] : -1);
            transaction->plan.raw.mask.push_back(
                    row < visible ? 0.0f : -std::numeric_limits<float>::infinity());
        }
    }
    for (uint32_t source : memory->pimpl->config.kv_sources) {
        llama_dsv41_source_plan source_plan;
        source_plan.source_layer = source;
        source_plan.ratio = memory->pimpl->config.ratios[source];
        source_plan.capacity = source_capacity(memory->pimpl->config, source);
        source_plan.compression = llama_dsv41_build_compression_plan(
                transaction->plan.positions, source_plan.ratio, source_plan.capacity);
        const uint32_t max_visible = source_plan.compression.n_visible.empty() ? 0 :
            *std::max_element(
                    source_plan.compression.n_visible.begin(),
                    source_plan.compression.n_visible.end());
        for (int32_t visible : source_plan.compression.n_visible) {
            for (uint32_t row = 0; row < max_visible; ++row) {
                source_plan.read_idxs.push_back(row < (uint32_t) visible ? row : -1);
            }
        }
        transaction->plan.sources.push_back(std::move(source_plan));
    }
    const uint32_t visible_blocks =
        (n_tokens + memory->pimpl->config.candidate_block_size - 1)/
        memory->pimpl->config.candidate_block_size;
    transaction->plan.candidate_width = std::min(
            visible_blocks, memory->pimpl->config.candidate_topk_blocks);
}

llama_memory_dsv41_context::llama_memory_dsv41_context(
        llama_memory_dsv41 * memory,
        std::vector<llama_ubatch> ubatches) :
    status(LLAMA_MEMORY_STATUS_SUCCESS),
    mem(memory),
    ubatches(std::move(ubatches)) {
}

llama_memory_dsv41_context::~llama_memory_dsv41_context() {
    rollback();
}

bool llama_memory_dsv41_context::next() {
    if (status != LLAMA_MEMORY_STATUS_SUCCESS || transaction) {
        return false;
    }
    if (++i_next >= ubatches.size()) {
        return false;
    }
    return true;
}

bool llama_memory_dsv41_context::apply() {
    if (status == LLAMA_MEMORY_STATUS_NO_UPDATE) {
        return true;
    }
    if (status != LLAMA_MEMORY_STATUS_SUCCESS || full || mem == nullptr || transaction ||
            i_next >= ubatches.size() || mem->pimpl->transaction_active) {
        status = LLAMA_MEMORY_STATUS_FAILED_PREPARE;
        return false;
    }

    const llama_ubatch & ubatch = ubatches[i_next];
    auto next = std::make_unique<transaction_state>();
    try {
        if (ubatch.n_tokens == 0 || ubatch.n_tokens > mem->pimpl->config.n_ubatch ||
                ubatch.n_pos != 1 || ubatch.pos == nullptr || ubatch.n_seq_id == nullptr ||
                ubatch.seq_id == nullptr) {
            throw std::invalid_argument("DeepSeek V4.1 ubatch shape is invalid");
        }

        next->next_sequences = mem->pimpl->sequences;
        std::vector<llama_seq_id> topology;
        for (uint32_t token = 0; token < ubatch.n_tokens; ++token) {
            if (ubatch.n_seq_id[token] <= 0 || ubatch.seq_id[token] == nullptr) {
                throw std::invalid_argument("DeepSeek V4.1 token has no sequence");
            }
            std::vector<llama_seq_id> token_sequences(
                    ubatch.seq_id[token], ubatch.seq_id[token] + ubatch.n_seq_id[token]);
            std::sort(token_sequences.begin(), token_sequences.end());
            if (std::adjacent_find(token_sequences.begin(), token_sequences.end()) != token_sequences.end()) {
                throw std::invalid_argument("DeepSeek V4.1 token repeats a sequence ID");
            }
            if (token == 0) {
                topology = token_sequences;
            } else if (topology != token_sequences) {
                throw std::invalid_argument("DeepSeek V4.1 ubatch changes sequence topology");
            }
            next->plan.token_seq_ids.push_back(token_sequences);
            next->plan.positions.push_back(ubatch.pos[token]);
        }
        if (topology.empty()) {
            throw std::invalid_argument("DeepSeek V4.1 ubatch has no sequence topology");
        }
        if (topology.size() != 1 || ubatch.n_seqs != 1) {
            throw std::invalid_argument("DeepSeek V4.1 memory supports one sequence per ubatch");
        }
        next->seq_ids = topology;
        for (llama_seq_id seq_id : topology) {
            if (!mem->pimpl->valid_seq(seq_id)) {
                throw std::invalid_argument("DeepSeek V4.1 sequence ID is out of range");
            }
            next->start_positions.push_back(mem->pimpl->sequences[seq_id].pos + 1);
        }
        if (!std::all_of(next->start_positions.begin(), next->start_positions.end(),
                [&](llama_pos pos) { return pos == next->start_positions.front(); })) {
            throw std::invalid_argument("DeepSeek V4.1 coupled sequences have different positions");
        }
        for (uint32_t token = 0; token < ubatch.n_tokens; ++token) {
            const llama_pos expected = next->start_positions.front() + token;
            if (ubatch.pos[token] != expected || expected < 0 || expected >= (llama_pos) mem->pimpl->config.n_ctx) {
                throw std::invalid_argument("DeepSeek V4.1 positions must be contiguous and within n_ctx");
            }
        }

        next->plan.generation = mem->pimpl->generation + 1;
        const uint32_t raw_width = mem->pimpl->config.raw_window;
        const uint32_t persist_first =
            ubatch.n_tokens > raw_width ? ubatch.n_tokens - raw_width : 0;
        next->plan.raw.n_visible.resize(ubatch.n_tokens);
        for (uint32_t token = 0; token < ubatch.n_tokens; ++token) {
            const llama_pos pos = ubatch.pos[token];
            for (llama_seq_id seq_id : topology) {
                if (token >= persist_first) {
                    next->plan.raw.persist_src_idxs.push_back(token);
                    next->plan.raw.write_idxs.push_back(
                            (int64_t) seq_id*raw_width + pos%raw_width);
                }
                next->next_sequences[seq_id].pos = pos;
            }
            const uint32_t visible = std::min<uint32_t>(raw_width, pos + 1);
            next->plan.raw.n_visible[token] = visible;
            const auto order = llama_dsv41_raw_ring_order(pos, raw_width);
            for (uint32_t row = 0; row < raw_width; ++row) {
                next->plan.raw.read_idxs.push_back(
                        row < visible ? (int32_t) ((uint32_t) topology.front()*raw_width + order[row]) : -1);
                next->plan.raw.mask.push_back(row < visible ? 0.0f : -std::numeric_limits<float>::infinity());
            }
        }

        for (uint32_t source : mem->pimpl->config.kv_sources) {
            llama_dsv41_source_plan source_plan;
            source_plan.source_layer = source;
            source_plan.ratio = mem->pimpl->config.ratios[source];
            source_plan.capacity = source_capacity(mem->pimpl->config, source);
            source_plan.compression = llama_dsv41_build_compression_plan(
                    next->plan.positions, source_plan.ratio, source_plan.capacity);
            const uint32_t max_visible = source_plan.compression.n_visible.empty() ? 0 :
                *std::max_element(source_plan.compression.n_visible.begin(), source_plan.compression.n_visible.end());
            for (int32_t visible : source_plan.compression.n_visible) {
                for (uint32_t row = 0; row < max_visible; ++row) {
                    source_plan.read_idxs.push_back(
                            row < (uint32_t) visible ?
                            (int32_t) ((uint32_t) topology.front()*source_plan.capacity + row) : -1);
                }
            }
            next->plan.sources.push_back(std::move(source_plan));
        }
        const llama_pos final_pos = next->plan.positions.back();
        const uint32_t candidate_visible = final_pos + 1;
        const uint32_t visible_blocks =
            (candidate_visible + mem->pimpl->config.candidate_block_size - 1)/
            mem->pimpl->config.candidate_block_size;
        next->plan.candidate_width = std::min(
                visible_blocks, mem->pimpl->config.candidate_topk_blocks);

        if (mem->pimpl->config.engram) {
            if (ubatch.token == nullptr) {
                throw std::invalid_argument("DeepSeek V4.1 Engram requires token inputs");
            }
            std::vector<llama_dsv41_engram_token> tokens;
            tokens.reserve(ubatch.n_tokens);
            for (uint32_t token = 0; token < ubatch.n_tokens; ++token) {
                tokens.push_back({
                    ubatch.token[token],
                    ubatch.pos[token],
                    next->plan.token_seq_ids[token],
                    1,
                });
            }
            for (llama_seq_id seq_id : topology) {
                next->engram_before[seq_id] = mem->pimpl->config.engram->sequence(seq_id);
            }
            next->engram = std::make_unique<llama_dsv41_engram_transaction>(
                    mem->pimpl->config.engram->prepare(tokens));
        }

        std::set<std::tuple<ggml_tensor *, size_t, size_t>> snapshot_keys;
        auto snapshot = [&](ggml_tensor * tensor, size_t offset, size_t size) {
            if (tensor == nullptr || mem->pimpl->config.no_alloc ||
                    !snapshot_keys.emplace(tensor, offset, size).second) {
                return;
            }
            llama_memory_dsv41::impl::tensor_snapshot entry;
            entry.tensor = tensor;
            entry.offset = offset;
            entry.data.resize(size);
            ggml_backend_tensor_get(tensor, entry.data.data(), offset, size);
            next->snapshots.push_back(std::move(entry));
        };
        for (llama_seq_id seq_id : topology) {
            for (ggml_tensor * raw : mem->pimpl->raw) {
                const size_t row_bytes = raw->nb[1];
                for (llama_pos pos : next->plan.positions) {
                    const size_t row = (size_t) seq_id*mem->pimpl->config.raw_window +
                        (uint32_t) pos%mem->pimpl->config.raw_window;
                    snapshot(raw, row*row_bytes, row_bytes);
                }
            }
            for (const auto & source_plan : next->plan.sources) {
                const auto & storage = mem->pimpl->sources.at(source_plan.source_layer);
                for (int64_t row : source_plan.compression.write_idxs) {
                    const size_t physical = (size_t) seq_id*storage.capacity + row;
                    snapshot(storage.kv, physical*storage.kv->nb[1], storage.kv->nb[1]);
                    snapshot(storage.index, physical*storage.index->nb[1], storage.index->nb[1]);
                }
                if (storage.carry_kv != nullptr) {
                    snapshot(
                            storage.carry_kv,
                            (size_t) seq_id*storage.carry_kv->nb[2],
                            storage.carry_kv->nb[2]);
                    snapshot(
                            storage.carry_score,
                            (size_t) seq_id*storage.carry_score->nb[2],
                            storage.carry_score->nb[2]);
                }
            }
        }

        mem->pimpl->transaction_active = true;
        transaction = std::move(next);
        return true;
    } catch (const std::exception & error) {
        if (next->engram) {
            mem->pimpl->config.engram->rollback(*next->engram);
        }
        LLAMA_LOG_ERROR("%s: %s\n", __func__, error.what());
        status = LLAMA_MEMORY_STATUS_FAILED_PREPARE;
        return false;
    }
}

void llama_memory_dsv41_context::commit() {
    if (!transaction || mem == nullptr) {
        return;
    }
    GGML_ASSERT(transaction->seq_ids.size() == 1);
    const llama_seq_id seq_id = transaction->seq_ids.front();
    llama_memory_dsv41::impl::rollback_state rollback;
    rollback.start_pos = transaction->start_positions.front();
    rollback.sequence = mem->pimpl->sequences[seq_id];
    if (transaction->engram) {
        rollback.engram = transaction->engram_before.at(seq_id);
        rollback.has_engram = true;
    }

    std::vector<int32_t> committed_candidates;
    if (!mem->pimpl->config.no_alloc && transaction->plan.candidate_width > 0) {
        const size_t row_bytes = mem->pimpl->candidate_ids->nb[1];
        std::vector<int32_t> ids(transaction->plan.candidate_width);
        uint32_t token = 0;
        for (uint32_t current = 0; current < transaction->plan.token_seq_ids.size(); ++current) {
            if (std::find(
                    transaction->plan.token_seq_ids[current].begin(),
                    transaction->plan.token_seq_ids[current].end(),
                    seq_id) != transaction->plan.token_seq_ids[current].end()) {
                token = current;
            }
        }
        ggml_backend_tensor_get(
                mem->pimpl->candidate_ids,
                ids.data(),
                (size_t) token*row_bytes,
                ids.size()*sizeof(int32_t));
        transaction->next_sequences[seq_id].candidates = std::move(ids);
    }

    const std::vector<int32_t> position_values =
        mem->pimpl->position_state_values(transaction->next_sequences[seq_id]);
    committed_candidates =
        mem->pimpl->candidate_state_values(transaction->next_sequences[seq_id]);

    auto rollback_slot = mem->pimpl->rollback_states.end();
    bool inserted_rollback = false;
    try {
        std::tie(rollback_slot, inserted_rollback) =
            mem->pimpl->rollback_states.emplace(seq_id, llama_memory_dsv41::impl::rollback_state {});
        if (transaction->engram) {
            mem->pimpl->config.engram->commit(*transaction->engram);
        }
    } catch (...) {
        if (inserted_rollback) {
            mem->pimpl->rollback_states.erase(rollback_slot);
        }
        throw;
    }

    if (!mem->pimpl->config.no_alloc) {
        ggml_backend_tensor_set(
                mem->pimpl->committed_candidate_ids,
                committed_candidates.data(),
                (size_t) seq_id*mem->pimpl->committed_candidate_ids->nb[1],
                committed_candidates.size()*sizeof(int32_t));
        ggml_backend_tensor_set(
                mem->pimpl->positions,
                position_values.data(),
                (size_t) seq_id*mem->pimpl->positions->nb[1],
                position_values.size()*sizeof(int32_t));
    }
    mem->pimpl->sequences.swap(transaction->next_sequences);
    rollback.snapshots.swap(transaction->snapshots);
    rollback_slot->second = std::move(rollback);
    mem->pimpl->generation = transaction->plan.generation;
    mem->pimpl->transaction_active = false;
    transaction.reset();
}

void llama_memory_dsv41_context::rollback() {
    if (!transaction || mem == nullptr) {
        return;
    }
    if (full) {
        transaction.reset();
        return;
    }
    if (transaction->engram) {
        mem->pimpl->config.engram->rollback(*transaction->engram);
    }
    if (!mem->pimpl->config.no_alloc) {
        for (const auto & snapshot : transaction->snapshots) {
            ggml_backend_tensor_set(
                    snapshot.tensor, snapshot.data.data(), snapshot.offset, snapshot.data.size());
        }
    }
    mem->pimpl->transaction_active = false;
    transaction.reset();
}

llama_memory_status llama_memory_dsv41_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_dsv41_context::get_ubatch() const {
    if (status != LLAMA_MEMORY_STATUS_SUCCESS || i_next >= ubatches.size()) {
        throw std::runtime_error("DeepSeek V4.1 memory context has no current ubatch");
    }
    return ubatches[i_next];
}

const llama_dsv41_memory_plan & llama_memory_dsv41_context::plan() const {
    if (!transaction) {
        throw std::runtime_error("DeepSeek V4.1 memory transaction is not prepared");
    }
    return transaction->plan;
}

const llama_dsv41_memory_plan & llama_memory_dsv41_context::graph_plan(
        const llama_ubatch & ubatch) const {
    if (!full) {
        return plan();
    }
    if (mem == nullptr || ubatch.n_tokens == 0 ||
            ubatch.n_tokens > mem->pimpl->config.n_ubatch) {
        throw std::invalid_argument("DeepSeek V4.1 full graph ubatch is invalid");
    }
    if (transaction && transaction->plan.positions.size() == ubatch.n_tokens) {
        return transaction->plan;
    }

    auto next = std::make_unique<transaction_state>();
    next->plan.generation = mem->pimpl->generation;
    next->next_sequences = mem->pimpl->sequences;
    next->seq_ids = { 0 };
    next->start_positions = { 0 };
    next->next_sequences[0].pos = ubatch.n_tokens - 1;

    const uint32_t raw_width = mem->pimpl->config.raw_window;
    const uint32_t persist_first =
        ubatch.n_tokens > raw_width ? ubatch.n_tokens - raw_width : 0;
    for (uint32_t token = 0; token < ubatch.n_tokens; ++token) {
        next->plan.token_seq_ids.push_back({ 0 });
        next->plan.positions.push_back(token);
        if (token >= persist_first) {
            next->plan.raw.persist_src_idxs.push_back(token);
            next->plan.raw.write_idxs.push_back(token%raw_width);
        }
        const uint32_t visible = std::min(raw_width, token + 1);
        next->plan.raw.n_visible.push_back(visible);
        const auto order = llama_dsv41_raw_ring_order(token, raw_width);
        for (uint32_t row = 0; row < raw_width; ++row) {
            next->plan.raw.read_idxs.push_back(row < visible ? order[row] : -1);
            next->plan.raw.mask.push_back(
                    row < visible ? 0.0f : -std::numeric_limits<float>::infinity());
        }
    }
    for (uint32_t source : mem->pimpl->config.kv_sources) {
        llama_dsv41_source_plan source_plan;
        source_plan.source_layer = source;
        source_plan.ratio = mem->pimpl->config.ratios[source];
        source_plan.capacity = source_capacity(mem->pimpl->config, source);
        source_plan.compression = llama_dsv41_build_compression_plan(
                next->plan.positions, source_plan.ratio, source_plan.capacity);
        const uint32_t max_visible = source_plan.compression.n_visible.empty() ? 0 :
            *std::max_element(
                    source_plan.compression.n_visible.begin(),
                    source_plan.compression.n_visible.end());
        for (int32_t visible : source_plan.compression.n_visible) {
            for (uint32_t row = 0; row < max_visible; ++row) {
                source_plan.read_idxs.push_back(row < (uint32_t) visible ? row : -1);
            }
        }
        next->plan.sources.push_back(std::move(source_plan));
    }
    const uint32_t visible_blocks =
        (ubatch.n_tokens + mem->pimpl->config.candidate_block_size - 1)/
        mem->pimpl->config.candidate_block_size;
    next->plan.candidate_width = std::min(
            visible_blocks, mem->pimpl->config.candidate_topk_blocks);
    transaction = std::move(next);
    return transaction->plan;
}

llama_dsv41_graph_topology llama_memory_dsv41_context::topology(
        const llama_ubatch & ubatch,
        uint32_t n_outputs) const {
    if (mem == nullptr) {
        throw std::runtime_error("DeepSeek V4.1 memory context has no memory");
    }
    if (full) {
        graph_plan(ubatch);
    }
    llama_dsv41_graph_topology result;
    result.n_tokens = ubatch.n_tokens;
    result.n_seqs = transaction ? transaction->seq_ids.size() : ubatch.n_seqs;
    result.n_outputs = n_outputs;
    result.backend_layout = mem->pimpl->backend_layout;
    result.engram_enabled = mem->pimpl->config.engram_enabled;
    result.expert_enabled = mem->pimpl->config.expert_enabled;
    result.transaction_generation = transaction ? transaction->plan.generation : mem->pimpl->generation;
    if (transaction) {
        result.seq_ids = transaction->seq_ids;
        result.start_positions = transaction->start_positions;
        for (llama_seq_id seq_id : transaction->seq_ids) {
            const llama_pos final_pos = transaction->next_sequences[seq_id].pos;
            result.visible_raw_widths.push_back(std::min<uint32_t>(
                    mem->pimpl->config.raw_window, final_pos + 1));
        }
        for (const auto & source : transaction->plan.sources) {
            result.source_ratios.push_back(source.ratio);
            result.visible_compressed_widths.push_back(
                    source.compression.n_visible.empty() ? 0 :
                    *std::max_element(source.compression.n_visible.begin(), source.compression.n_visible.end()));
            result.source_carry_counts.push_back(
                    source.ratio == 2 ? (transaction->plan.positions.back() + 1)%source.ratio : 0);
        }
        result.candidate_width = transaction->plan.candidate_width;
    } else {
        const uint32_t n_seqs = std::min(ubatch.n_seqs, mem->pimpl->config.n_seq);
        for (uint32_t seq = 0; seq < n_seqs; ++seq) {
            result.seq_ids.push_back(seq);
            result.start_positions.push_back(0);
            result.visible_raw_widths.push_back(mem->pimpl->config.raw_window);
        }
        for (uint32_t source : mem->pimpl->config.kv_sources) {
            result.source_ratios.push_back(mem->pimpl->config.ratios[source]);
            result.visible_compressed_widths.push_back(source_capacity(mem->pimpl->config, source));
            result.source_carry_counts.push_back(mem->pimpl->config.ratios[source] == 2 ? 1 : 0);
        }
        result.candidate_width = mem->pimpl->candidate_width;
    }
    return result;
}

const llama_dsv41_engram_transaction * llama_memory_dsv41_context::engram_transaction() const {
    return transaction && transaction->engram ? transaction->engram.get() : nullptr;
}

std::vector<int32_t> llama_memory_dsv41_context::engram_row_ids(uint32_t engram_layer) const {
    const auto * prepared = engram_transaction();
    if (prepared == nullptr) {
        return {};
    }
    if (engram_layer >= LLAMA_ENGRAM_LAYERS) {
        throw std::invalid_argument("DeepSeek V4.1 Engram layer index is out of range");
    }

    const uint32_t * source = prepared->row_ids(engram_layer);
    std::vector<int32_t> result(prepared->token_count()*LLAMA_ENGRAM_COLS);
    for (size_t token = 0; token < prepared->token_count(); ++token) {
        for (uint32_t column = 0; column < LLAMA_ENGRAM_COLS; ++column) {
            const uint32_t row =
                source[token*LLAMA_ENGRAM_LAYERS*LLAMA_ENGRAM_COLS + column];
            if (row > (uint32_t) INT32_MAX) {
                throw std::runtime_error("DeepSeek V4.1 Engram row ID exceeds I32");
            }
            result[token*LLAMA_ENGRAM_COLS + column] = row;
        }
    }
    return result;
}

void llama_memory_dsv41_context::stage_candidate_ids(
        uint32_t token,
        const std::vector<int32_t> & ids) {
    if (!transaction || mem == nullptr || token >= transaction->plan.positions.size() ||
            ids.size() != transaction->plan.candidate_width) {
        throw std::invalid_argument("DeepSeek V4.1 candidate staging shape is invalid");
    }
    if (!mem->pimpl->config.no_alloc && !ids.empty()) {
        ggml_backend_tensor_set(
                mem->pimpl->candidate_ids,
                ids.data(),
                (size_t) token*mem->pimpl->candidate_ids->nb[1],
                ids.size()*sizeof(int32_t));
    }
}
