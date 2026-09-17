#include "llama-dsv41-engram.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <stdexcept>
#include <utility>

static llama_dsv41_engram_sequence_state dsv41_engram_initial_state() {
    llama_dsv41_engram_sequence_state state;
    state.history.reset();
    return state;
}

void llama_dsv41_validate_engram_extent(const llama_dsv41_engram_extent & extent) {
    if (extent.fname.empty() || extent.rows == 0 || extent.columns != LLAMA_ENGRAM_ROW_BYTES ||
            extent.row_count != extent.rows || extent.type != GGML_TYPE_I8) {
        throw std::invalid_argument("DeepSeek V4.1 Engram tensor must be I8 [264, rows]");
    }
    const uint64_t bytes = (uint64_t) extent.rows*LLAMA_ENGRAM_ROW_BYTES;
    if (extent.offset > (uint64_t) INT64_MAX || bytes > (uint64_t) INT64_MAX - extent.offset) {
        throw std::invalid_argument("DeepSeek V4.1 Engram tensor extent overflows");
    }
}

struct llama_dsv41_engram_transaction::impl {
    const llama_dsv41_engram_runtime * owner = nullptr;
    uint64_t generation = 0;
    bool active = false;
    size_t count = 0;
    std::vector<uint32_t> ids;
    std::array<std::vector<float>, LLAMA_ENGRAM_LAYERS> decoded;
    std::vector<uint8_t> mask;
    std::map<llama_seq_id, llama_dsv41_engram_sequence_state> next;
};

llama_dsv41_engram_transaction::llama_dsv41_engram_transaction() : pimpl(std::make_unique<impl>()) {}
llama_dsv41_engram_transaction::~llama_dsv41_engram_transaction() = default;
llama_dsv41_engram_transaction::llama_dsv41_engram_transaction(llama_dsv41_engram_transaction && other) noexcept = default;
llama_dsv41_engram_transaction & llama_dsv41_engram_transaction::operator=(llama_dsv41_engram_transaction && other) noexcept = default;

size_t llama_dsv41_engram_transaction::token_count() const {
    return pimpl->count;
}

const uint32_t * llama_dsv41_engram_transaction::row_ids(uint32_t layer) const {
    if (layer >= LLAMA_ENGRAM_LAYERS || pimpl->count == 0) {
        return nullptr;
    }
    return pimpl->ids.data() + layer*LLAMA_ENGRAM_COLS;
}

const float * llama_dsv41_engram_transaction::rows(uint32_t layer) const {
    if (layer >= LLAMA_ENGRAM_LAYERS || pimpl->decoded[layer].empty()) {
        return nullptr;
    }
    return pimpl->decoded[layer].data();
}

const uint8_t * llama_dsv41_engram_transaction::text_mask() const {
    return pimpl->mask.empty() ? nullptr : pimpl->mask.data();
}

void llama_dsv41_engram_transaction::upload_layer(
        uint32_t layer,
        size_t token_offset,
        size_t token_count,
        ggml_tensor * rows_input,
        ggml_tensor * text_select_input) const {
    if (!pimpl->active) {
        throw std::invalid_argument("DeepSeek V4.1 Engram transaction is not active");
    }
    if (layer >= LLAMA_ENGRAM_LAYERS || token_offset > pimpl->count ||
            token_count > pimpl->count - token_offset) {
        throw std::invalid_argument("DeepSeek V4.1 Engram upload range is invalid");
    }
    const int64_t row_width = LLAMA_ENGRAM_COLS*LLAMA_ENGRAM_DIM;
    if (rows_input == nullptr || text_select_input == nullptr ||
            rows_input->type != GGML_TYPE_F32 || rows_input->ne[0] != row_width ||
            ggml_nelements(rows_input) != row_width*(int64_t) token_count ||
            text_select_input->type != GGML_TYPE_I32 ||
            ggml_nelements(text_select_input) != (int64_t) token_count) {
        throw std::invalid_argument("DeepSeek V4.1 Engram input tensor shape mismatch");
    }

    const size_t row_offset = token_offset*row_width;
    ggml_backend_tensor_set(
            rows_input,
            pimpl->decoded[layer].data() + row_offset,
            0,
            token_count*row_width*sizeof(float));
    std::vector<int32_t> select(token_count);
    for (size_t i = 0; i < token_count; ++i) {
        select[i] = pimpl->mask[token_offset + i] != 0 ? (int32_t) (token_count + i) : (int32_t) i;
    }
    ggml_backend_tensor_set(
            text_select_input,
            select.data(),
            0,
            token_count*sizeof(int32_t));
}

struct llama_dsv41_engram_runtime::impl {
    llama_engram_hasher hasher;
    std::array<std::unique_ptr<llama_engram_table>, LLAMA_ENGRAM_LAYERS> tables;
    std::map<llama_seq_id, llama_dsv41_engram_sequence_state> sequences;
    size_t max_tokens;
    uint64_t generation = 0;

    impl(
            llama_engram_layout layout,
            const std::array<llama_dsv41_engram_extent, LLAMA_ENGRAM_LAYERS> & extents,
            size_t max_tokens) :
        hasher(std::move(layout)),
        max_tokens(max_tokens) {
        if (max_tokens == 0 || max_tokens > (size_t) INT32_MAX/2) {
            throw std::invalid_argument("DeepSeek V4.1 Engram token bound is invalid");
        }
        for (size_t i = 0; i < LLAMA_ENGRAM_LAYERS; ++i) {
            llama_dsv41_validate_engram_extent(extents[i]);
            if (extents[i].rows != hasher.layout().rows[i]) {
                throw std::invalid_argument("DeepSeek V4.1 Engram tensor rows do not match metadata");
            }
            tables[i] = std::make_unique<llama_engram_table>(
                    extents[i].fname, extents[i].offset, extents[i].rows);
        }
    }
};

llama_dsv41_engram_runtime::llama_dsv41_engram_runtime(
        llama_engram_layout layout,
        const std::array<llama_dsv41_engram_extent, LLAMA_ENGRAM_LAYERS> & extents,
        size_t max_tokens) :
    pimpl(std::make_unique<impl>(std::move(layout), extents, max_tokens)) {}

llama_dsv41_engram_runtime::~llama_dsv41_engram_runtime() = default;

llama_dsv41_engram_transaction llama_dsv41_engram_runtime::prepare(
        const std::vector<llama_dsv41_engram_token> & tokens) {
    if (tokens.size() > pimpl->max_tokens) {
        throw std::invalid_argument("DeepSeek V4.1 Engram batch exceeds the configured bound");
    }

    llama_dsv41_engram_transaction result;
    result.pimpl->owner = this;
    result.pimpl->generation = pimpl->generation;
    result.pimpl->count = tokens.size();
    result.pimpl->ids.resize(tokens.size()*LLAMA_ENGRAM_LAYERS*LLAMA_ENGRAM_COLS);
    result.pimpl->mask.resize(tokens.size());
    result.pimpl->next = pimpl->sequences;

    for (size_t i = 0; i < tokens.size(); ++i) {
        const llama_dsv41_engram_token & token = tokens[i];
        if (token.seq_ids.empty()) {
            throw std::invalid_argument("DeepSeek V4.1 Engram token has no sequence");
        }

        uint32_t expected[LLAMA_ENGRAM_LAYERS*LLAMA_ENGRAM_COLS] = {};
        bool have_expected = false;
        std::set<llama_seq_id> seen;
        for (llama_seq_id seq_id : token.seq_ids) {
            if (seq_id < 0) {
                throw std::invalid_argument("DeepSeek V4.1 Engram sequence ID is negative");
            }
            if (!seen.insert(seq_id).second) {
                throw std::invalid_argument("DeepSeek V4.1 Engram token repeats a sequence ID");
            }
            auto inserted = result.pimpl->next.emplace(seq_id, dsv41_engram_initial_state());
            llama_dsv41_engram_sequence_state & state = inserted.first->second;
            if (token.pos != state.pos + 1) {
                throw std::invalid_argument("DeepSeek V4.1 Engram sequence position is not contiguous");
            }

            uint32_t current[LLAMA_ENGRAM_LAYERS*LLAMA_ENGRAM_COLS] = {};
            pimpl->hasher.hash(state.history, &token.token, &token.text, 1, current);
            state.pos = token.pos;
            if (have_expected && std::memcmp(expected, current, sizeof(expected)) != 0) {
                throw std::invalid_argument("DeepSeek V4.1 coupled sequence histories differ");
            }
            std::memcpy(expected, current, sizeof(expected));
            have_expected = true;
        }

        std::memcpy(
                result.pimpl->ids.data() + i*LLAMA_ENGRAM_LAYERS*LLAMA_ENGRAM_COLS,
                expected,
                sizeof(expected));
        result.pimpl->mask[i] = token.text != 0;
    }

    const size_t row_values = tokens.size()*LLAMA_ENGRAM_COLS*LLAMA_ENGRAM_DIM;
    for (size_t layer = 0; layer < LLAMA_ENGRAM_LAYERS; ++layer) {
        result.pimpl->decoded[layer].resize(row_values);
        pimpl->tables[layer]->read_batch(
                result.pimpl->ids.data() + layer*LLAMA_ENGRAM_COLS,
                tokens.size(),
                LLAMA_ENGRAM_LAYERS*LLAMA_ENGRAM_COLS,
                result.pimpl->decoded[layer].data());
    }
    result.pimpl->active = true;
    return result;
}

void llama_dsv41_engram_runtime::commit(llama_dsv41_engram_transaction & transaction) {
    if (!transaction.pimpl->active || transaction.pimpl->owner != this) {
        throw std::invalid_argument("DeepSeek V4.1 Engram transaction is not active");
    }
    if (transaction.pimpl->generation != pimpl->generation) {
        throw std::runtime_error("DeepSeek V4.1 Engram transaction is stale");
    }
    auto next = transaction.pimpl->next;
    pimpl->sequences.swap(next);
    ++pimpl->generation;
    transaction.pimpl->active = false;
}

void llama_dsv41_engram_runtime::rollback(llama_dsv41_engram_transaction & transaction) {
    if (transaction.pimpl->owner != this) {
        throw std::invalid_argument("DeepSeek V4.1 Engram transaction belongs to another runtime");
    }
    transaction.pimpl->active = false;
}

void llama_dsv41_engram_runtime::seq_reset(llama_seq_id seq_id) {
    if (seq_id < 0) {
        throw std::invalid_argument("DeepSeek V4.1 Engram sequence ID is negative");
    }
    pimpl->sequences[seq_id] = dsv41_engram_initial_state();
    ++pimpl->generation;
}

void llama_dsv41_engram_runtime::seq_copy(llama_seq_id seq_id_src, llama_seq_id seq_id_dst) {
    if (seq_id_src < 0 || seq_id_dst < 0) {
        throw std::invalid_argument("DeepSeek V4.1 Engram sequence ID is negative");
    }
    const auto it = pimpl->sequences.find(seq_id_src);
    pimpl->sequences[seq_id_dst] = it == pimpl->sequences.end() ?
            dsv41_engram_initial_state() : it->second;
    ++pimpl->generation;
}

void llama_dsv41_engram_runtime::seq_remove(llama_seq_id seq_id) {
    if (seq_id < 0) {
        throw std::invalid_argument("DeepSeek V4.1 Engram sequence ID is negative");
    }
    pimpl->sequences.erase(seq_id);
    ++pimpl->generation;
}

llama_dsv41_engram_snapshot llama_dsv41_engram_runtime::checkpoint() const {
    return { pimpl->sequences };
}

void llama_dsv41_engram_runtime::restore(const llama_dsv41_engram_snapshot & snapshot) {
    auto restored = snapshot.sequences;
    for (auto & item : restored) {
        if (item.first < 0 || item.second.pos < -1) {
            throw std::invalid_argument("DeepSeek V4.1 Engram snapshot is invalid");
        }
        pimpl->hasher.hash(item.second.history, nullptr, nullptr, 0, nullptr);
    }
    pimpl->sequences.swap(restored);
    ++pimpl->generation;
}

llama_dsv41_engram_sequence_state llama_dsv41_engram_runtime::sequence(llama_seq_id seq_id) const {
    const auto it = pimpl->sequences.find(seq_id);
    return it == pimpl->sequences.end() ? dsv41_engram_initial_state() : it->second;
}

size_t llama_dsv41_engram_runtime::max_tokens() const {
    return pimpl->max_tokens;
}

static ggml_tensor * dsv41_bf16_f32(ggml_context * ctx, ggml_tensor * tensor) {
    return ggml_cast(ctx, ggml_cast(ctx, tensor, GGML_TYPE_BF16), GGML_TYPE_F32);
}

static void dsv41_engram_gate_f32(
        ggml_tensor * dst,
        const ggml_tensor * src,
        int ith,
        int nth,
        void *) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && src->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src));
    const float * input = static_cast<const float *>(src->data);
    float * output = static_cast<float *>(dst->data);
    const int64_t count = ggml_nelements(src);
    for (int64_t i = ith; i < count; i += nth) {
        const float signed_root = std::copysign(std::sqrt(std::max(std::abs(input[i]), 1.0e-6f)), input[i]);
        output[i] = 1.0f/(1.0f + std::exp(-signed_root));
    }
}

ggml_tensor * llama_dsv41_build_engram_gate(
        ggml_context * ctx,
        ggml_tensor * dot,
        ggml_backend_sched_t sched,
        ggml_backend_t backend_cpu) {
    if (ctx == nullptr || dot == nullptr || sched == nullptr || backend_cpu == nullptr) {
        throw std::invalid_argument("DeepSeek V4.1 Engram gate input is null");
    }
    if (!ggml_backend_is_cpu(backend_cpu)) {
        throw std::invalid_argument("DeepSeek V4.1 Engram gate backend is not local CPU");
    }
    bool found = false;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        found = found || ggml_backend_sched_get_backend(sched, i) == backend_cpu;
    }
    if (!found) {
        throw std::invalid_argument("DeepSeek V4.1 Engram gate CPU backend is not in the scheduler");
    }
    ggml_tensor * gate = ggml_map_custom1(ctx, dot, dsv41_engram_gate_f32, GGML_N_TASKS_MAX, nullptr);
    ggml_backend_sched_set_tensor_backend(sched, gate, backend_cpu);
    return gate;
}

ggml_tensor * llama_dsv41_build_engram_add(
        ggml_context * ctx,
        ggml_tensor * residual,
        ggml_tensor * projected,
        ggml_tensor * q_norm,
        ggml_tensor * k_norm,
        ggml_tensor * text_select,
        float rms_eps,
        ggml_backend_sched_t sched,
        ggml_backend_t backend_cpu) {
    if (ctx == nullptr || residual == nullptr || projected == nullptr || q_norm == nullptr || k_norm == nullptr) {
        throw std::invalid_argument("DeepSeek V4.1 Engram graph input is null");
    }

    const int64_t width = residual->ne[0];
    const int64_t streams = residual->ne[1];
    const int64_t tokens = residual->ne[2];
    if (width <= 0 || streams != 4 || tokens <= 0 || tokens > INT32_MAX/2 ||
            projected->ne[0] != 5*width || projected->ne[1] != tokens ||
            q_norm->ne[0] != width || q_norm->ne[1] != streams ||
            k_norm->ne[0] != width || k_norm->ne[1] != streams ||
            (text_select != nullptr &&
             (text_select->type != GGML_TYPE_I32 || ggml_nelements(text_select) != tokens))) {
        throw std::invalid_argument("DeepSeek V4.1 Engram graph shape mismatch");
    }

    projected = dsv41_bf16_f32(ctx, projected);
    ggml_tensor * value = ggml_view_2d(
            ctx, projected, width, tokens, projected->nb[1], 4*projected->nb[0]*width);
    ggml_tensor * result = nullptr;
    for (int64_t stream = 0; stream < streams; ++stream) {
        ggml_tensor * hidden = ggml_view_2d(
                ctx, residual, width, tokens, residual->nb[2], stream*residual->nb[1]);
        ggml_tensor * key = ggml_view_2d(
                ctx, projected, width, tokens, projected->nb[1], stream*projected->nb[0]*width);
        ggml_tensor * qw = ggml_view_1d(ctx, q_norm, width, stream*q_norm->nb[1]);
        ggml_tensor * kw = ggml_view_1d(ctx, k_norm, width, stream*k_norm->nb[1]);

        ggml_tensor * hidden_norm = ggml_rms_norm(ctx, hidden, rms_eps);
        ggml_tensor * key_norm = ggml_rms_norm(ctx, key, rms_eps);
        ggml_tensor * dot = ggml_mul(ctx, hidden_norm, qw);
        dot = ggml_mul(ctx, dot, kw);
        dot = ggml_mul(ctx, dot, key_norm);
        dot = ggml_scale(ctx, ggml_sum_rows(ctx, dot), 1.0f/std::sqrt((float) width));

        ggml_tensor * gate = llama_dsv41_build_engram_gate(ctx, dot, sched, backend_cpu);
        ggml_tensor * updated = dsv41_bf16_f32(ctx, ggml_add(ctx, hidden, ggml_mul(ctx, value, gate)));
        if (text_select != nullptr) {
            updated = ggml_get_rows(ctx, ggml_concat(ctx, hidden, updated, 1), text_select);
        }
        updated = ggml_reshape_3d(ctx, updated, width, 1, tokens);
        result = result == nullptr ? updated : ggml_concat(ctx, result, updated, 1);
    }
    return result;
}

ggml_tensor * llama_dsv41_build_engram(
        ggml_context * ctx,
        ggml_tensor * residual,
        ggml_tensor * rows,
        ggml_tensor * engram_kv,
        ggml_tensor * q_norm,
        ggml_tensor * k_norm,
        ggml_tensor * text_select,
        float rms_eps,
        ggml_backend_sched_t sched,
        ggml_backend_t backend_cpu) {
    if (ctx == nullptr || residual == nullptr || rows == nullptr || engram_kv == nullptr ||
            rows->ne[0] != LLAMA_ENGRAM_COLS*LLAMA_ENGRAM_DIM ||
            engram_kv->ne[0] != rows->ne[0] ||
            engram_kv->ne[1] != 5*residual->ne[0]) {
        throw std::invalid_argument("DeepSeek V4.1 Engram projection shape mismatch");
    }
    ggml_tensor * projected = ggml_mul_mat(ctx, engram_kv, rows);
    return llama_dsv41_build_engram_add(
            ctx, residual, projected, q_norm, k_norm, text_select, rms_eps, sched, backend_cpu);
}
