#include "llama-dsv41.h"
#include "llama-dsv41-engram.h"
#include "llama-dsv41-expert.h"
#include "llama-hparams.h"
#include "llama-memory-dsv41.h"
#include "models.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static float dsv41_rope_attn_factor(float freq_scale) {
    return 1.0f/(1.0f + 0.1f*logf(1.0f/freq_scale));
}

std::string llama_dsv41_graph_trace_name(const char * trace, uint32_t layer);
ggml_tensor * llama_dsv41_graph_append_zero_row(ggml_context * ctx, ggml_tensor * tensor);
ggml_tensor * llama_dsv41_graph_completion_zero(
        ggml_context * ctx,
        ggml_tensor * dependency,
        ggml_type type);

std::string llama_dsv41_graph_trace_name(
        const char * trace,
        uint32_t layer) {
    return "dsv41.trace." + std::string(trace) + ".l" +
        std::to_string(layer);
}

struct llama_model_deepseek41::engram_model {
    llama_engram_layout layout;
    std::array<llama_dsv41_engram_extent, LLAMA_ENGRAM_LAYERS> extents;
};

std::unique_ptr<llama_dsv41_engram_runtime> llama_model_deepseek41::create_memory_engram_runtime(
        size_t max_tokens) const {
    if (hparams.no_alloc) {
        return nullptr;
    }
    return engram ?
        std::make_unique<llama_dsv41_engram_runtime>(engram->layout, engram->extents, max_tokens) :
        nullptr;
}

namespace {

struct dsv41_graph_source_input {
    uint32_t layer = 0;
    uint32_t ratio = 0;
    uint32_t capacity = 0;
    uint32_t read_width = 0;

    ggml_tensor * read_idxs = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * carry_read_idxs = nullptr;
    ggml_tensor * state_read_idxs = nullptr;
    ggml_tensor * state_persist_src_idxs = nullptr;
    ggml_tensor * state_persist_dst_idxs = nullptr;
    ggml_tensor * write_idxs = nullptr;
    ggml_tensor * write_pos = nullptr;
    ggml_tensor * candidate_pad_mask = nullptr;
    ggml_tensor * candidate_block_bias = nullptr;
    ggml_tensor * candidate_final_block = nullptr;
    ggml_tensor * row_blocks = nullptr;
};

class dsv41_graph_input final : public llm_graph_input_i {
public:
    dsv41_graph_input(
            ggml_context * ctx,
            const llama_cparams & cparams,
            const llama_hparams & hparams,
            const llama_memory_dsv41_context * mctx,
            const llama_ubatch & ubatch,
            uint32_t n_outputs) :
        cparams(cparams),
        mctx(mctx),
        topology(mctx->topology(ubatch, n_outputs)) {
        const llama_memory_dsv41 * memory = mctx->memory();
        const uint32_t n_tokens = ubatch.n_tokens;
        const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
        const llama_dsv41_memory_plan & plan = mctx->graph_plan(ubatch);

        initial_pre = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hparams.dsv4_hc_mult, n_tokens);
        ggml_set_input(initial_pre);
        ggml_set_name(initial_pre, "dsv41.inp.carried_pre");

        raw_persist_src_idxs = ggml_new_tensor_1d(
                ctx, GGML_TYPE_I32, plan.raw.persist_src_idxs.size());
        raw_write_idxs = ggml_new_tensor_1d(
                ctx, GGML_TYPE_I64, plan.raw.write_idxs.size());
        raw_read_idxs = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, hparams.n_swa, n_tokens);
        raw_mask = ggml_new_tensor_4d(ctx, type_mask, hparams.n_swa, 1, 1, n_tokens);
        ggml_set_input(raw_persist_src_idxs);
        ggml_set_input(raw_write_idxs);
        ggml_set_input(raw_read_idxs);
        ggml_set_input(raw_mask);
        ggml_set_name(raw_persist_src_idxs, "dsv41.inp.raw.persist_src_idxs");
        ggml_set_name(raw_write_idxs, "dsv41.inp.raw.write_idxs");
        ggml_set_name(raw_read_idxs, "dsv41.inp.raw.read_idxs");
        ggml_set_name(raw_mask, "dsv41.inp.raw.mask");

        for (const llama_dsv41_source_plan & source_plan : plan.sources) {
            dsv41_graph_source_input source;
            source.layer = source_plan.source_layer;
            source.ratio = source_plan.ratio;
            source.capacity = source_plan.capacity;
            source.read_width = source_plan.read_idxs.empty() ? 0 :
                source_plan.read_idxs.size()/n_tokens;
            if (source.read_width > 0) {
                source.read_idxs = ggml_new_tensor_2d(
                        ctx, GGML_TYPE_I32, source.read_width, n_tokens);
                source.mask = ggml_new_tensor_4d(
                        ctx, type_mask, source.read_width, 1, 1, n_tokens);
                ggml_set_input(source.read_idxs);
                ggml_set_input(source.mask);
            }
            if (!source_plan.compression.state_read_idxs.empty()) {
                source.carry_read_idxs = ggml_new_tensor_1d(
                        ctx, GGML_TYPE_I32, source.ratio);
                source.state_read_idxs = ggml_new_tensor_1d(
                        ctx, GGML_TYPE_I32, source_plan.compression.state_read_idxs.size());
                ggml_set_input(source.carry_read_idxs);
                ggml_set_input(source.state_read_idxs);
            }
            if (!source_plan.compression.state_persist_src_idxs.empty()) {
                source.state_persist_src_idxs = ggml_new_tensor_1d(
                        ctx, GGML_TYPE_I32, source_plan.compression.state_persist_src_idxs.size());
                source.state_persist_dst_idxs = ggml_new_tensor_1d(
                        ctx, GGML_TYPE_I64,
                        source_plan.compression.state_persist_dst_idxs.size()*topology.n_seqs);
                ggml_set_input(source.state_persist_src_idxs);
                ggml_set_input(source.state_persist_dst_idxs);
            }
            if (!source_plan.compression.write_idxs.empty()) {
                source.write_idxs = ggml_new_tensor_1d(
                        ctx, GGML_TYPE_I64,
                        source_plan.compression.write_idxs.size()*topology.n_seqs);
                source.write_pos = ggml_new_tensor_1d(
                        ctx, GGML_TYPE_I32, source_plan.compression.write_pos.size());
                ggml_set_input(source.write_idxs);
                ggml_set_input(source.write_pos);
            }
            if (source.layer == hparams.dsv41_candidate_source_layer && source.read_width > 0) {
                const uint32_t block_size = hparams.dsv41_candidate_block_size;
                const uint32_t n_blocks = (source.read_width + block_size - 1)/block_size;
                const uint32_t padded = n_blocks*block_size;
                source.candidate_pad_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, padded, n_tokens);
                source.candidate_block_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_blocks, n_tokens);
                source.candidate_final_block = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_tokens);
                source.row_blocks = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, source.read_width);
                ggml_set_input(source.candidate_pad_mask);
                ggml_set_input(source.candidate_block_bias);
                ggml_set_input(source.candidate_final_block);
                ggml_set_input(source.row_blocks);
            }
            sources.emplace(source.layer, source);
        }

        if (topology.engram_enabled) {
            for (uint32_t index = 0; index < LLAMA_ENGRAM_LAYERS; ++index) {
                engram_rows[index] = ggml_new_tensor_2d(
                        ctx, GGML_TYPE_F32, LLAMA_ENGRAM_COLS*LLAMA_ENGRAM_DIM, n_tokens);
                engram_select[index] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
                engram_row_ids[index] = ggml_new_tensor_2d(
                        ctx, GGML_TYPE_I32, LLAMA_ENGRAM_COLS, n_tokens);
                ggml_set_input(engram_rows[index]);
                ggml_set_input(engram_select[index]);
                ggml_set_input(engram_row_ids[index]);
                const uint32_t layer = index == 0 ? 1 : 14;
                ggml_format_name(
                        engram_row_ids[index],
                        "dsv41.inp.engram.row_ids.l%u", layer);
            }
        }

        GGML_UNUSED(memory);
    }

    void set_input(const llama_ubatch * ubatch) override {
        const llama_dsv41_memory_plan & plan = mctx->plan();
        const llama_memory_dsv41 * memory = mctx->memory();
        const uint32_t n_tokens = ubatch->n_tokens;
        const uint32_t n_streams = plan.token_seq_ids.empty() ? 1 : plan.token_seq_ids.front().size();
        if (n_streams != 1) {
            throw std::runtime_error(
                    "DeepSeek V4.1 graph supports one active sequence per ubatch");
        }

        std::vector<float> pre((size_t) LLAMA_DSV41_HC_MULT*n_tokens, 0.0f);
        for (uint32_t token = 0; token < n_tokens; ++token) {
            pre[(size_t) token*LLAMA_DSV41_HC_MULT] = 1.0f;
        }
        set_tensor(initial_pre, pre);
        set_tensor(raw_persist_src_idxs, plan.raw.persist_src_idxs);
        set_tensor(raw_write_idxs, plan.raw.write_idxs);

        std::vector<int32_t> raw_idxs = plan.raw.read_idxs;
        const int32_t raw_cache_rows =
            (int32_t) (memory->config().raw_window*memory->config().n_seq);
        const int32_t raw_sentinel = raw_cache_rows + n_tokens;
        const llama_pos batch_start = plan.positions.front();
        for (uint32_t token = 0; token < n_tokens; ++token) {
            const uint32_t visible = plan.raw.n_visible[token];
            const llama_pos first = plan.positions[token] + 1 - visible;
            for (uint32_t row = 0; row < visible; ++row) {
                const llama_pos pos = first + row;
                if (pos >= batch_start) {
                    raw_idxs[(size_t) token*memory->config().raw_window + row] =
                        raw_cache_rows + pos - batch_start;
                }
            }
        }
        std::replace(raw_idxs.begin(), raw_idxs.end(), -1, raw_sentinel);
        set_tensor(raw_read_idxs, raw_idxs);
        set_mask(raw_mask, plan.raw.mask);

        for (const llama_dsv41_source_plan & source_plan : plan.sources) {
            dsv41_graph_source_input & source = sources.at(source_plan.source_layer);
            if (source.read_idxs != nullptr) {
                std::vector<int32_t> read_idxs = source_plan.read_idxs;
                const int32_t sentinel =
                    (int32_t) (source_plan.capacity*memory->config().n_seq);
                std::replace(read_idxs.begin(), read_idxs.end(), -1, sentinel);
                set_tensor(source.read_idxs, read_idxs);
                std::vector<float> mask(read_idxs.size());
                std::transform(
                        source_plan.read_idxs.begin(),
                        source_plan.read_idxs.end(),
                        mask.begin(),
                        [](int32_t idx) {
                            return idx >= 0 ? 0.0f : -std::numeric_limits<float>::infinity();
                        });
                set_mask(source.mask, mask);
            }
            if (source.state_read_idxs != nullptr) {
                std::vector<int32_t> carry_read_idxs(source.ratio);
                const llama_seq_id seq_id = plan.token_seq_ids.front().front();
                for (uint32_t row = 0; row < source.ratio; ++row) {
                    carry_read_idxs[row] = seq_id*source.ratio + row;
                }
                set_tensor(source.carry_read_idxs, carry_read_idxs);
                set_tensor(source.state_read_idxs, source_plan.compression.state_read_idxs);
            }
            if (source.state_persist_src_idxs != nullptr) {
                set_tensor(source.state_persist_src_idxs, source_plan.compression.state_persist_src_idxs);
                std::vector<int64_t> dst;
                dst.reserve(source_plan.compression.state_persist_dst_idxs.size()*n_streams);
                for (int32_t row : source_plan.compression.state_persist_dst_idxs) {
                    for (llama_seq_id seq_id : plan.token_seq_ids.front()) {
                        dst.push_back((int64_t) seq_id*source.ratio + row);
                    }
                }
                set_tensor(source.state_persist_dst_idxs, dst);
            }
            if (source.write_idxs != nullptr) {
                std::vector<int64_t> write_idxs;
                write_idxs.reserve(source_plan.compression.write_idxs.size()*n_streams);
                for (int64_t row : source_plan.compression.write_idxs) {
                    for (llama_seq_id seq_id : plan.token_seq_ids.front()) {
                        write_idxs.push_back((int64_t) seq_id*source.capacity + row);
                    }
                }
                set_tensor(source.write_idxs, write_idxs);
                set_tensor(source.write_pos, source_plan.compression.write_pos);
            }
            if (source.candidate_pad_mask != nullptr) {
                const uint32_t block_size = memory->config().candidate_block_size;
                const uint32_t n_blocks =
                    (source.read_width + block_size - 1)/block_size;
                const uint32_t padded = n_blocks*block_size;
                std::vector<float> pad((size_t) padded*n_tokens, 0.0f);
                for (uint32_t token = 0; token < n_tokens; ++token) {
                    for (uint32_t row = source.read_width; row < padded; ++row) {
                        pad[(size_t) token*padded + row] =
                            -std::numeric_limits<float>::infinity();
                    }
                }
                set_tensor(source.candidate_pad_mask, pad);

                std::vector<float> bias((size_t) n_blocks*n_tokens, 0.0f);
                std::vector<int32_t> final_blocks(n_tokens);
                for (uint32_t token = 0; token < n_tokens; ++token) {
                    const uint32_t visible =
                        source_plan.compression.n_visible[token];
                    const uint32_t visible_blocks =
                        (visible + block_size - 1)/block_size;
                    if (visible_blocks == 0) {
                        throw std::runtime_error(
                                "DeepSeek V4.1 candidate source has no visible block");
                    }
                    for (uint32_t block = visible_blocks; block < n_blocks; ++block) {
                        bias[(size_t) token*n_blocks + block] =
                            -std::numeric_limits<float>::infinity();
                    }
                    final_blocks[token] = (int32_t) visible_blocks - 1;
                }
                set_tensor(source.candidate_block_bias, bias);
                set_tensor(source.candidate_final_block, final_blocks);

                std::vector<int32_t> row_blocks(source.read_width);
                for (uint32_t row = 0; row < source.read_width; ++row) {
                    row_blocks[row] = row/block_size;
                }
                set_tensor(source.row_blocks, row_blocks);
            }
        }

        const llama_dsv41_engram_transaction * transaction = mctx->engram_transaction();
        for (uint32_t index = 0; index < LLAMA_ENGRAM_LAYERS; ++index) {
            if (engram_rows[index] == nullptr) {
                continue;
            }
            if (transaction == nullptr) {
                std::vector<float> zeros(
                        (size_t) LLAMA_ENGRAM_COLS*LLAMA_ENGRAM_DIM*n_tokens);
                std::vector<int32_t> select(n_tokens);
                std::vector<int32_t> ids((size_t) LLAMA_ENGRAM_COLS*n_tokens);
                set_tensor(engram_rows[index], zeros);
                set_tensor(engram_select[index], select);
                set_tensor(engram_row_ids[index], ids);
                continue;
            }
            transaction->upload_layer(
                    index, 0, n_tokens, engram_rows[index], engram_select[index]);
            set_tensor(
                    engram_row_ids[index],
                    mctx->engram_row_ids(index));
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        const auto * next = static_cast<const llama_memory_dsv41_context *>(params.mctx);
        const llama_dsv41_graph_topology next_topology =
            next->topology(params.ubatch, params.n_outputs);
        if (!topology.same_topology(next_topology)) {
            return false;
        }
        mctx = next;
        topology = next_topology;
        return true;
    }

    dsv41_graph_source_input * source(uint32_t layer) {
        const auto found = sources.find(layer);
        return found == sources.end() ? nullptr : &found->second;
    }

    const llama_cparams cparams;
    const llama_memory_dsv41_context * mctx;
    llama_dsv41_graph_topology topology;
    ggml_tensor * initial_pre = nullptr;
    ggml_tensor * raw_persist_src_idxs = nullptr;
    ggml_tensor * raw_write_idxs = nullptr;
    ggml_tensor * raw_read_idxs = nullptr;
    ggml_tensor * raw_mask = nullptr;
    std::map<uint32_t, dsv41_graph_source_input> sources;
    std::array<ggml_tensor *, LLAMA_ENGRAM_LAYERS> engram_rows = {};
    std::array<ggml_tensor *, LLAMA_ENGRAM_LAYERS> engram_select = {};
    std::array<ggml_tensor *, LLAMA_ENGRAM_LAYERS> engram_row_ids = {};

private:
    template <typename T>
    static void set_tensor(ggml_tensor * tensor, const std::vector<T> & values) {
        if (tensor == nullptr || tensor->buffer == nullptr) {
            return;
        }
        if (ggml_nelements(tensor) != (int64_t) values.size()) {
            throw std::runtime_error("DeepSeek V4.1 graph input shape changed");
        }
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size()*sizeof(T));
    }

    static void set_mask(ggml_tensor * tensor, const std::vector<float> & values) {
        if (tensor == nullptr || tensor->buffer == nullptr) {
            return;
        }
        if (ggml_nelements(tensor) != (int64_t) values.size()) {
            throw std::runtime_error("DeepSeek V4.1 graph mask shape changed");
        }
        if (tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> converted(values.size());
            ggml_fp32_to_fp16_row(values.data(), converted.data(), values.size());
            ggml_backend_tensor_set(
                    tensor, converted.data(), 0, converted.size()*sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(
                    tensor, values.data(), 0, values.size()*sizeof(float));
        }
    }
};

static ggml_tensor * dsv41_flatten_memory(
        ggml_context * ctx,
        ggml_tensor * tensor) {
    return ggml_reshape_2d(
            ctx, tensor, tensor->ne[0], tensor->ne[1]*tensor->ne[2]);
}

static ggml_tensor * dsv41_append_zero_row(
        ggml_context * ctx,
        ggml_tensor * tensor) {
    const ggml_type type = tensor->type;
    ggml_tensor * row = ggml_view_2d(
            ctx, tensor, tensor->ne[0], 1, tensor->nb[1], 0);
    if (row->type != GGML_TYPE_F32) {
        row = ggml_cast(ctx, row, GGML_TYPE_F32);
    }
    row = ggml_scale(ctx, row, 0.0f);
    if (type != GGML_TYPE_F32) {
        row = ggml_cast(ctx, row, type);
    }
    return ggml_concat(ctx, tensor, row, 1);
}

static ggml_tensor * dsv41_hc_mean(
        ggml_context * ctx,
        ggml_tensor * streams) {
    ggml_tensor * result = ggml_view_2d(
            ctx, streams, streams->ne[0], streams->ne[2], streams->nb[2], 0);
    for (int64_t stream = 1; stream < streams->ne[1]; ++stream) {
        result = ggml_add(ctx, result, ggml_view_2d(
                ctx, streams, streams->ne[0], streams->ne[2],
                streams->nb[2], stream*streams->nb[1]));
    }
    return ggml_scale(ctx, result, 1.0f/streams->ne[1]);
}

static ggml_tensor * dsv41_view_1d(
        ggml_context * ctx,
        ggml_tensor * tensor,
        int64_t ne0,
        int64_t offset) {
    return ggml_view_1d(ctx, tensor, ne0, ggml_row_size(tensor->type, offset));
}

static ggml_tensor * dsv41_view_2d(
        ggml_context * ctx,
        ggml_tensor * tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t offset) {
    return ggml_view_2d(
            ctx, tensor, ne0, ne1, tensor->nb[1],
            ggml_row_size(tensor->type, offset));
}

static ggml_tensor * dsv41_sort_row_ids(
        ggml_context * ctx,
        ggml_tensor * ids) {
    ggml_tensor * order = ggml_argsort(
            ctx, ggml_cast(ctx, ids, GGML_TYPE_F32),
            GGML_SORT_ORDER_ASC);
    ggml_tensor * sorted = ggml_get_rows(
            ctx,
            ggml_reshape_3d(ctx, ids, 1, ids->ne[0], ids->ne[1]),
            order);
    return ggml_cont(
            ctx,
            ggml_reshape_2d(ctx, sorted, ids->ne[0], ids->ne[1]));
}

}

void llama_model_deepseek41::load_arch_hparams(llama_model_loader & ml) {
    llama_dsv41_config config = {};
    std::string raw_config;
    ml.get_key(LLM_KV_DSV41_CONFIG, raw_config);
    ml.get_key(LLM_KV_DSV41_MAX_POSITION_EMBEDDINGS, config.n_ctx_train);
    ml.get_key(LLM_KV_DSV41_HIDDEN_SIZE, config.n_embd);
    ml.get_key(LLM_KV_DSV41_NUM_HIDDEN_LAYERS, config.n_layer);
    ml.get_key(LLM_KV_DSV41_VOCAB_SIZE, config.n_vocab);
    ml.get_key(LLM_KV_DSV41_NUM_ATTENTION_HEADS, config.n_head);
    ml.get_key(LLM_KV_DSV41_NUM_KEY_VALUE_HEADS, config.n_head_kv);
    ml.get_key(LLM_KV_DSV41_HEAD_DIM, config.n_head_dim);
    ml.get_key(LLM_KV_DSV41_QK_ROPE_HEAD_DIM, config.n_rot);
    ml.get_key(LLM_KV_DSV41_Q_LORA_RANK, config.n_lora_q);
    ml.get_key(LLM_KV_DSV41_O_LORA_RANK, config.n_lora_o);
    ml.get_key(LLM_KV_DSV41_O_GROUPS, config.n_o_group);
    ml.get_key(LLM_KV_DSV41_MOE_INTERMEDIATE_SIZE, config.n_ff_expert);
    ml.get_key(LLM_KV_DSV41_N_ROUTED_EXPERTS, config.n_expert);
    ml.get_key(LLM_KV_DSV41_NUM_EXPERTS_PER_TOK, config.n_expert_used);
    ml.get_key(LLM_KV_DSV41_N_SHARED_EXPERTS, config.n_expert_shared);
    ml.get_key(LLM_KV_DSV41_INDEX_N_HEADS, config.indexer_n_head);
    ml.get_key(LLM_KV_DSV41_INDEX_HEAD_DIM, config.indexer_head_size);
    ml.get_key(LLM_KV_DSV41_INDEX_TOPK, config.indexer_top_k);
    ml.get_key(LLM_KV_DSV41_HC_MULT, config.hc_count);
    ml.get_key(LLM_KV_DSV41_HC_SINKHORN_ITERS, config.hc_sinkhorn_iters);
    ml.get_key(LLM_KV_DSV41_SLIDING_WINDOW, config.raw_window);
    ml.get_key(LLM_KV_DSV41_CANDIDATE_SOURCE_LAYER_ID, config.candidate_source_layer);
    ml.get_key(LLM_KV_DSV41_CANDIDATE_TOPK_BLOCKS, config.candidate_topk_blocks);
    ml.get_key(LLM_KV_DSV41_CANDIDATE_BLOCK_SIZE, config.candidate_block_size);
    ml.get_key(LLM_KV_DSV41_RMS_NORM_EPS, config.f_norm_rms_eps);
    ml.get_key(LLM_KV_DSV41_HC_EPS, config.hc_eps);
    ml.get_key(LLM_KV_DSV41_SWIGLU_LIMIT, config.swiglu_clamp);
    ml.get_key(LLM_KV_DSV41_ROUTED_SCALING_FACTOR, config.routed_scale);
    ml.get_key(LLM_KV_DSV41_ROPE_THETA, config.rope_theta);
    ml.get_key(LLM_KV_DSV41_COMPRESS_ROPE_THETA, config.compress_rope_theta);
    ml.get_key(LLM_KV_DSV41_ROPE_SCALING_FACTOR, config.yarn_factor);
    ml.get_key(LLM_KV_DSV41_ROPE_SCALING_BETA_FAST, config.yarn_beta_fast);
    ml.get_key(LLM_KV_DSV41_ROPE_SCALING_BETA_SLOW, config.yarn_beta_slow);
    ml.get_key(LLM_KV_DSV41_ROPE_SCALING_ORIG_CTX_LEN, config.yarn_original_context);
    ml.get_key(LLM_KV_DSV41_NORM_TOPK_PROB, config.expert_weights_norm);
    ml.get_key(LLM_KV_DSV41_HIDDEN_ACT, config.hidden_act);
    ml.get_key(LLM_KV_DSV41_SCORING_FUNC, config.scoring_func);
    ml.get_key(LLM_KV_DSV41_TOPK_METHOD, config.topk_method);
    ml.get_arr(LLM_KV_DSV41_COMPRESS_RATIOS, config.compress_ratios);
    ml.get_arr(LLM_KV_DSV41_KV_SOURCE_LAYER_IDS, config.kv_sources);
    ml.get_arr(LLM_KV_DSV41_INDEX_SOURCE_LAYER_IDS, config.index_sources);
    ml.get_key(LLM_KV_DSV41_ENGRAM_ENCODING, config.engram_encoding);
    ml.get_arr(LLM_KV_DSV41_ENGRAM_LAYER_IDS, config.engram_layers);
    ml.get_arr(LLM_KV_DSV41_ENGRAM_ROWS, config.engram_rows);
    ml.get_key(LLM_KV_DSV41_ENGRAM_COMPRESSED_VOCAB_SIZE, config.engram_compressed_vocab_size);
    ml.get_key(LLM_KV_DSV41_ENGRAM_PAD_ID, config.engram_pad_id);
    ml.get_arr_n(LLM_KV_DSV41_ENGRAM_TOKEN_MAP, config.engram_token_map_size);
    ml.get_arr_n(LLM_KV_DSV41_ENGRAM_PRIMES, config.engram_primes_size);
    ml.get_arr_n(LLM_KV_DSV41_ENGRAM_MULTIPLIERS, config.engram_multipliers_size);
    ml.get_arr(LLM_KV_DSV41_ENGRAM_TOKEN_MAP, config.engram_token_map);
    ml.get_arr(LLM_KV_DSV41_ENGRAM_PRIMES, config.engram_primes);
    ml.get_arr(LLM_KV_DSV41_ENGRAM_MULTIPLIERS, config.engram_multipliers);

    config.n_ff_dense = LLAMA_DSV41_N_FF_DENSE;
    llama_dsv41_validate_config(config);
    engram = std::make_shared<engram_model>();
    engram->layout = llama_dsv41_make_engram_layout(config);

    if (raw_config.empty()) {
        throw std::runtime_error("DeepSeek V4.1 metadata: config must not be empty");
    }
    hparams.n_ctx_train = config.n_ctx_train;
    hparams.n_embd = config.n_embd;
    hparams.n_embd_out_impl = config.n_embd;
    hparams.n_layer_all = config.n_layer;
    hparams.n_layer_nextn = 0;
    hparams.n_expert = config.n_expert;
    hparams.n_expert_shared = config.n_expert_shared;
    hparams.n_lora_q = config.n_lora_q;
    hparams.n_ff_shexp = config.n_ff_expert;
    hparams.n_embd_head_k_full = config.n_head_dim;
    hparams.n_embd_head_v_full = config.n_head_dim;
    hparams.n_embd_head_k_swa = config.n_head_dim;
    hparams.n_embd_head_v_swa = config.n_head_dim;
    hparams.n_rot_full = config.n_rot;
    hparams.n_rot_swa = config.n_rot;
    hparams.n_swa = config.raw_window;
    hparams.indexer_n_head = config.indexer_n_head;
    hparams.indexer_head_size = config.indexer_head_size;
    hparams.indexer_top_k = config.indexer_top_k;
    hparams.dsv4_o_group_count = config.n_o_group;
    hparams.dsv4_o_lora_rank = config.n_lora_o;
    hparams.dsv4_hc_mult = config.hc_count;
    hparams.dsv4_hc_sinkhorn_iters = config.hc_sinkhorn_iters;
    hparams.dsv4_compress_rope_base = (float) config.compress_rope_theta;
    hparams.dsv4_hc_eps = config.hc_eps;
    hparams.dsv41_candidate_source_layer = config.candidate_source_layer;
    hparams.dsv41_candidate_topk_blocks = config.candidate_topk_blocks;
    hparams.dsv41_candidate_block_size = config.candidate_block_size;
    hparams.f_norm_rms_eps = config.f_norm_rms_eps;
    hparams.expert_weights_scale = config.routed_scale;
    hparams.expert_weights_norm = config.expert_weights_norm;
    hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS;
    hparams.rope_freq_base_train = (float) config.rope_theta;
    hparams.rope_freq_base_train_swa = (float) config.rope_theta;
    hparams.rope_freq_scale_train = 1.0f/config.yarn_factor;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    hparams.n_ctx_orig_yarn = (uint32_t) config.yarn_original_context;
    hparams.yarn_beta_fast = config.yarn_beta_fast;
    hparams.yarn_beta_slow = config.yarn_beta_slow;
    hparams.yarn_ext_factor = 1.0f;
    hparams.rope_attn_factor = dsv41_rope_attn_factor(hparams.rope_freq_scale_train);
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.causal_attn = true;

    for (uint32_t il = 0; il < config.n_layer; ++il) {
        hparams.n_head_arr[il] = config.n_head;
        hparams.n_head_kv_arr[il] = config.n_head_kv;
        hparams.n_ff_arr[il] = config.n_ff_dense;
        hparams.n_ff_exp_arr[il] = config.n_ff_expert;
        hparams.n_expert_used_arr[il] = config.n_expert_used;
        hparams.swiglu_clamp_exp[il] = config.swiglu_clamp;
        hparams.swiglu_clamp_shexp[il] = config.swiglu_clamp;
        hparams.dsv4_compress_ratios[il] = config.compress_ratios[il];
        hparams.dsv41_kv_source_layer[il] = llama_dsv41_kv_source_layer(il);
        hparams.dsv41_index_source_layer[il] = llama_dsv41_index_source_layer(il);
        hparams.is_swa_impl[il] = 1;
    }
    for (uint32_t il : config.engram_layers) {
        hparams.dsv41_engram_layers.set(il);
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_deepseek41::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t n_embd_head     = hparams.n_embd_head_k();
    const int64_t o_groups        = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank     = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult         = hparams.dsv4_hc_mult;
    const int64_t hc_dim          = hc_mult*n_embd;
    const int64_t hc_mix_dim      = (2 + hc_mult)*hc_mult;

    const std::vector<llama_expert_store_tensor> expert_tensors =
        llama_dsv41_register_expert_tensors([&](const std::string & name,
                                                int32_t layer,
                                                llama_expert_projection projection,
                                                const std::initializer_list<int64_t> & ne) {
            return ml.register_external_tensor(name, layer, projection, ne);
        });
    if (params.expert_cache_bytes == 0 || params.expert_cache_slots <= 0) {
        throw std::runtime_error(
                "DeepSeek V4.1 requires non-zero expert_cache_bytes and expert_cache_slots before tensor allocation");
    }
    llama_dsv41_expert_runtime_params expert_params;
    expert_params.cache_bytes = params.expert_cache_bytes;
    expert_params.cache_slots = params.expert_cache_slots;
    expert_params.direct_io = true;
    expert_params.allow_buffered_io = false;
    expert_params.no_alloc = ml.no_alloc;
    experts = std::make_shared<llama_dsv41_expert_runtime>(
            expert_tensors,
            expert_params,
            [this](const llama_expert_store_tensor & tensor) {
                return select_moe_buft(
                        tensor.layer, tensor.type, tensor.ne[0], tensor.ne[1], params.expert_cache_slots);
            });

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

    for (int32_t il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), { n_embd }, 0);
        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", il), { n_head }, 0);
        layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", il), { n_embd, q_lora_rank }, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", il), { q_lora_rank }, 0);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", il), { q_lora_rank, n_head*n_embd_head }, 0);
        layer.wkv = create_tensor(tn(LLM_TENSOR_ATTN_KV, "weight", il), { n_embd, n_embd_head }, 0);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", il), { n_embd_head }, 0);
        layer.wo_a = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A, "weight", il), { n_head*n_embd_head/o_groups, o_lora_rank, o_groups }, TENSOR_ALLOW_RESHAPE);
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B, "weight", il), { o_groups*o_lora_rank, n_embd }, 0);

        layer.hc_attn_fn = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN, "weight", il), { hc_dim, hc_mix_dim }, 0);
        layer.hc_attn_base = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE, "weight", il), { hc_mix_dim }, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", il), { 3 }, 0);
        layer.hc_ffn_fn = create_tensor(tn(LLM_TENSOR_HC_FFN_FN, "weight", il), { hc_dim, hc_mix_dim }, 0);
        layer.hc_ffn_base = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE, "weight", il), { hc_mix_dim }, 0);
        layer.hc_ffn_scale = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE, "weight", il), { 3 }, 0);

        if (hparams.dsv41_is_kv_source(il)) {
            layer.attn_comp_wkv = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV, "weight", il), { n_embd, n_embd_head }, 0);
            layer.attn_comp_norm = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM, "weight", il), { n_embd_head }, 0);
            if (hparams.dsv4_compress_ratios[il] == 2) {
                layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", il), { n_embd, n_embd_head }, 0);
            }
            layer.indexer_attn_k = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K, "weight", il), { n_embd_head, hparams.indexer_head_size }, 0);
            layer.indexer_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { hparams.indexer_head_size }, 0);
        }
        if (hparams.dsv41_is_index_source(il)) {
            layer.indexer_proj = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ, "weight", il), { n_embd, hparams.indexer_n_head }, 0);
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", il), { q_lora_rank, hparams.indexer_n_head*hparams.indexer_head_size }, 0);
        }

        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", il), { n_embd, n_expert }, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", il), { n_expert }, 0);
        layer.ffn_exp_probs_b_vl = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", il), { n_expert }, TENSOR_NOT_REQUIRED);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", il), { n_embd }, 0);
        layer.ffn_gate_exps = experts->cache_tensor(il, LLAMA_EXPERT_PROJECTION_GATE);
        layer.ffn_down_exps = experts->cache_tensor(il, LLAMA_EXPERT_PROJECTION_DOWN);
        layer.ffn_up_exps = experts->cache_tensor(il, LLAMA_EXPERT_PROJECTION_UP);
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", il), { n_embd, n_ff_exp*n_expert_shared }, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", il), { n_ff_exp*n_expert_shared, n_embd }, 0);
        layer.ffn_up_shexp = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", il), { n_embd, n_ff_exp*n_expert_shared }, 0);

        if (hparams.dsv41_engram_layers.test(il)) {
            const size_t index = il == (int32_t) engram->layout.layer_ids[0] ? 0 : 1;
            const std::string table_name = tn(LLM_TENSOR_ENGRAM_EMBD, "weight", il).str();
            const auto * table = ml.get_weight(table_name.c_str());
            if (table == nullptr) {
                throw std::runtime_error("DeepSeek V4.1 is missing required Engram tensor " + table_name);
            }
            llama_dsv41_engram_extent & extent = engram->extents[index];
            extent.fname = ml.no_alloc ? "(no_alloc)" : ml.fnames.at(table->idx);
            extent.offset = table->offs;
            extent.rows = engram->layout.rows[index];
            extent.columns = table->tensor->ne[0];
            extent.row_count = table->tensor->ne[1];
            extent.type = table->tensor->type;
            llama_dsv41_validate_engram_extent(extent);

            create_tensor(
                    tn(LLM_TENSOR_ENGRAM_EMBD, "weight", il),
                    { LLAMA_ENGRAM_ROW_BYTES, (int64_t) extent.rows },
                    TENSOR_SKIP);
            layer.engram_q_norm = create_tensor(
                    tn(LLM_TENSOR_ENGRAM_Q_NORM, "weight", il),
                    { n_embd, hc_mult },
                    0);
            layer.engram_k_norm = create_tensor(
                    tn(LLM_TENSOR_ENGRAM_K_NORM, "weight", il),
                    { n_embd, hc_mult },
                    0);
            layer.engram_kv = create_tensor(
                    tn(LLM_TENSOR_ENGRAM_KV, "weight", il),
                    { LLAMA_ENGRAM_COLS*LLAMA_ENGRAM_DIM, (hc_mult + 1)*n_embd },
                    0);
        }
    }

}

bool llama_model_deepseek41::requires_synchronous_graph() const {
    return experts != nullptr;
}

std::string llama_model_deepseek41::consume_runtime_error() const {
    return experts ? experts->consume_error() : std::string();
}

void llama_model_deepseek41::release_runtime_work() const {
    if (experts) {
        experts->release_all();
    }
}

void llama_model_deepseek41::release_runtime_work_after_sync(ggml_backend_sched_t sched) const {
    if (experts) {
        experts->release_all_after_sync(sched);
    }
}

void llama_model_deepseek41::acquire_runtime_context() const {
    if (experts) {
        experts->acquire_context();
    }
}

void llama_model_deepseek41::release_runtime_context() const {
    if (experts) {
        experts->release_context();
    }
}

namespace {

struct dsv41_hc_mix {
    ggml_tensor * pre;
    ggml_tensor * post;
    ggml_tensor * comb;
};

static dsv41_hc_mix dsv41_build_hc_mix(
        const llama_model_deepseek41::graph & graph,
        ggml_tensor * streams,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        int il) {
    const int64_t hc = graph.hparams.dsv4_hc_mult;
    const int64_t nt = streams->ne[2];
    ggml_tensor * flat = ggml_reshape_2d(graph.ctx0, streams, graph.n_embd*hc, nt);
    flat = ggml_rms_norm(graph.ctx0, flat, graph.norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(graph.ctx0, hc_fn, flat);
    graph.cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre = dsv41_view_1d(graph.ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = dsv41_view_1d(graph.ctx0, hc_scale, 1, 1);
    ggml_tensor * scale_comb = dsv41_view_1d(graph.ctx0, hc_scale, 1, 2);
    ggml_tensor * base_pre = dsv41_view_1d(graph.ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = dsv41_view_1d(graph.ctx0, hc_base, hc, hc);
    ggml_tensor * base_comb = dsv41_view_1d(graph.ctx0, hc_base, hc*hc, 2*hc);

    ggml_tensor * pre = dsv41_view_2d(graph.ctx0, mixes, hc, nt, 0);
    pre = ggml_add(graph.ctx0, ggml_mul(graph.ctx0, pre, scale_pre), base_pre);
    pre = ggml_scale_bias(
            graph.ctx0, ggml_sigmoid(graph.ctx0, pre), 1.0f,
            graph.hparams.dsv4_hc_eps);
    graph.cb(pre, "hc_pre", il);

    ggml_tensor * post = dsv41_view_2d(graph.ctx0, mixes, hc, nt, hc);
    post = ggml_add(graph.ctx0, ggml_mul(graph.ctx0, post, scale_post), base_post);
    post = ggml_scale(graph.ctx0, ggml_sigmoid(graph.ctx0, post), 2.0f);
    graph.cb(post, "hc_post", il);

    ggml_tensor * comb = nullptr;
    if (graph.cparams.fused_dsv4_hc_comb) {
        comb = ggml_dsv4_hc_comb(
                graph.ctx0, mixes, hc_scale, hc_base,
                graph.hparams.dsv4_hc_eps,
                (int32_t) graph.hparams.dsv4_hc_sinkhorn_iters);
        graph.res->add_fused_node({LLM_FUSED_OP_DSV4_HC_COMB, comb, il});
    } else {
        comb = dsv41_view_2d(graph.ctx0, mixes, hc*hc, nt, 2*hc);
        comb = ggml_add(graph.ctx0, ggml_mul(graph.ctx0, comb, scale_comb), base_comb);
        comb = ggml_reshape_3d(graph.ctx0, comb, hc, hc, nt);
        comb = graph.build_hc_sinkhorn(comb, il);
    }
    graph.cb(comb, "hc_comb", il);
    return { pre, post, comb };
}

static ggml_tensor * dsv41_cast_for_store(
        ggml_context * ctx,
        ggml_tensor * source,
        const ggml_tensor * destination) {
    return source->type == destination->type ?
        source : ggml_cast(ctx, source, destination->type);
}

static ggml_tensor * dsv41_concat(
        ggml_context * ctx,
        ggml_tensor * a,
        ggml_tensor * b,
        int dim,
        const char * label) {
    if (a == nullptr || b == nullptr || a->type != b->type) {
        throw std::runtime_error(format(
                "DeepSeek V4.1 %s concat type mismatch: %s and %s",
                label,
                a == nullptr ? "null" : ggml_type_name(a->type),
                b == nullptr ? "null" : ggml_type_name(b->type)));
    }
    return ggml_concat(ctx, a, b, dim);
}

static ggml_tensor * dsv41_completion_zero(
        ggml_context * ctx,
        ggml_tensor * dependency,
        ggml_type type) {
    ggml_tensor * marker = ggml_view_1d(ctx, dependency, 1, 0);
    marker = ggml_argsort_top_k(ctx, marker, 1);
    marker = ggml_cast(ctx, marker, GGML_TYPE_F32);
    marker = ggml_scale(ctx, marker, 0.0f);
    return type == GGML_TYPE_F32 ? marker : ggml_cast(ctx, marker, type);
}

static ggml_tensor * dsv41_build_candidate_mask(
        const llama_model_deepseek41::graph & graph,
        ggml_tensor * candidate_blocks,
        const dsv41_graph_source_input & source) {
    const int64_t n_blocks = source.candidate_block_bias->ne[0];
    const int64_t n_tokens = candidate_blocks->ne[1];
    ggml_tensor * all = ggml_new_tensor_3d(
            graph.ctx0, GGML_TYPE_F32, 1, n_blocks, n_tokens);
    all = ggml_fill(graph.ctx0, all, -INFINITY);
    ggml_tensor * zeros = ggml_scale(
            graph.ctx0, ggml_cast(graph.ctx0, candidate_blocks, GGML_TYPE_F32), 0.0f);
    zeros = ggml_reshape_3d(
            graph.ctx0, zeros, 1, candidate_blocks->ne[0], n_tokens);
    ggml_tensor * blocks = ggml_set_rows(graph.ctx0, all, zeros, candidate_blocks);
    blocks = ggml_reshape_2d(graph.ctx0, blocks, n_blocks, n_tokens);
    blocks = ggml_cont(graph.ctx0, ggml_transpose(graph.ctx0, blocks));
    ggml_tensor * rows = ggml_get_rows(graph.ctx0, blocks, source.row_blocks);
    return ggml_cont(graph.ctx0, ggml_transpose(graph.ctx0, rows));
}

static ggml_tensor * dsv41_build_index_selection(
        const llama_model_deepseek41::graph & graph,
        const llama_model & model,
        dsv41_graph_input & input,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * & candidate_blocks,
        ggml_tensor * & selected_local,
        int il) {
    const auto & layer = model.layers[il];
    const uint32_t source_layer = graph.hparams.dsv41_kv_source_layer[il];
    dsv41_graph_source_input * source = input.source(source_layer);
    if (source == nullptr || source->read_width == 0) {
        return nullptr;
    }

    const llama_memory_dsv41 * memory = input.mctx->memory();
    ggml_tensor * index_cache = dsv41_flatten_memory(
            graph.ctx0, memory->index_keys(source_layer));
    index_cache = dsv41_append_zero_row(graph.ctx0, index_cache);
    ggml_tensor * index_k = ggml_get_rows(
            graph.ctx0, index_cache,
            ggml_reshape_1d(
                graph.ctx0, source->read_idxs,
                source->read_width*graph.n_tokens));
    index_k = ggml_reshape_3d(
            graph.ctx0, index_k,
            graph.hparams.indexer_head_size,
            source->read_width, graph.n_tokens);
    graph.cb(index_k, "dsv41_index_k", il);

    ggml_tensor * index_q = graph.build_lora_mm(layer.indexer_attn_q_b, qr);
    index_q = ggml_reshape_3d(
            graph.ctx0, index_q,
            graph.hparams.indexer_head_size,
            graph.hparams.indexer_n_head,
            graph.n_tokens);
    index_q = ggml_rope_ext(
            graph.ctx0, index_q, inp_pos, nullptr, graph.hparams.n_rot(),
            graph.rope_type, graph.n_ctx_orig,
            graph.hparams.dsv4_compress_rope_base,
            graph.freq_scale, graph.ext_factor,
            dsv41_rope_attn_factor(graph.freq_scale),
            graph.beta_fast, graph.beta_slow);
    index_q = ggml_rope_set_offset(
            index_q, graph.hparams.indexer_head_size - graph.hparams.n_rot());
    graph.cb(index_q, "dsv41_index_q", il);

    ggml_tensor * weights = graph.build_lora_mm(layer.indexer_proj, cur);
    weights = ggml_scale(
            graph.ctx0, weights,
            1.0f/std::sqrt(
                (float) (graph.hparams.indexer_head_size*graph.hparams.indexer_n_head)));
    weights = ggml_reshape_3d(
            graph.ctx0, weights, graph.hparams.indexer_n_head, 1, graph.n_tokens);

    ggml_tensor * scores = ggml_mul_mat(graph.ctx0, index_k, index_q);
    ggml_prec_set_acc(scores, GGML_PREC_F32);
    scores = ggml_relu(graph.ctx0, scores);
    scores = ggml_cont(graph.ctx0, ggml_permute(graph.ctx0, scores, 1, 0, 2, 3));
    scores = ggml_mul(graph.ctx0, scores, weights);
    scores = ggml_sum_rows(graph.ctx0, scores);
    scores = ggml_reshape_2d(graph.ctx0, scores, source->read_width, graph.n_tokens);
    scores = ggml_add(
            graph.ctx0, scores,
            ggml_reshape_2d(
                graph.ctx0, source->mask, source->read_width, graph.n_tokens));

    if (candidate_blocks != nullptr) {
        scores = ggml_add(
                graph.ctx0, scores,
                dsv41_build_candidate_mask(graph, candidate_blocks, *source));
    }
    graph.cb(scores, "dsv41_index_scores", il);

    if ((uint32_t) il == graph.hparams.dsv41_candidate_source_layer) {
        const uint32_t block_size = graph.hparams.dsv41_candidate_block_size;
        const int64_t padded = source->candidate_pad_mask->ne[0];
        ggml_tensor * padded_scores = ggml_pad(
                graph.ctx0, scores, padded - source->read_width, 0, 0, 0);
        padded_scores = ggml_add(
                graph.ctx0, padded_scores, source->candidate_pad_mask);
        ggml_tensor * block_scores = ggml_pool_1d(
                graph.ctx0, padded_scores, GGML_OP_POOL_MAX,
                block_size, block_size, 0);
        block_scores = ggml_add(
                graph.ctx0, block_scores, source->candidate_block_bias);
        graph.cb(block_scores, "dsv41_candidate_scores", il);

        const uint32_t n_candidate = input.topology.candidate_width;
        candidate_blocks = llama_dsv41_build_candidate_blocks(
                graph.ctx0, block_scores,
                source->candidate_final_block, n_candidate);
        ggml_set_name(
                candidate_blocks,
                llama_dsv41_graph_trace_name(
                    "attn.candidate_blocks", il).c_str());
        ggml_build_forward_expand(graph.gf, candidate_blocks);

        ggml_tensor * score_store = memory->candidate_scores();
        ggml_tensor * score_view = ggml_view_2d(
                graph.ctx0, score_store, block_scores->ne[0], graph.n_tokens,
                score_store->nb[1], 0);
        ggml_build_forward_expand(
                graph.gf, ggml_cpy(graph.ctx0, block_scores, score_view));
        ggml_tensor * id_store = memory->candidate_ids();
        ggml_tensor * id_view = ggml_view_2d(
                graph.ctx0, id_store, candidate_blocks->ne[0], graph.n_tokens,
                id_store->nb[1], 0);
        ggml_build_forward_expand(
                graph.gf, ggml_cpy(graph.ctx0, candidate_blocks, id_view));
    }

    const uint32_t n_top_k = std::min<uint32_t>(
            source->read_width, graph.hparams.indexer_top_k);
    selected_local = ggml_cont(
            graph.ctx0,
            ggml_argsort_top_k(graph.ctx0, scores, n_top_k));
    selected_local = dsv41_sort_row_ids(
            graph.ctx0, selected_local);
    ggml_tensor * selected = ggml_get_rows(
            graph.ctx0,
            ggml_reshape_3d(
                graph.ctx0, source->read_idxs, 1,
                source->read_width, graph.n_tokens),
            selected_local);
    selected = ggml_cont(
            graph.ctx0,
            ggml_reshape_2d(
                graph.ctx0, selected,
                n_top_k, graph.n_tokens));
    if ((uint32_t) il > graph.hparams.dsv41_candidate_source_layer) {
        ggml_set_name(
                selected,
                llama_dsv41_graph_trace_name(
                    "attn.candidates", il).c_str());
        ggml_build_forward_expand(graph.gf, selected);
    }
    return selected;
}

static ggml_tensor * dsv41_build_attention(
        const llama_model_deepseek41::graph & graph,
        const llama_model & model,
        dsv41_graph_input & input,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * & selected,
        ggml_tensor * & selected_local,
        ggml_tensor * & candidate_blocks,
        int il) {
    const auto & layer = model.layers[il];
    const int64_t n_embd_head = graph.hparams.n_embd_head_k();
    const int64_t n_rot = graph.hparams.n_rot();
    const int64_t n_nope = n_embd_head - n_rot;
    const int64_t ratio = graph.hparams.dsv4_compress_ratios[il];
    const int64_t n_groups = graph.hparams.dsv4_o_group_count;
    const int64_t heads_per_group = graph.n_head/n_groups;
    const int64_t group_width = heads_per_group*n_embd_head;
    const int64_t o_lora_rank = graph.hparams.dsv4_o_lora_rank;
    const float rope_base = ratio == 0 ?
        graph.freq_base : graph.hparams.dsv4_compress_rope_base;
    const float rope_scale = ratio == 0 ? 1.0f : graph.freq_scale;
    const float rope_ext = ratio == 0 ? 0.0f : graph.ext_factor;
    const float rope_attn = ratio == 0 ?
        1.0f : dsv41_rope_attn_factor(graph.freq_scale);
    const float rope_beta_fast = ratio == 0 ? 0.0f : graph.beta_fast;
    const float rope_beta_slow = ratio == 0 ? 0.0f : graph.beta_slow;
    const int32_t rope_ctx = ratio == 0 ? 0 : graph.n_ctx_orig;

    ggml_tensor * qr = graph.build_lora_mm(layer.wq_a, cur);
    qr = graph.build_norm(
            qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    graph.cb(qr, "dsv41_qr", il);

    ggml_tensor * q = graph.build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(
            graph.ctx0, q, n_embd_head, graph.n_head, graph.n_tokens);
    q = ggml_rms_norm(graph.ctx0, q, graph.norm_rms_eps);
    q = ggml_rope_ext(
            graph.ctx0, q, inp_pos, nullptr, n_rot, graph.rope_type,
            rope_ctx, rope_base, rope_scale, rope_ext, rope_attn,
            rope_beta_fast, rope_beta_slow);
    q = ggml_rope_set_offset(q, n_nope);
    graph.cb(q, "dsv41_q", il);

    ggml_tensor * kv = graph.build_lora_mm(layer.wkv, cur);
    kv = graph.build_norm(
            kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(
            graph.ctx0, kv, n_embd_head, 1, graph.n_tokens);
    ggml_tensor * raw_kv = ggml_rope_ext(
            graph.ctx0, kv, inp_pos, nullptr, n_rot, graph.rope_type,
            rope_ctx, rope_base, rope_scale, rope_ext, rope_attn,
            rope_beta_fast, rope_beta_slow);
    raw_kv = ggml_rope_set_offset(raw_kv, n_nope);
    graph.cb(raw_kv, "dsv41_raw_kv", il);

    const llama_memory_dsv41 * memory = input.mctx->memory();
    ggml_tensor * raw_store = dsv41_flatten_memory(
            graph.ctx0, memory->raw_k(il));
    ggml_tensor * raw_write = ggml_reshape_2d(
            graph.ctx0, raw_kv, n_embd_head, graph.n_tokens);
    ggml_tensor * raw_persist = ggml_get_rows(
            graph.ctx0, raw_write, input.raw_persist_src_idxs);
    raw_write = dsv41_cast_for_store(
            graph.ctx0, raw_write, raw_store);

    ggml_tensor * raw_read = dsv41_concat(
            graph.ctx0, raw_store, raw_write, 1, "raw state");
    raw_read = dsv41_append_zero_row(graph.ctx0, raw_read);
    raw_read = ggml_get_rows(
            graph.ctx0, raw_read,
            ggml_reshape_1d(
                graph.ctx0, input.raw_read_idxs,
                graph.hparams.n_swa*graph.n_tokens));
    raw_read = ggml_reshape_4d(
            graph.ctx0, raw_read, n_embd_head, 1,
            graph.hparams.n_swa, graph.n_tokens);

    const int32_t kv_source = graph.hparams.dsv41_kv_source_layer[il];
    dsv41_graph_source_input * source =
        kv_source >= 0 ? input.source(kv_source) : nullptr;
    if (source != nullptr && source->layer == (uint32_t) il) {
        ggml_tensor * compressed = graph.build_lora_mm(
                layer.attn_comp_wkv, cur);
        ggml_tensor * gate = ratio == 2 ?
            graph.build_lora_mm(layer.attn_comp_wgate, cur) : nullptr;
        ggml_tensor * compressed_current = compressed;
        ggml_tensor * gate_current = gate;
        ggml_tensor * carry_kv = nullptr;
        ggml_tensor * carry_gate = nullptr;
        ggml_tensor * persist_kv = nullptr;
        ggml_tensor * persist_gate = nullptr;

        if (ratio == 2) {
            carry_kv = dsv41_flatten_memory(
                    graph.ctx0, memory->compressor_carry_kv(source->layer));
            carry_gate = dsv41_flatten_memory(
                    graph.ctx0, memory->compressor_carry_score(source->layer));
            if (source->write_idxs != nullptr) {
                ggml_tensor * carry_kv_first = ggml_get_rows(
                        graph.ctx0, carry_kv, source->carry_read_idxs);
                ggml_tensor * carry_gate_first = ggml_get_rows(
                        graph.ctx0, carry_gate, source->carry_read_idxs);
                carry_kv_first = dsv41_cast_for_store(
                        graph.ctx0, carry_kv_first, compressed);
                carry_gate_first = dsv41_cast_for_store(
                        graph.ctx0, carry_gate_first, gate);
                ggml_tensor * source_kv = dsv41_concat(
                        graph.ctx0, carry_kv_first, compressed, 1, "compressor carry KV");
                ggml_tensor * source_gate = dsv41_concat(
                        graph.ctx0, carry_gate_first, gate, 1, "compressor carry score");
                source_kv = ggml_get_rows(
                        graph.ctx0, source_kv, source->state_read_idxs);
                source_gate = ggml_get_rows(
                        graph.ctx0, source_gate, source->state_read_idxs);
                const int64_t n_write = source->write_pos->ne[0];
                source_kv = ggml_reshape_3d(
                        graph.ctx0, source_kv, n_embd_head, ratio, n_write);
                source_gate = ggml_reshape_3d(
                        graph.ctx0, source_gate, n_embd_head, ratio, n_write);
                compressed = llama_dsv41_build_ratio_pool(
                        graph.ctx0, source_kv, source_gate, ratio);
            } else {
                compressed = nullptr;
            }

            persist_kv = ggml_get_rows(
                    graph.ctx0, compressed_current,
                    source->state_persist_src_idxs);
            persist_gate = ggml_get_rows(
                    graph.ctx0, gate_current,
                    source->state_persist_src_idxs);
        } else {
            compressed = llama_dsv41_build_ratio_pool(
                    graph.ctx0,
                    ggml_reshape_3d(
                        graph.ctx0, compressed, n_embd_head, 1,
                        graph.n_tokens),
                    nullptr, 1);
        }

        if (compressed != nullptr) {
            compressed = graph.build_norm(
                    compressed, layer.attn_comp_norm, nullptr,
                    LLM_NORM_RMS, il);
            graph.cb(compressed, "dsv41_compressed_unrotated", il);

            ggml_tensor * index_k = graph.build_lora_mm(
                    layer.indexer_attn_k, compressed);
            index_k = graph.build_norm(
                    index_k, layer.indexer_k_norm, nullptr,
                    LLM_NORM_RMS, il);
            index_k = ggml_reshape_3d(
                    graph.ctx0, index_k,
                    graph.hparams.indexer_head_size, 1, index_k->ne[1]);
            index_k = ggml_rope_ext(
                    graph.ctx0, index_k, source->write_pos, nullptr,
                    graph.hparams.n_rot(), graph.rope_type, graph.n_ctx_orig,
                    graph.hparams.dsv4_compress_rope_base,
                    graph.freq_scale, graph.ext_factor,
                    dsv41_rope_attn_factor(graph.freq_scale),
                    graph.beta_fast, graph.beta_slow);
            index_k = ggml_rope_set_offset(
                    index_k,
                    graph.hparams.indexer_head_size - graph.hparams.n_rot());

            ggml_tensor * compressed_rope = ggml_reshape_3d(
                    graph.ctx0, compressed, n_embd_head, 1, compressed->ne[1]);
            compressed_rope = ggml_rope_ext(
                    graph.ctx0, compressed_rope, source->write_pos, nullptr,
                    n_rot, graph.rope_type, graph.n_ctx_orig,
                    graph.hparams.dsv4_compress_rope_base,
                    graph.freq_scale, graph.ext_factor,
                    dsv41_rope_attn_factor(graph.freq_scale),
                    graph.beta_fast, graph.beta_slow);
            compressed_rope = ggml_rope_set_offset(compressed_rope, n_nope);

            ggml_tensor * comp_store = dsv41_flatten_memory(
                    graph.ctx0, memory->compressed_kv(source->layer));
            ggml_tensor * index_store = dsv41_flatten_memory(
                    graph.ctx0, memory->index_keys(source->layer));
            ggml_tensor * comp_write = ggml_reshape_2d(
                    graph.ctx0, compressed_rope, n_embd_head,
                    compressed_rope->ne[2]);
            ggml_tensor * index_write = ggml_reshape_2d(
                    graph.ctx0, index_k, graph.hparams.indexer_head_size,
                    index_k->ne[2]);
            comp_write = dsv41_cast_for_store(
                    graph.ctx0, comp_write, comp_store);
            index_write = dsv41_cast_for_store(
                    graph.ctx0, index_write, index_store);
            ggml_build_forward_expand(
                    graph.gf, ggml_set_rows(
                        graph.ctx0, comp_store, comp_write,
                        source->write_idxs));
            ggml_build_forward_expand(
                    graph.gf, ggml_set_rows(
                        graph.ctx0, index_store, index_write,
                        source->write_idxs));
        }

        if (persist_kv != nullptr) {
            if (compressed != nullptr) {
                ggml_tensor * completion = dsv41_completion_zero(
                        graph.ctx0, compressed, persist_kv->type);
                persist_kv = ggml_add(
                        graph.ctx0, persist_kv, completion);
                persist_gate = ggml_add(
                        graph.ctx0, persist_gate, completion);
            }
            persist_kv = dsv41_cast_for_store(
                    graph.ctx0, persist_kv, carry_kv);
            persist_gate = dsv41_cast_for_store(
                    graph.ctx0, persist_gate, carry_gate);
            ggml_build_forward_expand(
                    graph.gf, ggml_set_rows(
                        graph.ctx0, carry_kv, persist_kv,
                        source->state_persist_dst_idxs));
            ggml_build_forward_expand(
                    graph.gf, ggml_set_rows(
                        graph.ctx0, carry_gate, persist_gate,
                        source->state_persist_dst_idxs));
        }
    }

    if (graph.hparams.dsv41_is_index_source(il)) {
        selected = dsv41_build_index_selection(
                graph, model, input, qr, cur, inp_pos,
                candidate_blocks, selected_local,
                il);
    }

    ggml_tensor * source_trace = nullptr;
    if (selected != nullptr) {
        source_trace = ggml_cont(graph.ctx0, selected);
        ggml_set_name(
                source_trace,
                llama_dsv41_graph_trace_name(
                    "attn.source", il).c_str());
    } else {
        source_trace = ggml_cont(graph.ctx0, input.raw_read_idxs);
        ggml_set_name(
                source_trace,
                llama_dsv41_graph_trace_name(
                    "attn.source", il).c_str());
    }
    ggml_build_forward_expand(graph.gf, source_trace);

    ggml_tensor * k_all = raw_read;
    ggml_tensor * mask_all = input.raw_mask;
    int64_t n_kv_max = graph.hparams.n_swa;
    if (selected != nullptr) {
        ggml_tensor * comp_store = dsv41_flatten_memory(
                graph.ctx0, memory->compressed_kv(kv_source));
        comp_store = dsv41_append_zero_row(
                graph.ctx0, comp_store);
        ggml_tensor * compressed = ggml_get_rows(
                graph.ctx0, comp_store,
                ggml_reshape_1d(
                    graph.ctx0, selected,
                    selected->ne[0]*graph.n_tokens));
        compressed = ggml_reshape_4d(
                graph.ctx0, compressed, n_embd_head, 1,
                selected->ne[0], graph.n_tokens);
        k_all = dsv41_concat(
                graph.ctx0, raw_read, compressed, 2, "raw and compressed attention");

        ggml_tensor * source_mask = ggml_reshape_3d(
                graph.ctx0, source->mask, 1,
                source->read_width, graph.n_tokens);
        ggml_tensor * compressed_mask = ggml_get_rows(
                graph.ctx0, source_mask, selected_local);
        compressed_mask = ggml_cont(
                graph.ctx0,
                ggml_permute(
                    graph.ctx0, compressed_mask, 1, 0, 2, 3));
        compressed_mask = ggml_reshape_4d(
                graph.ctx0, compressed_mask,
                selected->ne[0], 1, 1, graph.n_tokens);
        compressed_mask = dsv41_cast_for_store(
                graph.ctx0, compressed_mask, input.raw_mask);
        mask_all = dsv41_concat(
                graph.ctx0, input.raw_mask, compressed_mask, 0, "attention mask");
        n_kv_max += selected->ne[0];
    }

    ggml_tensor * out = graph.build_attn_mha(
            q, k_all, k_all, nullptr, mask_all,
            layer.attn_sinks, nullptr, n_kv_max,
            1.0f/std::sqrt((float) n_embd_head), il);

    // Keep the prior ring intact until attention has consumed it.
    ggml_tensor * completion = dsv41_completion_zero(
            graph.ctx0, out, raw_persist->type);
    raw_persist = ggml_add(
            graph.ctx0, raw_persist, completion);
    raw_persist = dsv41_cast_for_store(
            graph.ctx0, raw_persist, raw_store);
    ggml_tensor * raw_update = ggml_set_rows(
            graph.ctx0, raw_store, raw_persist,
            input.raw_write_idxs);
    ggml_build_forward_expand(graph.gf, raw_update);
    out = ggml_reshape_3d(
            graph.ctx0, out, n_embd_head, graph.n_head,
            graph.n_tokens);
    out = ggml_rope_ext_back(
            graph.ctx0, out, inp_pos, nullptr, n_rot,
            graph.rope_type, rope_ctx, rope_base, rope_scale,
            rope_ext, rope_attn, rope_beta_fast, rope_beta_slow);
    out = ggml_rope_set_offset(out, n_nope);
    graph.cb(out, "dsv41_attn_derope", il);

    out = ggml_reshape_3d(
            graph.ctx0, out, group_width, n_groups,
            graph.n_tokens);
    out = ggml_permute(graph.ctx0, out, 0, 2, 1, 3);
    if (graph.n_tokens > 1 && graph.n_tokens <= 8) {
        out = ggml_cont(graph.ctx0, out);
    }
    ggml_tensor * oa = ggml_mul_mat(graph.ctx0, layer.wo_a, out);
    oa = ggml_permute(graph.ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(
            graph.ctx0, oa, o_lora_rank*n_groups,
            graph.n_tokens);
    out = graph.build_lora_mm(layer.wo_b, oa);
    graph.cb(out, "dsv41_attn_out", il);
    return out;
}

static std::pair<ggml_tensor *, ggml_tensor *> dsv41_build_router(
        const llama_model_deepseek41::graph & graph,
        const llama_layer & layer,
        ggml_tensor * cur,
        int il) {
    ggml_tensor * logits = graph.build_lora_mm(layer.ffn_gate_inp, cur);
    ggml_prec_set_acc(logits, GGML_PREC_F32);
    ggml_tensor * probs = ggml_sqrt(
            graph.ctx0, ggml_softplus(graph.ctx0, logits));
    ggml_tensor * selection = ggml_add(
            graph.ctx0, probs, layer.ffn_exp_probs_b);
    ggml_tensor * ids = ggml_cont(
            graph.ctx0,
            ggml_argsort_top_k(
                graph.ctx0, selection,
                graph.hparams.n_expert_used()));
    ids = dsv41_sort_row_ids(graph.ctx0, ids);
    ggml_set_name(
            ids,
            llama_dsv41_graph_trace_name(
                "expert.ids", il).c_str());

    ggml_tensor * weights = ggml_get_rows(
            graph.ctx0,
            ggml_reshape_3d(
                graph.ctx0, probs, 1, graph.n_expert,
                graph.n_tokens),
            ids);
    weights = ggml_reshape_2d(
            graph.ctx0, weights,
            graph.hparams.n_expert_used(), graph.n_tokens);
    ggml_tensor * sum = ggml_clamp(
            graph.ctx0, ggml_sum_rows(graph.ctx0, weights),
            6.103515625e-5, INFINITY);
    weights = ggml_div(graph.ctx0, weights, sum);
    weights = ggml_scale(
            graph.ctx0, weights,
            graph.hparams.expert_weights_scale);
    weights = ggml_cont(graph.ctx0, weights);
    ggml_set_name(
            weights,
            llama_dsv41_graph_trace_name(
                "expert.weights", il).c_str());
    ggml_build_forward_expand(graph.gf, weights);
    return { logits, ids };
}

}

ggml_tensor * llama_dsv41_graph_append_zero_row(
        ggml_context * ctx,
        ggml_tensor * tensor) {
    return dsv41_append_zero_row(ctx, tensor);
}

ggml_tensor * llama_dsv41_graph_completion_zero(
        ggml_context * ctx,
        ggml_tensor * dependency,
        ggml_type type) {
    return dsv41_completion_zero(ctx, dependency, type);
}

llama_model_deepseek41::graph::graph(
        const llama_model & model,
        const llm_graph_params & params) :
    llama_model_deepseek4::graph(params) {
    GGML_ASSERT(n_layer == LLAMA_DSV41_N_LAYER);
    GGML_ASSERT(hparams.dsv4_hc_mult == LLAMA_DSV41_HC_MULT);
    GGML_ASSERT(mctx != nullptr);

    const auto * dsv41_mctx =
        static_cast<const llama_memory_dsv41_context *>(mctx);
    auto input_owner = std::make_unique<dsv41_graph_input>(
            ctx0, cparams, hparams, dsv41_mctx,
            ubatch, n_outputs);
    auto * input = static_cast<dsv41_graph_input *>(
            res->add_input(std::move(input_owner)));
    if (input->topology.n_seqs != 1) {
        throw std::invalid_argument(
                "DeepSeek V4.1 graph supports one active sequence per ubatch");
    }

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    ggml_tensor * streams = ggml_reshape_3d(
            ctx0, inp, n_embd, 1, n_tokens);
    streams = ggml_repeat_4d(
            ctx0, streams, n_embd,
            hparams.dsv4_hc_mult, n_tokens, 1);
    ggml_tensor * carried_pre = input->initial_pre;
    ggml_tensor * selected = nullptr;
    ggml_tensor * selected_local = nullptr;
    ggml_tensor * candidate_blocks = nullptr;
    cb(streams, "dsv41_hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        if ((size_t) il < cparams.embeddings_layer_inp.size() &&
                cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = dsv41_hc_mean(ctx0, streams);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        if (hparams.dsv41_engram_layers.test(il)) {
            const uint32_t index = il == 1 ? 0 : 1;
            ggml_tensor * row_ids = ggml_cont(
                    ctx0, input->engram_row_ids[index]);
            ggml_set_name(
                    row_ids,
                    llama_dsv41_graph_trace_name(
                        "engram.row_ids", il).c_str());
            ggml_build_forward_expand(gf, row_ids);
            streams = llama_dsv41_build_engram(
                    ctx0, streams,
                    input->engram_rows[index],
                    model.layers[il].engram_kv,
                    model.layers[il].engram_q_norm,
                    model.layers[il].engram_k_norm,
                    input->engram_select[index],
                    norm_rms_eps, sched, backend_cpu);
            cb(streams, "dsv41_engram", il);
        }

        ggml_tensor * residual = streams;
        const dsv41_hc_mix attn_mix = dsv41_build_hc_mix(
                *this, streams,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                il);
        ggml_tensor * cur = build_hc_pre(
                streams, carried_pre, il);
        cb(cur, "dsv41_hc_attn_carried_pre", il);
        cur = build_norm(
                cur, model.layers[il].attn_norm,
                nullptr, LLM_NORM_RMS, il);
        cur = dsv41_build_attention(
                *this, model, *input, cur, inp_pos,
                selected, selected_local, candidate_blocks, il);
        streams = build_hc_post(
                cur, residual, attn_mix.post, attn_mix.comb, il);
        cb(streams, "dsv41_hc_attn_post", il);

        residual = streams;
        const dsv41_hc_mix ffn_mix = dsv41_build_hc_mix(
                *this, streams,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                il);
        cur = build_hc_pre(streams, attn_mix.pre, il);
        cb(cur, "dsv41_hc_ffn_attn_pre", il);
        cur = build_norm(
                cur, model.layers[il].ffn_norm,
                nullptr, LLM_NORM_RMS, il);

        const auto [router_logits, original_ids] =
            dsv41_build_router(*this, model.layers[il], cur, il);
        ggml_tensor * slot_ids = llama_dsv41_build_expert_remap(
                ctx0, original_ids,
                *static_cast<const llama_model_deepseek41 &>(model).experts,
                il, sched, backend_cpu);
        ggml_tensor * moe_out = build_moe_ffn(
                cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, hparams.n_expert_used(),
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                router_logits,
                nullptr, nullptr, nullptr, nullptr,
                original_ids,
                slot_ids);
        cb(moe_out, "dsv41_ffn_moe", il);

        ggml_tensor * shared = build_ffn(
                cur,
                model.layers[il].ffn_up_shexp, nullptr, nullptr,
                model.layers[il].ffn_gate_shexp, nullptr, nullptr,
                model.layers[il].ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(shared, "dsv41_ffn_shared", il);
        cur = ggml_add(ctx0, moe_out, shared);
        cb(cur, "dsv41_ffn_out", il);

        ggml_tensor * release = llama_dsv41_build_expert_release(
                ctx0, moe_out,
                *static_cast<const llama_model_deepseek41 &>(model).experts,
                il, sched, backend_cpu);
        ggml_build_forward_expand(gf, release);

        streams = build_hc_post(
                cur, residual, ffn_mix.post, ffn_mix.comb, il);
        streams = build_cvec(streams, il);
        carried_pre = ffn_mix.pre;
        cb(streams, "dsv41_layer_out", il);
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() &&
            cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = dsv41_hc_mean(ctx0, streams);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    if (inp_out_ids != nullptr) {
        ggml_tensor * flat = ggml_reshape_2d(
                ctx0, streams,
                n_embd*hparams.dsv4_hc_mult, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        streams = ggml_reshape_3d(
                ctx0, flat, n_embd,
                hparams.dsv4_hc_mult, n_outputs);
        carried_pre = ggml_get_rows(ctx0, carried_pre, inp_out_ids);
    }

    ggml_tensor * cur = llama_dsv41_build_output_collapse(
            ctx0, streams, carried_pre, n_embd,
            hparams.dsv4_hc_mult,
            inp_out_ids ? n_outputs : n_tokens);
    cb(cur, "dsv41_output_collapse", -1);
    cur = llama_dsv41_build_output_norm_input(ctx0, cur);
    cb(cur, "dsv41_output_collapse_f32", -1);
    cur = build_norm(
            cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}

std::unique_ptr<llm_graph_context> llama_model_deepseek41::build_arch_graph(
        const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}
