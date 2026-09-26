#include "llama-dsv41.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

static constexpr int32_t DSV41_KV_SOURCES[] = { 2, 8, 14, 20 };
static constexpr int32_t DSV41_INDEX_SOURCES[] = { 2, 8, 14, 20, 24, 28, 32, 36 };
static constexpr uint32_t DSV41_ENGRAM_LAYERS[] = { 1, 14 };
static constexpr uint32_t DSV41_ENGRAM_ROWS[] = { 384006168, 384016682 };

static void dsv41_require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(std::string("DeepSeek V4.1 metadata: ") + message);
    }
}

void llama_dsv41_validate_config(const llama_dsv41_config & config) {
    dsv41_require(config.n_ctx_train == LLAMA_DSV41_N_CTX, "max_position_embeddings must be 1048576");
    dsv41_require(config.n_embd == LLAMA_DSV41_N_EMBD, "hidden_size must be 5120");
    dsv41_require(config.n_layer == LLAMA_DSV41_N_LAYER, "num_hidden_layers must be 40");
    dsv41_require(config.n_vocab == LLAMA_DSV41_N_VOCAB, "vocab_size must be 129280");
    dsv41_require(config.n_head == LLAMA_DSV41_N_HEAD, "num_attention_heads must be 64");
    dsv41_require(config.n_head_kv == LLAMA_DSV41_N_HEAD_KV, "num_key_value_heads must be 1");
    dsv41_require(config.n_head_dim == LLAMA_DSV41_N_HEAD_DIM, "head_dim must be 512");
    dsv41_require(config.n_rot == LLAMA_DSV41_N_ROT, "qk_rope_head_dim must be 64");
    dsv41_require(config.n_lora_q == LLAMA_DSV41_N_LORA_Q, "q_lora_rank must be 1280");
    dsv41_require(config.n_lora_o == LLAMA_DSV41_N_LORA_O, "o_lora_rank must be 1024");
    dsv41_require(config.n_o_group == LLAMA_DSV41_N_O_GROUP, "o_group_num must be 8");
    dsv41_require(config.n_ff_dense == LLAMA_DSV41_N_FF_DENSE, "intermediate_size must be 18432");
    dsv41_require(config.n_ff_expert == LLAMA_DSV41_N_FF_EXP, "moe_intermediate_size must be 2304");
    dsv41_require(config.n_expert == LLAMA_DSV41_N_EXPERT, "n_routed_experts must be 384");
    dsv41_require(config.n_expert_used == LLAMA_DSV41_N_EXPERT_USED, "num_experts_per_tok must be 6");
    dsv41_require(config.n_expert_shared == LLAMA_DSV41_N_EXPERT_SHARED, "n_shared_experts must be 1");
    dsv41_require(config.indexer_n_head == LLAMA_DSV41_N_INDEX_HEAD, "index_n_heads must be 32");
    dsv41_require(config.indexer_head_size == LLAMA_DSV41_N_INDEX_HEAD_DIM, "index_head_dim must be 128");
    dsv41_require(config.indexer_top_k == LLAMA_DSV41_N_INDEX_TOP_K, "index_topk must be 512");
    dsv41_require(config.hc_count == LLAMA_DSV41_HC_MULT, "hc_num_streams must be 4");
    dsv41_require(config.hc_sinkhorn_iters == LLAMA_DSV41_HC_SINKHORN_ITERS, "hc_sinkhorn_iters must be 20");
    dsv41_require(config.raw_window == LLAMA_DSV41_N_SWA, "sliding_window must be 128");
    dsv41_require(config.candidate_source_layer == LLAMA_DSV41_CANDIDATE_SOURCE_LAYER, "candidate_source_layer must be 20");
    dsv41_require(config.candidate_topk_blocks == LLAMA_DSV41_CANDIDATE_TOPK_BLOCKS, "candidate_topk_blocks must be 2048");
    dsv41_require(config.candidate_block_size == LLAMA_DSV41_CANDIDATE_BLOCK_SIZE, "candidate_block_size must be 8");
    dsv41_require(config.f_norm_rms_eps == 1.0e-20f, "rms_norm_eps must be 1e-20");
    dsv41_require(config.hc_eps == 1.0e-6f, "hc_eps must be 1e-6");
    dsv41_require(config.swiglu_clamp == 10.0f, "swiglu_clamp_limit must be 10");
    dsv41_require(config.routed_scale == 1.5f, "routed_scaling_factor must be 1.5");
    dsv41_require(config.rope_theta == 10000, "rope_theta must be 10000");
    dsv41_require(config.compress_rope_theta == 160000, "compress_rope_theta must be 160000");
    dsv41_require(config.yarn_factor == 16.0f, "rope_scaling.factor must be 16");
    dsv41_require(config.yarn_beta_fast == 32.0f, "rope_scaling.beta_fast must be 32");
    dsv41_require(config.yarn_beta_slow == 1.0f, "rope_scaling.beta_slow must be 1");
    dsv41_require(config.yarn_original_context == 65536.0f, "rope_scaling.original_max_position_embeddings must be 65536");
    dsv41_require(config.expert_weights_norm, "norm_topk_prob must be true");
    dsv41_require(config.hidden_act == "silu", "hidden_act must be silu");
    dsv41_require(config.scoring_func == "sqrtsoftplus", "scoring_func must be sqrtsoftplus");
    dsv41_require(config.topk_method == "noaux_tc", "topk_method must be noaux_tc");

    dsv41_require(config.compress_ratios.size() == LLAMA_DSV41_N_LAYER, "compress_ratios must have 40 entries");
    for (uint32_t il = 0; il < LLAMA_DSV41_N_LAYER; ++il) {
        dsv41_require(config.compress_ratios[il] == llama_dsv41_compress_ratio(il), "compress_ratios has an invalid main-layer value");
    }
    dsv41_require(config.kv_sources == std::vector<uint32_t>(std::begin(DSV41_KV_SOURCES), std::end(DSV41_KV_SOURCES)), "kv_source_layers must be [2,8,14,20]");
    dsv41_require(config.index_sources == std::vector<uint32_t>(std::begin(DSV41_INDEX_SOURCES), std::end(DSV41_INDEX_SOURCES)), "index_source_layers must be [2,8,14,20,24,28,32,36]");
    dsv41_require(config.engram_layers == std::vector<uint32_t>(std::begin(DSV41_ENGRAM_LAYERS), std::end(DSV41_ENGRAM_LAYERS)), "engram.layer_ids must be [1,14]");
    dsv41_require(config.engram_rows == std::vector<uint32_t>(std::begin(DSV41_ENGRAM_ROWS), std::end(DSV41_ENGRAM_ROWS)), "engram.rows must be [384006168,384016682]");
    dsv41_require(config.engram_encoding == LLAMA_DSV41_ENGRAM_ENCODING, "engram.encoding must be e4m3_e8m0_32_row264");
    dsv41_require(config.engram_compressed_vocab_size == LLAMA_DSV41_ENGRAM_COMPRESSED_VOCAB, "engram.compressed_vocab_size must be 99092");
    dsv41_require(config.engram_pad_id == LLAMA_DSV41_ENGRAM_PAD_ID, "engram.pad_id must be 2");
    dsv41_require(config.engram_token_map_size == LLAMA_DSV41_N_VOCAB, "engram.token_map must contain 129280 entries");
    dsv41_require(config.engram_primes_size == LLAMA_DSV41_ENGRAM_PRIMES_COUNT, "engram.primes must contain 48 entries");
    dsv41_require(config.engram_multipliers_size == LLAMA_DSV41_ENGRAM_MULTIPLIERS_COUNT, "engram.multipliers must contain 8 entries");
    if (!config.engram_token_map.empty() || !config.engram_primes.empty() || !config.engram_multipliers.empty()) {
        llama_dsv41_make_engram_layout(config);
    }
}

llama_engram_layout llama_dsv41_make_engram_layout(const llama_dsv41_config & config) {
    dsv41_require(config.engram_token_map.size() == LLAMA_DSV41_N_VOCAB, "engram.token_map data must contain 129280 entries");
    dsv41_require(config.engram_primes.size() == LLAMA_DSV41_ENGRAM_PRIMES_COUNT, "engram.primes data must contain 48 entries");
    dsv41_require(config.engram_multipliers.size() == LLAMA_DSV41_ENGRAM_MULTIPLIERS_COUNT, "engram.multipliers data must contain 8 entries");

    llama_engram_layout layout;
    layout.encoding = config.engram_encoding;
    std::copy(config.engram_layers.begin(), config.engram_layers.end(), layout.layer_ids.begin());
    layout.token_map = config.engram_token_map;
    layout.compressed_vocab_size = config.engram_compressed_vocab_size;
    layout.pad_id = config.engram_pad_id;
    std::copy(config.engram_rows.begin(), config.engram_rows.end(), layout.rows.begin());
    for (size_t layer = 0; layer < LLAMA_ENGRAM_LAYERS; ++layer) {
        std::copy_n(
                config.engram_primes.begin() + layer*LLAMA_ENGRAM_COLS,
                LLAMA_ENGRAM_COLS,
                layout.primes[layer].begin());
        std::copy_n(
                config.engram_multipliers.begin() + layer*LLAMA_ENGRAM_NGRAM,
                LLAMA_ENGRAM_NGRAM,
                layout.multipliers[layer].begin());
    }
    llama_engram_hasher validate(layout);
    return layout;
}



static int32_t dsv41_source_layer(const int32_t * sources, size_t n, uint32_t il) {
    int32_t result = -1;
    for (size_t i = 0; i < n; ++i) {
        if ((uint32_t) sources[i] > il) {
            break;
        }
        result = sources[i];
    }
    return result;
}

llama_dsv41_cache_state::llama_dsv41_cache_state(uint32_t compressed_cache_size) :
        compressed_cache_size(compressed_cache_size),
        raw(LLAMA_DSV41_N_SWA, -1) {
    if (compressed_cache_size == 0) {
        throw std::runtime_error("DeepSeek V4.1 compressed cache is empty");
    }
    for (int32_t source : DSV41_KV_SOURCES) {
        const uint32_t ratio = llama_dsv41_compress_ratio(source);
        sources.emplace(source, source_state {
            ratio,
            std::vector<llama_pos>(compressed_cache_size, -1),
            std::vector<llama_pos>(ratio, -1),
        });
    }
}

void llama_dsv41_cache_state::clear() {
    pos = -1;
    std::fill(raw.begin(), raw.end(), -1);
    for (auto & item : sources) {
        std::fill(item.second.compressed.begin(), item.second.compressed.end(), -1);
        std::fill(item.second.pending.begin(), item.second.pending.end(), -1);
    }
    candidates.clear();
}

void llama_dsv41_cache_state::append(llama_pos next_pos) {
    if (next_pos != pos + 1) {
        throw std::runtime_error("DeepSeek V4.1 cache requires contiguous single-sequence positions");
    }

    for (const auto & item : sources) {
        const source_state & state = item.second;
        if ((next_pos + 1)%state.ratio == 0 && (uint64_t) (next_pos/state.ratio) >= compressed_cache_size) {
            throw std::runtime_error("DeepSeek V4.1 compressed cache overflow");
        }
    }

    raw[next_pos%raw.size()] = next_pos;
    for (auto & item : sources) {
        source_state & state = item.second;
        state.pending[next_pos%state.ratio] = next_pos;
        if ((next_pos + 1)%state.ratio != 0) {
            continue;
        }

        const uint64_t dst = next_pos/state.ratio;
        state.compressed[dst] = next_pos + 1 - state.ratio;
    }
    pos = next_pos;
}

void llama_dsv41_cache_state::set_candidate_blocks(const std::vector<int32_t> & blocks) {
    candidates = blocks;
}

llama_pos llama_dsv41_cache_state::position() const {
    return pos;
}

const std::vector<llama_pos> & llama_dsv41_cache_state::raw_slots() const {
    return raw;
}

const std::vector<llama_pos> & llama_dsv41_cache_state::compressed_slots(uint32_t source_layer) const {
    const auto it = sources.find(source_layer);
    if (it == sources.end()) {
        throw std::runtime_error("DeepSeek V4.1 compressed cache source layer is invalid");
    }
    return it->second.compressed;
}

const std::vector<llama_pos> & llama_dsv41_cache_state::pending_slots(uint32_t source_layer) const {
    const auto it = sources.find(source_layer);
    if (it == sources.end()) {
        throw std::runtime_error("DeepSeek V4.1 compressor state source layer is invalid");
    }
    return it->second.pending;
}

const std::vector<int32_t> & llama_dsv41_cache_state::candidate_blocks() const {
    return candidates;
}

uint64_t llama_dsv41_memory_accounting::total() const {
    return raw_kv + compressed_kv + index_keys + compressor_carry +
        candidate_scores + candidate_ids + position_state + graph_workspace;
}

llama_dsv41_memory_accounting llama_dsv41_account_memory(
        uint32_t n_ctx,
        uint32_t n_seq,
        uint32_t n_tokens,
        uint32_t kv_element_size,
        uint32_t index_element_size,
        uint64_t graph_workspace) {
    if (n_ctx == 0 || n_seq == 0 || n_tokens == 0 || kv_element_size == 0 || index_element_size == 0) {
        throw std::runtime_error("DeepSeek V4.1 memory accounting dimensions must be non-zero");
    }

    llama_dsv41_memory_accounting result;
    const uint64_t raw_rows = (uint64_t) LLAMA_DSV41_N_LAYER*LLAMA_DSV41_N_SWA*n_seq;
    const uint64_t ratio_2_rows = ((uint64_t) n_ctx + 1)/2;
    const uint64_t ratio_1_rows = n_ctx;
    const uint64_t compressed_rows = (3*ratio_2_rows + ratio_1_rows)*n_seq;
    const uint64_t candidate_blocks = ((uint64_t) n_ctx + LLAMA_DSV41_CANDIDATE_BLOCK_SIZE - 1)/
        LLAMA_DSV41_CANDIDATE_BLOCK_SIZE;
    uint64_t pending_rows = 0;
    uint64_t gated_pending_rows = 0;
    for (int32_t source : DSV41_KV_SOURCES) {
        const uint32_t ratio = llama_dsv41_compress_ratio(source);
        pending_rows += ratio;
        if (ratio == 2) {
            gated_pending_rows += ratio;
        }
    }

    result.raw_kv = raw_rows*LLAMA_DSV41_N_HEAD_DIM*kv_element_size;
    result.compressed_kv = compressed_rows*LLAMA_DSV41_N_HEAD_DIM*kv_element_size;
    result.index_keys = compressed_rows*LLAMA_DSV41_N_INDEX_HEAD_DIM*index_element_size;
    result.compressor_carry =
        (pending_rows + gated_pending_rows)*LLAMA_DSV41_N_HEAD_DIM*kv_element_size*n_seq;
    result.candidate_scores = candidate_blocks*sizeof(float)*n_tokens;
    result.candidate_ids = std::min<uint64_t>(candidate_blocks, LLAMA_DSV41_CANDIDATE_TOPK_BLOCKS)*
        sizeof(int32_t)*n_tokens;
    const uint64_t position_rows =
        ((uint64_t) LLAMA_DSV41_N_SWA + compressed_rows/n_seq + pending_rows)*n_seq;
    result.position_state = position_rows*sizeof(llama_pos);
    result.graph_workspace = graph_workspace;
    return result;
}

int32_t llama_dsv41_kv_source_layer(uint32_t il) {
    return dsv41_source_layer(DSV41_KV_SOURCES, sizeof(DSV41_KV_SOURCES)/sizeof(DSV41_KV_SOURCES[0]), il);
}

int32_t llama_dsv41_index_source_layer(uint32_t il) {
    return dsv41_source_layer(DSV41_INDEX_SOURCES, sizeof(DSV41_INDEX_SOURCES)/sizeof(DSV41_INDEX_SOURCES[0]), il);
}

uint32_t llama_dsv41_compress_ratio(uint32_t il) {
    if (il < 2) {
        return 0;
    }
    if (il < 20) {
        return 2;
    }
    if (il < LLAMA_DSV41_N_LAYER) {
        return 1;
    }
    throw std::runtime_error("DeepSeek V4.1 layer index is out of range");
}

llama_dsv41_layer_plan llama_dsv41_build_layer_plan(
        uint32_t il,
        const std::vector<llama_pos> & positions,
        uint32_t compressed_cache_size) {
    if (positions.empty()) {
        throw std::runtime_error("DeepSeek V4.1 graph plan requires at least one token");
    }
    if (positions.front() < 0) {
        throw std::runtime_error("DeepSeek V4.1 graph plan requires non-negative positions");
    }
    for (size_t i = 1; i < positions.size(); ++i) {
        if (positions[i] != positions[i - 1] + 1) {
            throw std::runtime_error("DeepSeek V4.1 graph plan requires one contiguous sequence");
        }
    }

    llama_dsv41_layer_plan plan = {};
    plan.layer = il;
    plan.ratio = llama_dsv41_compress_ratio(il);
    plan.kv_source_layer = llama_dsv41_kv_source_layer(il);
    plan.index_source_layer = llama_dsv41_index_source_layer(il);
    plan.owns_kv_source = plan.kv_source_layer == (int32_t) il;
    plan.owns_index_source = plan.index_source_layer == (int32_t) il;
    plan.builds_candidates = il == LLAMA_DSV41_CANDIDATE_SOURCE_LAYER;
    plan.uses_candidates = plan.owns_index_source && il > LLAMA_DSV41_CANDIDATE_SOURCE_LAYER;
    plan.reuses_index_selection = !plan.owns_index_source && plan.index_source_layer >= 0;
    plan.collapses_output = il + 1 == LLAMA_DSV41_N_LAYER;
    plan.raw_ring_order = llama_dsv41_raw_ring_order(positions.back(), LLAMA_DSV41_N_SWA);
    if (plan.owns_kv_source && plan.ratio != 0) {
        plan.compression = llama_dsv41_build_compression_plan(positions, plan.ratio, compressed_cache_size);
    }
    return plan;
}

llama_dsv41_compression_plan llama_dsv41_build_compression_plan(
        const std::vector<llama_pos> & positions,
        uint32_t ratio,
        uint32_t cache_size) {
    if (ratio != 1 && ratio != 2) {
        throw std::runtime_error("DeepSeek V4.1 compression ratio must be 1 or 2");
    }
    if (cache_size == 0) {
        throw std::runtime_error("DeepSeek V4.1 compressed cache is empty");
    }

    llama_dsv41_compression_plan plan;
    plan.n_visible.resize(positions.size(), 0);

    std::vector<int32_t> latest_state_src(ratio, -1);
    std::vector<llama_pos> latest_state_pos(ratio, -1);

    const int32_t scratch_offset = ratio;
    for (size_t i = 0; i < positions.size(); ++i) {
        const llama_pos pos = positions[i];
        if (pos < 0) {
            continue;
        }

        plan.n_visible[i] = (int32_t) ((pos + 1)/ratio);
        plan.n_kv = std::max<int64_t>(plan.n_kv, plan.n_visible[i]);

        if (ratio == 1) {
            if ((uint64_t) pos >= cache_size) {
                throw std::runtime_error("DeepSeek V4.1 compressed cache overflow");
            }
            plan.write_idxs.push_back(pos);
            plan.write_pos.push_back((int32_t) pos);
            continue;
        }

        const int32_t row = (int32_t) (pos%ratio);
        plan.state_pos.push_back(row);
        if (latest_state_src[row] < 0 || pos > latest_state_pos[row]) {
            latest_state_src[row] = (int32_t) i;
            latest_state_pos[row] = pos;
        }

        if ((pos + 1)%ratio != 0) {
            continue;
        }

        const llama_pos source_start = pos + 1 - ratio;
        const int64_t write_idx = pos/ratio;
        if ((uint64_t) write_idx >= cache_size) {
            throw std::runtime_error("DeepSeek V4.1 compressed cache overflow");
        }

        for (uint32_t j = 0; j < ratio; ++j) {
            const llama_pos source_pos = source_start + j;
            int32_t source_idx = (int32_t) (source_pos%ratio);
            for (size_t k = 0; k < positions.size(); ++k) {
                if (positions[k] == source_pos) {
                    source_idx = scratch_offset + (int32_t) k;
                    break;
                }
            }
            plan.state_read_idxs.push_back(source_idx);
        }

        plan.write_idxs.push_back(write_idx);
        plan.write_pos.push_back((int32_t) source_start);
    }

    if (ratio == 2) {
        for (uint32_t row = 0; row < ratio; ++row) {
            if (latest_state_src[row] >= 0) {
                plan.state_persist_src_idxs.push_back(latest_state_src[row]);
                plan.state_persist_dst_idxs.push_back((int32_t) row);
            }
        }
    }

    plan.n_kv = std::min<int64_t>(cache_size, std::max<int64_t>(1, plan.n_kv));
    return plan;
}

std::vector<int32_t> llama_dsv41_select_candidate_blocks(
        const std::vector<float> & scores,
        uint32_t n_visible,
        uint32_t block_size,
        uint32_t top_k_blocks) {
    if (block_size == 0) {
        throw std::runtime_error("DeepSeek V4.1 candidate block size must be non-zero");
    }

    n_visible = std::min<uint32_t>(n_visible, scores.size());
    if (n_visible == 0 || top_k_blocks == 0) {
        return {};
    }

    const uint32_t n_blocks = (n_visible + block_size - 1)/block_size;
    std::vector<float> block_scores(n_blocks, -std::numeric_limits<float>::infinity());
    for (uint32_t block = 0; block < n_blocks; ++block) {
        const uint32_t i0 = block*block_size;
        const uint32_t i1 = std::min<uint32_t>(n_visible, i0 + block_size);
        for (uint32_t i = i0; i < i1; ++i) {
            if (std::isnan(scores[i])) {
                throw std::runtime_error("DeepSeek V4.1 candidate score is NaN");
            }
            block_scores[block] = std::max(block_scores[block], scores[i]);
        }
    }

    const int32_t final_block = (int32_t) n_blocks - 1;
    std::vector<int32_t> blocks(n_blocks - 1);
    std::iota(blocks.begin(), blocks.end(), 0);
    std::stable_sort(blocks.begin(), blocks.end(), [&](int32_t a, int32_t b) {
        if (block_scores[a] != block_scores[b]) {
            return block_scores[a] > block_scores[b];
        }
        return a < b;
    });

    const uint32_t n_selected = std::min<uint32_t>(top_k_blocks, n_blocks);
    std::vector<int32_t> selected;
    selected.reserve(n_selected);
    selected.push_back(final_block);
    selected.insert(selected.end(), blocks.begin(), blocks.begin() + n_selected - 1);
    return selected;
}

std::vector<int32_t> llama_dsv41_candidate_rows(
        const std::vector<int32_t> & blocks,
        uint32_t n_visible,
        uint32_t block_size) {
    if (block_size == 0) {
        throw std::runtime_error("DeepSeek V4.1 candidate block size must be non-zero");
    }

    std::vector<int32_t> rows;
    for (int32_t block : blocks) {
        if (block < 0) {
            throw std::runtime_error("DeepSeek V4.1 candidate block index must be non-negative");
        }
        const uint64_t i0 = (uint64_t) block*block_size;
        const uint64_t i1 = std::min<uint64_t>(n_visible, i0 + block_size);
        for (uint64_t i = i0; i < i1; ++i) {
            rows.push_back((int32_t) i);
        }
    }
    return rows;
}

std::vector<uint32_t> llama_dsv41_raw_ring_order(llama_pos pos, uint32_t window) {
    if (window == 0 || pos < 0) {
        return {};
    }

    const uint32_t n_raw = std::min<uint64_t>((uint64_t) pos + 1, window);
    const uint32_t start = (uint32_t) ((pos + 1 - n_raw)%window);

    std::vector<uint32_t> result(n_raw);
    for (uint32_t i = 0; i < n_raw; ++i) {
        result[i] = (start + i)%window;
    }
    return result;
}

std::vector<float> llama_dsv41_output_collapse(
        const std::vector<float> & residual,
        const std::vector<float> & pre,
        uint32_t n_embd,
        uint32_t hc_mult) {
    if (pre.size() != hc_mult || residual.size() != (size_t) n_embd*hc_mult) {
        throw std::runtime_error("DeepSeek V4.1 output collapse shape mismatch");
    }

    std::vector<float> result(n_embd, 0.0f);
    for (uint32_t h = 0; h < hc_mult; ++h) {
        for (uint32_t d = 0; d < n_embd; ++d) {
            result[d] += residual[(size_t) h*n_embd + d]*pre[h];
        }
    }
    return result;
}

ggml_tensor * llama_dsv41_build_ratio_pool(
        ggml_context * ctx,
        ggml_tensor * kv,
        ggml_tensor * gate,
        uint32_t ratio) {
    if (ratio != 1 && ratio != 2) {
        throw std::runtime_error("DeepSeek V4.1 graph compression ratio must be 1 or 2");
    }
    if (kv->ne[1] != ratio) {
        throw std::runtime_error("DeepSeek V4.1 graph compressor input shape mismatch");
    }
    if (ratio == 1) {
        return ggml_reshape_2d(ctx, kv, kv->ne[0], kv->ne[2]);
    }
    if (gate == nullptr || !ggml_are_same_shape(kv, gate)) {
        throw std::runtime_error("DeepSeek V4.1 ratio-2 graph requires matching KV and gate tensors");
    }

    ggml_tensor * kv_t = ggml_cont(ctx, ggml_permute(ctx, kv, 1, 0, 2, 3));
    ggml_tensor * gate_t = ggml_cont(ctx, ggml_permute(ctx, gate, 1, 0, 2, 3));
    ggml_tensor * weights = ggml_soft_max(ctx, gate_t);
    ggml_tensor * pooled = ggml_sum_rows(ctx, ggml_mul(ctx, kv_t, weights));
    return ggml_reshape_2d(ctx, pooled, kv->ne[0], kv->ne[2]);
}

ggml_tensor * llama_dsv41_build_shared_softmax(
        ggml_context * ctx,
        ggml_tensor * raw_scores,
        ggml_tensor * compressed_scores) {
    if (raw_scores == nullptr && compressed_scores == nullptr) {
        throw std::runtime_error("DeepSeek V4.1 attention requires raw or compressed scores");
    }
    ggml_tensor * scores = raw_scores;
    if (scores == nullptr) {
        scores = compressed_scores;
    } else if (compressed_scores != nullptr) {
        if (raw_scores->ne[1] != compressed_scores->ne[1] ||
                raw_scores->ne[2] != compressed_scores->ne[2] ||
                raw_scores->ne[3] != compressed_scores->ne[3]) {
            throw std::runtime_error("DeepSeek V4.1 raw and compressed score shapes are incompatible");
        }
        scores = ggml_concat(ctx, raw_scores, compressed_scores, 0);
    }
    return ggml_soft_max(ctx, scores);
}

ggml_tensor * llama_dsv41_build_candidate_blocks(
        ggml_context * ctx,
        ggml_tensor * block_scores,
        ggml_tensor * final_blocks,
        uint32_t n_candidate) {
    if (block_scores == nullptr || final_blocks == nullptr ||
            block_scores->type != GGML_TYPE_F32 ||
            final_blocks->type != GGML_TYPE_I32 ||
            final_blocks->ne[0] != 1 ||
            final_blocks->ne[1] != block_scores->ne[1]) {
        throw std::runtime_error("DeepSeek V4.1 candidate graph shape mismatch");
    }

    const int64_t n_blocks = block_scores->ne[0];
    const int64_t n_tokens = block_scores->ne[1];
    if (n_candidate == 0 || n_candidate > (uint32_t) n_blocks) {
        throw std::runtime_error("DeepSeek V4.1 candidate graph width is invalid");
    }
    if (n_candidate == 1) {
        return ggml_cont(ctx, final_blocks);
    }

    ggml_tensor * local = ggml_reshape_2d(
            ctx, ggml_arange(ctx, 0.0f, (float) (n_blocks - 1), 1.0f),
            n_blocks - 1, 1);
    local = ggml_repeat_4d(
            ctx, local, n_blocks - 1, n_tokens, 1, 1);
    ggml_tensor * final_f32 = ggml_cast(ctx, final_blocks, GGML_TYPE_F32);
    ggml_tensor * shift = ggml_step(
            ctx, ggml_scale_bias(
                ctx, ggml_sub(ctx, local, final_f32), 1.0f, 0.5f));
    ggml_tensor * ordinary_ids = ggml_cast(
            ctx, ggml_add(ctx, local, shift), GGML_TYPE_I32);

    ggml_tensor * ordinary_scores = ggml_get_rows(
            ctx,
            ggml_reshape_3d(ctx, block_scores, 1, n_blocks, n_tokens),
            ordinary_ids);
    ordinary_scores = ggml_reshape_2d(
            ctx, ordinary_scores, n_blocks - 1, n_tokens);
    ggml_tensor * selected_local = ggml_argsort_top_k(
            ctx, ordinary_scores, n_candidate - 1);
    ggml_tensor * selected = ggml_get_rows(
            ctx,
            ggml_reshape_3d(ctx, ordinary_ids, 1, n_blocks - 1, n_tokens),
            selected_local);
    selected = ggml_reshape_2d(
            ctx, selected, n_candidate - 1, n_tokens);
    return ggml_cont(ctx, ggml_concat(ctx, final_blocks, selected, 0));
}

ggml_tensor * llama_dsv41_build_output_collapse(
        ggml_context * ctx,
        ggml_tensor * residual,
        ggml_tensor * pre,
        uint32_t n_embd,
        uint32_t hc_mult,
        uint32_t n_tokens) {
    if (residual->ne[0] != n_embd || residual->ne[1] != hc_mult || residual->ne[2] != n_tokens ||
            pre->ne[0] != hc_mult || pre->ne[1] != n_tokens) {
        throw std::runtime_error("DeepSeek V4.1 graph output collapse shape mismatch");
    }

    ggml_tensor * residual_t = ggml_cont(ctx, ggml_permute(ctx, residual, 1, 0, 2, 3));
    ggml_tensor * pre_t = ggml_reshape_3d(ctx, pre, hc_mult, 1, n_tokens);
    ggml_tensor * collapsed = ggml_sum_rows(ctx, ggml_mul(ctx, residual_t, pre_t));
    collapsed = ggml_reshape_2d(ctx, collapsed, n_embd, n_tokens);
    return ggml_cast(ctx, collapsed, GGML_TYPE_BF16);
}

ggml_tensor * llama_dsv41_build_output_norm_input(
        ggml_context * ctx,
        ggml_tensor * collapsed) {
    if (collapsed == nullptr || collapsed->type != GGML_TYPE_BF16) {
        throw std::runtime_error("DeepSeek V4.1 output collapse must be BF16");
    }
    return ggml_cast(ctx, collapsed, GGML_TYPE_F32);
}

ggml_tensor * llama_dsv41_build_output(
        ggml_context * ctx,
        ggml_tensor * residual,
        ggml_tensor * pre,
        ggml_tensor * output_norm,
        ggml_tensor * output,
        float rms_eps,
        uint32_t hc_mult) {
    if (output_norm->ne[0] != residual->ne[0] || output->ne[0] != residual->ne[0]) {
        throw std::runtime_error("DeepSeek V4.1 output tensor shape mismatch");
    }

    ggml_tensor * collapsed = llama_dsv41_build_output_collapse(
            ctx, residual, pre, residual->ne[0], hc_mult, residual->ne[2]);
    ggml_tensor * normalized = ggml_rms_norm(
            ctx, llama_dsv41_build_output_norm_input(ctx, collapsed), rms_eps);
    normalized = ggml_mul(ctx, normalized, output_norm);
    return ggml_mul_mat(ctx, output, normalized);
}
