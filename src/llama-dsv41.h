#pragma once

#include "llama-engram.h"
#include "llama.h"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

static constexpr uint32_t LLAMA_DSV41_N_LAYER                 = 40;
static constexpr uint32_t LLAMA_DSV41_N_EMBD                  = 5120;
static constexpr uint32_t LLAMA_DSV41_N_VOCAB                 = 129280;
static constexpr uint32_t LLAMA_DSV41_N_HEAD                  = 64;
static constexpr uint32_t LLAMA_DSV41_N_HEAD_KV               = 1;
static constexpr uint32_t LLAMA_DSV41_N_HEAD_DIM              = 512;
static constexpr uint32_t LLAMA_DSV41_N_ROT                   = 64;
static constexpr uint32_t LLAMA_DSV41_N_LORA_Q                = 1280;
static constexpr uint32_t LLAMA_DSV41_N_LORA_O                = 1024;
static constexpr uint32_t LLAMA_DSV41_N_O_GROUP               = 8;
static constexpr uint32_t LLAMA_DSV41_N_FF_DENSE              = 18432;
static constexpr uint32_t LLAMA_DSV41_N_EXPERT                = 384;
static constexpr uint32_t LLAMA_DSV41_N_EXPERT_USED           = 6;
static constexpr uint32_t LLAMA_DSV41_N_EXPERT_SHARED         = 1;
static constexpr uint32_t LLAMA_DSV41_N_FF_EXP                = 2304;
static constexpr uint32_t LLAMA_DSV41_N_INDEX_HEAD            = 32;
static constexpr uint32_t LLAMA_DSV41_N_INDEX_HEAD_DIM        = 128;
static constexpr uint32_t LLAMA_DSV41_N_INDEX_TOP_K           = 512;
static constexpr uint32_t LLAMA_DSV41_N_SWA                   = 128;
static constexpr uint32_t LLAMA_DSV41_N_CTX                   = 1048576;
static constexpr uint32_t LLAMA_DSV41_HC_MULT                 = 4;
static constexpr uint32_t LLAMA_DSV41_HC_SINKHORN_ITERS       = 20;
static constexpr uint32_t LLAMA_DSV41_CANDIDATE_SOURCE_LAYER  = 20;
static constexpr uint32_t LLAMA_DSV41_CANDIDATE_TOPK_BLOCKS   = 2048;
static constexpr uint32_t LLAMA_DSV41_CANDIDATE_BLOCK_SIZE    = 8;
static constexpr uint32_t LLAMA_DSV41_ENGRAM_COMPRESSED_VOCAB = 99092;
static constexpr uint32_t LLAMA_DSV41_ENGRAM_PAD_ID            = 2;
static constexpr uint32_t LLAMA_DSV41_ENGRAM_PRIMES_COUNT     = 48;
static constexpr uint32_t LLAMA_DSV41_ENGRAM_MULTIPLIERS_COUNT = 8;
static constexpr const char * LLAMA_DSV41_ENGRAM_ENCODING = "e4m3_e8m0_32_row264";

struct llama_dsv41_config {
    uint32_t n_ctx_train;
    uint32_t n_embd;
    uint32_t n_layer;
    uint32_t n_vocab;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_head_dim;
    uint32_t n_rot;
    uint32_t n_lora_q;
    uint32_t n_lora_o;
    uint32_t n_o_group;
    uint32_t n_ff_dense;
    uint32_t n_ff_expert;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_expert_shared;
    uint32_t indexer_n_head;
    uint32_t indexer_head_size;
    uint32_t indexer_top_k;
    uint32_t hc_count;
    uint32_t hc_sinkhorn_iters;
    uint32_t raw_window;
    uint32_t candidate_source_layer;
    uint32_t candidate_topk_blocks;
    uint32_t candidate_block_size;
    float f_norm_rms_eps;
    float hc_eps;
    float swiglu_clamp;
    float routed_scale;
    uint32_t rope_theta;
    uint32_t compress_rope_theta;
    float yarn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    float yarn_original_context;
    bool expert_weights_norm;
    std::string hidden_act;
    std::string scoring_func;
    std::string topk_method;
    std::vector<uint32_t> compress_ratios;
    std::vector<uint32_t> kv_sources;
    std::vector<uint32_t> index_sources;
    std::vector<uint32_t> engram_layers;
    std::vector<uint32_t> engram_rows;
    std::string engram_encoding;
    uint32_t engram_compressed_vocab_size;
    uint32_t engram_pad_id;
    uint32_t engram_token_map_size;
    uint32_t engram_primes_size;
    uint32_t engram_multipliers_size;
    std::vector<uint32_t> engram_token_map;
    std::vector<uint32_t> engram_primes;
    std::vector<uint64_t> engram_multipliers;
};

void llama_dsv41_validate_config(const llama_dsv41_config & config);
llama_engram_layout llama_dsv41_make_engram_layout(const llama_dsv41_config & config);

struct llama_dsv41_compression_plan {
    std::vector<int32_t> state_pos;
    std::vector<int32_t> state_persist_src_idxs;
    std::vector<int32_t> state_persist_dst_idxs;
    std::vector<int32_t> state_read_idxs;
    std::vector<int64_t> write_idxs;
    std::vector<int32_t> write_pos;
    std::vector<int32_t> n_visible;
    int64_t n_kv = 0;
};

struct llama_dsv41_layer_plan {
    uint32_t layer;
    uint32_t ratio;
    int32_t kv_source_layer;
    int32_t index_source_layer;
    bool owns_kv_source;
    bool owns_index_source;
    bool builds_candidates;
    bool uses_candidates;
    bool reuses_index_selection;
    bool collapses_output;
    std::vector<uint32_t> raw_ring_order;
    llama_dsv41_compression_plan compression;
};

class llama_dsv41_cache_state {
public:
    explicit llama_dsv41_cache_state(uint32_t compressed_cache_size);

    void clear();
    void append(llama_pos pos);
    void set_candidate_blocks(const std::vector<int32_t> & blocks);

    llama_pos position() const;
    const std::vector<llama_pos> & raw_slots() const;
    const std::vector<llama_pos> & compressed_slots(uint32_t source_layer) const;
    const std::vector<llama_pos> & pending_slots(uint32_t source_layer) const;
    const std::vector<int32_t> & candidate_blocks() const;

private:
    struct source_state {
        uint32_t ratio;
        std::vector<llama_pos> compressed;
        std::vector<llama_pos> pending;
    };

    uint32_t compressed_cache_size;
    llama_pos pos = -1;
    std::vector<llama_pos> raw;
    std::map<uint32_t, source_state> sources;
    std::vector<int32_t> candidates;
};

struct llama_dsv41_memory_accounting {
    uint64_t raw_kv = 0;
    uint64_t compressed_kv = 0;
    uint64_t index_keys = 0;
    uint64_t compressor_carry = 0;
    uint64_t candidate_scores = 0;
    uint64_t candidate_ids = 0;
    uint64_t position_state = 0;
    uint64_t graph_workspace = 0;

    uint64_t total() const;
};

llama_dsv41_memory_accounting llama_dsv41_account_memory(
        uint32_t n_ctx,
        uint32_t n_seq,
        uint32_t n_tokens,
        uint32_t kv_element_size,
        uint32_t index_element_size,
        uint64_t graph_workspace);

int32_t llama_dsv41_kv_source_layer(uint32_t il);
int32_t llama_dsv41_index_source_layer(uint32_t il);
uint32_t llama_dsv41_compress_ratio(uint32_t il);

llama_dsv41_layer_plan llama_dsv41_build_layer_plan(
        uint32_t il,
        const std::vector<llama_pos> & positions,
        uint32_t compressed_cache_size);

llama_dsv41_compression_plan llama_dsv41_build_compression_plan(
        const std::vector<llama_pos> & positions,
        uint32_t ratio,
        uint32_t cache_size);

std::vector<int32_t> llama_dsv41_select_candidate_blocks(
        const std::vector<float> & scores,
        uint32_t n_visible,
        uint32_t block_size,
        uint32_t top_k_blocks);

std::vector<int32_t> llama_dsv41_candidate_rows(
        const std::vector<int32_t> & blocks,
        uint32_t n_visible,
        uint32_t block_size);

std::vector<uint32_t> llama_dsv41_raw_ring_order(llama_pos pos, uint32_t window);

std::vector<float> llama_dsv41_output_collapse(
        const std::vector<float> & residual,
        const std::vector<float> & pre,
        uint32_t n_embd,
        uint32_t hc_mult);

ggml_tensor * llama_dsv41_build_ratio_pool(
        ggml_context * ctx,
        ggml_tensor * kv,
        ggml_tensor * gate,
        uint32_t ratio);

ggml_tensor * llama_dsv41_build_shared_softmax(
        ggml_context * ctx,
        ggml_tensor * raw_scores,
        ggml_tensor * compressed_scores);

ggml_tensor * llama_dsv41_build_candidate_blocks(
        ggml_context * ctx,
        ggml_tensor * block_scores,
        ggml_tensor * final_blocks,
        uint32_t n_candidate);

ggml_tensor * llama_dsv41_build_output_collapse(
        ggml_context * ctx,
        ggml_tensor * residual,
        ggml_tensor * pre,
        uint32_t n_embd,
        uint32_t hc_mult,
        uint32_t n_tokens);

ggml_tensor * llama_dsv41_build_output_norm_input(
        ggml_context * ctx,
        ggml_tensor * collapsed);

ggml_tensor * llama_dsv41_build_output(
        ggml_context * ctx,
        ggml_tensor * residual,
        ggml_tensor * pre,
        ggml_tensor * output_norm,
        ggml_tensor * output,
        float rms_eps,
        uint32_t hc_mult);
