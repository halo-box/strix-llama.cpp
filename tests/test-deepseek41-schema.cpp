#include "../src/llama-arch.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

static void check(bool condition, const std::string & message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message.c_str());
        std::exit(1);
    }
}

int main() {
    check(llm_arch_from_string("deepseek41") == LLM_ARCH_DEEPSEEK41, "deepseek41 architecture lookup failed");
    check(std::string(llm_arch_name(LLM_ARCH_DEEPSEEK41)) == "deepseek41", "deepseek41 architecture name failed");

    const LLM_KV kv(LLM_ARCH_DEEPSEEK41);
    const std::vector<std::pair<llm_kv, const char *>> keys = {
        { LLM_KV_DSV41_CONFIG,                           "deepseek41.config" },
        { LLM_KV_DSV41_VOCAB_SIZE,                       "deepseek41.vocab_size" },
        { LLM_KV_DSV41_HIDDEN_SIZE,                      "deepseek41.hidden_size" },
        { LLM_KV_DSV41_MOE_INTERMEDIATE_SIZE,            "deepseek41.moe_intermediate_size" },
        { LLM_KV_DSV41_NUM_HIDDEN_LAYERS,                "deepseek41.num_hidden_layers" },
        { LLM_KV_DSV41_NUM_ATTENTION_HEADS,              "deepseek41.num_attention_heads" },
        { LLM_KV_DSV41_NUM_KEY_VALUE_HEADS,              "deepseek41.num_key_value_heads" },
        { LLM_KV_DSV41_HEAD_DIM,                         "deepseek41.head_dim" },
        { LLM_KV_DSV41_QK_ROPE_HEAD_DIM,                 "deepseek41.qk_rope_head_dim" },
        { LLM_KV_DSV41_Q_LORA_RANK,                      "deepseek41.q_lora_rank" },
        { LLM_KV_DSV41_O_LORA_RANK,                      "deepseek41.o_lora_rank" },
        { LLM_KV_DSV41_O_GROUPS,                         "deepseek41.o_groups" },
        { LLM_KV_DSV41_N_ROUTED_EXPERTS,                 "deepseek41.n_routed_experts" },
        { LLM_KV_DSV41_N_SHARED_EXPERTS,                 "deepseek41.n_shared_experts" },
        { LLM_KV_DSV41_NUM_EXPERTS_PER_TOK,              "deepseek41.num_experts_per_tok" },
        { LLM_KV_DSV41_MAX_POSITION_EMBEDDINGS,          "deepseek41.max_position_embeddings" },
        { LLM_KV_DSV41_SLIDING_WINDOW,                   "deepseek41.sliding_window" },
        { LLM_KV_DSV41_INDEX_N_HEADS,                    "deepseek41.index_n_heads" },
        { LLM_KV_DSV41_INDEX_HEAD_DIM,                   "deepseek41.index_head_dim" },
        { LLM_KV_DSV41_INDEX_TOPK,                       "deepseek41.index_topk" },
        { LLM_KV_DSV41_CANDIDATE_SOURCE_LAYER_ID,        "deepseek41.candidate_source_layer_id" },
        { LLM_KV_DSV41_CANDIDATE_TOPK_BLOCKS,            "deepseek41.candidate_topk_blocks" },
        { LLM_KV_DSV41_CANDIDATE_BLOCK_SIZE,             "deepseek41.candidate_block_size" },
        { LLM_KV_DSV41_HC_MULT,                          "deepseek41.hc_mult" },
        { LLM_KV_DSV41_HC_SINKHORN_ITERS,                "deepseek41.hc_sinkhorn_iters" },
        { LLM_KV_DSV41_ROPE_THETA,                       "deepseek41.rope_theta" },
        { LLM_KV_DSV41_COMPRESS_ROPE_THETA,              "deepseek41.compress_rope_theta" },
        { LLM_KV_DSV41_RMS_NORM_EPS,                     "deepseek41.rms_norm_eps" },
        { LLM_KV_DSV41_HC_EPS,                           "deepseek41.hc_eps" },
        { LLM_KV_DSV41_SWIGLU_LIMIT,                     "deepseek41.swiglu_limit" },
        { LLM_KV_DSV41_ROUTED_SCALING_FACTOR,            "deepseek41.routed_scaling_factor" },
        { LLM_KV_DSV41_SCORING_FUNC,                     "deepseek41.scoring_func" },
        { LLM_KV_DSV41_HIDDEN_ACT,                       "deepseek41.hidden_act" },
        { LLM_KV_DSV41_TOPK_METHOD,                      "deepseek41.topk_method" },
        { LLM_KV_DSV41_NORM_TOPK_PROB,                   "deepseek41.norm_topk_prob" },
        { LLM_KV_DSV41_COMPRESS_RATIOS,                  "deepseek41.compress_ratios" },
        { LLM_KV_DSV41_KV_SOURCE_LAYER_IDS,              "deepseek41.kv_source_layer_ids" },
        { LLM_KV_DSV41_INDEX_SOURCE_LAYER_IDS,           "deepseek41.index_source_layer_ids" },
        { LLM_KV_DSV41_ROPE_SCALING_FACTOR,              "deepseek41.rope_scaling.factor" },
        { LLM_KV_DSV41_ROPE_SCALING_BETA_FAST,           "deepseek41.rope_scaling.beta_fast" },
        { LLM_KV_DSV41_ROPE_SCALING_BETA_SLOW,           "deepseek41.rope_scaling.beta_slow" },
        { LLM_KV_DSV41_ROPE_SCALING_ORIG_CTX_LEN,        "deepseek41.rope_scaling.original_max_position_embeddings" },
        { LLM_KV_DSV41_ENGRAM_ENCODING,                  "deepseek41.engram.encoding" },
        { LLM_KV_DSV41_ENGRAM_LAYER_IDS,                 "deepseek41.engram.layer_ids" },
        { LLM_KV_DSV41_ENGRAM_ROWS,                      "deepseek41.engram.rows" },
        { LLM_KV_DSV41_ENGRAM_COMPRESSED_VOCAB_SIZE,     "deepseek41.engram.compressed_vocab_size" },
        { LLM_KV_DSV41_ENGRAM_PAD_ID,                    "deepseek41.engram.pad_id" },
        { LLM_KV_DSV41_ENGRAM_TOKEN_MAP,                 "deepseek41.engram.token_map" },
        { LLM_KV_DSV41_ENGRAM_PRIMES,                    "deepseek41.engram.primes" },
        { LLM_KV_DSV41_ENGRAM_MULTIPLIERS,               "deepseek41.engram.multipliers" },
    };

    for (const auto & item : keys) {
        check(kv(item.first) == item.second, std::string("metadata key mismatch: ") + item.second);
    }

    const LLM_TN tn(LLM_ARCH_DEEPSEEK41);
    check(tn(LLM_TENSOR_ENGRAM_EMBD,   "weight", 1).str() == "blk.1.engram_embd.weight", "Engram embedding tensor name failed");
    check(tn(LLM_TENSOR_ENGRAM_Q_NORM, "weight", 1).str() == "blk.1.engram_q_norm.weight", "Engram query norm tensor name failed");
    check(tn(LLM_TENSOR_ENGRAM_K_NORM, "weight", 1).str() == "blk.1.engram_k_norm.weight", "Engram key norm tensor name failed");
    check(tn(LLM_TENSOR_ENGRAM_KV,     "weight", 1).str() == "blk.1.engram_kv.weight", "Engram projection tensor name failed");

    return 0;
}
