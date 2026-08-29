#pragma once

#include "llama.h"

#include "common.h"

#include <cstdint>
#include <vector>

enum common_reasoning_budget_state {
    REASONING_BUDGET_IDLE,           // Waiting for start sequence.
    REASONING_BUDGET_INTRO_FORCING,  // Forcing the intro message.
    REASONING_BUDGET_COUNTING,       // Counting down tokens.
    REASONING_BUDGET_SOFT_PENDING,   // Waiting for a soft-warning newline boundary.
    REASONING_BUDGET_SOFT_FORCING,   // Forcing the soft warning message.
    REASONING_BUDGET_HARD_PENDING,   // Waiting for a hard-cutoff paragraph boundary.
    REASONING_BUDGET_FORCING,        // Forcing budget message and end sequence.
    REASONING_BUDGET_WAITING_UTF8,   // Waiting for UTF-8 completion before forcing.
    REASONING_BUDGET_DONE,           // Passthrough forever.
};

struct common_reasoning_budget_soft_point {
    int32_t      threshold;  // Fire when remaining <= threshold.
    llama_tokens tokens;     // Message forced at this point. Empty disables it.
};

// Creates a sampler that limits generation inside a reasoning block.
// Intro, soft, and forced tokens do not count against the budget.
//
// State machine:
// IDLE -> INTRO_FORCING -> COUNTING -> SOFT_PENDING -> SOFT_FORCING ->
// COUNTING -> HARD_PENDING -> WAITING_UTF8 -> FORCING -> DONE
//
// A soft point fires once at the next newline after its threshold is crossed.
// The hard cutoff takes priority when the budget is exhausted.
//
// A positive grace_tokens value waits for a paragraph boundary after exhaustion.
// It forces after at most grace_tokens more tokens.
//
// Parameters:
//   vocab               - vocabulary for UTF-8 and paragraph boundary detection
//   start_seqs          - sequences that activate counting
//   end_seqs            - sequences that naturally deactivate
//   forced_tokens       - sequence forced when the budget expires
//   soft_points         - soft warning points
//   intro_forced_tokens - sequence forced when the block starts
//   budget              - max generated tokens in the reasoning block
//   grace_tokens        - max tokens to wait for a paragraph boundary
//   initial_state       - initial state
struct llama_sampler * common_reasoning_budget_init(
    const struct llama_vocab *                              vocab,
    const std::vector<llama_tokens> &                       start_seqs,
    const std::vector<llama_tokens> &                       end_seqs,
    const llama_tokens &                                    forced_tokens,
    const std::vector<common_reasoning_budget_soft_point> & soft_points,
    const llama_tokens &                                    intro_forced_tokens,
    int32_t                                                 budget,
    int32_t                                                 grace_tokens  = 0,
    common_reasoning_budget_state                           initial_state = REASONING_BUDGET_IDLE,
    bool                                                    intro_once    = false);

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl);

// The end sequence that transitioned the sampler to DONE, or nullptr if none
// was recorded. Cleared when a new start sequence re-arms the sampler.
const llama_tokens * common_reasoning_budget_get_end_match(const struct llama_sampler * smpl);

// Manually transition the reasoning budget sampler into the FORCING state.
// Returns true if the transition occurred.
bool common_reasoning_budget_force(struct llama_sampler * smpl);
