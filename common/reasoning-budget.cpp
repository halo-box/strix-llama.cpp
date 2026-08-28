#include "reasoning-budget.h"
#include "common.h"
#include "trie.h"
#include "unicode.h"

#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct token_matcher {
    std::vector<llama_tokens> seqs;
    common_aho_corasick ac;
    size_t state = 0;

    token_matcher(const std::vector<llama_tokens> & seqs) : seqs(collect(seqs)), ac(build_trie(this->seqs)) {}

    static std::vector<llama_tokens> collect(const std::vector<llama_tokens> & seqs) {
        std::vector<llama_tokens> res;
        for (const auto & seq : seqs) {
            if (!seq.empty() && std::find(res.begin(), res.end(), seq) == res.end()) {
                res.push_back(seq);
            }
        }
        return res;
    }

    static common_trie build_trie(const std::vector<llama_tokens> & seqs) {
        common_trie t;
        for (const auto & seq : seqs) {
            t.insert(std::vector<uint32_t>(seq.begin(), seq.end()));
        }
        return t;
    }

    // Returns the index of the longest sequence ending at this token, or -1.
    int32_t advance(llama_token token) {
        state = ac.next(state, (uint32_t) token);
        const int32_t p = ac.match_pattern(state);
        if (p >= 0) {
            state = 0;
        }
        return p;
    }

    void reset() { state = 0; }
};

struct common_reasoning_budget_ctx {
    const llama_vocab * vocab;

    token_matcher start_matcher;
    token_matcher end_matcher;
    llama_tokens forced_tokens;

    int32_t budget;           // Maximum tokens in reasoning block.
    int32_t remaining;        // Tokens remaining in budget.

    common_reasoning_budget_state state;

    size_t force_pos;         // Next position in forced_tokens to force.

    int32_t end_match;        // Index into end_matcher.seqs that transitioned to DONE, or -1.

    struct soft_point_state {
        int32_t      threshold;
        llama_tokens tokens;
        bool         triggered;
        size_t       force_pos;
    };

    std::vector<soft_point_state> soft_points;
    size_t                        active_soft;

    llama_tokens intro_forced_tokens;
    size_t       intro_force_pos;

    int32_t grace_tokens;
    int32_t grace_remaining;
    bool    hard_pending_prev_nl;
};

static const char * common_reasoning_budget_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-budget";
}

static bool token_utf8_complete(const common_reasoning_budget_ctx * ctx, llama_token token) {
    if (ctx->vocab == nullptr) {
        return true;
    }
    const std::string piece = common_token_to_piece(ctx->vocab, token, false);
    return common_utf8_is_complete(piece);
}

static void common_reasoning_budget_begin_forcing(common_reasoning_budget_ctx * ctx, llama_token token) {
    ctx->end_matcher.reset();
    if (token_utf8_complete(ctx, token)) {
        ctx->state     = REASONING_BUDGET_FORCING;
        ctx->force_pos = 0;
    } else {
        ctx->state = REASONING_BUDGET_WAITING_UTF8;
    }
}

static void common_reasoning_budget_enter_hard_exhausted(common_reasoning_budget_ctx * ctx, llama_token token) {
    if (ctx->grace_tokens > 0) {
        ctx->state                = REASONING_BUDGET_HARD_PENDING;
        ctx->grace_remaining      = ctx->grace_tokens;
        ctx->hard_pending_prev_nl = false;
        ctx->end_matcher.reset();
        COM_TRC("budget exhausted, waiting up to %d tokens for a paragraph break\n", ctx->grace_tokens);
        return;
    }

    common_reasoning_budget_begin_forcing(ctx, token);
    COM_TRC("%s", "budget exhausted, forcing end sequence\n");
}

static void common_reasoning_budget_activate(common_reasoning_budget_ctx * ctx) {
    ctx->remaining = ctx->budget;
    for (auto & sp : ctx->soft_points) {
        sp.triggered = false;
    }
    ctx->end_match = -1;

    if (!ctx->intro_forced_tokens.empty()) {
        ctx->state           = REASONING_BUDGET_INTRO_FORCING;
        ctx->intro_force_pos = 0;
        COM_TRC("activated, budget=%d tokens, forcing intro message\n", ctx->budget);
    } else if (ctx->remaining <= 0) {
        ctx->state     = REASONING_BUDGET_FORCING;
        ctx->force_pos = 0;
        COM_TRC("%s", "budget=0, forcing immediately\n");
    } else {
        ctx->state = REASONING_BUDGET_COUNTING;
        COM_TRC("activated, budget=%d tokens\n", ctx->budget);
    }
}

static void common_reasoning_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    switch (ctx->state) {
        case REASONING_BUDGET_IDLE:
        {
            if (ctx->start_matcher.advance(token) >= 0) {
                common_reasoning_budget_activate(ctx);
            }
            break;
        }
        case REASONING_BUDGET_INTRO_FORCING:
            ctx->intro_force_pos++;
            if (ctx->intro_force_pos >= ctx->intro_forced_tokens.size()) {
                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "intro complete, budget=0, forcing immediately\n");
                } else {
                    ctx->state = REASONING_BUDGET_COUNTING;
                    COM_TRC("%s", "intro complete, resuming countdown\n");
                }
            }
            break;
        case REASONING_BUDGET_COUNTING:
        {
            const int32_t match = ctx->end_matcher.advance(token);
            if (match >= 0) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_TRC("%s", "deactivated (natural end)\n");
                break;
            }

            ctx->remaining--;
            if (ctx->remaining <= 0) {
                common_reasoning_budget_enter_hard_exhausted(ctx, token);
                break;
            }

            for (size_t i = 0; i < ctx->soft_points.size(); i++) {
                auto & sp = ctx->soft_points[i];
                if (!sp.triggered && ctx->remaining <= sp.threshold) {
                    ctx->active_soft = i;
                    ctx->state       = REASONING_BUDGET_SOFT_PENDING;
                    COM_TRC("soft threshold reached (remaining=%d, point %zu), waiting for newline\n", ctx->remaining,
                            i);
                    break;
                }
            }
            break;
        }
        case REASONING_BUDGET_SOFT_PENDING:
        {
            const int32_t match = ctx->end_matcher.advance(token);
            if (match >= 0) {
                ctx->state     = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_TRC("%s", "deactivated (natural end)\n");
                break;
            }

            ctx->remaining--;
            if (ctx->remaining <= 0) {
                COM_TRC("%s", "budget exhausted before newline, soft warning skipped\n");
                common_reasoning_budget_enter_hard_exhausted(ctx, token);
                break;
            }

            if (ctx->vocab != nullptr) {
                const std::string piece = common_token_to_piece(ctx->vocab, token, false);
                if (piece.find('\n') != std::string::npos) {
                    ctx->state   = REASONING_BUDGET_SOFT_FORCING;
                    auto & sp    = ctx->soft_points[ctx->active_soft];
                    sp.force_pos = 0;
                    sp.triggered = true;
                    COM_TRC("%s", "newline boundary found, forcing soft warning\n");
                }
            }
            break;
        }
        case REASONING_BUDGET_SOFT_FORCING:
        {
            auto & sp = ctx->soft_points[ctx->active_soft];
            sp.force_pos++;
            if (sp.force_pos >= sp.tokens.size()) {
                ctx->state = REASONING_BUDGET_COUNTING;
                COM_TRC("%s", "soft warning complete, resuming countdown\n");
            }
            break;
        }
        case REASONING_BUDGET_HARD_PENDING:
        {
            const int32_t match = ctx->end_matcher.advance(token);
            if (match >= 0) {
                ctx->state     = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_TRC("%s", "deactivated (natural end)\n");
                break;
            }

            ctx->grace_remaining--;

            const std::string piece =
                ctx->vocab != nullptr ? common_token_to_piece(ctx->vocab, token, false) : std::string();
            const bool paragraph_boundary = piece.find("\n\n") != std::string::npos ||
                                            (ctx->hard_pending_prev_nl && !piece.empty() && piece[0] == '\n');
            ctx->hard_pending_prev_nl     = !piece.empty() && piece.back() == '\n';

            if (paragraph_boundary) {
                common_reasoning_budget_begin_forcing(ctx, token);
                COM_TRC("%s", "paragraph boundary found, forcing end sequence\n");
            } else if (ctx->grace_remaining <= 0) {
                common_reasoning_budget_begin_forcing(ctx, token);
                COM_TRC("%s", "grace period expired, forcing end sequence\n");
            }
            break;
        }
        case REASONING_BUDGET_WAITING_UTF8:
        {
            const int32_t match = ctx->end_matcher.advance(token);
            if (match >= 0) {
                ctx->state     = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_TRC("%s", "deactivated (natural end)\n");
                break;
            }

            if (token_utf8_complete(ctx, token)) {
                common_reasoning_budget_begin_forcing(ctx, token);
                COM_TRC("%s", "UTF-8 complete, now forcing end sequence\n");
            }
            break;
        }
        case REASONING_BUDGET_FORCING:
        {
            // Track the end sequence in forced_tokens for grammar replay.
            const int32_t match = ctx->end_matcher.advance(token);
            ctx->force_pos++;
            if (ctx->force_pos >= ctx->forced_tokens.size()) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_TRC("%s", "forced sequence complete, done\n");
            }
            break;
        }
        case REASONING_BUDGET_DONE:
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->end_matcher.reset();
                common_reasoning_budget_activate(ctx);
            }
            break;
    }
}

static void common_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    llama_token forced;

    if (ctx->state == REASONING_BUDGET_FORCING) {
        if (ctx->force_pos >= ctx->forced_tokens.size()) {
            return;
        }
        forced = ctx->forced_tokens[ctx->force_pos];
    } else if (ctx->state == REASONING_BUDGET_SOFT_FORCING) {
        const auto & sp = ctx->soft_points[ctx->active_soft];
        if (sp.force_pos >= sp.tokens.size()) {
            return;
        }
        forced = sp.tokens[sp.force_pos];
    } else if (ctx->state == REASONING_BUDGET_INTRO_FORCING) {
        if (ctx->intro_force_pos >= ctx->intro_forced_tokens.size()) {
            return;
        }
        forced = ctx->intro_forced_tokens[ctx->intro_force_pos];
    } else {
        return;
    }

    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id != forced) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
}

static void common_reasoning_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    ctx->state = REASONING_BUDGET_IDLE;
    ctx->remaining = ctx->budget;
    ctx->start_matcher.reset();
    ctx->end_matcher.reset();
    ctx->force_pos = 0;
    ctx->end_match = -1;
    for (auto & sp : ctx->soft_points) {
        sp.triggered = false;
        sp.force_pos = 0;
    }
    ctx->active_soft          = 0;
    ctx->intro_force_pos      = 0;
    ctx->grace_remaining      = ctx->grace_tokens;
    ctx->hard_pending_prev_nl = false;
}

static struct llama_sampler * common_reasoning_budget_init_state(
    const struct llama_vocab *                              vocab,
    const std::vector<llama_tokens> &                       start_seqs,
    const std::vector<llama_tokens> &                       end_seqs,
    const llama_tokens &                                    forced_tokens,
    const std::vector<common_reasoning_budget_soft_point> & soft_points,
    const llama_tokens &                                    intro_forced_tokens,
    int32_t                                                 budget,
    int32_t                                                 grace_tokens,
    common_reasoning_budget_state                           initial_state);

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl);

static void common_reasoning_budget_free(struct llama_sampler * smpl) {
    delete (common_reasoning_budget_ctx *) smpl->ctx;
}

static struct llama_sampler_i common_reasoning_budget_i = {
    /* .name              = */ common_reasoning_budget_name,
    /* .accept            = */ common_reasoning_budget_accept,
    /* .apply             = */ common_reasoning_budget_apply,
    /* .reset             = */ common_reasoning_budget_reset,
    /* .clone             = */ common_reasoning_budget_clone,
    /* .free              = */ common_reasoning_budget_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx(*ctx)
    );
}

static struct llama_sampler * common_reasoning_budget_init_state(
    const struct llama_vocab *                              vocab,
    const std::vector<llama_tokens> &                       start_seqs,
    const std::vector<llama_tokens> &                       end_seqs,
    const llama_tokens &                                    forced_tokens,
    const std::vector<common_reasoning_budget_soft_point> & soft_points,
    const llama_tokens &                                    intro_forced_tokens,
    int32_t                                                 budget,
    int32_t                                                 grace_tokens,
    common_reasoning_budget_state                           initial_state) {
    if (initial_state == REASONING_BUDGET_COUNTING && budget <= 0) {
        initial_state = REASONING_BUDGET_FORCING;
    }

    std::vector<common_reasoning_budget_ctx::soft_point_state> soft_pts;
    for (const auto & p : soft_points) {
        if (!p.tokens.empty() && p.threshold >= 0) {
            soft_pts.push_back({ p.threshold, p.tokens, false, 0 });
        }
    }
    std::sort(soft_pts.begin(), soft_pts.end(),
              [](const common_reasoning_budget_ctx::soft_point_state & a,
                 const common_reasoning_budget_ctx::soft_point_state & b) { return a.threshold > b.threshold; });

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx{
            /* .vocab                = */ vocab,
            /* .start_matcher        = */ token_matcher(start_seqs),
            /* .end_matcher          = */ token_matcher(end_seqs),
            /* .forced_tokens        = */ forced_tokens,
            /* .budget               = */ budget,
            /* .remaining            = */ budget,
            /* .state                = */ initial_state,
            /* .force_pos            = */ 0,
            /* .end_match            = */ -1,
            /* .soft_points          = */ std::move(soft_pts),
            /* .active_soft          = */ 0,
            /* .intro_forced_tokens  = */ intro_forced_tokens,
            /* .intro_force_pos      = */ 0,
            /* .grace_tokens         = */ grace_tokens,
            /* .grace_remaining      = */ grace_tokens,
            /* .hard_pending_prev_nl = */ false,
        });
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab *                              vocab,
        const std::vector<llama_tokens> &                       start_seqs,
        const std::vector<llama_tokens> &                       end_seqs,
        const llama_tokens &                                    forced_tokens,
        const std::vector<common_reasoning_budget_soft_point> & soft_points,
        const llama_tokens &                                    intro_forced_tokens,
        int32_t                                                 budget,
        int32_t                                                 grace_tokens,
        common_reasoning_budget_state                           initial_state) {
    return common_reasoning_budget_init_state(vocab, start_seqs, end_seqs, forced_tokens, soft_points,
                                              intro_forced_tokens, budget, grace_tokens, initial_state);
}

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return REASONING_BUDGET_IDLE;
    }
    return ((const common_reasoning_budget_ctx *)smpl->ctx)->state;
}

const llama_tokens * common_reasoning_budget_get_end_match(const struct llama_sampler * smpl) {
    if (!smpl) {
        return nullptr;
    }

    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;
    if (ctx->end_match < 0) {
        return nullptr;
    }

    return &ctx->end_matcher.seqs[ctx->end_match];
}

bool common_reasoning_budget_force(struct llama_sampler * smpl) {
    if (!smpl) {
        return false;
    }

    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    if (ctx->state != REASONING_BUDGET_COUNTING && ctx->state != REASONING_BUDGET_INTRO_FORCING &&
        ctx->state != REASONING_BUDGET_SOFT_PENDING && ctx->state != REASONING_BUDGET_SOFT_FORCING &&
        ctx->state != REASONING_BUDGET_HARD_PENDING) {
        return false;
    }

    ctx->state = REASONING_BUDGET_FORCING;
    ctx->force_pos = 0;
    ctx->end_matcher.reset();
    COM_TRC("%s", "forced into forcing state (manual transition)\n");

    return true;
}
