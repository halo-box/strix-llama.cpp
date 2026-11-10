// The reserve views of the hybrid-idx memory (llama_context::sched_reserve): init_full views every stream of the cache,
// init_full_n_kv(n_kv) limits the views to n_kv cells (reserve pass B), and init_full_n_kv(n_kv, 1) to the first stream as
// well (the one-stream reserve graphs of a split cache with several sequences: a one-sequence ubatch there is a one-stream
// graph, which a view of every stream cannot build; Prove's -np 2 split run at ub8192 grew past the reserve on them).
#include "qsa-mem-fixture.h"

#include <cstdio>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static const llama_memory_hybrid_idx_context * as_idx(const llama_memory_context_ptr & p) {
    return static_cast<const llama_memory_hybrid_idx_context *>(p.get());
}

static void check_views(uint32_t kv_size, uint32_t n_seq_max, bool unified) {
    qsa_fixture fx(kv_size, n_seq_max, unified);
    const uint32_t n_stream_cache = unified ? 1 : n_seq_max;

    auto full = fx.mem->init_full();
    CHECK(full && full->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
    CHECK(as_idx(full)->get_n_stream() == n_stream_cache);
    CHECK(as_idx(full)->get_idx()->get_n_kv() == kv_size);
    CHECK(as_idx(full)->get_attn()->get_n_kv() == kv_size);

    // pass B: n_kv cells, every stream
    auto b = fx.mem->init_full_n_kv(kv_size / 2);
    CHECK(b && b->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
    CHECK(as_idx(b)->get_n_stream() == n_stream_cache);
    CHECK(as_idx(b)->get_idx()->get_n_kv() == kv_size / 2);
    CHECK(as_idx(b)->get_attn()->get_n_kv() == kv_size / 2);

    // the one-stream view at the full per-stream n_kv
    auto one = fx.mem->init_full_n_kv(kv_size, 1);
    CHECK(one && one->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
    CHECK(as_idx(one)->get_n_stream() == 1);
    CHECK(as_idx(one)->get_idx()->get_n_kv() == kv_size);
    CHECK(as_idx(one)->get_attn()->get_n_kv() == kv_size);

    // n_kv past the cache is clamped, n_stream past the cache's streams is the cache's
    auto big = fx.mem->init_full_n_kv(2*kv_size, n_stream_cache + 3);
    CHECK(as_idx(big)->get_n_stream() == n_stream_cache);
    CHECK(as_idx(big)->get_idx()->get_n_kv() == kv_size);

    fprintf(stderr, "views kv_size=%u n_seq_max=%u unified=%d: full %u streams, pass B %u streams, one-stream %u\n", kv_size, n_seq_max,
            (int) unified, as_idx(full)->get_n_stream(), as_idx(b)->get_n_stream(), as_idx(one)->get_n_stream());
}

// The compact decision of the reserve graphs (llama_memory_hybrid_idx_context::qsa_graph_compact, the call qwen4exp's
// build_qsa_top_k makes with LLM_GRAPH_QSA_RESERVE_GRAPH): sched_reserve builds every reserve graph on init_full, which
// views the live cells when a reserve runs mid-session (sched_need_reserve). The pp reserve ubatch (graph_reserve: n_seqs
// sequences 0 .. n_seqs - 1 in one unified stream, all-zero positions, ubatch_prepare_reserve's seq ids) must get the
// verdict it gets on the empty cache of init, whatever the cells hold: an image in one sequence next to text in another
// (QSA_NC_2D_MULTI_SEQ), or a block shared through seq_cp (QSA_NC_SHARED_BLOCK). Runtime graphs keep the live verdict.
static llama_ubatch reserve_ubatch(llama_batch_allocr & balloc, uint32_t n_tokens, uint32_t n_seqs) {
    llama_ubatch u = balloc.ubatch_reserve(n_tokens / n_seqs, n_seqs);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < u.n_seq_tokens; ++t) {
            const uint32_t i = s * u.n_seq_tokens + t;
            u.n_seq_id[i] = 1;
            u.seq_id[i]   = &u.seq_id_unq[s];
        }
    }
    return u;
}

static void check_reserve_compact(const char * name, qsa_fixture & fx, int want_live) {
    const uint32_t r = 4;
    auto full = fx.mem->init_full();
    CHECK(full && full->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
    const auto * m = as_idx(full);
    llama_batch_allocr balloc(4);
    for (const uint32_t n_tokens : { 4096u, 512u, 4u }) {
        const llama_ubatch u = reserve_ubatch(balloc, n_tokens, fx.n_seq_max);
        const int  live    = m->qsa_compact_reason(u, r);
        const bool reserve = m->qsa_graph_compact(u, r, true);
        const bool runtime = m->qsa_graph_compact(u, r, false);
        fprintf(stderr, "%s: reserve ubatch of %u tokens, %u seqs: live cells %s, reserve graph %s, runtime graph %s\n", name, n_tokens,
                fx.n_seq_max, qsa_compact_reason_name(live), reserve ? "compact" : "non-compact", runtime ? "compact" : "non-compact");
        CHECK(live == want_live);
        CHECK(reserve);                                    // the init verdict, whatever the live cells hold
        CHECK(runtime == m->qsa_scalar_visibility(u, r));  // a runtime graph keeps the live verdict
    }
}

static void check_reserve_compact_all() {
    {   // init: the empty cache, where the reserve and live verdicts agree
        qsa_fixture fx(8192, 4, true);
        check_reserve_compact("empty", fx, QSA_COMPACT_OK);
    }
    {   // slot 0 holds an image, slot 1 text: a sequence of the stream holds 2-D cells while another has cells
        qsa_fixture fx(8192, 4, true);
        CHECK(fx.feed(fx.text(0, 100), 512));
        CHECK(fx.feed(fx.image(0, 32, 32), 512));
        CHECK(fx.feed(fx.text(0, 100), 512));
        CHECK(fx.feed(fx.text(1, 3000), 512));
        check_reserve_compact("image+text", fx, QSA_NC_2D_MULTI_SEQ);
    }
    {   // slot 1 copied slot 0 (seq_cp) when it ended inside a block (3001 tokens, ratio 4), then slot 0 went on: the block
        // 3000 .. 3003 holds sequence 0 under {0, 1} and {0} (R1-CB, test-qsa-visibility "seqcp")
        qsa_fixture fx(8192, 4, true);
        CHECK(fx.feed(fx.text(0, 3001), 512));
        fx.mem->seq_cp(0, 1, -1, -1);
        fx.next_pos[1] = fx.next_pos[0];
        CHECK(fx.feed(fx.text(0, 40), 512));
        check_reserve_compact("seq_cp", fx, QSA_NC_SHARED_BLOCK);
    }
}

int main() {
    check_reserve_compact_all();
    check_views(4096, 2, false);
    check_views(4096, 4, false);
    check_views(4096, 2, true);
    check_views(4096, 1, false);
    if (n_fail) {
        fprintf(stderr, "test-qsa-reserve-view: %d FAILED\n", n_fail);
        return 1;
    }
    fprintf(stderr, "test-qsa-reserve-view: OK\n");
    return 0;
}
