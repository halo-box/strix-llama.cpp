// Unit checks of the llama-side QSA pieces (no model file), one section per ctest (tests/CMakeLists.txt registers each
// section under its own test name):
//
//   dup-key           the duplicate-key term of the compact predicate reads a kept count (src/prefix.h)
//   reserve-view      the reserve views of the hybrid-idx memory and the reserve graphs' compact decision
//   reserve-passes    the QSA reserve pass list builds the one-stream shortcut graph of a split multi-sequence context
//   noncompact-check  the graph-build check fails non-compact ubatches only where block selection is eligible
//   supported-types   the "supported here" list of the QSA init error probes nine types without running out of its context
//
// usage: test-qsa-checks [section]   (no section: every section)

#include "qsa-mem-fixture.h"
#include "../src/llama-context.h"
#include "../src/models/qwen4exp-qsa-check.h"
#include "../src/prefix.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;

// printf-style message; the first 20 failures of a section are printed
#define CHECK(cond, ...) do { if (!(cond)) { if (g_fail++ < 20) { printf("  FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); } } } while (0)
#define CHECK0(cond) CHECK(cond, "%s", "check failed")

// ---- dup-key ---------------------------------------------------------------------------------------------------------
// The duplicate-key term of the QSA compact predicate (src/prefix.h qsa_seq_has_dup_key, QSA_NC_DUP_KEY) runs for every
// sequence of a ubatch on every graph build and every can_reuse of a multi-sequence context (qwen4exp can_reuse ->
// qsa_graph_compact -> qsa_compact_reason -> qsa_compact_visibility_stream). It used to walk the whole std::set of
// (pos, cell) pairs of each such sequence: with 4 server slots sharing a long seq_cp prompt, every decode step visited
// every shared cell 4 times. It now reads llama_kv_cells::seq_pos_n_dup, a count kept by the set's own insert / erase.
//
//   exact: random cell edits (pos_set + seq_add, seq_add to a used cell, seq_rm, seq_keep, rm, pos_add with removal,
//          pos_div, set from a cp copy, reset) on a small cache with colliding positions and (y, x) extents; after every
//          edit seq_pos_n_dup equals the count recomputed from seq_pos_get, and qsa_seq_has_dup_key equals the old full
//          walk (kept below as the oracle) for pos_2d false and true;
//   work:  2^20 cells shared by 4 sequences at unique positions (the seq_cp server layout), plus one sequence with an
//          M-RoPE image: qsa_seq_has_dup_key visits 0 set entries for the 4 text sequences and still finds (or rules
//          out) the image's duplicate keys. The timings are printed, not asserted.
// Exit status 0 = every check held.

namespace dup_key {

// the walk qsa_seq_has_dup_key did before the count existed (prefix.h at e31548e7d), the oracle of the exact part
static bool dup_key_walk(const llama_kv_cells & cells, llama_seq_id seq, bool pos_2d) {
    const auto & sp = cells.seq_pos_get(seq);
    std::vector<std::pair<llama_pos, llama_pos>> yx;
    for (auto it = sp.begin(); it != sp.end();) {
        auto end = it;
        size_t n = 0;
        while (end != sp.end() && end->first == it->first) { ++end; ++n; }
        if (n > 1) {
            if (!pos_2d) { return true; }
            yx.clear();
            for (auto c = it; c != end; ++c) {
                const auto & e = cells.ext_get(c->second);
                yx.emplace_back(e.y, e.x);
            }
            std::sort(yx.begin(), yx.end());
            if (std::adjacent_find(yx.begin(), yx.end()) != yx.end()) { return true; }
        }
        it = end;
    }
    return false;
}

static uint32_t n_dup_recount(const llama_kv_cells & cells, llama_seq_id s) {
    const auto & sp = cells.seq_pos_get(s);
    uint32_t n = 0;
    for (auto it = sp.begin(); it != sp.end(); ++it) {
        if (it != sp.begin() && std::prev(it)->first == it->first) { n++; }
    }
    return n;
}

static constexpr int N_SEQ = 4;

static void check_all(const llama_kv_cells & cells, const char * what, int step) {
    for (llama_seq_id s = 0; s < N_SEQ; ++s) {
        const uint32_t want = n_dup_recount(cells, s);
        CHECK(cells.seq_pos_n_dup(s) == want, "%s step %d seq %d: seq_pos_n_dup %u, recount %u", what, step, s, cells.seq_pos_n_dup(s), want);
        for (bool p2 : { false, true }) {
            size_t walked = 0;
            const bool got = qsa_seq_has_dup_key(cells, s, p2, &walked);
            CHECK(got == dup_key_walk(cells, s, p2), "%s step %d seq %d pos_2d %d: dup key %d, the full walk says %d", what, step, s, (int) p2, (int) got, (int) !got);
        }
    }
}

static void test_exact() {
    std::mt19937 rng(12345);
    const uint32_t N = 512;
    llama_kv_cells cells;
    cells.resize(N);
    auto rnd = [&](uint32_t n) { return (uint32_t) (rng() % n); };
    int n_dup_seen = 0;
    for (int step = 0; step < 40000; ++step) {
        const uint32_t i = rnd(N);
        const int op = rnd(10);
        if (cells.is_empty(i)) {
            // a new cell: positions collide (range 0..63), an image cell gets a (y, x) on a 3 x 3 grid
            cells.pos_set(i, (llama_pos) rnd(64));
            llama_kv_cell_ext e;
            e.y = (llama_pos) rnd(3);
            e.x = (llama_pos) rnd(3);
            cells.ext_set(i, e);
            cells.seq_add(i, (llama_seq_id) rnd(N_SEQ));
        } else if (op == 0) {
            const llama_seq_id s = (llama_seq_id) rnd(N_SEQ);
            if (!cells.seq_has(i, s)) { cells.seq_add(i, s); }
        } else if (op == 1) {
            const llama_seq_id s = (llama_seq_id) rnd(N_SEQ);
            if (cells.seq_has(i, s)) { cells.seq_rm(i, s); }
        } else if (op == 2) {
            cells.seq_keep(i, (llama_seq_id) rnd(N_SEQ));
        } else if (op == 3) {
            cells.rm(i);
        } else if (op == 4) {
            cells.pos_add(i, (llama_pos) rnd(9) - 6);   // may remove the cell (pos < 0)
        } else if (op == 5) {
            cells.pos_div(i, 2);
        } else if (op == 6 && step % 97 == 0) {
            // save / restore of a range, as state_write / state_read do
            const uint32_t i0 = rnd(N - 16);
            llama_kv_cells saved = cells.cp(i0, 16);
            for (uint32_t j = i0; j < i0 + 16; ++j) {
                if (!cells.is_empty(j)) { cells.rm(j); }
            }
            cells.set(i0, saved);
        } else if (op == 7 && step % 4999 == 0) {
            cells.reset();
        }
        cells.reset_shift();
        for (llama_seq_id s = 0; s < N_SEQ; ++s) { n_dup_seen += cells.seq_pos_n_dup(s) > 0; }
        check_all(cells, "exact", step);
    }
    // a copy of the whole cache (llama_kv_cells is copied by value, e.g. the hybrid memory's simulated prepare)
    llama_kv_cells copy = cells;
    check_all(copy, "copy", 0);
    printf("exact: 40000 edits, %d (step, sequence) pairs with a repeated position\n", n_dup_seen);
    CHECK(n_dup_seen > 1000, "the edits rarely made a repeated position: the exact part proves little");
}

static void test_work() {
    const uint32_t N = 1u << 20;
    llama_kv_cells cells;
    cells.resize(N + 1200);
    // 4 sequences sharing every cell at unique positions (a seq_cp prompt in a unified cache, -np 4)
    for (uint32_t i = 0; i < N; ++i) {
        cells.pos_set(i, (llama_pos) i);
        for (llama_seq_id s = 0; s < N_SEQ; ++s) { cells.seq_add(i, s); }
    }
    // sequence 4: an M-RoPE image at one position (32 x 32 grid: no duplicate key), then a duplicate (y, x)
    const llama_seq_id img = 4;
    for (uint32_t k = 0; k < 1024; ++k) {
        const uint32_t i = N + k;
        cells.pos_set(i, 7);
        llama_kv_cell_ext e;
        e.y = (llama_pos) (k / 32);
        e.x = (llama_pos) (k % 32);
        cells.ext_set(i, e);
        cells.seq_add(i, img);
    }
    for (llama_seq_id s = 0; s < N_SEQ; ++s) {
        CHECK(cells.seq_pos_get(s).size() == N, "seq %d holds %zu cells", s, cells.seq_pos_get(s).size());
        CHECK(cells.seq_pos_n_dup(s) == 0, "seq %d: seq_pos_n_dup %u on unique positions", s, cells.seq_pos_n_dup(s));
    }
    const int reps = 200;
    size_t walked = 0;
    const auto t0 = std::chrono::steady_clock::now();
    bool any = false;
    for (int r = 0; r < reps; ++r) {
        for (llama_seq_id s = 0; s < N_SEQ; ++s) {
            any |= qsa_seq_has_dup_key(cells, s, true, &walked);
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    printf("work: %d sequences x %u shared cells: %zu set entries visited per decode-step check (want 0), %.3f us per step\n",
           N_SEQ, N, walked / reps, us);
    CHECK(!any, "a duplicate key on unique positions");
    CHECK(walked == 0, "the text sequences walked %zu set entries per step: the check must not visit them", walked / reps);

    size_t w_img = 0;
    CHECK(!qsa_seq_has_dup_key(cells, img, true, &w_img), "the 32 x 32 image has no duplicate (pos, y, x) key");
    CHECK(qsa_seq_has_dup_key(cells, img, false, nullptr), "the image repeats its position: a duplicate key when the key is pos alone");
    CHECK(w_img > 0, "the image sequence must still be compared on (y, x)");
    // one more cell at (7, 3, 5): a duplicate key
    const uint32_t i = N + 1024;
    cells.pos_set(i, 7);
    llama_kv_cell_ext e;
    e.y = 3;
    e.x = 5;
    cells.ext_set(i, e);
    cells.seq_add(i, img);
    CHECK(qsa_seq_has_dup_key(cells, img, true, nullptr), "the image now holds (7, 3, 5) twice");
    printf("work: image sequence (1024 cells at one position): %zu entries compared, duplicate key found after the extra cell\n", w_img);
}

static int run() {
    test_exact();
    test_work();
    printf("test-qsa-dup-key: %s (%d failed checks)\n", g_fail == 0 ? "OK" : "FAILED", g_fail);
    return g_fail == 0 ? 0 : 1;
}

} // namespace dup_key

// ---- reserve-view ----------------------------------------------------------------------------------------------------
// The reserve views of the hybrid-idx memory (llama_context::sched_reserve): init_full views every stream of the cache,
// init_full_n_kv(n_kv) limits the views to n_kv cells (reserve pass B), and init_full_n_kv(n_kv, 1) to the first stream as
// well (the one-stream reserve graphs of a split cache with several sequences: a one-sequence ubatch there is a one-stream
// graph, which a view of every stream cannot build; Prove's -np 2 split run at ub8192 grew past the reserve on them).

namespace reserve_view {

static const llama_memory_hybrid_idx_context * as_idx(const llama_memory_context_ptr & p) {
    return static_cast<const llama_memory_hybrid_idx_context *>(p.get());
}

static void check_views(uint32_t kv_size, uint32_t n_seq_max, bool unified) {
    qsa_fixture fx(kv_size, n_seq_max, unified);
    const uint32_t n_stream_cache = unified ? 1 : n_seq_max;

    auto full = fx.mem->init_full();
    CHECK0(full && full->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
    CHECK0(as_idx(full)->get_n_stream() == n_stream_cache);
    CHECK0(as_idx(full)->get_idx()->get_n_kv() == kv_size);
    CHECK0(as_idx(full)->get_attn()->get_n_kv() == kv_size);

    // pass B: n_kv cells, every stream
    auto b = fx.mem->init_full_n_kv(kv_size / 2);
    CHECK0(b && b->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
    CHECK0(as_idx(b)->get_n_stream() == n_stream_cache);
    CHECK0(as_idx(b)->get_idx()->get_n_kv() == kv_size / 2);
    CHECK0(as_idx(b)->get_attn()->get_n_kv() == kv_size / 2);

    // the one-stream view at the full per-stream n_kv
    auto one = fx.mem->init_full_n_kv(kv_size, 1);
    CHECK0(one && one->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
    CHECK0(as_idx(one)->get_n_stream() == 1);
    CHECK0(as_idx(one)->get_idx()->get_n_kv() == kv_size);
    CHECK0(as_idx(one)->get_attn()->get_n_kv() == kv_size);

    // n_kv past the cache is clamped, n_stream past the cache's streams is the cache's
    auto big = fx.mem->init_full_n_kv(2*kv_size, n_stream_cache + 3);
    CHECK0(as_idx(big)->get_n_stream() == n_stream_cache);
    CHECK0(as_idx(big)->get_idx()->get_n_kv() == kv_size);

    printf("views kv_size=%u n_seq_max=%u unified=%d: full %u streams, pass B %u streams, one-stream %u\n", kv_size, n_seq_max,
            (int) unified, as_idx(full)->get_n_stream(), as_idx(b)->get_n_stream(), as_idx(one)->get_n_stream());
}

// The compact decision of the reserve graphs (llama_memory_hybrid_idx_context::qsa_graph_compact, the call qwen4exp's
// build_qsa_top_k makes with LLM_GRAPH_QSA_RESERVE_GRAPH): sched_reserve builds every reserve graph on init_full, which
// views the live cells when a reserve runs mid-session (sched_need_reserve). The pp reserve ubatch (graph_reserve: n_seqs
// sequences 0 .. n_seqs - 1 in one unified stream, all-zero positions, ubatch_prepare_reserve's seq ids) must get the
// verdict it gets on the empty cache of init, whatever the cells hold: an image in one sequence next to text in another
// (QSA_NC_2D_MULTI_SEQ), or a block shared through seq_cp (QSA_NC_SHARED_BLOCK). Runtime graphs keep the live verdict;
// the reservation context itself (init_full) sees no cell (#262), so its own predicate gives the empty cache's verdict too.
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
    CHECK0(full && full->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
    const auto * m = as_idx(full);
    llama_batch_allocr balloc(4);
    for (const uint32_t n_tokens : { 4096u, 512u, 4u }) {
        const llama_ubatch u = reserve_ubatch(balloc, n_tokens, fx.n_seq_max);
        // the live verdict: the predicate over the cells the runtime graph reads (init_full is a reservation context, which
        // itself sees no cell since #262, so it is asked of the memory's indexer cache directly)
        const int  live    = llama_memory_hybrid_idx::qsa_compact_reason(fx.mem->get_mem_idx(), m->get_n_stream(),
                m->get_idx()->get_n_kv(), r, u);
        const bool reserve = m->qsa_graph_compact(u, r, true);
        const bool ctx_res = m->qsa_graph_compact(u, r, false);
        printf("%s: reserve ubatch of %u tokens, %u seqs: live cells %s, reserve graph %s, reservation context %s\n", name,
                n_tokens, fx.n_seq_max, qsa_compact_reason_name(live), reserve ? "compact" : "non-compact",
                ctx_res ? "compact" : "non-compact");
        CHECK0(live == want_live);
        CHECK0(reserve);                                    // the init verdict, whatever the live cells hold
        // #262: a reservation context sees no cell, whatever the graph flag: its predicate is the empty cache's
        CHECK0(ctx_res);
        CHECK0(m->qsa_compact_reason(u, r) == QSA_COMPACT_OK);
        CHECK0(m->qsa_scalar_visibility(u, r));
    }
}

static void check_reserve_compact_all() {
    {   // init: the empty cache, where the reserve and live verdicts agree
        qsa_fixture fx(8192, 4, true);
        check_reserve_compact("empty", fx, QSA_COMPACT_OK);
    }
    {   // slot 0 holds an image, slot 1 text: a sequence of the stream holds 2-D cells while another has cells
        qsa_fixture fx(8192, 4, true);
        CHECK0(fx.feed(fx.text(0, 100), 512));
        CHECK0(fx.feed(fx.image(0, 32, 32), 512));
        CHECK0(fx.feed(fx.text(0, 100), 512));
        CHECK0(fx.feed(fx.text(1, 3000), 512));
        check_reserve_compact("image+text", fx, QSA_NC_2D_MULTI_SEQ);
    }
    {   // slot 1 copied slot 0 (seq_cp) when it ended inside a block (3001 tokens, ratio 4), then slot 0 went on: the block
        // 3000 .. 3003 holds sequence 0 under {0, 1} and {0} (R1-CB, test-qsa-visibility "seqcp")
        qsa_fixture fx(8192, 4, true);
        CHECK0(fx.feed(fx.text(0, 3001), 512));
        fx.mem->seq_cp(0, 1, -1, -1);
        fx.next_pos[1] = fx.next_pos[0];
        CHECK0(fx.feed(fx.text(0, 40), 512));
        check_reserve_compact("seq_cp", fx, QSA_NC_SHARED_BLOCK);
    }
}

static int run() {
    check_reserve_compact_all();
    check_views(4096, 2, false);
    check_views(4096, 4, false);
    check_views(4096, 2, true);
    check_views(4096, 1, false);
    printf("test-qsa-reserve-view: %s (%d failed checks)\n", g_fail == 0 ? "OK" : "FAILED", g_fail);
    return g_fail == 0 ? 0 : 1;
}

} // namespace reserve_view

// ---- reserve-passes --------------------------------------------------------------------------------------------------
// The QSA reserve graphs of llama_context::sched_reserve (llama_qsa_reserve_passes, PLAN-v3 §7.1) cover every graph shape a
// runtime ubatch can build. Here: the one-stream dense-shortcut graph of a split multi-sequence context. Pass B reserves the
// shortcut at n_kv_b over n_seqs streams, so each stream holds n_tok_b / n_seqs rows and the route of that graph is decided
// at that per-stream n_q (prefer_dense in qwen4exp.cpp; on HIP a non-f16 pair is dense FA at n_q <= 127 and the AV rows +
// SPARSE_ATTN route at n_q >= 128). A slot's own prompt ubatch at n_kv <= n_kv_b is a one-stream graph of up to
// min(ub, n_kv_b) rows, which only a one-stream pass at n_kv_b builds: -np 8 at ub 512 reserved 8 x 64 rows (dense) while
// a 512-token one-sequence ubatch ran the AV route. Pure function of the context shape: no model, no device.

namespace reserve_passes {

static std::string names(const std::vector<llama_qsa_reserve_pass_desc> & p) {
    std::string s;
    for (const auto & d : p) {
        s += std::string(s.empty() ? "" : " ") + d.name;
    }
    return s;
}

// the qwen4exp shape: indexer_top_k 2048, ratio 4 -> width 2051, shortcut n_kv_b 2048
static llama_qsa_reserve_shape shape(uint32_t n_tokens, uint32_t n_seqs, bool unified, uint32_t n_kv_full) {
    llama_qsa_reserve_shape s = {};
    s.n_tokens       = n_tokens;
    s.n_seqs         = n_seqs;
    s.n_outputs_pp   = n_tokens;
    s.kv_unified     = unified;
    s.mtp            = false;
    s.width          = 2048 + 4 - 1;
    s.n_kv_full      = n_kv_full;
    s.noncompact_cap = std::min<uint32_t>(n_tokens, 2 * 512);
    return s;
}

static int run() {
    for (uint32_t n_tokens : { 256u, 512u, 2048u, 4096u, 8192u }) {
        for (uint32_t n_seqs : { 1u, 2u, 4u, 8u }) {
            for (bool unified : { false, true }) {
                for (uint32_t n_kv_full : { 2048u, 4096u, 32768u }) {
                    const llama_qsa_reserve_shape s = shape(n_tokens, n_seqs, unified, n_kv_full);
                    const auto p = llama_qsa_reserve_passes(s);
                    const uint32_t n_kv_b = s.width / 256 * 256;
                    const bool shortcut = n_kv_b < n_kv_full;
                    const bool split_multi = !unified && n_seqs > 1;
                    printf("ub %u np %u %s n_kv %u: %s\n", n_tokens, n_seqs, unified ? "unified" : "split", n_kv_full, names(p).c_str());

                    // the one-stream graph of the shortcut: one sequence, n_stream 1, at n_kv_b, with the largest row count a
                    // one-sequence ubatch can have there
                    const uint32_t n_tok_1 = std::min(n_tokens, n_kv_b);
                    bool one_stream_b = false;
                    for (const auto & d : p) {
                        // the streams of the pass's graph: one in a unified cache; else the view's (every stream: n_seqs)
                        const uint32_t streams = unified ? 1 : (d.n_kv != 0 && d.n_stream != 0 ? d.n_stream : n_seqs);
                        const bool one_stream = streams == 1;
                        const uint32_t view_kv = d.n_kv == 0 ? n_kv_full : d.n_kv;
                        if (one_stream && view_kv == n_kv_b && d.n_tokens >= n_tok_1 && d.qsa == 0) {
                            one_stream_b = true;
                        }
                        CHECK(d.n_tokens >= 1 && d.n_tokens <= n_tokens, "%s: n_tokens %u", d.name, d.n_tokens);
                        CHECK(d.n_outputs <= d.n_tokens || strcmp(d.name, "noncompact") == 0, "%s: n_outputs %u > n_tokens %u", d.name, d.n_outputs, d.n_tokens);
                    }
                    if (shortcut) {
                        CHECK(one_stream_b, "ub %u np %u %s n_kv %u: no reserve graph builds the one-stream shortcut at n_kv %u with %u rows",
                              n_tokens, n_seqs, unified ? "unified" : "split", n_kv_full, n_kv_b, n_tok_1);
                    } else {
                        for (const auto & d : p) {
                            CHECK(d.n_kv == 0 || d.n_kv == n_kv_full, "%s: a reduced view at n_kv %u without a shortcut", d.name, d.n_kv);
                        }
                    }
                    // the passes before this change keep their shapes (the refactor of sched_reserve into this function)
                    bool has_b = false, has_1s = false;
                    for (const auto & d : p) {
                        if (strcmp(d.name, "passB") == 0) {
                            has_b = true;
                            CHECK(d.n_kv == n_kv_b && d.n_stream == 0 && d.n_seqs == n_seqs &&
                                  d.n_tokens == std::min(n_tokens, n_kv_b * (unified ? 1 : n_seqs)), "passB shape");
                        }
                        if (strncmp(d.name, "1s-", 3) == 0) {
                            has_1s = true;
                            CHECK(d.n_stream == 1 && d.n_seqs == 1, "%s: not one-stream", d.name);
                        }
                    }
                    CHECK(has_b == shortcut, "passB present %d, shortcut %d", (int) has_b, (int) shortcut);
                    CHECK(has_1s == split_multi, "1s passes present %d, split multi-sequence %d", (int) has_1s, (int) split_multi);
                }
            }
        }
    }
    printf("test-qsa-reserve-passes: %s (%d failed checks)\n", g_fail == 0 ? "OK" : "FAILED", g_fail);
    return g_fail == 0 ? 0 : 1;
}

} // namespace reserve_passes

// ---- noncompact-check ------------------------------------------------------------------------------------------------
// The always-on QSA graph-build check (qwen4exp_qsa_noncompact_error, src/models/qwen4exp-qsa-check.h) fails only ubatches
// that are non-compact on the complete-block selection path. Where block selection is not eligible (a non-causal context:
// LLAMA_ATTENTION_TYPE_NON_CAUSAL or llama_set_causal_attn(ctx, false); a ratio-1 layer; top_k % ratio != 0; ...) every
// ubatch takes the TE rows, which the reserve's own pp / tg graphs build, so decode must go on: the check tested `scalar`
// alone (which also requires `blocks`) and failed every ubatch past the shortcut there, in a single-sequence context with
// the self-contradictory reason 'compact'. Pure function: no model.

namespace noncompact_check {

static int run() {
    const int64_t width = 2048 + 4 - 1;   // indexer_top_k + ratio - 1
    const uint32_t U = 1024;
    for (uint32_t n_tokens : { 1u, 512u, 1024u, 4096u }) {
        for (int64_t n_kv : { (int64_t) 2048, (int64_t) 4096, (int64_t) 131072 }) {
            for (bool multi : { false, true }) {
                const bool past = n_kv > width;
                // block selection not eligible (non-causal, ratio 1, ...): scalar is false by construction, never an error
                CHECK(!qwen4exp_qsa_noncompact_error(false, false, false, false, n_kv, width, n_tokens, U, multi),
                      "blocks ineligible: a decode error");
                // SWA layer: the TE rows on every ubatch, never an error
                CHECK(!qwen4exp_qsa_noncompact_error(true, true, false, false, n_kv, width, n_tokens, U, multi),
                      "SWA layer: a decode error");
                // the reserve's non-compact variant, and compact ubatches: never an error
                CHECK(!qwen4exp_qsa_noncompact_error(true, false, true, false, n_kv, width, n_tokens, U, multi), "forced variant: error");
                CHECK(!qwen4exp_qsa_noncompact_error(true, false, false, true, n_kv, width, n_tokens, U, multi), "compact ubatch: error");
                // eligible and non-compact past the shortcut: an error in a single-sequence context, and past U tokens
                const bool want = past && (n_tokens > U || !multi);
                CHECK(qwen4exp_qsa_noncompact_error(true, false, false, false, n_kv, width, n_tokens, U, multi) == want,
                      "eligible non-compact ubatch: wrong decision");
            }
        }
    }
    // the per-context eligibility terms shared by qwen4exp_use_block_selection and the NC0 re-split
    CHECK(qwen4exp_block_selection_eligible(true, true, 4, 2048, 0.0f, false, 256, 256), "Qwen3.8 QSA layer: not eligible");
    CHECK(!qwen4exp_block_selection_eligible(true, false, 4, 2048, 0.0f, false, 256, 256), "non-causal: eligible");
    CHECK(!qwen4exp_block_selection_eligible(false, true, 4, 2048, 0.0f, false, 256, 256), "no blk_bias (alibi): eligible");
    CHECK(!qwen4exp_block_selection_eligible(true, true, 1, 2048, 0.0f, false, 256, 256), "ratio 1: eligible");
    CHECK(!qwen4exp_block_selection_eligible(true, true, 0, 2048, 0.0f, false, 256, 256), "ratio 0: eligible");
    CHECK(!qwen4exp_block_selection_eligible(true, true, 3, 2048, 0.0f, false, 256, 256), "top_k %% ratio != 0: eligible");
    CHECK(!qwen4exp_block_selection_eligible(true, true, 4, 2048, 8.0f, false, 256, 256), "alibi bias: eligible");
    CHECK(!qwen4exp_block_selection_eligible(true, true, 4, 2048, 0.0f, true, 256, 256), "soft cap: eligible");
    CHECK(!qwen4exp_block_selection_eligible(true, true, 4, 2048, 0.0f, false, 128, 256), "head 128: eligible");
    printf("test-qsa-noncompact-check: %s (%d failed checks)\n", g_fail == 0 ? "OK" : "FAILED", g_fail);
    return g_fail == 0 ? 0 : 1;
}

} // namespace noncompact_check

// ---- supported-types -------------------------------------------------------------------------------------------------
// The "supported here" list of the QSA init error (PLAN-v3 §10, llama_qsa_supported_types in src/llama-context.cpp): it
// probes the nine K / V types of the sparse op on the node's device. It ran out of its ggml context (16 tensor objects for
// 9 x 2) and aborted in ggml_new_object / GGML_ASSERT(obj_new) instead of returning the list, so the designed init error
// turned into a process abort. CPU only, no model file: the node is a no_alloc SPARSE_ATTN of the qwen4exp shape.

namespace supported_types {

// the comma-separated list as its entries: a type is in it only as a whole entry ("f16" is not in "bf16, q8_0": a
// substring search found it inside "bf16", so a list without f16 passed the f16 check)
static std::vector<std::string> split_list(const std::string & list) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (pos <= list.size() && !list.empty()) {
        const size_t end = list.find(',', pos);
        std::string item = list.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        const size_t a = item.find_first_not_of(' '), b = item.find_last_not_of(' ');
        out.push_back(a == std::string::npos ? "" : item.substr(a, b - a + 1));
        if (end == std::string::npos) {
            break;
        }
        pos = end + 1;
    }
    return out;
}

static bool has_entry(const std::vector<std::string> & items, const std::string & t) {
    for (const auto & i : items) {
        if (i == t) {
            return true;
        }
    }
    return false;
}

static int run() {
    // the checker itself: a list with bf16 but not f16 misses f16
    {
        const auto items = split_list("bf16, q8_0, q4_0, f32, q4_1, q5_0, q5_1, iq4_nl");
        CHECK(!has_entry(items, "f16") && has_entry(items, "bf16") && has_entry(items, "iq4_nl") && items.size() == 8,
              "the list checker is wrong (f16 found in a list without it)");
        if (g_fail) {
            return 1;
        }
    }
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    CHECK(cpu != nullptr, "no CPU device");
    if (cpu == nullptr) {
        return 1;
    }
    ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx { ggml_init(ip) };
    // q [256, n_q, n_head, n_stream], K / V views of 2052 cells and 2 KV heads (GQA 12), a G = 1 plan over 2051 cells
    ggml_tensor * q    = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 256, 9, 24, 1);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_Q8_0, 256, 2052, 2, 1);
    ggml_tensor * v    = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_Q8_0, 256, 2052, 2, 1);
    ggml_tensor * sel  = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_I32, 2051, 9, 1, 1);
    ggml_tensor * plan = ggml_sparse_attn_plan(ctx.get(), sel, 2051, 1);
    ggml_tensor * attn = ggml_sparse_attn(ctx.get(), q, k, v, plan, 0.0625f, 0, 0);

    const std::string list = llama_qsa_supported_types(cpu, attn);
    printf("supported here (CPU): '%s'\n", list.c_str());
    // the CPU op takes every type of S: the list names all nine
    const char * want[] = { "f16", "bf16", "q8_0", "q4_0", "f32", "q4_1", "q5_0", "q5_1", "iq4_nl" };
    const std::vector<std::string> items = split_list(list);
    for (const char * t : want) {
        CHECK(has_entry(items, t), "missing %s", t);
    }
    CHECK(items.size() == std::size(want), "%zu entries, want %zu", items.size(), std::size(want));
    printf("test-qsa-supported-types: %s (%d failed checks)\n", g_fail == 0 ? "OK" : "FAILED", g_fail);
    return g_fail == 0 ? 0 : 1;
}

} // namespace supported_types

struct section {
    const char * name;
    int (*run)();
};

static const section sections[] = {
    { "dup-key",          dup_key::run },
    { "reserve-view",     reserve_view::run },
    { "reserve-passes",   reserve_passes::run },
    { "noncompact-check", noncompact_check::run },
    { "supported-types",  supported_types::run },
};

int main(int argc, char ** argv) {
    int status = 0;
    bool found = false;
    for (const section & s : sections) {
        if (argc > 1 && strcmp(argv[1], s.name) != 0) {
            continue;
        }
        found = true;
        printf("== %s\n", s.name);
        g_fail = 0;
        status |= s.run();
    }
    if (!found) {
        printf("unknown section '%s'\n", argv[1]);
        return 2;
    }
    return status;
}
