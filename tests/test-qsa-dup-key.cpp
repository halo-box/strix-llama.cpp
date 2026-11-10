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
#include "../src/prefix.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <random>

static int g_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { if (g_fail++ < 20) { printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } } while (0)

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

int main() {
    test_exact();
    test_work();
    printf("test-qsa-dup-key: %s (%d failed checks)\n", g_fail == 0 ? "OK" : "FAILED", g_fail);
    return g_fail == 0 ? 0 : 1;
}
