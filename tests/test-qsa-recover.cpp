// qsa_rebuild_prefix: rebuild the QSA prefix of one sequence from a unified KV cache that also holds other sequences
#include "../src/prefix.h"
#include "ggml.h"
#include <algorithm>
#include <iostream>
#include <random>

static llama_kv_cells make_cells(uint32_t n) { llama_kv_cells c; c.resize(n); return c; }
static void put(llama_kv_cells & c, uint32_t cell, llama_seq_id seq, llama_pos pos) { c.pos_set(cell, pos); c.seq_add(cell, seq); }
// an mrope cell as llama_kv_cache::apply_ubatch stores it: text at (pos, pos, pos), an image's cells at one position over (y, x)
static void put2(llama_kv_cells & c, uint32_t cell, llama_seq_id seq, llama_pos pos, llama_pos y, llama_pos x) {
    put(c, cell, seq, pos); c.ext_set(cell, { x, y });
}

// every field the selection metadata reads; begin/end/previous_* describe the last ubatch, which qsa_apply applies after a rebuild
static void same_tracker(const qsa_prefix_state & a, const qsa_prefix_state & b) {
    GGML_ASSERT(a.valid && b.valid && (a.cells.empty() || a.sequence == b.sequence));
    GGML_ASSERT(a.cells == b.cells && a.positions == b.positions && a.block_positions == b.block_positions);
    GGML_ASSERT(a.pos_of == b.pos_of && a.y_of == b.y_of && a.x_of == b.x_of);
    GGML_ASSERT(a.first_dup == b.first_dup && a.first_gap == b.first_gap && a.blk_rank == b.blk_rank && a.blk_start == b.blk_start);
}

// randomized: seq 0 next to two other conversations in one cache, cells in any order; text, mrope images, position holes
// (an MTP draft), rollbacks and a cell shared for a while. The rebuilt tracker must equal the one kept ubatch by ubatch.
static void test_rebuild_matches_incremental() {
    std::mt19937 rng(1163);
    const uint32_t n_cells = 2048;
    for (int round = 0; round < 200; ++round) {
        auto c = make_cells(n_cells);
        qsa_prefix_state inc(n_cells);
        std::vector<uint32_t> free_cells(n_cells);
        for (uint32_t i = 0; i < n_cells; ++i) { free_cells[i] = i; }
        std::shuffle(free_cells.begin(), free_cells.end(), rng);
        const int mode = round % 3; // 0: target with images, 1: draft with holes, 2: both
        int32_t next[3] = { 0, 0, 0 };
        int32_t shared = -1;
        for (int step = 0; step < 80 && free_cells.size() > 64; ++step) {
            const int what = rng() % 8;
            if (what == 0) {        // rollback of seq 0 from a position it holds (seq_rm)
                if (inc.cells.empty()) { continue; }
                const int32_t p = inc.pos_of[rng() % inc.cells.size()];
                if (shared >= 0 && c.pos_get(shared) >= p) { c.seq_rm(shared, 1); shared = -1; }
                inc.truncate(inc.rank_from(p)); inc.relayout = false;
                next[0] = 0;
                for (uint32_t i = 0; i < n_cells; ++i) {
                    if (c.is_empty(i) || !c.seq_has(i, 0)) { continue; }
                    if (c.pos_get(i) >= p) { c.rm(i); free_cells.push_back(i); continue; }
                    next[0] = std::max({ next[0], c.pos_get(i) + 1, c.ext_get(i).y + 1, c.ext_get(i).x + 1 });
                }
                if (mode != 0) { next[0] = std::max(next[0], p); }
            } else if (what <= 2) { // another conversation appends or rolls back
                const llama_seq_id o = 1 + rng() % 2;
                if (rng() % 3 == 0 && next[o] > 0) {
                    const int32_t p = rng() % next[o];
                    for (uint32_t i = 0; i < n_cells; ++i) {
                        if (int32_t(i) == shared || c.is_empty(i) || !c.seq_has(i, o) || c.pos_get(i) < p) { continue; }
                        c.rm(i); free_cells.push_back(i);
                    }
                    next[o] = p;
                } else {
                    for (int k = 1 + rng() % 9; k > 0; --k) { put2(c, free_cells.back(), o, next[o], next[o], next[o]); free_cells.pop_back(); ++next[o]; }
                }
            } else if (what == 3) { // share a cell of seq 0 with seq 1 for a while (seq_cp of one cell), or stop sharing
                if (shared >= 0) { c.seq_rm(shared, 1); shared = -1; }
                else if (!inc.cells.empty()) { shared = inc.cells[rng() % inc.cells.size()]; c.seq_add(shared, 1); }
            } else {                // a ubatch of seq 0
                std::vector<uint32_t> slots; std::vector<int32_t> p, y, x;
                if (mode != 1 && what == 4) {          // an image of h x w cells at one position
                    const int h = 1 + rng() % 4, w = 2 + rng() % 4;
                    for (int r = 0; r < h; ++r) { for (int k = 0; k < w; ++k) { p.push_back(next[0]); y.push_back(next[0] + r); x.push_back(next[0] + k); } }
                    next[0] += std::max(h, w);
                } else {
                    if (mode != 0 && what == 5) { next[0] += 1 + rng() % 9; } // the draft skips an image's positions
                    for (int k = 1 + rng() % 9; k > 0; --k) { p.push_back(next[0]); y.push_back(next[0]); x.push_back(next[0]); ++next[0]; }
                }
                for (size_t i = 0; i < p.size(); ++i) { slots.push_back(free_cells.back()); free_cells.pop_back(); put2(c, slots.back(), 0, p[i], y[i], x[i]); }
                GGML_ASSERT(inc.apply(0, (int32_t) inc.cells.size(), slots, p, y, x));
                inc.relayout = false;
            }
            qsa_prefix_state rebuilt(n_cells);
            const bool ok = qsa_rebuild_prefix(c, 0, rebuilt);
            GGML_ASSERT(ok == (shared < 0));
            if (ok) { same_tracker(inc, rebuilt); }
        }
    }
}

int main() {
    // one sequence alone, in position order: what already worked
    {
        auto c = make_cells(64);
        for (int p = 0; p < 10; ++p) { put(c, p, 0, p); }
        qsa_prefix_state s(64);
        GGML_ASSERT(qsa_rebuild_prefix(c, 0, s));
        GGML_ASSERT(s.sequence == 0 && s.cells.size() == 10 && s.block_positions.size() == 2);
        for (int p = 0; p < 10; ++p) { GGML_ASSERT(s.cells[p] == p && s.positions[p] == p); }
    }
    // another conversation resident first (cells 0..14), ours after it, stored in reverse cell order
    {
        auto c = make_cells(64);
        for (int p = 0; p < 15; ++p) { put(c, p, 1, p); }
        for (int p = 0; p < 10; ++p) { put(c, 29 - p, 0, p); }
        qsa_prefix_state s(64);
        GGML_ASSERT(qsa_rebuild_prefix(c, 0, s));
        GGML_ASSERT(s.sequence == 0 && s.cells.size() == 10 && s.block_positions.size() == 2);
        for (int p = 0; p < 10; ++p) { GGML_ASSERT(s.cells[p] == 29 - p && s.positions[29 - p] == p); }
        for (int i = 0; i < 15; ++i) { GGML_ASSERT(s.positions[i] == -1); }
        qsa_prefix_state t(64);
        GGML_ASSERT(qsa_rebuild_prefix(c, 1, t) && t.cells.size() == 15 && t.positions[29] == -1);
    }
    // two conversations decoded together: their cells alternate
    {
        auto c = make_cells(64);
        for (int p = 0; p < 20; ++p) { put(c, 2*p, 0, p); put(c, 2*p + 1, 1, p); }
        qsa_prefix_state s(64);
        GGML_ASSERT(qsa_rebuild_prefix(c, 1, s) && s.cells.size() == 20);
        for (int p = 0; p < 20; ++p) { GGML_ASSERT(s.cells[p] == 2*p + 1); }
    }
    // a new conversation, nothing cached yet, others resident: an empty prefix its first ubatch extends
    {
        auto c = make_cells(64);
        for (int p = 0; p < 15; ++p) { put(c, p, 1, p); }
        qsa_prefix_state s(64);
        GGML_ASSERT(qsa_rebuild_prefix(c, 0, s) && s.cells.empty() && s.sequence == 0);
        GGML_ASSERT(s.apply(0, 0, {20, 21, 22}));
    }
    // an image next to another conversation: the cells of seq in the mask's order (position, then y, then x), ranked,
    // the same tracker as applying them in that order; the other sequence's text is a plain prefix
    {
        auto c = make_cells(64);
        std::vector<uint32_t> slots; std::vector<int32_t> p, y, x;
        auto add = [&](uint32_t cell, llama_pos pos, llama_pos yy, llama_pos xx) {
            put2(c, cell, 0, pos, yy, xx); slots.push_back(cell); p.push_back(pos); y.push_back(yy); x.push_back(xx);
        };
        for (int q = 0; q < 12; ++q) { put2(c, 2*q + 1, 1, q, q, q); }      // the other conversation, interleaved
        for (int q = 0; q < 3; ++q) { add(40 + q, q, q, q); }
        for (int r = 0; r < 2; ++r) { for (int k = 0; k < 3; ++k) { add(60 - 3*r - k, 3, 3 + r, 3 + k); } } // 2 x 3 image, cells reversed
        for (int q = 6; q < 11; ++q) { add(20 + q, q, q, q); }
        qsa_prefix_state s(64), ref(64);
        GGML_ASSERT(qsa_rebuild_prefix(c, 0, s));
        GGML_ASSERT(ref.apply(0, 0, {40, 41, 42}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}));
        GGML_ASSERT(ref.apply(0, 3, {60, 59, 58, 57, 56, 55}, {3, 3, 3, 3, 3, 3}, {3, 3, 3, 4, 4, 4}, {3, 4, 5, 3, 4, 5}));
        GGML_ASSERT(ref.apply(0, 9, {26, 27, 28, 29, 30}, {6, 7, 8, 9, 10}, {6, 7, 8, 9, 10}, {6, 7, 8, 9, 10}));
        GGML_ASSERT(s.valid && s.sequence == 0 && s.ranked() && !s.identity());
        GGML_ASSERT(s.cells == ref.cells && s.positions == ref.positions && s.block_positions == ref.block_positions);
        GGML_ASSERT(s.pos_of == ref.pos_of && s.y_of == ref.y_of && s.x_of == ref.x_of);
        GGML_ASSERT(s.first_dup == ref.first_dup && s.first_gap == ref.first_gap && s.first_dup == 4);
        for (int i = 0; i < 64; i += 2) { GGML_ASSERT(i >= 20 || s.positions[i + 1] == -1); }
        qsa_prefix_state t(64);
        GGML_ASSERT(qsa_rebuild_prefix(c, 1, t) && t.cells.size() == 12 && t.identity() && !t.ranked());
        // two image cells with one key are refused
        put2(c, 61, 0, 3, 4, 5);
        qsa_prefix_state u(64);
        GGML_ASSERT(!qsa_rebuild_prefix(c, 0, u));
    }
    // a hole in the positions (an MTP draft skips an image's positions): position blocks, only complete ones count
    {
        auto c = make_cells(64);
        for (int p = 0; p < 10; ++p) { if (p != 5) { put(c, p, 0, p); } }
        qsa_prefix_state s(64);
        GGML_ASSERT(qsa_rebuild_prefix(c, 0, s) && s.cells.size() == 9 && !s.ranked() && !s.identity());
        GGML_ASSERT(s.complete() == 1 && s.blk_rank[0] == 0 && s.blk_start[0] == 0 && s.tail_start(7) == 8);
    }
    // still refused: a cell shared with another sequence, a duplicated position
    {
        auto c = make_cells(64);
        for (int p = 0; p < 10; ++p) { put(c, p, 0, p); }
        c.seq_add(3, 1);
        qsa_prefix_state s(64);
        GGML_ASSERT(!qsa_rebuild_prefix(c, 0, s));
    }
    {
        auto c = make_cells(64);
        for (int p = 0; p < 10; ++p) { put(c, p, 0, p); }
        put(c, 40, 0, 3);
        qsa_prefix_state s(64);
        GGML_ASSERT(!qsa_rebuild_prefix(c, 0, s));
    }
    test_rebuild_matches_incremental();
    std::cout << "PASS: prefixes rebuilt next to other sequences, with images and holes; shared cells and duplicates refused\n";
}
