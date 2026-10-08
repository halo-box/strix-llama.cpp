// qsa_rebuild_prefix: rebuild the QSA prefix of one sequence from a unified KV cache that also holds other sequences
#include "../src/prefix.h"
#include "ggml.h"
#include <iostream>

static llama_kv_cells make_cells(uint32_t n) { llama_kv_cells c; c.resize(n); return c; }
static void put(llama_kv_cells & c, uint32_t cell, llama_seq_id seq, llama_pos pos) { c.pos_set(cell, pos); c.seq_add(cell, seq); }
// an mrope cell as llama_kv_cache::apply_ubatch stores it: text at (pos, pos, pos), an image's cells at one position over (y, x)
static void put2(llama_kv_cells & c, uint32_t cell, llama_seq_id seq, llama_pos pos, llama_pos y, llama_pos x) {
    put(c, cell, seq, pos); c.ext_set(cell, { x, y });
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
    // still refused: a hole in the positions, a cell shared with another sequence, a duplicated position
    {
        auto c = make_cells(64);
        for (int p = 0; p < 10; ++p) { if (p != 5) { put(c, p, 0, p); } }
        qsa_prefix_state s(64);
        GGML_ASSERT(!qsa_rebuild_prefix(c, 0, s));
    }
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
    std::cout << "PASS: prefixes rebuilt next to other sequences; holes, shared cells and duplicates refused\n";
}
