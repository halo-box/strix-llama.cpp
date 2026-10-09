// qsa_rebuild_prefix: rebuild the QSA prefix of one sequence from a unified KV cache that also holds other sequences
#include "../src/prefix.h"
#include "ggml.h"
#include <iostream>

static llama_kv_cells make_cells(uint32_t n) { llama_kv_cells c; c.resize(n); return c; }
static void put(llama_kv_cells & c, uint32_t cell, llama_seq_id seq, llama_pos pos) { c.pos_set(cell, pos); c.seq_add(cell, seq); }

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
