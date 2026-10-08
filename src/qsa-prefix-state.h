#pragma once
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <tuple>
#include <vector>

// The tracked prefix holds one sequence's cells in the attention mask's order: rank k is the k-th cell by
// (position, then 2-D y, then x), as the QSA selection ranks them. Without an image that order is the position
// (rank == position). An mrope image puts all its cells at one position, so ranks run ahead of positions after
// it; the selection then ranks too, and the blocks are rank blocks. A prefix that is neither (a position hole
// without an image) does not match the selection's blocks and is invalid.
struct qsa_prefix_state {
    bool valid = true;
    int32_t sequence = -1;
    int32_t begin = 0, end = 0;     // ranks of the last applied ubatch
    size_t previous_size = 0;
    std::vector<int32_t> cells;     // rank -> cell
    std::vector<int32_t> positions; // cell -> rank
    std::vector<int32_t> block_positions;
    std::vector<int32_t> pos_of, y_of, x_of; // rank -> mrope key (position, y, x)
    size_t first_dup = SIZE_MAX;    // first rank at the same position as the rank before it
    size_t first_gap = SIZE_MAX;    // first rank whose position is not its rank

    explicit qsa_prefix_state(size_t capacity = 0) : positions(capacity, -1) {}
    bool ranked()   const { return first_dup < cells.size(); }
    bool identity() const { return first_gap >= cells.size(); }
    std::tuple<int32_t, int32_t, int32_t> key(size_t k) const { return { pos_of[k], y_of[k], x_of[k] }; }
    void reset() {
        valid = true; sequence = -1; begin = end = 0; previous_size = 0;
        cells.clear(); block_positions.clear(); std::fill(positions.begin(), positions.end(), -1);
        pos_of.clear(); y_of.clear(); x_of.clear(); first_dup = first_gap = SIZE_MAX;
    }
    void invalidate() { valid = false; }
    void truncate(size_t size) {
        size = std::min(size, cells.size());
        for (size_t i = size; i < cells.size(); ++i) { positions[cells[i]] = -1; }
        cells.resize(size); block_positions.resize(size/4);
        pos_of.resize(size); y_of.resize(size); x_of.resize(size);
        if (first_dup >= size) { first_dup = SIZE_MAX; }
        if (first_gap >= size) { first_gap = SIZE_MAX; }
        if (cells.empty()) { sequence = -1; }
    }
    // the first rank whose position is >= p (the ranks a seq_rm from position p removes)
    size_t rank_from(int32_t p) const { return std::lower_bound(pos_of.begin(), pos_of.end(), p) - pos_of.begin(); }
    // a 1-D ubatch: every axis is the position
    bool apply(int32_t seq, int32_t start, const std::vector<uint32_t> & slots) {
        std::vector<int32_t> p(slots.size());
        for (size_t i = 0; i < slots.size(); ++i) { p[i] = start + int32_t(i); }
        return apply(seq, start, slots, p, p, p);
    }
    // place slots at ranks [start, start + n) with keys (pos, y, x); an overlap with the prefix must be the same cells and keys
    bool apply(int32_t seq, int32_t start, const std::vector<uint32_t> & slots,
               const std::vector<int32_t> & pos, const std::vector<int32_t> & y, const std::vector<int32_t> & x) {
        if (!valid || slots.empty() || start < 0 || size_t(start) > cells.size() ||
            (sequence >= 0 && sequence != seq) || size_t(start)+slots.size() > positions.size()) {
            invalidate(); return false;
        }
        const int32_t finish = start + slots.size();
        for (size_t i = 0; i < slots.size(); ++i) {
            const uint32_t cell = slots[i];
            if (cell >= positions.size()) { invalidate(); return false; }
            const int32_t old = positions[cell];
            if (old >= 0 && (old < start || old >= finish)) { invalidate(); return false; }
        }
        for (size_t i = start; i < std::min(size_t(finish), cells.size()); ++i) {
            if (slots[i-start] != uint32_t(cells[i]) || pos_of[i] != pos[i-start] || y_of[i] != y[i-start] || x_of[i] != x[i-start]) {
                invalidate(); return false;
            }
        }
        // the new ranks continue the mask order
        for (size_t i = std::max(size_t(start), cells.size()); i < size_t(finish); ++i) {
            const auto k = std::make_tuple(pos[i-start], y[i-start], x[i-start]);
            const bool after = i == 0 || (i == size_t(start) ? key(i-1) < k :
                std::make_tuple(pos[i-start-1], y[i-start-1], x[i-start-1]) < k);
            if (!after) { invalidate(); return false; }
        }
        previous_size = cells.size(); begin = start; end = finish; sequence = seq;
        for (size_t i = start; i < std::min(size_t(finish), cells.size()); ++i) { positions[cells[i]] = -1; }
        const size_t old_size = cells.size();
        cells.resize(std::max(cells.size(), size_t(finish)), -1);
        pos_of.resize(cells.size()); y_of.resize(cells.size()); x_of.resize(cells.size());
        for (size_t i = 0; i < slots.size(); ++i) {
            if (positions[slots[i]] >= 0) { invalidate(); return false; }
            cells[start+i] = slots[i]; positions[slots[i]] = start+i;
            pos_of[start+i] = pos[i]; y_of[start+i] = y[i]; x_of[start+i] = x[i];
        }
        for (size_t i = old_size; i < cells.size(); ++i) {
            if (first_dup == SIZE_MAX && i > 0 && pos_of[i] == pos_of[i-1]) { first_dup = i; }
            if (first_gap == SIZE_MAX && pos_of[i] != int32_t(i)) { first_gap = i; }
        }
        if (!ranked() && !identity()) { invalidate(); return false; }
        while (block_positions.size() < cells.size()/4) { block_positions.push_back(block_positions.size()*4); }
        return true;
    }
    // the rope position of block b's pooled key on axis a, as the selection gives it: the block start on every axis
    // for position blocks; the key of the block's first cell (position, y, x, position) for rank blocks
    int32_t block_axis(size_t b, int a) const {
        if (!ranked()) { return int32_t(b*4); }
        const size_t k = b*4;
        return a == 1 ? y_of[k] : a == 2 ? x_of[k] : pos_of[k];
    }
};
