#pragma once
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <vector>

struct qsa_prefix_state {
    bool valid = true;
    int32_t sequence = -1;
    int32_t begin = 0, end = 0;
    size_t previous_size = 0;
    std::vector<int32_t> cells;
    std::vector<int32_t> positions;
    std::vector<int32_t> block_positions;

    explicit qsa_prefix_state(size_t capacity = 0) : positions(capacity, -1) {}
    void reset() {
        valid = true; sequence = -1; begin = end = 0; previous_size = 0;
        cells.clear(); block_positions.clear(); std::fill(positions.begin(), positions.end(), -1);
    }
    void invalidate() { valid = false; }
    void truncate(size_t size) {
        size = std::min(size, cells.size());
        for (size_t i = size; i < cells.size(); ++i) { positions[cells[i]] = -1; }
        cells.resize(size); block_positions.resize(size/4);
        if (cells.empty()) { sequence = -1; }
    }
    bool apply(int32_t seq, int32_t start, const std::vector<uint32_t> & slots) {
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
            if (slots[i-start] != uint32_t(cells[i])) { invalidate(); return false; }
        }
        previous_size = cells.size(); begin = start; end = finish; sequence = seq;
        for (size_t i = start; i < std::min(size_t(finish), cells.size()); ++i) { positions[cells[i]] = -1; }
        cells.resize(std::max(cells.size(), size_t(finish)), -1);
        for (size_t i = 0; i < slots.size(); ++i) {
            if (positions[slots[i]] >= 0) { invalidate(); return false; }
            cells[start+i] = slots[i]; positions[slots[i]] = start+i;
        }
        while (block_positions.size() < cells.size()/4) { block_positions.push_back(block_positions.size()*4); }
        return true;
    }
};
