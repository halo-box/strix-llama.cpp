#include "../src/qsa-prefix-state.h"
#include "ggml.h"
#include <random>
#include <iostream>

int main() {
    qsa_prefix_state s(4096);
    std::mt19937 rng(414);
    for (int round=0; round<200; ++round) {
        s.reset();
        for (int step=0; step<100; ++step) {
            if (!s.cells.empty() && rng()%3==0) { s.truncate(rng()%(s.cells.size()+1)); }
            size_t start=s.cells.size();
            int n=1+rng()%8;
            std::vector<uint32_t> slots;
            while (int(slots.size())<n) {
                uint32_t c=rng()%4096;
                if (s.positions[c]<0 && std::find(slots.begin(),slots.end(),c)==slots.end()) slots.push_back(c);
            }
            auto before=s.cells;
            GGML_ASSERT(s.apply(0,start,slots));
            GGML_ASSERT(s.previous_size==before.size());
            GGML_ASSERT(std::equal(before.begin(),before.end(),s.cells.begin()));
            for (size_t i=0;i<s.cells.size();++i) GGML_ASSERT(s.positions[s.cells[i]]==int(i));
            for (size_t b=0;b<s.block_positions.size();++b) GGML_ASSERT(s.block_positions[b]==int(4*b));
            if (s.cells.size()>8) {
                int at=rng()%(s.cells.size()-4);
                std::vector<uint32_t> same(s.cells.begin()+at,s.cells.begin()+at+4);
                auto old=s.cells;GGML_ASSERT(s.apply(0,at,same));GGML_ASSERT(s.cells==old);
            }
        }
    }
    s.reset(); GGML_ASSERT(!s.apply(0,2,{3}));
    s.reset(); GGML_ASSERT(!s.apply(0,0,{3,3}));
    s.reset(); GGML_ASSERT(s.apply(0,0,{3,7})); GGML_ASSERT(!s.apply(1,2,{4}));
    s.reset(); GGML_ASSERT(s.apply(0,0,{3,7})); GGML_ASSERT(!s.apply(0,2,{3}));
    s.reset(); GGML_ASSERT(s.apply(0,0,{3,7})); GGML_ASSERT(!s.apply(0,0,{9}));
    s.reset(); GGML_ASSERT(s.apply(0,0,{3,7})); s.truncate(0); GGML_ASSERT(s.apply(1,0,{3}));
    s.invalidate(); GGML_ASSERT(!s.apply(1,1,{4}));
    std::cout << "PASS: 20000 randomized append/rollback steps, in-place rewrites, holes, duplicates, collisions, sequence changes\n";
}
