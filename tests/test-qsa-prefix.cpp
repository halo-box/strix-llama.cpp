#include "../src/qsa-prefix-state.h"
#include "../src/prefix.h"
#include "ggml.h"
#include <random>
#include <iostream>

// cells filled in order, one per position
struct qsa_cells {
    llama_kv_cells cells;
    uint32_t next = 0;
    explicit qsa_cells(uint32_t n) { cells.resize(n); }
    void add(llama_pos p0, llama_pos p1, std::vector<llama_seq_id> seqs, llama_pos ext_dy = 0) {
        for (llama_pos p=p0; p<p1; ++p, ++next) {
            cells.pos_set(next, p);
            for (auto s : seqs) { cells.seq_add(next, s); }
            if (ext_dy) { cells.ext_set(next, { p, p+ext_dy }); }
        }
    }
};

// one token per entry, n_pos axes all equal to the linear position
struct qsa_ubatch {
    std::vector<llama_token> tok; std::vector<llama_pos> pos; std::vector<int32_t> n_seq_id;
    std::vector<std::vector<llama_seq_id>> ids; std::vector<llama_seq_id *> seq_id; llama_ubatch u{};
    qsa_ubatch(std::vector<std::pair<llama_pos, std::vector<llama_seq_id>>> tokens, uint32_t n_pos = 1) {
        const uint32_t n = tokens.size();
        tok.assign(n, 0); pos.resize(n*n_pos); ids.resize(n);
        for (uint32_t i=0; i<n; ++i) {
            for (uint32_t a=0; a<n_pos; ++a) { pos[i+a*n] = tokens[i].first; }
            ids[i] = tokens[i].second; n_seq_id.push_back(ids[i].size()); seq_id.push_back(ids[i].data());
        }
        u.n_tokens = n; u.n_pos = n_pos; u.token = tok.data(); u.pos = pos.data(); u.n_seq_id = n_seq_id.data(); u.seq_id = seq_id.data();
    }
};

static bool scalar(const qsa_cells & c, const qsa_ubatch & ub) {
    return qsa_scalar_visibility_cells(c.cells, c.cells.size(), 4, ub.u);
}

// the gate that picks the compact (maskless) QSA visibility over the masked one
static void test_scalar_visibility() {
    {   // independent sequences decode together
        qsa_cells c(64); c.add(0, 13, {0}); c.add(0, 13, {1});
        GGML_ASSERT(scalar(c, qsa_ubatch({{13, {0}}, {13, {1}}})));
    }
    {   // seq_cp inside a block: position 12 is shared, 13 and 14 are not, so block [12, 16) has no complete group
        qsa_cells c(64); c.add(0, 13, {0, 1}); c.add(13, 15, {0}); c.add(13, 15, {1}); c.add(0, 6, {2});
        GGML_ASSERT(!scalar(c, qsa_ubatch({{15, {0}}, {15, {1}}})));
        GGML_ASSERT(!scalar(c, qsa_ubatch({{15, {1}}})));
        GGML_ASSERT(scalar(c, qsa_ubatch({{6, {2}}})));
    }
    {   // seq_cp on a block boundary splits nothing
        qsa_cells c(64); c.add(0, 12, {0, 1}); c.add(12, 14, {0}); c.add(12, 14, {1});
        GGML_ASSERT(scalar(c, qsa_ubatch({{14, {0}}, {14, {1}}})));
    }
    {   // a prompt shared at token level reads the row of its first seq id, as the attention mask does
        qsa_cells c(64); c.add(0, 8, {0, 1, 2, 3});
        GGML_ASSERT(scalar(c, qsa_ubatch({{8, {0, 1, 2, 3}}, {9, {0, 1, 2, 3}}})));
    }
    {   // a 2-D image cell only matters to the sequences that hold it
        qsa_cells c(64); c.add(0, 8, {0}); c.add(0, 4, {1}); c.add(4, 5, {1}, 1);
        GGML_ASSERT(scalar(c, qsa_ubatch({{8, {0}}}, 3)));
        GGML_ASSERT(!scalar(c, qsa_ubatch({{5, {1}}}, 3)));
        GGML_ASSERT(!scalar(c, qsa_ubatch({{8, {0}}, {5, {1}}}, 3)));
    }
    {   // one sequence with an image (mrope: the image's cells share one position, extents past it): text after the
        // image is ranked in the mask's order, so the compact rule holds; a query at the image's own position does not
        qsa_cells c(64); c.add(0, 6, {0});
        for (llama_pos k = 0; k < 6; ++k) { c.cells.pos_set(c.next, 6); c.cells.seq_add(c.next, 0); c.cells.ext_set(c.next, { 6 + k%3, 6 + k/3 }); ++c.next; }
        c.add(9, 12, {0});
        GGML_ASSERT(scalar(c, qsa_ubatch({{12, {0}}}, 4)));
        GGML_ASSERT(scalar(c, qsa_ubatch({{12, {0}}, {13, {0}}, {14, {0}}, {15, {0}}}, 4)));
        GGML_ASSERT(!scalar(c, qsa_ubatch({{6, {0}}}, 4)));
        GGML_ASSERT(!scalar(c, qsa_ubatch({{12, {0}}, {6, {0}}}, 4)));
        // another sequence anywhere in the cache: the selection does not rank, keep the masked path
        qsa_cells d = c; d.add(0, 3, {1});
        GGML_ASSERT(!scalar(d, qsa_ubatch({{12, {0}}}, 4)));
        // a cell shared by two sequences
        qsa_cells e = c; e.cells.seq_add(0, 1);
        GGML_ASSERT(!scalar(e, qsa_ubatch({{12, {0}}}, 4)));
        // without 2-D cells nothing changes: a second sequence is fine
        qsa_cells f(64); f.add(0, 12, {0}); f.add(0, 5, {1});
        GGML_ASSERT(scalar(f, qsa_ubatch({{12, {0}}}, 4)));
    }
    {   // no seq id, or a position with 2-D extents in the ubatch
        qsa_cells c(64); c.add(0, 8, {0});
        qsa_ubatch none({{8, {0}}}); none.n_seq_id[0] = 0;
        GGML_ASSERT(!scalar(c, none));
        qsa_ubatch ub({{8, {0}}}, 3); ub.pos[1] = 9;
        GGML_ASSERT(!scalar(c, ub));
    }
}

// mrope: an image's cells share one position and run over (y, x); text resumes at the image position + max(h, w).
// The tracked prefix must hold exactly the live cells in the mask's order (the order the selection ranks them in),
// be ranked while an image is in it and position-ordered otherwise, and give each rank block its first cell's key.
static void test_mrope_ranks() {
    std::mt19937 rng(4711);
    for (int round=0; round<300; ++round) {
        qsa_prefix_state s(8192);
        struct cell_key { int32_t p, y, x; uint32_t cell; };
        std::vector<cell_key> live;   // reference: every cell the cache holds
        std::vector<uint32_t> free_cells(8192); for (uint32_t i=0;i<8192;++i) free_cells[i]=8191-i;
        int32_t next_pos = 0;
        for (int step=0; step<60; ++step) {
            const int what = rng()%6;
            if (what == 0 && !live.empty()) {   // rollback: remove every cell at position >= p (seq_rm)
                const int32_t p = live[rng()%live.size()].p;
                s.truncate(s.rank_from(p));
                for (auto it=live.begin(); it!=live.end();) { if (it->p >= p) { free_cells.push_back(it->cell); it=live.erase(it); } else { ++it; } }
                next_pos = live.empty() ? 0 : std::max_element(live.begin(), live.end(), [](auto & a, auto & b){ return std::tie(a.p,a.y,a.x) < std::tie(b.p,b.y,b.x); })->p + 1;
                // after an image the next text position is past the image's extent
                for (auto & c : live) { next_pos = std::max(next_pos, std::max(c.y, c.x) + 1); }
                continue;
            }
            std::vector<uint32_t> slots; std::vector<int32_t> p, y, x;
            if (what == 1) {                  // an image of h x w cells
                const int h = 1+rng()%5, w = 2+rng()%5;
                for (int r=0;r<h;++r) for (int c=0;c<w;++c) { p.push_back(next_pos); y.push_back(next_pos+r); x.push_back(next_pos+c); }
                next_pos += std::max(h, w);
            } else {                          // text
                const int n = 1+rng()%9;
                for (int i=0;i<n;++i) { p.push_back(next_pos); y.push_back(next_pos); x.push_back(next_pos); ++next_pos; }
            }
            for (size_t i=0;i<p.size();++i) { slots.push_back(free_cells.back()); free_cells.pop_back(); live.push_back({p[i], y[i], x[i], slots[i]}); }
            GGML_ASSERT(s.apply(0, (int32_t) s.cells.size(), slots, p, y, x));
        }
        std::sort(live.begin(), live.end(), [](auto & a, auto & b){ return std::tie(a.p,a.y,a.x) < std::tie(b.p,b.y,b.x); });
        GGML_ASSERT(s.valid && s.cells.size() == live.size());
        bool dup = false, ident = true;
        for (size_t k=0;k<live.size();++k) {
            GGML_ASSERT(s.cells[k] == int32_t(live[k].cell) && s.positions[live[k].cell] == int32_t(k));
            GGML_ASSERT(s.pos_of[k] == live[k].p && s.y_of[k] == live[k].y && s.x_of[k] == live[k].x);
            dup |= k > 0 && live[k].p == live[k-1].p; ident &= live[k].p == int32_t(k);
        }
        GGML_ASSERT(s.ranked() == dup && (dup || s.identity()) && (s.identity() == ident || dup));
        GGML_ASSERT(s.complete() == live.size()/4);
        for (size_t b=0;b<live.size()/4;++b) for (int a=0;a<4;++a) {
            const auto & c = live[4*b];
            GGML_ASSERT(s.blk_rank[b] == int32_t(4*b) && s.blk_start[b] == int32_t(4*b));
            GGML_ASSERT(s.block_axis(b, a) == (dup ? (a == 1 ? c.y : a == 2 ? c.x : c.p) : int32_t(4*b)));
        }
    }
    {   // a position hole without an image (an MTP draft skips the image's positions): position blocks, only complete ones count
        qsa_prefix_state s(64);
        GGML_ASSERT(s.apply(0, 0, {1, 2, 3}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}));
        GGML_ASSERT(s.apply(0, 3, {4, 5, 6, 7, 8}, {5, 8, 9, 10, 11}, {5, 8, 9, 10, 11}, {5, 8, 9, 10, 11}));
        GGML_ASSERT(s.valid && !s.ranked() && !s.identity() && s.complete() == 1 && s.blk_rank[0] == 4 && s.blk_start[0] == 8);
        GGML_ASSERT(s.tail_start(7) == 12 && s.tail_start(3) == 4 && s.tail_cell(4, 0) == -1 && s.tail_cell(4, 1) == 4);
        s.truncate(s.rank_from(9)); GGML_ASSERT(s.complete() == 0 && s.cells.size() == 5);
    }
    {   // keys must keep the mask's order; a rewrite must repeat the cells and keys it overwrites
        qsa_prefix_state s(64);
        GGML_ASSERT(s.apply(0, 0, {1, 2, 3, 4}, {0, 1, 1, 1}, {0, 1, 1, 2}, {0, 1, 2, 1}));
        GGML_ASSERT(s.ranked() && !s.identity());
        GGML_ASSERT(s.apply(0, 2, {3, 4}, {1, 1}, {1, 2}, {2, 1}));
        GGML_ASSERT(!s.apply(0, 2, {3, 4}, {1, 1}, {1, 2}, {2, 2}));
        qsa_prefix_state t(64);
        GGML_ASSERT(!t.apply(0, 0, {1, 2}, {0, 0}, {0, 0}, {1, 0}));
    }
}

// the draft side of an mrope conversation: text only, but the positions skip each image's extent. The selection
// pools position blocks that have all four positions; a query's tail is its own position block, holes as -1.
static void test_position_holes() {
    std::mt19937 rng(9001);
    for (int round=0; round<300; ++round) {
        qsa_prefix_state s(8192);
        std::vector<std::pair<int32_t, uint32_t>> live;   // (position, cell), increasing
        std::vector<uint32_t> free_cells(8192); for (uint32_t i=0;i<8192;++i) free_cells[i]=8191-i;
        int32_t next_pos = 0;
        for (int step=0; step<80; ++step) {
            const int what = rng()%6;
            if (what == 0 && !live.empty()) {          // rollback from a position
                const int32_t p = live[rng()%live.size()].first;
                s.truncate(s.rank_from(p));
                while (!live.empty() && live.back().first >= p) { free_cells.push_back(live.back().second); live.pop_back(); }
                next_pos = p;
                continue;
            }
            if (what == 1) { next_pos += 1 + rng()%9; }   // an image the draft never sees
            const int n = 1+rng()%9;
            std::vector<uint32_t> slots; std::vector<int32_t> p;
            for (int i=0;i<n;++i) { p.push_back(next_pos++); slots.push_back(free_cells.back()); free_cells.pop_back(); live.push_back({p.back(), slots.back()}); }
            GGML_ASSERT(s.apply(0, (int32_t) s.cells.size(), slots, p, p, p));
        }
        GGML_ASSERT(s.valid && !s.ranked() && s.cells.size() == live.size());
        // reference: complete position blocks in order, as the scan numbers them
        std::vector<int32_t> starts;
        for (size_t k=0;k+3<live.size();++k) { if (live[k].first%4 == 0 && live[k+3].first == live[k].first+3) { starts.push_back(live[k].first); } }
        GGML_ASSERT(s.complete() == starts.size());
        for (size_t b=0;b<starts.size();++b) {
            GGML_ASSERT(s.blk_start[b] == starts[b] && s.pos_of[s.blk_rank[b]] == starts[b]);
            for (int a=0;a<4;++a) { GGML_ASSERT(s.block_axis(b, a) == starts[b]); }
        }
        for (size_t k=0;k<live.size();++k) {
            const int32_t q = live[k].first, t = (q+1)/4*4;
            GGML_ASSERT(s.tail_start(k) == t);
            for (int j=0;j<q+1-t;++j) {
                const auto it = std::find_if(live.begin(), live.end(), [&](auto & e){ return e.first == t+j; });
                GGML_ASSERT(s.tail_cell(t, j) == (it == live.end() ? -1 : int32_t(it->second)));
            }
        }
    }
}

int main() {
    test_scalar_visibility();
    test_mrope_ranks();
    test_position_holes();
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
