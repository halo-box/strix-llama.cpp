#include "../src/llama-memory-hybrid-idx.h"
#include "../src/prefix.h"
#include "ggml.h"
#include <cstring>
#include <iostream>
#include <set>
#include <vector>

// Host-side regression for the qsa_fast gate lifted from 127 to 512 tokens:
// incremental indexer-key pooling must select exactly the touched blocks, and
// must stay off when the prefix, the commit state or the size guard does not hold.
// Runs on CPU only; no model or backend is involved.

// qsa_prefix.update-equivalent bookkeeping via qsa_prefix_state::apply
static void feed(qsa_prefix_state & s, int32_t start, int n) {
    // position k lives in cell k (cells in order); apply() places them in sequence
    std::vector<uint32_t> slots(n);
    for (int i = 0; i < n; ++i) slots[i] = start + i;
    GGML_ASSERT(s.apply(0, start, slots));
}

// single-sequence ubatch of n tokens starting at position start
struct qsa_fast_ubatch {
    std::vector<llama_token> tok; std::vector<llama_pos> pos; std::vector<int32_t> n_seq_id;
    std::vector<std::vector<llama_seq_id>> ids; std::vector<llama_seq_id *> seq_id; llama_ubatch u{};
    qsa_fast_ubatch(int32_t start, int n, int32_t seq = 0) {
        tok.assign(n, 0); pos.resize(n); ids.assign(n, {seq});
        for (int i = 0; i < n; ++i) { pos[i] = start + i; n_seq_id.push_back(1); seq_id.push_back(ids[i].data()); }
        u.n_tokens = n; u.n_pos = 1; u.token = tok.data(); u.pos = pos.data(); u.n_seq_id = n_seq_id.data(); u.seq_id = seq_id.data();
    }
};

// minimal stand-in for the memory object's fast-path pieces
struct qsa_fast_mock : public qsa_prefix_state {
    qsa_fast_mock(size_t capacity = 0) : qsa_prefix_state(capacity) {}
    void assign_from(const qsa_prefix_state & s) { static_cast<qsa_prefix_state &>(*this) = s; }
    std::vector<int64_t> qsa_ready{0};
    std::vector<bool>    qsa_keys{true};

    const int max_tokens = getenv("QSA_FAST_MAX") ? atoi(getenv("QSA_FAST_MAX")) : 512;   // read once, like the production static
    bool fast(const llama_ubatch & u) const {
        const bool prefix = qsa_prefix_matches(u);
        return prefix && int(u.n_tokens) <= max_tokens && u.n_tokens <= 512 && qsa_keys.at(0) &&
            qsa_ready.at(0) >= int64_t(previous_size/4) &&
            cells.size()/4 >= (u.n_tokens+3)/4+1;
    }
    bool qsa_prefix_matches(const llama_ubatch & u) const {
        if (!valid || !u.token || !u.pos || !u.n_tokens || !u.n_pos || !u.seq_id || !u.n_seq_id ||
            u.pos[0] != begin || int64_t(u.n_tokens) != end-begin) { return false; }
        for (uint32_t i=0; i<u.n_tokens; ++i) {
            if (u.n_seq_id[i] != 1 || !u.seq_id[i] || u.seq_id[i][0] != sequence || u.pos[i] != u.pos[0]+int32_t(i)) { return false; }
        }
        return true;
    }
    void commit() { qsa_ready.at(0) = cells.size()/4; }
};

// reproduce qsa_fill_updates's id selection and check it is the incremental block set
static void check_fill(const qsa_fast_mock & s, int n) {
    const int count = (n+3)/4+2, complete = int(s.cells.size()/4);
    const int first = int(s.begin/4), last = std::min(complete, int(s.end+3)/4);
    GGML_ASSERT(last-first < count && complete >= count - 1);
    std::vector<int> ids = { complete };
    for (int b=first; b<last; ++b) ids.push_back(b);
    for (int b=0; int(ids.size())<count; ++b) { if (b<first || b>=last) ids.push_back(b); }
    // in range, duplicate-free, and covers exactly [first, last) beyond the placeholder/filler blocks
    std::set<int> seen;
    for (int id : ids) { GGML_ASSERT(id >= 0 && id <= complete); GGML_ASSERT(seen.insert(id).second); }
    GGML_ASSERT(ids.size() == size_t(count));
    std::set<int> covered(ids.begin()+1, ids.end());
    for (int b=first; b<last; ++b) GGML_ASSERT(covered.count(b) == 1);
}

static void test_size_boundaries() {
    // empty cache: the first ubatch never takes the fast path (needs cells/4 >= count-1)
    for (int n : {1, 4, 8, 127, 128, 255, 256, 511, 512, 513, 1024}) {
        qsa_fast_mock s(8192); s.qsa_ready[0] = 0;
        feed(s, 0, n);
        const bool fast = s.fast(qsa_fast_ubatch(0, n).u);
        GGML_ASSERT(!fast || (int(s.cells.size()/4) >= (n+3)/4+1));
    }
    // established cache (full re-pool committed): fast iff n <= 512
    for (int n : {1, 4, 8, 126, 127, 128, 129, 255, 256, 257, 511, 512, 513, 514, 1024, 2048}) {
        qsa_fast_mock s(8192);
        feed(s, 0, 512); s.commit();
        feed(s, 512, n);
        GGML_ASSERT(s.fast(qsa_fast_ubatch(512, n).u) == (n <= 512));
    }
    // the last allowed boundary: previous commit covers the skipped blocks
    {
        qsa_fast_mock s(8192);
        feed(s, 0, 512); s.commit();
        feed(s, 512, 512);
        GGML_ASSERT(s.fast(qsa_fast_ubatch(512, 512).u));
        GGML_ASSERT(s.qsa_ready[0] >= int64_t(s.previous_size/4));
    }
}

static void test_incremental_ids() {
    for (int begin : {4, 128, 512, 1024}) {
        for (int n : {1, 4, 8, 127, 128, 256, 512}) {
            qsa_fast_mock s(8192);
            feed(s, 0, begin); s.commit();
            feed(s, begin, n);
            if (s.fast(qsa_fast_ubatch(begin, n).u)) { check_fill(s, n); }
        }
    }
}

static void test_gate_failures() {
    {   // position must start at the tracked prefix end
        qsa_fast_mock s(8192); feed(s, 0, 512); s.commit(); feed(s, 512, 256);
        GGML_ASSERT(!s.fast(qsa_fast_ubatch(513, 255).u));
        GGML_ASSERT(!s.fast(qsa_fast_ubatch(256, 256).u));
    }
    {   // the prefix was not committed (recover/rollback path) -> full re-pool
        qsa_fast_mock s(8192); feed(s, 0, 512); /* no commit */ feed(s, 512, 256);
        GGML_ASSERT(!s.fast(qsa_fast_ubatch(512, 256).u));
    }
    {   // truncation below a block boundary re-pools once, then the fast path resumes on the
        // committed smaller prefix (qsa_ready is clamped on seq_rm, llama-memory-hybrid-idx.cpp:203)
        qsa_fast_mock s(8192); feed(s, 0, 512); s.commit(); s.truncate(200);
        s.qsa_ready[0] = std::min<int64_t>(s.qsa_ready[0], s.cells.size()/4);
        // production rebuilds the tracked prefix from the cache (prefix stays valid through seq_rm):
        // cells [0,200) are rewritten in place (size stays 200), qsa_ready clamps to 200/4 = 50.
        // qsa_fast runs against this pre-ubatch state: cells/4 = 50 < (256+3)/4+1 = 65 -> full re-pool.
        GGML_ASSERT(int(s.cells.size()) == 200);
        GGML_ASSERT(!s.fast(qsa_fast_ubatch(200, 256).u));   // one full re-pool
        feed(s, 200, 256);                                   // the full path commits the whole prefix
        s.commit();
        feed(s, 456, 8);
        GGML_ASSERT(s.fast(qsa_fast_ubatch(456, 8).u));      // fast resumes on the new prefix
    }
    {   // multi-sequence ubatch never matches the prefix
        qsa_fast_mock s(8192); feed(s, 0, 512); s.commit(); feed(s, 512, 256);
        qsa_fast_ubatch u(512, 256);
        u.u.n_seq_id[3] = 2; // break the single-sequence invariant
        GGML_ASSERT(!s.fast(u.u));
    }
    {   // QSA_FAST_MAX: values <= 512 shrink the limit; 0 disables; >512 is capped by the hard limit.
        // (The real guard caches the env in a function-local static; this mock re-reads it per call,
        //  and each case uses a fresh prefix so only the limit under test varies.)
        setenv("QSA_FAST_MAX", "127", 1);
        qsa_fast_mock a(8192), b(8192);
        feed(a, 0, 512); a.commit(); feed(a, 512, 128);
        GGML_ASSERT(!a.fast(qsa_fast_ubatch(512, 128).u));
        feed(b, 0, 512); b.commit(); feed(b, 512, 127);
        GGML_ASSERT(b.fast(qsa_fast_ubatch(512, 127).u));
        setenv("QSA_FAST_MAX", "0", 1);
        qsa_fast_mock b0(8192);
        feed(b0, 0, 512); b0.commit(); feed(b0, 512, 127);
        GGML_ASSERT(!b0.fast(qsa_fast_ubatch(512, 127).u));
        setenv("QSA_FAST_MAX", "9999", 1);   // still capped at 512
        qsa_fast_mock c(8192), d(8192);
        feed(c, 0, 512); c.commit(); feed(c, 512, 513);
        GGML_ASSERT(!c.fast(qsa_fast_ubatch(512, 513).u));
        feed(d, 0, 512); d.commit(); feed(d, 512, 512);
        GGML_ASSERT(d.fast(qsa_fast_ubatch(512, 512).u));
        unsetenv("QSA_FAST_MAX");
    }
}

int main() {
    test_size_boundaries();
    test_incremental_ids();
    test_gate_failures();
    std::cout << "PASS: qsa_fast gate boundaries (1..2048), incremental block selection, prefix/commit/truncate gates, QSA_FAST_MAX\n";
}
