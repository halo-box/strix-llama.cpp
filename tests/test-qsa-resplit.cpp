// PLAN-v3 §4.3 (P2-2, R2-RES spec): the non-compact re-split of llama_memory_hybrid_idx::init_batch, on the production memory.
//
// Layouts: (a) one sequence with 2-D cells at several depths; (b) R1-CB, two sequences of one unified stream with a seq_cp
// prefix ending inside a block; (c) R1-C layout 3, images in two slots of one unified stream; (d) -np 2 split with one image
// slot. Batches: random token batches with 1-4 sequences and 1..3 x n_ubatch tokens each, and embedding batches of images
// with 2-D positions. Grid: n_keep_tail {0, 4} x n_ubatch {512, 1024, 4096, 8192} x S {1, 2}.
// For every ubatch of init_batch (with its NC0 hook), applied to the real attention and indexer caches, with the graph-time
// predicate (qsa_scalar_visibility on the indexer cells, the function build_qsa_top_k calls):
//   (i)   graph-NC(u) => NC0(u)
//   (ii)  NC0(u) => u.n_tokens <= U = min(n_ubatch, S x 512)
//   (iii) every batch token in exactly one ubatch
//   (iv)  the last n_keep_tail tokens of every sequence share one ubatch
//   (v)   attention cells == indexer cells after every apply
//   (vi)  replays <= #ubatches
// Pass: 0 violations over the fixed seeds, and the re-split engaged (replays > 0) on the multi-sequence layouts.
#include "qsa-mem-fixture.h"

#include <cinttypes>
#include <map>
#include <random>
#include <set>
#include <tuple>

static int64_t g_viol[7] = { 0 };
static int64_t g_ubatches = 0, g_nc = 0, g_nc0 = 0, g_replays = 0, g_batches = 0;

#define VIOL(k, ...) do { if (g_viol[k]++ < 10) { fprintf(stderr, "violation (%d): ", k); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

using qsa_key = std::tuple<llama_seq_id, llama_pos, llama_pos, llama_pos>;

// (v): the two caches hold the same cells (slot, position, extents, sequences) in every stream
static bool same_cells(const llama_memory_hybrid_idx * mem, uint32_t n_seq_max, bool unified) {
    const uint32_t n_streams = unified ? 1 : n_seq_max;
    for (uint32_t s = 0; s < n_streams; ++s) {
        const auto & a = mem->get_mem_attn()->get_cells((llama_seq_id) s);
        const auto & b = mem->get_mem_idx()->get_cells((llama_seq_id) s);
        if (a.size() != b.size() || a.get_used() != b.get_used()) { return false; }
        for (uint32_t j = 0; j < a.size(); ++j) {
            if (a.is_empty(j) != b.is_empty(j)) { return false; }
            if (a.is_empty(j)) { continue; }
            if (a.pos_get(j) != b.pos_get(j) || a.seq_get_all(j) != b.seq_get_all(j) ||
                    a.ext_get(j).x != b.ext_get(j).x || a.ext_get(j).y != b.ext_get(j).y) { return false; }
        }
    }
    return true;
}

static void run_batch(qsa_fixture & f, const std::vector<qsa_fixture::tok> & toks, uint32_t n_ubatch, uint32_t n_keep_tail, bool multi) {
    const uint32_t U = f.mem->qsa_noncompact_cap(n_ubatch);
    std::vector<uint8_t> graph_nc;
    std::vector<uint32_t> n_tok;
    std::map<qsa_key, std::vector<uint32_t>> where;   // batch token -> the ubatches that carried it
    const bool ok = f.feed(toks, n_ubatch, [&](const llama_memory_hybrid_idx_context * mctx, const llama_ubatch & u, uint32_t k) {
        const int64_t n_kv = mctx->get_idx()->get_n_kv();
        graph_nc.push_back(n_kv > 2048 + 4 - 1 && !mctx->qsa_scalar_visibility(u, 4) ? 1 : 0);
        n_tok.push_back(u.n_tokens);
        for (uint32_t i = 0; i < u.n_tokens; ++i) {
            where[{ u.seq_id[i][0], u.pos[i], u.pos[i + u.n_tokens], u.pos[i + 2*u.n_tokens] }].push_back(k);
        }
        if (!same_cells(f.mem.get(), f.n_seq_max, f.unified)) {
            VIOL(5, "attention and indexer cells differ after ubatch %u", k);
        }
    });
    if (!ok) {
        VIOL(0, "a batch of %zu tokens was refused", toks.size());
        return;
    }
    g_batches++;
    const auto & nc0 = f.mem->qsa_last_nc0();
    const uint32_t n_ub = (uint32_t) n_tok.size();
    g_ubatches += n_ub;
    if (multi && nc0.size() != n_ub) {
        VIOL(1, "NC0 has %zu entries for %u ubatches", nc0.size(), n_ub);
        return;
    }
    for (uint32_t k = 0; k < n_ub; ++k) {
        const bool n0 = multi && nc0[k];
        g_nc += graph_nc[k];
        g_nc0 += n0;
        if (graph_nc[k] && !n0) {
            VIOL(1, "ubatch %u of %u tokens is non-compact at graph time but not NC0 (%s)", k, n_tok[k], multi ? "multi" : "single-sequence context");
        }
        if (n0 && n_tok[k] > U) {
            VIOL(2, "ubatch %u is NC0 with %u tokens > U = %u", k, n_tok[k], U);
        }
    }
    // (iii)
    size_t n_seen = 0;
    for (const auto & t : toks) {
        const auto it = where.find({ t.seq, t.p, t.y, t.x });
        if (it == where.end() || it->second.size() != 1) {
            VIOL(3, "token seq %d pos (%d, %d, %d) is in %zu ubatches", t.seq, t.p, t.y, t.x, it == where.end() ? 0 : it->second.size());
        } else {
            n_seen++;
        }
    }
    if (where.size() != toks.size() || n_seen != toks.size()) {
        VIOL(3, "%zu batch tokens, %zu distinct ubatch tokens", toks.size(), where.size());
    }
    // (iv)
    if (n_keep_tail > 0) {
        std::map<llama_seq_id, std::vector<qsa_key>> per_seq;
        for (const auto & t : toks) { per_seq[t.seq].push_back({ t.seq, t.p, t.y, t.x }); }
        for (const auto & e : per_seq) {
            const size_t n = std::min<size_t>(n_keep_tail, e.second.size());
            std::set<uint32_t> ub;
            for (size_t i = e.second.size() - n; i < e.second.size(); ++i) { ub.insert(where[e.second[i]].front()); }
            if (ub.size() != 1) {
                VIOL(4, "the last %zu tokens of seq %d span %zu ubatches", n, e.first, ub.size());
            }
        }
    }
    // (vi)
    g_replays += f.mem->qsa_last_replays();
    if (f.mem->qsa_last_replays() > n_ub) {
        VIOL(6, "%u replays for %u ubatches", f.mem->qsa_last_replays(), n_ub);
    }
}

// a random batch: tokens for 1-4 of the active sequences (1..3 x n_ubatch each), or 1-2 images; at most `room` tokens
static std::vector<qsa_fixture::tok> random_batch(qsa_fixture & f, std::mt19937 & rng, const std::vector<llama_seq_id> & seqs,
        uint32_t n_ubatch, size_t room) {
    std::vector<llama_seq_id> pick = seqs;
    std::shuffle(pick.begin(), pick.end(), rng);
    pick.resize(1 + rng() % std::min<size_t>(4, pick.size()));
    std::sort(pick.begin(), pick.end());
    std::vector<qsa_fixture::tok> toks;
    if (rng() % 3 == 0) {
        for (size_t k = 0; k < std::min<size_t>(2, pick.size()); ++k) {
            const uint32_t h = 4 + rng() % 37, w = 4 + rng() % 37;
            if (toks.size() + h*w > room) { break; }
            const auto img = f.image(pick[k], h, w);
            toks.insert(toks.end(), img.begin(), img.end());
        }
    } else {
        for (const auto s : pick) {
            const uint32_t n = std::min<size_t>(1 + rng() % (3*n_ubatch), room > toks.size() ? (room - toks.size()) / pick.size() : 0);
            if (n == 0) { continue; }
            const auto t = f.text(s, n);
            toks.insert(toks.end(), t.begin(), t.end());
        }
    }
    return toks;
}

static size_t free_cells(const qsa_fixture & f, llama_seq_id s) {
    const auto & c = f.mem->get_mem_attn()->get_cells(s);
    return c.size() - c.get_used();
}

int main() {
    const char * layouts[] = { "a:single-2d", "b:seqcp", "c:layout3", "d:split-np2", "a1:np1" };
    int64_t replays_multi = 0;
    for (const char * layout : layouts) {
        const std::string L = layout;
        for (uint32_t n_keep_tail : { 0u, 4u }) {
            for (uint32_t n_ubatch : { 512u, 1024u, 4096u, 8192u }) {
                for (uint32_t S : { 1u, 2u }) {
                    const bool split = L[0] == 'd';
                    const bool np1 = L == "a1:np1";
                    const uint32_t n_seq_max = np1 ? 1 : split ? 2 : 4;
                    const uint32_t kv = split ? 131072 : 327680;
                    qsa_fixture f(kv, n_seq_max, !split, n_keep_tail > 0 ? n_keep_tail - 1 : 0, S);
                    std::mt19937 rng(1000003u * n_ubatch + 97u * S + n_keep_tail + (uint32_t) L[0]);
                    std::vector<llama_seq_id> active;
                    const uint32_t ub0 = 4096;
                    if (L[0] == 'a') {
                        // 2-D cells at depth 0, about 5k and past 32k
                        f.feed(f.image(0, 32, 32, 0), ub0);
                        f.feed(f.text(0, 5000), ub0);
                        f.feed(f.image(0, 24, 40), ub0);
                        f.feed(f.text(0, 28000), ub0);
                        f.feed(f.image(0, 30, 30), ub0);
                        f.feed(f.text(0, 300), ub0);
                        active = { 0 };
                        if (!np1) { active = { 0, 1, 2, 3 }; }
                    } else if (L[0] == 'b') {
                        f.feed(f.text(0, 8193), ub0);
                        f.mem->seq_cp(0, 1, -1, -1);
                        f.next_pos[1] = f.next_pos[0];
                        active = { 0, 1, 2, 3 };
                    } else if (L[0] == 'c') {
                        f.feed(f.text(0, 20000), ub0);
                        f.feed(f.image(0, 32, 32), ub0);
                        f.feed(f.text(0, 1200), ub0);
                        f.feed(f.image(1, 32, 32, 0), ub0);
                        f.feed(f.text(1, 2000), ub0);
                        active = { 0, 1, 2, 3 };
                    } else {
                        f.feed(f.text(0, 9000), ub0);
                        f.feed(f.image(0, 32, 32), ub0);
                        f.feed(f.text(1, 6000), ub0);
                        active = { 0, 1 };
                    }
                    for (int b = 0; b < 4; ++b) {
                        size_t room = SIZE_MAX;
                        for (const auto s : active) { room = std::min(room, free_cells(f, s)); }
                        room = room > 1024 ? room - 1024 : 0;
                        if (room < 64) { break; }
                        const auto toks = random_batch(f, rng, active, n_ubatch, room);
                        if (toks.empty()) { continue; }
                        const int64_t before = g_replays;
                        run_batch(f, toks, n_ubatch, n_keep_tail, !np1);
                        if (!np1) { replays_multi += g_replays - before; }
                    }
                }
            }
        }
        fprintf(stderr, "layout %s done: batches %" PRId64 ", ubatches %" PRId64 ", graph-NC %" PRId64 ", NC0 %" PRId64 ", replays %" PRId64 "\n",
                layout, g_batches, g_ubatches, g_nc, g_nc0, g_replays);
    }
    int64_t total = 0;
    for (int k = 0; k < 7; ++k) { total += g_viol[k]; }
    printf("test-qsa-resplit: %" PRId64 " batches, %" PRId64 " ubatches, %" PRId64 " graph-time non-compact, %" PRId64 " NC0, %" PRId64 " replays (multi-sequence %" PRId64 ")\n",
            g_batches, g_ubatches, g_nc, g_nc0, g_replays, replays_multi);
    printf("violations: refused %" PRId64 ", (i) %" PRId64 ", (ii) %" PRId64 ", (iii) %" PRId64 ", (iv) %" PRId64 ", (v) %" PRId64 ", (vi) %" PRId64 "\n",
            g_viol[0], g_viol[1], g_viol[2], g_viol[3], g_viol[4], g_viol[5], g_viol[6]);
    const bool engaged = replays_multi > 0 && g_nc > 0;
    if (!engaged) {
        printf("the re-split never engaged: the test proves nothing\n");
    }
    printf("%s\n", total == 0 && engaged ? "PASS" : "FAILED");
    return total == 0 && engaged ? 0 : 1;
}
