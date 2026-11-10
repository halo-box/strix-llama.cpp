// The reserve views of the hybrid-idx memory (llama_context::sched_reserve): init_full views every stream of the cache,
// init_full_n_kv(n_kv) limits the views to n_kv cells (reserve pass B), and init_full_n_kv(n_kv, 1) to the first stream as
// well (the one-stream reserve graphs of a split cache with several sequences: a one-sequence ubatch there is a one-stream
// graph, which a view of every stream cannot build; Prove's -np 2 split run at ub8192 grew past the reserve on them).
#include "qsa-mem-fixture.h"

#include <cstdio>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static const llama_memory_hybrid_idx_context * as_idx(const llama_memory_context_ptr & p) {
    return static_cast<const llama_memory_hybrid_idx_context *>(p.get());
}

static void check_views(uint32_t kv_size, uint32_t n_seq_max, bool unified) {
    qsa_fixture fx(kv_size, n_seq_max, unified);
    const uint32_t n_stream_cache = unified ? 1 : n_seq_max;

    auto full = fx.mem->init_full();
    CHECK(full && full->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
    CHECK(as_idx(full)->get_n_stream() == n_stream_cache);
    CHECK(as_idx(full)->get_idx()->get_n_kv() == kv_size);
    CHECK(as_idx(full)->get_attn()->get_n_kv() == kv_size);

    // pass B: n_kv cells, every stream
    auto b = fx.mem->init_full_n_kv(kv_size / 2);
    CHECK(b && b->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
    CHECK(as_idx(b)->get_n_stream() == n_stream_cache);
    CHECK(as_idx(b)->get_idx()->get_n_kv() == kv_size / 2);
    CHECK(as_idx(b)->get_attn()->get_n_kv() == kv_size / 2);

    // the one-stream view at the full per-stream n_kv
    auto one = fx.mem->init_full_n_kv(kv_size, 1);
    CHECK(one && one->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
    CHECK(as_idx(one)->get_n_stream() == 1);
    CHECK(as_idx(one)->get_idx()->get_n_kv() == kv_size);
    CHECK(as_idx(one)->get_attn()->get_n_kv() == kv_size);

    // n_kv past the cache is clamped, n_stream past the cache's streams is the cache's
    auto big = fx.mem->init_full_n_kv(2*kv_size, n_stream_cache + 3);
    CHECK(as_idx(big)->get_n_stream() == n_stream_cache);
    CHECK(as_idx(big)->get_idx()->get_n_kv() == kv_size);

    fprintf(stderr, "views kv_size=%u n_seq_max=%u unified=%d: full %u streams, pass B %u streams, one-stream %u\n", kv_size, n_seq_max,
            (int) unified, as_idx(full)->get_n_stream(), as_idx(b)->get_n_stream(), as_idx(one)->get_n_stream());
}

int main() {
    check_views(4096, 2, false);
    check_views(4096, 4, false);
    check_views(4096, 2, true);
    check_views(4096, 1, false);
    if (n_fail) {
        fprintf(stderr, "test-qsa-reserve-view: %d FAILED\n", n_fail);
        return 1;
    }
    fprintf(stderr, "test-qsa-reserve-view: OK\n");
    return 0;
}
