#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// Gathers the per-layer-embedding (PLE) rows of a batch on a worker thread while the GPU computes the
// current ubatch, so the next ubatch's set_input finds its rows ready instead of reading them with the
// GPU idle (A1, 2026-10-10).
//
// The batch is walked in token order in windows of `window` tokens into a ring of 2 windows. The row
// indices come from the caller (hashed from the batch's own tokens); set_input recomputes them from the
// cache as before and takes a token's rows from the ring only when the two index sets are identical, so
// the bytes handed to the graph are exactly what the synchronous gather would have produced. Anything else
// (unknown predecessors, a different ubatch order) is gathered synchronously as a miss.
struct llama_ple_async {
    // gathers n_rows rows idx[0..n_rows) into dst laid out [dim, n_rows]; must be safe to call from the worker
    using gather_fn = std::function<void(const int32_t * idx, size_t n_rows, float * dst)>;

    llama_ple_async(int64_t n_heads, int64_t dim, gather_fn fn);
    ~llama_ple_async();

    // begin a batch: idx_all is [n_heads, n_tokens] in batch order, -1 for a token whose rows are not known
    void start(std::vector<int32_t> idx_all, int64_t n_tokens, int64_t window);

    // take the next n tokens at the cursor: the rows of token i go to dst + i*n_heads*dim when the ring holds
    // exactly idx[i*n_heads .. (i+1)*n_heads); miss[i] = 1 otherwise. Blocks until the worker has produced
    // them. Returns the number of hits.
    int64_t consume(const int32_t * idx, int64_t n, float * dst, std::vector<uint8_t> & miss);

    // end the batch: stops the worker and drops the ring contents
    void stop();

    bool active() const { return running; }

    // running totals, for the engagement log
    uint64_t st_ubatches = 0, st_hits = 0, st_misses = 0;
    double   st_wait_ms  = 0;

private:
    void worker_main();

    const int64_t n_heads;
    const int64_t dim;
    gather_fn     gather;

    std::vector<int32_t> idx_truth;   // as given, -1 = unknown
    std::vector<int32_t> idx_fill;    // unknown rows replaced by row 0 so the gather never sees -1
    std::vector<float>   ring;        // cap tokens x n_heads x dim
    int64_t n_tokens = 0;
    int64_t window   = 0;
    int64_t cap      = 0;             // ring capacity in tokens, a multiple of window

    std::mutex              mtx;
    std::condition_variable cv;
    int64_t produced = 0;             // tokens gathered so far (batch order)
    int64_t consumed = 0;             // tokens taken so far
    bool    stop_req = false;
    bool    done     = false;         // worker finished (or stopped)
    bool    running  = false;
    std::thread th;
};
