#include "llama-ple-async.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstring>

llama_ple_async::llama_ple_async(int64_t n_heads, int64_t dim, gather_fn fn)
    : n_heads(n_heads), dim(dim), gather(std::move(fn)) {}

llama_ple_async::~llama_ple_async() {
    stop();
}

void llama_ple_async::start(std::vector<int32_t> idx_all, int64_t n_tok, int64_t win) {
    stop();
    GGML_ASSERT(win > 0 && n_tok > 0 && (int64_t) idx_all.size() == n_tok * n_heads);

    idx_truth = std::move(idx_all);
    idx_fill  = idx_truth;
    for (auto & r : idx_fill) {
        if (r < 0) {
            r = 0;
        }
    }
    n_tokens = n_tok;
    window   = win;
    cap      = 2 * window;
    ring.resize((size_t) cap * n_heads * dim);

    produced = 0;
    consumed = 0;
    stop_req = false;
    done     = false;
    running  = true;
    th = std::thread([this] { worker_main(); });
}

void llama_ple_async::worker_main() {
    for (int64_t t0 = 0; t0 < n_tokens; t0 += window) {
        const int64_t n = std::min(window, n_tokens - t0);
        {
            // the window's slots must be free: tokens below consumed + cap
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [&] { return stop_req || t0 + n <= consumed + cap; });
            if (stop_req) {
                break;
            }
        }
        // windows start at multiples of `window` and cap is a multiple of it, so a window is contiguous in the ring
        const int64_t slot = t0 % cap;
        gather(idx_fill.data() + t0 * n_heads, (size_t) (n * n_heads), ring.data() + (size_t) slot * n_heads * dim);
        {
            std::lock_guard<std::mutex> lk(mtx);
            produced = t0 + n;
        }
        cv.notify_all();
    }
    {
        std::lock_guard<std::mutex> lk(mtx);
        done = true;
    }
    cv.notify_all();
}

int64_t llama_ple_async::consume(const int32_t * idx, int64_t n, float * dst, std::vector<uint8_t> & miss) {
    miss.assign((size_t) n, 1);
    if (!running) {
        return 0;
    }
    const int64_t c0 = consumed;
    const int64_t c1 = std::min(c0 + n, n_tokens);   // tokens past the batch end are plain misses

    const auto t0 = std::chrono::steady_clock::now();
    {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [&] { return done || produced >= c1; });
    }
    st_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    const int64_t avail = std::min(c1, produced);     // a stopped worker may have produced less
    const size_t  tok_bytes = (size_t) n_heads * dim * sizeof(float);

    int64_t hits = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t t = c0 + i;
        if (t >= avail) {
            continue;
        }
        const int32_t * truth = idx_truth.data() + t * n_heads;
        if (truth[0] < 0 || memcmp(truth, idx + i * n_heads, (size_t) n_heads * sizeof(int32_t)) != 0) {
            continue;
        }
        memcpy(dst + (size_t) i * n_heads * dim, ring.data() + (size_t) (t % cap) * n_heads * dim, tok_bytes);
        miss[(size_t) i] = 0;
        hits++;
    }
    {
        std::lock_guard<std::mutex> lk(mtx);
        consumed = c0 + n;
    }
    cv.notify_all();

    st_ubatches += 1;
    st_hits     += (uint64_t) hits;
    st_misses   += (uint64_t) (n - hits);
    return hits;
}

void llama_ple_async::stop() {
    if (!running) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mtx);
        stop_req = true;
    }
    cv.notify_all();
    if (th.joinable()) {
        th.join();
    }
    running = false;
    idx_truth.clear();
    idx_fill.clear();
}
