#include "llama-ple-disk.h"

#include "llama-bounded-file.h"
#include "llama-impl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

struct llama_ple_disk::impl {
    std::string fname;
    bool        direct = false;
    size_t      offs   = 0;
    ggml_type   type   = GGML_TYPE_COUNT;
    int64_t     ne0    = 0;
    int64_t     nrows  = 0;
    size_t      rs     = 0;     // bytes per row
    ggml_to_float_t to_float = nullptr;
    std::unique_ptr<llama_bounded_file> file;
    int         fd     = -1;    // buffered-mode fd, used only for prefetch() fadvise hints

    // direct-mapped cache of raw rows: slot = row & mask, tag = row
    size_t               n_slots = 0;
    size_t               mask    = 0;
    std::vector<int64_t> tags;
    std::vector<uint8_t> slab;

    // per-gather scratch, reused
    std::vector<std::pair<int32_t, uint32_t>>  order;    // (row, position in idx) sorted by row: equal rows are adjacent
    std::vector<uint32_t>                      runs;     // order[runs[u]..runs[u+1]) are the positions of distinct row u
    std::vector<uint8_t>                       raw;      // [distinct rows, rs]
    std::vector<std::pair<int32_t, uint32_t>>  misses;   // (row, index into raw)
    float *                                    dst     = nullptr; // destination of the gather in progress
    llama_bounded_file::buffer                 bounce0;  // calling-thread bounce buffer

    std::mutex mtx; // one gather at a time; a model is shared by every context built on it

    // worker pool, started on first use; a job is n_items items handed out `grain` at a time
    enum class job_kind { read, dequant };
    job_kind                 job       = job_kind::read;
    size_t                   n_items   = 0;
    size_t                   grain     = 1;
    int32_t                  n_threads = 1;
    std::vector<std::thread> workers;
    std::mutex               pm;
    std::condition_variable  cv_work;
    std::condition_variable  cv_done;
    uint64_t                 gen     = 0;
    size_t                   pending = 0;
    std::atomic<size_t>      next{0};
    std::exception_ptr       worker_error; // first error seen by a worker, rethrown on the calling thread
    bool                     stop    = false;

    uint64_t st_calls = 0, st_rows = 0, st_uniq = 0, st_hits = 0, st_reads = 0, st_bytes = 0;
    double   st_ms = 0;

    impl(const std::string & fname, size_t offs, ggml_type type, int64_t ne0, int64_t nrows, const params & p)
        : fname(fname), offs(offs), type(type), ne0(ne0), nrows(nrows) {
#if defined(_WIN32)
        throw std::runtime_error("llama_ple_disk: not supported on Windows");
#else
        if (ne0 <= 0 || nrows <= 0) {
            throw std::runtime_error("llama_ple_disk: invalid table shape");
        }
        if (ggml_is_quantized(type) && ne0 % ggml_blck_size(type) != 0) {
            throw std::runtime_error(format("llama_ple_disk: row of %lld %s elements is not a whole number of blocks",
                                            (long long) ne0, ggml_type_name(type)));
        }
        rs = ggml_row_size(type, ne0);
        if (type != GGML_TYPE_F32) {
            to_float = ggml_get_type_traits(type)->to_float;
            if (to_float == nullptr) {
                throw std::runtime_error(format("llama_ple_disk: no dequantizer for %s", ggml_type_name(type)));
            }
        }

        llama_bounded_file::params fp;
        fp.direct_io = p.direct_io;
        file   = std::make_unique<llama_bounded_file>(fname, fp);
        direct = file->direct_io();
        if (rs == 0 || (uint64_t) nrows > UINT64_MAX / rs) {
            throw std::runtime_error("llama_ple_disk: table extent overflow");
        }
        const uint64_t bytes = (uint64_t) nrows * rs;
        if (offs > file->size() || bytes > file->size() - offs) {
            throw std::runtime_error("llama_ple_disk: table extent is outside the file");
        }
        if (!direct) {
            // best-effort fd used only to hint the page cache in prefetch(); failure is not fatal
            fd = open(fname.c_str(), O_RDONLY | O_CLOEXEC);
        }

        n_threads = std::max<int32_t>(1, p.n_threads);

        if (p.cache_bytes >= rs) {
            size_t n = p.cache_bytes / rs;
            n_slots = 1;
            while (n_slots * 2 <= n) {
                n_slots *= 2;
            }
            mask = n_slots - 1;
            tags.assign(n_slots, -1);
            slab.resize(n_slots * rs);
        }
#endif
    }

    ~impl() {
#if !defined(_WIN32)
        {
            std::lock_guard<std::mutex> lk(pm);
            stop = true;
        }
        cv_work.notify_all();
        for (auto & w : workers) {
            w.join();
        }
        if (fd >= 0) {
            close(fd);
        }
#endif
    }

#if !defined(_WIN32)
    void read_row(int64_t row, uint8_t * dst, llama_bounded_file::buffer & bounce) const {
        const uint64_t off = (uint64_t) offs + (uint64_t) row * rs;
        file->read(off, dst, rs, bounce);
    }

    // dequantize distinct row u into the first position that asked for it, then copy it to the others
    void dequant_row(size_t u) const {
        const uint8_t * src   = raw.data() + u * rs;
        float *         first = dst + (size_t) order[runs[u]].second * ne0;
        if (to_float) {
            to_float(src, first, ne0);
        } else {
            memcpy(first, src, rs);
        }
        for (uint32_t j = runs[u] + 1; j < runs[u + 1]; ++j) {
            memcpy(dst + (size_t) order[j].second * ne0, first, (size_t) ne0 * sizeof(float));
        }
    }

    void run_items(llama_bounded_file::buffer & bounce) {
        for (;;) {
            const size_t i0 = next.fetch_add(grain);
            if (i0 >= n_items) {
                break;
            }
            const size_t i1 = std::min(i0 + grain, n_items);
            for (size_t i = i0; i < i1; ++i) {
                if (job == job_kind::read) {
                    read_row(misses[i].first, raw.data() + (size_t) misses[i].second * rs, bounce);
                } else {
                    dequant_row(i);
                }
            }
        }
    }

    void worker() {
        llama_bounded_file::buffer bounce;
        uint64_t                   seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(pm);
                cv_work.wait(lk, [&] { return stop || gen != seen; });
                if (stop) {
                    break;
                }
                seen = gen;
            }
            std::exception_ptr error;
            try {
                if (direct && bounce.empty()) {
                    bounce = file->make_buffer(rs);
                }
                run_items(bounce);
            } catch (...) {
                error = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lk(pm);
                if (error && !worker_error) {
                    worker_error = error;
                }
                if (--pending == 0) {
                    cv_done.notify_one();
                }
            }
        }
    }

    // run `kind` over n items; up to serial_max of them stay on the calling thread,
    // which otherwise works alongside the pool instead of waiting for it
    void run_job(job_kind kind, size_t n, size_t g, size_t serial_max) {
        job     = kind;
        n_items = n;
        grain   = g;
        if (direct && bounce0.empty()) {
            bounce0 = file->make_buffer(rs);
        }
        if (n_threads <= 1 || n <= serial_max) {
            next = 0;
            run_items(bounce0);
            return;
        }
        if (workers.empty()) {
            workers.reserve(n_threads);
            for (int32_t i = 0; i < n_threads; ++i) {
                workers.emplace_back([this] { worker(); });
            }
        }
        {
            std::lock_guard<std::mutex> lk(pm);
            next         = 0;
            pending      = workers.size();
            worker_error = nullptr;
            ++gen;
        }
        cv_work.notify_all();
        // the calling thread also helps drain items, so its own read errors join the same
        // worker_error slot instead of throwing early and abandoning still-running workers
        std::exception_ptr error;
        try {
            run_items(bounce0);
        } catch (...) {
            error = std::current_exception();
        }
        std::unique_lock<std::mutex> lk(pm);
        cv_done.wait(lk, [&] { return pending == 0; });
        if (!error) {
            error = worker_error;
        }
        lk.unlock();
        if (error) {
            std::rethrow_exception(error);
        }
    }

    void gather(const int32_t * idx, size_t n, float * out) {
        std::lock_guard<std::mutex> lk(mtx);
        const auto t0 = std::chrono::steady_clock::now();
        GGML_ASSERT(n < UINT32_MAX);

        // sort the positions by row so equal rows are adjacent: each distinct row is read and dequantized once
        order.resize(n);
        for (size_t k = 0; k < n; ++k) {
            order[k] = { idx[k], (uint32_t) k };
        }
        std::sort(order.begin(), order.end());
        if (n && (order.front().first < 0 || (int64_t) order.back().first >= nrows)) {
            GGML_ABORT("llama_ple_disk: row index out of range (%d..%d of %lld rows)",
                       order.front().first, order.back().first, (long long) nrows);
        }
        runs.clear();
        for (size_t k = 0; k < n; ++k) {
            if (k == 0 || order[k].first != order[k - 1].first) {
                runs.push_back((uint32_t) k);
            }
        }
        const size_t n_uniq = runs.size();
        runs.push_back((uint32_t) n);

        raw.resize(n_uniq * rs);
        misses.clear();
        for (size_t u = 0; u < n_uniq; ++u) {
            const int32_t row = order[runs[u]].first;
            if (n_slots) {
                const size_t slot = (size_t) row & mask;
                if (tags[slot] == row) {
                    memcpy(raw.data() + u * rs, slab.data() + slot * rs, rs);
                    continue;
                }
            }
            misses.emplace_back(row, (uint32_t) u);
        }

        if (!misses.empty()) {
            run_job(job_kind::read, misses.size(), 1, 2);
            if (n_slots) {
                for (const auto & m : misses) {
                    const size_t slot = (size_t) m.first & mask;
                    tags[slot] = m.first;
                    memcpy(slab.data() + slot * rs, raw.data() + (size_t) m.second * rs, rs);
                }
            }
        }

        dst = out;
        run_job(job_kind::dequant, n_uniq, 64, 256);
        dst = nullptr;

        st_calls += 1;
        st_rows  += n;
        st_uniq  += n_uniq;
        st_hits  += n_uniq - misses.size();
        st_reads += misses.size();
        for (const auto & m : misses) {
            st_bytes += file->read_size((uint64_t) offs + (uint64_t) m.first * rs, rs);
        }
        st_ms    += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
#endif
};

llama_ple_disk::llama_ple_disk(const std::string & fname, size_t offs, ggml_type type, int64_t ne0, int64_t nrows, const params & p)
    : pimpl(std::make_unique<impl>(fname, offs, type, ne0, nrows, p)) {}

llama_ple_disk::~llama_ple_disk() {
    if (pimpl->st_calls) {
        LLAMA_LOG_INFO("%s: %llu batches, %llu rows gathered (%llu unique): %llu cache hits, %llu disk reads (%.1f MiB), %.1f ms total\n",
                       __func__, (unsigned long long) pimpl->st_calls, (unsigned long long) pimpl->st_rows,
                       (unsigned long long) pimpl->st_uniq, (unsigned long long) pimpl->st_hits,
                       (unsigned long long) pimpl->st_reads, pimpl->st_bytes / (1024.0 * 1024.0), pimpl->st_ms);
    }
}

void llama_ple_disk::gather(const int32_t * idx, size_t n, float * dst) {
#if defined(_WIN32)
    GGML_UNUSED(idx); GGML_UNUSED(n); GGML_UNUSED(dst);
    GGML_ABORT("llama_ple_disk: not supported on Windows");
#else
    pimpl->gather(idx, n, dst);
#endif
}

void llama_ple_disk::prefetch(const int32_t * idx, size_t n) const {
#if defined(_WIN32)
    GGML_UNUSED(idx); GGML_UNUSED(n);
#else
    if (pimpl->direct || pimpl->fd < 0 || n == 0) {
        return;
    }
    std::vector<int32_t> uniq;
    uniq.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (idx[i] >= 0 && (int64_t) idx[i] < pimpl->nrows) {
            uniq.push_back(idx[i]);
        }
    }
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
#if defined(POSIX_FADV_WILLNEED)
    for (const int32_t row : uniq) {
        const off_t off = (off_t) pimpl->offs + (off_t) row * (off_t) pimpl->rs;
        posix_fadvise(pimpl->fd, off, (off_t) pimpl->rs, POSIX_FADV_WILLNEED);
    }
#endif
#endif
}

bool llama_ple_disk::page_cached() const { return !pimpl->direct; }

int64_t llama_ple_disk::n_rows()   const { return pimpl->nrows; }
int64_t llama_ple_disk::ne0()      const { return pimpl->ne0;   }
size_t  llama_ple_disk::row_size() const { return pimpl->rs;    }

std::string llama_ple_disk::describe() const {
    return format("%s @ %zu: %lld rows x %lld %s (%zu B/row, %.2f GiB), %s I/O, %d threads, row cache %zu MiB",
                  pimpl->fname.c_str(), pimpl->offs, (long long) pimpl->nrows, (long long) pimpl->ne0,
                  ggml_type_name(pimpl->type), pimpl->rs, (double) pimpl->nrows * pimpl->rs / (1024.0 * 1024.0 * 1024.0),
                  pimpl->direct ? "direct" : "buffered", pimpl->n_threads,
                  pimpl->n_slots * pimpl->rs / (1024 * 1024));
}
