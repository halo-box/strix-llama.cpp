// hooks: src/llama-ple-disk.h of the fork (3b76dee48, upstream 9048bad8d, new in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/lib/README.md

} // namespace ggml_fork
#include "ggml.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/lib/README.md

} // namespace ggml_fork
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/lib/README.md

// A weight table that stays in the GGUF file. Nothing is mapped or loaded for it at
// model load; each batch reads exactly the rows it gathers, with pread, and hands
// them over dequantized.
//
// Built for the qwen4exp n-gram hash embedding (per_layer_token_embd): 320 M rows of
// 90 bytes, a third of the model's bytes, of which one token touches 16 at unrelated
// hashed offsets. The working set is a few thousand rows per batch, so the table has
// no business being resident -- on a unified-memory machine it competes with the
// weights and the KV cache for the same RAM.
struct llama_ple_disk {
    struct params {
        int32_t n_threads   = 64;         // parallel readers: random reads on NVMe need queue depth
        size_t  cache_bytes = 256u << 20; // direct-mapped cache of raw rows; 0 disables
        bool    direct_io   = true;       // O_DIRECT, so the rows never enter the page cache either
        int64_t ne_out      = 0;          // > 0: rows are not a ggml type, gather() decodes them into ne_out floats
    };

    // decodes stored row `row` (row_size() bytes at src) into params.ne_out floats; called from several threads at once
    using row_decoder = std::function<void(int32_t row, const uint8_t * src, float * dst)>;

    llama_ple_disk(const std::string & fname, size_t offs, ggml_type type, int64_t ne0, int64_t nrows, const params & p);
    ~llama_ple_disk();

    // dequantize rows idx[0..n) into dst, laid out [ne0, n] ([params.ne_out, n] when set); aborts on an out-of-range index
    // a table opened with params.ne_out > 0 passes its decoder, which then replaces the type's to_float
    void gather(const int32_t * idx, size_t n, float * dst, const row_decoder & decode = nullptr);

    // ask the kernel to read rows idx[0..n) into the page cache ahead of a later gather; only advice, so a
    // wrong prediction costs readahead and nothing else. A no-op with O_DIRECT, whose reads bypass the
    // page cache. Safe to call from another thread while a gather runs.
    void prefetch(const int32_t * idx, size_t n) const;

    bool page_cached() const; // false when reads bypass the page cache (O_DIRECT)

    int64_t     n_rows()   const;
    int64_t     ne0()      const;
    size_t      row_size() const;
    std::string describe() const;

    struct impl;
    std::unique_ptr<impl> pimpl;
};
} // namespace ggml_fork
