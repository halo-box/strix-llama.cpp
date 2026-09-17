#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum {
    LLAMA_ENGRAM_LAYERS    = 2,
    LLAMA_ENGRAM_NGRAM     = 4,
    LLAMA_ENGRAM_HEADS     = 8,
    LLAMA_ENGRAM_COLS      = 24,
    LLAMA_ENGRAM_DIM       = 256,
    LLAMA_ENGRAM_ROW_BYTES = 264,
    LLAMA_ENGRAM_DEAD      = -1,
};

struct llama_engram_layout {
    std::string encoding;
    std::array<uint32_t, LLAMA_ENGRAM_LAYERS> layer_ids = {};
    std::vector<uint32_t> token_map;
    uint32_t compressed_vocab_size = 0;
    uint32_t pad_id = 0;
    std::array<uint32_t, LLAMA_ENGRAM_LAYERS> rows = {};
    std::array<std::array<uint64_t, LLAMA_ENGRAM_NGRAM>, LLAMA_ENGRAM_LAYERS> multipliers = {};
    std::array<std::array<uint32_t, LLAMA_ENGRAM_COLS>, LLAMA_ENGRAM_LAYERS> primes = {};
};

struct llama_engram_history {
    // Newest compressed token first. LLAMA_ENGRAM_DEAD breaks all n-grams that cross it.
    std::array<int32_t, LLAMA_ENGRAM_NGRAM - 1> tail = {};

    void reset();
};

struct llama_engram_hasher {
    explicit llama_engram_hasher(llama_engram_layout layout);
    ~llama_engram_hasher();

    // Output is [token][layer][column]. Invalid input does not change history or output.
    void hash(
            llama_engram_history & history,
            const int32_t * tokens,
            const uint8_t * mask,
            size_t count,
            uint32_t * rows) const;

    const llama_engram_layout & layout() const;

    struct impl;
    std::unique_ptr<impl> pimpl;
};

// Decode 256 E4M3 values and eight E8M0 scales to BF16-rounded values in F32 storage.
void llama_engram_decode_row(const uint8_t row[LLAMA_ENGRAM_ROW_BYTES], float output[LLAMA_ENGRAM_DIM]);

struct llama_engram_table {
    // The table is never mapped or cached. Construction fails unless uncached reads are available.
    llama_engram_table(const std::string & fname, uint64_t offset, uint32_t rows);
    ~llama_engram_table();

    void read(const uint32_t * rows, size_t count, float * output);
    // Read 24 rows per token. Input uses row-ID stride; output is packed [token][column][dimension].
    void read_batch(const uint32_t * rows, size_t tokens, size_t stride, float * output);

    uint32_t n_rows() const;
    std::string describe() const;

    struct impl;
    std::unique_ptr<impl> pimpl;
};
