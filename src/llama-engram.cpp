#include "llama-engram.h"

#include "llama-bounded-file.h"
#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>

static void llama_engram_validate_layout(const llama_engram_layout & layout) {
    if (layout.encoding != "e4m3_e8m0_32_row264") {
        throw std::invalid_argument("llama_engram: unsupported row encoding");
    }
    if (layout.layer_ids[0] == layout.layer_ids[1]) {
        throw std::invalid_argument("llama_engram: layer IDs must be distinct");
    }
    if (layout.token_map.empty() || layout.token_map.size() > (size_t) INT32_MAX) {
        throw std::invalid_argument("llama_engram: invalid token map size");
    }
    if (layout.compressed_vocab_size == 0 || layout.compressed_vocab_size > (uint32_t) INT32_MAX ||
        layout.pad_id >= layout.compressed_vocab_size) {
        throw std::invalid_argument("llama_engram: invalid compressed vocabulary");
    }
    for (uint32_t token : layout.token_map) {
        if (token >= layout.compressed_vocab_size) {
            throw std::invalid_argument("llama_engram: token map entry is out of range");
        }
    }

    for (size_t layer = 0; layer < LLAMA_ENGRAM_LAYERS; ++layer) {
        for (size_t i = 0; i < LLAMA_ENGRAM_NGRAM; ++i) {
            const uint64_t multiplier = layout.multipliers[layer][i];
            if ((multiplier & 1) == 0 ||
                multiplier > (uint64_t) INT64_MAX / layout.compressed_vocab_size) {
                throw std::invalid_argument("llama_engram: invalid hash multiplier");
            }
        }

        uint64_t row_count = 0;
        for (size_t col = 0; col < LLAMA_ENGRAM_COLS; ++col) {
            const uint32_t prime = layout.primes[layer][col];
            if (prime < 2) {
                throw std::invalid_argument("llama_engram: invalid hash prime");
            }
            row_count += prime;
        }
        if (row_count > UINT32_MAX || row_count != layout.rows[layer]) {
            throw std::invalid_argument("llama_engram: row extent does not match hash buckets");
        }
    }
}

void llama_engram_history::reset() {
    tail.fill(LLAMA_ENGRAM_DEAD);
}

struct llama_engram_hasher::impl {
    llama_engram_layout layout;

    explicit impl(llama_engram_layout layout) : layout(std::move(layout)) {
        llama_engram_validate_layout(this->layout);
    }

    void validate_history(const llama_engram_history & history) const {
        for (int32_t token : history.tail) {
            if (token < LLAMA_ENGRAM_DEAD ||
                (token >= 0 && (uint32_t) token >= layout.compressed_vocab_size)) {
                throw std::invalid_argument("llama_engram: invalid token history");
            }
        }
    }

    void hash(
            llama_engram_history & history,
            const int32_t * tokens,
            const uint8_t * mask,
            size_t count,
            uint32_t * rows) const {
        if (count > SIZE_MAX / (LLAMA_ENGRAM_LAYERS * LLAMA_ENGRAM_COLS * sizeof(*rows)) ||
            (count != 0 && (tokens == nullptr || rows == nullptr))) {
            throw std::invalid_argument("llama_engram: invalid hash buffers");
        }
        validate_history(history);
        for (size_t i = 0; i < count; ++i) {
            if (tokens[i] < 0 || (size_t) tokens[i] >= layout.token_map.size()) {
                throw std::invalid_argument("llama_engram: token ID is out of range");
            }
        }

        llama_engram_history next = history;
        uint32_t * output = rows;
        for (size_t pos = 0; pos < count; ++pos) {
            const int32_t current = mask != nullptr && mask[pos] == 0 ?
                    LLAMA_ENGRAM_DEAD : (int32_t) layout.token_map[tokens[pos]];
            uint32_t ids[LLAMA_ENGRAM_NGRAM];
            bool blocked = false;
            for (size_t depth = 0; depth < LLAMA_ENGRAM_NGRAM; ++depth) {
                const int32_t id = depth == 0 ? current : next.tail[depth - 1];
                blocked = blocked || id == LLAMA_ENGRAM_DEAD;
                ids[depth] = blocked ? layout.pad_id : (uint32_t) id;
            }

            for (size_t layer = 0; layer < LLAMA_ENGRAM_LAYERS; ++layer) {
                uint64_t hash = (uint64_t) ids[0] * layout.multipliers[layer][0];
                uint64_t offset = 0;
                for (size_t depth = 1; depth < LLAMA_ENGRAM_NGRAM; ++depth) {
                    hash ^= (uint64_t) ids[depth] * layout.multipliers[layer][depth];
                    for (size_t head = 0; head < LLAMA_ENGRAM_HEADS; ++head) {
                        const size_t col = (depth - 1) * LLAMA_ENGRAM_HEADS + head;
                        const uint32_t prime = layout.primes[layer][col];
                        *output++ = (uint32_t) (hash % prime + offset);
                        offset += prime;
                    }
                }
            }

            for (size_t i = next.tail.size() - 1; i > 0; --i) {
                next.tail[i] = next.tail[i - 1];
            }
            next.tail[0] = current;
        }
        history = next;
    }
};

llama_engram_hasher::llama_engram_hasher(llama_engram_layout layout)
    : pimpl(std::make_unique<impl>(std::move(layout))) {}

llama_engram_hasher::~llama_engram_hasher() = default;

void llama_engram_hasher::hash(
        llama_engram_history & history,
        const int32_t * tokens,
        const uint8_t * mask,
        size_t count,
        uint32_t * rows) const {
    pimpl->hash(history, tokens, mask, count, rows);
}

const llama_engram_layout & llama_engram_hasher::layout() const {
    return pimpl->layout;
}

static float llama_engram_e4m3(uint8_t code) {
    const int exponent = (code >> 3) & 15;
    const int mantissa = code & 7;
    const float value = exponent != 0 ?
            std::ldexp((float) (8 + mantissa), exponent - 10) :
            std::ldexp((float) mantissa, -9);
    return (code & 128) != 0 ? -value : value;
}

void llama_engram_decode_row(const uint8_t row[LLAMA_ENGRAM_ROW_BYTES], float output[LLAMA_ENGRAM_DIM]) {
    if (row == nullptr || output == nullptr) {
        throw std::invalid_argument("llama_engram: invalid row decode buffers");
    }

    float decoded[LLAMA_ENGRAM_DIM];
    for (size_t i = 0; i < LLAMA_ENGRAM_DIM; ++i) {
        const uint8_t code = row[i];
        const uint8_t scale = row[LLAMA_ENGRAM_DIM + i / 32];
        if ((code & 127) == 127) {
            throw std::domain_error("llama_engram: E4M3 NaN encoding");
        }
        if (scale == 255) {
            throw std::domain_error("llama_engram: E8M0 scale 255");
        }

        float value = std::ldexp(llama_engram_e4m3(code), (int) scale - 127);
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
        memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value)) {
            throw std::domain_error("llama_engram: decoded value is not finite");
        }
        decoded[i] = value;
    }
    memcpy(output, decoded, sizeof(decoded));
}

struct llama_engram_table::impl {
    struct request {
        uint32_t row;
        uint32_t output;
    };

    static constexpr size_t BATCH_TOKENS = 2048;

    llama_bounded_file file;
    llama_bounded_file::buffer scratch;
    uint64_t offset;
    uint32_t rows;
    std::mutex mutex;
    std::vector<request> requests;

    impl(const std::string & fname, uint64_t offset, uint32_t rows)
        : file(fname, { true, true }),
          scratch(file.make_buffer(LLAMA_ENGRAM_ROW_BYTES)),
          offset(offset),
          rows(rows) {
        const uint64_t bytes = (uint64_t) rows * LLAMA_ENGRAM_ROW_BYTES;
        if (rows == 0 || offset > (uint64_t) INT64_MAX || bytes > (uint64_t) INT64_MAX - offset ||
            offset > file.size() || bytes > file.size() - offset) {
            throw std::invalid_argument("llama_engram: invalid table extent");
        }
        requests.reserve(BATCH_TOKENS * LLAMA_ENGRAM_COLS);
    }

    void validate_rows(const uint32_t * row_ids, size_t count) const {
        if (count > SIZE_MAX / (LLAMA_ENGRAM_DIM * sizeof(float)) ||
            (count != 0 && row_ids == nullptr)) {
            throw std::invalid_argument("llama_engram: invalid row list");
        }
        for (size_t i = 0; i < count; ++i) {
            if (row_ids[i] >= rows) {
                throw std::invalid_argument("llama_engram: row ID is out of range");
            }
        }
    }

    void read_one(uint32_t row, float * output) {
        uint8_t raw[LLAMA_ENGRAM_ROW_BYTES];
        file.read(offset + (uint64_t) row * LLAMA_ENGRAM_ROW_BYTES, raw, sizeof(raw), scratch);
        llama_engram_decode_row(raw, output);
    }

    void read(const uint32_t * row_ids, size_t count, float * output) {
        validate_rows(row_ids, count);
        if (count != 0 && output == nullptr) {
            throw std::invalid_argument("llama_engram: invalid row output");
        }

        std::lock_guard<std::mutex> lock(mutex);
        for (size_t i = 0; i < count; ++i) {
            read_one(row_ids[i], output + i * LLAMA_ENGRAM_DIM);
        }
    }

    void read_batch(const uint32_t * row_ids, size_t tokens, size_t stride, float * output) {
        if (tokens > SIZE_MAX / (LLAMA_ENGRAM_COLS * LLAMA_ENGRAM_DIM * sizeof(float)) ||
            (tokens != 0 && (row_ids == nullptr || output == nullptr || stride < LLAMA_ENGRAM_COLS)) ||
            (tokens != 0 && tokens - 1 > (SIZE_MAX / sizeof(*row_ids) - LLAMA_ENGRAM_COLS) / stride)) {
            throw std::invalid_argument("llama_engram: invalid batch buffers");
        }
        for (size_t token = 0; token < tokens; ++token) {
            validate_rows(row_ids + token * stride, LLAMA_ENGRAM_COLS);
        }
        if (tokens == 0) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex);
        for (size_t start = 0; start < tokens; start += BATCH_TOKENS) {
            const size_t count_tokens = std::min(tokens - start, BATCH_TOKENS);
            const size_t count_rows = count_tokens * LLAMA_ENGRAM_COLS;
            requests.resize(count_rows);
            for (size_t i = 0; i < count_rows; ++i) {
                requests[i].row = row_ids[(start + i / LLAMA_ENGRAM_COLS) * stride + i % LLAMA_ENGRAM_COLS];
                requests[i].output = (uint32_t) i;
            }
            std::sort(requests.begin(), requests.end(), [](const request & a, const request & b) {
                return a.row < b.row;
            });

            float * chunk = output + start * LLAMA_ENGRAM_COLS * LLAMA_ENGRAM_DIM;
            const float * previous = nullptr;
            for (size_t i = 0; i < count_rows; ++i) {
                float * dst = chunk + (size_t) requests[i].output * LLAMA_ENGRAM_DIM;
                if (i != 0 && requests[i].row == requests[i - 1].row) {
                    memcpy(dst, previous, LLAMA_ENGRAM_DIM * sizeof(*dst));
                } else {
                    read_one(requests[i].row, dst);
                    previous = dst;
                }
            }
        }
    }
};

llama_engram_table::llama_engram_table(const std::string & fname, uint64_t offset, uint32_t rows)
    : pimpl(std::make_unique<impl>(fname, offset, rows)) {}

llama_engram_table::~llama_engram_table() = default;

void llama_engram_table::read(const uint32_t * rows, size_t count, float * output) {
    pimpl->read(rows, count, output);
}

void llama_engram_table::read_batch(const uint32_t * rows, size_t tokens, size_t stride, float * output) {
    pimpl->read_batch(rows, tokens, stride, output);
}

uint32_t llama_engram_table::n_rows() const {
    return pimpl->rows;
}

std::string llama_engram_table::describe() const {
    return format("%s @ %llu: %u rows x %u bytes, uncached aligned reads",
                  pimpl->file.name().c_str(), (unsigned long long) pimpl->offset,
                  pimpl->rows, LLAMA_ENGRAM_ROW_BYTES);
}
