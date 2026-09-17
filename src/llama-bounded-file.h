#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct llama_bounded_file {
    struct params {
        bool direct_io          = true;
        bool direct_io_required = false; // fail instead of using the page cache
    };

    struct buffer {
        buffer();
        ~buffer();
        buffer(buffer && other) noexcept;
        buffer & operator=(buffer && other) noexcept;

        buffer(const buffer &) = delete;
        buffer & operator=(const buffer &) = delete;

        bool empty() const;

        struct impl;
        std::unique_ptr<impl> pimpl;
    };

    llama_bounded_file(const std::string & fname, const params & p);
    ~llama_bounded_file();

    buffer make_buffer(size_t size) const;
    // Read only the requested extent. Direct I/O may read its containing aligned blocks into scratch.
    void read(uint64_t offset, void * dst, size_t size, buffer & scratch) const;

    uint64_t size() const;
    size_t alignment() const;
    size_t read_size(uint64_t offset, size_t size) const;
    bool direct_io() const;
    const std::string & name() const;

    struct impl;
    std::unique_ptr<impl> pimpl;
};
