#include "llama-bounded-file.h"

#include "llama-impl.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

struct llama_bounded_file::buffer::impl {
    void * ptr   = nullptr;
    size_t bytes = 0;

    ~impl() {
        free(ptr);
    }
};

llama_bounded_file::buffer::buffer() : pimpl(std::make_unique<impl>()) {}
llama_bounded_file::buffer::~buffer() = default;
llama_bounded_file::buffer::buffer(buffer && other) noexcept = default;
llama_bounded_file::buffer & llama_bounded_file::buffer::operator=(buffer && other) noexcept = default;

bool llama_bounded_file::buffer::empty() const {
    return pimpl->ptr == nullptr;
}

struct llama_bounded_file::impl {
    std::string fname;
    int fd = -1;
    uint64_t file_size = 0;
    size_t block = 4096;
    bool direct = false;

    impl(const std::string & fname, const params & p) : fname(fname) {
#if defined(_WIN32)
        GGML_UNUSED(p);
        throw std::runtime_error("llama_bounded_file: not supported on Windows");
#else
        int direct_error = 0;
        if (p.direct_io) {
#if defined(O_DIRECT)
            fd = open(fname.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
            direct = fd >= 0;
            direct_error = direct ? 0 : errno;
#else
            fd = open(fname.c_str(), O_RDONLY | O_CLOEXEC);
            direct_error = fd >= 0 ? ENOTSUP : errno;
#if defined(F_NOCACHE)
            if (fd >= 0) {
                if (fcntl(fd, F_NOCACHE, 1) == 0) {
                    direct = true;
#if defined(F_RDAHEAD)
                    if (fcntl(fd, F_RDAHEAD, 0) != 0) {
                        direct = false;
                        direct_error = errno;
                    }
#endif
                } else {
                    direct_error = errno;
                }
            }
#endif
#endif
            if (!direct && p.direct_io_required) {
                const int saved = direct_error ? direct_error : ENOTSUP;
                if (fd >= 0) {
                    close(fd);
                    fd = -1;
                }
                throw std::runtime_error(format("llama_bounded_file: uncached open of %s failed: %s",
                                                fname.c_str(), strerror(saved)));
            }
            if (!direct) {
                LLAMA_LOG_WARN("%s: uncached open of %s failed (%s); falling back to buffered reads\n",
                               __func__, fname.c_str(), strerror(direct_error ? direct_error : ENOTSUP));
            }
            if (!direct && fd >= 0) {
                close(fd);
                fd = -1;
            }
        }
        if (fd < 0) {
            fd = open(fname.c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0) {
                throw std::runtime_error(format("llama_bounded_file: failed to open %s: %s",
                                                fname.c_str(), strerror(errno)));
            }
        }

        struct stat st = {};
        if (fstat(fd, &st) != 0) {
            const int saved = errno;
            close(fd);
            fd = -1;
            throw std::runtime_error(format("llama_bounded_file: fstat of %s failed: %s",
                                            fname.c_str(), strerror(saved)));
        }
        if (!S_ISREG(st.st_mode) || st.st_size < 0) {
            close(fd);
            fd = -1;
            throw std::runtime_error(format("llama_bounded_file: %s is not a regular file", fname.c_str()));
        }
        file_size = (uint64_t) st.st_size;
#endif
    }

    ~impl() {
#if !defined(_WIN32)
        if (fd >= 0) {
            close(fd);
        }
#endif
    }

    size_t read_size(uint64_t offset, size_t size) const {
        if (!direct || size == 0) {
            return size;
        }
        const size_t prefix = (size_t) (offset & (block - 1));
        if (size > SIZE_MAX - prefix || size + prefix > SIZE_MAX - (block - 1)) {
            throw std::overflow_error("llama_bounded_file: aligned read size overflow");
        }
        return (size + prefix + block - 1) & ~(block - 1);
    }

#if !defined(_WIN32)
    void pread_full(void * dst, size_t len, uint64_t offset, size_t need) const {
        size_t done = 0;
        while (done < len) {
            const size_t remaining = len - done;
            const ssize_t n = pread(fd, (uint8_t *) dst + done, remaining, (off_t) (offset + done));
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error(format("llama_bounded_file: pread of %s failed at %llu: %s",
                                                fname.c_str(), (unsigned long long) (offset + done), strerror(errno)));
            }
            if (n == 0) {
                break;
            }
            done += (size_t) n;
            if (direct && done >= need) {
                return;
            }
            if (direct && (size_t) n < remaining) {
                break;
            }
        }
        if (done < need) {
            throw std::runtime_error(format("llama_bounded_file: short read in %s: %zu of %zu bytes at %llu",
                                            fname.c_str(), done, need, (unsigned long long) offset));
        }
    }
#endif
};

llama_bounded_file::llama_bounded_file(const std::string & fname, const params & p)
    : pimpl(std::make_unique<impl>(fname, p)) {}

llama_bounded_file::~llama_bounded_file() = default;

llama_bounded_file::buffer llama_bounded_file::make_buffer(size_t size) const {
    buffer result;
    if (!pimpl->direct || size == 0) {
        return result;
    }
#if defined(_WIN32)
    GGML_UNUSED(size);
    throw std::runtime_error("llama_bounded_file: not supported on Windows");
#else
    const size_t bytes = pimpl->read_size(pimpl->block - 1, size);
    void * ptr = nullptr;
    const int err = posix_memalign(&ptr, pimpl->block, bytes);
    if (err != 0) {
        throw std::runtime_error(format("llama_bounded_file: posix_memalign failed: %s", strerror(err)));
    }
    result.pimpl->ptr = ptr;
    result.pimpl->bytes = bytes;
    return result;
#endif
}

void llama_bounded_file::read(uint64_t offset, void * dst, size_t size, buffer & scratch) const {
#if defined(_WIN32)
    GGML_UNUSED(offset);
    GGML_UNUSED(dst);
    GGML_UNUSED(size);
    GGML_UNUSED(scratch);
    throw std::runtime_error("llama_bounded_file: not supported on Windows");
#else
    if (size == 0) {
        return;
    }
    if (dst == nullptr || offset > pimpl->file_size || size > pimpl->file_size - offset ||
        offset > (uint64_t) INT64_MAX || size > (uint64_t) INT64_MAX - offset) {
        throw std::invalid_argument(format("llama_bounded_file: invalid read of %zu bytes at %llu in %s",
                                           size, (unsigned long long) offset, pimpl->fname.c_str()));
    }
    if (!pimpl->direct) {
        pimpl->pread_full(dst, size, offset, size);
        return;
    }

    const uint64_t aligned_offset = offset & ~(uint64_t) (pimpl->block - 1);
    const size_t prefix = (size_t) (offset - aligned_offset);
    const size_t aligned_size = pimpl->read_size(offset, size);
    if (aligned_size > (uint64_t) INT64_MAX - aligned_offset) {
        throw std::overflow_error("llama_bounded_file: aligned read offset overflow");
    }
    if (scratch.pimpl->ptr == nullptr || scratch.pimpl->bytes < aligned_size) {
        throw std::invalid_argument("llama_bounded_file: aligned scratch buffer is too small");
    }
    pimpl->pread_full(scratch.pimpl->ptr, aligned_size, aligned_offset, prefix + size);
    memcpy(dst, (uint8_t *) scratch.pimpl->ptr + prefix, size);
#endif
}

uint64_t llama_bounded_file::size() const {
    return pimpl->file_size;
}

size_t llama_bounded_file::alignment() const {
    return pimpl->block;
}

size_t llama_bounded_file::read_size(uint64_t offset, size_t size) const {
    return pimpl->read_size(offset, size);
}

bool llama_bounded_file::direct_io() const {
    return pimpl->direct;
}

const std::string & llama_bounded_file::name() const {
    return pimpl->fname;
}
