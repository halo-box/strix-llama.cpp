#include "llama-expert-store.h"

#include "llama-impl.h"
#include "llama-mmap.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>

#if !defined(_WIN32)
#include <unistd.h>
#endif

// The positional I/O and reservation model is adapted from ggml-org/llama.cpp#25294.
// Leases add the in-flight publication safety described in ggml-org/llama.cpp#27861.

namespace {

bool checked_add_u64(uint64_t a, uint64_t b, uint64_t * result) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }
    *result = a + b;
    return true;
}

bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t * result) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    *result = a * b;
    return true;
}

bool is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

struct expert_key {
    int32_t                 layer;
    llama_expert_projection projection;
    int32_t                 expert_id;

    bool operator<(const expert_key & other) const {
        if (layer != other.layer) {
            return layer < other.layer;
        }
        if (projection != other.projection) {
            return projection < other.projection;
        }
        return expert_id < other.expert_id;
    }

    bool operator==(const expert_key & other) const {
        return layer == other.layer && projection == other.projection && expert_id == other.expert_id;
    }
};

struct tensor_key {
    int32_t                 layer;
    llama_expert_projection projection;

    bool operator<(const tensor_key & other) const {
        if (layer != other.layer) {
            return layer < other.layer;
        }
        return projection < other.projection;
    }
};

struct aligned_buffer {
    uint8_t * data = nullptr;
    size_t size = 0;

    aligned_buffer() = default;

    aligned_buffer(size_t size, size_t alignment) {
        reset(size, alignment);
    }

    aligned_buffer(aligned_buffer && other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }

    aligned_buffer & operator=(aligned_buffer && other) noexcept {
        if (this != &other) {
            clear();
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }

    ~aligned_buffer() {
        clear();
    }

    aligned_buffer(const aligned_buffer &) = delete;
    aligned_buffer & operator=(const aligned_buffer &) = delete;

    void reset(size_t new_size, size_t alignment) {
        clear();
        if (new_size == 0) {
            return;
        }
        alignment = std::max(alignment, alignof(void *));
#if defined(_WIN32)
        data = static_cast<uint8_t *>(_aligned_malloc(new_size, alignment));
        if (data == nullptr) {
            throw std::bad_alloc();
        }
#else
        void * ptr = nullptr;
        if (posix_memalign(&ptr, alignment, new_size) != 0) {
            throw std::bad_alloc();
        }
        data = static_cast<uint8_t *>(ptr);
#endif
        size = new_size;
    }

    void clear() {
#if defined(_WIN32)
        _aligned_free(data);
#else
        free(data);
#endif
        data = nullptr;
        size = 0;
    }
};

struct expert_file {
    std::string fname;
    uint64_t size = 0;
    bool direct = false;
    std::unique_ptr<llama_file> file;

    expert_file(const std::string & fname, bool direct_io, bool allow_buffered_io) : fname(fname) {
        reopen(direct_io);
        if (direct_io && !direct && !allow_buffered_io) {
            throw std::runtime_error(format("llama_expert_store: direct I/O is required but unavailable for %s", fname.c_str()));
        }
        if (direct_io && !direct) {
            LLAMA_LOG_WARN("%s: direct I/O is unavailable for %s; using explicitly allowed buffered reads\n",
                    __func__, fname.c_str());
        }
    }

    expert_file(const expert_file &) = delete;
    expert_file & operator=(const expert_file &) = delete;

    void reopen(bool direct_io) {
        file = std::make_unique<llama_file>(fname.c_str(), "rb", direct_io);
        size = file->size();
        direct = direct_io && file->has_direct_io();
    }

    size_t pread_at_least(void * dst, size_t len, uint64_t offset, size_t need) const {
        if (need > len) {
            throw std::runtime_error("llama_expert_store: invalid read requirement");
        }
        size_t total = 0;
        while (total < need) {
#if defined(_WIN32)
            file->seek(offset + total, SEEK_SET);
            file->read_raw(static_cast<uint8_t *>(dst) + total, len - total);
            const size_t n = len - total;
#else
            if (offset + total > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
                throw std::runtime_error("llama_expert_store: file offset exceeds off_t");
            }
            const ssize_t result = pread(file->file_id(), static_cast<uint8_t *>(dst) + total, len - total,
                    static_cast<off_t>(offset + total));
            if (result < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error(format("llama_expert_store: pread failed for %s at %llu: %s",
                        fname.c_str(), (unsigned long long) (offset + total), strerror(errno)));
            }
            const size_t n = static_cast<size_t>(result);
#endif
            if (n == 0) {
                break;
            }
            total += n;
        }
        if (total < need) {
            throw std::runtime_error(format("llama_expert_store: short read for %s: %zu bytes, need %zu at %llu",
                    fname.c_str(), total, need, (unsigned long long) offset));
        }
        return total;
    }
};

}

llama_expert_store_aligned_read llama_expert_store_align_read(
        uint64_t offset, size_t size, size_t alignment, uint64_t file_size) {
    if (!is_power_of_two(alignment)) {
        throw std::runtime_error("llama_expert_store: I/O alignment must be a power of two");
    }

    uint64_t end;
    if (!checked_add_u64(offset, size, &end) || end > file_size) {
        throw std::runtime_error("llama_expert_store: read is outside the source file");
    }

    const uint64_t aligned_offset = offset & ~static_cast<uint64_t>(alignment - 1);
    const size_t prefix = static_cast<size_t>(offset - aligned_offset);
    uint64_t needed;
    if (!checked_add_u64(prefix, size, &needed)) {
        throw std::runtime_error("llama_expert_store: aligned read size overflow");
    }
    uint64_t rounded;
    if (!checked_add_u64(needed, alignment - 1, &rounded)) {
        throw std::runtime_error("llama_expert_store: aligned read size overflow");
    }
    rounded &= ~static_cast<uint64_t>(alignment - 1);
    llama_expert_store_aligned_read result;
    result.offset = aligned_offset;
    if (rounded > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("llama_expert_store: aligned read exceeds addressable memory");
    }
    result.size = static_cast<size_t>(rounded);
    result.prefix = prefix;
    return result;
}

void llama_expert_store_validate_tensor(const llama_expert_store_tensor & tensor) {
    if (tensor.name.empty() || tensor.fname.empty()) {
        throw std::runtime_error("llama_expert_store: tensor name and source file are required");
    }
    if (tensor.layer < 0) {
        throw std::runtime_error(format("llama_expert_store: tensor %s has an invalid layer", tensor.name.c_str()));
    }
    if (tensor.projection < LLAMA_EXPERT_PROJECTION_GATE || tensor.projection > LLAMA_EXPERT_PROJECTION_DOWN) {
        throw std::runtime_error(format("llama_expert_store: tensor %s has an invalid projection", tensor.name.c_str()));
    }
    const ggml_type expected_type = tensor.projection == LLAMA_EXPERT_PROJECTION_DOWN ? GGML_TYPE_Q2_K : GGML_TYPE_IQ2_XXS;
    if (tensor.type != expected_type) {
        throw std::runtime_error(format("llama_expert_store: tensor %s must be %s, got %s",
                tensor.name.c_str(), ggml_type_name(expected_type), ggml_type_name(tensor.type)));
    }
    if (tensor.ne[0] <= 0 || tensor.ne[1] <= 0 || tensor.ne[2] <= 0) {
        throw std::runtime_error(format("llama_expert_store: tensor %s has invalid dimensions", tensor.name.c_str()));
    }
    if (tensor.ne[2] > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(format("llama_expert_store: tensor %s has too many experts", tensor.name.c_str()));
    }
    if (tensor.ne[0] % ggml_blck_size(tensor.type) != 0) {
        throw std::runtime_error(format("llama_expert_store: tensor %s rows are not whole quantization blocks", tensor.name.c_str()));
    }

    const size_t row_size = ggml_row_size(tensor.type, tensor.ne[0]);
    uint64_t plane_size;
    uint64_t tensor_size;
    if (!checked_mul_u64(row_size, static_cast<uint64_t>(tensor.ne[1]), &plane_size) ||
            !checked_mul_u64(plane_size, static_cast<uint64_t>(tensor.ne[2]), &tensor_size)) {
        throw std::runtime_error(format("llama_expert_store: tensor %s size overflows", tensor.name.c_str()));
    }
    if (tensor.nb[0] != ggml_type_size(tensor.type) || tensor.nb[1] != row_size || tensor.nb[2] != plane_size) {
        throw std::runtime_error(format("llama_expert_store: tensor %s is not a contiguous merged-expert tensor", tensor.name.c_str()));
    }
    uint64_t tensor_end;
    if (!checked_add_u64(tensor.file_offset, tensor_size, &tensor_end) || tensor_end > tensor.file_size) {
        throw std::runtime_error(format("llama_expert_store: tensor %s is outside the source file", tensor.name.c_str()));
    }
}

struct llama_expert_store::impl {
    struct slot {
        bool occupied = false;
        expert_key key = {};
        const llama_expert_store_tensor * tensor = nullptr;
        aligned_buffer bytes;
        uint64_t last_use = 0;
        uint32_t pins = 0;
    };

    llama_expert_store_params params;
    std::map<tensor_key, llama_expert_store_tensor> tensors;
    std::map<std::string, std::unique_ptr<expert_file>> files;
    std::vector<slot> slots;
    size_t bytes_resident = 0;
    uint64_t use_clock = 0;
    llama_expert_store_stats counters;
    mutable std::mutex mutex;

    impl(std::vector<llama_expert_store_tensor> tensors, const llama_expert_store_params & params) : params(params) {
        if (params.cache_bytes == 0 || params.cache_slots == 0) {
            throw std::runtime_error("llama_expert_store: cache byte and slot budgets must be non-zero");
        }
        if (params.cache_slots > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("llama_expert_store: cache slot budget exceeds the slot ID range");
        }
        if (!is_power_of_two(params.io_alignment)) {
            throw std::runtime_error("llama_expert_store: I/O alignment must be a power of two");
        }

        for (auto & tensor : tensors) {
            llama_expert_store_validate_tensor(tensor);
            const tensor_key key = { tensor.layer, tensor.projection };
            if (this->tensors.count(key) != 0) {
                throw std::runtime_error(format("llama_expert_store: duplicate tensor for layer %d projection %d",
                        tensor.layer, static_cast<int>(tensor.projection)));
            }
            if (tensor.nb[2] > params.cache_bytes) {
                throw std::runtime_error(format("llama_expert_store: tensor %s expert plane exceeds the cache byte budget",
                        tensor.name.c_str()));
            }
            auto file_it = files.find(tensor.fname);
            if (file_it == files.end()) {
                file_it = files.emplace(tensor.fname,
                        std::make_unique<expert_file>(tensor.fname, params.direct_io, params.allow_buffered_io)).first;
            }
            if (file_it->second->size != tensor.file_size) {
                throw std::runtime_error(format("llama_expert_store: source file size changed for %s", tensor.fname.c_str()));
            }
            this->tensors.emplace(key, std::move(tensor));
        }

        if (this->tensors.empty()) {
            throw std::runtime_error("llama_expert_store: no tensors registered");
        }
        for (auto it = this->tensors.begin(); it != this->tensors.end();) {
            const int32_t layer = it->first.layer;
            const auto gate = this->tensors.find({ layer, LLAMA_EXPERT_PROJECTION_GATE });
            const auto up   = this->tensors.find({ layer, LLAMA_EXPERT_PROJECTION_UP });
            const auto down = this->tensors.find({ layer, LLAMA_EXPERT_PROJECTION_DOWN });
            if (gate == this->tensors.end() || up == this->tensors.end() || down == this->tensors.end()) {
                throw std::runtime_error(format("llama_expert_store: layer %d must register gate, up, and down tensors", layer));
            }
            if (gate->second.ne[0] != up->second.ne[0] ||
                    gate->second.ne[1] != up->second.ne[1] ||
                    gate->second.ne[2] != up->second.ne[2] ||
                    down->second.ne[0] != gate->second.ne[1] ||
                    down->second.ne[1] != gate->second.ne[0] ||
                    down->second.ne[2] != gate->second.ne[2]) {
                throw std::runtime_error(format("llama_expert_store: layer %d expert tensor dimensions do not match", layer));
            }
            it = this->tensors.upper_bound({ layer, LLAMA_EXPERT_PROJECTION_DOWN });
        }
        slots.resize(params.cache_slots);
    }

    const llama_expert_store_tensor & get_tensor(const expert_key & key) const {
        const auto it = tensors.find({ key.layer, key.projection });
        if (it == tensors.end()) {
            throw std::runtime_error(format("llama_expert_store: no tensor for layer %d projection %d",
                    key.layer, static_cast<int>(key.projection)));
        }
        if (key.expert_id < 0 || key.expert_id >= it->second.ne[2]) {
            throw std::runtime_error(format("llama_expert_store: expert ID %d is outside [0, %lld)",
                    key.expert_id, (long long) it->second.ne[2]));
        }
        return it->second;
    }

    aligned_buffer read_expert(const llama_expert_store_tensor & tensor, int32_t expert_id, uint64_t * bytes_read) const {
        uint64_t expert_delta;
        uint64_t expert_offset;
        if (!checked_mul_u64(static_cast<uint64_t>(expert_id), tensor.nb[2], &expert_delta) ||
                !checked_add_u64(tensor.file_offset, expert_delta, &expert_offset)) {
            throw std::runtime_error(format("llama_expert_store: expert offset overflow for %s", tensor.name.c_str()));
        }

        aligned_buffer payload(tensor.nb[2], params.io_alignment);
        auto & file = *files.at(tensor.fname);
        if (file.direct) {
            const llama_expert_store_aligned_read read =
                    llama_expert_store_align_read(expert_offset, tensor.nb[2], params.io_alignment, tensor.file_size);
            aligned_buffer bounce(read.size, params.io_alignment);
            try {
                *bytes_read += file.pread_at_least(bounce.data, read.size, read.offset, read.prefix + tensor.nb[2]);
                memcpy(payload.data, bounce.data + read.prefix, tensor.nb[2]);
            } catch (const std::runtime_error & e) {
                if (!params.allow_buffered_io) {
                    throw std::runtime_error(format("llama_expert_store: direct I/O failed for %s and buffered fallback is disabled: %s",
                            tensor.fname.c_str(), e.what()));
                }
                LLAMA_LOG_WARN("%s: direct read failed for %s; retrying with buffered I/O: %s\n",
                        __func__, tensor.fname.c_str(), e.what());
                file.reopen(false);
                *bytes_read += file.pread_at_least(payload.data, tensor.nb[2], expert_offset, tensor.nb[2]);
            }
        } else {
            *bytes_read += file.pread_at_least(payload.data, tensor.nb[2], expert_offset, tensor.nb[2]);
        }

        if (!ggml_validate_row_data(tensor.type, payload.data, tensor.nb[2])) {
            throw std::runtime_error(format("llama_expert_store: tensor %s expert %d has invalid payload",
                    tensor.name.c_str(), expert_id));
        }
        return payload;
    }

    void unpin(const std::vector<uint32_t> & slot_ids) {
        std::lock_guard<std::mutex> lock(mutex);
        for (uint32_t slot_id : slot_ids) {
            if (slot_id >= slots.size() || slots[slot_id].pins == 0) {
                GGML_ABORT("llama_expert_store: invalid lease slot");
            }
            slots[slot_id].pins--;
        }
    }

    std::vector<llama_expert_store::payload> get_payloads(const std::vector<uint32_t> & slot_ids) const {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<llama_expert_store::payload> result;
        result.reserve(slot_ids.size());
        for (uint32_t slot_id : slot_ids) {
            const slot & entry = slots.at(slot_id);
            GGML_ASSERT(entry.occupied && entry.pins > 0);
            result.push_back({
                entry.key.layer,
                entry.key.projection,
                entry.key.expert_id,
                slot_id,
                entry.tensor->type,
                entry.bytes.data,
                entry.bytes.size,
            });
        }
        return result;
    }
};

struct llama_expert_store::lease::impl {
    std::shared_ptr<llama_expert_store::impl> store;
    std::vector<uint32_t> pinned_slots;
    std::vector<std::vector<uint32_t>> remapped_slots;

    ~impl() {
        if (store) {
            store->unpin(pinned_slots);
        }
    }
};

llama_expert_store::lease::lease() = default;
llama_expert_store::lease::lease(lease && other) noexcept = default;
llama_expert_store::lease & llama_expert_store::lease::operator=(lease && other) noexcept = default;
llama_expert_store::lease::~lease() = default;

const std::vector<std::vector<uint32_t>> & llama_expert_store::lease::slot_ids() const {
    static const std::vector<std::vector<uint32_t>> empty;
    return pimpl ? pimpl->remapped_slots : empty;
}

std::vector<llama_expert_store::payload> llama_expert_store::lease::payloads() const {
    return pimpl ? pimpl->store->get_payloads(pimpl->pinned_slots) : std::vector<payload>();
}

llama_expert_store::llama_expert_store(
        std::vector<llama_expert_store_tensor> tensors, const llama_expert_store_params & params)
    : pimpl(std::make_shared<impl>(std::move(tensors), params)) {
}

llama_expert_store::~llama_expert_store() = default;

llama_expert_store::lease llama_expert_store::acquire(const std::vector<llama_expert_store_request> & requests) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    std::vector<std::vector<expert_key>> request_keys;
    std::vector<expert_key> unique_keys;
    request_keys.reserve(requests.size());
    for (const auto & request : requests) {
        std::vector<expert_key> keys;
        keys.reserve(request.expert_ids.size());
        for (int32_t expert_id : request.expert_ids) {
            const expert_key key = { request.layer, request.projection, expert_id };
            pimpl->get_tensor(key);
            keys.push_back(key);
            unique_keys.push_back(key);
        }
        request_keys.push_back(std::move(keys));
    }
    std::sort(unique_keys.begin(), unique_keys.end());
    unique_keys.erase(std::unique(unique_keys.begin(), unique_keys.end()), unique_keys.end());

    std::map<expert_key, uint32_t> resident;
    for (uint32_t i = 0; i < pimpl->slots.size(); ++i) {
        if (pimpl->slots[i].occupied) {
            resident.emplace(pimpl->slots[i].key, i);
        }
    }

    std::vector<expert_key> misses;
    std::vector<uint32_t> hit_slots;
    size_t miss_bytes = 0;
    for (const expert_key & key : unique_keys) {
        const auto hit = resident.find(key);
        if (hit != resident.end()) {
            hit_slots.push_back(hit->second);
            continue;
        }
        const auto & tensor = pimpl->get_tensor(key);
        if (miss_bytes > pimpl->params.cache_bytes || tensor.nb[2] > pimpl->params.cache_bytes - miss_bytes) {
            throw std::runtime_error("llama_expert_store: requested expert union exceeds the cache byte budget");
        }
        miss_bytes += tensor.nb[2];
        misses.push_back(key);
    }

    std::vector<uint32_t> empty_slots;
    std::vector<uint32_t> candidates;
    std::sort(hit_slots.begin(), hit_slots.end());
    for (uint32_t i = 0; i < pimpl->slots.size(); ++i) {
        const auto & entry = pimpl->slots[i];
        if (!entry.occupied) {
            empty_slots.push_back(i);
        } else if (entry.pins == 0 && !std::binary_search(hit_slots.begin(), hit_slots.end(), i)) {
            candidates.push_back(i);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [&](uint32_t a, uint32_t b) {
        const auto & lhs = pimpl->slots[a];
        const auto & rhs = pimpl->slots[b];
        if (lhs.last_use != rhs.last_use) {
            return lhs.last_use < rhs.last_use;
        }
        return a < b;
    });

    const size_t min_victims = misses.size() > empty_slots.size() ? misses.size() - empty_slots.size() : 0;
    std::vector<uint32_t> victims;
    if (miss_bytes > std::numeric_limits<size_t>::max() - pimpl->bytes_resident) {
        throw std::runtime_error("llama_expert_store: cache byte accounting overflow");
    }
    size_t bytes_after = pimpl->bytes_resident + miss_bytes;
    for (uint32_t candidate : candidates) {
        if (victims.size() >= min_victims && bytes_after <= pimpl->params.cache_bytes) {
            break;
        }
        victims.push_back(candidate);
        bytes_after -= pimpl->slots[candidate].bytes.size;
    }
    if (victims.size() < min_victims || bytes_after > pimpl->params.cache_bytes) {
        throw std::runtime_error("llama_expert_store: cache capacity is pinned or too small for the requested expert union");
    }

    std::vector<uint32_t> target_slots = empty_slots;
    target_slots.insert(target_slots.end(), victims.begin(), victims.end());
    std::sort(target_slots.begin(), target_slots.end());

    uint64_t bytes_read = 0;
    std::vector<aligned_buffer> staged;
    staged.reserve(misses.size());
    for (const expert_key & key : misses) {
        const auto & tensor = pimpl->get_tensor(key);
        staged.push_back(pimpl->read_expert(tensor, key.expert_id, &bytes_read));
    }

    for (uint32_t victim : victims) {
        auto & entry = pimpl->slots[victim];
        resident.erase(entry.key);
        pimpl->bytes_resident -= entry.bytes.size;
        entry.bytes.clear();
        entry.occupied = false;
        entry.tensor = nullptr;
        entry.last_use = 0;
        pimpl->counters.evictions++;
    }

    for (size_t i = 0; i < misses.size(); ++i) {
        const expert_key & key = misses[i];
        const auto & tensor = pimpl->get_tensor(key);
        auto & entry = pimpl->slots[target_slots[i]];
        entry.bytes = std::move(staged[i]);
        entry.occupied = true;
        entry.key = key;
        entry.tensor = &tensor;
        entry.pins = 0;
        pimpl->bytes_resident += entry.bytes.size;
        resident[entry.key] = target_slots[i];
    }

    auto lease_impl = std::make_unique<lease::impl>();
    lease_impl->remapped_slots.reserve(request_keys.size());
    for (const auto & keys : request_keys) {
        std::vector<uint32_t> remapped;
        remapped.reserve(keys.size());
        for (const expert_key & key : keys) {
            remapped.push_back(resident.at(key));
        }
        lease_impl->remapped_slots.push_back(std::move(remapped));
    }
    lease_impl->pinned_slots.reserve(unique_keys.size());
    for (const expert_key & key : unique_keys) {
        lease_impl->pinned_slots.push_back(resident.at(key));
    }
    for (uint32_t slot_id : lease_impl->pinned_slots) {
        auto & entry = pimpl->slots[slot_id];
        entry.last_use = ++pimpl->use_clock;
        entry.pins++;
    }
    lease_impl->store = pimpl;

    pimpl->counters.hits += unique_keys.size() - misses.size();
    pimpl->counters.misses += misses.size();
    pimpl->counters.bytes_read += bytes_read;

    lease result;
    result.pimpl = std::move(lease_impl);
    return result;
}

llama_expert_store_stats llama_expert_store::stats() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->counters;
}

size_t llama_expert_store::resident_bytes() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->bytes_resident;
}

size_t llama_expert_store::resident_entries() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    size_t result = 0;
    for (const auto & slot : pimpl->slots) {
        result += slot.occupied ? 1 : 0;
    }
    return result;
}

bool llama_expert_store::direct_io_active() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    for (const auto & item : pimpl->files) {
        if (!item.second->direct) {
            return false;
        }
    }
    return true;
}
