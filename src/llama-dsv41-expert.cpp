#include "llama-dsv41-expert.h"

#include "llama-dsv41.h"
#include "llama-impl.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

// The remap node follows ggml-org/llama.cpp#25294 commit 4260e4608.
// Lease publication and release follow ggml-org/llama.cpp#27861 commit bccbacdb8.

namespace {

size_t checked_add(size_t a, size_t b, const char * message) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::runtime_error(message);
    }
    return a + b;
}

size_t checked_mul(size_t a, size_t b, const char * message) {
    if (a != 0 && b > std::numeric_limits<size_t>::max()/a) {
        throw std::runtime_error(message);
    }
    return a*b;
}

size_t projection_index(llama_expert_projection projection) {
    if (projection < LLAMA_EXPERT_PROJECTION_GATE || projection > LLAMA_EXPERT_PROJECTION_DOWN) {
        throw std::invalid_argument("DeepSeek V4.1 expert projection is invalid");
    }
    return static_cast<size_t>(projection);
}

void require_local_cpu(ggml_backend_sched_t sched, ggml_backend_t backend_cpu) {
    if (sched == nullptr || backend_cpu == nullptr || !ggml_backend_is_cpu(backend_cpu)) {
        throw std::invalid_argument("DeepSeek V4.1 expert callback requires a local CPU backend");
    }
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        if (ggml_backend_sched_get_backend(sched, i) == backend_cpu) {
            return;
        }
    }
    throw std::invalid_argument("DeepSeek V4.1 expert CPU backend is not in the scheduler");
}

struct dsv41_expert_callback_state {
    llama_dsv41_expert_runtime * runtime = nullptr;
    int32_t layer = -1;
};

}

struct llama_dsv41_expert_runtime::impl {
    struct logical_slot {
        int32_t expert_id = -1;
        uint32_t pins = 0;
        uint64_t last_use = 0;
    };

    struct layer_state {
        std::array<ggml_tensor *, 3> tensors = {};
        std::vector<logical_slot> slots;
        std::unique_ptr<llama_expert_store::lease> lease;
        std::vector<uint32_t> pinned_slots;
        std::vector<int32_t> active_ids;
        std::vector<int32_t> active_remap;
    };

    llama_dsv41_expert_runtime_params params;
    std::unique_ptr<llama_expert_store> store;
    std::vector<layer_state> layers;
    std::vector<dsv41_expert_callback_state> callbacks;
    std::vector<std::pair<ggml_backend_buffer_type_t, ggml_context_ptr>> contexts;
    std::vector<ggml_backend_buffer_ptr> buffers;
    upload_fn upload;
    publish_fn before_publish;
    size_t bytes_cache = 0;
    size_t bytes_staging = 0;
    uint64_t use_clock = 0;
    std::string error;
    bool context_active = false;
    mutable std::mutex mutex;

    impl(
            llama_dsv41_expert_runtime * owner,
            std::vector<llama_expert_store_tensor> tensors,
            const llama_dsv41_expert_runtime_params & params,
            buft_selector select_buft,
            upload_fn upload,
            publish_fn before_publish)
        : params(params), upload(std::move(upload)), before_publish(std::move(before_publish)) {
        if (params.cache_bytes == 0 || params.cache_slots == 0) {
            throw std::runtime_error("DeepSeek V4.1 expert cache byte and slot capacity must be non-zero");
        }
        if (!select_buft) {
            throw std::invalid_argument("DeepSeek V4.1 expert cache buffer selector is empty");
        }
        if (tensors.size() != LLAMA_DSV41_N_LAYER*3) {
            throw std::runtime_error("DeepSeek V4.1 must register 40 gate/up/down routed tensor sets");
        }

        layers.resize(LLAMA_DSV41_N_LAYER);
        callbacks.resize(LLAMA_DSV41_N_LAYER);
        std::vector<size_t> layer_plane_bytes(LLAMA_DSV41_N_LAYER, 0);
        for (const auto & tensor : tensors) {
            llama_expert_store_validate_tensor(tensor);
            if (tensor.layer < 0 || tensor.layer >= (int32_t) LLAMA_DSV41_N_LAYER) {
                throw std::runtime_error("DeepSeek V4.1 routed tensor layer is invalid");
            }
            if (tensor.ne[2] != LLAMA_DSV41_N_EXPERT) {
                throw std::runtime_error("DeepSeek V4.1 routed tensor expert count mismatch");
            }
            layer_plane_bytes[tensor.layer] = checked_add(
                    layer_plane_bytes[tensor.layer],
                    tensor.nb[2],
                    "DeepSeek V4.1 expert plane byte count overflow");
            bytes_cache = checked_add(
                    bytes_cache,
                    checked_mul(tensor.nb[2], params.cache_slots, "DeepSeek V4.1 expert cache byte count overflow"),
                    "DeepSeek V4.1 expert cache byte count overflow");

            ggml_backend_buffer_type_t buft = select_buft(tensor);
            if (buft == nullptr) {
                throw std::runtime_error("DeepSeek V4.1 expert cache has no backend buffer type");
            }
            ggml_context * ctx = nullptr;
            for (auto & item : contexts) {
                if (item.first == buft) {
                    ctx = item.second.get();
                    break;
                }
            }
            if (ctx == nullptr) {
                ggml_init_params ctx_params = {
                    /*.mem_size   =*/ ggml_tensor_overhead()*(LLAMA_DSV41_N_LAYER*3 + 1),
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx = ggml_init(ctx_params);
                if (ctx == nullptr) {
                    throw std::runtime_error("DeepSeek V4.1 failed to create expert cache tensor context");
                }
                contexts.emplace_back(buft, ctx);
            }

            ggml_tensor * cache = ggml_new_tensor_3d(
                    ctx, tensor.type, tensor.ne[0], tensor.ne[1], params.cache_slots);
            ggml_format_name(cache, "%s.cache", tensor.name.c_str());
            layers[tensor.layer].tensors[projection_index(tensor.projection)] = cache;
        }
        if (bytes_cache > params.cache_bytes) {
            throw std::runtime_error(format(
                    "DeepSeek V4.1 expert cache requires %zu bytes for %zu slots per layer, configured %zu",
                    bytes_cache, params.cache_slots, params.cache_bytes));
        }

        for (int32_t il = 0; il < (int32_t) LLAMA_DSV41_N_LAYER; ++il) {
            for (ggml_tensor * tensor : layers[il].tensors) {
                if (tensor == nullptr) {
                    throw std::runtime_error(format("DeepSeek V4.1 layer %d is missing an expert cache tensor", il));
                }
            }
            layers[il].slots.resize(params.cache_slots);
            callbacks[il] = { owner, il };
            bytes_staging = std::max(
                    bytes_staging,
                    checked_mul(layer_plane_bytes[il], params.cache_slots, "DeepSeek V4.1 expert staging byte count overflow"));
        }

        for (auto & item : contexts) {
            ggml_backend_buffer_t buffer = nullptr;
            if (params.no_alloc) {
                buffer = ggml_backend_buft_alloc_buffer(item.first, 0);
                for (ggml_tensor * tensor = ggml_get_first_tensor(item.second.get());
                        tensor != nullptr;
                        tensor = ggml_get_next_tensor(item.second.get(), tensor)) {
                    tensor->buffer = buffer;
                }
            } else {
                buffer = ggml_backend_alloc_ctx_tensors_from_buft(item.second.get(), item.first);
            }
            if (buffer == nullptr) {
                throw std::runtime_error(format(
                        "DeepSeek V4.1 failed to allocate %s expert cache buffer",
                        ggml_backend_buft_name(item.first)));
            }
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            buffers.emplace_back(buffer);
        }

        if (!params.no_alloc) {
            llama_expert_store_params store_params;
            store_params.cache_bytes = bytes_staging;
            store_params.cache_slots = checked_mul(params.cache_slots, 3, "DeepSeek V4.1 expert staging slot count overflow");
            store_params.direct_io = params.direct_io;
            store_params.allow_buffered_io = params.allow_buffered_io;
            store = std::make_unique<llama_expert_store>(std::move(tensors), store_params);
        }

        if (!this->upload) {
            this->upload = [](ggml_tensor * tensor, size_t offset, const void * data, size_t size) {
                ggml_backend_tensor_set(tensor, data, offset, size);
            };
        }
    }

    const llama_expert_store::payload & payload(
            const std::vector<llama_expert_store::payload> & payloads,
            int32_t layer,
            llama_expert_projection projection,
            int32_t expert_id) const {
        for (const auto & payload : payloads) {
            if (payload.layer == layer && payload.projection == projection && payload.expert_id == expert_id) {
                return payload;
            }
        }
        throw std::runtime_error("DeepSeek V4.1 expert store did not return a requested payload");
    }

    std::vector<int32_t> remap(int32_t layer, const std::vector<int32_t> & expert_ids) {
        std::lock_guard<std::mutex> lock(mutex);
        if (store == nullptr) {
            throw std::runtime_error("DeepSeek V4.1 expert cache is metadata-only");
        }
        if (layer < 0 || layer >= (int32_t) layers.size()) {
            throw std::invalid_argument("DeepSeek V4.1 expert layer is invalid");
        }
        layer_state & state = layers[layer];
        if (state.lease) {
            if (state.active_ids == expert_ids) {
                return state.active_remap;
            }
            throw std::runtime_error(format("DeepSeek V4.1 expert layer %d still has an in-flight lease", layer));
        }

        std::vector<int32_t> unique = expert_ids;
        for (int32_t expert_id : unique) {
            if (expert_id < 0 || expert_id >= (int32_t) LLAMA_DSV41_N_EXPERT) {
                throw std::runtime_error("DeepSeek V4.1 selected expert ID is out of range");
            }
        }
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        if (unique.size() > params.cache_slots) {
            throw std::runtime_error(format(
                    "DeepSeek V4.1 selected expert union has %zu entries, cache has %zu slots",
                    unique.size(), params.cache_slots));
        }

        std::map<int32_t, uint32_t> resident;
        std::vector<uint32_t> empty;
        std::vector<uint32_t> victims;
        for (uint32_t slot = 0; slot < state.slots.size(); ++slot) {
            const logical_slot & entry = state.slots[slot];
            if (entry.expert_id >= 0) {
                resident.emplace(entry.expert_id, slot);
                if (entry.pins == 0 && !std::binary_search(unique.begin(), unique.end(), entry.expert_id)) {
                    victims.push_back(slot);
                }
            } else {
                empty.push_back(slot);
            }
        }
        std::sort(victims.begin(), victims.end(), [&](uint32_t a, uint32_t b) {
            if (state.slots[a].last_use != state.slots[b].last_use) {
                return state.slots[a].last_use < state.slots[b].last_use;
            }
            return a < b;
        });

        std::vector<int32_t> misses;
        for (int32_t expert_id : unique) {
            if (resident.count(expert_id) == 0) {
                misses.push_back(expert_id);
            }
        }
        if (misses.size() > empty.size() + victims.size()) {
            throw std::runtime_error("DeepSeek V4.1 expert cache capacity is pinned");
        }

        std::vector<uint32_t> targets = empty;
        targets.insert(targets.end(), victims.begin(), victims.end());
        targets.resize(misses.size());
        std::sort(targets.begin(), targets.end());

        auto lease = std::make_unique<llama_expert_store::lease>();
        if (!misses.empty()) {
            *lease = store->acquire({
                { layer, LLAMA_EXPERT_PROJECTION_GATE, misses },
                { layer, LLAMA_EXPERT_PROJECTION_UP,   misses },
                { layer, LLAMA_EXPERT_PROJECTION_DOWN, misses },
            });
            const auto payloads = lease->payloads();
            try {
                for (size_t i = 0; i < misses.size(); ++i) {
                    const uint32_t slot = targets[i];
                    for (llama_expert_projection projection : {
                            LLAMA_EXPERT_PROJECTION_GATE,
                            LLAMA_EXPERT_PROJECTION_UP,
                            LLAMA_EXPERT_PROJECTION_DOWN }) {
                        const auto & item = payload(payloads, layer, projection, misses[i]);
                        ggml_tensor * tensor = state.tensors[projection_index(projection)];
                        upload(tensor, slot*tensor->nb[2], item.data, item.size);
                    }
                }
            } catch (...) {
                for (uint32_t slot : targets) {
                    state.slots[slot] = {};
                }
                throw;
            }

            for (size_t i = 0; i < misses.size(); ++i) {
                const uint32_t slot = targets[i];
                state.slots[slot].expert_id = misses[i];
                resident[misses[i]] = slot;
            }
        }

        std::vector<uint32_t> pinned_slots;
        pinned_slots.reserve(unique.size());
        for (int32_t expert_id : unique) {
            const uint32_t slot = resident.at(expert_id);
            pinned_slots.push_back(slot);
        }

        std::vector<int32_t> result;
        result.reserve(expert_ids.size());
        for (int32_t expert_id : expert_ids) {
            result.push_back((int32_t) resident.at(expert_id));
        }
        std::vector<int32_t> active_ids = expert_ids;
        std::vector<int32_t> active_remap = result;

        if (before_publish) {
            before_publish();
        }
        for (uint32_t slot : pinned_slots) {
            logical_slot & entry = state.slots[slot];
            entry.last_use = ++use_clock;
            entry.pins++;
        }
        state.pinned_slots = std::move(pinned_slots);
        state.lease = std::move(lease);
        state.active_ids = std::move(active_ids);
        state.active_remap = std::move(active_remap);
        return result;
    }

    void release(int32_t layer) {
        std::lock_guard<std::mutex> lock(mutex);
        if (layer < 0 || layer >= (int32_t) layers.size()) {
            return;
        }
        layer_state & state = layers[layer];
        for (uint32_t slot : state.pinned_slots) {
            if (state.slots[slot].pins == 0) {
                GGML_ABORT("DeepSeek V4.1 expert slot pin underflow");
            }
            state.slots[slot].pins--;
        }
        state.pinned_slots.clear();
        state.lease.reset();
        state.active_ids.clear();
        state.active_remap.clear();
    }
};

llama_dsv41_expert_runtime::llama_dsv41_expert_runtime(
        std::vector<llama_expert_store_tensor> tensors,
        const llama_dsv41_expert_runtime_params & params,
        buft_selector select_buft,
        upload_fn upload,
        publish_fn before_publish)
    : pimpl(std::make_unique<impl>(
            this, std::move(tensors), params, std::move(select_buft), std::move(upload), std::move(before_publish))) {
}

llama_dsv41_expert_runtime::~llama_dsv41_expert_runtime() = default;

std::vector<int32_t> llama_dsv41_expert_runtime::remap(
        int32_t layer, const std::vector<int32_t> & expert_ids) {
    return pimpl->remap(layer, expert_ids);
}

void llama_dsv41_expert_runtime::release(int32_t layer) {
    pimpl->release(layer);
}

void llama_dsv41_expert_runtime::release_all() {
    for (int32_t il = 0; il < (int32_t) LLAMA_DSV41_N_LAYER; ++il) {
        pimpl->release(il);
    }
}

void llama_dsv41_expert_runtime::release_all_after_sync(ggml_backend_sched_t sched) {
    if (sched != nullptr) {
        ggml_backend_sched_synchronize(sched);
    }
    release_all();
}

ggml_tensor * llama_dsv41_expert_runtime::cache_tensor(
        int32_t layer, llama_expert_projection projection) const {
    if (layer < 0 || layer >= (int32_t) pimpl->layers.size()) {
        throw std::invalid_argument("DeepSeek V4.1 expert layer is invalid");
    }
    return pimpl->layers[layer].tensors[projection_index(projection)];
}

size_t llama_dsv41_expert_runtime::cache_slots() const {
    return pimpl->params.cache_slots;
}

size_t llama_dsv41_expert_runtime::cache_bytes() const {
    return pimpl->bytes_cache;
}

size_t llama_dsv41_expert_runtime::staging_bytes() const {
    return pimpl->bytes_staging;
}

void llama_dsv41_expert_runtime::set_error(const std::string & error) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->error.empty()) {
        pimpl->error = error;
    }
}

std::string llama_dsv41_expert_runtime::consume_error() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    std::string result;
    result.swap(pimpl->error);
    return result;
}

void llama_dsv41_expert_runtime::acquire_context() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->context_active) {
        throw std::runtime_error("DeepSeek V4.1 bounded expert runtime supports one context per model");
    }
    pimpl->context_active = true;
}

void llama_dsv41_expert_runtime::release_context() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->context_active = false;
}

std::vector<llama_expert_store_tensor> llama_dsv41_register_expert_tensors(
        const std::function<llama_expert_store_tensor(
                const std::string & name,
                int32_t layer,
                llama_expert_projection projection,
                const std::initializer_list<int64_t> & ne)> & register_tensor) {
    if (!register_tensor) {
        throw std::invalid_argument("DeepSeek V4.1 expert tensor registrar is empty");
    }
    std::vector<llama_expert_store_tensor> result;
    result.reserve(LLAMA_DSV41_N_LAYER*3);
    for (int32_t il = 0; il < (int32_t) LLAMA_DSV41_N_LAYER; ++il) {
        result.push_back(register_tensor(
                "blk." + std::to_string(il) + ".ffn_gate_exps.weight",
                il, LLAMA_EXPERT_PROJECTION_GATE,
                { LLAMA_DSV41_N_EMBD, LLAMA_DSV41_N_FF_EXP, LLAMA_DSV41_N_EXPERT }));
        result.push_back(register_tensor(
                "blk." + std::to_string(il) + ".ffn_up_exps.weight",
                il, LLAMA_EXPERT_PROJECTION_UP,
                { LLAMA_DSV41_N_EMBD, LLAMA_DSV41_N_FF_EXP, LLAMA_DSV41_N_EXPERT }));
        result.push_back(register_tensor(
                "blk." + std::to_string(il) + ".ffn_down_exps.weight",
                il, LLAMA_EXPERT_PROJECTION_DOWN,
                { LLAMA_DSV41_N_FF_EXP, LLAMA_DSV41_N_EMBD, LLAMA_DSV41_N_EXPERT }));
    }
    return result;
}

static void dsv41_expert_remap_callback(
        ggml_tensor * dst,
        const ggml_tensor * src,
        int,
        int,
        void * userdata) {
    auto * state = static_cast<dsv41_expert_callback_state *>(userdata);
    GGML_ASSERT(dst->type == GGML_TYPE_I32 && src->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src));
    const int32_t * ids = static_cast<const int32_t *>(src->data);
    const size_t count = ggml_nelements(src);
    try {
        const std::vector<int32_t> remapped = state->runtime->remap(
                state->layer, std::vector<int32_t>(ids, ids + count));
        memcpy(dst->data, remapped.data(), remapped.size()*sizeof(int32_t));
    } catch (const std::exception & error) {
        std::fill_n(static_cast<int32_t *>(dst->data), count, 0);
        state->runtime->set_error(error.what());
    }
}

static void dsv41_expert_release_callback(
        ggml_tensor * dst,
        const ggml_tensor * src,
        int,
        int,
        void * userdata) {
    auto * state = static_cast<dsv41_expert_callback_state *>(userdata);
    memcpy(dst->data, src->data, ggml_nbytes(src));
    state->runtime->release(state->layer);
}

ggml_tensor * llama_dsv41_build_expert_remap(
        ggml_context * ctx,
        ggml_tensor * selected_experts,
        llama_dsv41_expert_runtime & runtime,
        int32_t layer,
        ggml_backend_sched_t sched,
        ggml_backend_t backend_cpu) {
    if (ctx == nullptr || selected_experts == nullptr || selected_experts->type != GGML_TYPE_I32) {
        throw std::invalid_argument("DeepSeek V4.1 expert remap input is invalid");
    }
    require_local_cpu(sched, backend_cpu);
    ggml_tensor * original = ggml_cont(ctx, selected_experts);
    ggml_tensor * remapped = ggml_map_custom1(
            ctx, original, dsv41_expert_remap_callback, 1, &runtime.pimpl->callbacks.at(layer));
    ggml_backend_sched_set_tensor_backend(sched, remapped, backend_cpu);
    return remapped;
}

ggml_tensor * llama_dsv41_build_expert_release(
        ggml_context * ctx,
        ggml_tensor * experts,
        llama_dsv41_expert_runtime & runtime,
        int32_t layer,
        ggml_backend_sched_t sched,
        ggml_backend_t backend_cpu) {
    if (ctx == nullptr || experts == nullptr) {
        throw std::invalid_argument("DeepSeek V4.1 expert release input is invalid");
    }
    require_local_cpu(sched, backend_cpu);
    ggml_tensor * completion = ggml_sum(ctx, experts);
    completion = ggml_map_custom1(
            ctx, completion, dsv41_expert_release_callback, 1, &runtime.pimpl->callbacks.at(layer));
    ggml_backend_sched_set_tensor_backend(sched, completion, backend_cpu);
    return completion;
}
