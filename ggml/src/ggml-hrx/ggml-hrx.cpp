#include "ggml-hrx.h"

#include "ggml-hrx-common.h"
#include "ggml-hrx-graph.h"
#include "ggml-hrx-ops.h"
#include "ops/autofuse.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <optional>

static constexpr size_t GGML_HRX_ALIGNMENT = 256;

// Device memory has no host address. Tensor data pointers are offsets from this fake base. The Vulkan backend uses the same method.
static uint8_t * const GGML_HRX_BASE = (uint8_t *) (uintptr_t) 0x1000;

bool ggml_hrx_check(hrx_status_t status, const char * expr, const char * file, int line) {
    if (hrx_status_is_ok(status)) {
        return true;
    }
    char * message = nullptr;
    size_t length  = 0;
    hrx_status_t format_status = hrx_status_to_string(status, &message, &length);
    if (!hrx_status_is_ok(format_status)) {
        hrx_status_ignore(format_status);
    }
    GGML_LOG_ERROR("%s:%d: %s failed: %s\n", file, line, expr, message != nullptr ? message : "unknown HRX error");
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return false;
}

static std::atomic<uint64_t> ggml_hrx_next_buffer_id{1};

struct ggml_hrx_buffer_context {
    const uint64_t id = ggml_hrx_next_buffer_id.fetch_add(1);
    std::atomic<uint64_t> weight_version{0};
    bool shared_pool = false;
    ggml_hrx_device_context * dev    = nullptr;
    hrx_buffer_t              buffer = nullptr;
};

static void ggml_hrx_weight_changed(ggml_backend_buffer_t buffer) {
    auto * b = static_cast<ggml_hrx_buffer_context *>(buffer->context);
    if (b->dev->npu && ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        b->weight_version = ++b->dev->weight_version;
    }
}

static void * ggml_backend_hrx_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return GGML_HRX_BASE;
}

static size_t ggml_hrx_tensor_offset(const ggml_tensor * t) {
    return (size_t) ((const uint8_t *) t->data - GGML_HRX_BASE);
}

static ggml_backend_buffer_t ggml_hrx_tensor_buffer(const ggml_tensor * t) {
    return t->view_src != nullptr ? t->view_src->buffer : t->buffer;
}

bool ggml_hrx_tensor_on_device(const ggml_tensor * t) {
    ggml_backend_buffer_t buffer = ggml_hrx_tensor_buffer(t);
    return buffer != nullptr && buffer->iface.get_base == ggml_backend_hrx_buffer_get_base;
}

// buffer_stream records uploads up to this size. They run before the next graph or read.
static constexpr size_t GGML_HRX_MAX_RECORDED_UPLOAD = 1024 * 1024;
// This is the HAL limit for one update_buffer command.
static constexpr size_t GGML_HRX_UPDATE_CHUNK = 64 * 1024;

// Hold buffer_mutex when you call this function.
static bool ggml_hrx_sync_buffer_stream(ggml_hrx_device_context * dev) {
    dev->upload_pending = false;
    return GGML_HRX_CHECK(hrx_stream_synchronize(dev->buffer_stream));
}

static bool ggml_hrx_finish_uploads(ggml_hrx_device_context * dev) {
    std::lock_guard<std::mutex> lock(dev->buffer_mutex);
    return !dev->upload_pending || ggml_hrx_sync_buffer_stream(dev);
}

static void ggml_backend_hrx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_hrx_buffer_context *) buffer->context;
    if (ctx->buffer != nullptr) {
        ggml_hrx_weight_changed(buffer);
        ggml_hrx_finish_uploads(ctx->dev);
        if (!ctx->shared_pool && ctx->dev->npu != nullptr) {
            ggml_hrx_npu_forget(ctx->dev->npu, ctx->buffer);
        }
        hrx_buffer_release(ctx->buffer);
    }
    delete ctx;
}

struct ggml_hrx_fill_args {
    hrx_buffer_t buffer;
    size_t       offset;
    size_t       size;
    uint8_t      value;
};

uint64_t ggml_hrx_weight_version(const ggml_tensor * t) {
    return static_cast<ggml_hrx_buffer_context *>(ggml_hrx_tensor_buffer(t)->context)->weight_version.load();
}

static bool ggml_hrx_fill(ggml_hrx_device_context * dev, const ggml_hrx_fill_args & args) {
    std::lock_guard<std::mutex> lock(dev->buffer_mutex);
    if (!GGML_HRX_CHECK(hrx_stream_fill_buffer(dev->buffer_stream, args.buffer, args.offset, args.size, &args.value, sizeof(args.value)))) {
        return false;
    }
    return ggml_hrx_sync_buffer_stream(dev);
}

static void ggml_backend_hrx_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                  size_t offset, size_t size) {
    if (size == 0) {
        return;
    }
    ggml_hrx_weight_changed(buffer);
    auto *       ctx = (ggml_hrx_buffer_context *) buffer->context;
    const size_t dst = ggml_hrx_tensor_offset(tensor) + offset;
    GGML_ASSERT(dst <= buffer->size && size <= buffer->size - dst);
    // Like a small upload, the fill runs before the next graph and before any read (ggml_hrx_wait_uploads). A
    // speculative step frees KV cells in every layer, and a host wait for each fill cost several ms for each step.
    ggml_hrx_device_context * dev = ctx->dev;
    std::lock_guard<std::mutex> lock(dev->buffer_mutex);
    if (!GGML_HRX_CHECK(hrx_stream_fill_buffer(dev->buffer_stream, ctx->buffer, dst, size, &value, sizeof(value)))) {
        GGML_ABORT("%s: HRX buffer fill failed", __func__);
    }
    dev->upload_pending = true;
}

static void ggml_backend_hrx_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
                                               size_t offset, size_t size) {
    if (size == 0) {
        return;
    }
    ggml_hrx_weight_changed(buffer);
    auto *       ctx = (ggml_hrx_buffer_context *) buffer->context;
    const size_t dst = ggml_hrx_tensor_offset(tensor) + offset;
    GGML_ASSERT(dst <= buffer->size && size <= buffer->size - dst);
    ggml_hrx_device_context * dev = ctx->dev;
    if (size <= GGML_HRX_MAX_RECORDED_UPLOAD) {
        std::lock_guard<std::mutex> lock(dev->buffer_mutex);
        // The stream copies the data now. The caller can reuse it.
        for (size_t i = 0; i < size; i += GGML_HRX_UPDATE_CHUNK) {
            const size_t n = std::min(GGML_HRX_UPDATE_CHUNK, size - i);
            if (!GGML_HRX_CHECK(hrx_stream_update_buffer(dev->buffer_stream, (const uint8_t *) data + i, n, ctx->buffer, dst + i))) {
                GGML_ABORT("%s: HRX upload failed", __func__);
            }
        }
        dev->upload_pending = true;
        return;
    }
    if (!ggml_hrx_finish_uploads(dev) || !GGML_HRX_CHECK(hrx_synchronous_h2d(dev->transfer_queue, data, ctx->buffer, dst, size))) {
        GGML_ABORT("%s: HRX upload failed", __func__);
    }
}

static void ggml_backend_hrx_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
                                               size_t offset, size_t size) {
    if (size == 0) {
        return;
    }
    auto *       ctx = (ggml_hrx_buffer_context *) buffer->context;
    const size_t src = ggml_hrx_tensor_offset(tensor) + offset;
    GGML_ASSERT(src <= buffer->size && size <= buffer->size - src);
    if (!ggml_hrx_finish_uploads(ctx->dev) ||
        !GGML_HRX_CHECK(hrx_synchronous_d2h(ctx->dev->transfer_queue, ctx->buffer, src, data, size))) {
        GGML_ABORT("%s: HRX download failed", __func__);
    }
}

// A state save or restore moves many small ranges. The stream queues all copies, then the host waits one time.
// The tensors can be in other buffers of the same type, on the same device.
static void ggml_backend_hrx_buffer_set_tensor_n(ggml_backend_buffer_t buffer, ggml_tensor * const * tensors, const void * const * data,
                                                 const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    ggml_hrx_device_context * dev = ((ggml_hrx_buffer_context *) buffer->context)->dev;
    std::lock_guard<std::mutex> lock(dev->buffer_mutex);
    for (size_t i = 0; i < n_tensors; ++i) {
        if (sizes[i] == 0) {
            continue;
        }
        ggml_backend_buffer_t buf = ggml_hrx_tensor_buffer(tensors[i]);
        ggml_hrx_weight_changed(buf);
        const size_t          dst = ggml_hrx_tensor_offset(tensors[i]) + offsets[i];
        GGML_ASSERT(dst <= buf->size && sizes[i] <= buf->size - dst);
        auto * bctx = (ggml_hrx_buffer_context *) buf->context;
        if (!GGML_HRX_CHECK(hrx_stream_copy_h2d(dev->buffer_stream, data[i], bctx->buffer, dst, sizes[i]))) {
            GGML_ABORT("%s: HRX upload failed", __func__);
        }
    }
    if (!ggml_hrx_sync_buffer_stream(dev)) {
        GGML_ABORT("%s: HRX upload failed", __func__);
    }
}

static void ggml_backend_hrx_buffer_get_tensor_n(ggml_backend_buffer_t buffer, const ggml_tensor * const * tensors, void * const * data,
                                                 const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    ggml_hrx_device_context * dev = ((ggml_hrx_buffer_context *) buffer->context)->dev;
    std::lock_guard<std::mutex> lock(dev->buffer_mutex);
    for (size_t i = 0; i < n_tensors; ++i) {
        if (sizes[i] == 0) {
            continue;
        }
        ggml_backend_buffer_t buf = ggml_hrx_tensor_buffer(tensors[i]);
        const size_t          src = ggml_hrx_tensor_offset(tensors[i]) + offsets[i];
        GGML_ASSERT(src <= buf->size && sizes[i] <= buf->size - src);
        auto * bctx = (ggml_hrx_buffer_context *) buf->context;
        if (!GGML_HRX_CHECK(hrx_stream_copy_d2h(dev->buffer_stream, bctx->buffer, src, data[i], sizes[i]))) {
            GGML_ABORT("%s: HRX download failed", __func__);
        }
    }
    if (!ggml_hrx_sync_buffer_stream(dev)) {
        GGML_ABORT("%s: HRX download failed", __func__);
    }
}

static bool ggml_backend_hrx_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_hrx_tensor_on_device(src)) {
        return false;
    }
    auto * src_ctx = (ggml_hrx_buffer_context *) ggml_hrx_tensor_buffer(src)->context;
    auto * dst_ctx = (ggml_hrx_buffer_context *) buffer->context;
    if (src_ctx->dev != dst_ctx->dev) {
        return false;
    }
    const size_t size = ggml_nbytes(src);
    if (size == 0) {
        return true;
    }
    ggml_hrx_weight_changed(buffer);
    ggml_hrx_device_context * dev = dst_ctx->dev;
    std::lock_guard<std::mutex> lock(dev->buffer_mutex);
    if (!GGML_HRX_CHECK(hrx_stream_copy_buffer(dev->buffer_stream, src_ctx->buffer, ggml_hrx_tensor_offset(src),
                                               dst_ctx->buffer, ggml_hrx_tensor_offset(dst), size))) {
        return false;
    }
    return ggml_hrx_sync_buffer_stream(dev);
}

static void ggml_backend_hrx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_hrx_buffer_context *) buffer->context;
    if (buffer->size == 0 || ctx->buffer == nullptr) {
        return;
    }
    ggml_hrx_weight_changed(buffer);
    if (!ggml_hrx_fill(ctx->dev, { ctx->buffer, 0, buffer->size, value })) {
        GGML_ABORT("%s: HRX buffer clear failed", __func__);
    }
}

static const ggml_backend_buffer_i ggml_backend_hrx_buffer_interface = {
    /* .free_buffer   = */ ggml_backend_hrx_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_hrx_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_hrx_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_hrx_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_hrx_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ ggml_backend_hrx_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_hrx_buffer_clear,
    /* .reset         = */ nullptr,
    /* .set_tensor_n  = */ ggml_backend_hrx_buffer_set_tensor_n,
    /* .get_tensor_n  = */ ggml_backend_hrx_buffer_get_tensor_n,
    /* .cpy_tensor_n  = */ nullptr,
};

static const char * ggml_backend_hrx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return ((ggml_hrx_device_context *) buft->context)->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_hrx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * dev = (ggml_hrx_device_context *) buft->context;

    hrx_buffer_t allocation = nullptr;
    if (size > 0) {
        const hrx_buffer_params_t params = {
            HRX_MEMORY_TYPE_DEVICE_LOCAL,
            HRX_MEMORY_ACCESS_ALL,
            HRX_BUFFER_USAGE_DEFAULT,
        };
        if (!GGML_HRX_CHECK(hrx_allocator_allocate_buffer(hrx_device_allocator(dev->device), params, size, &allocation))) {
            GGML_LOG_ERROR("%s: failed to allocate %.2f MiB on %s\n", __func__, size / 1024.0 / 1024.0, dev->name.c_str());
            return nullptr;
        }
    }

    auto * ctx   = new ggml_hrx_buffer_context;
    ctx->dev     = dev;
    ctx->buffer  = allocation;
    return ggml_backend_buffer_init(buft, ggml_backend_hrx_buffer_interface, ctx, size);
}

ggml_backend_buffer_t ggml_hrx_wrap_buffer(ggml_hrx_device_context * dev, hrx_buffer_t buffer, size_t size) {
    auto * ctx   = new ggml_hrx_buffer_context;
    ctx->dev     = dev;
    ctx->buffer  = buffer;
    ctx->shared_pool = true;
    return ggml_backend_buffer_init(&dev->buft, ggml_backend_hrx_buffer_interface, ctx, size);
}

hrx_buffer_t ggml_hrx_buffer_handle(ggml_backend_buffer_t buffer) {
    return ((ggml_hrx_buffer_context *) buffer->context)->buffer;
}

size_t ggml_hrx_offset(const ggml_tensor * t) {
    return ggml_hrx_tensor_offset(t);
}

static size_t ggml_backend_hrx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_HRX_ALIGNMENT;
}

static size_t ggml_backend_hrx_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return ((ggml_hrx_device_context *) buft->context)->memory_total;
}

static const ggml_backend_buffer_type_i ggml_backend_hrx_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_hrx_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_hrx_buffer_type_alloc_buffer,
    /* .alloc_buffer_n   = */ nullptr,
    /* .get_alignment    = */ ggml_backend_hrx_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_hrx_buffer_type_get_max_size,
    /* .get_alloc_size   = */ nullptr,
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ nullptr,
};

ggml_hrx_kernel::~ggml_hrx_kernel() {
    ggml_hrx_loom_launch_program_free(launch);
    if (executable != nullptr) {
        hrx_executable_release(executable);
    }
}

static const ggml_hrx_kernel_source * ggml_hrx_find_source(const char * path) {
    for (size_t i = 0; i < ggml_hrx_kernel_source_count; ++i) {
        if (std::strcmp(ggml_hrx_kernel_sources[i].path, path) == 0) {
            return &ggml_hrx_kernel_sources[i];
        }
    }
    GGML_LOG_ERROR("%s: kernel source %s is not embedded\n", __func__, path);
    return nullptr;
}

static std::unique_ptr<ggml_hrx_kernel> ggml_hrx_build_kernel(ggml_hrx_device_context * dev, const ggml_hrx_kernel_desc & desc) {
    if (dev->jit == nullptr && !GGML_HRX_CHECK(ggml_hrx_loom_jit_create(dev->arch->name, dev->arch->loom_target, &dev->jit))) {
        return nullptr;
    }

    const ggml_hrx_kernel_source   generated = { desc.file, (const unsigned char *) (desc.source != nullptr ? desc.source->data() : nullptr),
                                                 desc.source != nullptr ? desc.source->size() : 0 };
    const ggml_hrx_kernel_source * source    = desc.source != nullptr ? &generated : ggml_hrx_find_source(desc.file);
    if (source == nullptr) {
        return nullptr;
    }
    std::vector<ggml_hrx_loom_source> deps;
    for (const char * lib : desc.libs) {
        const ggml_hrx_kernel_source * dep = ggml_hrx_find_source(lib);
        if (dep == nullptr) {
            return nullptr;
        }
        deps.push_back({ dep->data, dep->size, dep->path });
    }
    std::vector<ggml_hrx_loom_config_binding> config;
    for (const auto & kv : desc.config) {
        config.push_back({ kv.first.c_str(), kv.second.c_str() });
    }

    ggml_hrx_loom_compile_options options;
    options.source   = { source->data, source->size, source->path };
    options.deps     = deps.data();
    options.n_deps   = deps.size();
    options.config   = config.data();
    options.n_config = config.size();
    options.symbol   = desc.symbol;

    ggml_hrx_loom_compile_result compiled;
    if (!GGML_HRX_CHECK(ggml_hrx_loom_compile(dev->jit, &options, &compiled))) {
        GGML_LOG_ERROR("%s: failed to compile %s from %s\n", __func__, desc.symbol, desc.file);
        return nullptr;
    }

    auto kernel    = std::make_unique<ggml_hrx_kernel>();
    kernel->launch = compiled.launch;
    kernel->symbol = desc.symbol;
    if (!GGML_HRX_CHECK(hrx_executable_load_data(dev->device, compiled.hsaco.data(), compiled.hsaco.size(), "amdgpu",
                                                 dev->arch->name, &kernel->executable)) ||
        !GGML_HRX_CHECK(hrx_executable_lookup_export_by_name(kernel->executable, desc.symbol, &kernel->export_ordinal)) ||
        !GGML_HRX_CHECK(hrx_executable_export_info(kernel->executable, kernel->export_ordinal, &kernel->export_info))) {
        return nullptr;
    }

    const hrx_executable_export_info_t & info = kernel->export_info;
    if (info.parameter_count < info.binding_count) {
        GGML_LOG_ERROR("%s: %s has an invalid parameter count\n", __func__, desc.symbol);
        return nullptr;
    }
    kernel->n_scalars = info.parameter_count - info.binding_count;
    if (kernel->n_scalars == 0) {
        if (info.constant_byte_length != 0) {
            GGML_LOG_ERROR("%s: %s has constants but no scalar parameters\n", __func__, desc.symbol);
            return nullptr;
        }
    } else {
        kernel->scalar_size = info.constant_byte_length / kernel->n_scalars;
        if (info.constant_byte_length % kernel->n_scalars != 0 || (kernel->scalar_size != 4 && kernel->scalar_size != 8)) {
            GGML_LOG_ERROR("%s: %s has an unsupported scalar ABI\n", __func__, desc.symbol);
            return nullptr;
        }
    }
    return kernel;
}

ggml_hrx_kernel * ggml_hrx_get_kernel(ggml_hrx_device_context * dev, const ggml_hrx_kernel_desc & desc) {
    std::string key = std::string(desc.file) + "|" + desc.symbol;
    if (desc.source != nullptr) {
        key += "|source=" + std::to_string(std::hash<std::string>()(*desc.source)) + ":" + std::to_string(desc.source->size());
    }
    for (const auto & kv : desc.config) {
        key += "|" + kv.first + "=" + kv.second;
    }

    std::lock_guard<std::mutex> lock(dev->kernel_mutex);
    auto it = dev->kernels.find(key);
    if (it != dev->kernels.end()) {
        return it->second.get();
    }
    // Cache a failed build as nullptr. This prevents a retry for each node.
    auto & slot = dev->kernels[key];
    slot = ggml_hrx_build_kernel(dev, desc);
    return slot.get();
}

static bool ggml_hrx_ref_overlaps(const hrx_buffer_ref_t & ref, const ggml_tensor * t) {
    if (t == nullptr || !ggml_hrx_tensor_on_device(t)) {
        return false;
    }
    const auto * bctx   = (const ggml_hrx_buffer_context *) ggml_hrx_tensor_buffer(t)->context;
    const size_t offset = ggml_hrx_tensor_offset(t);
    return bctx->buffer == ref.buffer && offset < ref.offset + ref.length && ref.offset < offset + ggml_nbytes(t);
}

bool ggml_hrx_dispatch(ggml_backend_hrx_context * ctx, ggml_hrx_kernel * kernel, const std::vector<int64_t> & workload,
                       const std::vector<int64_t> & scalars, const std::vector<const ggml_tensor *> & buffers) {
    if (kernel == nullptr) {
        return false;
    }
    if (scalars.size() != kernel->n_scalars || buffers.size() != kernel->export_info.binding_count) {
        GGML_LOG_ERROR("%s: argument count mismatch: %zu scalars, %zu buffers\n", __func__, scalars.size(), buffers.size());
        return false;
    }

    ggml_hrx_loom_launch_config launch;
    if (!GGML_HRX_CHECK(ggml_hrx_loom_launch_program_eval(kernel->launch, workload.data(), workload.size(), &launch))) {
        return false;
    }
    const uint32_t max_yz = ctx->device->arch->max_workgroups_yz;
    if (launch.workgroup_count[1] > max_yz || launch.workgroup_count[2] > max_yz) {
        GGML_LOG_ERROR("%s: workgroup count y or z is above %u\n", __func__, max_yz);
        return false;
    }

    std::vector<uint8_t> constants(kernel->n_scalars * kernel->scalar_size);
    for (size_t i = 0; i < scalars.size(); ++i) {
        const int64_t value = scalars[i];
        if (value < 0 || (kernel->scalar_size == 4 && (uint64_t) value > std::numeric_limits<uint32_t>::max())) {
            GGML_LOG_ERROR("%s: launch scalar %zu out of range: %" PRId64 "\n", __func__, i, value);
            return false;
        }
        if (kernel->scalar_size == 4) {
            const uint32_t v = (uint32_t) value;
            std::memcpy(constants.data() + i * 4, &v, 4);
        } else {
            const uint64_t v = (uint64_t) value;
            std::memcpy(constants.data() + i * 8, &v, 8);
        }
    }

    std::vector<hrx_buffer_ref_t> refs;
    refs.reserve(buffers.size());
    for (const ggml_tensor * t : buffers) {
        if (t == nullptr || !ggml_hrx_tensor_on_device(t)) {
            GGML_LOG_ERROR("%s: tensor %s is not in an HRX buffer\n", __func__, t != nullptr ? t->name : "(null)");
            return false;
        }
        auto * bctx = (ggml_hrx_buffer_context *) ggml_hrx_tensor_buffer(t)->context;
        refs.push_back({ bctx->buffer, ggml_hrx_tensor_offset(t), ggml_nbytes(t) });
    }

    const hrx_dispatch_config_t config = {
        { launch.workgroup_count[0], launch.workgroup_count[1], launch.workgroup_count[2] },
        { launch.workgroup_size[0],  launch.workgroup_size[1],  launch.workgroup_size[2] },
        launch.subgroup_size,
    };
    if (ctx->trace) {
        const ggml_tensor * n = ctx->cur_node;
        fprintf(stderr, "%" PRId64 "\t%s\t%s\t%s\t%u,%u,%u\n", ctx->n_graphs_seen, n != nullptr ? n->name : "-",
                n != nullptr ? ggml_op_desc(n) : "-", kernel->symbol.c_str(), launch.workgroup_count[0], launch.workgroup_count[1],
                launch.workgroup_count[2]);
    }
    if (ctx->npu != nullptr && !ggml_hrx_npu_sync(ctx, refs)) {
        return false;
    }
    if (ctx->rec != nullptr) {
        // A binding is written if it overlaps the node output, a declared write or the scratch.
        std::vector<bool> written(refs.size());
        for (size_t k = 0; k < refs.size(); ++k) {
            bool w = buffers[k] == &ctx->scratch_tensor || ggml_hrx_ref_overlaps(refs[k], ctx->cur_node);
            if (ctx->dispatch_writes != nullptr) {
                for (const ggml_tensor * t : *ctx->dispatch_writes) {
                    w = w || ggml_hrx_ref_overlaps(refs[k], t);
                }
            }
            written[k] = w;
        }
        ggml_hrx_graph_rec_dispatch(ctx->rec, kernel, config, constants, refs, written);
        return true;
    }
    return GGML_HRX_CHECK(hrx_stream_dispatch(ctx->stream, kernel->executable, kernel->export_ordinal, &config,
                                              constants.data(), constants.size(), refs.data(), refs.size(), 0));
}

const ggml_tensor * ggml_hrx_scratch(ggml_backend_hrx_context * ctx, size_t size) {
    if (ctx->scratch.empty() || ggml_backend_buffer_get_size(ctx->scratch.back()) < size) {
        // Dispatches in flight can use the old buffer. The context frees it at the end.
        size = std::max(size, 2 * (ctx->scratch.empty() ? 0 : ggml_backend_buffer_get_size(ctx->scratch.back())));
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(&ctx->device->buft, size);
        if (buffer == nullptr) {
            return nullptr;
        }
        ctx->scratch.push_back(buffer);
        ggml_tensor & t = ctx->scratch_tensor;
        t               = {};
        t.type          = GGML_TYPE_I8;
        t.buffer        = buffer;
        t.data          = ggml_backend_buffer_get_base(buffer);
        t.ne[0]         = (int64_t) size;
        t.ne[1] = t.ne[2] = t.ne[3] = 1;
        t.nb[0]         = 1;
        t.nb[1] = t.nb[2] = t.nb[3] = size;
    }
    return &ctx->scratch_tensor;
}

static ggml_guid_t ggml_backend_hrx_guid() {
    static ggml_guid guid = { 0xd2, 0x3d, 0x72, 0x83, 0xb2, 0x82, 0x4d, 0xe0, 0x8a, 0x3e, 0x21, 0x1d, 0x68, 0x87, 0x2f, 0x4b };
    return &guid;
}

static const char * ggml_backend_hrx_get_name(ggml_backend_t backend) {
    return ((ggml_backend_hrx_context *) backend->context)->name.c_str();
}

static void ggml_hrx_readback_release(ggml_backend_hrx_context * ctx) {
    if (ctx->readback != nullptr) {
        GGML_HRX_CHECK(hrx_buffer_unmap(ctx->readback));
        hrx_buffer_release(ctx->readback);
    }
    ctx->readback      = nullptr;
    ctx->readback_host = nullptr;
    ctx->readback_size = 0;
}

// Returns false if the readback buffer cannot take size more bytes now.
static bool ggml_hrx_readback_reserve(ggml_backend_hrx_context * ctx, size_t size) {
    if (ctx->readback_used + size <= ctx->readback_size) {
        return true;
    }
    // Pending copies use the current buffer.
    if (ctx->readback_failed || !ctx->readback_copies.empty()) {
        return false;
    }
    size = std::max(size, 2 * ctx->readback_size);
    ggml_hrx_readback_release(ctx);
    const hrx_buffer_params_t params = {
        HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_HOST_COHERENT,
        HRX_MEMORY_ACCESS_ALL,
        HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
    };
    void * host = nullptr;
    if (!GGML_HRX_CHECK(hrx_allocator_allocate_buffer(hrx_device_allocator(ctx->device->device), params, size, &ctx->readback)) ||
        !GGML_HRX_CHECK(hrx_buffer_map(ctx->readback, HRX_MAP_READ, 0, size, &host))) {
        if (ctx->readback != nullptr) {
            hrx_buffer_release(ctx->readback);
            ctx->readback = nullptr;
        }
        ctx->readback_failed = true;
        return false;
    }
    ctx->readback_host = (const uint8_t *) host;
    ctx->readback_size = size;
    return true;
}

static bool ggml_hrx_stream_finish(ggml_backend_hrx_context * ctx) {
    if (!GGML_HRX_CHECK(hrx_stream_synchronize(ctx->stream))) {
        return false;
    }
    for (const auto & c : ctx->readback_copies) {
        std::memcpy(c.dst, ctx->readback_host + c.offset, c.size);
    }
    ctx->readback_copies.clear();
    ctx->readback_used = 0;
    return true;
}

static void ggml_hrx_profile_print(ggml_backend_hrx_context * ctx);

static void ggml_backend_hrx_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_hrx_context *) backend->context;
    GGML_HRX_CHECK(hrx_stream_synchronize(ctx->stream));
    if (ctx->prof != nullptr) {
        ggml_hrx_profile_print(ctx);
    }
    ggml_hrx_autofuse_free(ctx->autofuse);
    ggml_hrx_fuse_free(ctx->fuse);
    GGML_LOG_DEBUG("%s: graphs: %" PRId64 " replayed, %" PRId64 " recorded, %" PRId64 " run directly\n", ctx->name.c_str(),
                   ctx->n_replayed, ctx->n_recorded, ctx->n_direct);
    hrx_stream_release(ctx->stream);
    ggml_hrx_readback_release(ctx);
    for (auto & kv : ctx->graphs) {
        if (kv.second.exec != nullptr) {
            hrx_graph_exec_release(kv.second.exec);
        }
        ggml_hrx_npu_replay_free(kv.second.npu);
    }
    ggml_hrx_npu_state_free(ctx->npu);
    for (ggml_backend_buffer_t buffer : ctx->fa_cls_buffers) {
        ggml_backend_buffer_free(buffer);
    }
    for (ggml_backend_buffer_t buffer : ctx->scratch) {
        ggml_backend_buffer_free(buffer);
    }
    delete ctx;
    delete backend;
}

static void ggml_backend_hrx_synchronize(ggml_backend_t backend) {
    ggml_hrx_stream_finish((ggml_backend_hrx_context *) backend->context);
}

// Make the backend stream wait for the uploads recorded in buffer_stream.
static bool ggml_hrx_wait_uploads(ggml_backend_hrx_context * ctx) {
    ggml_hrx_device_context * dev = ctx->device;
    hrx_timeline_point_t      point;
    {
        std::lock_guard<std::mutex> lock(dev->buffer_mutex);
        if (dev->upload_pending && !GGML_HRX_CHECK(hrx_stream_flush(dev->buffer_stream))) {
            return false;
        }
        if (!GGML_HRX_CHECK(hrx_stream_get_timeline_position(dev->buffer_stream, &point))) {
            return false;
        }
    }
    if (point.value <= ctx->upload_wait) {
        return true;
    }
    ctx->upload_wait = point.value;
    return GGML_HRX_CHECK(hrx_stream_wait_on(ctx->stream, point));
}

static void ggml_backend_hrx_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data,
                                              size_t offset, size_t size) {
    auto * ctx = (ggml_backend_hrx_context *) backend->context;
    if (size == 0) {
        return;
    }
    GGML_ASSERT(ggml_hrx_tensor_on_device(tensor));
    ggml_hrx_weight_changed(ggml_hrx_tensor_buffer(tensor));
    auto *       bctx = (ggml_hrx_buffer_context *) ggml_hrx_tensor_buffer(tensor)->context;
    const size_t dst  = ggml_hrx_tensor_offset(tensor) + offset;
    const bool   ok   = ggml_hrx_wait_uploads(ctx) &&
                    (size <= GGML_HRX_UPDATE_CHUNK ?
                         GGML_HRX_CHECK(hrx_stream_update_buffer(ctx->stream, data, size, bctx->buffer, dst)) :
                         GGML_HRX_CHECK(hrx_stream_copy_h2d(ctx->stream, data, bctx->buffer, dst, size)));
    if (!ok) {
        GGML_ABORT("%s: HRX upload failed", __func__);
    }
}

static void ggml_backend_hrx_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data,
                                              size_t offset, size_t size) {
    auto * ctx = (ggml_backend_hrx_context *) backend->context;
    if (size == 0) {
        return;
    }
    GGML_ASSERT(ggml_hrx_tensor_on_device(tensor));
    auto *       bctx = (ggml_hrx_buffer_context *) ggml_hrx_tensor_buffer(tensor)->context;
    const size_t src  = ggml_hrx_tensor_offset(tensor) + offset;
    if (!ggml_hrx_wait_uploads(ctx)) {
        GGML_ABORT("%s: HRX download failed", __func__);
    }
    if (ctx->device->tune->mapped_readback && ggml_hrx_readback_reserve(ctx, size)) {
        const size_t dst = ctx->readback_used;
        if (!GGML_HRX_CHECK(hrx_stream_copy_buffer(ctx->stream, bctx->buffer, src, ctx->readback, dst, size))) {
            GGML_ABORT("%s: HRX download failed", __func__);
        }
        ctx->readback_copies.push_back({ data, dst, size });
        ctx->readback_used = dst + GGML_PAD(size, GGML_HRX_ALIGNMENT);
        return;
    }
    if (!GGML_HRX_CHECK(hrx_stream_copy_d2h(ctx->stream, bctx->buffer, src, data, size))) {
        GGML_ABORT("%s: HRX download failed", __func__);
    }
}

static void ggml_backend_hrx_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * ctx = (ggml_backend_hrx_context *) backend->context;
    // The host copies of earlier get_tensor_async calls must be done when the event completes.
    if (!ctx->readback_copies.empty()) {
        ggml_hrx_stream_finish(ctx);
    }
    GGML_HRX_CHECK(hrx_event_record((hrx_event_t) event->context, ctx->stream));
}

static void ggml_backend_hrx_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * ctx = (ggml_backend_hrx_context *) backend->context;
    GGML_HRX_CHECK(hrx_stream_wait_event(ctx->stream, (hrx_event_t) event->context));
}

static bool ggml_hrx_compute_node(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:            return ggml_hrx_op_binary(ctx, node);
        case GGML_OP_UNARY:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_CLAMP:
        case GGML_OP_FILL:
        case GGML_OP_REPEAT:         return ggml_hrx_op_unary(ctx, node);
        case GGML_OP_ARANGE:         return ggml_hrx_op_arange(ctx, node);
        case GGML_OP_CONCAT:         return ggml_hrx_op_concat(ctx, node);
        case GGML_OP_PAD:            return ggml_hrx_op_pad(ctx, node);
        case GGML_OP_SCALE:          return ggml_hrx_op_scale(ctx, node);
        case GGML_OP_GLU:            return ggml_hrx_op_glu(ctx, node);
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:           return ggml_hrx_op_cpy(ctx, node);
        case GGML_OP_GET_ROWS:       return ggml_hrx_op_get_rows(ctx, node);
        case GGML_OP_SET_ROWS:       return ggml_hrx_op_set_rows(ctx, node);
        case GGML_OP_RMS_NORM:       return ggml_hrx_op_rms_norm(ctx, node);
        case GGML_OP_ROPE:           return ggml_hrx_op_rope(ctx, node);
        case GGML_OP_SOFT_MAX:       return ggml_hrx_op_soft_max(ctx, node);
        case GGML_OP_SUM_ROWS:       return ggml_hrx_op_sum_rows(ctx, node);
        case GGML_OP_ARGSORT:        return ggml_hrx_op_argsort(ctx, node);
        case GGML_OP_TOP_K:          return ggml_hrx_op_top_k(ctx, node);
        case GGML_OP_ARGMAX:         return ggml_hrx_op_argmax(ctx, node);
        case GGML_OP_CUMSUM:         return ggml_hrx_op_cumsum(ctx, node);
        case GGML_OP_SUM:            return ggml_hrx_op_sum(ctx, node);
        case GGML_OP_TOKEN_MASK:     return ggml_hrx_op_token_mask(ctx, node);
        case GGML_OP_SSM_CONV:       return ggml_hrx_op_ssm_conv(ctx, node);
        case GGML_OP_MUL_MAT:        return ggml_hrx_op_mul_mat(ctx, node);
        case GGML_OP_MUL_MAT_ID:     return ggml_hrx_op_mul_mat_id(ctx, node);
        case GGML_OP_FLASH_ATTN_EXT: return ggml_hrx_op_flash_attn_ext(ctx, node);
        default:                     return false;
    }
}

// GGML_HRX_PROFILE_OPS: a timed event after each node. The time between two events is the node's kernels and the gap
// before them. The totals print when the backend is freed.
struct ggml_hrx_op_profile {
    std::vector<hrx_event_t>                           events;
    std::vector<std::string>                           keys;
    std::map<std::string, std::pair<double, int64_t>> totals; // ms, count
    double                                             graph_ms = 0.0;
    int64_t                                            graphs   = 0;
};

static std::string ggml_hrx_profile_key(const ggml_tensor * node) {
    std::string key = ggml_op_desc(node);
    if (node->src[0] != nullptr) {
        key += std::string(" ") + ggml_type_name(node->src[0]->type);
    }
    key += " [" + std::to_string(node->ne[0]) + "," + std::to_string(node->ne[1]) + "," + std::to_string(node->ne[2]) + "]";
    return key;
}

static bool ggml_hrx_profile_record(ggml_backend_hrx_context * ctx, size_t slot) {
    ggml_hrx_op_profile * p = ctx->prof;
    while (p->events.size() <= slot) {
        hrx_event_t event = nullptr;
        if (!GGML_HRX_CHECK(hrx_event_create(ctx->device->device, HRX_EVENT_FLAG_NONE, &event))) {
            return false;
        }
        p->events.push_back(event);
    }
    return GGML_HRX_CHECK(hrx_event_record(p->events[slot], ctx->stream));
}

static void ggml_hrx_profile_collect(ggml_backend_hrx_context * ctx, size_t n) {
    ggml_hrx_op_profile * p = ctx->prof;
    if (n == 0 || !GGML_HRX_CHECK(hrx_stream_synchronize(ctx->stream))) {
        return;
    }
    for (size_t k = 1; k <= n; ++k) {
        float ms = 0.0f;
        if (GGML_HRX_CHECK(hrx_event_elapsed_time(p->events[k - 1], p->events[k], &ms))) {
            auto & t = p->totals[p->keys[k - 1]];
            t.first += ms;
            t.second += 1;
            p->graph_ms += ms;
        }
    }
    p->graphs += 1;
}

static void ggml_hrx_profile_print(ggml_backend_hrx_context * ctx) {
    ggml_hrx_op_profile * p = ctx->prof;
    std::vector<std::pair<std::string, std::pair<double, int64_t>>> rows(p->totals.begin(), p->totals.end());
    std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) { return a.second.first > b.second.first; });
    fprintf(stderr, "ggml_hrx: op profile: %" PRId64 " graphs, %.3f ms device time\n", p->graphs, p->graph_ms);
    for (const auto & r : rows) {
        fprintf(stderr, "ggml_hrx: %10.3f ms %5.1f%% %8" PRId64 " x %9.2f us  %s\n", r.second.first,
                100.0 * r.second.first / std::max(p->graph_ms, 1e-9), r.second.second, 1000.0 * r.second.first / r.second.second,
                r.first.c_str());
    }
    for (hrx_event_t event : p->events) {
        hrx_event_release(event);
    }
    delete p;
    ctx->prof = nullptr;
}

// capture: the stream records the dispatches into a graph. Then do not flush.
static bool ggml_hrx_run_nodes(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph, bool capture) {
    const ggml_hrx_tune * tune      = ctx->device->tune;
    int                   n_pending = 0;
    // The GPU starts work while the host records the next nodes. A short first interval starts a small graph sooner.
    int                   n_submit  = tune->nodes_first_submit;
    size_t                n_timed   = 0;

    if (ggml_hrx_npu_plan(ctx, cgraph) && capture) {
        ggml_hrx_npu_record_begin(ctx);
    }
    ggml_hrx_fuse_plan(ctx, cgraph);
    ggml_hrx_autofuse_plan(ctx, cgraph);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || ggml_op_is_empty(node->op) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            ggml_hrx_fuse_absorbed(ctx, i)) {
            continue;
        }
        if (ctx->prof != nullptr && n_timed == 0 && !ggml_hrx_profile_record(ctx, 0)) {
            return false;
        }
        ctx->cur_node = node;
        const char * fused = nullptr;
        const char * auto_fused = nullptr;
        bool         npu_ok = false;
        if (ggml_hrx_npu_gate(ctx, cgraph, i, &npu_ok)) {
            if (!npu_ok) {
                return false;
            }
            ++i;
            continue;
        }
        if (ggml_hrx_npu_run(ctx, i, &npu_ok)) {
            if (!npu_ok) {
                GGML_LOG_ERROR("%s: failed to run node %d (%s, %s) on the NPU\n", __func__, i, ggml_op_desc(node), node->name);
                return false;
            }
            continue;
        }
        int          last  = ggml_hrx_autofuse_run(ctx, i, &auto_fused);
        if (last < 0) {
            last = ggml_hrx_fuse(ctx, cgraph, i, &fused);
            ctx->dispatch_writes = nullptr;
        }
        if (last < 0 && !ggml_hrx_compute_node(ctx, node)) {
            GGML_LOG_ERROR("%s: failed to run node %d (%s, %s)\n", __func__, i, ggml_op_desc(node), node->name);
            return false;
        }
        if (last >= 0) {
            i = last;
        }
        if (ctx->prof != nullptr) {
            ctx->prof->keys.resize(std::max(ctx->prof->keys.size(), n_timed + 1));
            ctx->prof->keys[n_timed] = auto_fused != nullptr ? std::string(auto_fused) + " " + ggml_hrx_profile_key(cgraph->nodes[i]) :
                                       fused != nullptr      ? std::string("FUSED ") + fused + " " + ggml_hrx_profile_key(cgraph->nodes[i]) :
                                                               ggml_hrx_profile_key(node);
            if (!ggml_hrx_profile_record(ctx, ++n_timed)) {
                return false;
            }
        }
        if (!capture && ++n_pending >= n_submit) {
            n_pending = 0;
            n_submit  = std::min(2 * n_submit, tune->nodes_per_submit);
            if (!GGML_HRX_CHECK(hrx_stream_flush(ctx->stream))) {
                return false;
            }
        }
    }
    if (!ggml_hrx_npu_finish(ctx)) {
        return false;
    }
    if (ctx->prof != nullptr) {
        ggml_hrx_profile_collect(ctx, n_timed);
    }
    return true;
}

static ggml_backend_hrx_context::node_props ggml_hrx_node_props(const ggml_tensor * node) {
    ggml_backend_hrx_context::node_props p;
    std::memset(&p, 0, sizeof(p));
    std::memcpy(&p.node, node, sizeof(ggml_tensor));
    // All HRX buffers have the same fake base address. Thus keep the buffer of each tensor too.
    p.node.buffer = ggml_hrx_tensor_buffer(node);
    // Buffer wrapper addresses can be reused while recordings still retain the old HRX allocation.
    p.buffer_id = ggml_hrx_tensor_on_device(node) ? static_cast<ggml_hrx_buffer_context *>(p.node.buffer->context)->id : 0;
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        const ggml_tensor * s = node->src[j];
        if (s == nullptr) {
            continue;
        }
        p.src_data[j] = s->data;
        p.src_buf[j]  = ggml_hrx_tensor_buffer(s);
        p.src_buffer_id[j] = ggml_hrx_tensor_on_device(s) ? static_cast<ggml_hrx_buffer_context *>(p.src_buf[j]->context)->id : 0;
        std::memcpy(p.src_ne[j], s->ne, sizeof(s->ne));
        std::memcpy(p.src_nb[j], s->nb, sizeof(s->nb));
        p.src_type[j] = s->type;
    }
    return p;
}

// Returns true if the graph has the same nodes, shapes and addresses as the last time it ran.
static bool ggml_hrx_graph_unchanged(ggml_backend_hrx_context::graph_cache & g, const ggml_cgraph * cgraph) {
    if (cgraph->uid != 0 && cgraph->uid == g.uid && (int) g.props.size() == cgraph->n_nodes) {
        return true;
    }
    g.uid     = cgraph->uid;
    bool same = (int) g.props.size() == cgraph->n_nodes;
    g.props.resize(cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_backend_hrx_context::node_props p = ggml_hrx_node_props(cgraph->nodes[i]);
        if (!same || std::memcmp(&g.props[i], &p, sizeof(p)) != 0) {
            g.props[i] = p;
            same       = false;
        }
    }
    return same;
}

// Records the graph into g.exec. The caller launches it.
// libhrx 0.2.0 has no stream capture. Thus the dispatches are collected and then added as kernel nodes to an explicit graph.
static bool ggml_hrx_graph_record(ggml_backend_hrx_context * ctx, ggml_backend_hrx_context::graph_cache & g, const ggml_cgraph * cgraph) {
    hrx_graph_t graph = nullptr;
    if (!GGML_HRX_CHECK(hrx_graph_create(ctx->device->device, 0, &graph))) {
        return false;
    }
    ctx->rec      = ggml_hrx_graph_rec_init();
    const bool ok = ggml_hrx_run_nodes(ctx, cgraph, true) && ggml_hrx_graph_rec_build(ctx->rec, graph);
    ggml_hrx_graph_rec_free(ctx->rec);
    ctx->rec = nullptr;
    ggml_hrx_npu_replay_free(g.npu);
    g.npu = ggml_hrx_npu_record_end(ctx);
    if (!ok) {
        hrx_graph_release(graph);
        return false;
    }
    const bool recorded = g.exec == nullptr ?
        GGML_HRX_CHECK(hrx_graph_instantiate(graph, ctx->device->stream_queue, 0, &g.exec)) :
        GGML_HRX_CHECK(hrx_graph_exec_update(g.exec, graph));
    hrx_graph_release(graph);
    g.exec_valid = recorded;
    return recorded;
}

static enum ggml_status ggml_backend_hrx_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_hrx_context *) backend->context;
    ctx->n_graphs_seen += 1;

    if (!ggml_hrx_wait_uploads(ctx)) {
        return GGML_STATUS_FAILED;
    }

    bool              replay    = false;
    if (ctx->device->npu && ctx->weight_version != ctx->device->weight_version.load()) {
        for (auto & entry : ctx->graphs) {
            entry.second.exec_valid = false;
        }
        ctx->weight_version = ctx->device->weight_version.load();
    }
    if (ctx->prof == nullptr && !ctx->graphs_failed && cgraph->n_nodes > 0) {
        // A small cache. Graphs of other shapes take new keys, so drop all of them when it fills.
        if (ctx->graphs.size() >= 16 && ctx->graphs.find(cgraph->nodes[0]) == ctx->graphs.end()) {
            for (auto & kv : ctx->graphs) {
                if (kv.second.exec != nullptr) {
                    hrx_graph_exec_release(kv.second.exec);
                }
                ggml_hrx_npu_replay_free(kv.second.npu);
            }
            ctx->graphs.clear();
        }
        auto & g = ctx->graphs[cgraph->nodes[0]];
        // A changed graph runs directly. It is recorded the next time it runs unchanged, so a graph that changes
        // for each call (for example a growing prompt) does not pay for a recording.
        if (ggml_hrx_graph_unchanged(g, cgraph)) {
            ctx->n_replayed += g.exec_valid ? 1 : 0;
            ctx->n_recorded += g.exec_valid ? 0 : 1;
            replay = g.exec_valid || ggml_hrx_graph_record(ctx, g, cgraph);
            ctx->graphs_failed = !replay;
            if (replay && ((g.npu != nullptr && !ggml_hrx_npu_replay_launch(ctx, g.npu)) ||
                           !GGML_HRX_CHECK(hrx_graph_exec_launch(g.exec, ctx->stream)))) {
                return GGML_STATUS_FAILED;
            }
        } else {
            g.exec_valid = false;
        }
    }

    ctx->n_direct += replay ? 0 : 1;
    if (!replay && !ggml_hrx_run_nodes(ctx, cgraph, false)) {
        return GGML_STATUS_FAILED;
    }

    // synchronize waits for the graph.
    if (!GGML_HRX_CHECK(hrx_stream_flush(ctx->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_hrx_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_hrx_mul_mat_group_order(((ggml_backend_hrx_context *) backend->context)->device, cgraph);
    ggml_hrx_npu_graph_order(((ggml_backend_hrx_context *) backend->context)->device, cgraph);
    ggml_hrx_flash_attn_alloc_deps(cgraph, params);
}

static const ggml_backend_i ggml_backend_hrx_interface = {
    /* .get_name            = */ ggml_backend_hrx_get_name,
    /* .free                = */ ggml_backend_hrx_free,
    /* .set_tensor_async    = */ ggml_backend_hrx_set_tensor_async,
    /* .get_tensor_async    = */ ggml_backend_hrx_get_tensor_async,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ ggml_backend_hrx_synchronize,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ ggml_backend_hrx_graph_compute,
    /* .event_record        = */ ggml_backend_hrx_event_record,
    /* .event_wait          = */ ggml_backend_hrx_event_wait,
    /* .graph_optimize      = */ ggml_backend_hrx_graph_optimize,
};

static ggml_hrx_device_context * ggml_hrx_dev_ctx(ggml_backend_dev_t dev) {
    return (ggml_hrx_device_context *) dev->context;
}

static const char * ggml_backend_hrx_device_get_name(ggml_backend_dev_t dev) {
    return ggml_hrx_dev_ctx(dev)->name.c_str();
}

static const char * ggml_backend_hrx_device_get_description(ggml_backend_dev_t dev) {
    return ggml_hrx_dev_ctx(dev)->description.c_str();
}

static void ggml_backend_hrx_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // HRX does not report free memory yet.
    *free  = ggml_hrx_dev_ctx(dev)->memory_total;
    *total = ggml_hrx_dev_ctx(dev)->memory_total;
}

static enum ggml_backend_dev_type ggml_backend_hrx_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_hrx_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hrx_device_get_name(dev);
    props->description = ggml_backend_hrx_device_get_description(dev);
    props->type        = ggml_backend_hrx_device_get_type(dev);
    props->device_id   = nullptr;
    ggml_backend_hrx_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ true,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ true,
        /* .mmap_support         = */ true,
    };
}

static ggml_backend_t ggml_backend_hrx_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_hrx_device_context * dev_ctx = ggml_hrx_dev_ctx(dev);

    hrx_stream_t stream = nullptr;
    if (!GGML_HRX_CHECK(hrx_stream_create(dev_ctx->stream_queue, 0, &stream))) {
        return nullptr;
    }
    auto * ctx   = new ggml_backend_hrx_context;
    ctx->device  = dev_ctx;
    ctx->stream  = stream;
    ctx->name    = dev_ctx->name;
    if (params && strcmp(params, "profile") == 0) {
        ctx->prof = new ggml_hrx_op_profile;
    }
    ctx->trace = params && strcmp(params, "trace") == 0;
    return new ggml_backend{ ggml_backend_hrx_guid(), ggml_backend_hrx_interface, dev, ctx };
}

static ggml_backend_buffer_type_t ggml_backend_hrx_device_get_buffer_type(ggml_backend_dev_t dev) {
    return &ggml_hrx_dev_ctx(dev)->buft;
}

static bool ggml_backend_hrx_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const ggml_hrx_device_context * ctx = ggml_hrx_dev_ctx(dev);
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:      return true;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:            return ggml_hrx_supports_binary(ctx, op);
        case GGML_OP_UNARY:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_CLAMP:
        case GGML_OP_FILL:
        case GGML_OP_REPEAT:         return ggml_hrx_supports_unary(ctx, op);
        case GGML_OP_ARANGE:         return ggml_hrx_supports_arange(ctx, op);
        case GGML_OP_CONCAT:         return ggml_hrx_supports_concat(ctx, op);
        case GGML_OP_PAD:            return ggml_hrx_supports_pad(ctx, op);
        case GGML_OP_SCALE:          return ggml_hrx_supports_scale(ctx, op);
        case GGML_OP_GLU:            return ggml_hrx_supports_glu(ctx, op);
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:           return ggml_hrx_supports_cpy(ctx, op);
        case GGML_OP_GET_ROWS:       return ggml_hrx_supports_get_rows(ctx, op);
        case GGML_OP_SET_ROWS:       return ggml_hrx_supports_set_rows(ctx, op);
        case GGML_OP_RMS_NORM:       return ggml_hrx_supports_rms_norm(ctx, op);
        case GGML_OP_ROPE:           return ggml_hrx_supports_rope(ctx, op);
        case GGML_OP_SOFT_MAX:       return ggml_hrx_supports_soft_max(ctx, op);
        case GGML_OP_SUM_ROWS:       return ggml_hrx_supports_sum_rows(ctx, op);
        case GGML_OP_ARGSORT:        return ggml_hrx_supports_argsort(ctx, op);
        case GGML_OP_TOP_K:          return ggml_hrx_supports_top_k(ctx, op);
        case GGML_OP_ARGMAX:         return ggml_hrx_supports_argmax(ctx, op);
        case GGML_OP_CUMSUM:         return ggml_hrx_supports_cumsum(ctx, op);
        case GGML_OP_SUM:            return ggml_hrx_supports_sum(ctx, op);
        case GGML_OP_TOKEN_MASK:     return ggml_hrx_supports_token_mask(ctx, op);
        case GGML_OP_SSM_CONV:       return ggml_hrx_supports_ssm_conv(ctx, op);
        case GGML_OP_MUL_MAT:        return ggml_hrx_supports_mul_mat(ctx, op);
        case GGML_OP_MUL_MAT_ID:     return ggml_hrx_supports_mul_mat_id(ctx, op);
        case GGML_OP_FLASH_ATTN_EXT: return ggml_hrx_supports_flash_attn_ext(ctx, op);
        default:                     return false;
    }
}

static bool ggml_backend_hrx_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == &ggml_hrx_dev_ctx(dev)->buft;
}

static ggml_backend_event_t ggml_backend_hrx_device_event_new(ggml_backend_dev_t dev) {
    hrx_event_t event = nullptr;
    if (!GGML_HRX_CHECK(hrx_event_create(ggml_hrx_dev_ctx(dev)->device, HRX_EVENT_FLAG_DISABLE_TIMING, &event))) {
        return nullptr;
    }
    return new ggml_backend_event{ dev, event };
}

static void ggml_backend_hrx_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    hrx_event_release((hrx_event_t) event->context);
    delete event;
}

static void ggml_backend_hrx_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    GGML_HRX_CHECK(hrx_event_synchronize((hrx_event_t) event->context));
}

static const ggml_backend_device_i ggml_backend_hrx_device_interface = {
    /* .get_name             = */ ggml_backend_hrx_device_get_name,
    /* .get_description      = */ ggml_backend_hrx_device_get_description,
    /* .get_memory           = */ ggml_backend_hrx_device_get_memory,
    /* .get_type             = */ ggml_backend_hrx_device_get_type,
    /* .get_props            = */ ggml_backend_hrx_device_get_props,
    /* .init_backend         = */ ggml_backend_hrx_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_hrx_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_hrx_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hrx_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ ggml_backend_hrx_device_event_new,
    /* .event_free           = */ ggml_backend_hrx_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hrx_device_event_synchronize,
};

struct ggml_hrx_reg_context {
    hrx_runtime_t                   runtime = nullptr;
    std::vector<hrx_device_group_t> groups;

    std::vector<std::unique_ptr<ggml_hrx_device_context>> contexts;
    std::vector<ggml_backend_device>                      devices;

    ~ggml_hrx_reg_context() {
        for (auto & dev : contexts) {
            ggml_hrx_npu * npu = dev->npu;
            dev->npu = nullptr;
            ggml_hrx_npu_free(npu);
            ggml_backend_buffer_free(dev->relative_pos_scratch);
            dev->kernels.clear();
            ggml_hrx_loom_jit_free(dev->jit);
            if (dev->buffer_stream != nullptr) {
                hrx_stream_release(dev->buffer_stream);
            }
            hrx_device_release(dev->device);
        }
        for (hrx_device_group_t group : groups) {
            hrx_device_group_release(group);
        }
        if (runtime != nullptr) {
            hrx_runtime_release(runtime);
        }
    }
};

static std::optional<std::string> ggml_hrx_device_string(hrx_device_t device, hrx_device_property_t property) {
    std::vector<char> buffer(64);
    while (buffer.size() <= 4096) {
        hrx_status_t status = hrx_device_get_property(device, property, buffer.data(), buffer.size());
        if (hrx_status_is_ok(status)) {
            return std::string(buffer.data());
        }
        if (hrx_status_code(status) != HRX_STATUS_OUT_OF_RANGE) {
            GGML_HRX_CHECK(status);
            return std::nullopt;
        }
        hrx_status_ignore(status);
        buffer.resize(buffer.size() * 2);
    }
    return std::nullopt;
}

static bool ggml_hrx_find_queue(hrx_device_t device, hrx_queue_role_flags_t roles, uint32_t index, hrx_queue_t * queue) {
    *queue = nullptr;
    const size_t n_families = hrx_device_queue_family_count(device);
    for (size_t i = 0; i < n_families; ++i) {
        hrx_queue_family_info_t family = {};
        if (!GGML_HRX_CHECK(hrx_device_query_queue_family(device, (uint32_t) i, &family))) {
            return false;
        }
        if ((family.roles & roles) == roles && family.provisioned_queue_count > index) {
            *queue = hrx_device_get_queue(device, (uint32_t) i, index);
            if (*queue != nullptr) {
                return true;
            }
        }
    }
    return false;
}

// Returns nullptr if the device is not usable. In that case, it releases all that it took.
static std::unique_ptr<ggml_hrx_device_context> ggml_hrx_init_device(hrx_device_t device, int index) {
    const std::optional<std::string> name = ggml_hrx_device_string(device, HRX_DEVICE_PROPERTY_NAME);
    const std::optional<std::string> arch = ggml_hrx_device_string(device, HRX_DEVICE_PROPERTY_ARCHITECTURE);
    if (!name || !arch) {
        return nullptr;
    }
    const ggml_hrx_arch * info = ggml_hrx_arch_find(arch->c_str());
    if (info == nullptr) {
        GGML_LOG_INFO("ggml_hrx: skipping %s (%s): architecture is not supported\n", name->c_str(), arch->c_str());
        return nullptr;
    }

    auto ctx    = std::make_unique<ggml_hrx_device_context>();
    ctx->device = device;
    ctx->arch   = info;
    ctx->tune   = info->tune;
    ctx->name   = GGML_HRX_NAME + std::to_string(index);
    ctx->index  = index;
    ctx->description = *name + " (" + *arch + ")";

    uint64_t memory = 0;
    if (!GGML_HRX_CHECK(hrx_device_get_property(device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY, &memory, sizeof(memory))) ||
        !GGML_HRX_CHECK(hrx_device_get_property(device, HRX_DEVICE_PROPERTY_COMPUTE_UNITS, &ctx->n_cu, sizeof(ctx->n_cu)))) {
        return nullptr;
    }
    ctx->memory_total = (size_t) memory;

    // Synchronous transfers use a second queue if there is one. Then they do not wait for queued graphs.
    if ((!ggml_hrx_find_queue(device, HRX_QUEUE_ROLE_TRANSFER, 1, &ctx->transfer_queue) &&
         !ggml_hrx_find_queue(device, HRX_QUEUE_ROLE_TRANSFER, 0, &ctx->transfer_queue)) ||
        !ggml_hrx_find_queue(device, HRX_QUEUE_ROLE_DISPATCH | HRX_QUEUE_ROLE_TRANSFER, 0, &ctx->stream_queue)) {
        GGML_LOG_ERROR("ggml_hrx: %s has no usable queues\n", name->c_str());
        return nullptr;
    }
    if (!GGML_HRX_CHECK(hrx_stream_create(ctx->stream_queue, 0, &ctx->buffer_stream))) {
        return nullptr;
    }
    return ctx;
}

static void ggml_hrx_discover(ggml_hrx_reg_context * reg, ggml_backend_reg_t backend_reg, bool with_npu) {
    hrx_runtime_options_t      options   = {};
    const hrx_host_allocator_t allocator = hrx_host_allocator_system();
    if (!GGML_HRX_CHECK(hrx_runtime_create(&options, allocator, &reg->runtime))) {
        return;
    }

    size_t              count = 0;
    hrx_device_info_t * infos = nullptr;
    if (!GGML_HRX_CHECK(hrx_runtime_query_available_devices(reg->runtime, "amdgpu", allocator, &count, &infos))) {
        return;
    }
    std::string npu_uri;
    if (with_npu) {
        size_t              n_npu  = 0;
        hrx_device_info_t * npus   = nullptr;
        if (GGML_HRX_CHECK(hrx_runtime_query_available_devices(reg->runtime, "xdna", allocator, &n_npu, &npus)) && n_npu > 0) {
            npu_uri = std::string(npus[0].uri) + "?columns=8";
        }
        hrx_host_allocator_free(allocator, npus);
        if (npu_uri.empty()) {
            hrx_host_allocator_free(allocator, infos);
            return;
        }
    }

    // One group per device.
    for (size_t i = 0; i < count; ++i) {
        hrx_device_create_params_t params[2] = { { infos[i].uri, HRX_WAIT_FLAG_NONE }, { npu_uri.c_str(), HRX_WAIT_FLAG_NONE } };
        hrx_device_group_t group = nullptr;
        if (!GGML_HRX_CHECK(hrx_device_group_create(reg->runtime, with_npu ? 2 : 1, params, &group))) {
            continue;
        }
        hrx_device_t device = hrx_device_group_get_device(group, 0);
        if (device == nullptr) {
            hrx_device_group_release(group);
            continue;
        }
        hrx_device_retain(device);
        auto ctx = ggml_hrx_init_device(device, (int) reg->contexts.size());
        if (!ctx) {
            hrx_device_release(device);
            hrx_device_group_release(group);
            continue;
        }
        if (with_npu) {
            ctx->name += "+XDNA";
            ctx->npu = ggml_hrx_npu_create(ctx.get(), hrx_device_group_get_device(group, 1));
        }
        reg->groups.push_back(group);
        reg->contexts.push_back(std::move(ctx));
        if (with_npu) {
            break;
        }
    }
    hrx_host_allocator_free(allocator, infos);

    reg->devices.reserve(reg->contexts.size());
    for (auto & ctx : reg->contexts) {
        reg->devices.push_back({ ggml_backend_hrx_device_interface, backend_reg, ctx.get() });
        ctx->buft = { ggml_backend_hrx_buffer_type_interface, &reg->devices.back(), ctx.get() };
    }
}

static const char * ggml_backend_hrx_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_HRX_NAME;
}

static size_t ggml_backend_hrx_reg_get_device_count(ggml_backend_reg_t reg) {
    return ((ggml_hrx_reg_context *) reg->context)->devices.size();
}

static ggml_backend_dev_t ggml_backend_hrx_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * ctx = (ggml_hrx_reg_context *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return &ctx->devices[index];
}

// Position data contains (query position, key array offset) pairs, followed by key positions.
static bool ggml_backend_hrx_set_relative_positions(ggml_tensor * dst, const int32_t * positions, size_t count, uint32_t extent) {
    if (!ggml_hrx_tensor_on_device(dst) || dst->type != GGML_TYPE_I32 || !ggml_is_contiguous(dst) ||
        ggml_n_dims(dst) > 2 || dst->ne[0] <= 0 || dst->ne[0] >= INT32_MAX - 255 || dst->ne[1] <= 1 ||
        count > INT32_MAX / sizeof(int32_t) || extent > INT32_MAX) {
        return false;
    }

    auto * dev = static_cast<ggml_hrx_buffer_context *>(ggml_hrx_tensor_buffer(dst)->context)->dev;
    if (dst->ne[1] > dev->arch->max_workgroups_yz) {
        return false;
    }
    const size_t query_words = 2 * dst->ne[1];
    if (!positions || count < query_words + (size_t) dst->ne[0]) {
        return false;
    }
    const size_t last_key_offset = count - dst->ne[0];
    for (size_t i = 1; i < query_words; i += 2) {
        const size_t offset = positions[i];
        if (offset < query_words || offset > last_key_offset) {
            return false;
        }
    }

    ggml_hrx_kernel_desc desc;
    desc.file   = "flash_attn/relative_positions.loom";
    desc.symbol = "ggml_hrx_relative_positions";
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(dev, desc);
    if (kernel == nullptr) {
        return false;
    }

    const size_t bytes = count * sizeof(int32_t);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> old(nullptr, ggml_backend_buffer_free);
    std::lock_guard<std::mutex> lock(dev->buffer_mutex);
    if (!dev->relative_pos_scratch || ggml_backend_buffer_get_size(dev->relative_pos_scratch) < bytes) {
        const size_t capacity = dev->relative_pos_scratch ? ggml_backend_buffer_get_size(dev->relative_pos_scratch) : 0;
        auto * buffer = ggml_backend_buft_alloc_buffer(&dev->buft, std::max(bytes, 2 * capacity));
        if (buffer == nullptr) {
            return false;
        }
        old.reset(dev->relative_pos_scratch);
        dev->relative_pos_scratch = buffer;
    }

    ggml_tensor src = {};
    src.type   = GGML_TYPE_I32;
    src.ne[0]  = count;
    src.ne[1]  = src.ne[2] = src.ne[3] = 1;
    src.nb[0]  = sizeof(int32_t);
    src.nb[1]  = src.nb[2] = src.nb[3] = bytes;
    src.buffer = dev->relative_pos_scratch;
    src.data   = ggml_backend_buffer_get_base(src.buffer);

    const auto buffer = ggml_hrx_buffer_handle(src.buffer);
    dev->upload_pending = true;
    for (size_t offset = 0; offset < bytes; offset += GGML_HRX_UPDATE_CHUNK) {
        if (!GGML_HRX_CHECK(hrx_stream_update_buffer(dev->buffer_stream, positions + offset / sizeof(int32_t),
                              std::min(GGML_HRX_UPDATE_CHUNK, bytes - offset), buffer, offset))) {
            return false;
        }
    }
    ggml_backend_hrx_context upload;
    upload.device = dev;
    upload.stream = dev->buffer_stream;
    return GGML_HRX_CHECK(hrx_stream_barrier(upload.stream, nullptr)) &&
           ggml_hrx_dispatch(&upload, kernel, { (dst->ne[0] + 255) / 256, dst->ne[1] },
                             { dst->ne[0], (int64_t) count, extent }, { &src, dst }) &&
           GGML_HRX_CHECK(hrx_stream_barrier(upload.stream, nullptr));
}

static ggml_backend_dev_t ggml_backend_hrx_npu_device();

static ggml_backend_feature * ggml_backend_hrx_get_features(ggml_backend_reg_t) {
    static ggml_backend_feature features[] = {
        {"STACKED_DEPTH_MIX", "1"},
        {"DEVICE_TOKEN_EMBEDDINGS", "1"},
        {nullptr, nullptr},
    };
    return features;
}

static void * ggml_backend_hrx_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_hrx_npu_device") == 0) {
        return reinterpret_cast<void *>(ggml_backend_hrx_npu_device);
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return reinterpret_cast<void *>(ggml_backend_hrx_get_features);
    }
    if (strcmp(name, "ggml_backend_set_relative_positions") == 0) {
        return reinterpret_cast<void *>(ggml_backend_hrx_set_relative_positions);
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_hrx_reg_interface = {
    /* .get_name         = */ ggml_backend_hrx_reg_get_name,
    /* .get_device_count = */ ggml_backend_hrx_reg_get_device_count,
    /* .get_device       = */ ggml_backend_hrx_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hrx_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hrx_reg(void) {
    static ggml_hrx_reg_context reg_ctx;
    static ggml_backend_reg     reg = { GGML_BACKEND_API_VERSION, ggml_backend_hrx_reg_interface, &reg_ctx };
    static std::once_flag once;
    std::call_once(once, [] { ggml_hrx_discover(&reg_ctx, &reg, false); });
    return &reg;
}

static ggml_backend_dev_t ggml_backend_hrx_npu_device() {
    static std::mutex mutex;
    static std::unique_ptr<ggml_hrx_reg_context> paired;
    std::lock_guard<std::mutex> lock(mutex);
    if (!paired) {
        auto ctx = std::make_unique<ggml_hrx_reg_context>();
        ggml_hrx_discover(ctx.get(), ggml_backend_hrx_reg(), true);
        if (ctx->devices.empty() || !ctx->contexts.front()->npu) {
            return nullptr;
        }
        paired = std::move(ctx);
    }
    return &paired->devices.front();
}

ggml_backend_t ggml_backend_hrx_init(size_t device) {
    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (device >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, device), nullptr);
}

bool ggml_backend_is_hrx(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_hrx_guid());
}

int ggml_backend_hrx_get_device_count(void) {
    return (int) ggml_backend_reg_dev_count(ggml_backend_hrx_reg());
}

ggml_backend_buffer_type_t ggml_backend_hrx_buffer_type(size_t device) {
    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (device >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, device));
}

GGML_BACKEND_DL_IMPL(ggml_backend_hrx_reg)
