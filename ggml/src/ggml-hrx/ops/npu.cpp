#include "ggml-hrx-ops.h"
#include "ggml-hrx-graph.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "loomc/loomc.h"
#include "loomc/target/amd/xdna.h"

#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>

// A Q8_0 MUL_MAT with a large batch can give a part of its rows (tokens) to the NPU. The NPU runs an int8 GEMM
// (kernels/npu/gemm_i8_k*_g*.loom, all 32 AIE2P cores) with int8 weights and one scale for each weight row, derived from
// the Q8_0 weights at first use, and int8 activations with one scale for each token. The GPU quantizes the NPU rows
// (kernels/npu/npu_i8.loom), computes the other rows as usual and later scales the int32 sums into the output.
//
// Two kinds of nodes use the NPU:
// - ahead: the first use of the output is far away (graph_order moves the node next to its input). The NPU takes all
//   rows it can and runs while the GPU computes the nodes in between.
// - split: the output is used soon. The NPU takes a part of the rows and runs while the GPU computes the rest.
// The NPU runs one GEMM at a time, so a node uses the NPU only if no other NPU result is in flight.
// A recorded graph with NPU work is a list of HRX graphs with NPU dispatches and waits between them.
// GGML_HRX_NPU=1 turns it on. On Inkling 2 Mini prefill it gave no measured gain (2026-11-24): the GPU loses ~7 % of its
// speed while the NPU streams DRAM, and the quantize and scale passes cost most of the GPU time that the NPU saves.

// Strix Halo, NPU alone (npu_gemm, 2026-11-24): M=2048 [2304,3072] 2.09 ms, [3072,2304] 1.98 ms, M=1024 1.29 / 1.22 ms;
// the GPU needs 0.95 and 1.92 ms for M=2048. The split share balances the two devices on [3072,2304].
static constexpr int64_t GGML_HRX_NPU_MIN_ROWS   = 512;            // smaller batches stay on the GPU
static constexpr int64_t GGML_HRX_NPU_MIN_WEIGHT = 2304 * 2304;    // smaller weights stay on the GPU
static constexpr float   GGML_HRX_NPU_AHEAD      = 1.0f;           // share of the rows for an ahead node
static constexpr float   GGML_HRX_NPU_SPLIT      = 0.5f;           // share of the rows for a split node
static constexpr int     GGML_HRX_NPU_AHEAD_DIST = 8;              // nodes between a node and its first use for ahead
static constexpr int     GGML_HRX_NPU_SLOTS      = 2;              // exchange buffers in flight
static constexpr int64_t GGML_HRX_NPU_MAX_N      = 16384;          // the int8 copy of a larger weight (an output head) costs too much memory

struct ggml_hrx_npu_weight {
    ggml_backend_buffer_t buf    = nullptr; // panel (N x K int8) then the scales (N f32)
    ggml_tensor           panel  = {};
    ggml_tensor           scales = {};
};

struct ggml_hrx_npu_slot {
    ggml_backend_buffer_t buf  = nullptr; // A, then C, then the row scales
    size_t                size = 0;
};

struct ggml_hrx_npu {
    ggml_hrx_device_context * dev    = nullptr;
    hrx_device_t              device = nullptr;
    hrx_queue_t               queue  = nullptr;
    hrx_semaphore_t           sem    = nullptr;
    uint64_t                  value  = 0;
    hrx_mem_pool_t            pool   = nullptr;
    std::string               arch;
    float                     ahead  = GGML_HRX_NPU_AHEAD;
    float                     split  = GGML_HRX_NPU_SPLIT;

    std::mutex                                       mutex;
    std::map<std::string, std::pair<hrx_executable_t, uint32_t>> kernels; // by kernel config, nullptr if the build failed
    std::map<std::tuple<hrx_buffer_t, size_t, int64_t, int64_t, uint64_t>, ggml_hrx_npu_weight> weights; // by buffer, offset, K, N
    size_t                                           active_contexts = 0;
    std::vector<hrx_buffer_t>                        allocations;
};

struct ggml_hrx_npu_job {
    const ggml_tensor * node   = nullptr;
    hrx_buffer_t        buffer = nullptr; // node output, NPU rows
    size_t              offset = 0;
    size_t              length = 0;
    uint64_t            value  = 0;
    int                 slot   = 0;
    int64_t             rows   = 0;
    const ggml_hrx_npu_weight * weight = nullptr;
};

// A recorded graph with NPU work: HRX graphs that run in order, each followed by an NPU dispatch or an NPU wait.
struct ggml_hrx_npu_replay {
    struct op {
        hrx_executable_t exe     = nullptr;
        uint32_t         ordinal = 0;
        hrx_buffer_ref_t bindings[3] = {};
    };
    struct part {
        hrx_graph_exec_t exec = nullptr; // nullptr if no dispatch came before the action
        bool             wait = false;   // wait for op, else dispatch op
        int              op   = 0;
    };
    std::vector<op>       ops;
    std::vector<part>     parts;
    std::vector<uint64_t> values; // NPU semaphore value of each op in the current launch
};

struct ggml_hrx_npu_state {
    ggml_hrx_npu *                owner = nullptr;
    ggml_hrx_npu_slot             slots[GGML_HRX_NPU_SLOTS] = {};
    int                           next_slot = 0;
    std::vector<ggml_backend_buffer_t> retired;
    std::vector<float>            share; // by node index, 0 for the GPU only
    std::vector<ggml_hrx_npu_job> jobs;
    bool                          completing = false;
    ggml_hrx_npu_replay *         replay     = nullptr; // the recording that takes the NPU work while a graph is recorded
};

static float ggml_hrx_npu_env_float(const char * name, float def) {
    const char * v = getenv(name);
    if (v == nullptr) {
        return def;
    }
    char * end;
    const float f = std::strtof(v, &end);
    return end != v && *end == '\0' && std::isfinite(f) ? std::clamp(f, 0.0f, 1.0f) : def;
}

static hrx_queue_t ggml_hrx_npu_find_queue(hrx_device_t device, hrx_queue_role_flags_t roles) {
    for (size_t i = 0; i < hrx_device_queue_family_count(device); ++i) {
        hrx_queue_family_info_t family = {};
        if (!GGML_HRX_CHECK(hrx_device_query_queue_family(device, (uint32_t) i, &family))) {
            return nullptr;
        }
        if ((family.roles & roles) == roles && family.provisioned_queue_count > 0) {
            return hrx_device_get_queue(device, (uint32_t) i, 0);
        }
    }
    return nullptr;
}

ggml_hrx_npu * ggml_hrx_npu_create(ggml_hrx_device_context * dev, hrx_device_t npu_device) {
    auto npu    = std::make_unique<ggml_hrx_npu>();
    npu->dev    = dev;
    npu->device = npu_device;
    npu->queue  = ggml_hrx_npu_find_queue(npu_device, HRX_QUEUE_ROLE_DISPATCH | HRX_QUEUE_ROLE_TRANSFER);
    if (npu->queue == nullptr) {
        GGML_LOG_INFO("ggml_hrx: the NPU has no dispatch queue\n");
        return nullptr;
    }
    char arch[64] = {};
    if (!GGML_HRX_CHECK(hrx_device_get_property(npu_device, HRX_DEVICE_PROPERTY_ARCHITECTURE, arch, sizeof(arch))) ||
        !GGML_HRX_CHECK(hrx_semaphore_create(npu_device, 0, &npu->sem))) {
        return nullptr;
    }
    npu->arch = arch;

    // Buffers of this pool are storage for the GPU dispatch queue and for the NPU.
    hrx_pool_family_access_t families[2] = {};
    families[0].device         = dev->device;
    families[0].family_ordinal = hrx_queue_get_family_ordinal(dev->stream_queue);
    families[0].usage          = HRX_BUFFER_USAGE_STORAGE | HRX_BUFFER_USAGE_TRANSFER;
    families[1].device         = npu_device;
    families[1].family_ordinal = hrx_queue_get_family_ordinal(npu->queue);
    families[1].usage          = HRX_BUFFER_USAGE_STORAGE | HRX_BUFFER_USAGE_TRANSFER;
    hrx_pool_scope_t scope = {};
    scope.family_count     = 2;
    scope.families         = families;
    if (!GGML_HRX_CHECK(hrx_slab_pool_create(hrx_device_get_device_group(dev->device), scope, nullptr, &npu->pool))) {
        hrx_semaphore_release(npu->sem);
        return nullptr;
    }
    npu->ahead = ggml_hrx_npu_env_float("GGML_HRX_NPU_AHEAD", GGML_HRX_NPU_AHEAD);
    npu->split = ggml_hrx_npu_env_float("GGML_HRX_NPU_SPLIT", GGML_HRX_NPU_SPLIT);
    GGML_LOG_INFO("ggml_hrx: %s uses the NPU (%s) for large Q8_0 MUL_MATs\n", dev->name.c_str(), arch);
    return npu.release();
}

static void ggml_hrx_npu_release_buffers(ggml_hrx_npu * npu) {
    if (npu->allocations.empty()) {
        return;
    }
    uint64_t done = npu->value + 1;
    hrx_semaphore_list_t waits = { &npu->sem, &npu->value, 1 };
    hrx_semaphore_list_t signals = { &npu->sem, &done, 1 };
    if (!GGML_HRX_CHECK(hrx_queue_dealloca(npu->queue, &waits, &signals, npu->allocations.size(), npu->allocations.data()))) {
        return;
    }
    npu->value = done;
    GGML_HRX_CHECK(hrx_semaphore_wait(npu->sem, done, UINT64_MAX));
    for (auto & kv : npu->weights) {
        ggml_backend_buffer_free(kv.second.buf);
    }
    npu->weights.clear();
    for (hrx_buffer_t b : npu->allocations) {
        hrx_buffer_release(b);
    }
    npu->allocations.clear();
}

void ggml_hrx_npu_free(ggml_hrx_npu * npu) {
    if (npu == nullptr) {
        return;
    }
    ggml_hrx_npu_release_buffers(npu);
    for (auto & kv : npu->kernels) {
        if (kv.second.first != nullptr) {
            hrx_executable_release(kv.second.first);
        }
    }
    hrx_mem_pool_release(npu->pool);
    hrx_semaphore_release(npu->sem);
    delete npu;
}

void ggml_hrx_npu_forget(ggml_hrx_npu * npu, hrx_buffer_t buffer) {
    std::vector<ggml_backend_buffer_t> dead;
    uint64_t value;
    {
        std::lock_guard<std::mutex> lock(npu->mutex);
        value = npu->value;
        for (auto it = npu->weights.begin(); it != npu->weights.end();) {
            if (std::get<0>(it->first) == buffer) {
                dead.push_back(it->second.buf);
                it = npu->weights.erase(it);
            } else {
                ++it;
            }
        }
    }
    // NPU GEMMs in flight can still read these panels.
    if (!dead.empty()) {
        GGML_HRX_CHECK(hrx_semaphore_wait(npu->sem, value, UINT64_MAX));
    }
    for (ggml_backend_buffer_t b : dead) {
        ggml_backend_buffer_free(b);
    }
}

// A buffer from the shared pool, wrapped for ggml_hrx_dispatch.
static ggml_backend_buffer_t ggml_hrx_npu_alloc(ggml_hrx_npu * npu, size_t size) {
    hrx_pool_reservation_request_t request = { size, 4096 };
    hrx_buffer_t                   buffer  = nullptr;
    const uint64_t                 value   = npu->value + 1;
    uint64_t                       wait    = npu->value;
    hrx_semaphore_list_t           waits   = { &npu->sem, &wait, 1 };
    uint64_t                       signal  = value;
    hrx_semaphore_list_t           signals = { &npu->sem, &signal, 1 };
    if (!GGML_HRX_CHECK(hrx_queue_alloca(npu->queue, &waits, &signals, npu->pool, 1, &request, &buffer))) {
        return nullptr;
    }
    npu->value = value;
    if (!GGML_HRX_CHECK(hrx_semaphore_wait(npu->sem, value, UINT64_MAX))) {
        hrx_buffer_release(buffer);
        return nullptr;
    }
    npu->allocations.push_back(buffer);
    hrx_buffer_retain(buffer);
    return ggml_hrx_wrap_buffer(npu->dev, buffer, size);
}

static ggml_tensor ggml_hrx_npu_bytes(ggml_backend_buffer_t buf, size_t offset, size_t size) {
    ggml_tensor t = {};
    t.type        = GGML_TYPE_I8;
    t.buffer      = buf;
    t.data        = (uint8_t *) ggml_backend_buffer_get_base(buf) + offset;
    t.ne[0]       = (int64_t) size;
    t.ne[1] = t.ne[2] = t.ne[3] = 1;
    t.nb[0]       = 1;
    t.nb[1] = t.nb[2] = t.nb[3] = size;
    return t;
}

// int8 weights with one scale for each row (max |w| / 127), in the B order of the NPU kernel.
static const ggml_hrx_npu_weight * ggml_hrx_npu_weight_get(ggml_hrx_npu * npu, hrx_stream_t stream, const ggml_tensor * w) {
    const auto key = std::make_tuple(ggml_hrx_buffer_handle(w->view_src != nullptr ? w->view_src->buffer : w->buffer), ggml_hrx_offset(w),
                                     w->ne[0], w->ne[1], ggml_hrx_weight_version(w));
    auto         it   = npu->weights.find(key);
    if (it != npu->weights.end()) {
        return &it->second;
    }
    const int64_t K = w->ne[0], N = w->ne[1];
    if (!GGML_HRX_CHECK(hrx_stream_synchronize(stream))) {
        return nullptr;
    }
    std::vector<uint8_t> q8(ggml_nbytes(w));
    ggml_backend_tensor_get(w, q8.data(), 0, q8.size());

    std::vector<int8_t> panel((size_t) (N * K), 0);
    std::vector<float>  scales((size_t) N, 0.0f);
    std::vector<float>  row((size_t) K);
    const auto * blocks = (const block_q8_0 *) q8.data();
    for (int64_t n = 0; n < N; ++n) {
        float amax = 0.0f;
        for (int64_t b = 0; b < K / 32; ++b) {
            const block_q8_0 & blk = blocks[n * (K / 32) + b];
            const float d = GGML_FP16_TO_FP32(blk.d);
            for (int j = 0; j < 32; ++j) {
                row[b * 32 + j] = d * blk.qs[j];
                amax = std::max(amax, std::fabs(row[b * 32 + j]));
            }
        }
        const float scale = amax / 127.0f;
        const float inv   = amax > 0.0f ? 127.0f / amax : 0.0f;
        scales[n]         = scale;
        for (int64_t k = 0; k < K; ++k) {
            const float q = std::nearbyint(row[k] * inv);
            const size_t at = (size_t) ((((n / 64) % 2) * (N / 128) + n / 128) * 64 * K + (k / 128) * 8192 + ((n % 64) / 8) * 1024 +
                                        ((k % 128) / 8) * 64 + (k % 8) * 8 + n % 8);
            panel[at] = (int8_t) std::max(-127.0f, std::min(127.0f, q));
        }
    }

    const size_t panel_bytes = (size_t) (N * K);
    const size_t total       = GGML_PAD(panel_bytes, 256) + (size_t) N * sizeof(float);
    ggml_backend_buffer_t buf = ggml_hrx_npu_alloc(npu, total);
    if (buf == nullptr) {
        return nullptr;
    }
    hrx_buffer_t hb = npu->allocations.back();
    if (!GGML_HRX_CHECK(hrx_synchronous_h2d(npu->queue, panel.data(), hb, 0, panel_bytes)) ||
        !GGML_HRX_CHECK(hrx_synchronous_h2d(npu->queue, scales.data(), hb, GGML_PAD(panel_bytes, 256), (size_t) N * sizeof(float)))) {
        ggml_backend_buffer_free(buf);
        return nullptr;
    }
    ggml_hrx_npu_weight & e = npu->weights[key];
    e.buf    = buf;
    e.panel  = ggml_hrx_npu_bytes(buf, 0, panel_bytes);
    e.scales = ggml_hrx_npu_bytes(buf, GGML_PAD(panel_bytes, 256), (size_t) N * sizeof(float));
    return &e;
}

static bool ggml_hrx_loom_ok(loomc_status_t status, const char * what) {
    if (loomc_status_is_ok(status)) {
        return true;
    }
    char              text[1024] = {};
    loomc_host_size_t length     = 0;
    loomc_status_format(status, sizeof(text), text, &length);
    GGML_LOG_ERROR("ggml_hrx: NPU kernel %s failed: %s\n", what, text);
    loomc_status_free(status);
    return false;
}

static bool ggml_hrx_loom_result_ok(const loomc_result_t * result, const char * what) {
    if (result != nullptr && loomc_result_succeeded(result)) {
        return true;
    }
    GGML_LOG_ERROR("ggml_hrx: NPU kernel %s failed\n", what);
    for (loomc_host_size_t i = 0; result != nullptr && i < loomc_result_diagnostic_count(result); ++i) {
        const loomc_diagnostic_t * d = loomc_result_diagnostic_at(result, i);
        GGML_LOG_ERROR("  %.*s: %.*s @ %u:%u\n", (int) d->code.size, d->code.data, (int) d->message.size, d->message.data,
                       (unsigned) d->range.start_line, (unsigned) d->range.start_column);
    }
    return false;
}

// The NPU GEMM for K, with 1024 (groups 1) or 512 (groups 2) rows in each pass. nullptr if there is none for K.
static const ggml_hrx_kernel_source * ggml_hrx_npu_source(int64_t k, int groups) {
    const std::string path = "npu/gemm_i8_k" + std::to_string(k) + "_g" + std::to_string(groups) + ".loom";
    for (size_t i = 0; i < ggml_hrx_kernel_source_count; ++i) {
        if (path == ggml_hrx_kernel_sources[i].path) {
            return &ggml_hrx_kernel_sources[i];
        }
    }
    return nullptr;
}

// Compiles the NPU GEMM for one shape. The NPU image ABI has no scalars, thus each shape is a config.
static std::vector<uint8_t> ggml_hrx_npu_compile(const ggml_hrx_npu * npu, const ggml_hrx_kernel_source * src, int64_t tsteps,
                                                 int64_t passes) {
    std::vector<uint8_t> out;
    const loomc_allocator_t      alloc   = loomc_allocator_system();
    loomc_target_environment_t * env     = nullptr;
    loomc_context_t *            context = nullptr;
    loomc_workspace_t *          ws      = nullptr;
    loomc_target_profile_t *     profile = nullptr;
    loomc_compiler_t *           comp    = nullptr;
    loomc_source_t *             source  = nullptr;
    loomc_module_t *             module  = nullptr;
    loomc_result_t *             result  = nullptr;

    bool ok = ggml_hrx_loom_ok(loomc_target_environment_create_xdna(alloc, &env), "environment");
    loomc_context_target_options_t target_options = {};
    target_options.type               = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target_options.structure_size     = sizeof(target_options);
    target_options.target_environment = env;
    loomc_context_options_t context_options = {};
    context_options.type           = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS;
    context_options.structure_size = sizeof(context_options);
    context_options.next           = &target_options;
    ok = ok && ggml_hrx_loom_ok(loomc_context_create(&context_options, alloc, &context), "context");
    ok = ok && ggml_hrx_loom_ok(loomc_workspace_create(nullptr, alloc, &ws), "workspace");
    ok = ok && ggml_hrx_loom_ok(loomc_target_profile_create_xdna(env, loomc_make_cstring_view(npu->arch.c_str()), alloc, &profile), "profile");
    ok = ok && ggml_hrx_loom_ok(loomc_compiler_create(context, nullptr, alloc, &comp), "compiler");
    if (ok) {
        loomc_source_options_t so = {};
        so.type           = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS;
        so.structure_size = sizeof(so);
        so.format         = LOOMC_SOURCE_FORMAT_TEXT;
        so.identifier     = loomc_make_cstring_view(src->path);
        so.contents       = loomc_make_byte_span(src->data, src->size);
        so.storage        = LOOMC_SOURCE_STORAGE_BORROWED;
        ok = ggml_hrx_loom_ok(loomc_source_create(&so, alloc, &source), "source");
    }
    ok = ok && ggml_hrx_loom_ok(loomc_module_deserialize_from_source(context, ws, source, nullptr, alloc, &module, &result), "parse") &&
         ggml_hrx_loom_result_ok(result, "parse");
    loomc_result_release(result);
    result = nullptr;

    const std::string t = std::to_string(tsteps), p = std::to_string(passes);
    const loomc_config_binding_t bindings[] = {
        { loomc_make_cstring_view("gemm.tsteps"), loomc_make_cstring_view(t.c_str()) },
        { loomc_make_cstring_view("gemm.passes"), loomc_make_cstring_view(p.c_str()) },
    };
    loomc_config_options_t config = {};
    config.bindings      = bindings;
    config.binding_count = 2;
    config.flags         = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
    loomc_compile_artifact_options_t options = {};
    options.type           = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS;
    options.structure_size = sizeof(options);
    options.target_profile = profile;
    options.config         = &config;
    ok = ok && ggml_hrx_loom_ok(loomc_compile_artifact(comp, ws, nullptr, module, &options, alloc, &result), "compile") &&
         ggml_hrx_loom_result_ok(result, "compile");
    for (loomc_host_size_t i = 0; ok && i < loomc_result_artifact_count(result); ++i) {
        const loomc_artifact_t * a = loomc_result_artifact_at(result, i);
        if (a->kind != LOOMC_ARTIFACT_KIND_EXECUTABLE) {
            continue;
        }
        loomc_byte_span_t span     = loomc_byte_span_empty();
        const bool        borrowed = loomc_byte_sequence_try_get_contiguous_span(a->contents, &span);
        if (!borrowed && !ggml_hrx_loom_ok(loomc_byte_sequence_clone(a->contents, alloc, &span), "copy")) {
            break;
        }
        out.assign(span.data, span.data + span.data_length);
        if (!borrowed) {
            loomc_allocator_free(alloc, const_cast<uint8_t *>(span.data));
        }
    }
    loomc_result_release(result);
    loomc_module_release(module);
    loomc_source_release(source);
    loomc_compiler_release(comp);
    loomc_target_profile_release(profile);
    loomc_workspace_release(ws);
    loomc_context_release(context);
    loomc_target_environment_release(env);
    return out;
}

// m rows: groups 1 if m is a multiple of 1024, else groups 2 (m is a multiple of 512).
static std::pair<hrx_executable_t, uint32_t> ggml_hrx_npu_kernel(ggml_hrx_npu * npu, int64_t m, int64_t n, int64_t k) {
    const int     groups = m % 1024 == 0 ? 1 : 2;
    const int64_t passes = m / (1024 / groups);
    const std::string key = std::to_string(k) + "," + std::to_string(groups) + "," + std::to_string(n / 128) + "," + std::to_string(passes);
    auto it = npu->kernels.find(key);
    if (it != npu->kernels.end()) {
        return it->second;
    }
    hrx_executable_t               exe     = nullptr;
    uint32_t                       ordinal = 0;
    const ggml_hrx_kernel_source * src     = ggml_hrx_npu_source(k, groups);
    const std::vector<uint8_t>     image   = src != nullptr ? ggml_hrx_npu_compile(npu, src, n / 128, passes) : std::vector<uint8_t>();
    if (image.empty() ||
        !GGML_HRX_CHECK(hrx_executable_load_data(npu->device, image.data(), image.size(), "xdna", npu->arch.c_str(), &exe)) ||
        !GGML_HRX_CHECK(hrx_executable_lookup_export_by_name(exe, "gemm_i8", &ordinal))) {
        if (exe != nullptr) {
            hrx_executable_release(exe);
        }
        exe = nullptr;
    }
    npu->kernels[key] = { exe, ordinal };
    return npu->kernels[key];
}

static bool ggml_hrx_npu_eligible(const ggml_tensor * node) {
    if (node->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * w = node->src[0];
    const ggml_tensor * x = node->src[1];
    return w->type == GGML_TYPE_Q8_0 && x->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 && ggml_hrx_tensor_on_device(w) &&
           w->op == GGML_OP_NONE && w->view_src == nullptr && ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
           w->ne[0] % 128 == 0 && w->ne[2] == 1 && w->ne[3] == 1 && x->ne[2] == 1 && x->ne[3] == 1 && ggml_is_contiguous(w) &&
           x->nb[0] == sizeof(float) && node->nb[0] == sizeof(float) && x->ne[1] >= GGML_HRX_NPU_MIN_ROWS &&
           w->ne[0] * w->ne[1] >= GGML_HRX_NPU_MIN_WEIGHT && ggml_hrx_npu_source(w->ne[0], 1) != nullptr &&
           ggml_hrx_npu_source(w->ne[0], 2) != nullptr && w->ne[1] % 128 == 0 && w->ne[1] <= GGML_HRX_NPU_MAX_N &&
           ggml_prec_get_acc(node) == GGML_PREC_UNDEFINED && ggml_prec_get_src(node, 1) == GGML_PREC_UNDEFINED &&
           ggml_get_op_params_i32(node, 1) != GGML_HINT_EXACT_BATCH;
}

// NPU rows m: a multiple of 512 rows up to share * rows.
static bool ggml_hrx_npu_shape(int64_t rows, int64_t n, float share, int64_t * m) {
    *m = std::min((int64_t) (share * (float) rows), rows) / 512 * 512;
    return *m >= 512 && n / 128 >= 2;
}

static bool ggml_hrx_npu_reads(const ggml_tensor * node, const ggml_tensor * t) {
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        const ggml_tensor * s = node->src[j];
        if (s != nullptr && (s == t || s->view_src == t || ggml_hrx_overlap(s, t))) {
            return true;
        }
    }
    return false;
}

void ggml_hrx_npu_graph_order(const ggml_hrx_device_context * dev, ggml_cgraph * cgraph) {
    if (dev->npu == nullptr) {
        return;
    }
    // Move each eligible node up to just after the last node that makes one of its inputs.
    for (int i = 1; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (!ggml_hrx_npu_eligible(node)) {
            continue;
        }
        // Tensors have no memory yet: follow the sources and their view chains.
        int target = i;
        while (target > 0) {
            const ggml_tensor * prev = cgraph->nodes[target - 1];
            bool dep = false;
            for (int j = 0; j < GGML_MAX_SRC && !dep; ++j) {
                for (const ggml_tensor * s = node->src[j]; s != nullptr && !dep; s = s->view_src) {
                    dep = s == prev || (prev->view_src == s && !ggml_op_is_empty(prev->op));
                }
            }
            if (dep) {
                break;
            }
            --target;
        }
        if (target < i) {
            std::memmove(&cgraph->nodes[target + 1], &cgraph->nodes[target], (size_t) (i - target) * sizeof(ggml_tensor *));
            cgraph->nodes[target] = node;
        }
    }
}

bool ggml_hrx_npu_plan(ggml_backend_hrx_context * ctx, const ggml_cgraph * cgraph) {
    ggml_hrx_npu * npu = ctx->device->npu;
    if (npu == nullptr || ctx->prof != nullptr) {
        return false;
    }
    if (ctx->npu == nullptr) {
        std::lock_guard<std::mutex> lock(npu->mutex);
        ctx->npu = new ggml_hrx_npu_state;
        ctx->npu->owner = npu;
        ++npu->active_contexts;
    }
    ggml_hrx_npu_state * st = ctx->npu;
    st->share.assign(cgraph->n_nodes, 0.0f);
    bool any  = false;
    int  busy = -1; // node index until which the NPU has a result in flight
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (i <= busy || !ggml_hrx_npu_eligible(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        int first = cgraph->n_nodes;
        for (int j = i + 1; j < cgraph->n_nodes; ++j) {
            if (ggml_hrx_npu_reads(cgraph->nodes[j], node)) {
                first = j;
                break;
            }
        }
        // An output that no later node reads is done at the end of the graph.
        const float share = first - i > GGML_HRX_NPU_AHEAD_DIST || first == cgraph->n_nodes ? npu->ahead : npu->split;
        int64_t m = 0;
        if (share <= 0.0f || !ggml_hrx_npu_shape(node->src[1]->ne[1], node->src[0]->ne[1], share, &m)) {
            continue;
        }
        st->share[i] = share;
        busy         = first;
        any          = true;
    }
    return any;
}

// Ends the graph that is recorded now with an NPU action and starts a new one.
static bool ggml_hrx_npu_cut(ggml_backend_hrx_context * ctx, bool wait, int op) {
    ggml_hrx_npu_replay::part part;
    part.wait = wait;
    part.op   = op;
    if (ggml_hrx_graph_rec_size(ctx->rec) > 0) {
        hrx_graph_t graph = nullptr;
        if (!GGML_HRX_CHECK(hrx_graph_create(ctx->device->device, 0, &graph))) {
            return false;
        }
        const bool ok = ggml_hrx_graph_rec_build(ctx->rec, graph) &&
                        GGML_HRX_CHECK(hrx_graph_instantiate(graph, ctx->device->stream_queue, 0, &part.exec));
        hrx_graph_release(graph);
        if (!ok) {
            return false;
        }
    }
    ctx->npu->replay->parts.push_back(part);
    ggml_hrx_graph_rec_free(ctx->rec);
    ctx->rec = ggml_hrx_graph_rec_init();
    return true;
}

void ggml_hrx_npu_record_begin(ggml_backend_hrx_context * ctx) {
    if (ctx->npu != nullptr) {
        ctx->npu->replay = new ggml_hrx_npu_replay;
    }
}

ggml_hrx_npu_replay * ggml_hrx_npu_record_end(ggml_backend_hrx_context * ctx) {
    if (ctx->npu == nullptr) {
        return nullptr;
    }
    ggml_hrx_npu_replay * r = ctx->npu->replay;
    ctx->npu->replay        = nullptr;
    if (r != nullptr && r->parts.empty()) {
        delete r;
        r = nullptr;
    }
    return r;
}

void ggml_hrx_npu_replay_free(ggml_hrx_npu_replay * r) {
    if (r == nullptr) {
        return;
    }
    for (auto & p : r->parts) {
        if (p.exec != nullptr) {
            hrx_graph_exec_release(p.exec);
        }
    }
    delete r;
}

static bool ggml_hrx_npu_submit(ggml_backend_hrx_context * ctx, const ggml_hrx_npu_replay::op & op, uint64_t * out_value) {
    ggml_hrx_npu *       npu = ctx->device->npu;
    hrx_timeline_point_t point;
    if (!GGML_HRX_CHECK(hrx_stream_flush(ctx->stream)) || !GGML_HRX_CHECK(hrx_stream_get_timeline_position(ctx->stream, &point))) {
        return false;
    }
    hrx_semaphore_t             sems[] = { point.semaphore, npu->sem };
    uint64_t                    values[] = { point.value, npu->value };
    hrx_semaphore_list_t        waits = { sems, values, 2 };
    uint64_t                    value      = npu->value + 1;
    hrx_semaphore_list_t        signals    = { &npu->sem, &value, 1 };
    const hrx_dispatch_config_t config     = { { 1, 1, 1 }, { 1, 1, 1 }, 0 };
    if (!GGML_HRX_CHECK(hrx_queue_dispatch(npu->queue, &waits, &signals, op.exe, op.ordinal, &config, nullptr, 0, op.bindings, 3, nullptr, 0))) {
        return false;
    }
    npu->value = value;
    *out_value = value;
    return true;
}

bool ggml_hrx_npu_replay_launch(ggml_backend_hrx_context * ctx, ggml_hrx_npu_replay * r) {
    ggml_hrx_npu * npu = ctx->device->npu;
    std::lock_guard<std::mutex> lock(npu->mutex);
    r->values.assign(r->ops.size(), 0);
    for (const auto & p : r->parts) {
        if (p.exec != nullptr && !GGML_HRX_CHECK(hrx_graph_exec_launch(p.exec, ctx->stream))) {
            return false;
        }
        if (p.wait) {
            if (!GGML_HRX_CHECK(hrx_stream_wait_on(ctx->stream, { npu->sem, r->values[p.op] }))) {
                return false;
            }
        } else if (!ggml_hrx_npu_submit(ctx, r->ops[p.op], &r->values[p.op])) {
            return false;
        }
    }
    return true;
}

// Waits for the NPU job and scales its sums into the node output.
static bool ggml_hrx_npu_complete(ggml_backend_hrx_context * ctx, const ggml_hrx_npu_job & job,
                                  const ggml_tensor * out, const ggml_tensor * factor) {
    ggml_hrx_npu * npu = ctx->device->npu;
    if (ctx->rec != nullptr ? !ggml_hrx_npu_cut(ctx, true, (int) job.value) :
                              !GGML_HRX_CHECK(hrx_stream_wait_on(ctx->stream, { npu->sem, job.value }))) {
        return false;
    }
    const ggml_tensor *       node = job.node;
    const ggml_hrx_npu_slot & slot = ctx->npu->slots[job.slot];
    const int64_t             K    = node->src[0]->ne[0];
    const int64_t             N    = node->ne[0];
    const size_t a_bytes = (size_t) (job.rows * K);
    const size_t c_bytes = (size_t) (job.rows * N * 4);
    const ggml_tensor c  = ggml_hrx_npu_bytes(slot.buf, GGML_PAD(a_bytes, 256), c_bytes);
    const ggml_tensor sa = ggml_hrx_npu_bytes(slot.buf, GGML_PAD(a_bytes, 256) + GGML_PAD(c_bytes, 256), (size_t) job.rows * 4);
    ggml_tensor dst = *out;
    dst.ne[1]       = job.rows;
    dst.view_src    = nullptr;
    dst.buffer      = out->view_src != nullptr ? out->view_src->buffer : out->buffer;

    ggml_hrx_kernel_desc desc;
    desc.file   = "npu/npu_i8.loom";
    desc.symbol = "ggml_npu_dequant";
    desc.config = { { "ggml.npu.wg_size", "256" }, { "ggml.npu.attn_gate", factor ? "1" : "0" } };
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);
    const int64_t lanes = job.rows * (N / 8);
    const int64_t d1    = (int64_t) (out->nb[1] / sizeof(float));
    // Completion can run while the graph recorder is visiting the first consumer.
    const auto * saved_writes = ctx->dispatch_writes;
    const std::vector<const ggml_tensor *> writes = { &dst };
    ctx->dispatch_writes = &writes;
    const bool ok = ggml_hrx_dispatch(ctx, kernel, { lanes },
                                      { lanes, job.rows, N, N, d1, (int64_t) (ggml_nbytes(&dst) / sizeof(float)), (int64_t) (c_bytes / 4), N },
                                      { &c, &sa, &job.weight->scales, &dst, factor ? factor : &dst });
    ctx->dispatch_writes = saved_writes;
    return ok;
}

static bool ggml_hrx_npu_complete_at(ggml_backend_hrx_context * ctx, size_t k) {
    ggml_hrx_npu_state * st  = ctx->npu;
    const ggml_hrx_npu_job job = st->jobs[k];
    st->jobs.erase(st->jobs.begin() + (std::ptrdiff_t) k);
    st->completing = true;
    const bool ok  = ggml_hrx_npu_complete(ctx, job, job.node, nullptr);
    st->completing = false;
    return ok;
}

bool ggml_hrx_npu_gate(ggml_backend_hrx_context * ctx, const ggml_cgraph * graph, int i, bool * ok) {
    ggml_hrx_npu_state * st = ctx->npu;
    static const bool disabled = getenv("GGML_HRX_DISABLE_FUSION") != nullptr;
    if (disabled || st == nullptr || st->jobs.size() != 1 || i + 1 >= graph->n_nodes || ggml_hrx_fuse_next(ctx, i) <= i + 1) {
        return false;
    }
    const ggml_hrx_npu_job job = st->jobs.back();
    const ggml_tensor * sig = graph->nodes[i];
    const ggml_tensor * out = graph->nodes[i + 1];
    if (sig->op != GGML_OP_UNARY || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID || sig->src[0] != job.node ||
        out->op != GGML_OP_MUL || out->src[1] != sig || (out->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        job.rows != job.node->ne[1] || !ggml_hrx_only_used_by(graph, sig, 1) || !ggml_hrx_only_used_by(graph, job.node, 1)) {
        return false;
    }
    const ggml_tensor * factor = out->src[0];
    if (factor->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 || !ggml_are_same_shape(factor, sig) ||
        !ggml_is_contiguous(factor) || !ggml_is_contiguous(out) ||
        (ggml_hrx_overlap(factor, out) && factor->data != out->data)) {
        return false;
    }
    st->jobs.pop_back();
    ctx->cur_node = out;
    *ok = ggml_hrx_npu_complete(ctx, job, out, factor);
    return true;
}

bool ggml_hrx_npu_sync(ggml_backend_hrx_context * ctx, const std::vector<hrx_buffer_ref_t> & refs) {
    ggml_hrx_npu_state * st = ctx->npu;
    if (st == nullptr || st->completing || st->jobs.empty()) {
        return true;
    }
    for (size_t k = 0; k < st->jobs.size();) {
        const ggml_hrx_npu_job & job = st->jobs[k];
        bool hit = false;
        for (const hrx_buffer_ref_t & r : refs) {
            hit = hit || (r.buffer == job.buffer && r.offset < job.offset + job.length && job.offset < r.offset + r.length);
        }
        if (!hit) {
            ++k;
            continue;
        }
        if (!ggml_hrx_npu_complete_at(ctx, k)) {
            return false;
        }
    }
    return true;
}

bool ggml_hrx_npu_finish(ggml_backend_hrx_context * ctx) {
    ggml_hrx_npu_state * st = ctx->npu;
    while (st != nullptr && !st->jobs.empty()) {
        if (!ggml_hrx_npu_complete_at(ctx, 0)) {
            return false;
        }
    }
    return true;
}

void ggml_hrx_npu_state_free(ggml_hrx_npu_state * st) {
    if (st == nullptr) {
        return;
    }
    ggml_hrx_npu * npu = st->owner;
    std::lock_guard<std::mutex> lock(npu->mutex);
    for (auto & slot : st->slots) {
        if (slot.buf != nullptr) {
            ggml_backend_buffer_free(slot.buf);
        }
    }
    for (auto buf : st->retired) {
        ggml_backend_buffer_free(buf);
    }
    if (--npu->active_contexts == 0) {
        ggml_hrx_npu_release_buffers(npu);
    }
    delete st;
}

bool ggml_hrx_npu_run(ggml_backend_hrx_context * ctx, int i, bool * ok) {
    ggml_hrx_npu_state * st = ctx->npu;
    if (st == nullptr || i >= (int) st->share.size() || st->share[i] <= 0.0f || (ctx->rec != nullptr && st->replay == nullptr)) {
        return false;
    }
    ggml_hrx_npu *      npu  = ctx->device->npu;
    const ggml_tensor * node = ctx->cur_node;
    const ggml_tensor * w    = node->src[0];
    const ggml_tensor * x    = node->src[1];
    const int64_t       K    = w->ne[0];
    const int64_t       N    = w->ne[1];
    int64_t m = 0;
    if (!ggml_hrx_npu_shape(x->ne[1], N, st->share[i], &m)) {
        return false;
    }
    *ok = false;
    std::lock_guard<std::mutex> lock(npu->mutex);
    const auto [exe, ordinal] = ggml_hrx_npu_kernel(npu, m, N, K);
    const ggml_hrx_npu_weight * weight = exe != nullptr ? ggml_hrx_npu_weight_get(npu, ctx->stream, w) : nullptr;
    if (weight == nullptr) {
        st->share[i] = 0.0f;
        return false;
    }

    // The exchange slot: A (m x K int8), C (m x N int32), the row scales. Its last GEMM must be done.
    const int slot_id = st->next_slot;
    st->next_slot    = (st->next_slot + 1) % GGML_HRX_NPU_SLOTS;
    for (size_t k = 0; k < st->jobs.size();) {
        if (st->jobs[k].slot != slot_id) {
            ++k;
        } else if (!ggml_hrx_npu_complete_at(ctx, k)) {
            return true;
        }
    }
    ggml_hrx_npu_slot & slot = st->slots[slot_id];
    const size_t a_bytes = (size_t) (m * K);
    const size_t c_bytes = (size_t) (m * N * 4);
    const size_t need    = GGML_PAD(a_bytes, 256) + GGML_PAD(c_bytes, 256) + (size_t) m * 4;
    if (slot.size < need) {
        if (slot.buf != nullptr) {
            st->retired.push_back(slot.buf);
        }
        slot.buf  = ggml_hrx_npu_alloc(npu, need);
        slot.size = slot.buf != nullptr ? need : 0;
        if (slot.buf == nullptr) {
            return true;
        }
    }
    const ggml_tensor a  = ggml_hrx_npu_bytes(slot.buf, 0, a_bytes);
    const ggml_tensor sa = ggml_hrx_npu_bytes(slot.buf, GGML_PAD(a_bytes, 256) + GGML_PAD(c_bytes, 256), (size_t) m * 4);

    // 1. GPU: quantize the first m rows of x.
    ggml_hrx_kernel_desc desc;
    desc.file   = "npu/npu_i8.loom";
    desc.symbol = "ggml_npu_quant_rows";
    desc.config = { { "ggml.npu.wg_size", std::to_string(GGML_PAD(K / 8, 64)) } };
    ggml_hrx_kernel * quant = ggml_hrx_get_kernel(ctx->device, desc);
    const int64_t s1 = (int64_t) (x->nb[1] / sizeof(float));
    const auto * saved_writes = ctx->dispatch_writes;
    const std::vector<const ggml_tensor *> writes = { &a, &sa };
    ctx->dispatch_writes = &writes;
    const bool quant_ok = ggml_hrx_dispatch(ctx, quant, { m }, { m, K, s1, (int64_t) (ggml_nbytes(x) / sizeof(float)), (int64_t) a_bytes }, { x, &a, &sa });
    ctx->dispatch_writes = saved_writes;
    if (!quant_ok) {
        return true;
    }

    // 2. NPU: the GEMM after the quantization. A recording only notes it.
    ggml_hrx_npu_replay::op op;
    op.exe         = exe;
    op.ordinal     = ordinal;
    op.bindings[0] = { ggml_hrx_buffer_handle(slot.buf), 0, a_bytes };
    op.bindings[1] = { ggml_hrx_buffer_handle(weight->buf), 0, (size_t) (N * K) };
    op.bindings[2] = { ggml_hrx_buffer_handle(slot.buf), GGML_PAD(a_bytes, 256), c_bytes };
    uint64_t value = 0;
    if (ctx->rec != nullptr) {
        value = st->replay->ops.size();
        st->replay->ops.push_back(op);
        if (!ggml_hrx_npu_cut(ctx, false, (int) value)) {
            return true;
        }
    } else if (!ggml_hrx_npu_submit(ctx, op, &value)) {
        return true;
    }
    ggml_hrx_npu_job job;
    job.node   = node;
    job.buffer = ggml_hrx_buffer_handle(node->view_src != nullptr ? node->view_src->buffer : node->buffer);
    job.offset = ggml_hrx_offset(node);
    job.length = (size_t) (m - 1) * node->nb[1] + (size_t) N * sizeof(float);
    job.value  = value;
    job.slot   = slot_id;
    job.rows   = m;
    job.weight = weight;
    st->jobs.push_back(job);

    // 3. GPU: the other rows, while the NPU runs.
    if (m < x->ne[1]) {
        ggml_tensor xr = *x;
        ggml_tensor dr = *node;
        xr.ne[1]  = dr.ne[1] = x->ne[1] - m;
        xr.data   = (char *) x->data + m * x->nb[1];
        dr.data   = (char *) node->data + m * node->nb[1];
        dr.src[1] = &xr;
        if (!ggml_hrx_op_mul_mat(ctx, &dr)) {
            return true;
        }
    }
    *ok = true;
    return true;
}
