#pragma once

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-hrx-arch.h"
#include "hrx_runtime.h"
#include "loom-jit.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// embed-kernels.cmake generates these sources.
struct ggml_hrx_kernel_source {
    const char *          path;
    const unsigned char * data;
    size_t                size;
};

extern const ggml_hrx_kernel_source ggml_hrx_kernel_sources[];
extern const size_t                 ggml_hrx_kernel_source_count;

struct ggml_hrx_kernel {
    hrx_executable_t               executable     = nullptr;
    uint32_t                       export_ordinal = 0;
    hrx_executable_export_info_t   export_info    = {};
    ggml_hrx_loom_launch_program * launch         = nullptr;
    size_t                         n_scalars      = 0;
    size_t                         scalar_size    = 0;
    std::string                    symbol;

    ~ggml_hrx_kernel();
};

struct ggml_hrx_kernel_desc {
    const char *                                     file   = nullptr; // Path relative to kernels/. With source, only a name.
    const char *                                     symbol = nullptr;
    std::vector<const char *>                        libs;
    std::vector<std::pair<std::string, std::string>> config;
    const std::string *                              source = nullptr; // Generated Loom source instead of the embedded file.
};

struct ggml_hrx_device_context {
    int         index = 0;
    std::string name;
    std::string description;
    size_t      memory_total = 0;
    uint32_t    n_cu         = 0;

    const ggml_hrx_arch * arch = nullptr;
    const ggml_hrx_tune * tune = nullptr;

    hrx_device_t device         = nullptr;
    hrx_queue_t  transfer_queue = nullptr;
    hrx_queue_t  stream_queue   = nullptr;

    // Locks buffer fills, copies and uploads that run outside a backend stream.
    std::mutex   buffer_mutex;
    hrx_stream_t buffer_stream  = nullptr;
    // True until the host waits for buffer_stream. A flush is not enough: transfer_queue reads do not wait for buffer_stream.
    bool         upload_pending = false;
    ggml_backend_buffer_t relative_pos_scratch = nullptr;

    ggml_backend_buffer_type buft = {};

    std::mutex                                              kernel_mutex;
    ggml_hrx_loom_jit *                                     jit = nullptr;
    std::map<std::string, std::unique_ptr<ggml_hrx_kernel>> kernels;

    std::atomic<uint64_t> weight_version{0};
    struct ggml_hrx_npu * npu = nullptr; // XDNA device in the same group, see ops/npu.cpp. nullptr if there is none.
};

struct ggml_hrx_fa_cls_slot {
    int64_t     graph   = -1;
    int64_t     key[12] = {};
    uint64_t    used    = 0;
    ggml_tensor tensor  = {};
};

struct ggml_backend_hrx_context {
    ggml_hrx_device_context * device      = nullptr;
    hrx_stream_t              stream      = nullptr;
    std::string               name;
    uint64_t                  weight_version = 0;
    uint64_t                  upload_wait = 0; // Last buffer_stream timeline value that this stream waits on.

    // Grow-only device scratch for op partial results. Old buffers stay alive until the context is freed.
    std::vector<ggml_backend_buffer_t> scratch;
    ggml_tensor                        scratch_tensor = {};
    ggml_hrx_fa_cls_slot               fa_cls_slots[4] = {};
    uint64_t                           fa_cls_clock = 0;
    std::vector<ggml_backend_buffer_t> fa_cls_buffers;

    // Mapped host memory for get_tensor_async. The stream copies into it. synchronize copies it to the caller memory.
    struct readback_copy {
        void * dst;
        size_t offset;
        size_t size;
    };
    hrx_buffer_t               readback        = nullptr;
    const uint8_t *            readback_host   = nullptr;
    size_t                     readback_size   = 0;
    size_t                     readback_used   = 0;
    bool                       readback_failed = false;
    std::vector<readback_copy> readback_copies;

    // Recorded graphs, keyed by the first node. A graph that runs again unchanged is replayed with one launch.
    struct node_props {
        ggml_tensor           node;
        void *                src_data[GGML_MAX_SRC];
        ggml_backend_buffer_t src_buf[GGML_MAX_SRC];
        int64_t               src_ne[GGML_MAX_SRC][GGML_MAX_DIMS];
        size_t                src_nb[GGML_MAX_SRC][GGML_MAX_DIMS];
        ggml_type             src_type[GGML_MAX_SRC];
    };
    struct graph_cache {
        uint64_t                uid  = 0;
        std::vector<node_props> props;
        hrx_graph_exec_t        exec = nullptr;
        bool                    exec_valid = false; // The recording in exec matches props.
        struct ggml_hrx_npu_replay * npu = nullptr; // NPU work of the recording, it runs before exec (ops/npu.cpp)
    };
    std::map<const void *, graph_cache> graphs;
    bool                                graphs_failed = false; // A recording failed. Do not try again.
    // While a graph is recorded, dispatches go here (ggml-hrx-graph.h).
    struct ggml_hrx_graph_rec *         rec           = nullptr;
    // Tensors that the next dispatches write, in addition to cur_node. Set it on paths that run several nodes.
    const std::vector<const ggml_tensor *> * dispatch_writes = nullptr;
    int64_t                             n_replayed    = 0; // Graph runs by replay, by a new recording and directly.
    int64_t                             n_recorded    = 0;
    int64_t                             n_direct      = 0;

    struct ggml_hrx_op_profile * prof = nullptr; // GGML_HRX_PROFILE_OPS: device time of each node, by op.

    struct ggml_hrx_autofuse * autofuse = nullptr; // Region plan of the graph that runs, see ops/autofuse.h.
    struct ggml_hrx_npu_state * npu      = nullptr; // NPU plan and results in flight of the graph that runs, see ops/npu.cpp.
    struct ggml_hrx_fusion_plan * fuse   = nullptr; // Fusions of the graph that runs, see ops/fuse.cpp.

    // GGML_HRX_DISPATCH_LOG=<file>: one line for each dispatch, to join with device profiles by order.
    FILE *              dispatch_log  = nullptr;
    const ggml_tensor * cur_node      = nullptr;
    int64_t             n_graphs_seen = 0;
};

// Logs a failed status and returns false. It frees the status.
bool ggml_hrx_check(hrx_status_t status, const char * expr, const char * file, int line);
#define GGML_HRX_CHECK(expr) ggml_hrx_check((expr), #expr, __FILE__, __LINE__)

// Compiles each kernel one time for each device. Returns nullptr on failure.
ggml_hrx_kernel * ggml_hrx_get_kernel(ggml_hrx_device_context * dev, const ggml_hrx_kernel_desc & desc);

// workload: the root kernel.def arguments. They set the launch config.
// scalars: the launch() index arguments, in order. Each value must be >= 0.
// buffers: the launch() buffer arguments, in order. Each binds the data range of its tensor, with the view offset.
bool ggml_hrx_dispatch(ggml_backend_hrx_context *         ctx,
                       ggml_hrx_kernel *                  kernel,
                       const std::vector<int64_t> &       workload,
                       const std::vector<int64_t> &       scalars,
                       const std::vector<const ggml_tensor *> & buffers);

bool ggml_hrx_tensor_on_device(const ggml_tensor * t);

// Wraps an HRX buffer (it takes one reference) as a backend buffer of dev, for ggml_hrx_dispatch bindings.
ggml_backend_buffer_t ggml_hrx_wrap_buffer(ggml_hrx_device_context * dev, hrx_buffer_t buffer, size_t size);
// Byte offset of t in its buffer.
size_t ggml_hrx_offset(const ggml_tensor * t);
uint64_t ggml_hrx_weight_version(const ggml_tensor * t);
// The HRX buffer of a backend buffer of this backend.
hrx_buffer_t ggml_hrx_buffer_handle(ggml_backend_buffer_t buffer);

// Returns a device scratch tensor of at least size bytes. Its contents do not persist between ops.
// Returns nullptr on failure.
const ggml_tensor * ggml_hrx_scratch(ggml_backend_hrx_context * ctx, size_t size);

inline bool ggml_hrx_elem_strides(const ggml_tensor * t, int64_t s[GGML_MAX_DIMS]) {
    const size_t ts = ggml_type_size(t->type);
    if (ggml_blck_size(t->type) != 1) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->nb[i] % ts != 0) {
            return false;
        }
        s[i] = (int64_t) (t->nb[i] / ts);
    }
    return true;
}

// Config of the kernels that use @ggml.elementwise.wg_size.
inline std::pair<std::string, std::string> ggml_hrx_elementwise_wg_config(const ggml_hrx_device_context * dev) {
    return { "ggml.elementwise.wg_size", std::to_string(dev->tune->elementwise_wg_size) };
}

// Workgroups of a grid-stride kernel that moves n f32 values, 4 for each lane in one pass.
// Returns 0 if the tune disables the grid-stride kernels.
inline int64_t ggml_hrx_stream_groups(const ggml_hrx_device_context * dev, int64_t n) {
    const int64_t per_group = 4 * dev->tune->elementwise_wg_size;
    const int64_t limit     = dev->tune->stream_wg_per_cu * dev->n_cu;
    return limit > 0 ? std::min((n + per_group - 1) / per_group, limit) : 0;
}
