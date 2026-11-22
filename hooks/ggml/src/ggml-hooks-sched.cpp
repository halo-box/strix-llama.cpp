// The sched switch (GGML_HOOKS=+sched, off by default): the scheduler and graph allocator of the fork
// (sched/backend.cpp, sched/alloc.c), the buffer allocation of a tensor list and the sanitizer
// calls. The upstream functions are renamed *__upstream (hooks_rename); each public function here calls one side.
// The switch is read once: the objects of one side are never passed to the other.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-hooks.h"

static bool fork_sched_on() {
    static const bool on = ggml_hook_switch("sched", false);
    return on;
}

extern "C" {
int ggml_hook_sched_n_graph_nodes__upstream(ggml_backend_sched_t sched); // ggml-backend-upstream.cpp

bool fork_ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst);
bool ggml_backend_buffer_copy_tensor__upstream(const struct ggml_tensor * src, struct ggml_tensor * dst);
void fork_ggml_backend_buffer_free(ggml_backend_buffer_t buffer);
void ggml_backend_buffer_free__upstream(ggml_backend_buffer_t buffer);
ggml_backend_buffer_t fork_ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors);
ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n__upstream(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors);
void fork_ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend);
void ggml_backend_event_record__upstream(ggml_backend_event_t event, ggml_backend_t backend);
void fork_ggml_backend_event_synchronize(ggml_backend_event_t event);
void ggml_backend_event_synchronize__upstream(ggml_backend_event_t event);
void fork_ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event);
void ggml_backend_event_wait__upstream(ggml_backend_t backend, ggml_backend_event_t event);
enum ggml_status fork_ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph);
enum ggml_status ggml_backend_graph_compute_async__upstream(ggml_backend_t backend, struct ggml_cgraph * cgraph);
bool fork_ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
bool ggml_backend_sched_alloc_graph__upstream(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
void fork_ggml_backend_sched_free(ggml_backend_sched_t sched);
void ggml_backend_sched_free__upstream(ggml_backend_sched_t sched);
ggml_backend_t fork_ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i);
ggml_backend_t ggml_backend_sched_get_backend__upstream(ggml_backend_sched_t sched, int i);
size_t fork_ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend);
size_t ggml_backend_sched_get_buffer_size__upstream(ggml_backend_sched_t sched, ggml_backend_t backend);
ggml_backend_buffer_type_t fork_ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend);
ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type__upstream(ggml_backend_sched_t sched, ggml_backend_t backend);
int fork_ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched);
int ggml_backend_sched_get_n_backends__upstream(ggml_backend_sched_t sched);
int fork_ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched);
int ggml_backend_sched_get_n_copies__upstream(ggml_backend_sched_t sched);
int fork_ggml_backend_sched_get_n_graph_nodes(ggml_backend_sched_t sched);
int fork_ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched);
int ggml_backend_sched_get_n_splits__upstream(ggml_backend_sched_t sched);
ggml_backend_t fork_ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node);
ggml_backend_t ggml_backend_sched_get_tensor_backend__upstream(ggml_backend_sched_t sched, struct ggml_tensor * node);
enum ggml_status fork_ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
enum ggml_status ggml_backend_sched_graph_compute__upstream(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
enum ggml_status fork_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
enum ggml_status ggml_backend_sched_graph_compute_async__upstream(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
ggml_backend_sched_t fork_ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel, bool op_offload);
ggml_backend_sched_t ggml_backend_sched_new__upstream(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel, bool op_offload);
void fork_ggml_backend_sched_prepare_inputs(ggml_backend_sched_t sched);
bool fork_ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph);
bool ggml_backend_sched_reserve__upstream(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph);
void fork_ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes);
void ggml_backend_sched_reserve_size__upstream(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes);
void fork_ggml_backend_sched_reset(ggml_backend_sched_t sched);
void ggml_backend_sched_reset__upstream(ggml_backend_sched_t sched);
void fork_ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, void * user_data);
void ggml_backend_sched_set_copy_callback__upstream(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, void * user_data);
void fork_ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data);
void ggml_backend_sched_set_eval_callback__upstream(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data);
void fork_ggml_backend_sched_set_lazy_realloc_sync(ggml_backend_sched_t sched, bool lazy);
void fork_ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, void * user_data);
void ggml_backend_sched_set_realloc_callback__upstream(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, void * user_data);
void fork_ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend);
void ggml_backend_sched_set_tensor_backend__upstream(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend);
void fork_ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
void ggml_backend_sched_split_graph__upstream(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
void fork_ggml_backend_sched_synchronize(ggml_backend_sched_t sched);
void ggml_backend_sched_synchronize__upstream(ggml_backend_sched_t sched);
void fork_ggml_backend_synchronize(ggml_backend_t backend);
void ggml_backend_synchronize__upstream(ggml_backend_t backend);
void fork_ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst);
void ggml_backend_tensor_copy__upstream(const struct ggml_tensor * src, struct ggml_tensor * dst);
void fork_ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);
void ggml_backend_tensor_copy_async__upstream(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);
void fork_ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
void ggml_backend_tensor_get__upstream(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
void fork_ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
void ggml_backend_tensor_get_2d__upstream(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
void fork_ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
void ggml_backend_tensor_get_2d_async__upstream(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
void fork_ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
void ggml_backend_tensor_get_async__upstream(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
void fork_ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
void ggml_backend_tensor_memset__upstream(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
void fork_ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_backend_tensor_set__upstream(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void fork_ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
void ggml_backend_tensor_set_2d__upstream(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
void fork_ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
void ggml_backend_tensor_set_2d_async__upstream(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
void fork_ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_backend_tensor_set_async__upstream(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void fork_ggml_backend_tensor_set_direct(struct ggml_tensor * tensor, size_t offset, size_t size);
bool fork_ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
bool ggml_gallocr_alloc_graph__upstream(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
bool fork_ggml_gallocr_alloc_graph_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids);
void fork_ggml_gallocr_clear_pins(ggml_gallocr_t galloc);
void fork_ggml_gallocr_free(ggml_gallocr_t galloc);
void ggml_gallocr_free__upstream(ggml_gallocr_t galloc);
size_t fork_ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id);
size_t ggml_gallocr_get_buffer_size__upstream(ggml_gallocr_t galloc, int buffer_id);
ggml_gallocr_t fork_ggml_gallocr_new(ggml_backend_buffer_type_t buft);
ggml_gallocr_t ggml_gallocr_new__upstream(ggml_backend_buffer_type_t buft);
ggml_gallocr_t fork_ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs);
ggml_gallocr_t ggml_gallocr_new_n__upstream(ggml_backend_buffer_type_t * bufts, int n_bufs);
void fork_ggml_gallocr_pin_tensor(ggml_gallocr_t galloc, struct ggml_tensor * t);
bool fork_ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
bool ggml_gallocr_reserve__upstream(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
bool fork_ggml_gallocr_reserve_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids);
bool ggml_gallocr_reserve_n__upstream(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids);
void fork_ggml_gallocr_reserve_n_size(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids, size_t * sizes);
void ggml_gallocr_reserve_n_size__upstream(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids, size_t * sizes);
void fork_ggml_gallocr_set_free_callback(ggml_gallocr_t galloc, void (*cb)(void * user_data), void * user_data);
bool fork_ggml_op_alloc_size_may_expand(enum ggml_op op);
bool ggml_op_alloc_size_may_expand__upstream(enum ggml_op op);
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (fork_sched_on()) {
        return fork_ggml_backend_buffer_copy_tensor(src, dst);
    }
    return ggml_backend_buffer_copy_tensor__upstream(src, dst);
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (fork_sched_on()) {
        fork_ggml_backend_buffer_free(buffer);
        return;
    }
    ggml_backend_buffer_free__upstream(buffer);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    if (fork_sched_on()) {
        return fork_ggml_backend_buft_alloc_buffer_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_alloc_buffer_n__upstream(buft, tensors, n_tensors);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    if (fork_sched_on()) {
        fork_ggml_backend_event_record(event, backend);
        return;
    }
    ggml_backend_event_record__upstream(event, backend);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    if (fork_sched_on()) {
        fork_ggml_backend_event_synchronize(event);
        return;
    }
    ggml_backend_event_synchronize__upstream(event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    if (fork_sched_on()) {
        fork_ggml_backend_event_wait(backend, event);
        return;
    }
    ggml_backend_event_wait__upstream(backend, event);
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    if (fork_sched_on()) {
        return fork_ggml_backend_graph_compute_async(backend, cgraph);
    }
    return ggml_backend_graph_compute_async__upstream(backend, cgraph);
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    if (fork_sched_on()) {
        return fork_ggml_backend_sched_alloc_graph(sched, graph);
    }
    return ggml_backend_sched_alloc_graph__upstream(sched, graph);
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (fork_sched_on()) {
        fork_ggml_backend_sched_free(sched);
        return;
    }
    ggml_backend_sched_free__upstream(sched);
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    if (fork_sched_on()) {
        return fork_ggml_backend_sched_get_backend(sched, i);
    }
    return ggml_backend_sched_get_backend__upstream(sched, i);
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    if (fork_sched_on()) {
        return fork_ggml_backend_sched_get_buffer_size(sched, backend);
    }
    return ggml_backend_sched_get_buffer_size__upstream(sched, backend);
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    if (fork_sched_on()) {
        return fork_ggml_backend_sched_get_buffer_type(sched, backend);
    }
    return ggml_backend_sched_get_buffer_type__upstream(sched, backend);
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    if (fork_sched_on()) {
        return fork_ggml_backend_sched_get_n_backends(sched);
    }
    return ggml_backend_sched_get_n_backends__upstream(sched);
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    if (fork_sched_on()) {
        return fork_ggml_backend_sched_get_n_copies(sched);
    }
    return ggml_backend_sched_get_n_copies__upstream(sched);
}

int ggml_backend_sched_get_n_graph_nodes(ggml_backend_sched_t sched) {
    if (fork_sched_on()) {
        return fork_ggml_backend_sched_get_n_graph_nodes(sched);
    }
    return ggml_hook_sched_n_graph_nodes__upstream(sched);
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    if (fork_sched_on()) {
        return fork_ggml_backend_sched_get_n_splits(sched);
    }
    return ggml_backend_sched_get_n_splits__upstream(sched);
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    if (fork_sched_on()) {
        return fork_ggml_backend_sched_get_tensor_backend(sched, node);
    }
    return ggml_backend_sched_get_tensor_backend__upstream(sched, node);
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    if (fork_sched_on()) {
        return fork_ggml_backend_sched_graph_compute(sched, graph);
    }
    return ggml_backend_sched_graph_compute__upstream(sched, graph);
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    if (fork_sched_on()) {
        return fork_ggml_backend_sched_graph_compute_async(sched, graph);
    }
    return ggml_backend_sched_graph_compute_async__upstream(sched, graph);
}

ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel, bool op_offload) {
    if (fork_sched_on()) {
        return fork_ggml_backend_sched_new(backends, bufts, n_backends, graph_size, parallel, op_offload);
    }
    return ggml_backend_sched_new__upstream(backends, bufts, n_backends, graph_size, parallel, op_offload);
}

void ggml_backend_sched_prepare_inputs(ggml_backend_sched_t sched) {
    if (fork_sched_on()) {
        fork_ggml_backend_sched_prepare_inputs(sched);
        return;
    }
    GGML_UNUSED(sched); // upstream: no ring of graph inputs
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    if (fork_sched_on()) {
        return fork_ggml_backend_sched_reserve(sched, measure_graph);
    }
    return ggml_backend_sched_reserve__upstream(sched, measure_graph);
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    if (fork_sched_on()) {
        fork_ggml_backend_sched_reserve_size(sched, measure_graph, sizes);
        return;
    }
    ggml_backend_sched_reserve_size__upstream(sched, measure_graph, sizes);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    if (fork_sched_on()) {
        fork_ggml_backend_sched_reset(sched);
        return;
    }
    ggml_backend_sched_reset__upstream(sched);
}

void ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, void * user_data) {
    if (fork_sched_on()) {
        fork_ggml_backend_sched_set_copy_callback(sched, callback, user_data);
        return;
    }
    ggml_backend_sched_set_copy_callback__upstream(sched, callback, user_data);
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    if (fork_sched_on()) {
        fork_ggml_backend_sched_set_eval_callback(sched, callback, user_data);
        return;
    }
    ggml_backend_sched_set_eval_callback__upstream(sched, callback, user_data);
}

void ggml_backend_sched_set_lazy_realloc_sync(ggml_backend_sched_t sched, bool lazy) {
    if (fork_sched_on()) {
        fork_ggml_backend_sched_set_lazy_realloc_sync(sched, lazy);
        return;
    }
    GGML_UNUSED(sched); GGML_UNUSED(lazy); // upstream: always syncs
}

void ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, void * user_data) {
    if (fork_sched_on()) {
        fork_ggml_backend_sched_set_realloc_callback(sched, callback, user_data);
        return;
    }
    ggml_backend_sched_set_realloc_callback__upstream(sched, callback, user_data);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    if (fork_sched_on()) {
        fork_ggml_backend_sched_set_tensor_backend(sched, node, backend);
        return;
    }
    ggml_backend_sched_set_tensor_backend__upstream(sched, node, backend);
}

void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    if (fork_sched_on()) {
        fork_ggml_backend_sched_split_graph(sched, graph);
        return;
    }
    ggml_backend_sched_split_graph__upstream(sched, graph);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    if (fork_sched_on()) {
        fork_ggml_backend_sched_synchronize(sched);
        return;
    }
    ggml_backend_sched_synchronize__upstream(sched);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    if (fork_sched_on()) {
        fork_ggml_backend_synchronize(backend);
        return;
    }
    ggml_backend_synchronize__upstream(backend);
}

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (fork_sched_on()) {
        fork_ggml_backend_tensor_copy(src, dst);
        return;
    }
    ggml_backend_tensor_copy__upstream(src, dst);
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (fork_sched_on()) {
        fork_ggml_backend_tensor_copy_async(backend_src, backend_dst, src, dst);
        return;
    }
    ggml_backend_tensor_copy_async__upstream(backend_src, backend_dst, src, dst);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (fork_sched_on()) {
        fork_ggml_backend_tensor_get(tensor, data, offset, size);
        return;
    }
    ggml_backend_tensor_get__upstream(tensor, data, offset, size);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    if (fork_sched_on()) {
        fork_ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
        return;
    }
    ggml_backend_tensor_get_2d__upstream(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    if (fork_sched_on()) {
        fork_ggml_backend_tensor_get_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
        return;
    }
    ggml_backend_tensor_get_2d_async__upstream(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (fork_sched_on()) {
        fork_ggml_backend_tensor_get_async(backend, tensor, data, offset, size);
        return;
    }
    ggml_backend_tensor_get_async__upstream(backend, tensor, data, offset, size);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    if (fork_sched_on()) {
        fork_ggml_backend_tensor_memset(tensor, value, offset, size);
        return;
    }
    ggml_backend_tensor_memset__upstream(tensor, value, offset, size);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (fork_sched_on()) {
        fork_ggml_backend_tensor_set(tensor, data, offset, size);
        return;
    }
    ggml_backend_tensor_set__upstream(tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    if (fork_sched_on()) {
        fork_ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
        return;
    }
    ggml_backend_tensor_set_2d__upstream(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    if (fork_sched_on()) {
        fork_ggml_backend_tensor_set_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
        return;
    }
    ggml_backend_tensor_set_2d_async__upstream(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (fork_sched_on()) {
        fork_ggml_backend_tensor_set_async(backend, tensor, data, offset, size);
        return;
    }
    ggml_backend_tensor_set_async__upstream(backend, tensor, data, offset, size);
}

void ggml_backend_tensor_set_direct(struct ggml_tensor * tensor, size_t offset, size_t size) {
    if (fork_sched_on()) {
        fork_ggml_backend_tensor_set_direct(tensor, offset, size);
        return;
    }
    GGML_UNUSED(tensor); GGML_UNUSED(offset); GGML_UNUSED(size); // the sanitizer runs with sched only
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (fork_sched_on()) {
        return fork_ggml_gallocr_alloc_graph(galloc, graph);
    }
    return ggml_gallocr_alloc_graph__upstream(galloc, graph);
}

bool ggml_gallocr_alloc_graph_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    if (fork_sched_on()) {
        return fork_ggml_gallocr_alloc_graph_n(galloc, graph, node_buffer_ids, leaf_buffer_ids);
    }
    GGML_UNUSED(node_buffer_ids); GGML_UNUSED(leaf_buffer_ids);
    return ggml_gallocr_alloc_graph__upstream(galloc, graph);
}

void ggml_gallocr_clear_pins(ggml_gallocr_t galloc) {
    if (fork_sched_on()) {
        fork_ggml_gallocr_clear_pins(galloc);
        return;
    }
    GGML_UNUSED(galloc);
}

void ggml_gallocr_free(ggml_gallocr_t galloc) {
    if (fork_sched_on()) {
        fork_ggml_gallocr_free(galloc);
        return;
    }
    ggml_gallocr_free__upstream(galloc);
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id) {
    if (fork_sched_on()) {
        return fork_ggml_gallocr_get_buffer_size(galloc, buffer_id);
    }
    return ggml_gallocr_get_buffer_size__upstream(galloc, buffer_id);
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft) {
    if (fork_sched_on()) {
        return fork_ggml_gallocr_new(buft);
    }
    return ggml_gallocr_new__upstream(buft);
}

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs) {
    if (fork_sched_on()) {
        return fork_ggml_gallocr_new_n(bufts, n_bufs);
    }
    return ggml_gallocr_new_n__upstream(bufts, n_bufs);
}

void ggml_gallocr_pin_tensor(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    if (fork_sched_on()) {
        fork_ggml_gallocr_pin_tensor(galloc, t);
        return;
    }
    GGML_UNUSED(galloc); GGML_UNUSED(t); // the upstream scheduler does not read across backends in place
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (fork_sched_on()) {
        return fork_ggml_gallocr_reserve(galloc, graph);
    }
    return ggml_gallocr_reserve__upstream(galloc, graph);
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    if (fork_sched_on()) {
        return fork_ggml_gallocr_reserve_n(galloc, graph, node_buffer_ids, leaf_buffer_ids);
    }
    return ggml_gallocr_reserve_n__upstream(galloc, graph, node_buffer_ids, leaf_buffer_ids);
}

void ggml_gallocr_reserve_n_size(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids, size_t * sizes) {
    if (fork_sched_on()) {
        fork_ggml_gallocr_reserve_n_size(galloc, graph, node_buffer_ids, leaf_buffer_ids, sizes);
        return;
    }
    ggml_gallocr_reserve_n_size__upstream(galloc, graph, node_buffer_ids, leaf_buffer_ids, sizes);
}

void ggml_gallocr_set_free_callback(ggml_gallocr_t galloc, void (*cb)(void * user_data), void * user_data) {
    if (fork_sched_on()) {
        fork_ggml_gallocr_set_free_callback(galloc, cb, user_data);
        return;
    }
    GGML_UNUSED(galloc); GGML_UNUSED(cb); GGML_UNUSED(user_data); // upstream syncs before every re-reserve
}

bool ggml_op_alloc_size_may_expand(enum ggml_op op) {
    if (fork_sched_on()) {
        return fork_ggml_op_alloc_size_may_expand(op);
    }
    return ggml_op_alloc_size_may_expand__upstream(op);
}
