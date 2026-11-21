// CPU hook kernels of the EXL3 ops (GGML_OP_CUSTOM nodes, see hooks/ggml/exl3/ggml-exl3-ops.c). Built with GGML_EXL3,
// like ggml/src/ggml-cpu/exl3-ops.cpp in the fork. The compute, supports and work size functions are the fork ones.

#include "ggml-hooks-cpu.h"

#include "exl3-ops.h"

namespace {

bool mul_mat_supports(const ggml_hook_dev *, const ggml_tensor * t) {
    return ggml_hook_exl3_op(t) == GGML_HOOK_EXL3_MUL_MAT && ggml_cpu_exl3_supports_op(t);
}

bool mul_mat_id_supports(const ggml_hook_dev *, const ggml_tensor * t) {
    return ggml_hook_exl3_op(t) == GGML_HOOK_EXL3_MUL_MAT_ID && ggml_cpu_exl3_mul_mat_id_supports_op(t);
}

bool get_rows_supports(const ggml_hook_dev *, const ggml_tensor * t) {
    return ggml_hook_exl3_op(t) == GGML_HOOK_EXL3_GET_ROWS && ggml_cpu_exl3_get_rows_supports_op(t);
}

void mul_mat_compute(const ggml_compute_params * params, ggml_tensor * dst) {
    ggml_compute_forward_exl3_mul_mat(params, dst);
}

void mul_mat_id_compute(const ggml_compute_params * params, ggml_tensor * dst) {
    ggml_compute_forward_exl3_mul_mat_id(params, dst);
}

void get_rows_compute(const ggml_compute_params * params, ggml_tensor * dst) {
    ggml_compute_forward_exl3_get_rows(params, dst);
}

// the fork graph_plan cases of the EXL3 ops (n_tasks = n_threads)
size_t mul_mat_work_size(int n_threads, const ggml_tensor * op) {
    return ggml_cpu_exl3_work_size(op, n_threads);
}

size_t mul_mat_id_work_size(int n_threads, const ggml_tensor * op) {
    return ggml_cpu_exl3_mul_mat_id_work_size(op, n_threads);
}

} // namespace

GGML_HOOK_REGISTER_KERNEL([] {
    auto k = ggml_hook_op_kernel("fork_cpu_exl3_mul_mat", GGML_HOOK_BACKEND_CPU, GGML_OP_CUSTOM, 0, mul_mat_supports,
                                 ggml_hook_fn(static_cast<ggml_hook_cpu_op_fn>(mul_mat_compute)));
    k.cpu_work_size = mul_mat_work_size;
    return k;
}());

GGML_HOOK_REGISTER_KERNEL([] {
    auto k = ggml_hook_op_kernel("fork_cpu_exl3_mul_mat_id", GGML_HOOK_BACKEND_CPU, GGML_OP_CUSTOM, 0, mul_mat_id_supports,
                                 ggml_hook_fn(static_cast<ggml_hook_cpu_op_fn>(mul_mat_id_compute)));
    k.cpu_work_size = mul_mat_id_work_size;
    return k;
}());

GGML_HOOK_REGISTER_KERNEL(ggml_hook_op_kernel("fork_cpu_exl3_get_rows", GGML_HOOK_BACKEND_CPU, GGML_OP_CUSTOM, 0, get_rows_supports,
                                              ggml_hook_fn(static_cast<ggml_hook_cpu_op_fn>(get_rows_compute))));
