// Hook kernels of the fork CPU code: the fork versions of upstream CPU functions (mul-mat.c, ops-q.inc,
// binary-ops.inc). The cases that upstream does not run are on by default; the cases where the
// fork changes what upstream runs are variants, off by default (GGML_HOOKS=+name), until a benchmark decides.

#include "ggml-hooks-cpu.h"

#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ops.h"
#include "simd-mappings.h"
#include "vec.h"

#include <algorithm>
#include <cmath>
#include <cstring>

extern "C" {
void   ggml_hook_fork_mul_mat(const ggml_compute_params * params, ggml_tensor * dst);
void   ggml_hook_fork_mul_mat_id(const ggml_compute_params * params, ggml_tensor * dst);
size_t ggml_hook_fork_mul_mat_id_work_size(int n_tasks, const ggml_tensor * node);
}

#include "binary-ops.inc"
#include "ops-q.inc"

namespace {

// mul_mat

// the fork: an F16 src1 against a quantized vec_dot_type, converted through F32 (upstream: F32, the vec_dot_type, or
// F16/BF16 widened to an F32 vec_dot_type)
bool f16_act(const ggml_tensor * t, const ggml_tensor * src1) {
    const ggml_type vdt = ggml_get_type_traits_cpu(t->src[0]->type)->vec_dot_type;
    return src1->type == GGML_TYPE_F16 && vdt != GGML_TYPE_F16 && vdt != GGML_TYPE_F32 && ggml_get_type_traits_cpu(vdt)->from_float != nullptr;
}

bool mul_mat_f16_act_supports(const ggml_hook_dev *, const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && f16_act(t, t->src[1]);
}

void mul_mat_compute(const ggml_compute_params * params, ggml_tensor * dst) {
    ggml_hook_fork_mul_mat(params, dst);
}

bool mul_mat_id_f16_act_supports(const ggml_hook_dev *, const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && t->src[1]->type == GGML_TYPE_F16;
}

bool mul_mat_id_supports(const ggml_hook_dev *, const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && t->src[1]->type == GGML_TYPE_F32;
}

void mul_mat_id_compute(const ggml_compute_params * params, ggml_tensor * dst) {
    ggml_hook_fork_mul_mat_id(params, dst);
}

size_t mul_mat_id_work_size(int n_threads, const ggml_tensor * op) {
    return ggml_hook_fork_mul_mat_id_work_size(n_threads, op);
}


// the generic quantized paths of add, out_prod and get_rows, for the fork types (the fork added their case labels)

bool hook_src0(const ggml_tensor * t) {
    return ggml_hook_type_is_hook(t->src[0]->type);
}

bool add_q_supports(const ggml_hook_dev *, const ggml_tensor * t) {
    return hook_src0(t) && t->src[1]->type == GGML_TYPE_F32;
}

void add_q_compute(const ggml_compute_params * params, ggml_tensor * dst) {
    ggml_compute_forward_add_q_f32(params, dst);
}

bool out_prod_q_supports(const ggml_hook_dev *, const ggml_tensor * t) {
    return hook_src0(t);
}

void out_prod_q_compute(const ggml_compute_params * params, ggml_tensor * dst) {
    ggml_compute_forward_out_prod_q_f32(params, dst);
}

bool get_rows_q_supports(const ggml_hook_dev *, const ggml_tensor * t) {
    return hook_src0(t) && t->type == GGML_TYPE_F32;
}

void get_rows_q_compute(const ggml_compute_params * params, ggml_tensor * dst) {
    ggml_compute_forward_get_rows_q(params, dst);
}

// binary ops: the two f16 cases the fork added to binary_op (f16 + f16 -> f32, f32 + f16 -> f32)

bool binary_f16_supports(const ggml_hook_dev *, const ggml_tensor * t) {
    const ggml_type a = t->src[0]->type, b = t->src[1]->type;
    return t->type == GGML_TYPE_F32 && b == GGML_TYPE_F16 && (a == GGML_TYPE_F16 || a == GGML_TYPE_F32);
}

void add_f16_compute(const ggml_compute_params * params, ggml_tensor * dst) { binary_op<op_add>(params, dst); }
void sub_f16_compute(const ggml_compute_params * params, ggml_tensor * dst) { binary_op<op_sub>(params, dst); }
void mul_f16_compute(const ggml_compute_params * params, ggml_tensor * dst) { binary_op<op_mul>(params, dst); }
void div_f16_compute(const ggml_compute_params * params, ggml_tensor * dst) { binary_op<op_div>(params, dst); }

} // namespace

#define FORK_CPU_KERNEL(NAME, OP, SUPPORTS, COMPUTE, ON) \
    GGML_HOOK_REGISTER_KERNEL([] { \
        auto k = ggml_hook_op_kernel(NAME, GGML_HOOK_BACKEND_CPU, OP, 0, SUPPORTS, ggml_hook_fn(static_cast<ggml_hook_cpu_op_fn>(COMPUTE))); \
        k.enabled_by_default = ON; \
        return k; \
    }())

FORK_CPU_KERNEL("cpu_mul_mat_f16_act", GGML_OP_MUL_MAT, mul_mat_f16_act_supports, mul_mat_compute, true);

GGML_HOOK_REGISTER_KERNEL([] {
    auto k = ggml_hook_op_kernel("cpu_mul_mat_id_f16_act", GGML_HOOK_BACKEND_CPU, GGML_OP_MUL_MAT_ID, 0, mul_mat_id_f16_act_supports,
                                 ggml_hook_fn(static_cast<ggml_hook_cpu_op_fn>(mul_mat_id_compute)));
    k.cpu_work_size = mul_mat_id_work_size;
    return k;
}());

// variant: the compact row map of the fork for every MUL_MAT_ID
GGML_HOOK_REGISTER_KERNEL([] {
    auto k = ggml_hook_op_kernel("cpu_mul_mat_id", GGML_HOOK_BACKEND_CPU, GGML_OP_MUL_MAT_ID, 0, mul_mat_id_supports,
                                 ggml_hook_fn(static_cast<ggml_hook_cpu_op_fn>(mul_mat_id_compute)));
    k.cpu_work_size      = mul_mat_id_work_size;
    k.enabled_by_default = false;
    return k;
}());


FORK_CPU_KERNEL("cpu_add_q",      GGML_OP_ADD,      add_q_supports,      add_q_compute,      true);
FORK_CPU_KERNEL("cpu_out_prod_q", GGML_OP_OUT_PROD, out_prod_q_supports, out_prod_q_compute, true);
FORK_CPU_KERNEL("cpu_get_rows_q", GGML_OP_GET_ROWS, get_rows_q_supports, get_rows_q_compute, true);

FORK_CPU_KERNEL("cpu_add_f16", GGML_OP_ADD, binary_f16_supports, add_f16_compute, true);
FORK_CPU_KERNEL("cpu_sub_f16", GGML_OP_SUB, binary_f16_supports, sub_f16_compute, true);
FORK_CPU_KERNEL("cpu_mul_f16", GGML_OP_MUL, binary_f16_supports, mul_f16_compute, true);
FORK_CPU_KERNEL("cpu_div_f16", GGML_OP_DIV, binary_f16_supports, div_f16_compute, true);
