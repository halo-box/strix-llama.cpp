// copied from ggml/src/ggml-cpu/exl3-ops.h of the fork (1fe05fa45, upstream 9048bad8d). Adapted: the op parameter through ggml_hook_exl3_param (0 uses).
#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_compute_params;

bool   ggml_cpu_exl3_supports_op(const struct ggml_tensor * op);
size_t ggml_cpu_exl3_work_size(const struct ggml_tensor * op, int n_tasks);
void   ggml_compute_forward_exl3_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);

bool   ggml_cpu_exl3_mul_mat_id_supports_op(const struct ggml_tensor * op);
size_t ggml_cpu_exl3_mul_mat_id_work_size(const struct ggml_tensor * op, int n_tasks);
void   ggml_compute_forward_exl3_mul_mat_id(const struct ggml_compute_params * params, struct ggml_tensor * dst);

bool   ggml_cpu_exl3_get_rows_supports_op(const struct ggml_tensor * op);
void   ggml_compute_forward_exl3_get_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
