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

#ifdef __cplusplus
}
#endif
