#pragma once

// CPU hook kernels. Include from files built in the CPU backend (kernels/cpu/).
//
// kernel signature for GGML_HOOK_BACKEND_CPU ops (ggml_hook_cpu_op_fn):
//   void kernel(const ggml_compute_params * params, ggml_tensor * dst);
// it runs on every thread of the op, like the ggml_compute_forward_* functions, with the same work buffer.
// supports() must be a pure function of the node: every thread must take the same decision.

#include "ggml-hooks.h"
