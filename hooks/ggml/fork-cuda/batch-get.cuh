// hooks: ggml/src/ggml-cuda/batch-get.cuh of the fork (1fe05fa45, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/fork-cuda/README.md

} // namespace ggml_fork
#include "ggml-backend.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/fork-cuda/README.md

bool ggml_cuda_get_tensor_n(int device,
        const ggml_tensor * const * tensors, void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors);
} // namespace ggml_fork
