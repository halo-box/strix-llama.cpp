// hooks: ggml/src/ggml-cuda/lightning-indexer.cuh of the fork (1fe05fa45, upstream 9048bad8d, changed in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#include "common.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md

void ggml_cuda_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_lightning_indexer_supported(int device, const ggml_tensor * dst);
} // namespace ggml_fork
