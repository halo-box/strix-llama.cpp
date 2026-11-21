// hooks: ggml/src/ggml-cuda/fattn-tile-rdna3-5.cu of the fork (1fe05fa45, upstream 9048bad8d, new in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/fork-cuda/README.md
#if defined(GGML_USE_HIP)

#define GGML_CUDA_FATTN_VGPR192
} // namespace ggml_fork
#include "fattn-tile.cuh"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/fork-cuda/README.md

fattn_kernel_t ggml_cuda_fattn_tile_d256_ncols32_rdna3_5(const bool use_logit_softcap) {
    return use_logit_softcap
        ? flash_attn_tile<256, 256, 4, 8, true,  false>
        : flash_attn_tile<256, 256, 4, 8, false, false>;
}

#endif // defined(GGML_USE_HIP)
} // namespace ggml_fork
