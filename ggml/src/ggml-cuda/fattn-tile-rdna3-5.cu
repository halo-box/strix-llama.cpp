#if defined(GGML_USE_HIP)

#define GGML_CUDA_FATTN_VGPR192
#include "fattn-tile.cuh"

fattn_kernel_t ggml_cuda_fattn_tile_d256_ncols32_rdna3_5(const bool use_logit_softcap) {
    return use_logit_softcap
        ? flash_attn_tile<256, 256, 4, 8, true,  false>
        : flash_attn_tile<256, 256, 4, 8, false, false>;
}

#endif // defined(GGML_USE_HIP)
