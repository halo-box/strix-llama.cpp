#pragma once

#include <cstdlib>
#include <cstring>

enum ggml_cuda_q7_view_mode {
    GGML_CUDA_Q7_VIEW_NONE,
    GGML_CUDA_Q7_VIEW_PANEL16,
    GGML_CUDA_Q7_VIEW_Q8,
    GGML_CUDA_Q7_VIEW_Q8_NO_OUTPUT,
    GGML_CUDA_Q7_VIEW_CONFLICT,
};

static inline bool ggml_cuda_q7_view_env_enabled(const char * value) {
    return
        value != nullptr &&
        (std::strcmp(value, "1") == 0 ||
         std::strcmp(value, "true") == 0 ||
         std::strcmp(value, "on") == 0 ||
         std::strcmp(value, "force") == 0);
}

static inline bool ggml_cuda_q7_q8_view_env_no_output(
        const char * value) {
    return
        value != nullptr &&
        (std::strcmp(value, "no-output") == 0 ||
         std::strcmp(value, "exclude-output") == 0);
}

// Both experimental views reserve a private tail behind the same canonical
// Q7 tensor. Treat requesting both as an error and disable both rather than
// silently choosing one allocation/layout.
static inline ggml_cuda_q7_view_mode ggml_cuda_q7_view_mode_get() {
#if defined(GGML_USE_HIP)
    static const bool panel16 =
        ggml_cuda_q7_view_env_enabled(
            std::getenv("GGML_ROCM_GFX1151_Q7_PANEL16_CACHE"));
    static const char * q8_value =
        std::getenv("GGML_ROCM_GFX1151_Q7_Q8_VIEW");
    static const bool q8_no_output =
        ggml_cuda_q7_q8_view_env_no_output(q8_value);
    static const bool q8 =
        ggml_cuda_q7_view_env_enabled(q8_value) || q8_no_output;
    static const ggml_cuda_q7_view_mode mode =
        panel16 && q8 ? GGML_CUDA_Q7_VIEW_CONFLICT :
        panel16       ? GGML_CUDA_Q7_VIEW_PANEL16 :
        q8_no_output  ? GGML_CUDA_Q7_VIEW_Q8_NO_OUTPUT :
        q8            ? GGML_CUDA_Q7_VIEW_Q8 :
                        GGML_CUDA_Q7_VIEW_NONE;
    return mode;
#else
    return GGML_CUDA_Q7_VIEW_NONE;
#endif
}
