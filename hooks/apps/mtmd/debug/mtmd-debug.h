// hooks: tools/mtmd/debug/mtmd-debug.h of the fork (403dfcb03, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include "mtmd.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include <vector>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

// INTERNAL HEADER FOR DEBUGGING PURPOSES ONLY
// NOT INTENDED FOR PUBLIC USE
// Do not raise issues related to this debugging API

// encode take the pre-processed f32 values, print the intermidiate values via cb_eval callback
MTMD_API void mtmd_debug_encode_image(mtmd_context * ctx, const std::vector<std::vector<float>> & image);
MTMD_API void mtmd_debug_encode_audio(mtmd_context * ctx, const std::vector<float> & input); // will be broadcasted to fit n_mel

// preprocess take the raw input values
MTMD_API void mtmd_debug_preprocess_image(mtmd_context * ctx, const std::vector<uint8_t> & rgb_values, int nx, int ny);
MTMD_API void mtmd_debug_preprocess_audio(mtmd_context * ctx, const std::vector<float> & pcm_samples);
} // namespace ggml_fork
