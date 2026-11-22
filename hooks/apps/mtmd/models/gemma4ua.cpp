// hooks: tools/mtmd/models/gemma4ua.cpp of the fork (403dfcb03, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
#include "models.h"
#include <cmath>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

ggml_cgraph * mtmd_graph_gemma4ua::build() {
    ggml_tensor * inp = build_inp_raw(1);

    auto cur = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));

    // Gemma4UnifiedMultimodalEmbedder
    {
        // embedding_pre_projection_norm
        cur = ggml_rms_norm(ctx0, cur, hparams.eps);
        cur = build_mm(model.mm_input_proj_w, cur);
        cb(cur, "projected", -1);
    }

    ggml_build_forward_expand(gf, cur);
    return gf;
}
} // namespace ggml_fork
