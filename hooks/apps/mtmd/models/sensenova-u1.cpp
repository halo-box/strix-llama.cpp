// hooks: tools/mtmd/models/sensenova-u1.cpp of the fork (3b76dee48, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
#include "models.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

ggml_cgraph * mtmd_graph_sensenova_u1::build() {
    GGML_ASSERT(model.patch_embeddings_0 != nullptr);
    GGML_ASSERT(model.patch_bias != nullptr);
    GGML_ASSERT(model.mm_0_w != nullptr);
    GGML_ASSERT(model.mm_0_b != nullptr);
    GGML_ASSERT(hparams.n_merge == 2);

    ggml_tensor * pos_x = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_x, "pos_x");
    ggml_set_input(pos_x);

    ggml_tensor * pos_y = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_y, "pos_y");
    ggml_set_input(pos_y);

    ggml_tensor * inp = build_inp();
    inp = ggml_gelu(ctx0, inp);

    // Apply adjacent-pair 2D RoPE to patch embeddings before downsampling.
    inp = ggml_reshape_4d(ctx0, inp, n_embd, 1, n_patches, n_batch);
    inp = build_rope_2d(ctx0, inp, pos_x, pos_y, hparams.rope_theta, false);
    inp = ggml_reshape_3d(ctx0, inp, n_embd, n_patches, n_batch);
    cb(inp, "patch_embd_rope", -1);

    inp = ggml_reshape_4d(ctx0, inp, n_embd, n_patches_x, n_patches_y, n_batch);
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 2, 0, 1, 3));
    inp = ggml_conv_2d(ctx0, model.mm_0_w, inp,
            hparams.n_merge, hparams.n_merge, 0, 0, 1, 1);

    const int n_output_patches = n_patches / (hparams.n_merge * hparams.n_merge);
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 2, 0, 3));
    inp = ggml_reshape_3d(ctx0, inp, n_mmproj_embd, n_output_patches, n_batch);
    inp = ggml_add(ctx0, inp, model.mm_0_b);
    cb(inp, "proj_out", -1);

    ggml_build_forward_expand(gf, inp);
    return gf;
}
} // namespace ggml_fork
