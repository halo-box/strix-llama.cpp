// hooks: tools/mtmd/models/inkling2v.cpp of the fork (403dfcb03, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
#include "models.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

// ViT with a CLS token, learned position embeddings and 2D RoPE; the patches are merged before the projector
ggml_cgraph * mtmd_graph_inkling2v::build() {
    const int n_pos = n_patches + 1;
    auto * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);
    auto * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    auto * cur = build_inp();
    cb(cur, "patch_embd", -1);
    cur = ggml_add(ctx0, cur, resize_position_embeddings(GGML_SCALE_MODE_BILINEAR));
    auto * cls = ggml_add(ctx0, model.class_embedding, model.class_position_embedding);
    cur = ggml_concat(ctx0, cls, cur, 1);
    cb(cur, "embeddings", -1);

    auto add_pos = [&](ggml_tensor * x, const mtmd_layer &) {
        return build_rope_2d(ctx0, x, pos_w, pos_h, hparams.rope_theta, false);
    };
    cur = build_vit(cur, n_pos, NORM_TYPE_NORMAL, hparams.ffn_op, nullptr, add_pos);
    cb(cur, "encoder_out", -1);

    // remove CLS token
    cur = ggml_view_2d(ctx0, cur, n_embd, n_patches, cur->nb[1], cur->nb[1]);
    cur = build_patch_merge_permute(cur, hparams.n_merge);
    cur = build_ffn(cur, model.mm_1_w, model.mm_1_b, nullptr, nullptr, model.mm_2_w, model.mm_2_b, FFN_GELU_ERF, -1);
    cb(cur, "projected", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}
} // namespace ggml_fork
