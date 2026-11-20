// example: SCALE f32 op kernel and SCALE + ADD fusion on Vulkan. Off by default, used by test-ggml-hooks.

#include "ggml-vulkan-common.h"
#include "ggml-impl.h"

#include "ggml-hooks-vk.h"

#include "example_scale.spv.h"
#include "example_scale_add.spv.h"

struct example_scale_pc {
    uint32_t ne;
    uint32_t a_offs;
    uint32_t b_offs;
    uint32_t d_offs;
    float    scale;
    float    bias;
};

// one workgroup of 256 per 256 elements, on dimension x only
static constexpr int64_t EXAMPLE_MAX_NE = 256ll * 65535;

static bool example_f32_contiguous(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

static bool example_scale_supports(const ggml_hook_dev *, const ggml_tensor * t) {
    return example_f32_contiguous(t) && example_f32_contiguous(t->src[0]) && ggml_nelements(t) <= EXAMPLE_MAX_NE;
}

static bool example_scale_compute(ggml_backend_vk_context * ctx, vk_context & subctx, ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor *       dst = cgraph->nodes[node_idx];
    const ggml_tensor * a   = dst->src[0];

    vk_pipeline pipeline = ggml_hook_vk_pipeline(ctx, "ggml_hook_example_scale_f32", ggml_hook_spv_example_scale_data, ggml_hook_spv_example_scale_len, 2,
                                             sizeof(example_scale_pc), { 256, 1, 1 });
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    const uint32_t         ne = (uint32_t) ggml_nelements(dst);
    const example_scale_pc pc = { ne, get_misalign_bytes(ctx, a) / 4, 0, get_misalign_bytes(ctx, dst) / 4, ggml_get_op_params_f32(dst, 0), ggml_get_op_params_f32(dst, 1) };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { ggml_vk_tensor_subbuffer(ctx, a, true), ggml_vk_tensor_subbuffer(ctx, dst, true) }, pc, { ne, 1, 1 });
    return true;
}

// returns the ADD src that is not the SCALE output, or nullptr
static const ggml_tensor * example_add_other(const ggml_tensor * scale, const ggml_tensor * add) {
    if (add->src[0] == scale) {
        return add->src[1];
    }
    if (add->src[1] == scale) {
        return add->src[0];
    }
    return nullptr;
}

static int example_scale_add_match(const ggml_hook_dev *, const ggml_cgraph * cgraph, int i) {
    if (!ggml_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_ADD })) {
        return 0;
    }
    const ggml_tensor * scale = cgraph->nodes[i];
    const ggml_tensor * add   = cgraph->nodes[i + 1];
    const ggml_tensor * b     = example_add_other(scale, add);
    if (!b || !example_scale_supports(nullptr, scale) || !example_f32_contiguous(add) || !example_f32_contiguous(b) ||
        !ggml_are_same_shape(scale, b) || !ggml_are_same_shape(scale, add)) {
        return 0;
    }
    return 2;
}

static void example_scale_add_compute(ggml_backend_vk_context * ctx, vk_context & subctx, ggml_cgraph * cgraph, int node_idx, int) {
    const ggml_tensor * scale = cgraph->nodes[node_idx];
    ggml_tensor *       dst   = cgraph->nodes[node_idx + 1];
    const ggml_tensor * a     = scale->src[0];
    const ggml_tensor * b     = example_add_other(scale, dst);

    vk_pipeline pipeline = ggml_hook_vk_pipeline(ctx, "ggml_hook_example_scale_add_f32", ggml_hook_spv_example_scale_add_data, ggml_hook_spv_example_scale_add_len, 3,
                                             sizeof(example_scale_pc), { 256, 1, 1 });
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    const uint32_t         ne = (uint32_t) ggml_nelements(dst);
    const example_scale_pc pc = { ne, get_misalign_bytes(ctx, a) / 4, get_misalign_bytes(ctx, b) / 4, get_misalign_bytes(ctx, dst) / 4,
                                  ggml_get_op_params_f32(scale, 0), ggml_get_op_params_f32(scale, 1) };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                              { ggml_vk_tensor_subbuffer(ctx, a, true), ggml_vk_tensor_subbuffer(ctx, b, true), ggml_vk_tensor_subbuffer(ctx, dst, true) }, pc,
                              { ne, 1, 1 });
}

GGML_HOOK_REGISTER_KERNEL([] {
    ggml_hook_kernel k = ggml_hook_op_kernel("example_vk_scale_f32", GGML_HOOK_BACKEND_VULKAN, GGML_OP_SCALE, 0, example_scale_supports, ggml_hook_fn(example_scale_compute));
    k.enabled_by_default = false;
    return k;
}());

GGML_HOOK_REGISTER_KERNEL([] {
    ggml_hook_kernel k = ggml_hook_fusion_kernel("example_vk_scale_add_f32", GGML_HOOK_BACKEND_VULKAN, GGML_OP_SCALE, 0, example_scale_add_match, ggml_hook_fn(example_scale_add_compute));
    k.enabled_by_default = false;
    return k;
}());
