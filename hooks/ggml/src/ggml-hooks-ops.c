// Builders of the fork that upstream ggml.c does not have, or has with fewer types. Verbatim from ggml/src/ggml.c of
// the fork (1fe05fa45, upstream 9048bad8d):
// - ggml_flash_attn_ext_set/get_causal_prefix and ggml_flash_attn_ext_add_top_k (after upstream lines 5506, 5522);
// - ggml_dsv4_hc_pre_impl and its three builders (fork line 6614): the gated form also takes f16 inputs and may
//   write f16. ggml_dsv4_hc_pre and ggml_dsv4_hc_pre_gated of upstream are renamed (hooks_rename).

#include "ggml.h"
#include "ggml-impl.h"

void ggml_flash_attn_ext_set_causal_prefix(
        struct ggml_tensor * a,
        bool                 causal_prefix) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    ggml_set_op_params_i32(a, 5, causal_prefix ? 1 : 0);
}

bool ggml_flash_attn_ext_get_causal_prefix(
        const struct ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    return ggml_get_op_params_i32(a, 5) != 0;
}

void ggml_flash_attn_ext_add_top_k(
        struct ggml_tensor * a,
        struct ggml_tensor * top_k,
                 int64_t     n_kv_raw) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    // src[5] is kv_rows (ggml_flash_attn_ext_rows): the selected-key list indexes the dense K/V view, so the two do not mix
    GGML_ASSERT(a->src[5] == NULL);
    GGML_ASSERT(a->src[6] == NULL);
    GGML_ASSERT(top_k->type == GGML_TYPE_I32);
    GGML_ASSERT(top_k->ne[1] == a->src[0]->ne[1]);
    GGML_ASSERT(n_kv_raw >= 0 && n_kv_raw <= a->src[1]->ne[1]);

    a->src[6] = top_k;
    ggml_set_op_params_i32(a, 4, (int32_t) n_kv_raw);
}

static struct ggml_tensor * ggml_dsv4_hc_pre_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * weights,
        float                 scale,
        bool                  gated,
        enum ggml_type        out_type) {
    // the gated form may write an f16 mixed stream (every consumer is a matmul B operand)
    GGML_ASSERT(out_type == GGML_TYPE_F32 || (gated && out_type == GGML_TYPE_F16));
    // the gated form also takes an f16 x and an f16 gate (backends opt in through supports_op)
    GGML_ASSERT(x->type == GGML_TYPE_F32 || (gated && x->type == GGML_TYPE_F16));
    GGML_ASSERT(weights->type == GGML_TYPE_F32 || (gated && weights->type == GGML_TYPE_F16));

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[3] == 1);
    if (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } else {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        GGML_ASSERT(weights->ne[2] == 1);
    }
    GGML_ASSERT(weights->ne[3] == 1);

    struct ggml_tensor * result = ggml_new_tensor_2d(ctx, out_type, n_embd, n_tokens);

    ggml_set_op_params_f32(result, 0, scale);
    ggml_set_op_params_i32(result, 1, gated ? 1 : 0);

    result->op     = GGML_OP_DSV4_HC_PRE;
    result->src[0] = x;
    result->src[1] = weights;

    return result;
}

struct ggml_tensor * ggml_dsv4_hc_pre(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * weights) {
    return ggml_dsv4_hc_pre_impl(ctx, x, weights, 1.0f, false, GGML_TYPE_F32);
}

struct ggml_tensor * ggml_dsv4_hc_pre_gated(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * gate,
        float                 scale) {
    return ggml_dsv4_hc_pre_impl(ctx, x, gate, scale, true, GGML_TYPE_F32);
}

struct ggml_tensor * ggml_dsv4_hc_pre_gated_ext(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * gate,
        float                 scale,
        enum ggml_type        out_type) {
    return ggml_dsv4_hc_pre_impl(ctx, x, gate, scale, true, out_type);
}

// the check of gguf_init_from_reader in ggml/src/gguf.cpp of the fork (after upstream line 735), called by HOOK(gguf-type)
bool ggml_hook_gguf_type_unassigned(const char * name, enum ggml_type type) {
            // ids inside the range can still be unassigned (e.g. 142, PQ2_0, which is not ported)
            if (ggml_blck_size(type) == 0) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d (unassigned type id)\n",
                    "gguf_init_from_reader", name, type);
                return true;
            }
            return false;
}
