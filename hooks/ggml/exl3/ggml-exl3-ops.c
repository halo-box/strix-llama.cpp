// The EXL3 ops of the fork, as GGML_OP_CUSTOM nodes. The builders are the fork ones (ggml/src/ggml.c after upstream
// line 6547, fork 1fe05fa45). Adapted: result->op = GGML_OP_EXL3_* becomes a custom node whose function identifies the
// op, and op_params[0] of the fork moves after the custom op params (GGML_HOOK_EXL3_PARAM).

#include "ggml.h"
#include "ggml-impl.h"

#define GGML_HOOK_EXL3_PARAM 6 // i32 slot of the fork parameter: after struct ggml_custom_op_params

static_assert(sizeof(struct ggml_custom_op_params) <= GGML_HOOK_EXL3_PARAM*sizeof(int32_t), "custom op params overlap the EXL3 parameter");

// The function of the custom node. It runs only if no hook kernel took the node: as the fork without GGML_EXL3.
static void exl3_mul_mat_fn(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(dst); GGML_UNUSED(ith); GGML_UNUSED(nth); GGML_UNUSED(userdata);
    GGML_ABORT("EXL3_MUL_MAT needs a build with GGML_EXL3=ON");
}

static void exl3_get_rows_fn(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(dst); GGML_UNUSED(ith); GGML_UNUSED(nth); GGML_UNUSED(userdata);
    GGML_ABORT("EXL3_GET_ROWS needs a build with GGML_EXL3=ON");
}

static void exl3_mul_mat_id_fn(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(dst); GGML_UNUSED(ith); GGML_UNUSED(nth); GGML_UNUSED(userdata);
    GGML_ABORT("EXL3_MUL_MAT_ID needs a build with GGML_EXL3=ON");
}

static ggml_custom_op_t exl3_fn(enum ggml_hook_exl3_op_kind kind) {
    switch (kind) {
        case GGML_HOOK_EXL3_MUL_MAT:    return exl3_mul_mat_fn;
        case GGML_HOOK_EXL3_GET_ROWS:   return exl3_get_rows_fn;
        case GGML_HOOK_EXL3_MUL_MAT_ID: return exl3_mul_mat_id_fn;
        default:                        return NULL;
    }
}

static void ggml_hook_exl3_set(struct ggml_tensor * result, enum ggml_hook_exl3_op_kind kind, int32_t param) {
    struct ggml_custom_op_params p = { exl3_fn(kind), GGML_N_TASKS_MAX, NULL };
    memcpy(result->op_params, &p, sizeof(p));
    ggml_set_op_params_i32(result, GGML_HOOK_EXL3_PARAM, param);
    result->op = GGML_OP_CUSTOM;
}

enum ggml_hook_exl3_op_kind ggml_hook_exl3_op(const struct ggml_tensor * t) {
    if (t->op != GGML_OP_CUSTOM) {
        return GGML_HOOK_EXL3_NONE;
    }
    struct ggml_custom_op_params p;
    memcpy(&p, t->op_params, sizeof(p));
    for (int k = GGML_HOOK_EXL3_MUL_MAT; k <= GGML_HOOK_EXL3_MUL_MAT_ID; ++k) {
        if (p.fun == exl3_fn((enum ggml_hook_exl3_op_kind) k)) {
            return (enum ggml_hook_exl3_op_kind) k;
        }
    }
    return GGML_HOOK_EXL3_NONE;
}

int32_t ggml_hook_exl3_param(const struct ggml_tensor * t) {
    return ggml_get_op_params_i32(t, GGML_HOOK_EXL3_PARAM);
}

bool ggml_hook_is_hook_custom(const struct ggml_tensor * t) {
    return ggml_hook_exl3_op(t) != GGML_HOOK_EXL3_NONE;
}

// ggml_exl3_mul_mat

struct ggml_tensor * ggml_exl3_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * trellis,
        struct ggml_tensor  * x,
        struct ggml_tensor  * suh,
        struct ggml_tensor  * svh,
        int                   codebook) {
    GGML_ASSERT(trellis->type == GGML_TYPE_I16 && ggml_is_contiguous(trellis) && trellis->ne[3] == 1);
    GGML_ASSERT(suh->type == GGML_TYPE_F16 && svh->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(codebook >= 0 && codebook <= 2);

    // tiles are 16x16 and the Hadamard blocks 128 wide
    const int64_t k = trellis->ne[2]*16;
    const int64_t n = trellis->ne[1]*16;
    GGML_ASSERT(k % 128 == 0 && n % 128 == 0);
    GGML_ASSERT(ggml_is_vector(suh) && ggml_is_contiguous(suh) && suh->ne[0] == k);
    GGML_ASSERT(ggml_is_vector(svh) && ggml_is_contiguous(svh) && svh->ne[0] == n);
    GGML_ASSERT(x->ne[0] == k);

    const int64_t ne[4] = { n, x->ne[1], x->ne[2], x->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_hook_exl3_set(result, GGML_HOOK_EXL3_MUL_MAT, codebook); // adapted: a GGML_OP_CUSTOM node
    result->src[0] = trellis;
    result->src[1] = x;
    result->src[2] = suh;
    result->src[3] = svh;

    return result;
}

// ggml_exl3_mul_mat_id

struct ggml_tensor * ggml_exl3_mul_mat_id(
        struct ggml_context * ctx,
        struct ggml_tensor  * trellis,
        struct ggml_tensor  * x,
        struct ggml_tensor  * ids,
        struct ggml_tensor  * suh,
        struct ggml_tensor  * svh,
        int                   codebook) {
    GGML_ASSERT(trellis->type == GGML_TYPE_I16 && ggml_is_contiguous(trellis));
    GGML_ASSERT(suh->type == GGML_TYPE_F16 && svh->type == GGML_TYPE_F16 && ggml_is_contiguous(suh) && ggml_is_contiguous(svh));
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->ne[3] == 1);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[2] == 1 && ids->ne[3] == 1);
    GGML_ASSERT(codebook >= 0 && codebook <= 2);

    // one EXL3 linear per expert, as ggml_exl3_mul_mat
    const int64_t k = trellis->ne[2]*16;
    const int64_t n = trellis->ne[1]*16;
    const int64_t E = trellis->ne[3];
    GGML_ASSERT(k % 128 == 0 && n % 128 == 0);
    GGML_ASSERT(suh->ne[0] == k && suh->ne[1] == E && suh->ne[2] == 1 && suh->ne[3] == 1);
    GGML_ASSERT(svh->ne[0] == n && svh->ne[1] == E && svh->ne[2] == 1 && svh->ne[3] == 1);
    // as ggml_mul_mat_id: an expert list per token, x broadcast over the slots
    GGML_ASSERT(x->ne[0] == k && ids->ne[1] == x->ne[2] && ids->ne[0] % x->ne[1] == 0);

    const int64_t ne[4] = { n, ids->ne[0], x->ne[2], 1 };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_hook_exl3_set(result, GGML_HOOK_EXL3_MUL_MAT_ID, codebook); // adapted: a GGML_OP_CUSTOM node
    result->src[0] = trellis;
    result->src[1] = x;
    result->src[2] = ids;
    result->src[3] = suh;
    result->src[4] = svh;

    return result;
}

// ggml_exl3_get_rows

struct ggml_tensor * ggml_exl3_get_rows(
        struct ggml_context * ctx,
        struct ggml_tensor  * table,
        struct ggml_tensor  * ids,
        struct ggml_tensor  * signs,
        int64_t               n_out) {
    GGML_ASSERT(table->type == GGML_TYPE_I16 && ggml_is_contiguous(table) && ggml_n_dims(table) <= 2);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ggml_is_vector(ids));
    GGML_ASSERT(signs->type == GGML_TYPE_F16 && ggml_is_contiguous(signs) && signs->ne[0] == 256 && ggml_n_dims(signs) <= 2);

    // a row is one ring of 1 + 16*K words per group of 256, K = 1..8
    const int64_t groups = signs->ne[1];
    const int64_t rw     = table->ne[0] % groups == 0 ? table->ne[0]/groups - 1 : 0;
    GGML_ASSERT(rw >= 16 && rw <= 128 && rw % 16 == 0);
    GGML_ASSERT(n_out > 0 && n_out <= groups*256);

    struct ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_out, ids->ne[0]);

    ggml_hook_exl3_set(result, GGML_HOOK_EXL3_GET_ROWS, (int32_t) n_out); // adapted: a GGML_OP_CUSTOM node
    result->src[0] = table;
    result->src[1] = ids;
    result->src[2] = signs;

    return result;
}

