// The alloc size of GGML_OP_SPARSE_ATTN as an RPC server computes it (ggml-rpc.cpp rpc_server::get_alloc_size): the server
// deserializes the node and its direct srcs only, so the plan (src[3]) arrives with its op, op_params and shape but with
// src[0] == NULL. The workspace past ggml_nbytes(dst) must not depend on that missing source of a source: for every device of
// the build (CPU, Vulkan, HIP) and every shape the device takes, the size of the server's view must equal the size of the
// fully linked node. Before the fix the Vulkan select dereferenced the NULL selection (a server crash on the first query) and
// the HIP select returned "malformed node", so its alloc size fell back to ggml_nbytes(dst) and the client reserved no
// workspace (a kernel write past the allocation). Without such a device the CPU rows pass trivially.
#include "ggml.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>

static int g_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); ++g_fail; } } while (0)

// what deserialize_tensor rebuilds: type, ne, nb, op, op_params and flags; no src, no view, no data
static ggml_tensor * rpc_copy(ggml_context * ctx, const ggml_tensor * t) {
    ggml_tensor * c = ggml_new_tensor(ctx, t->type, GGML_MAX_DIMS, t->ne);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        c->nb[i] = t->nb[i];
    }
    c->op = t->op;
    memcpy(c->op_params, t->op_params, sizeof(c->op_params));
    c->flags = t->flags;
    return c;
}

// rpc_server::get_alloc_size: the node, then its direct srcs linked to it (and nothing below them)
static ggml_tensor * rpc_server_view(ggml_context * ctx, const ggml_tensor * t) {
    ggml_tensor * c = rpc_copy(ctx, t);
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (t->src[i]) {
            c->src[i] = rpc_copy(ctx, t->src[i]);
        }
    }
    return c;
}

int main() {
    ggml_backend_load_all();
    int n_rows = 0;
    for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        printf("device %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        for (ggml_type t : { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 }) {
            for (int32_t G : { 1, 4 }) {
                for (int64_t n_q : { 3, 9, 512 }) {
                    ggml_init_params ip = { 64 * ggml_tensor_overhead(), nullptr, true };
                    ggml_context * ctx = ggml_init(ip);
                    const int64_t n_kv = 8192, n_sel = 2051, kv_head = 2, gqa = 12;
                    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, n_q, kv_head * gqa, 1);
                    ggml_tensor * k    = ggml_new_tensor_4d(ctx, t, 256, n_kv, kv_head, 1);
                    ggml_tensor * v    = ggml_new_tensor_4d(ctx, t, 256, n_kv, kv_head, 1);
                    ggml_tensor * sel  = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, n_sel, n_q, 1, 1);
                    ggml_tensor * plan = ggml_sparse_attn_plan(ctx, sel, n_kv, G);
                    ggml_tensor * attn = ggml_sparse_attn(ctx, q, k, v, plan, 0.0625f, 0, 0);
                    if (!ggml_backend_dev_supports_op(dev, attn) || !ggml_backend_dev_supports_op(dev, plan)) {
                        ggml_free(ctx);
                        continue;
                    }
                    const size_t a_linked = ggml_backend_buft_get_alloc_size(buft, attn);
                    const size_t p_linked = ggml_backend_buft_get_alloc_size(buft, plan);
                    ggml_tensor * attn_v = rpc_server_view(ctx, attn);
                    ggml_tensor * plan_v = rpc_server_view(ctx, plan);
                    const size_t a_view = ggml_backend_buft_get_alloc_size(buft, attn_v);
                    const size_t p_view = ggml_backend_buft_get_alloc_size(buft, plan_v);
                    printf("  %-4s G %d n_q %-3" PRId64 ": attend nbytes %zu, alloc linked %zu, server view %zu; plan alloc linked %zu, view %zu\n",
                           ggml_type_name(t), G, n_q, ggml_nbytes(attn), a_linked, a_view, p_linked, p_view);
                    CHECK(a_view == a_linked, "%s: attend alloc size of the server's view %zu != linked %zu", ggml_backend_dev_name(dev), a_view, a_linked);
                    CHECK(p_view == p_linked, "%s: plan alloc size of the server's view %zu != linked %zu", ggml_backend_dev_name(dev), p_view, p_linked);
                    n_rows++;
                    ggml_free(ctx);
                }
            }
        }
    }
    printf("%d rows, %d failed checks: %s\n", n_rows, g_fail, g_fail == 0 ? "OK" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}
