// Hook kernels of the fork flash attention and DSV4 hc_pre on the CPU (fork-fa.inc). On by default: upstream does not
// run these cases. The flash attention kernel takes the nodes with selected keys (src[6], ggml_flash_attn_ext_add_top_k),
// which upstream computes densely through the mask; hc_pre takes the f16 forms of ggml_dsv4_hc_pre_gated_ext.

#include "ggml-hooks-cpu.h"

#include "common.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ops.h"
#include "simd-gemm.h"
#include "simd-mappings.h"
#include "vec.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>

#include "fork-fa.inc"

namespace {

bool fa_supports(const ggml_hook_dev *, const ggml_tensor * t) {
    return t->op == GGML_OP_FLASH_ATTN_EXT && t->src[6] != nullptr;
}

void fa_compute(const ggml_compute_params * params, ggml_tensor * dst) {
    fork_flash_attn_ext(params, dst);
}

// the GGML_OP_FLASH_ATTN_EXT case of ggml_graph_plan in the fork (ggml/src/ggml-cpu/ggml-cpu.c line 3317), n_tasks = n_threads
size_t fa_work_size(int n_tasks, const ggml_tensor * node) {
    const int64_t neq2 = node->src[0]->ne[2]; // number of query heads
    const int64_t DK = node->src[1]->ne[0];
    const int64_t DV = node->src[2]->ne[0];

    // Tiled flash attention scratch (tile sizes defined in common.h)
    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
    size_t prefill  = sizeof(float)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
    size_t n_chunks = n_tasks;
    size_t decode   = sizeof(float)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

    // selected-key rows (src[6]) are gathered per thread after the decode partials
    if (node->src[6]) {
        decode += sizeof(float)*n_tasks*CACHE_LINE_SIZE_F32 + sizeof(int32_t)*n_tasks*node->src[6]->ne[0];
    }

    return MAX(prefill, decode);
}

bool hc_pre_supports(const ggml_hook_dev *, const ggml_tensor * t) {
    return t->op == GGML_OP_DSV4_HC_PRE &&
           (t->type == GGML_TYPE_F16 || t->src[0]->type == GGML_TYPE_F16 || t->src[1]->type == GGML_TYPE_F16);
}

void hc_pre_compute(const ggml_compute_params * params, ggml_tensor * dst) {
    fork_dsv4_hc_pre(params, dst);
}

} // namespace

GGML_HOOK_REGISTER_KERNEL([] {
    auto k = ggml_hook_op_kernel("fork_cpu_fa_selected_keys", GGML_HOOK_BACKEND_CPU, GGML_OP_FLASH_ATTN_EXT, 0, fa_supports,
                                 ggml_hook_fn(static_cast<ggml_hook_cpu_op_fn>(fa_compute)));
    k.cpu_work_size = fa_work_size;
    return k;
}());

GGML_HOOK_REGISTER_KERNEL(ggml_hook_op_kernel("fork_cpu_dsv4_hc_pre_f16", GGML_HOOK_BACKEND_CPU, GGML_OP_DSV4_HC_PRE, 0, hc_pre_supports,
                                              ggml_hook_fn(static_cast<ggml_hook_cpu_op_fn>(hc_pre_compute))));
