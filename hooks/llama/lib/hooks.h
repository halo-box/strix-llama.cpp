// The hooks side of the fork libllama (hooks/llama/lib): what the fork code needs from the hooks build. Included
// at the end of the fork llama-impl.h.
#pragma once

#include "ggml.h"

namespace ggml_fork {

// the EXL3 ops of the fork are GGML_OP_CUSTOM nodes (hooks/ggml/exl3/ggml-exl3-ops.c): pseudo op ids for the fork code
#define GGML_OP_EXL3_MUL_MAT    ((ggml_op) (GGML_OP_COUNT + 1))
#define GGML_OP_EXL3_GET_ROWS   ((ggml_op) (GGML_OP_COUNT + 2))
#define GGML_OP_EXL3_MUL_MAT_ID ((ggml_op) (GGML_OP_COUNT + 3))

static inline ggml_op ggml_fork_op(const ggml_tensor * t) {
    switch (ggml_hook_exl3_op(t)) {
        case GGML_HOOK_EXL3_MUL_MAT:    return GGML_OP_EXL3_MUL_MAT;
        case GGML_HOOK_EXL3_GET_ROWS:   return GGML_OP_EXL3_GET_ROWS;
        case GGML_HOOK_EXL3_MUL_MAT_ID: return GGML_OP_EXL3_MUL_MAT_ID;
        default:                        return t->op;
    }
}

} // namespace ggml_fork
