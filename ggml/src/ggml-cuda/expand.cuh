#pragma once
#include "ggml.h"
struct ggml_backend_cuda_context;
struct ggml_cuda_qsa_expand_args {
    const ggml_tensor * blocks;
    const ggml_tensor * scores;
    const ggml_tensor * cells;
    const ggml_tensor * tail;
    ggml_tensor * cast_blocks;
    ggml_tensor * cast_cells;
    ggml_tensor * expanded;
    ggml_tensor * output;
};
void ggml_cuda_op_qsa_expand(ggml_backend_cuda_context &, const ggml_cuda_qsa_expand_args &);
