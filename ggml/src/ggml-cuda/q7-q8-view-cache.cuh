#pragma once

#include "q7-q8-view.cuh"

// Returns a valid standard-Q8 compute view only for the explicit Qwen MMQ
// projection allowlist and a safe canonical-base contiguous Q7 tensor in a
// CUDA/HIP weights buffer. This includes routed-expert 3D weights, flattened
// only in the compute view while preserving expert-channel strides. Embedding
// lookup storage is deliberately excluded; GET_ROWS and MMVQ always consume
// the canonical Q7 representation.
const block_q8_0 * ggml_cuda_q7_q8_view_get(
        const ggml_tensor * tensor);

void ggml_cuda_q7_q8_view_note_mmq_handoff(
        const ggml_tensor * tensor);
void ggml_cuda_q7_q8_view_note_kernel_dispatch(
        const ggml_tensor * tensor,
        int64_t nrows_x,
        int64_t ncols_x,
        int64_t ncols_dst);
uint64_t ggml_cuda_q7_q8_view_mmq_handoff_count();
uint64_t ggml_cuda_q7_q8_view_kernel_dispatch_count();
uint64_t ggml_cuda_q7_q8_view_output_dispatch_count();
