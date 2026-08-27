#pragma once

#include "q7-panel16.cuh"

// Returns the phase-one gfx1151 panel cache only for an exact, valid,
// canonical-base weight tensor in a normal CUDA/HIP buffer. All unsupported
// storage, view, mutation, and upload cases return nullptr.
const block_rocmfp7_panel16 * ggml_cuda_q7_panel16_cache_get(
        const ggml_tensor * tensor);

// Process-wide MMQ instrumentation. These are deliberately non-inline so
// template instantiations in separate translation units share one set of
// counters and one first-launch log.
void ggml_cuda_q7_panel16_note_mmq_handoff(
        const ggml_tensor * tensor);
void ggml_cuda_q7_panel16_note_kernel_launch(
        int mmq_x,
        int64_t nrows_x,
        int64_t ncols_x,
        int64_t ncols_dst);
uint64_t ggml_cuda_q7_panel16_mmq_handoff_count();
uint64_t ggml_cuda_q7_panel16_kernel_launch_count();
