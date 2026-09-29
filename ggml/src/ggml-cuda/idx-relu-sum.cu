#include "idx-relu-sum.cuh"

// Preserve the head addition order of the RELU/CONT/ADD graph.
// With tails/starts it also applies the compact QSA visibility exactly as CPY(I32->F32), REPEAT, SUB, STEP, LOG
// and ADD compute it, so the result is bit-identical to the unfused graph.
static __global__ void idx_relu_sum_f32(const float * __restrict__ src, float * __restrict__ dst,
                                        const int n_blocks, const int heads,
                                        const int32_t * __restrict__ tails, const int32_t * __restrict__ starts) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const int64_t tt = blockIdx.y;                         // token within stream, streams stacked
    const float * p  = src + tt * (int64_t) heads * n_blocks + b;
    float acc = fmaxf(p[0], 0.0f);
    for (int h = 1; h < heads; ++h) {
        acc = acc + fmaxf(p[(int64_t) h * n_blocks], 0.0f);
    }
    if (tails) {
        const float d    = (float) tails[tt] - (float) starts[b];
        const float step = d > 0.0f;
        acc = acc + logf(step);
    }
    dst[tt * (int64_t) n_blocks + b] = acc;
}

void ggml_cuda_op_idx_relu_sum(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_relu_sum_args & args) {
    const ggml_tensor * s = args.score;
    const int n_blocks = (int) s->ne[0];
    const int64_t rows = s->ne[2] * s->ne[3];
    constexpr int threads = 256;
    const dim3 grid((n_blocks + threads - 1) / threads, (unsigned) rows, 1);
    const int32_t * tails  = args.vis_tails  ? (const int32_t *) args.vis_tails->data  : nullptr;
    const int32_t * starts = args.vis_starts ? (const int32_t *) args.vis_starts->data : nullptr;
    idx_relu_sum_f32<<<grid, threads, 0, ctx.stream()>>>((const float *) s->data, (float *) args.dst->data, n_blocks, args.heads,
        tails, starts);
    CUDA_CHECK(cudaGetLastError());
}
