#include "norm-gated.cuh"
#include "ggml-impl.h"

// One wave per row, with the same reduction tree as the block-per-row norm.
static __device__ __forceinline__ float xor_tree(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { v += __shfl_xor_sync(0xffffffff, v, off, 32); }
    return v;
}
template <bool GATE>
static __global__ void __launch_bounds__(256) rms_rows_f32(const float * x, const float * w, const float * z, float * dst,
        const int ncols, const uint3 row_div, const uint3 channel_div, const int64_t total_rows,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps) {
    const int lane = threadIdx.x & 31;
    const int64_t g = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    if (g >= total_rows) return;
    uint32_t row = uint32_t(g), channel = 0, sample = 0;
    if (uint64_t(row_div.z) != uint64_t(total_rows)) {
        const uint2 rc = fast_div_modulo(row, row_div);
        const uint2 cs = fast_div_modulo(rc.x, channel_div);
        row = rc.y; channel = cs.y; sample = cs.x;
    }
    x += sample * stride_sample + channel * stride_channel + row * stride_row;
    dst += g * ncols; if (GATE) z += g * ncols;
    float part[8];
#pragma unroll
    for (int wv = 0; wv < 8; ++wv) {
        const int col = 32 * wv + lane;
        const float xi = col < ncols ? x[col] : 0.0f;
        part[wv] = xor_tree(xi * xi);
    }
    float v = 0.0f;
#pragma unroll
    for (int wv = 0; wv < 8; ++wv) { v = lane == wv ? part[wv] : v; }
    const float tmp = xor_tree(v);
    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
#pragma unroll
    for (int wv = 0; wv < 8; ++wv) {
        const int col = 32 * wv + lane;
        if (col < ncols) {
            const float t = scale * x[col] * w[col];
            if (GATE) { const float s = 1.0f / (1.0f + expf(-z[col])); dst[col] = t * s; } else { dst[col] = t; }
        }
    }
}

static bool rows_shape_ok(const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * out) {
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32) return false;
    if (x->ne[0] > 256 || x->ne[0] % 32 != 0 || x->nb[0] != 4 || !ggml_is_contiguous(w) || !ggml_is_contiguous(out)) return false;
    if (w->ne[0] != x->ne[0] || w->ne[1] != 1 || w->ne[2] != 1 || w->ne[3] != 1) return false;
    const int64_t rows = x->ne[1] * x->ne[2] * x->ne[3];
    if (rows < 4096 || uint64_t(rows) > UINT32_MAX) return false;
    return ggml_nelements(out) == ggml_nelements(x);
}

static bool norm_overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b || !a->data || !b->data) return false;
    const uintptr_t av = uintptr_t(a->data), bv = uintptr_t(b->data);
    return av <= bv ? bv - av < ggml_nbytes(a) : av - bv < ggml_nbytes(b);
}

static bool norm_elementwise_alias(const ggml_tensor * input, const ggml_tensor * out) {
    return !norm_overlaps(input, out) || (input->data == out->data &&
            ggml_is_contiguous(input) && ggml_are_same_shape(input, out));
}

static bool norm_needs_staging(const ggml_cuda_norm_gated_match & m) {
    if (!norm_elementwise_alias(m.x, m.dst) || norm_overlaps(m.w, m.dst) ||
            (m.z && !norm_elementwise_alias(m.z, m.dst))) return true;
    if (m.pre >= 0) {
        const auto * projected = m.z->view_src ? m.z->view_src : m.z;
        if (norm_overlaps(m.x, projected) || norm_overlaps(m.w, projected)) return true;
    }
    return false;
}

int ggml_cuda_norm_gated_match_at(const ggml_cgraph * cgraph, int i, ggml_cuda_norm_gated_match & m) {
    if (i + 3 >= cgraph->n_nodes) return 0;
    const ggml_tensor * rms = cgraph->nodes[i], * mul = cgraph->nodes[i+1];
    if (rms->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL) return 0;
    int k = i + 2; m.pre = -1;
    while (k + 1 < cgraph->n_nodes && k < i + 6) {
        const ggml_op op = cgraph->nodes[k]->op;
        if (op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE) { ++k; continue; }
        if (op == GGML_OP_MUL_MAT && m.pre < 0) { m.pre = k; ++k; continue; }
        break;
    }
    if (k + 1 >= cgraph->n_nodes) return 0;
    const ggml_tensor * sig = cgraph->nodes[k]; ggml_tensor * out = cgraph->nodes[k+1];
    if (sig->op != GGML_OP_UNARY || out->op != GGML_OP_MUL) return 0;
    if (ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID) return 0;
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : (mul->src[1] == rms ? mul->src[0] : nullptr);
    if (!w) return 0;
    const ggml_tensor * z = sig->src[0];
    if (m.pre >= 0) { const ggml_tensor * zr = z->view_src ? z->view_src : z; if (zr != cgraph->nodes[m.pre]) return 0; }
    if (!((out->src[0] == mul && out->src[1] == sig) || (out->src[0] == sig && out->src[1] == mul))) return 0;
    if (!ggml_node_has_n_uses(cgraph, i, 1) || !ggml_node_has_n_uses(cgraph, i+1, 1) || !ggml_node_has_n_uses(cgraph, k, 1)) return 0;
    if (!rows_shape_ok(x, w, out) || z->type != GGML_TYPE_F32 || !ggml_is_contiguous(z) || !ggml_is_contiguous(mul)) return 0;
    if (!ggml_are_same_shape(z, out) || !ggml_are_same_shape(mul, out) || !ggml_are_same_shape(rms, out)) return 0;
    float eps; memcpy(&eps, rms->op_params, sizeof(float)); if (eps < 0.0f) return 0;
    m.x = x; m.w = w; m.z = z; m.dst = out; m.eps = eps;
    m.staged = norm_needs_staging(m);
    return (k + 1) - i;
}

// plain RMS_NORM -> MUL(w) on narrow rows
int ggml_cuda_norm_rows_match_at(const ggml_cgraph * cgraph, int i, ggml_cuda_norm_gated_match & m) {
    if (i + 1 >= cgraph->n_nodes) return 0;
    const ggml_tensor * rms = cgraph->nodes[i]; ggml_tensor * mul = cgraph->nodes[i+1];
    if (rms->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL) return 0;
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : (mul->src[1] == rms ? mul->src[0] : nullptr);
    if (!w || !ggml_node_has_n_uses(cgraph, i, 1)) return 0;
    if (!rows_shape_ok(x, w, mul) || !ggml_are_same_shape(rms, mul)) return 0;
    float eps; memcpy(&eps, rms->op_params, sizeof(float)); if (eps < 0.0f) return 0;
    m.x = x; m.w = w; m.z = nullptr; m.dst = mul; m.eps = eps; m.pre = -1;
    m.staged = norm_needs_staging(m);
    return 1;
}

void ggml_cuda_op_norm_gated(ggml_backend_cuda_context & ctx, const ggml_cuda_norm_gated_match & m) {
    const ggml_tensor * x = m.x;
    const int64_t total = x->ne[1] * x->ne[2] * x->ne[3];
    const uint3 row_div = init_fastdiv_values(uint32_t(x->ne[1]));
    const uint3 channel_div = init_fastdiv_values(uint32_t(x->ne[2]));
    const dim3 grid((unsigned) ((total + 7) / 8)), block(256);
    const ggml_cuda_kernel_launch_params lp(grid, block, 0, ctx.stream());
    if (m.z) ggml_cuda_kernel_launch(rms_rows_f32<true>, lp, (const float *) x->data, (const float *) m.w->data, (const float *) m.z->data, (float *) m.dst->data,
        (int) x->ne[0], row_div, channel_div, total, (int64_t) (x->nb[1] / 4), (int64_t) (x->nb[2] / 4), (int64_t) (x->nb[3] / 4), m.eps);
    else     ggml_cuda_kernel_launch(rms_rows_f32<false>, lp, (const float *) x->data, (const float *) m.w->data, (const float *) nullptr, (float *) m.dst->data,
        (int) x->ne[0], row_div, channel_div, total, (int64_t) (x->nb[1] / 4), (int64_t) (x->nb[2] / 4), (int64_t) (x->nb[3] / 4), m.eps);
}
