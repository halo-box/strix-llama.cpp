#include "gdn-conv.cuh"
#include "conv-tail.cuh"
#include "unary.cuh"

// columns [tail_from, T+3) of every channel row of the concat: what the history copies read
static __global__ void gdn_concat_tail(const float * __restrict__ state, const float * __restrict__ x, float * __restrict__ out,
                                       const int C, const int T, const int tail_from, const int row_stride) {
    const int ncols = T + 3 - tail_from;
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t) C * ncols) {
        return;
    }
    const int c = (int) (idx / ncols), j = tail_from + (int) (idx % ncols);
    out[(size_t) c * row_stride + j] = (j < 3) ? state[c * 3 + j] : x[(size_t) (j - 3) * C + c];
}

// the same accumulation as ssm_conv_f32 with a zero bias, then SiLU
template <int TT>
static __global__ void __launch_bounds__(256) gdn_conv_direct_kernel(const float * __restrict__ state, const float * __restrict__ x,
        const float * __restrict__ w, float * __restrict__ y, const int C, const int T) {
    const int c  = blockIdx.x * 256 + threadIdx.x;
    const int t0 = blockIdx.y * TT;
    if (c >= C) {
        return;
    }
    float wr[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        wr[j] = w[c * 4 + j];
    }
    float xw[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int jp = t0 + k;
        xw[k] = (jp < 3) ? state[c * 3 + jp] : x[(size_t) (jp - 3) * C + c];
    }
    const float b = 0.0f;
    const int tend = min(T, t0 + TT);
    for (int t = t0; t < tend; ++t) {
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumf += xw[j] * wr[j];
        }
        sumf += b;
        y[(size_t) t * C + c] = ggml_cuda_op_silu_single(sumf);
        xw[0] = xw[1]; xw[1] = xw[2]; xw[2] = xw[3];
        const int jp = t + 4;
        xw[3] = (jp < 3) ? state[c * 3 + jp] : ((jp - 3) < T ? x[(size_t) (jp - 3) * C + c] : 0.0f);
    }
}

static bool gdn_conv_check(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m) {
    if (i < 0 || i + 2 >= cgraph->n_nodes) {
        return ggml_cuda_conv_reject("gdn_conv", "check 1", i);
    }
    const ggml_tensor * cc = cgraph->nodes[i];
    if (cc->op != GGML_OP_CONCAT || cc->type != GGML_TYPE_F32 || ggml_get_op_params_i32(cc, 0) != 0 ||
        (cc->flags & GGML_TENSOR_FLAG_OUTPUT) || !ggml_cuda_conv_next_is_noop(cgraph, i)) {
        return ggml_cuda_conv_reject("gdn_conv", "check 2", i);
    }
    const ggml_tensor * st = cc->src[0];
    const ggml_tensor * tr = cc->src[1];
    if (!st || !tr || st->type != GGML_TYPE_F32 || tr->type != GGML_TYPE_F32 || tr->op != GGML_OP_TRANSPOSE || !tr->view_src) {
        return ggml_cuda_conv_reject("gdn_conv", "check 3", i);
    }
    int64_t C = 0, T = 0;
    const ggml_tensor * x = ggml_cuda_conv_activation(tr, C, T);
    if (!x) {
        return ggml_cuda_conv_reject("gdn_conv", "check 4", i);
    }
    if (st->ne[0] != 3 || st->ne[1] != C || st->ne[2] != 1 || st->ne[3] != 1 || !ggml_is_contiguous(st)) {
        return ggml_cuda_conv_reject("gdn_conv", "check 5", i);
    }
    if (tr->ne[0] != T || tr->ne[1] != C || tr->ne[2] != 1 || tr->view_offs != 0) {
        return ggml_cuda_conv_reject("gdn_conv", "check 6", i);
    }
    if (cc->ne[0] != T + 3 || cc->ne[1] != C || cc->ne[2] != 1 || cc->ne[3] != 1 || !ggml_is_contiguous(cc) || T < 256 || C % 256 != 0) {
        return ggml_cuda_conv_reject("gdn_conv", "check 7", i);
    }

    // the conv is the only reader of the whole concat; everything else must be a tail reader
    int64_t tail_from = T + 3;
    int conv = -1;
    for (int n = i + 1; n < cgraph->n_nodes; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        if (!ggml_cuda_reads(t, cc)) {
            continue;
        }
        if (t->op == GGML_OP_SSM_CONV && ggml_cuda_conv_whole(t->src[0], cc)) {
            if (conv >= 0) {
                return ggml_cuda_conv_reject("gdn_conv", "check 8", n);
            }
            conv = n;
            continue;
        }
        if (ggml_cuda_conv_whole(t, cc) && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW)) {
            continue; // shape plumbing into the conv; its readers are checked as readers of the concat
        }
        if (!ggml_cuda_conv_tail_reader(t, cc, 3, tail_from)) {
            return ggml_cuda_conv_reject("gdn_conv", "check 9", n);
        }
    }
    if (conv < 0 || conv + 1 >= cgraph->n_nodes) {
        return ggml_cuda_conv_reject("gdn_conv", "check 10", i);
    }
    const ggml_tensor * cv = cgraph->nodes[conv];
    const ggml_tensor * w  = cv->src[1];
    if (cv->type != GGML_TYPE_F32 || cv->src[2] != nullptr || !w || w->type != GGML_TYPE_F32 || w->ne[0] != 4 || w->ne[1] != C ||
        !ggml_is_contiguous(w) || cv->ne[0] != C || cv->ne[1] != T || cv->ne[2] != 1 || cv->ne[3] != 1 || !ggml_is_contiguous(cv) ||
        (cv->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return ggml_cuda_conv_reject("gdn_conv", "check 11", i);
    }
    const ggml_tensor * su = cgraph->nodes[conv + 1];
    if (su->op != GGML_OP_UNARY || ggml_get_unary_op(su) != GGML_UNARY_OP_SILU || su->src[0] != cv || su->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(su) || ggml_cuda_conv_readers(cgraph, conv + 1, cv, nullptr) != 1) {
        return ggml_cuda_conv_reject("gdn_conv", "check 12", i);
    }
    m.concat_idx = i; m.conv_idx = conv; m.x = x; m.state = st; m.concat = cc; m.w = w; m.out = cgraph->nodes[conv + 1];
    m.C = C; m.T = T; m.tail_from = std::max<int64_t>(0, tail_from);
    return true;
}

bool ggml_cuda_gdn_conv_match_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m) {
    return gdn_conv_check(cgraph, i, m);
}

bool ggml_cuda_gdn_conv_match_at_conv(const ggml_cgraph * cgraph, int j, ggml_cuda_gdn_conv_match & m) {
    if (j < 1 || cgraph->nodes[j]->op != GGML_OP_SSM_CONV) {
        return false;
    }
    // the conv reads the concat directly or through a whole-tensor view; both entry points must find the same node
    const ggml_tensor * src = cgraph->nodes[j]->src[0];
    const ggml_tensor * cc  = src ? (src->op == GGML_OP_CONCAT ? src : src->view_src) : nullptr;
    if (!cc || cc->op != GGML_OP_CONCAT || !ggml_cuda_conv_whole(src, cc)) {
        return false;
    }
    for (int i = j - 1; i >= 0; --i) {
        if (cgraph->nodes[i] == cc) {
            return gdn_conv_check(cgraph, i, m) && m.conv_idx == j;
        }
    }
    return false;
}

void ggml_cuda_gdn_conv_write_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m) {
    const int ncols = (int) (m.T + 3 - m.tail_from);
    if (ncols <= 0) {
        return;
    }
    const int64_t n = m.C * ncols;
    gdn_concat_tail<<<(unsigned) ((n + 255) / 256), 256, 0, ctx.stream()>>>((const float *) m.state->data, (const float *) m.x->data,
        (float *) m.concat->data, (int) m.C, (int) m.T, (int) m.tail_from, (int) (m.concat->nb[1] / sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_gdn_conv_direct(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m) {
    constexpr int TT = 128;
    dim3 grid((unsigned) (m.C / 256), (unsigned) ((m.T + TT - 1) / TT));
    gdn_conv_direct_kernel<TT><<<grid, 256, 0, ctx.stream()>>>((const float *) m.state->data, (const float *) m.x->data,
        (const float *) m.w->data, (float *) m.out->data, (int) m.C, (int) m.T);
    CUDA_CHECK(cudaGetLastError());
}
