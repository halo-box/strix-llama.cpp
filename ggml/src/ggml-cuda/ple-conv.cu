#include "ple-conv.cuh"
#include "conv-tail.cuh"
#include "unary.cuh"

#if defined(__HIP_PLATFORM_AMD__)
// the unfused graph multiplies in one kernel and adds in another: keep the two roundings apart
static __device__ __forceinline__ float ple_mul_rn(const float a, const float b) { float r; asm("v_mul_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r; }
static __device__ __forceinline__ float ple_add_rn(const float a, const float b) { float r; asm("v_add_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r; }
#else
static __device__ __forceinline__ float ple_mul_rn(const float a, const float b) { return __fmul_rn(a, b); }
static __device__ __forceinline__ float ple_add_rn(const float a, const float b) { return __fadd_rn(a, b); }
#endif

// columns [tail_from, T+H) of every channel row of the concat: what the history copies read
static __global__ void ple_concat_tail(const float * __restrict__ state, const float * __restrict__ x, float * __restrict__ out,
                                       const int C, const int T, const int H, const int tail_from, const int row_stride) {
    const int ncols = T + H - tail_from;
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t) C * ncols) {
        return;
    }
    const int c = (int) (idx / ncols), j = tail_from + (int) (idx % ncols);
    out[(size_t) c * row_stride + j] = (j < H) ? state[c * H + j] : x[(size_t) (j - H) * C + c];
}

static __device__ __forceinline__ float ple_weight(const half w) { return __half2float(w); }
static __device__ __forceinline__ float ple_weight(const float w) { return w; }

template <int K, int DIL, int TT, typename TW>
static __global__ void __launch_bounds__(256) ple_conv_kernel(const float * __restrict__ state, const float * __restrict__ x,
        const TW * __restrict__ w, float * __restrict__ y, const int C, const int T) {
    constexpr int H = (K - 1) * DIL, WIN = H + 1;
    const int c = blockIdx.x * 256 + threadIdx.x, t0 = blockIdx.y * TT;
    if (c >= C) {
        return;
    }
    float wr[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        wr[k] = ple_weight(w[c * K + k]);
    }
    float win[WIN];
    auto ld = [&](int jp) -> float {
        return (jp < H) ? state[c * H + jp] : ((jp - H) < T ? x[(size_t) (jp - H) * C + c] : 0.0f);
    };
#pragma unroll
    for (int k = 0; k < WIN; ++k) {
        win[k] = ld(t0 + k);
    }
    const int tend = min(T, t0 + TT);
    for (int t = t0; t < tend; ++t) {
        float s = ple_mul_rn(win[0], wr[0]);
#pragma unroll
        for (int k = 1; k < K; ++k) {
            s = ple_add_rn(s, ple_mul_rn(win[k * DIL], wr[k]));
        }
        y[(size_t) t * C + c] = ggml_cuda_op_silu_single(s);
#pragma unroll
        for (int k = 0; k < WIN - 1; ++k) {
            win[k] = win[k + 1];
        }
        win[WIN - 1] = ld(t + WIN);
    }
}

// tap k: CONT(TRANSPOSE(VIEW of the concat at column k*dil, [T, C]))
static int ple_tap_index(const ggml_tensor * t, const ggml_tensor * cc, int64_t C, int64_t T, int64_t dil, int K) {
    if (t->op != GGML_OP_CONT || !t->src[0] || t->src[0]->op != GGML_OP_TRANSPOSE || t->type != GGML_TYPE_F32 ||
        t->ne[0] != C || t->ne[1] != T || t->ne[2] != 1 || t->ne[3] != 1 || !ggml_is_contiguous(t)) {
        return -1;
    }
    const ggml_tensor * v = t->src[0]->src[0];
    if (!v || v->op != GGML_OP_VIEW || v->view_src != cc || v->ne[0] != T || v->ne[1] != C || v->ne[2] != 1 ||
        v->nb[1] != cc->nb[1] || v->view_offs % (dil * sizeof(float)) != 0) {
        return -1;
    }
    const int64_t k = (int64_t) (v->view_offs / (dil * sizeof(float)));
    return (k >= 0 && k < K) ? (int) k : -1;
}

// the weight operand of a tap MUL: reshape <- CONT <- VIEW of column k of the [K, C] weight, behind a cast(F32) when it is F16
static const ggml_tensor * ple_weight_root(const ggml_tensor * wk, int64_t C, int k) {
    if (wk->type != GGML_TYPE_F32 || ggml_nelements(wk) != C) {
        return nullptr;
    }
    const bool cast = wk->op == GGML_OP_CPY;
    const ggml_tensor * src = cast ? wk->src[0] : wk;
    while (src && (src->op == GGML_OP_RESHAPE || src->op == GGML_OP_VIEW)) {
        src = src->src[0];
    }
    if (!src || src->op != GGML_OP_CONT) {
        return nullptr;
    }
    const ggml_tensor * wv = src->src[0];
    if (!wv || wv->op != GGML_OP_VIEW || !wv->view_src) {
        return nullptr;
    }
    const ggml_tensor * W = wv->view_src;
    if ((cast ? W->type != GGML_TYPE_F16 : W->type != GGML_TYPE_F32) || W->ne[1] != C || W->ne[2] != 1 || W->ne[3] != 1 ||
        !ggml_is_contiguous(W) || wv->ne[0] != 1 || wv->ne[1] != C || wv->nb[1] != W->nb[1] ||
        wv->view_offs != (size_t) k * ggml_element_size(W)) {
        return nullptr;
    }
    return W;
}

static bool ple_conv_check(const ggml_cgraph * cgraph, int i, ggml_cuda_ple_conv_match & m) {
    if (i < 0 || i + 2 >= cgraph->n_nodes) {
        return ggml_cuda_conv_reject("ple_conv", "check 1", i);
    }
    const ggml_tensor * cc = cgraph->nodes[i];
    if (cc->op != GGML_OP_CONCAT || cc->type != GGML_TYPE_F32 || ggml_get_op_params_i32(cc, 0) != 0 ||
        (cc->flags & GGML_TENSOR_FLAG_OUTPUT) || !ggml_cuda_conv_next_is_noop(cgraph, i)) {
        return ggml_cuda_conv_reject("ple_conv", "check 2", i);
    }
    const ggml_tensor * st = cc->src[0];
    const ggml_tensor * tr = cc->src[1];
    if (!st || !tr || st->type != GGML_TYPE_F32 || tr->type != GGML_TYPE_F32 || tr->op != GGML_OP_TRANSPOSE || !tr->view_src) {
        return ggml_cuda_conv_reject("ple_conv", "check 3", i);
    }
    int64_t C = 0, T = 0;
    const ggml_tensor * x = ggml_cuda_conv_activation(tr, C, T);
    if (!x) {
        return ggml_cuda_conv_reject("ple_conv", "check 4", i);
    }
    constexpr int K = 4, DIL = 3, H = (K - 1) * DIL;
    if (st->ne[0] != H || st->ne[1] != C || st->ne[2] != 1 || st->ne[3] != 1 || !ggml_is_contiguous(st)) {
        return ggml_cuda_conv_reject("ple_conv", "check 5", i);
    }
    if (tr->ne[0] != T || tr->ne[1] != C || tr->ne[2] != 1 || tr->view_offs != 0) {
        return ggml_cuda_conv_reject("ple_conv", "check 6", i);
    }
    if (cc->ne[0] != T + H || cc->ne[1] != C || cc->ne[2] != 1 || cc->ne[3] != 1 || !ggml_is_contiguous(cc) || T < 256 || C % 256 != 0) {
        return ggml_cuda_conv_reject("ple_conv", "check 7", i);
    }

    // Every later reader of the concat must be a tail reader or one of the four taps. Between the first tap and the
    // SiLU every node must belong to the chain, because the direct kernel replaces all of them.
    int64_t tail_from = T + H;
    int first_tap = -1, silu = -1;
    int n_taps = 0;
    const ggml_tensor * tap_out[K] = { nullptr, nullptr, nullptr, nullptr }; // latest value of tap k: CONT, then its MUL
    const ggml_tensor * W = nullptr;
    const ggml_tensor * chain = nullptr;                                      // running ADD chain (starts at tap 0's MUL)
    for (int n = i + 1; n < cgraph->n_nodes; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        const bool reads_cc = ggml_cuda_reads(t, cc);
        if (silu >= 0) {
            if (reads_cc && !ggml_cuda_conv_tail_reader(t, cc, H, tail_from)) {
                return ggml_cuda_conv_reject("ple_conv", "check 8", n);
            }
            continue;
        }
        if (reads_cc) {
            const int k = ple_tap_index(t, cc, C, T, DIL, K);
            if (k >= 0) {
                if (tap_out[k] != nullptr) {
                    return ggml_cuda_conv_reject("ple_conv", "check 9", n);
                }
                if (first_tap < 0) {
                    first_tap = n;
                }
                tap_out[k] = t;
                ++n_taps;
                continue;
            }
            if (t->op == GGML_OP_VIEW || t->op == GGML_OP_TRANSPOSE) {
                // a tap view or its transpose (no-ops), or a tail view
                if (t->op == GGML_OP_VIEW && ggml_cuda_conv_tail_view_col(t, cc, H) >= 0) {
                    tail_from = std::min(tail_from, ggml_cuda_conv_tail_view_col(t, cc, H));
                    continue;
                }
                if (t->op == GGML_OP_VIEW && t->ne[0] == T && t->ne[1] == C && t->nb[1] == cc->nb[1] && t->view_offs % (DIL * sizeof(float)) == 0 && (int64_t) (t->view_offs / (DIL * sizeof(float))) < K) {
                    continue;
                }
                if (t->op == GGML_OP_TRANSPOSE && t->src[0] && t->src[0]->op == GGML_OP_VIEW && t->src[0]->view_src == cc && t->src[0]->ne[0] == T) {
                    continue;
                }
                return ggml_cuda_conv_reject("ple_conv", "check 10", n);
            }
            if (first_tap >= 0) {
                return false; // a tail CONT/CPY inside the chain window would be skipped with it
            }
            if (!ggml_cuda_conv_tail_reader(t, cc, H, tail_from)) {
                return ggml_cuda_conv_reject("ple_conv", "check 11", n);
            }
            continue;
        }
        if (first_tap < 0) {
            continue; // unrelated work before the chain is computed normally
        }
        // inside the chain window: only the chain's own nodes
        if (t->op == GGML_OP_MUL) {
            const ggml_tensor * a = t->src[0];
            const ggml_tensor * b = t->src[1];
            int k = -1;
            for (int q = 0; q < K; ++q) {
                if (tap_out[q] && tap_out[q]->op == GGML_OP_CONT && (tap_out[q] == a || tap_out[q] == b)) {
                    k = q;
                }
            }
            if (k < 0 || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->ne[0] != C || t->ne[1] != T || t->ne[2] != 1) {
                return ggml_cuda_conv_reject("ple_conv", "check 12", n);
            }
            const ggml_tensor * Wk = ple_weight_root(tap_out[k] == a ? b : a, C, k);
            if (!Wk || (W && W != Wk)) {
                return ggml_cuda_conv_reject("ple_conv", "check 13", n);
            }
            W = Wk;
            tap_out[k] = t;
            if (k == 0) {
                chain = t;
            }
            continue;
        }
        if (t->op == GGML_OP_ADD) {
            if (!chain || t->src[0] != chain || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
                return ggml_cuda_conv_reject("ple_conv", "check 14", n);
            }
            int k = -1;
            for (int q = 1; q < K; ++q) {
                if (tap_out[q] && tap_out[q]->op == GGML_OP_MUL && tap_out[q] == t->src[1]) {
                    k = q;
                }
            }
            if (k < 0) {
                return ggml_cuda_conv_reject("ple_conv", "check 15", n);
            }
            tap_out[k] = t; // consumed
            chain = t;
            continue;
        }
        if (t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_SILU && chain && t->src[0] == chain) {
            if (n_taps != K || !W || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->ne[0] != C || t->ne[1] != T) {
                return ggml_cuda_conv_reject("ple_conv", "check 16", n);
            }
            silu = n;
            continue;
        }
        if (t->op == GGML_OP_CONT || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_CPY) {
            // weight column plumbing: must not read the concat (checked above) and must lead to a tap MUL
            continue;
        }
        return ggml_cuda_conv_reject("ple_conv", "check 17", n);
    }
    if (silu < 0 || first_tap < 0 || n_taps != K || !W) {
        return ggml_cuda_conv_reject("ple_conv", "check 18", i);
    }
    // chain intermediates feed only the chain: the direct kernel never produces them
    for (int n = first_tap; n < silu; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        if (t->op == GGML_OP_CONT || t->op == GGML_OP_MUL || t->op == GGML_OP_ADD || t->op == GGML_OP_CPY) {
            if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_cuda_conv_readers(cgraph, n + 1, t, nullptr) != 1) {
                return ggml_cuda_conv_reject("ple_conv", "chain intermediate has another reader", n);
            }
        }
    }
    m.concat_idx = i; m.first_tap_idx = first_tap; m.silu_idx = silu;
    m.x = x; m.state = st; m.concat = cc; m.w = W; m.out = cgraph->nodes[silu];
    m.C = C; m.T = T; m.H = H; m.K = K; m.dil = DIL; m.tail_from = std::max<int64_t>(0, tail_from);
    return true;
}

bool ggml_cuda_ple_conv_match_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_ple_conv_match & m) {
    return ple_conv_check(cgraph, i, m);
}

bool ggml_cuda_ple_conv_match_at_tap(const ggml_cgraph * cgraph, int i, ggml_cuda_ple_conv_match & m) {
    const ggml_tensor * t = cgraph->nodes[i];
    if (t->op != GGML_OP_CONT || !t->src[0] || t->src[0]->op != GGML_OP_TRANSPOSE || !t->src[0]->src[0] ||
        t->src[0]->src[0]->op != GGML_OP_VIEW || !t->src[0]->src[0]->view_src) {
        return false;
    }
    const ggml_tensor * cc = t->src[0]->src[0]->view_src;
    if (cc->op != GGML_OP_CONCAT) {
        return false;
    }
    for (int c = i - 1; c >= 0; --c) {
        if (cgraph->nodes[c] == cc) {
            return ple_conv_check(cgraph, c, m) && m.first_tap_idx == i;
        }
    }
    return false;
}

void ggml_cuda_ple_conv_write_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_ple_conv_match & m) {
    const int ncols = (int) (m.T + m.H - m.tail_from);
    if (ncols <= 0) {
        return;
    }
    const int64_t n = m.C * ncols;
    ple_concat_tail<<<(unsigned) ((n + 255) / 256), 256, 0, ctx.stream()>>>((const float *) m.state->data, (const float *) m.x->data,
        (float *) m.concat->data, (int) m.C, (int) m.T, (int) m.H, (int) m.tail_from, (int) (m.concat->nb[1] / sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_ple_conv_direct(ggml_backend_cuda_context & ctx, const ggml_cuda_ple_conv_match & m) {
    constexpr int TT = 128;
    GGML_ASSERT(m.K == 4 && m.dil == 3);
    dim3 grid((unsigned) (m.C / 256), (unsigned) ((m.T + TT - 1) / TT));
    if (m.w->type == GGML_TYPE_F16) {
        ple_conv_kernel<4, 3, TT, half><<<grid, 256, 0, ctx.stream()>>>((const float *) m.state->data, (const float *) m.x->data,
            (const half *) m.w->data, (float *) m.out->data, (int) m.C, (int) m.T);
    } else {
        ple_conv_kernel<4, 3, TT, float><<<grid, 256, 0, ctx.stream()>>>((const float *) m.state->data, (const float *) m.x->data,
            (const float *) m.w->data, (float *) m.out->data, (int) m.C, (int) m.T);
    }
    CUDA_CHECK(cudaGetLastError());
}
