// copied from ggml/src/ggml-cpu/ops.cpp line 8866 of the fork (1fe05fa45, upstream 9048bad8d): from
// ggml_compute_forward_flash_attn_ext_f16_one_chunk to ggml_compute_forward_flash_attn_ext. The fork reads the selected
// keys (src[6], ggml_flash_attn_ext_add_top_k). Adapted: ggml_compute_forward_flash_attn_ext is the static fork_flash_attn_ext.

static void ggml_compute_forward_flash_attn_ext_f16_one_chunk(
        const ggml_compute_params * params,
        ggml_tensor * dst,
        int ir0, int ir1,
        int64_t ic_start, int64_t ic_end,
        float * partials, int64_t partial_stride) {

    const bool write_partials = (partials != nullptr);
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * ids   = dst->src[6];
    const ggml_tensor * kv_rows = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t rk2 = neq2/nek2;
    const int64_t rk3 = neq3/nek3;

    const int64_t rv2 = neq2/nev2;
    const int64_t rv3 = neq3/nev3;

    // selected keys (src[6]): the first n_kv_raw keys plus key n_kv_raw + i for each valid index i in the query's row.
    // they are visited in ascending order, so with a mask that encodes the same selection the result is unchanged
    const int64_t n_kv_raw = ids ? ggml_get_op_params_i32(dst, 4) : nek1;
    int32_t * keys = ids ? (int32_t *) ((float *) params->wdata + params->nth*(DK + 2*DV + CACHE_LINE_SIZE_F32) +
            neq2*params->nth*(2 + DV)) + params->ith*ids->ne[0] : nullptr;

    // parallelize by q rows using ggml_vec_dot_f32

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) dst->op_params + 2, sizeof(float));

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ggml_type         const k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t const q_to_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    const kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;
    ggml_to_float_t   const v_to_float     = ggml_get_type_traits(v->type)->to_float;

    GGML_ASSERT((                            q_to_vec_dot) && "fattn: unsupported K-type");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v_to_float  ) && "fattn: unsupported V-type");

    int ith = params->ith;

    for (int ir = ir0; ir < ir1; ++ir) {
        // q indices
        const int iq3 = ir/(neq2*neq1);
        const int iq2 = (ir - iq3*neq2*neq1)/neq1;
        const int iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        const uint32_t h = iq2; // head index
        const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        float S = 0.0f;      // sum
        float M = -INFINITY; // maximum KQ value

        float       * VKQ32 = (float       *) params->wdata + ith*(1*DK + 2*DV + CACHE_LINE_SIZE_F32); // FP32 VKQ accumulator
        float       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        memset(VKQ32, 0, DV*sizeof(float));

        const ggml_fp16_t * mp = mask ? (ggml_fp16_t *)((char *) mask->data + iq1*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]) : NULL;

        // k indices
        const int ik3 = iq3 / rk3;
        const int ik2 = iq2 / rk2;

        // v indices
        const int iv3 = iq3 / rv3;
        const int iv2 = iq2 / rv2;

        const float * pq = (const float *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_to_vec_dot(pq, Q_q, DK);

        int64_t n_sel = 0;
        if (ids) {
            const int32_t * row = (const int32_t *) ((const char *) ids->data + iq1*ids->nb[1] + (iq3%ids->ne[3])*ids->nb[3]);
            for (int64_t j = 0; j < ids->ne[0]; ++j) {
                const int64_t ic = n_kv_raw + row[j];
                if (row[j] >= 0 && ic >= ic_start && ic < ic_end) {
                    keys[n_sel++] = (int32_t) ic;
                }
            }
            std::sort(keys, keys + n_sel);
            n_sel = std::unique(keys, keys + n_sel) - keys;
        }
        const int64_t n_dense = std::max<int64_t>(0, std::min(ic_end, n_kv_raw) - ic_start);
        const int32_t * rows = kv_rows ? (const int32_t *) ((const char *) kv_rows->data + iq3*kv_rows->nb[1]) : nullptr;

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf

        for (int64_t j = 0; j < n_dense + n_sel; ++j) {
            const int64_t ic = j < n_dense ? ic_start + j : keys[j - n_dense];
            const float mv = mp ? slope*GGML_CPU_FP16_TO_FP32(mp[ic]) : 0.0f;
            if (mv == -INFINITY) {
                continue;
            }

            const int64_t ik1 = rows ? rows[ic] : ic;
            if (ik1 < 0) {
                continue;
            }

            float s; // KQ value

            const char * k_data = (const char *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            if (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            const float Mold = M;

            float ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            float vs = 1.0f; // post-softmax KQ value, expf(s - M)

            const char * v_data = ((const char *) v->data + (ik1*nbv1 + iv2*nbv2 + iv3*nbv3));

            if (s > M) {
                // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                M = s;
                ms = expf(Mold - M);

                // V = V*expf(Mold - M)
                ggml_vec_scale_f32(DV, VKQ32, ms);
            } else {
                // no new maximum, ms == 1.0f, vs != 1.0f
                vs = expf(s - M);
            }

            // V += v*expf(s - M)
            if (v->type == GGML_TYPE_F16) {
                ggml_vec_mad_f16_f32(DV, VKQ32, (const ggml_fp16_t *) v_data, vs);
            } else if (v_to_float) {
                v_to_float(v_data, V32, DV);
                ggml_vec_mad_f32(DV, VKQ32, V32, vs);
            } else {
                // V is F32
                ggml_vec_mad_f32(DV, VKQ32, (const float *) v_data, vs);
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        // sinks - apply only on the first kv-chunk
        if (sinks && ic_start == 0) {
            const float s = ((float *)((char *) sinks->data))[h];

            float ms = 1.0f;
            float vs = 1.0f;

            if (s > M) {
                ms = expf(M - s);
                M = s;
                ggml_vec_scale_f32(DV, VKQ32, ms);
            } else {
                vs = expf(s - M);
            }

            S = S*ms + vs;
        }

        if (write_partials) {
            // Write M, S, VKQ to partials for later reduction
            // partials layout: [M, S, VKQ[DV]] per query head
            float * partial = partials + ir * partial_stride;
            partial[0] = M;
            partial[1] = S;
            memcpy(partial + 2, VKQ32, DV * sizeof(float));
        } else {
            // V /= S
            const float S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            ggml_vec_scale_f32(DV, VKQ32, S_inv);

            // dst indices
            const int i1 = iq1;
            const int i2 = iq2;
            const int i3 = iq3;

            // permute(0, 2, 1, 3)
            memcpy((char *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
        }
    }
}

static void ggml_compute_forward_flash_attn_ext_tiled(
        const ggml_compute_params * params,
        ggml_tensor * dst,
        int ir0, int ir1) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    const ggml_type kv_type = k->type;


    // broadcast factors
    const int64_t rk2 = neq2/nek2;
    const int64_t rk3 = neq3/nek3;

    const int64_t rv2 = neq2/nev2;
    const int64_t rv3 = neq3/nev3;

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) dst->op_params + 2, sizeof(float));

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    int ith = params->ith;

    static constexpr int Q_TILE_SZ  = ggml_fa_tile_config::Q;
    static constexpr int KV_TILE_SZ = ggml_fa_tile_config::KV;

    // with kv_rows the loop runs over the mask columns and reads K/V rows through kv_rows
    const ggml_tensor * kv_rows = dst->src[5];
    const int64_t n_kv_cols = kv_rows ? kv_rows->ne[0] : nek1;

    int ir = ir0;
    while (ir < ir1) {
        // q indices for the start of this tile
        const int iq3 = ir/(neq2*neq1);
        const int iq2 = (ir - iq3*neq2*neq1)/neq1;
        const int iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        // padding rows are masked, read row 0 for them
        const int32_t * rows = kv_rows ? (const int32_t *) ((const char *) kv_rows->data + iq3*kv_rows->nb[1]) : nullptr;
        auto kv_row = [rows](int64_t i) -> int64_t { return rows ? std::max<int32_t>(rows[i], 0) : i; };

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        const int tile_rows = MIN(Q_TILE_SZ, MIN((int)(ir1 - ir), (int)(neq1 - iq1)));
        GGML_ASSERT(tile_rows > 0);

        const uint32_t h = iq2; // head index
        const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        float S[Q_TILE_SZ];
        float M[Q_TILE_SZ];

        for (int i = 0 ; i < Q_TILE_SZ; ++i) {
            S[i] = 0.;
            M[i] = -INFINITY;
        }

        // Per-thread scratch layout:
        // Q_q:    Q_TILE_SZ * DK (converted Q tile — F32 for GEMM, KV type for scalar)
        // KQ:     Q_TILE_SZ * KV_TILE_SZ (attention scores in float)
        // mask:   Q_TILE_SZ * KV_TILE_SZ (mask in float)
        // VKQ32:  Q_TILE_SZ * DV (FP32 output accumulator)
        // V32:    KV_TILE_SZ * DV (F32 buffer for V tile)
        // K_f32:  KV_TILE_SZ * DK (F32 buffer for K tile — GEMM path)
        float * base  = (float *) params->wdata + ith*(Q_TILE_SZ*DK + 2*Q_TILE_SZ*KV_TILE_SZ + Q_TILE_SZ*DV + KV_TILE_SZ*DV + KV_TILE_SZ*DK + CACHE_LINE_SIZE_F32);

        void  * Q_q    = base;
        float * KQ     = (float *)((char *)base + Q_TILE_SZ * DK * sizeof(float));
        float * mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        float * VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        float * V32    = VKQ32 + Q_TILE_SZ * DV;
        float * K_f32  = V32 + KV_TILE_SZ * DV;

        memset(VKQ32, 0, Q_TILE_SZ * DV * sizeof(float));
        memset(mask32, 0, Q_TILE_SZ * KV_TILE_SZ * sizeof(float));

        // k indices
        const int ik3 = iq3 / rk3;
        const int ik2 = iq2 / rk2;

        // v indices
        const int iv3 = iq3 / rv3;
        const int iv2 = iq2 / rv2;

        {
            float * Q_f32 = (float *)Q_q;
            for (int tq = 0; tq < tile_rows; tq++) {
                const float * pq = (const float *) ((char *) q->data + ((iq1 + tq)*nbq1 + iq2*nbq2 + iq3*nbq3));
                memcpy(Q_f32 + tq * DK, pq, DK * sizeof(float));
            }
            for (int tq = tile_rows; tq < Q_TILE_SZ; tq++) {
                memset(Q_f32 + tq * DK, 0, DK * sizeof(float));
            }
        }

        memset(K_f32, 0, DK * KV_TILE_SZ * sizeof(float));
        memset(V32,   0, KV_TILE_SZ * DV * sizeof(float));

        for (int64_t ic = 0; ic < n_kv_cols; ic += KV_TILE_SZ) {
            const int kv_tile = (int)std::min((int64_t)KV_TILE_SZ, n_kv_cols - ic);

            // skip the tile entirely if all the masks are -inf
            if (mask) {
                bool can_skip = true;
                for (int tq = 0; tq < tile_rows; tq++) {
                    const ggml_fp16_t * mp_row = (const ggml_fp16_t *)((const char *) mask->data + (iq1 + tq)*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]);
                    for (int tk = 0; tk < kv_tile; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = slope * GGML_CPU_FP16_TO_FP32(mp_row[ic + tk]);
                        if (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = false;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    for (int tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                if (can_skip) {
                    continue;
                }
            }

            // Pack K tile transposed: K_f32[dk][kv] so KV_TILE is contiguous (SIMD dim)
            // Zero-pad the last tile so the GEMM always operates on KV_TILE_SZ columns
            for (int tk = 0; tk < kv_tile; tk++) {
                const char * k_data = (const char *)k->data + kv_row(ic + tk)*nbk1 + ik2*nbk2 + ik3*nbk3;
                if (kv_type == GGML_TYPE_F16) {
                    const ggml_fp16_t * k_f16 = (const ggml_fp16_t *)k_data;
                    for (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = GGML_CPU_FP16_TO_FP32(k_f16[dk]);
                    }
                } else {
                    const float * k_f32_src = (const float *)k_data;
                    for (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = k_f32_src[dk];
                    }
                }
            }
            memset(KQ, 0, Q_TILE_SZ * KV_TILE_SZ * sizeof(float));
            simd_gemm(KQ, (const float *)Q_q, K_f32, Q_TILE_SZ, DK, KV_TILE_SZ);
            ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, scale);

            if (logit_softcap != 0.0f) {
                ggml_vec_tanh_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, KQ);
                ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, logit_softcap);
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            if (kv_tile < KV_TILE_SZ) {
                for (int tq = 0; tq < Q_TILE_SZ; tq++) {
                    for (int tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            if (mask) {
                ggml_vec_add_f32(tile_rows * KV_TILE_SZ, KQ, KQ, mask32);
            }

            bool skip[Q_TILE_SZ] = {};

            for (int tq = 0; tq < Q_TILE_SZ; tq++) {
                float * kq_row = KQ + tq * KV_TILE_SZ;

                float tile_max;
                ggml_vec_max_f32(KV_TILE_SZ, &tile_max, kq_row);

                if (tile_max == -INFINITY) {
                    skip[tq] = true;
                    continue;
                }

                const float Mold = M[tq];
                const float Mnew = fmaxf(Mold, tile_max);

                if (Mnew > Mold) {
                    const float ms = expf(Mold - Mnew);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;


                S[tq] += ggml_vec_soft_max_f32(KV_TILE_SZ, kq_row, kq_row, Mnew);
            }

            // V accumulation: VKQ32 += softmax(KQ) * V
            // Pack V tile to contiguous F32, zero-padded
            for (int tk = 0; tk < kv_tile; tk++) {
                const char * v_data = (const char *)v->data + kv_row(ic + tk)*nbv1 + iv2*nbv2 + iv3*nbv3;
                if (kv_type == GGML_TYPE_F16) {
                    ggml_cpu_fp16_to_fp32((const ggml_fp16_t *)v_data, V32 + tk * DV, DV);
                } else {
                    memcpy(V32 + tk * DV, v_data, DV * sizeof(float));
                }
            }
            for (int tq = 0; tq < Q_TILE_SZ; tq++) {
                if (skip[tq]) {
                    memset(KQ + tq * KV_TILE_SZ, 0, KV_TILE_SZ * sizeof(float));
                }
            }
            simd_gemm(VKQ32, KQ, V32, Q_TILE_SZ, KV_TILE_SZ, DV);
        }

        // sinks (apply only to valid rows in the tile)
        if (sinks) {
            const float s = ((float *)((char *) sinks->data))[h];

            for (int tq = 0; tq < tile_rows; tq++) {
                float ms = 1.0f;
                float vs = 1.0f;

                if (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                } else {
                    vs = expf(s - M[tq]);
                }

                S[tq] = S[tq] * ms + vs;
            }
        }

        for (int tq = 0; tq < tile_rows; tq++) {
            // V /= S
            const float S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
            ggml_vec_scale_f32(DV, VKQ32 + tq * DV, S_inv);

            // dst indices
            const int i1 = iq1 + tq;
            const int i2 = iq2;
            const int i3 = iq3;

            // permute(0, 2, 1, 3)
            memcpy((char *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32 + tq * DV, nb1);
        }

        ir += tile_rows;
    }
}

// Reduction function: combines partial results across KV chunks
// Partials layout in wdata: [n_q_heads][n_chunks][2 + DV]
static void ggml_flash_attn_ext_reduce_partials(
        const ggml_compute_params * params,
        ggml_tensor * dst,
        const int64_t n_chunks,
        const int64_t chunk_size) {

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];

    const int64_t DK        = k->ne[0];
    const int64_t DV        = v->ne[0];
    const int64_t nek1      = k->ne[1];
    const int64_t n_q_heads = q->ne[2];

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t wdata_per_thread = DK + 2*DV + CACHE_LINE_SIZE_F32;
    float *       thread_wdata     = (float *) params->wdata + ith * wdata_per_thread;

    const int64_t partials_offset  = nth * (DK + 2*DV + CACHE_LINE_SIZE_F32);
    const int64_t partial_size     = 2 + DV;
    const float * partials_base    = (const float *) params->wdata + partials_offset;

    // Output layout
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const size_t  nb1 = dst->nb[1];

    // Each thread reduces a subset of query heads
    for (int64_t q_head = ith; q_head < n_q_heads; q_head += nth) {
        float   M_final   = -INFINITY;
        float   S_final   = 0.0f;
        float * VKQ_final = thread_wdata;
        memset(VKQ_final, 0, DV * sizeof(float));

        // Combine partials from all chunks
        for (int64_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
            const int64_t ic_start = chunk_idx * chunk_size;
            if (ic_start >= nek1) continue;

            const float * partial   = partials_base + (q_head * n_chunks + chunk_idx) * partial_size;
            const float   M_chunk   = partial[0];
            const float   S_chunk   = partial[1];
            const float * VKQ_chunk = partial + 2;

            if (S_chunk == 0.0f) continue;

            const float M_new     = fmaxf(M_final, M_chunk);
            const float scale_old = expf(M_final - M_new);
            const float scale_new = expf(M_chunk - M_new);

            for (int64_t d = 0; d < DV; ++d) {
                VKQ_final[d] = VKQ_final[d] * scale_old + VKQ_chunk[d] * scale_new;
            }
            S_final = S_final * scale_old + S_chunk * scale_new;
            M_final = M_new;
        }

        // Normalize and write to output
        if (S_final != 0.0f) {
            const float S_inv = 1.0f / S_final;
            ggml_vec_scale_f32(DV, VKQ_final, S_inv);
        }
        // iq1=0, iq3=0 for decode
        memcpy((char *) dst->data + (0*ne2*ne1 + q_head + 0*ne1)*nb1, VKQ_final, nb1);
    }
}

static void ggml_compute_forward_flash_attn_ext_f16(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;


    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int ith = params->ith;
    const int nth = params->nth;

    // When use_ref is set, force the vec-only reference implementation (no tiling, no KV-chunking)
    const bool use_ref = params->use_ref;

    // with kv_rows the loops run over the mask columns, the split KV path does not support it
    const ggml_tensor * kv_rows = dst->src[5];
    const int64_t n_kv_cols = kv_rows ? kv_rows->ne[0] : nek1;

    const bool kv_is_f32_or_f16 = (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    const bool use_split_kv_path = !use_ref && !kv_rows && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16 && (k->type == v->type) && q->type == GGML_TYPE_F32 && nek1 >= 512;

    if (use_split_kv_path) {
        const int64_t chunk_size = (nek1 + nth - 1) / nth;

        // Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
        const int64_t partial_size  = 2 + DV;
        float *       partials_base = (float *) params->wdata + nth * (DK + 2*DV + CACHE_LINE_SIZE_F32);

        const int64_t ic_start = ith * chunk_size;
        const int64_t ic_end   = std::min(ic_start + chunk_size, nek1);

        const int64_t partial_stride = nth * partial_size;
        float *       chunk_partials = partials_base + ith * partial_size;

        if (ic_start < nek1) {
            for (int64_t q_head = 0; q_head < neq2; q_head++) {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(
                    params, dst, q_head, q_head + 1, ic_start, ic_end,
                    chunk_partials, partial_stride);
            }
        } else {
            for (int64_t q_head = 0; q_head < neq2; q_head++) {
                float * q_partials = chunk_partials + q_head * partial_stride;
                q_partials[0] = -INFINITY;  // M
                q_partials[1] = 0.0f;       // S
            }
        }

        ggml_barrier(params->threadpool);
        ggml_flash_attn_ext_reduce_partials(params, dst, nth, chunk_size);
    } else {

        // total rows in q
        const int64_t nr = neq1*neq2*neq3;

        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        int nth_scaled = nth * 4;
        int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        if (nth == 1 || nchunk < nth || disable_chunking) {
            nchunk = nth;
        }

        if (ith == 0) {
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        const int64_t dr = (nr + nchunk - 1) / nchunk;

        static constexpr int64_t Q_TILE_SZ  = ggml_fa_tile_config::Q;
        // the tiled kernel attends every key the mask allows, so selected keys without a mask need the per-row path
        bool use_tiled = !use_ref &&
                               (q->type == GGML_TYPE_F32 &&
                                kv_is_f32_or_f16 &&
                                k->type == v->type &&
                                neq1 >= Q_TILE_SZ &&
                                (!dst->src[6] || dst->src[3]));
#if defined(GGML_SIMD) && !defined(__x86_64__) && !defined(_M_X64)
#if defined(__ARM_FEATURE_SVE)
        const int64_t f32_epr = svcntw();
#else
        const int64_t f32_epr = GGML_F32_EPR;
#endif
        use_tiled &= (DV % f32_epr == 0);
#endif
        int current_chunk = ith;

        while (current_chunk < nchunk) {
            const int64_t ir0 = dr * current_chunk;
            const int64_t ir1 = MIN(ir0 + dr, nr);

            if (use_tiled) {
                ggml_compute_forward_flash_attn_ext_tiled(params, dst, ir0, ir1);
            } else {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(params, dst, ir0, ir1, 0, n_kv_cols, nullptr, 0);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }
}

static void fork_flash_attn_ext(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    switch (ggml_prec_get_acc(dst)) {
        case GGML_PREC_UNDEFINED:
        case GGML_PREC_F32:
        case GGML_PREC_BF16:
            {
                // uses F32 accumulators
                ggml_compute_forward_flash_attn_ext_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// copied from ggml/src/ggml-cpu/ops.cpp line 11518 of the fork: ggml_compute_forward_dsv4_hc_pre and its f32 function,
// which also take f16 (ggml_dsv4_hc_pre_gated_ext). Adapted: renamed fork_dsv4_hc_pre(_f32), static.

static void fork_dsv4_hc_pre_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    const bool gated = ggml_get_op_params_i32(dst, 1) != 0;
    // the gated form also takes an f16 x and an f16 gate
    GGML_ASSERT(x->type == GGML_TYPE_F32 || (gated && x->type == GGML_TYPE_F16));
    GGML_ASSERT(weights->type == GGML_TYPE_F32 || (gated && weights->type == GGML_TYPE_F16));
    // the gated form may write an f16 mixed stream (ggml_dsv4_hc_pre_gated_ext)
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || (gated && dst->type == GGML_TYPE_F16));
    const bool d_f16 = dst->type == GGML_TYPE_F16;
    const bool x_f16 = x->type == GGML_TYPE_F16;
    const bool g_f16 = weights->type == GGML_TYPE_F16;

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    if (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } else {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
    }

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nr  = n_embd * n_tokens;
    const int64_t dr  = (nr + nth - 1) / nth;
    const int64_t ir0 = dr * ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i0 = ir % n_embd;
        const int64_t it = ir / n_embd;

        float sum = 0.0f;
        for (int64_t ih = 0; ih < hc; ++ih) {
            const char * xp = (const char *) x->data + i0*nbx0 + ih*nbx1 + it*nbx2;
            const float xv = x_f16 ? GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *) xp) : *(const float *) xp;
            float wv;
            if (gated) {
                const char * gp = (const char *) weights->data + i0*nbw0 + ih*nbw1 + it*nbw2;
                const float gv = g_f16 ? GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *) gp) : *(const float *) gp;
                wv = 1.0f / (1.0f + expf(-gv));
            } else {
                wv = *(const float *) ((const char *) weights->data + ih*nbw0 + it*nbw1);
            }
            sum += xv * wv;
        }

        char * dp = (char *) dst->data + i0*nbd0 + it*nbd1;
        if (d_f16) {
            *(ggml_fp16_t *) dp = GGML_CPU_FP32_TO_FP16(scale * sum);
        } else {
            *(float *) dp = scale * sum;
        }
    }
}

static void fork_dsv4_hc_pre(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:   // gated form only (asserted inside)
            {
                fork_dsv4_hc_pre_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}
