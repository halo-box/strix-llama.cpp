#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"
#include "q7-panel16-cache.cuh"
#include "q7-q8-view-cache.cuh"

#include <atomic>
#include <cinttypes>
#include <cstring>

namespace {

std::atomic<uint64_t> g_q7_panel16_mmq_handoffs{0};
std::atomic<uint64_t> g_q7_panel16_kernel_launches{0};
std::atomic<uint64_t> g_q7_q8_view_mmq_handoffs{0};
std::atomic<uint64_t> g_q7_q8_view_kernel_dispatches{0};
std::atomic<uint64_t> g_q7_q8_view_output_dispatches{0};
std::atomic<uint64_t> g_q7_q8_view_expert_handoffs{0};
std::atomic<uint64_t> g_q7_q8_view_expert_dispatches{0};

static size_t q7_q8_view_tensor_nbytes(
        const ggml_tensor * tensor) {
    if (tensor == nullptr ||
        tensor->ne[1] <= 0 ||
        tensor->ne[2] <= 0 ||
        tensor->ne[1] > INT64_MAX / tensor->ne[2]) {
        return 0;
    }
    return ggml_cuda_q7_q8_view_nbytes(
        tensor->ne[1] * tensor->ne[2],
        tensor->ne[0]);
}

} // namespace

void ggml_cuda_q7_panel16_note_mmq_handoff(
        const ggml_tensor * tensor) {
    const uint64_t count =
        g_q7_panel16_mmq_handoffs.fetch_add(
            1,
            std::memory_order_relaxed) +
        1;
    if (count == 1) {
        GGML_LOG_INFO(
            "gfx1151 Q7 panel16 MMQ cache handoff: "
            "handoffs=%" PRIu64 " tensor=%s panel_bytes=%zu\n",
            count,
            tensor->name,
            ggml_cuda_q7_panel16_nbytes(
                tensor->ne[1],
                tensor->ne[0]));
    }
}

void ggml_cuda_q7_panel16_note_kernel_launch(
        const int mmq_x,
        const int64_t nrows_x,
        const int64_t ncols_x,
        const int64_t ncols_dst) {
    const uint64_t count =
        g_q7_panel16_kernel_launches.fetch_add(
            1,
            std::memory_order_relaxed) +
        1;
    if (count == 1) {
        GGML_LOG_INFO(
            "gfx1151 Q7 panel16 MMQ kernel launched: "
            "launches=%" PRIu64 " mmq_x=%d "
            "m=%" PRId64 " n=%" PRId64 " k=%" PRId64 "\n",
            count,
            mmq_x,
            nrows_x,
            ncols_dst,
            ncols_x);
    }
}

uint64_t ggml_cuda_q7_panel16_mmq_handoff_count() {
    return g_q7_panel16_mmq_handoffs.load(std::memory_order_relaxed);
}

uint64_t ggml_cuda_q7_panel16_kernel_launch_count() {
    return g_q7_panel16_kernel_launches.load(std::memory_order_relaxed);
}

void ggml_cuda_q7_q8_view_note_mmq_handoff(
        const ggml_tensor * tensor) {
    const uint64_t count =
        g_q7_q8_view_mmq_handoffs.fetch_add(
            1,
            std::memory_order_relaxed) +
        1;
    if (count == 1) {
        GGML_LOG_INFO(
            "gfx1151 Q7 standard-Q8 compute-view MMQ handoff: "
            "handoffs=%" PRIu64 " tensor=%s view_bytes=%zu\n",
            count,
            tensor->name,
            q7_q8_view_tensor_nbytes(tensor));
    }
    if (std::strstr(tensor->name, "_exps.weight") != nullptr) {
        const uint64_t expert_count =
            g_q7_q8_view_expert_handoffs.fetch_add(
                1,
                std::memory_order_relaxed) +
            1;
        if (expert_count == 1) {
            GGML_LOG_INFO(
                "gfx1151 Q7 standard-Q8 routed-expert MMQ handoff: "
                "handoffs=%" PRIu64 " tensor=%s experts=%" PRId64
                " view_bytes=%zu\n",
                expert_count,
                tensor->name,
                tensor->ne[2],
                q7_q8_view_tensor_nbytes(tensor));
        }
    }
}

void ggml_cuda_q7_q8_view_note_kernel_dispatch(
        const ggml_tensor * tensor,
        const int64_t nrows_x,
        const int64_t ncols_x,
        const int64_t ncols_dst) {
    const uint64_t count =
        g_q7_q8_view_kernel_dispatches.fetch_add(
            1,
            std::memory_order_relaxed) +
        1;
    if (count == 1) {
        GGML_LOG_INFO(
            "gfx1151 Q7 standard-Q8 staged MMQ dispatched: "
            "dispatches=%" PRIu64 " m=%" PRId64
            " n=%" PRId64 " k=%" PRId64 "\n",
            count,
            nrows_x,
            ncols_dst,
            ncols_x);
    }
    if (std::strcmp(tensor->name, "output.weight") == 0) {
        const uint64_t output_count =
            g_q7_q8_view_output_dispatches.fetch_add(
                1,
                std::memory_order_relaxed) +
            1;
        if (output_count == 1) {
            GGML_LOG_INFO(
                "gfx1151 Q7 standard-Q8 output.weight MMQ dispatched: "
                "dispatches=%" PRIu64 " m=%" PRId64
                " n=%" PRId64 " k=%" PRId64
                " view_bytes=%zu\n",
                output_count,
                nrows_x,
                ncols_dst,
                ncols_x,
                q7_q8_view_tensor_nbytes(tensor));
        }
    }
    if (std::strstr(tensor->name, "_exps.weight") != nullptr) {
        const uint64_t expert_count =
            g_q7_q8_view_expert_dispatches.fetch_add(
                1,
                std::memory_order_relaxed) +
            1;
        if (expert_count == 1) {
            GGML_LOG_INFO(
                "gfx1151 Q7 standard-Q8 routed-expert staged MMQ "
                "dispatched: dispatches=%" PRIu64
                " tensor=%s experts=%" PRId64
                " m=%" PRId64 " n=%" PRId64 " k=%" PRId64 "\n",
                expert_count,
                tensor->name,
                tensor->ne[2],
                nrows_x,
                ncols_dst,
                ncols_x);
        }
    }
}

uint64_t ggml_cuda_q7_q8_view_mmq_handoff_count() {
    return g_q7_q8_view_mmq_handoffs.load(
        std::memory_order_relaxed);
}

uint64_t ggml_cuda_q7_q8_view_kernel_dispatch_count() {
    return g_q7_q8_view_kernel_dispatches.load(
        std::memory_order_relaxed);
}

uint64_t ggml_cuda_q7_q8_view_output_dispatch_count() {
    return g_q7_q8_view_output_dispatches.load(
        std::memory_order_relaxed);
}

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0_ROCMFP4:
            mul_mat_q_case<GGML_TYPE_Q4_0_ROCMFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
            mul_mat_q_case<GGML_TYPE_Q4_0_ROCMFP4_FAST>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_0_ROCMFPX:
            mul_mat_q_case<GGML_TYPE_Q3_0_ROCMFPX>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0_ROCMFPX:
            mul_mat_q_case<GGML_TYPE_Q2_0_ROCMFPX>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_0_ROCMFPX:
            mul_mat_q_case<GGML_TYPE_Q6_0_ROCMFPX>(ctx, args, stream);
            break;
        case GGML_TYPE_Q7_0_ROCMFPX:
            mul_mat_q_case<GGML_TYPE_Q7_0_ROCMFPX>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0_ROCMFPX:
            mul_mat_q_case<GGML_TYPE_Q8_0_ROCMFPX>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool use_stream_k = (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc);

    // TODO: tighter pool buffer size vs q8 path
    const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);

    if (!ids) {
        const block_rocmfp7_panel16 * src0_q7_panel16 =
            ggml_cuda_q7_panel16_cache_get(src0);
        const block_q8_0 * src0_q7_q8_view =
            ggml_cuda_q7_q8_view_get(src0);
        GGML_ASSERT(
            src0_q7_panel16 == nullptr ||
            src0_q7_q8_view == nullptr);
        if (src0_q7_panel16 != nullptr) {
            ggml_cuda_q7_panel16_note_mmq_handoff(src0);
        }
        if (src0_q7_q8_view != nullptr) {
            ggml_cuda_q7_q8_view_note_mmq_handoff(src0);
        }
        const ggml_type mmq_type_x =
            src0_q7_q8_view != nullptr ?
                GGML_TYPE_Q8_0 :
                src0->type;
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1 +
            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), nullptr, src0->type, false, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), mmq_type_x, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_K * sizeof(int)) :  // block_fp4_mmq holds 256 values
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            use_stream_k, ne1,
            src0_q7_panel16, src0_q7_panel16 != nullptr};
        if (src0_q7_q8_view != nullptr) {
            mmq_args q8_args = args;
            q8_args.x =
                reinterpret_cast<const char *>(src0_q7_q8_view);
            q8_args.type_x = GGML_TYPE_Q8_0;
            q8_args.stride_row_x *= NG_ROCMFP7;
            q8_args.stride_channel_x *= NG_ROCMFP7;
            q8_args.stride_sample_x *= NG_ROCMFP7;
            q8_args.x_q7_panel16 = nullptr;
            q8_args.has_q7_panel16_cache = false;
            q8_args.force_staged_q8 = true;
            ggml_cuda_q7_q8_view_note_kernel_dispatch(
                src0,
                q8_args.nrows_x,
                q8_args.ncols_x,
                q8_args.ncols_dst);
            ggml_cuda_mul_mat_q_switch_type(
                ctx,
                q8_args,
                stream);
            return;
        }
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    const block_q8_0 * src0_q7_q8_view =
        ggml_cuda_q7_q8_view_get(src0);
    if (src0_q7_q8_view != nullptr) {
        ggml_cuda_q7_q8_view_note_mmq_handoff(src0);
    }
    const ggml_type mmq_type_x =
        src0_q7_q8_view != nullptr ?
            GGML_TYPE_Q8_0 :
            src0->type;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, false, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * sizeof(block_q8_1)/QK8_1 +
        get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), nullptr, src0->type, false, ne10, s11, s12, s13,
                                    ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), mmq_type_x, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_K == 8 * QK_MXFP4, "QK_K needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_K * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        use_stream_k, ne12,
        nullptr, false};

    if (src0_q7_q8_view != nullptr) {
        mmq_args q8_args = args;
        q8_args.x =
            reinterpret_cast<const char *>(src0_q7_q8_view);
        q8_args.type_x = GGML_TYPE_Q8_0;
        q8_args.stride_row_x *= NG_ROCMFP7;
        q8_args.stride_channel_x *= NG_ROCMFP7;
        q8_args.stride_sample_x *= NG_ROCMFP7;
        q8_args.force_staged_q8 = true;
        ggml_cuda_q7_q8_view_note_kernel_dispatch(
            src0,
            q8_args.nrows_x,
            q8_args.ncols_x,
            q8_args.ncols_dst);
        ggml_cuda_mul_mat_q_switch_type(
            ctx,
            q8_args,
            stream);
        return;
    }

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;
    const int64_t stride01 = ne00 / ggml_blck_size(src0->type);

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    const bool use_stream_k = ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc))
                            && src1_ncols == ne11;
    const mmq_args args = {
        src0_dd_i, src0->type, (const int *) src1_ddq_i, nullptr, nullptr, dst_dd_i,
        ne00, row_diff, src1_ncols, stride01, ne11, nrows_dst,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k, src1_ncols,
        nullptr, false};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_padded_row_size);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q4_0_ROCMFP4:
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
        case GGML_TYPE_Q3_0_ROCMFPX:
        case GGML_TYPE_Q2_0_ROCMFPX:
        case GGML_TYPE_Q6_0_ROCMFPX:
        case GGML_TYPE_Q7_0_ROCMFPX:
        case GGML_TYPE_Q8_0_ROCMFPX:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10) lacks native dp4a, so MMQ is emulated and loses to
    // dequant + hipBLAS for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA) {
        return n_experts > 0;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
