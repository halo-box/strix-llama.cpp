#pragma once

// Where the fused gated SPARSE_ATTN writes its outputs (ggml-cuda.cu, the SPARSE_ATTN -> CONT -> SIGMOID -> MUL fusion).
// The kernel writes the gated f32 output (o32, the MUL's memory) and / or its 16-bit copy (o16, for the next mmb reader)
// while it reads Q, K, V, the plan, the gate and its own workspace. An output that overlaps a read cannot be written in
// place: it goes to graph memory that the fused subgraph leaves unwritten (the attend's own dst, the CONT and SIGMOID outputs,
// all in the reserved compute buffer), and one device copy moves it into place; when none fits, the fusion is refused (the
// unfused nodes run). Never pool memory: PLAN-v3 D12 (no ctx.pool() in a sparse kernel; the pool is outside the reserve, the
// fit probe and LLAMA_STRICT_RESERVE). Host-only: tests/test-sparse-attn-gate-scratch.cpp checks it on synthetic layouts.

#include <cstddef>
#include <cstdint>

struct ggml_cuda_sa_span {
    const void * p;
    size_t       n;
};

enum ggml_cuda_sa_gate_where {
    GGML_CUDA_SA_GATE_NONE    = 0,   // the output is not written
    GGML_CUDA_SA_GATE_DIRECT  = 1,   // written in place
    GGML_CUDA_SA_GATE_POOL    = 2,   // pool scratch: never chosen (D12), kept so that a caller can assert it
    GGML_CUDA_SA_GATE_SCRATCH = 3,   // written to scratch32 / scratch16 (graph memory), then copied into place
};

struct ggml_cuda_sa_gate_plan {
    bool ok;                         // false: do not fuse
    int  where32, where16;           // ggml_cuda_sa_gate_where
    void * scratch32, * scratch16;   // the scratch of a copied output
};

static inline bool ggml_cuda_sa_overlap(const void * a, size_t na, const void * b, size_t nb) {
    const uintptr_t x = (uintptr_t) a, y = (uintptr_t) b;
    return na > 0 && nb > 0 && x < y + nb && y < x + na;
}

static inline bool ggml_cuda_sa_overlaps_any(const ggml_cuda_sa_span * spans, int n, const void * p, size_t np) {
    for (int i = 0; i < n; ++i) {
        if (ggml_cuda_sa_overlap(spans[i].p, spans[i].n, p, np)) {
            return true;
        }
    }
    return false;
}

// reads: every range the kernel reads; o32 / o16: the outputs (n == 0 when not written); cands: memory of the graph that
// the fusion leaves unwritten (the attend's own dst, the CONT and SIGMOID outputs)
static inline ggml_cuda_sa_gate_plan ggml_cuda_sa_gate_plan_outputs(const ggml_cuda_sa_span * reads, int n_reads,
        ggml_cuda_sa_span o32, ggml_cuda_sa_span o16, const ggml_cuda_sa_span * cands, int n_cands) {
    ggml_cuda_sa_gate_plan r = { true, GGML_CUDA_SA_GATE_NONE, GGML_CUDA_SA_GATE_NONE, nullptr, nullptr };
    const bool hazard32 = o32.n > 0 && ggml_cuda_sa_overlaps_any(reads, n_reads, o32.p, o32.n);
    const bool hazard16 = o16.n > 0 && ggml_cuda_sa_overlaps_any(reads, n_reads, o16.p, o16.n);
    r.where32 = o32.n == 0 ? GGML_CUDA_SA_GATE_NONE : hazard32 ? GGML_CUDA_SA_GATE_SCRATCH : GGML_CUDA_SA_GATE_DIRECT;
    r.where16 = o16.n == 0 ? GGML_CUDA_SA_GATE_NONE : hazard16 ? GGML_CUDA_SA_GATE_SCRATCH : GGML_CUDA_SA_GATE_DIRECT;
    // a scratch: the start of a candidate that holds the output and overlaps no read, no final output and no other scratch
    // (the candidates can overlap each other: the SIGMOID may run in place over the CONT, the MUL over either)
    const auto pick = [&](size_t n, const void * other, size_t n_other) -> void * {
        for (int i = 0; i < n_cands; ++i) {
            const ggml_cuda_sa_span & c = cands[i];
            if (c.p == nullptr || c.n < n || ggml_cuda_sa_overlaps_any(reads, n_reads, c.p, n) ||
                ggml_cuda_sa_overlap(c.p, n, o32.p, o32.n) || ggml_cuda_sa_overlap(c.p, n, o16.p, o16.n) ||
                ggml_cuda_sa_overlap(c.p, n, other, n_other)) {
                continue;
            }
            return const_cast<void *>(c.p);
        }
        return nullptr;
    };
    if (hazard32) {
        r.scratch32 = pick(o32.n, nullptr, 0);
    }
    if (hazard16) {
        r.scratch16 = pick(o16.n, r.scratch32, hazard32 ? o32.n : 0);
    }
    r.ok = (!hazard32 || r.scratch32 != nullptr) && (!hazard16 || r.scratch16 != nullptr);
    return r;
}
