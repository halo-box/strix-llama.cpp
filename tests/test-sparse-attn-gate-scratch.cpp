// The output placement of the HIP gated SPARSE_ATTN fusion (ggml/src/ggml-cuda/sparse-attn-gate-scratch.h), on synthetic
// layouts of one compute buffer. PLAN-v3 D12: no sparse kernel uses ctx.pool(); every workspace is in the reserved compute
// buffer, so the reserve, the fit probe and LLAMA_STRICT_RESERVE see it. When gallocr puts the MUL output (or its 16-bit copy)
// over a tensor the kernel reads (Q is dead after the attend and has the MUL's size: the common case), the output must go to
// graph memory the fusion leaves unwritten (the attend's own dst, the CONT and SIGMOID outputs), or the fusion is refused;
// never to pool scratch. For every layout: no output in the pool; a scratch lies inside one candidate, overlaps no read, no
// final output and no other scratch; the fusion is refused only when no candidate fits. CPU only.
#include "../ggml/src/ggml-cuda/sparse-attn-gate-scratch.h"

#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); ++g_fail; } } while (0)

static char arena[1 << 20];

struct layout {
    const char * name;
    // offsets (bytes) in the arena; n32 bytes per f32 tensor of the attend's shape, n32 / 2 for the 16-bit copy
    size_t q, k, v, plan, gate, ws;    // reads (the workspace right after dst: its offset is given)
    size_t dst, cont, unary;           // candidates
    long   mul, d16;                   // outputs, -1 when not written
    bool   expect_fused;               // a candidate fits
};

int main() {
    const size_t n32 = 4096, n16 = n32 / 2, small = 512;
    // tensor slots of n32 bytes: slot(i) = i * n32
    auto slot = [&](size_t i) { return i * n32; };
    const std::vector<layout> layouts = {
        // Prove's case: the MUL over Q (dead after the attend); no 16-bit copy
        { "mul over q",                slot(0), slot(1), slot(2), slot(3), slot(4), slot(5) + n32, slot(5), slot(7), slot(8), (long) slot(0), -1, true },
        // the MUL over Q and the 16-bit copy over the gate
        { "mul over q, d16 over gate", slot(0), slot(1), slot(2), slot(3), slot(4), slot(5) + n32, slot(5), slot(7), slot(8), (long) slot(0), (long) slot(4), true },
        // the 16-bit copy only (bf16-only reader), over K
        { "d16 over k",                slot(0), slot(1), slot(2), slot(3), slot(4), slot(5) + n32, slot(5), slot(7), slot(8), -1, (long) slot(1), true },
        // the MUL in place over the attend's dst (gallocr's MUL in-place): no hazard, written in place
        { "mul over dst (in place)",   slot(0), slot(1), slot(2), slot(3), slot(4), slot(5) + n32, slot(5), slot(7), slot(8), (long) slot(5), -1, true },
        // the MUL over Q and the SIGMOID in place over the CONT: only dst is left for the f32 output
        { "mul over q, unary on cont", slot(0), slot(1), slot(2), slot(3), slot(4), slot(5) + n32, slot(5), slot(7), slot(7), (long) slot(0), -1, true },
        // both outputs hazardous, the CONT / SIGMOID over Q (freed) as well: dst takes one, nothing is left for the other
        { "no room for both",          slot(0), slot(1), slot(2), slot(3), slot(4), slot(5) + n32, slot(5), slot(0), slot(0), (long) slot(0), (long) slot(4), false },
    };
    for (const layout & L : layouts) {
        const ggml_cuda_sa_span reads[6] = {
            { arena + L.q, n32 }, { arena + L.k, n32 }, { arena + L.v, n32 }, { arena + L.plan, small }, { arena + L.gate, n32 }, { arena + L.ws, small } };
        const ggml_cuda_sa_span cands[3] = { { arena + L.dst, n32 }, { arena + L.cont, n32 }, { arena + L.unary, n32 } };
        const ggml_cuda_sa_span o32 = { L.mul >= 0 ? arena + L.mul : nullptr, L.mul >= 0 ? n32 : 0 };
        const ggml_cuda_sa_span o16 = { L.d16 >= 0 ? arena + L.d16 : nullptr, L.d16 >= 0 ? n16 : 0 };
        const ggml_cuda_sa_gate_plan gp = ggml_cuda_sa_gate_plan_outputs(reads, 6, o32, o16, cands, 3);
        const char * wn[] = { "none", "direct", "pool", "scratch" };
        printf("  %-26s: fused %d, f32 %s, 16-bit %s\n", L.name, gp.ok, wn[gp.where32 & 3], wn[gp.where16 & 3]);
        CHECK(gp.ok == L.expect_fused, "%s: fused %d, expected %d", L.name, gp.ok, L.expect_fused);
        if (!gp.ok) {
            continue;
        }
        CHECK(gp.where32 != GGML_CUDA_SA_GATE_POOL && gp.where16 != GGML_CUDA_SA_GATE_POOL, "%s: an output goes to pool scratch (unreserved, D12)", L.name);
        CHECK((gp.where32 == GGML_CUDA_SA_GATE_NONE) == (o32.n == 0) && (gp.where16 == GGML_CUDA_SA_GATE_NONE) == (o16.n == 0), "%s: an output is dropped", L.name);
        // where each output is finally written by the kernel
        struct w { const void * p; size_t n; bool scratch; };
        std::vector<w> writes;
        for (int o = 0; o < 2; ++o) {
            const int where = o == 0 ? gp.where32 : gp.where16;
            const ggml_cuda_sa_span out = o == 0 ? o32 : o16;
            void * s = o == 0 ? gp.scratch32 : gp.scratch16;
            if (where == GGML_CUDA_SA_GATE_NONE) {
                continue;
            }
            if (where == GGML_CUDA_SA_GATE_DIRECT) {
                CHECK(!ggml_cuda_sa_overlaps_any(reads, 6, out.p, out.n), "%s: output %d written in place over a read", L.name, o);
                writes.push_back({ out.p, out.n, false });
                continue;
            }
            CHECK(s != nullptr, "%s: output %d has no scratch", L.name, o);
            bool inside = false;
            for (const auto & c : cands) {
                inside = inside || ((const char *) s >= (const char *) c.p && (const char *) s + out.n <= (const char *) c.p + c.n);
            }
            CHECK(inside, "%s: the scratch of output %d is not inside a candidate", L.name, o);
            CHECK(!ggml_cuda_sa_overlaps_any(reads, 6, s, out.n), "%s: the scratch of output %d overlaps a read", L.name, o);
            CHECK(!ggml_cuda_sa_overlap(s, out.n, o32.p, o32.n) && !ggml_cuda_sa_overlap(s, out.n, o16.p, o16.n),
                  "%s: the scratch of output %d overlaps a final output", L.name, o);
            writes.push_back({ s, out.n, true });
        }
        for (size_t a = 0; a < writes.size(); ++a) {
            for (size_t b = a + 1; b < writes.size(); ++b) {
                CHECK(!ggml_cuda_sa_overlap(writes[a].p, writes[a].n, writes[b].p, writes[b].n), "%s: two kernel outputs overlap", L.name);
            }
        }
    }
    printf("test-sparse-attn-gate-scratch: %d failed checks: %s\n", g_fail, g_fail == 0 ? "OK" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}
