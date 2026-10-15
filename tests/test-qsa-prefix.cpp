#include "../src/qsa-prefix-state.h"
#include "../src/prefix.h"
#include "../src/llama-memory-hybrid-idx.h"
#include "../src/llama-model.h"
#include "ggml.h"
#include <random>
#include <iostream>

// cells filled in order, one per position
struct qsa_cells {
    llama_kv_cells cells;
    uint32_t next = 0;
    explicit qsa_cells(uint32_t n) { cells.resize(n); }
    void add(llama_pos p0, llama_pos p1, std::vector<llama_seq_id> seqs, llama_pos ext_dy = 0) {
        for (llama_pos p=p0; p<p1; ++p, ++next) {
            cells.pos_set(next, p);
            for (auto s : seqs) { cells.seq_add(next, s); }
            if (ext_dy) { cells.ext_set(next, { p, p+ext_dy }); }
        }
    }
};

// one token per entry, n_pos axes all equal to the linear position
struct qsa_ubatch {
    std::vector<llama_token> tok; std::vector<llama_pos> pos; std::vector<int32_t> n_seq_id;
    std::vector<std::vector<llama_seq_id>> ids; std::vector<llama_seq_id *> seq_id; llama_ubatch u{};
    qsa_ubatch(std::vector<std::pair<llama_pos, std::vector<llama_seq_id>>> tokens, uint32_t n_pos = 1) {
        const uint32_t n = tokens.size();
        tok.assign(n, 0); pos.resize(n*n_pos); ids.resize(n);
        for (uint32_t i=0; i<n; ++i) {
            for (uint32_t a=0; a<n_pos; ++a) { pos[i+a*n] = tokens[i].first; }
            ids[i] = tokens[i].second; n_seq_id.push_back(ids[i].size()); seq_id.push_back(ids[i].data());
        }
        u.n_tokens = n; u.n_pos = n_pos; u.token = tok.data(); u.pos = pos.data(); u.n_seq_id = n_seq_id.data(); u.seq_id = seq_id.data();
    }
};

static bool scalar(const qsa_cells & c, const qsa_ubatch & ub) {
    return qsa_scalar_visibility_cells(c.cells, c.cells.size(), 4, ub.u);
}

// the gate that picks the compact (maskless) QSA visibility over the masked one
static void test_scalar_visibility() {
    {   // independent sequences decode together
        qsa_cells c(64); c.add(0, 13, {0}); c.add(0, 13, {1});
        GGML_ASSERT(scalar(c, qsa_ubatch({{13, {0}}, {13, {1}}})));
    }
    {   // seq_cp inside a block: position 12 is shared, 13 and 14 are not, so block [12, 16) has no complete group
        qsa_cells c(64); c.add(0, 13, {0, 1}); c.add(13, 15, {0}); c.add(13, 15, {1}); c.add(0, 6, {2});
        GGML_ASSERT(!scalar(c, qsa_ubatch({{15, {0}}, {15, {1}}})));
        GGML_ASSERT(!scalar(c, qsa_ubatch({{15, {1}}})));
        GGML_ASSERT(scalar(c, qsa_ubatch({{6, {2}}})));
    }
    {   // seq_cp on a block boundary splits nothing
        qsa_cells c(64); c.add(0, 12, {0, 1}); c.add(12, 14, {0}); c.add(12, 14, {1});
        GGML_ASSERT(scalar(c, qsa_ubatch({{14, {0}}, {14, {1}}})));
    }
    {   // a prompt shared at token level reads the row of its first seq id, as the attention mask does
        qsa_cells c(64); c.add(0, 8, {0, 1, 2, 3});
        GGML_ASSERT(scalar(c, qsa_ubatch({{8, {0, 1, 2, 3}}, {9, {0, 1, 2, 3}}})));
    }
    {   // a 2-D image cell only matters to the sequences that hold it
        qsa_cells c(64); c.add(0, 8, {0}); c.add(0, 4, {1}); c.add(4, 5, {1}, 1);
        GGML_ASSERT(scalar(c, qsa_ubatch({{8, {0}}}, 3)));
        GGML_ASSERT(!scalar(c, qsa_ubatch({{5, {1}}}, 3)));
        GGML_ASSERT(!scalar(c, qsa_ubatch({{8, {0}}, {5, {1}}}, 3)));
    }
    {   // no seq id, or a position with 2-D extents in the ubatch
        qsa_cells c(64); c.add(0, 8, {0});
        qsa_ubatch none({{8, {0}}}); none.n_seq_id[0] = 0;
        GGML_ASSERT(!scalar(c, none));
        qsa_ubatch ub({{8, {0}}}, 3); ub.pos[1] = 9;
        GGML_ASSERT(!scalar(c, ub));
    }
}

// Exercise the production scan without model weights or device execution.
static void test_scan_foreign_image() {
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN4EXP, llama_model_default_params()));
    model->hparams = {};
    llama_memory_hybrid_idx mem(*model, GGML_TYPE_F16, GGML_TYPE_F16, false, 64, 1, 0, LLAMA_SWA_TYPE_NONE,
                               GGML_TYPE_F32, GGML_TYPE_F32, 4, 3, 0, false, true,
                               nullptr, nullptr, [](int32_t) { return false; });
    auto & cells = const_cast<llama_kv_cells &>(mem.get_mem_idx()->get_cells(0));
    for (uint32_t j = 0; j < 10; ++j) {
        cells.pos_set(j, j);
        cells.seq_add(j, 0);
        if (j < 4) { cells.seq_add(j, 2); }
    }
    for (uint32_t j = 16; j < 24; ++j) {
        cells.pos_set(j, 4);
        cells.seq_add(j, 1);
        cells.ext_set(j, {4, llama_pos(j - 15)});
    }
    for (uint32_t j = 32; j < 38; ++j) {
        cells.pos_set(j, j - 28);
        cells.seq_add(j, 2);
    }
    qsa_ubatch ub({{8, {0}}, {9, {2}}}, 3);
    GGML_ASSERT(qsa_scalar_visibility_cells(cells, 64, 4, ub.u));

    for (bool compact : {false, true}) {
        ggml_context_ptr ctx(ggml_init({8*ggml_tensor_overhead(), nullptr, true}));
        auto * map = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 64, 1);
        auto * members = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 64, 1);
        auto * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 64);
        auto * bias = compact ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 3*16 + 2*2)
                             : ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 16, 2);
        auto * tails = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I32, 3, 2, 1);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
        GGML_ASSERT(buffer);
        mem.set_input_qsa_blocks(map, members, pos, bias, tails, &ub.u, 4);
        const int32_t expected_tail[] = {8, -1, -1, 36, 37, -1};
        GGML_ASSERT(std::equal(expected_tail, expected_tail + 6, (int32_t *) tails->data));
        for (int query = 0; query < 2; ++query) {
            std::vector<int32_t> selected;
            for (int b = 0; b < 16; ++b) {
                const bool visible = compact ? ((int32_t *) bias->data)[ub.ids[query][0]*16 + b] < 8
                                             : ((float *) bias->data)[query*16 + b] == 0.0f;
                if (visible) {
                    const auto * block = (int32_t *) members->data + 4*b;
                    selected.insert(selected.end(), block, block + 4);
                }
            }
            std::sort(selected.begin(), selected.end());
            const std::vector<int32_t> expected = query == 0 ? std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}
                                                            : std::vector<int32_t>{0, 1, 2, 3, 32, 33, 34, 35};
            GGML_ASSERT(selected == expected);
        }
    }
}

int main() {
    test_scan_foreign_image();
    test_scalar_visibility();
    qsa_prefix_state s(4096);
    std::mt19937 rng(414);
    for (int round=0; round<200; ++round) {
        s.reset();
        for (int step=0; step<100; ++step) {
            if (!s.cells.empty() && rng()%3==0) { s.truncate(rng()%(s.cells.size()+1)); }
            size_t start=s.cells.size();
            int n=1+rng()%8;
            std::vector<uint32_t> slots;
            while (int(slots.size())<n) {
                uint32_t c=rng()%4096;
                if (s.positions[c]<0 && std::find(slots.begin(),slots.end(),c)==slots.end()) slots.push_back(c);
            }
            auto before=s.cells;
            GGML_ASSERT(s.apply(0,start,slots));
            GGML_ASSERT(s.previous_size==before.size());
            GGML_ASSERT(std::equal(before.begin(),before.end(),s.cells.begin()));
            for (size_t i=0;i<s.cells.size();++i) GGML_ASSERT(s.positions[s.cells[i]]==int(i));
            for (size_t b=0;b<s.block_positions.size();++b) GGML_ASSERT(s.block_positions[b]==int(4*b));
            if (s.cells.size()>8) {
                int at=rng()%(s.cells.size()-4);
                std::vector<uint32_t> same(s.cells.begin()+at,s.cells.begin()+at+4);
                auto old=s.cells;GGML_ASSERT(s.apply(0,at,same));GGML_ASSERT(s.cells==old);
            }
        }
    }
    s.reset(); GGML_ASSERT(!s.apply(0,2,{3}));
    s.reset(); GGML_ASSERT(!s.apply(0,0,{3,3}));
    s.reset(); GGML_ASSERT(s.apply(0,0,{3,7})); GGML_ASSERT(!s.apply(1,2,{4}));
    s.reset(); GGML_ASSERT(s.apply(0,0,{3,7})); GGML_ASSERT(!s.apply(0,2,{3}));
    s.reset(); GGML_ASSERT(s.apply(0,0,{3,7})); GGML_ASSERT(!s.apply(0,0,{9}));
    s.reset(); GGML_ASSERT(s.apply(0,0,{3,7})); s.truncate(0); GGML_ASSERT(s.apply(1,0,{3}));
    s.invalidate(); GGML_ASSERT(!s.apply(1,1,{4}));
    std::cout << "PASS: 20000 randomized append/rollback steps, in-place rewrites, holes, duplicates, collisions, sequence changes\n";
}
