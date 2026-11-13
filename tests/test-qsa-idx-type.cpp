// The indexer key cache type of llama_memory_hybrid_idx (CPU only, no model file):
//  1. llama_model_type_k_idx: qwen4exp auto maps the 4/5-bit types to q8_0, an explicit --cache-type-k-idx wins; every
//     other architecture (GLM5_NEXT) keeps type_k.
//  2. llama_model::create_memory builds the indexer cache with that type, main and MTP contexts.
//  3. Session state: a state with another indexer type is rejected with the hybrid-idx message; same-type states
//     round-trip; an attention-section mismatch keeps the cache's own message.
//  4. A restore gives the indexer cells the image 2D position of the attention cells, so the compact verdict after a restore is the one before the save.
#include "qsa-mem-fixture.h"

#include "../src/llama-cparams.h"
#include "../src/llama-io.h"
#include "../src/llama-memory.h"
#include "ggml-backend.h"

#include <cstring>
#include <stdexcept>
#include <string>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { ++n_fail; fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

// in-memory llama_io_*
struct io_vec_write : llama_io_write_i {
    std::vector<uint8_t> buf;
    void write(const void * src, size_t size) override { const auto * p = (const uint8_t *) src; buf.insert(buf.end(), p, p + size); }
    void write_tensor(ggml_tensor * t, size_t offset, size_t size) override {
        const size_t n = buf.size(); buf.resize(n + size); ggml_backend_tensor_get(t, buf.data() + n, offset, size);
    }
    size_t n_bytes() override { return buf.size(); }
};
struct io_vec_read : llama_io_read_i {
    const std::vector<uint8_t> & buf; size_t pos = 0;
    explicit io_vec_read(const std::vector<uint8_t> & b) : buf(b) {}
    void read(void * dst, size_t size) override {
        if (pos + size > buf.size()) { throw std::runtime_error("io_vec_read: past the end"); }
        memcpy(dst, buf.data() + pos, size); pos += size;
    }
    void read_tensor(ggml_tensor * t, size_t offset, size_t size) override {
        if (pos + size > buf.size()) { throw std::runtime_error("io_vec_read: past the end"); }
        ggml_backend_tensor_set(t, buf.data() + pos, offset, size); pos += size;
    }
    size_t n_bytes() override { return pos; }
};

static const ggml_type kv_types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,
                                      GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1 };

static void test_mapping() {
    for (ggml_type t : kv_types) {
        const bool sub8 = t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_1 || t == GGML_TYPE_Q5_0 || t == GGML_TYPE_Q5_1 || t == GGML_TYPE_IQ4_NL;
        const ggml_type q = llama_model_type_k_idx(LLM_ARCH_QWEN4EXP, t, GGML_TYPE_COUNT);
        CHECK(q == (sub8 ? GGML_TYPE_Q8_0 : t), "qwen4exp auto -ctk %s -> %s", ggml_type_name(t), ggml_type_name(q));
        const ggml_type g = llama_model_type_k_idx(LLM_ARCH_GLM5_NEXT, t, GGML_TYPE_COUNT);
        CHECK(g == t, "glm5-next auto -ctk %s -> %s", ggml_type_name(t), ggml_type_name(g));
        for (ggml_type o : kv_types) {
            const ggml_type qo = llama_model_type_k_idx(LLM_ARCH_QWEN4EXP, t, o);
            CHECK(qo == o, "qwen4exp -ctk %s --cache-type-k-idx %s -> %s", ggml_type_name(t), ggml_type_name(o), ggml_type_name(qo));
            const ggml_type go = llama_model_type_k_idx(LLM_ARCH_GLM5_NEXT, t, o);
            CHECK(go == t, "glm5-next -ctk %s --cache-type-k-idx %s -> %s", ggml_type_name(t), ggml_type_name(o), ggml_type_name(go));
        }
    }
    printf("mapping: %zu x %zu cases\n", std::size(kv_types), std::size(kv_types) + 1);
}

// a model of n_layer + n_nextn attention layers with one 128-wide KV head and a 128-wide indexer, no recurrent layer
static void set_hparams(llama_model & model, uint32_t n_layer, uint32_t n_nextn) {
    auto & hp = model.hparams;
    hp = {};
    hp.rope_type = LLAMA_ROPE_TYPE_IMROPE;
    hp.n_layer_all   = n_layer + n_nextn;
    hp.n_layer_nextn = n_nextn;
    hp.n_embd = 128;
    hp.n_embd_head_k_full = 128;
    hp.n_embd_head_v_full = 128;
    hp.indexer_head_size  = 128;
    hp.indexer_top_k      = 2048;
    for (uint32_t il = 0; il < hp.n_layer_all; ++il) {
        hp.n_head_arr[il]    = 1;
        hp.n_head_kv_arr[il] = 1;
        hp.is_recr_impl[il]  = false;
        hp.is_indexer_full_impl[il] = true;
    }
}

static llama_cparams make_cparams(uint32_t kv_size, llama_context_type ctx_type, uint32_t n_seq_max = 1) {
    llama_cparams cp = {};
    cp.n_ctx = cp.n_ctx_seq = kv_size;
    cp.n_batch = cp.n_ubatch = 64;
    cp.n_seq_max = n_seq_max;
    cp.n_rs_seq = 0;
    cp.flash_attn = true;
    cp.offload_kqv = false;   // CPU buffers: the model has no devices
    cp.kv_unified = true;
    cp.causal_attn = true;
    cp.ctx_type = ctx_type;
    return cp;
}

// the memory llama_context would create for this model, through llama_model::create_memory
static llama_memory_hybrid_idx * create(qsa_fixture & f, llm_arch arch, ggml_type tk, ggml_type tk_idx, llama_context_type ctx_type,
                                        uint32_t n_layer = 1, uint32_t n_nextn = 0, uint32_t kv_size = 256, uint32_t n_seq_max = 1) {
    f.mem.reset();
    f.model.reset(llama_model_create(arch, llama_model_default_params()));
    set_hparams(*f.model, n_layer, n_nextn);
    const llama_memory_params mp = { tk, tk, tk_idx, false, ctx_type, nullptr };
    const llama_cparams cp = make_cparams(kv_size, ctx_type, n_seq_max);
    llama_memory_i * m = f.model->create_memory(mp, cp);
    auto * h = dynamic_cast<llama_memory_hybrid_idx *>(m);
    if (!h) { delete m; }
    return h;
}

static void test_create_memory() {
    struct row { llm_arch arch; llama_context_type ctx; uint32_t nextn; ggml_type tk, tk_idx, want; const char * what; };
    const row rows[] = {
        { LLM_ARCH_QWEN4EXP,  LLAMA_CONTEXT_TYPE_DEFAULT, 0, GGML_TYPE_Q4_0,   GGML_TYPE_COUNT, GGML_TYPE_Q8_0,   "qwen4exp main q4_0 auto" },
        { LLM_ARCH_QWEN4EXP,  LLAMA_CONTEXT_TYPE_DEFAULT, 0, GGML_TYPE_IQ4_NL, GGML_TYPE_COUNT, GGML_TYPE_Q8_0,   "qwen4exp main iq4_nl auto" },
        { LLM_ARCH_QWEN4EXP,  LLAMA_CONTEXT_TYPE_DEFAULT, 0, GGML_TYPE_F16,    GGML_TYPE_COUNT, GGML_TYPE_F16,    "qwen4exp main f16 auto" },
        { LLM_ARCH_QWEN4EXP,  LLAMA_CONTEXT_TYPE_DEFAULT, 0, GGML_TYPE_Q8_0,   GGML_TYPE_COUNT, GGML_TYPE_Q8_0,   "qwen4exp main q8_0 auto" },
        { LLM_ARCH_QWEN4EXP,  LLAMA_CONTEXT_TYPE_DEFAULT, 0, GGML_TYPE_BF16,   GGML_TYPE_COUNT, GGML_TYPE_BF16,   "qwen4exp main bf16 auto" },
        { LLM_ARCH_QWEN4EXP,  LLAMA_CONTEXT_TYPE_DEFAULT, 0, GGML_TYPE_Q4_0,   GGML_TYPE_Q4_0,  GGML_TYPE_Q4_0,   "qwen4exp main q4_0 override q4_0" },
        { LLM_ARCH_QWEN4EXP,  LLAMA_CONTEXT_TYPE_DEFAULT, 0, GGML_TYPE_Q8_0,   GGML_TYPE_F16,   GGML_TYPE_F16,    "qwen4exp main q8_0 override f16" },
        { LLM_ARCH_QWEN4EXP,  LLAMA_CONTEXT_TYPE_MTP,     1, GGML_TYPE_Q4_0,   GGML_TYPE_COUNT, GGML_TYPE_Q8_0,   "qwen4exp mtp q4_0 auto" },
        { LLM_ARCH_QWEN4EXP,  LLAMA_CONTEXT_TYPE_MTP,     1, GGML_TYPE_F16,    GGML_TYPE_COUNT, GGML_TYPE_F16,    "qwen4exp mtp f16 auto" },
        { LLM_ARCH_GLM5_NEXT, LLAMA_CONTEXT_TYPE_DEFAULT, 0, GGML_TYPE_Q4_0,   GGML_TYPE_COUNT, GGML_TYPE_Q4_0,   "glm5-next main q4_0 auto" },
        { LLM_ARCH_GLM5_NEXT, LLAMA_CONTEXT_TYPE_DEFAULT, 0, GGML_TYPE_Q4_0,   GGML_TYPE_Q8_0,  GGML_TYPE_Q4_0,   "glm5-next main q4_0 override q8_0 (ignored)" },
        { LLM_ARCH_GLM5_NEXT, LLAMA_CONTEXT_TYPE_DEFAULT, 0, GGML_TYPE_Q8_0,   GGML_TYPE_COUNT, GGML_TYPE_Q8_0,   "glm5-next main q8_0 auto" },
        { LLM_ARCH_GLM5_NEXT, LLAMA_CONTEXT_TYPE_MTP,     1, GGML_TYPE_Q4_0,   GGML_TYPE_COUNT, GGML_TYPE_Q4_0,   "glm5-next mtp q4_0 auto" },
    };
    for (const auto & r : rows) {
        qsa_fixture f(256, 1, true);
        f.mem.reset();
        f.mem.reset(create(f, r.arch, r.tk, r.tk_idx, r.ctx, 1, r.nextn));
        CHECK(f.mem != nullptr, "%s: create_memory gave no hybrid-idx memory", r.what);
        if (!f.mem) { continue; }
        const llama_kv_cache * idx = f.mem->get_mem_idx();
        CHECK(idx != nullptr, "%s: no indexer cache", r.what);
        if (!idx) { continue; }
        const auto ids = idx->get_layer_ids();
        CHECK(ids.size() == 1, "%s: %zu indexer layers", r.what, ids.size());
        const ggml_type got_t = ids.empty() ? GGML_TYPE_COUNT : idx->type_k();
        const ggml_type attn_t = f.mem->get_mem_attn()->type_k();
        CHECK(f.mem->get_type_k_idx() == r.want && got_t == r.want && attn_t == r.tk,
              "%s: type_k_idx %s, indexer tensor %s, attention K %s (want idx %s, K %s)", r.what,
              ggml_type_name(f.mem->get_type_k_idx()), got_t == GGML_TYPE_COUNT ? "none" : ggml_type_name(got_t),
              ggml_type_name(attn_t), ggml_type_name(r.want), ggml_type_name(r.tk));
        printf("create_memory: %-46s K %-6s idx %-6s (layer %d)\n", r.what, ggml_type_name(attn_t),
               got_t == GGML_TYPE_COUNT ? "none" : ggml_type_name(got_t), ids.empty() ? -1 : (int) ids[0]);
    }
}

// one sequence of 40 text tokens through init_batch / apply, then its whole-context state
static std::vector<uint8_t> save_state(llm_arch arch, ggml_type tk, ggml_type tk_idx) {
    qsa_fixture f(256, 1, true);
    f.mem.reset();
    f.mem.reset(create(f, arch, tk, tk_idx, LLAMA_CONTEXT_TYPE_DEFAULT));
    GGML_ASSERT(f.mem);
    GGML_ASSERT(f.feed(f.text(0, 40), 64));
    io_vec_write w;
    f.mem->state_write(w, -1, 0);
    return w.buf;
}

// load into a fresh memory; "" on success, the exception's message otherwise
static std::string load_state(const std::vector<uint8_t> & buf, llm_arch arch, ggml_type tk, ggml_type tk_idx, uint32_t * n_cells = nullptr) {
    qsa_fixture f(256, 1, true);
    f.mem.reset();
    f.mem.reset(create(f, arch, tk, tk_idx, LLAMA_CONTEXT_TYPE_DEFAULT));
    GGML_ASSERT(f.mem);
    io_vec_read r(buf);
    try {
        f.mem->state_read(r, -1, 0);
    } catch (const std::exception & e) {
        // a failed restore drops what it read from every cache
        CHECK(f.mem->get_mem_attn()->get_cells(0).get_used() == 0 && f.mem->get_mem_idx()->get_cells(0).get_used() == 0,
              "a failed restore left cells behind");
        return e.what();
    }
    CHECK(r.pos == buf.size(), "read %zu of %zu bytes", r.pos, buf.size());
    if (n_cells) { *n_cells = f.mem->get_mem_idx()->get_cells(0).get_used(); }
    return "";
}

static void test_session() {
    // a qwen4exp file with a q4_0 indexer: auto now gives q8_0
    const auto old_q4 = save_state(LLM_ARCH_QWEN4EXP, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
    const std::string e = load_state(old_q4, LLM_ARCH_QWEN4EXP, GGML_TYPE_Q4_0, GGML_TYPE_COUNT);
    printf("session: old qwen4exp q4_0 file -> auto: \"%s\"\n", e.c_str());
    CHECK(e.find("failed to restore the sparse-attention indexer cache (first QSA layer 0)") != std::string::npos &&
          e.find("saved with indexer type q4_0, this context uses q8_0") != std::string::npos &&
          e.find("re-save the session, or load it with --cache-type-k-idx q4_0") != std::string::npos, "message: %s", e.c_str());
    // the override named in the message loads it
    uint32_t n = 0;
    std::string e2 = load_state(old_q4, LLM_ARCH_QWEN4EXP, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, &n);
    CHECK(e2.empty() && n == 40, "old q4_0 file with --cache-type-k-idx q4_0: \"%s\", %u cells", e2.c_str(), n);
    printf("session: old qwen4exp q4_0 file -> --cache-type-k-idx q4_0: %s (%u indexer cells)\n", e2.empty() ? "loaded" : e2.c_str(), n);

    // files whose indexer type does not change load as before
    struct row { llm_arch arch; ggml_type tk; const char * what; };
    const row rows[] = {
        { LLM_ARCH_QWEN4EXP,  GGML_TYPE_F16,  "qwen4exp f16" },
        { LLM_ARCH_QWEN4EXP,  GGML_TYPE_Q8_0, "qwen4exp q8_0" },
        { LLM_ARCH_GLM5_NEXT, GGML_TYPE_Q4_0, "glm5-next q4_0" },
        { LLM_ARCH_GLM5_NEXT, GGML_TYPE_F16,  "glm5-next f16" },
    };
    for (const auto & r : rows) {
        const auto buf = save_state(r.arch, r.tk, r.tk);
        n = 0;
        const std::string el = load_state(buf, r.arch, r.tk, GGML_TYPE_COUNT, &n);
        CHECK(el.empty() && n == 40, "%s: \"%s\", %u cells", r.what, el.c_str(), n);
        printf("session: %-16s file -> auto: %s (%u indexer cells)\n", r.what, el.empty() ? "loaded" : el.c_str(), n);
    }

    // a new qwen4exp q4_0 file (q8_0 indexer) round-trips
    const auto new_q4 = save_state(LLM_ARCH_QWEN4EXP, GGML_TYPE_Q4_0, GGML_TYPE_COUNT);
    n = 0;
    const std::string e3 = load_state(new_q4, LLM_ARCH_QWEN4EXP, GGML_TYPE_Q4_0, GGML_TYPE_COUNT, &n);
    CHECK(e3.empty() && n == 40, "new q4_0 file: \"%s\", %u cells", e3.c_str(), n);
    printf("session: new qwen4exp q4_0 file -> auto: %s (%u indexer cells)\n", e3.empty() ? "loaded" : e3.c_str(), n);

    // a mismatch in the attention section (another -ctk) is not the indexer's: the cache's own message, unwrapped
    const std::string e4 = load_state(new_q4, LLM_ARCH_QWEN4EXP, GGML_TYPE_Q8_0, GGML_TYPE_COUNT);
    printf("session: qwen4exp q4_0 file -> -ctk q8_0: \"%s\"\n", e4.c_str());
    CHECK(e4 == "failed to restore kv cache", "attention mismatch: %s", e4.c_str());
}

// the 2D cells of seq in a cache, and the cells whose (x, y) differ from those of the same cell in ref
static uint32_t n_2d_cells(const llama_kv_cache * kv, llama_seq_id seq) {
    const auto & c = kv->get_cells(seq);
    uint32_t n = 0;
    for (uint32_t j = 0; j < c.size(); ++j) {
        n += !c.is_empty(j) && c.seq_has(j, seq) && qsa_key_2d(c, j);
    }
    return n;
}
static uint32_t n_2d_mismatch(const llama_kv_cache * kv, const llama_kv_cache * ref, llama_seq_id seq) {
    const auto & c = kv->get_cells(seq);
    const auto & r = ref->get_cells(seq);
    uint32_t n = 0;
    for (uint32_t j = 0; j < c.size(); ++j) {
        if (c.is_empty(j)) { continue; }
        n += c.ext_get(j).x != r.ext_get(j).x || c.ext_get(j).y != r.ext_get(j).y;
    }
    return n;
}

// the compact verdict of a 10-token text ubatch of seq 0, after the cache took it
static int next_verdict(qsa_fixture & f) {
    int reason = -1;
    GGML_ASSERT(f.feed(f.text(0, 10), 64, [&](const llama_memory_hybrid_idx_context * m, const llama_ubatch & ub, uint32_t) {
        reason = m->qsa_compact_reason(ub, 4);
    }));
    return reason;
}

// the indexer has no rope of its own, so its saved state carries no x/y; the compact predicate and the scan's ranking read them from the indexer cells, so a restore takes them from the attention cells (same layout).
// Slot 0 holds text, an image and text; with two slots, slot 1 holds text in the saving and in the restoring memory, as in a server.
static void test_restore_2d() {
    for (const uint32_t n_seq : { 1u, 2u }) {
        qsa_fixture a(1024, n_seq, true);
        a.mem.reset(create(a, LLM_ARCH_QWEN4EXP, GGML_TYPE_F16, GGML_TYPE_COUNT, LLAMA_CONTEXT_TYPE_DEFAULT, 1, 0, 1024, n_seq));
        GGML_ASSERT(a.mem);
        GGML_ASSERT(a.feed(a.text(0, 20), 64));
        GGML_ASSERT(a.feed(a.image(0, 8, 8), 64));
        GGML_ASSERT(a.feed(a.text(0, 20), 64));
        if (n_seq > 1) { GGML_ASSERT(a.feed(a.text(1, 30), 64)); }
        io_vec_write w;
        a.mem->state_write(w, 0, 0);

        qsa_fixture b(1024, n_seq, true);
        b.mem.reset(create(b, LLM_ARCH_QWEN4EXP, GGML_TYPE_F16, GGML_TYPE_COUNT, LLAMA_CONTEXT_TYPE_DEFAULT, 1, 0, 1024, n_seq));
        GGML_ASSERT(b.mem);
        if (n_seq > 1) { GGML_ASSERT(b.feed(b.text(1, 30), 64)); }
        io_vec_read r(w.buf);
        std::string err;
        try {
            b.mem->state_read(r, 0, 0);
        } catch (const std::exception & e) {
            err = e.what();
        }
        CHECK(err.empty(), "n_seq %u: restore failed: %s", n_seq, err.c_str());
        if (!err.empty()) { continue; }
        b.next_pos[0] = a.next_pos[0];

        const uint32_t n_2d_a   = n_2d_cells(a.mem->get_mem_idx(), 0);
        const uint32_t n_2d_b   = n_2d_cells(b.mem->get_mem_idx(), 0);
        const uint32_t n_attn_b = n_2d_cells(b.mem->get_mem_attn(), 0);
        const uint32_t n_mis    = n_2d_mismatch(b.mem->get_mem_idx(), b.mem->get_mem_attn(), 0);
        const int      v_a      = next_verdict(a);
        const int      v_b      = next_verdict(b);
        printf("restore 2d: %u slot(s): indexer 2D cells saved %u, restored %u (attention %u), x/y mismatches %u; next text ubatch %s / %s\n",
               n_seq, n_2d_a, n_2d_b, n_attn_b, n_mis, qsa_compact_reason_name(v_a), qsa_compact_reason_name(v_b));
        CHECK(n_2d_a > 0 && n_2d_b == n_2d_a && n_attn_b == n_2d_a, "n_seq %u: 2D cells saved %u, restored %u, attention %u",
              n_seq, n_2d_a, n_2d_b, n_attn_b);
        CHECK(n_mis == 0, "n_seq %u: %u indexer cells with another x/y than the attention cell", n_seq, n_mis);
        CHECK(v_a == (n_seq > 1 ? QSA_NC_2D_MULTI_SEQ : QSA_COMPACT_OK) && v_b == v_a, "n_seq %u: verdict before %s, after the restore %s",
              n_seq, qsa_compact_reason_name(v_a), qsa_compact_reason_name(v_b));
    }
}

int main() {
    ggml_backend_load_all();
    test_mapping();
    test_create_memory();
    test_session();
    test_restore_2d();
    if (n_fail) {
        printf("test-qsa-idx-type: %d FAILED\n", n_fail);
        return 1;
    }
    printf("test-qsa-idx-type: OK\n");
    return 0;
}
