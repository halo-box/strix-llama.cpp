// PLAN-v3 §9 / W11: the indexer key cache type of the sparse-attention memory (llama_memory_hybrid_idx), no session format change.
//  1. llama_model_type_k_idx: qwen4exp auto maps q4_0, q4_1, q5_0, q5_1, iq4_nl to q8_0 and keeps every other type,
//     an explicit --cache-type-k-idx wins; every other architecture (GLM5_NEXT) keeps type_k whatever is asked (O26).
//  2. llama_model::create_memory builds the indexer cache with that type: qwen4exp main and MTP contexts, GLM5_NEXT
//     with -ctk q4_0 keeps q4_0 (the synthetic GLM5-Next load of the W11 done-when; no GLM5-Next GGUF is on strix-002).
//  3. Session state: a qwen4exp state written with the old q4_0 indexer is rejected by the existing per-layer key type
//     check of llama_kv_cache, through the wrapped hybrid-idx message; same-type states (f16, q8_0, q4_0 -> auto q8_0,
//     GLM5_NEXT q4_0) round-trip; an attention-section mismatch keeps the cache's own message (not wrapped).
// CPU only, no model file: the caches are the production objects over a one-layer model without weights.
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

// in-memory llama_io_* (the session file writer / reader of llama-context.cpp do the same over a FILE)
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

static llama_cparams make_cparams(uint32_t kv_size, llama_context_type ctx_type) {
    llama_cparams cp = {};
    cp.n_ctx = cp.n_ctx_seq = kv_size;
    cp.n_batch = cp.n_ubatch = 64;
    cp.n_seq_max = 1;
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
                                        uint32_t n_layer = 1, uint32_t n_nextn = 0) {
    f.mem.reset();
    f.model.reset(llama_model_create(arch, llama_model_default_params()));
    set_hparams(*f.model, n_layer, n_nextn);
    const llama_memory_params mp = { tk, tk, tk_idx, false, ctx_type, nullptr };
    const llama_cparams cp = make_cparams(256, ctx_type);
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
    // old (W0 .. W10) qwen4exp file: -ctk q4_0 built a q4_0 indexer; this build maps it to q8_0
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
        // the old build wrote type_k into the indexer, which is what auto still gives for these
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

int main() {
    ggml_backend_load_all();
    test_mapping();
    test_create_memory();
    test_session();
    if (n_fail) {
        printf("test-qsa-idx-type: %d FAILED\n", n_fail);
        return 1;
    }
    printf("test-qsa-idx-type: OK\n");
    return 0;
}
