// C3 capture: dump the selected-key rows (FLASH_ATTN_EXT src[5]) of every QSA layer for chosen ubatches.
// usage: c3_capture model.gguf (text.txt|RANDOM) n_tokens n_ubatch out_prefix ctk pos0[,pos0...]
// Rows file per ubatch: sequence of records {char name[64]; int32 n_q, ns, nk, has_mask, pos0, kt;} + n_q*ns int32.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <set>
#include <fstream>
#include <sstream>

struct state { int64_t pos0 = 0; std::string prefix; FILE * f = nullptr; int n_calls = 0; int kt = 0; };

static bool cb(ggml_tensor * t, bool ask, void * ud) {
    state & st = *(state *) ud;
    const bool hit = t->op == GGML_OP_FLASH_ATTN_EXT && t->src[5] != nullptr;
    if (ask) { return hit; }
    if (!hit) { return true; }
    const ggml_tensor * ids = t->src[5];
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == 4 && ids->ne[2] == 1 && ids->ne[3] == 1);
    struct { char name[64]; int32_t n_q, ns, nk, has_mask, pos0, kt; } h;
    memset(&h, 0, sizeof(h));
    snprintf(h.name, sizeof(h.name), "%s", t->name);
    h.n_q = (int32_t) ids->ne[1]; h.ns = (int32_t) ids->ne[0]; h.nk = (int32_t) t->src[1]->ne[1];
    h.has_mask = t->src[3] != nullptr; h.pos0 = (int32_t) st.pos0; h.kt = (int32_t) t->src[1]->type;
    std::vector<int32_t> buf((size_t) h.n_q * h.ns);
    for (int64_t q = 0; q < h.n_q; ++q) {
        ggml_backend_tensor_get(ids, buf.data() + q * h.ns, q * ids->nb[1], (size_t) h.ns * 4);
    }
    fwrite(&h, sizeof(h), 1, st.f);
    fwrite(buf.data(), 4, buf.size(), st.f);
    fflush(st.f);
    fprintf(stderr, "C3CAP pos0=%lld name=%s n_q=%d ns=%d nk=%d mask=%d kt=%s qtype=%s\n", (long long) st.pos0, h.name, h.n_q, h.ns, h.nk,
            h.has_mask, ggml_type_name(t->src[1]->type), ggml_type_name(t->src[0]->type));
    st.n_calls++;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 8) { fprintf(stderr, "usage: model text|RANDOM n_tokens n_ub out_prefix ctk pos0list\n"); return 1; }
    const char * model_path = argv[1];
    const std::string src = argv[2];
    const int n_tok = atoi(argv[3]), n_ub = atoi(argv[4]);
    state st; st.prefix = argv[5];
    const std::string ctk = argv[6];
    std::set<long long> want;
    { std::stringstream ss(argv[7]); std::string x; while (std::getline(ss, x, ',')) want.insert(atoll(x.c_str())); }
    ggml_type kt = ctk == "q8_0" ? GGML_TYPE_Q8_0 : ctk == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks;
    if (src == "RANDOM") {
        // as llama-bench test_prompt (tools/llama-bench/llama-bench.cpp:2219-2228): srand(1), rand() % n_vocab, BOS first
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        std::srand(1);
        toks.resize(n_tok);
        for (int i = 0; i < n_tok; ++i) toks[i] = (i == 0 && llama_vocab_get_add_bos(vocab)) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;
    } else {
        std::ifstream f(src); std::stringstream ss; ss << f.rdbuf(); std::string text = ss.str();
        toks.resize(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
        if (n < 0) return 1;
        toks.resize(std::min(n, n_tok));
    }
    fprintf(stderr, "tokens %zu\n", toks.size());
    if ((int) toks.size() < n_tok) { fprintf(stderr, "not enough tokens\n"); return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) ((toks.size() + 4096 + 255) / 256 * 256); cp.n_batch = n_ub; cp.n_ubatch = n_ub; cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = kt; cp.type_v = kt;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    for (size_t i = 0; i < toks.size(); i += n_ub) {
        const int m = (int) std::min<size_t>(n_ub, toks.size() - i);
        st.pos0 = (int64_t) i;
        const bool hit = want.count((long long) i) > 0;
        if (hit) {
            char fn[512]; snprintf(fn, sizeof(fn), "%s_p%lld.rows", st.prefix.c_str(), (long long) i);
            st.f = fopen(fn, "wb");
        }
        llama_set_eval_callback(ctx, hit ? cb : nullptr, &st);
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, m))) { fprintf(stderr, "decode failed\n"); return 1; }
        if (hit) { fclose(st.f); st.f = nullptr; fprintf(stderr, "C3CAP done pos0=%zu calls=%d\n", i, st.n_calls); st.n_calls = 0; }
    }
    llama_free(ctx); llama_model_free(model);
    return 0;
}
