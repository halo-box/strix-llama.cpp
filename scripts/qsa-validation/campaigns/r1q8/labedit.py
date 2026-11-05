# r1q8 lab edit: measurement-only defect hooks downstream of the QPIN pin copy + full-precision per-chunk KLD
import sys, re
root = sys.argv[1]
def sub(path, old, new, count=1):
    p = root + '/' + path
    s = open(p).read()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    s = s.replace(old, new)
    open(p, 'w').write(s)

# ---- defect helpers, declared before qpin_rows
sub('src/llama-context.cpp', 'void qpin_rows(ggml_tensor * t, int kind, int il) {', r'''// ---- R1Q8 (measurement only): planted defects. LAB_DEF=d1|d2|d3  LAB_DEF_PRE=1 (apply to the computed rows BEFORE the pin copy)
// d1: layer LAB_D1_L (23): drop the last valid tail cell (row slots W-3..W-1) of every query row (-> -1)
// d2: layer LAB_D2_L (3):  drop one selected block (4 cells) per query row: the median valid block of the row (-> -1)
// d3: layer LAB_D3_L (23): attn_output x (1 + LAB_D3_EPS) (default 1e-3) == V x (1+eps): softmax weights sum to 1, wo is linear
struct lab_def_state { int def = 0; bool pre = false; int l1 = 23, l2 = 3, l3 = 23; double eps = 1e-3;
    int64_t n_d1 = 0, n_d1_self = 0, n_d2 = 0, n_d3 = 0, n_rows_l = 0; bool init = false; };
lab_def_state g_lab;
void lab_init() {
    if (g_lab.init) { return; }
    g_lab.init = true;
    if (const char * d = getenv("LAB_DEF")) { g_lab.def = !strcmp(d, "d1") ? 1 : !strcmp(d, "d2") ? 2 : !strcmp(d, "d3") ? 3 : 0; }
    g_lab.pre = getenv("LAB_DEF_PRE") && atoi(getenv("LAB_DEF_PRE"));
    if (getenv("LAB_D1_L")) { g_lab.l1 = atoi(getenv("LAB_D1_L")); }
    if (getenv("LAB_D2_L")) { g_lab.l2 = atoi(getenv("LAB_D2_L")); }
    if (getenv("LAB_D3_L")) { g_lab.l3 = atoi(getenv("LAB_D3_L")); }
    if (getenv("LAB_D3_EPS")) { g_lab.eps = atof(getenv("LAB_D3_EPS")); }
    fprintf(stderr, "R1Q8: def=%d pre=%d l1=%d l2=%d l3=%d eps=%.3e\n", g_lab.def, (int) g_lab.pre, g_lab.l1, g_lab.l2, g_lab.l3, g_lab.eps);
}
// apply d1/d2 to one row in place; p = query position
void lab_def_row(int32_t * row, int64_t W, int il, int64_t p) {
    if (g_lab.def == 1 && il == g_lab.l1) {
        for (int64_t j = W - 1; j >= W - 3; --j) {
            if (row[j] >= 0) { g_lab.n_d1++; g_lab.n_d1_self += row[j] == p; row[j] = -1; break; }
        }
    } else if (g_lab.def == 2 && il == g_lab.l2) {
        const int64_t nb = (W - 3) / 4;
        int64_t nv = 0;
        for (int64_t b = 0; b < nb; ++b) { nv += row[4*b] >= 0; }
        if (nv == 0) { return; }
        int64_t want = nv / 2, k = 0;
        for (int64_t b = 0; b < nb; ++b) {
            if (row[4*b] < 0) { continue; }
            if (k++ == want) { for (int c = 0; c < 4; ++c) { row[4*b + c] = -1; } g_lab.n_d2++; break; }
        }
    }
}

void qpin_rows(ggml_tensor * t, int kind, int il) {''')

# producer-side variant: corrupt the computed row before the pin copy (must be erased)
sub('src/llama-context.cpp', '''        if (S.pin_qsa) {
            for (int64_t j = 0; j < W; ++j) {
                cur[j] = (kind == 2 && ref[j] < 0) ? first_valid : ref[j];
            }
            modified = true;
            S.n_rows_pin++;
        }''', '''        if (kind == 1 && (il == g_lab.l1 || il == g_lab.l2)) { g_lab.n_rows_l++; }
        if (kind == 1 && g_lab.pre) { lab_def_row(cur, W, il, p); modified = true; }
        if (S.pin_qsa) {
            for (int64_t j = 0; j < W; ++j) {
                cur[j] = (kind == 2 && ref[j] < 0) ? first_valid : ref[j];
            }
            modified = true;
            S.n_rows_pin++;
        }
        if (kind == 1 && !g_lab.pre && g_lab.def) { lab_def_row(cur, W, il, p); modified = true; }''')

# d3 + always-observed attn_output-<l3> (so every arm, the reference included, has the same graph splits)
sub('src/llama-context.cpp', '''    else if (strncmp(name, "ffn_moe_topk-", 13) == 0) { kind = 3; }
    if (ask) { return kind != 0; }''', '''    else if (strncmp(name, "ffn_moe_topk-", 13) == 0) { kind = 3; }
    else if (strncmp(name, "attn_output-", 12) == 0 && atoi(name + 12) == g_lab.l3) { kind = 4; }
    if (ask) { return kind != 0; }
    if (kind == 4) {
        if (g_lab.def == 3 && g_qpin.mode == 2 && g_qpin.chunk >= 0 && !g_qpin.pos.empty()) {
            GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
            std::vector<float> v(ggml_nelements(t));
            ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
            for (float & x : v) { x = (float) (x * (1.0 + g_lab.eps)); }
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
            g_lab.n_d3++;
        }
        return true;
    }''')

sub('src/llama-context.cpp', '''    LLAMA_LOG_WARN("QPIN: mode=%s dir=%s pin_qsa=%d pin_moe=%d\\n", m, d, g_qpin.pin_qsa, g_qpin.pin_moe);''',
'''    LLAMA_LOG_WARN("QPIN: mode=%s dir=%s pin_qsa=%d pin_moe=%d\\n", m, d, g_qpin.pin_qsa, g_qpin.pin_moe);
    lab_init();''')

sub('src/llama-context.cpp', '''    if (f) { fclose(f); }
    for (auto & [n, fd] : S.fds)''', '''    out("R1Q8_SUMMARY def=%d pre=%d rows_at_l=%lld d1_dropped=%lld d1_dropped_self=%lld d2_dropped_blocks=%lld d3_scaled_tensors=%lld\\n",
        g_lab.def, (int) g_lab.pre, (long long) g_lab.n_rows_l, (long long) g_lab.n_d1, (long long) g_lab.n_d1_self,
        (long long) g_lab.n_d2, (long long) g_lab.n_d3);
    if (f) { fclose(f); }
    for (auto & [n, fd] : S.fds)''')

# full-precision per-chunk KLD
sub('tools/perplexity/perplexity.cpp', '''            process_logits(n_vocab, all_logits, tokens.data() + start + seq*n_ctx + first, n_ctx - 1 - first,
                    workers, log_probs_uint16, kld, kld_ptr, p_diff_ptr);''', '''            const double r1q8_s0 = kld.sum_kld; const size_t r1q8_c0 = kld.count, r1q8_t0 = kld.n_same_top;
            process_logits(n_vocab, all_logits, tokens.data() + start + seq*n_ctx + first, n_ctx - 1 - first,
                    workers, log_probs_uint16, kld, kld_ptr, p_diff_ptr);
            LOG("QKLD_CHUNK %d n=%zu kld=%.9e same_top=%.9e\\n", i + seq + 1, kld.count - r1q8_c0,
                (kld.sum_kld - r1q8_s0) / (double) (kld.count - r1q8_c0), (double) (kld.n_same_top - r1q8_t0) / (double) (kld.count - r1q8_c0));''')
print("ok")
