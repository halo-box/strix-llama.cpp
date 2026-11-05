# R2-M-LAB-BANDS-OP: patch ~/dev/wt_lab/bands/tree (lab worktree at 94e086847): input/output dump of the GQA-12 maskless
# cases (BANDS_DUMP=<dir>) and a route print of the dispatched QSA kernel (BANDS_ROUTE=1). Idempotent.
import re, sys, os
T = os.path.expanduser('~/dev/wt_lab/bands/tree')
def patch(path, anchor, add, after=True):
    p = os.path.join(T, path); s = open(p).read()
    if add.strip() in s: print('already', path); return
    i = s.find(anchor); assert i >= 0 and s.find(anchor, i + 1) < 0, (path, anchor)
    j = i + len(anchor) if after else i
    s = s[:j] + add + s[j:]; open(p, 'w').write(s); print('patched', path)
patch('ggml/src/ggml-cuda/qsa-decode.cu', '    const int splits = (ns + chunk - 1)/chunk;\n',
      '    if (getenv("BANDS_ROUTE")) fprintf(stderr, "QSA_ROUTE decode G1 wmma %d nq %d ns %d nk %d splits %d chunk %d kt %s vt %s\\n", (int) wmma, nq, ns, (int) k->ne[1], splits, chunk, ggml_type_name(k->type), ggml_type_name(v->type));\n')
patch('ggml/src/ggml-cuda/qsa-prefill.cu', '    const bool qsa_kdirect = kv_direct_ok && (qsa_direct || qsa_direct_cfg == 2 || qsa_direct_cfg < 0);\n',
      '    if (getenv("BANDS_ROUTE")) fprintf(stderr, "QSA_ROUTE prefill G4 direct %d kdirect %d nq %d ns %d nk %d kt %s vt %s\\n", (int) qsa_direct, (int) qsa_kdirect, n_q, ns, nk, ggml_type_name(k->type), ggml_type_name(v->type));\n')
dump = ('        if (const char * dd = getenv("BANDS_DUMP")) if (ratio == 12 && streams == 1) { static int seq = 0; char fn[512]; snprintf(fn, sizeof fn, "%s/%s_%03d.bdmp", dd, op_desc(nullptr).c_str(), seq++); FILE * f = fopen(fn, "wb"); GGML_ASSERT(f); '
        'const int32_t h[8] = {queries, keys, selected, ratio, (int32_t) interleaved, blocks, (int32_t) type_k, (int32_t) type_v}; fwrite(h, 4, 8, f); '
        'auto rows = [&](ggml_tensor * t, int64_t n1, int64_t n2) { ggml_tensor * b = t->view_src ? t->view_src : t; std::vector<uint8_t> B(ggml_nbytes(b)); ggml_backend_tensor_get(b, B.data(), 0, B.size()); const size_t rb = ggml_row_size(t->type, t->ne[0]); '
        'for (int64_t i1 = 0; i1 < n1; ++i1) for (int64_t i2 = 0; i2 < n2; ++i2) fwrite(B.data() + t->view_offs + i1 * t->nb[1] + i2 * t->nb[2], 1, rb, f); }; '
        'rows(q, queries, 2*ratio); rows(k, keys, 2); rows(v, keys, 2); fwrite(picks.data(), 4, picks.size(), f); fwrite(actual, 4, n, f); fwrite(cpu, 4, n, f); fclose(f); '
        'fprintf(stderr, "BANDS_DUMP %s %s\\n", fn, vars().c_str()); }\n')
patch('tests/test-backend-ops.cpp', '        fprintf(stderr,"QSA_MASKLESS_FP64 q=%d keys=%d selected=%d gpu=%.9g cpu=%.9g\\n",queries,keys,selected,gpu_error,cpu_error);\n', dump)
