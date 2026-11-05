# r1e_patch.py FILE : add per-chunk QSA agreement lines (QPIN_QSA_CHUNK) to QPIN_STATS. Measurement only; inert unless QPIN is set.
import sys
p = sys.argv[1]; s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)
rep("    std::map<int, qpin_acc> moe;\n",
    "    std::map<int, qpin_acc> moe;\n"
    "    std::map<std::pair<int,int>, qpin_acc> qsa_chunk; // (chunk, src: 1 cb, 2 tk first fire, 3 tk second fire)\n"
    "    std::map<int, int> tk_fire; // per layer, reset per ubatch\n")
rep("    kind == 1 ? S.n_cb++ : S.n_tk++;\n",
    "    kind == 1 ? S.n_cb++ : S.n_tk++;\n"
    "    const int src = kind == 1 ? 1 : (++S.tk_fire[il] == 1 ? 2 : 3);\n")
rep("            qpin_add(S.qsa[{-1, -1}], ov, jc, ex);\n",
    "            qpin_add(S.qsa[{-1, -1}], ov, jc, ex);\n"
    "            qpin_add(S.qsa_chunk[{S.chunk, src}], ov, jc, ex);\n")
rep("    g_qpin.pos.clear();\n",
    "    g_qpin.pos.clear();\n    g_qpin.tk_fire.clear();\n")
rep("    for (const auto & [k, a] : S.moe) {\n",
    "    for (const auto & [k, a] : S.qsa_chunk) {\n"
    "        out(\"QPIN_QSA_CHUNK chunk=%d src=%s n=%lld overlap=%.9f jaccard=%.9f exact=%.9f\\n\", k.first,\n"
    "            k.second == 1 ? \"cb\" : (k.second == 2 ? \"tk1\" : \"tk2\"), (long long) a.n, a.overlap / std::max<int64_t>(a.n, 1),\n"
    "            a.jacc / std::max<int64_t>(a.n, 1), (double) a.exact / std::max<int64_t>(a.n, 1));\n"
    "    }\n"
    "    for (const auto & [k, a] : S.moe) {\n")
open(p, "w").write(s)
print("patched", p)
