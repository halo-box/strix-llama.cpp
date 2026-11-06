#!/usr/bin/env python3
# W3 kernel-time summary: res/*.out of w3_attn (SUM = mean over layers of the per-layer medians, ms per call)
import re, sys, os, glob, collections
R = sys.argv[1]
def sums(f):
    d = {}
    for l in open(f):
        m = re.match(r"SUM (\S+)\s+(\S+)", l)
        if m: d[m.group(1)] = float(m.group(2))
    return d
def chk(f):
    out = []
    for l in open(f):
        if " CHK " in l: out.append(l.strip())
    return out
T = ["f16", "bf16", "q8_0", "q4_0"]
target = {"wiki4k": {"f16": 1.0, "q8_0": 1.033, "q4_0": 1.017, "bf16": 1.018}, "wiki131k": {"f16": 1.0, "q8_0": 1.017, "q4_0": 0.986, "bf16": 1.047}}
print("## C18 rows (SYN data), mean of r1 / r2; ms per ub4096 call, mean of 12 per-layer medians\n")
allpass = True
for fx in ["wiki4k", "wiki131k"]:
    runs = [sums(f) for f in sorted(glob.glob(os.path.join(R, fx + "_r*.out")))]
    if not runs: continue
    m = {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}
    pr = m["PR-V0-f16"] + m["PACKV-f16"]; c18 = m["C18-Kf16D-Vf16P"] + m["PACKV-f16"]
    print("### %s (runs %d): PR-verbatim f16 route PR-V0-f16 + PACKV-f16 = %.4f ms; C18 Kf16D-Vf16P + PACKV-f16 = %.4f ms\n" % (fx, len(runs), pr, c18))
    print("| type | X+P (attend+fixup + pack) | X+X direct | best | best / PR route | gate (target x 1.04) | pass | best / C18 route | per-run best / PR |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t in T:
        xp = m["NEW-%s-P" % t] + m["NEWPACK-%s" % t]; xx = m["NEW-%s-D" % t]; b = min(xp, xx)
        gate = target[fx][t] * 1.04; ok = b / pr <= gate; allpass &= ok
        per = ["%.3f" % (min(r["NEW-%s-P" % t] + r["NEWPACK-%s" % t], r["NEW-%s-D" % t]) / (r["PR-V0-f16"] + r["PACKV-f16"])) for r in runs]
        print("| %s | %.4f | %.4f | %s | %.3f | %.3f | %s | %.3f | %s |" % (t, xp, xx, "X+P" if xp <= xx else "X+X", b / pr, gate, "yes" if ok else "NO", b / c18, " / ".join(per)))
    print()
print("C18 KERNEL-TIME GATE:", "PASS" if allpass else "FAIL")
print("\n## V-loader crossover (SYN plans, C6/C18 generator U 1.8; packed = attend + pack, direct = attend; ms)\n")
print("| n_q | n_kv | 64*n_q | rule | " + " | ".join("%s P / D / D/P" % t for t in T) + " |")
print("|---|---|---|---|" + "---|" * len(T))
for f in sorted(glob.glob(os.path.join(R, "syn_*.out")), key=lambda x: tuple(int(v) for v in re.findall(r"syn_(\d+)_(\d+)", x)[0])):
    nq, nk = (int(v) for v in re.findall(r"syn_(\d+)_(\d+)", f)[0]); m = sums(f)
    if not m: continue
    cells = []
    for t in T:
        p = m["NEW-%s-P" % t] + m["NEWPACK-%s" % t]; d = m["NEW-%s-D" % t]
        cells.append("%.2f / %.2f / %.3f" % (p, d, d / p))
    print("| %d | %d | %d | %s | %s |" % (nq, nk, 64 * nq, "packed" if nk < 64 * nq else "direct", " | ".join(cells)))
print("\n## correctness lines (new kernel vs PR-V0-f16 output; P vs D of one type)\n")
for f in sorted(glob.glob(os.path.join(R, "*.out"))):
    c = chk(f)
    mx = collections.defaultdict(float); nd = collections.defaultdict(int)
    for l in c:
        mm = re.search(r"CHK (\S+)\s+vs_PR-V0-f16 words_differ (\d+) nmse (\S+) nan (\d+) \| vs_same_type_first words_differ (\d+)", l)
        if mm:
            n = mm.group(1); mx[n] = max(mx[n], float(mm.group(3))); nd[n] += int(mm.group(5)) + int(mm.group(4)) * 10**9
    print(os.path.basename(f), " ".join("%s:nmse_max=%.2e,PvsD_words+nan=%d" % (k, mx[k], nd[k]) for k in sorted(mx)))
