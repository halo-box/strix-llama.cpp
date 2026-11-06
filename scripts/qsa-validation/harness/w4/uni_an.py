#!/usr/bin/env python3
# W4 U34: R2-BOP-style summary of bands_dec / w4_dec band outputs: per case (n_q, n_sel, n_kv, splits) the p100 of the
# GPU-vs-O_CON NMSE over seeds and types per variant family, 2 x p100, and the planted-defect detection margin
# k = min(d1, d2, d3 NMSE) / p100 (R2-BOP definition). Class = split class of the PR rule.
# usage: uni_an.py <dir> [variant-prefix (default PR-dec)]
import sys, re, glob, collections
d = sys.argv[1]; pref = sys.argv[2] if len(sys.argv) > 2 else "PR-dec"
def cls(s):
    return "1" if s <= 1 else "2-5" if s <= 5 else "6-16" if s <= 16 else "17-33" if s <= 33 else "34+"
cases = collections.defaultdict(lambda: dict(nmse=[], dmin=[], files=[]))
for f in sorted(glob.glob(d + "/*.out")):
    t = open(f).read()
    m = re.search(r"^CASE \S+ (?:L\d+ )?n_q (\d+) ns (\d+) nk (\d+) (?:blocks \S+ )?(?:pr_)?splits (\d+)", t, re.M)
    if not m: continue
    nq, ns, nk, sp = map(int, m.groups())
    key = (cls(sp), nq, ns, nk, sp)
    for mm in re.finditer(r"^BANDSUM \S+ (" + re.escape(pref) + r"\S*) (\S+) gpu_vs_contract nmse_layer_max (\S+)", t, re.M):
        cases[key]["nmse"].append((float(mm.group(3)), mm.group(1), f.split("/")[-1]))
    for mm in re.finditer(r"^DEFECT \S+ L\d+ pair (\S+) d(\d) cells_dropped \d+ rows \d+ nmse (\S+)", t, re.M):
        cases[key]["dmin"].append(float(mm.group(3)))
    cases[key]["files"].append(f)
byc = collections.defaultdict(list)
print("| class | splits | n_q | n_sel | n_kv | samples | NMSE p100 (" + pref + ") | 2 x p100 | defect Dmin | k = Dmin / p100 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for key in sorted(cases, key=lambda k: (k[0], k[4], k[1])):
    c = cases[key]
    if not c["nmse"]: continue
    p = max(x[0] for x in c["nmse"]); dm = min(c["dmin"]) if c["dmin"] else float("nan")
    print("| %s | %d | %d | %d | %d | %d | %.3e | %.3e | %.3e | %.1f |" % (key[0], key[4], key[1], key[2], key[3], len(c["nmse"]), p, 2 * p, dm, dm / p))
    byc[(key[0], "big" if key[2] > 257 else "small")].append((p, key, dm))
print()
for k in sorted(byc):
    p, key, _ = max(byc[k]); dm = min(x[2] for x in byc[k])
    print("CLASS splits %s n_sel %s: p100 %.4e (set by n_q %d n_sel %d n_kv %d splits %d) -> row 2 x p100 = %.4e; min Dmin %.4e, min k %.1f" %
          (k[0], k[1], p, key[1], key[2], key[3], key[4], 2 * p, dm, dm / p))
