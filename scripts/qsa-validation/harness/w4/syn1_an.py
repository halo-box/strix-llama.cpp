#!/usr/bin/env python3
# PLAN-v3 §14.2 (b): the decode n_q 1 SYN 262k fixture enlarged to 12 layers x 3 seeds. Per variant: NMSE vs O_CON per
# (seed, layer) (BAND ... gpu_vs_contract), p100 over the set, the re-set row = 2 x p100 of the PR-verbatim decode
# (PR-decM-f16 / PR-decM-q8_0: partial + qsa_decode_merge), the new G1 kernel's max, the planted-defect Dmin and
# k = Dmin / p100; the old row (R1-A, 1.15e-7) and seed 3 of R2-BOP (2 x 5.85e-8) for comparison.
# usage: syn1_an.py <files...>
import sys, re, collections
v = collections.defaultdict(list); dm = []; det = []
for f in sys.argv[1:]:
    for l in open(f):
        m = re.match(r"BAND (\S+) L(\d+) (\S+)\s+pair \S+ gpu_vs_contract rows \d+ nmse (\S+)", l)
        if m: v[m.group(3)].append((float(m.group(4)), m.group(1), int(m.group(2))))
        m = re.match(r"DEFECT \S+ L\d+ pair \S+ d\d cells_dropped \d+ rows \d+ nmse (\S+)", l)
        if m and float(m.group(1)) > 0: dm.append(float(m.group(1)))
        m = re.match(r"DETERMINISM \S+ L\d+ (\S+)\s+words_differ_run1_run2 (\d+)", l)
        if m: det.append(int(m.group(2)))
pr = max(x for k in v if k.startswith("PR-decM") for x in v[k])
print("samples per variant:", {k: len(x) for k, x in v.items()})
for k in sorted(v):
    x = max(v[k]); print("VARIANT %-14s p100 %.4e (%s L%d) mean %.4e" % (k, x[0], x[1], x[2], sum(y[0] for y in v[k]) / len(v[k])))
row = 2 * pr[0]
print("PR_p100 %.4e (%s %s L%d) -> re-set row 2 x p100 = %.4e (old row 1.15e-7)" % (pr[0], [k for k in v if pr in v[k]][0], pr[1], pr[2], row))
new = max(x for k in v if k.startswith("G1M") for x in v[k])
print("NEW_G1_max %.4e (%s L%d) <= row: %s ; ratio %.3f" % (new[0], new[1], new[2], new[0] <= row, new[0] / row))
print("DEFECT Dmin %.4e k = Dmin / PR_p100 %.1f ; Dmin / row %.1f" % (min(dm), min(dm) / pr[0], min(dm) / row))
print("DETERMINISM all run pairs bitwise equal:", all(d == 0 for d in det), "(%d pairs)" % len(det))
