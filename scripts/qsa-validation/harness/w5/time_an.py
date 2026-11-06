#!/usr/bin/env python3
# W5: summary of ~/dev/wt_lab/w5/time/res/*.out (w5_time). Per (rows, arm, type): the mean attend ms over the arm's layers per
# process (r1, r2), and on the SAME layers its ratio to f16 of the default arm and to f16 of the same arm. usage: time_an.py res_dir
import glob, os, re, sys, collections
L = collections.defaultdict(dict)   # (rows, arm, r, type) -> {layer: attend}
for fn in sorted(glob.glob(os.path.join(sys.argv[1], "*.out"))):
    m = re.match(r"(wiki\w+?)_(default|generic|vdirect|portable)_(r\d)\.out", os.path.basename(fn))
    if not m: continue
    rows, arm, r = m.groups()
    for line in open(fn, errors="replace"):
        s = re.match(r"L(\d+) RES (\S+)\s+n_q \d+ nk \d+ plan_med \S+ graph_med \S+ attend (\S+) ms", line)
        if s: L[(rows, arm, r, s.group(2))][int(s.group(1))] = float(s.group(3))
def mean(k, ls=None):
    d = L.get(k, {}); ls = sorted(d) if ls is None else ls
    return sum(d[x] for x in ls) / len(ls)
ARMS = ["default", "generic", "vdirect", "portable"]; TY = "f16 bf16 q8_0 q4_0 f32 q4_1 q5_0 q5_1 iq4_nl".split()
print("| rows | arm | type | layers | attend ms r1 | attend ms r2 | r2/r1 | / default-arm f16 same layers (r1, r2) | / same-arm f16 (r1, r2) |")
print("|---|---|---|---|---|---|---|---|---|")
for rows in sorted({k[0] for k in L}):
    for arm in ARMS:
        for t in TY:
            if (rows, arm, "r1", t) not in L: continue
            ls = sorted(L[(rows, arm, "r1", t)])
            a1, a2 = mean((rows, arm, "r1", t)), mean((rows, arm, "r2", t))
            d1, d2 = mean((rows, "default", "r1", "f16"), ls), mean((rows, "default", "r2", "f16"), ls)
            s1, s2 = mean((rows, arm, "r1", "f16"), ls), mean((rows, arm, "r2", "f16"), ls)
            print("| %s | %s | %s | %d | %.3f | %.3f | %.3f | %.3f, %.3f | %.3f, %.3f |" % (rows, arm, t, len(ls), a1, a2, a2 / a1, a1 / d1, a2 / d2, a1 / s1, a2 / s2))
