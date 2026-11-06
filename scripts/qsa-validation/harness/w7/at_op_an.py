#!/usr/bin/env python3
"""AT-W7PERF op level analysis (PLAN-v3 §16.5): per (pair, n_q, n_kv): mean over the 3 processes of each arm's mean-of-12-medians, the
run-to-run noise (p90 over all cells of |p_i - mean| / mean), the AT-A rule for the shortcut G (G4 where it beats G1 by > 3 x p90, else G1)
and the 0.98 rule for prefer_dense at the chosen G (sparse rate >= 0.98 x dense rate).
usage: at_op_an.py DIR"""
import re, sys, glob, collections
d = sys.argv[1]
A = collections.defaultdict(dict)
for f in sorted(glob.glob(d + '/p?.out')):
    for line in open(f):
        m = re.match(r'R (\S+) \S+ (\S+) (\S+) n_q=(\d+) n_kv=(\d+) arm=(\S+) ms=(\S+)', line)
        if m:
            p, k, v, nq, nk, arm, ms = m.groups()
            A[(k, v, int(nq), int(nk), arm)][p] = float(ms)
noise = sorted(abs(x - sum(c.values()) / len(c)) / (sum(c.values()) / len(c)) for c in A.values() for x in c.values())
p90 = noise[int(0.9 * len(noise))] if noise else 0
print(f'noise p90 = {p90:.4f} over {len(noise)} samples; G4 margin = {3 * p90:.4f}')
print('| K | V | n_q | n_kv | G1 ms | G4 ms | dense ms | G | sparse/dense rate | flip |')
print('|---|---|---|---|---|---|---|---|---|---|')
mean = lambda c: sum(c.values()) / len(c)
bands = collections.defaultdict(list)
def band(n): return '1' if n <= 1 else '2-8' if n <= 8 else '9-32' if n <= 32 else '33-127' if n <= 127 else '128-511' if n <= 511 else '512-1023' if n <= 1023 else '>=1024'
for key in sorted({k[:4] for k in A}):
    k, v, nq, nk = key
    g1, g4, de = [mean(A[key + (a,)]) if (key + (a,)) in A else None for a in ('G1', 'G4', 'dense')]
    if g1 is None or de is None: continue
    useg4 = g4 is not None and g1 / g4 > 1 + 3 * p90
    t = g4 if useg4 else g1
    r = de / t
    flip = r < 0.98
    bands[(k, v, band(nq))].append((nq, nk, 4 if useg4 else 1, r, flip))
    print(f'| {k} | {v} | {nq} | {nk} | {g1:.3f} | {g4 if g4 else float("nan"):.3f} | {de:.3f} | {4 if useg4 else 1} | {r:.3f} | {"DENSE" if flip else ""} |')
print()
print('| K | V | n_q band | G | worst sparse/dense | flip cells | prefer_dense |')
print('|---|---|---|---|---|---|---|')
for b in sorted(bands, key=lambda x: (x[0], x[1], ['1','2-8','9-32','33-127','128-511','512-1023','>=1024'].index(x[2]))):
    c = bands[b]
    print(f'| {b[0]} | {b[1]} | {b[2]} | {"/".join(sorted({str(x[2]) for x in c}))} | {min(x[3] for x in c):.3f} | {sum(x[4] for x in c)}/{len(c)} | {"dense" if any(x[4] for x in c) else "sparse"} |')
