#!/usr/bin/env python3
"""AT-W7PERF e2e (a) analysis: per (pair, ub, pp) and (pair, tg, depth): sparse / dense with the §16 statistic (mean of reps 1..n per process,
hierarchical bootstrap 4000 draws seed 1), against theta = 0.98. usage: at_e2e_an.py DIR"""
import glob, json, re, sys, random, collections
d = sys.argv[1]
DRAWS, SEED = 4000, 1
def load(f):
    try: return json.load(open(f))
    except Exception as e: print('BAD', f, e, file=sys.stderr); return []
A = collections.defaultdict(lambda: {'dense': [], 'sparse': []})
for f in sorted(glob.glob(d + '/*.json')):
    m = re.match(r'(.+)_(\d+)_(dense|sparse)_(pp|tg)\.json', f.split('/')[-1])
    if not m: continue
    pair, _, arm, kind = m.groups()
    for r in load(f):
        A[(pair, kind, r.get('n_depth', 0), r['n_ubatch'], r['n_prompt'], r['n_gen'])][arm].append(r['samples_ts'][1:])
def pt(ps): return sum(sum(p)/len(p) for p in ps)/len(ps)
def draw(ps, rng):
    t = 0.0
    for _ in ps:
        p = ps[rng.randrange(len(ps))]
        t += sum(p[rng.randrange(len(p))] for _ in p)/len(p)
    return t/len(ps)
def ci(R, N):
    rng = random.Random(SEED)
    xs = sorted(draw(N, rng)/draw(R, rng) for _ in range(DRAWS))
    return xs[int(.025*DRAWS)], xs[int(.975*DRAWS)-1]
print('| pair | kind | depth | ub | pp | tg | dense t/s | sparse t/s | ratio | 95% CI | verdict (0.98) |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for c in sorted(A):
    a = A[c]
    if not a['dense'] or not a['sparse']: continue
    lo, hi = ci(a['dense'], a['sparse'])
    r = pt(a['sparse'])/pt(a['dense'])
    v = 'PASS' if lo >= .98 else 'FAIL' if hi < .98 else 'RERUN'
    print(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} | {c[5]} | {pt(a['dense']):.2f} | {pt(a['sparse']):.2f} | {r:.4f} | [{lo:.4f}, {hi:.4f}] | {v} |")
