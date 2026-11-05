# C10 analysis: per cell (depth, type, ub, pp, workload) PR/base ratio of mean t/s with a hierarchical bootstrap
# (resample processes within each build, then reps within each resampled process), 20000 draws, seed 1.
# Also: text/random ratio per build, base drift (first vs last base process), outlier samples (< 0.6 x cell median).
import json, glob, os, sys, re, random, statistics as st
L = sys.argv[1]; DROP0 = len(sys.argv) > 2 and sys.argv[2] == 'drop0'
cells = {}  # key -> build -> list of per-process sample lists
for f in sorted(glob.glob(os.path.join(L, 'd*_*.json'))):
    m = re.match(r'd(\d+)_(f16|q8_0)_(\d)_(base|pr)_(text|rand)\.json', os.path.basename(f))
    if not m: continue
    d, t, k, b, wl = m.groups()
    try: rows = json.load(open(f))
    except Exception as e: print('BAD', f, e); continue
    for r in rows:
        s = r['samples_ts'][1:] if DROP0 else r['samples_ts']
        cells.setdefault((int(d), t, r['n_ubatch'], r['n_prompt'], wl), {}).setdefault(b, []).append((int(k), s))
rng = random.Random(1)
def hboot(procs, n):
    out = []
    for _ in range(n):
        ps = [procs[rng.randrange(len(procs))] for _ in procs]
        xs = []
        for p in ps: xs += [p[rng.randrange(len(p))] for _ in p]
        out.append(sum(xs) / len(xs))
    return out
def mean_all(procs): xs = [x for p in procs for x in p]; return sum(xs) / len(xs)
def med_all(procs): return st.median([x for p in procs for x in p])
def hboot_med(procs, n):
    out = []
    for _ in range(n):
        ps = [procs[rng.randrange(len(procs))] for _ in procs]
        xs = []
        for p in ps: xs += [p[rng.randrange(len(p))] for _ in p]
        out.append(st.median(xs))
    return out
NB = 20000
res = {}
print('depth type ub pp wl | base mean (cv%, nproc x nrep) | pr mean (cv%) | PR/base of means [95% CI] | medians base/pr, PR/base of medians [95% CI] | base drift last/first proc mean | outliers (<0.6x cell median)')
for key in sorted(cells):
    c = cells[key]
    if 'base' not in c or 'pr' not in c: continue
    B = [s for _, s in sorted(c['base'])]; P = [s for _, s in sorted(c['pr'])]
    mb, mp = mean_all(B), mean_all(P)
    bb, bp = hboot(B, NB), hboot(P, NB)
    rr = sorted(p / b for p, b in zip(bp, bb))
    lo, hi = rr[int(0.025 * NB)], rr[int(0.975 * NB)]
    mdb, mdp = med_all(B), med_all(P)
    qb, qp = hboot_med(B, 4000), hboot_med(P, 4000)
    rq = sorted(p / b for p, b in zip(qp, qb)); mlo, mhi = rq[100], rq[3899]
    allx = [x for p in B + P for x in p]; med = st.median(allx)
    outl = [(b, k, round(x)) for b in ('base', 'pr') for k, s in sorted(c[b]) for x in s if x < 0.6 * med]
    cvb = 100 * st.pstdev([x for p in B for x in p]) / mb; cvp = 100 * st.pstdev([x for p in P for x in p]) / mp
    drift = (sum(B[-1]) / len(B[-1])) / (sum(B[0]) / len(B[0]))
    res[key] = (mdb, mdp, mdp / mdb, lo, hi)
    d, t, ub, pp, wl = key
    print(f'd{d} {t} ub{ub} pp{pp} {wl} | {mb:8.1f} ({cvb:4.1f}%, {len(B)}x{len(B[0])}) | {mp:8.1f} ({cvp:4.1f}%, {len(P)}x{len(P[0])}) | '
          f'{mp/mb:6.4f} [{lo:6.4f}, {hi:6.4f}] {100*(mp/mb-1):+6.2f}% | med {mdb:7.1f}/{mdp:7.1f} {100*(mdp/mdb-1):+6.2f}% [{100*(mlo-1):+.2f}, {100*(mhi-1):+.2f}] | {drift:6.4f} | {outl if outl else "-"}')
print()
print('text / random (median t/s ratio) per build:')
for key in sorted(res):
    if key[4] != 'text': continue
    k2 = key[:4] + ('rand',)
    if k2 not in res: continue
    d, t, ub, pp, _ = key
    print(f'd{d} {t} ub{ub} pp{pp}: base {res[key][0]/res[k2][0]:6.4f}  pr {res[key][1]/res[k2][1]:6.4f}  | PR/base (medians) text {100*(res[key][2]-1):+6.2f}% vs rand {100*(res[k2][2]-1):+6.2f}%')
print()
print('q8_0 / f16 (median t/s ratio):')
for key in sorted(res):
    if key[1] != 'q8_0': continue
    k2 = (key[0], 'f16') + key[2:]
    if k2 not in res: continue
    d, t, ub, pp, wl = key
    print(f'd{d} ub{ub} pp{pp} {wl}: base {res[key][0]/res[k2][0]:6.4f}  pr {res[key][1]/res[k2][1]:6.4f}')
