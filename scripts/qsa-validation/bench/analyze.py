#!/usr/bin/env python3
"""§16 performance statistic for llama-bench JSON (PLAN-v3 §16; C10 analyze.py changed to the mean of reps).

Per process: rep 0 dropped, the mean of reps 1..n. Ratio new/R per cell with a 95 % CI from the hierarchical
bootstrap (resample processes within each build, then reps within each resampled process; 4000 draws, seed 1),
the statistic of `EV/bs_skeptic.py` with the mean. Verdict against a threshold theta (§16 stopping rule):
PASS if CI LB >= theta, FAIL if CI UB < theta, else RERUN (2x processes, then 4x; still unresolved -> FAIL).

usage:
  analyze.py ratio  --ref 'GLOB' --new 'GLOB' [--theta T]
  analyze.py rr     --ref-a 'GLOB' --new-a 'GLOB' --ref-b 'GLOB' --new-b 'GLOB' [--theta T]
      ratio of ratios (P3): (new_a/new_b) / (ref_a/ref_b), the four arms bootstrapped jointly per draw
  analyze.py c10    LOGDIR      C10 file naming d<depth>_<type>_<proc>_<base|pr>_<text|rand>.json, pr vs base
Each JSON is one llama-bench -o json output (one process); cells are keyed by (n_depth, type_k, type_v, n_ubatch,
n_prompt, n_gen). Samples with an eviction flag are not handled here: P6 excludes them upstream (C7 sampler).
"""
import argparse, glob, json, os, random, re, sys

DRAWS, SEED = 4000, 1

def cell_of(r):
    return (r.get('n_depth', 0), r['type_k'], r['type_v'], r['n_ubatch'], r['n_prompt'], r['n_gen'])

def load(pattern):
    cells = {}
    for f in sorted(glob.glob(pattern)):
        try: rows = json.load(open(f))
        except Exception as e: print(f'BAD {f}: {e}', file=sys.stderr); continue
        for r in rows:
            s = r['samples_ts'][1:]
            if s: cells.setdefault(cell_of(r), []).append(s)
    return cells

def point(procs):
    return sum(sum(p) / len(p) for p in procs) / len(procs)

def draw(procs, rng):
    tot = 0.0
    for _ in procs:
        p = procs[rng.randrange(len(procs))]
        tot += sum(p[rng.randrange(len(p))] for _ in p) / len(p)
    return tot / len(procs)

def verdict(lo, hi, theta):
    if theta is None: return ''
    return 'PASS' if lo >= theta else ('FAIL' if hi < theta else 'RERUN')

def ci(xs):
    xs = sorted(xs); return xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]

def ratio(R, N, theta, label='new/ref'):
    rng = random.Random(SEED)
    print(f'cell (depth, K, V, ub, pp, tg) | ref mean (nproc x nrep) | new mean | {label} [95% CI] | verdict')
    for c in sorted(set(R) & set(N)):
        r = [draw(R[c], rng) for _ in range(DRAWS)]; n = [draw(N[c], rng) for _ in range(DRAWS)]
        lo, hi = ci([b / a for a, b in zip(r, n)])
        pr, pn = point(R[c]), point(N[c])
        print(f'{c} | {pr:9.2f} ({len(R[c])}x{len(R[c][0])}) | {pn:9.2f} ({len(N[c])}x{len(N[c][0])}) | {pn/pr:.4f} [{lo:.4f}, {hi:.4f}] | {verdict(lo, hi, theta)}')

def rr(Ra, Na, Rb, Nb, theta):
    rng = random.Random(SEED)
    print('cell_a | cell_b | (new_a/new_b)/(ref_a/ref_b) [95% CI] | new_a/new_b | verdict')
    for ca in sorted(set(Ra) & set(Na)):
        cb = next((c for c in sorted(set(Rb) & set(Nb)) if c[0] == ca[0] and c[3:] == ca[3:]), None)
        if cb is None: continue
        xs = []
        for _ in range(DRAWS):
            ra, na, rb, nb = draw(Ra[ca], rng), draw(Na[ca], rng), draw(Rb[cb], rng), draw(Nb[cb], rng)
            xs.append((na / nb) / (ra / rb))
        lo, hi = ci(xs)
        pt = (point(Na[ca]) / point(Nb[cb])) / (point(Ra[ca]) / point(Rb[cb]))
        print(f'{ca} | {cb} | {pt:.4f} [{lo:.4f}, {hi:.4f}] | {point(Na[ca]) / point(Nb[cb]):.4f} | {verdict(lo, hi, theta)}')

def c10(L):
    sets = {}
    for f in sorted(glob.glob(os.path.join(L, 'd*_*.json'))):
        m = re.match(r'd(\d+)_(\w+?)_(\d+)_(base|pr)_(text|rand)\.json', os.path.basename(f))
        if not m: continue
        sets.setdefault((m.group(4), m.group(5)), []).append(f)
    for wl in ('text', 'rand'):
        if ('base', wl) not in sets or ('pr', wl) not in sets: continue
        print(f'## {wl}')
        R, N = {}, {}
        for f in sets[('base', wl)]:
            for k, v in load(f).items(): R.setdefault(k, []).extend(v)
        for f in sets[('pr', wl)]:
            for k, v in load(f).items(): N.setdefault(k, []).extend(v)
        ratio(R, N, None, 'pr/base')

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['ratio', 'rr', 'c10'])
    ap.add_argument('logdir', nargs='?')
    for k in ('ref', 'new', 'ref-a', 'new-a', 'ref-b', 'new-b'): ap.add_argument('--' + k)
    ap.add_argument('--theta', type=float)
    a = ap.parse_args()
    if a.mode == 'c10': c10(a.logdir); return
    if a.mode == 'ratio': ratio(load(a.ref), load(a.new), a.theta); return
    rr(load(a.ref_a), load(a.new_a), load(a.ref_b), load(a.new_b), a.theta)

if __name__ == '__main__':
    main()
