#!/usr/bin/env python3
"""W7 perf verdicts (PLAN-v3 §16): the §16 statistic (per process the mean of reps 1..n; ratio CI from the hierarchical bootstrap, 4000 draws, seed 1), keyed by (depth, ub, pp, tg) so
cross-type ratios work. A sample below 0.9 x its cell median is an eviction sample (dropped, counted) when its process's C7 sampler saw KFD evictions after the first 60 s; otherwise it is a P6 (a) failure.
usage: perf_an.py DIR PHASE [PHASE...]   phases: P1 P2 P3 P4 P5 P6 ND NDB NPIN D2B D1V B1"""
import glob, json, os, random, re, sys, collections
DRAWS, SEED = 4000, 1
d = sys.argv[1]
def evicted(tag):
    """the KFD evicted_ms the process accumulated after its first 60 s (model load and depth fill evict by design)"""
    p = f'{d}/{tag}.samp'
    if not os.path.exists(p): return 0.0
    t0 = first = last = None
    for i, l in enumerate(open(p)):
        if i == 0: continue
        x = l.split()
        if len(x) < 2 or x[1] == '-1': continue
        t = int(x[0]) / 1e9
        if t0 is None: t0 = t
        if t - t0 >= 60 and first is None: first = float(x[1])
        last = float(x[1])
    return (last - first) if first is not None else 0.0
def key(r): return (r.get('n_depth', 0), r['n_ubatch'], r['n_prompt'], r['n_gen'])
EV = collections.Counter()
EVMS = {}
P6A = []
def load(pat):
    cells = collections.defaultdict(list)
    for f in sorted(glob.glob(f'{d}/{pat}.json')):
        tag = os.path.basename(f)[:-5]
        try: rows = json.load(open(f))
        except Exception: continue
        ev = evicted(tag)
        EVMS[tag] = ev
        for r in rows:
            s = r['samples_ts'][1:]
            if not s: continue
            m = sorted(s)[len(s)//2]
            # §16 / P6: the bench JSON has no per-rep time stamps, so a sample below 0.9 x its cell median counts as an eviction sample when its
            # process saw evictions (dropped from the statistic and counted), and as a P6 (a) failure when it did not
            low = [x for x in s if x < 0.9*m]
            if low and ev > 0:
                EV[tag] += len(low)
                s = [x for x in s if x >= 0.9*m]
            elif low:
                P6A.extend((tag, key(r), x, m) for x in low)
            cells[key(r)].append(s)
    return cells
def pt(ps): return sum(sum(p)/len(p) for p in ps)/len(ps)
def dr(ps, rng):
    t = 0.0
    for _ in ps:
        p = ps[rng.randrange(len(ps))]
        t += sum(p[rng.randrange(len(p))] for _ in p)/len(p)
    return t/len(ps)
def ci(xs):
    xs = sorted(xs); return xs[int(.025*len(xs))], xs[int(.975*len(xs))-1]
def verdict(lo, hi, th): return 'PASS' if lo >= th else 'FAIL' if hi < th else 'RERUN'
RES = collections.Counter()
def ratio(name, R, N, th, filt=None):
    print(f'\n### {name} (theta {th})\n\n| depth | ub | pp | tg | ref t/s (procs) | new t/s (procs) | ratio | 95% CI | verdict |\n|---|---|---|---|---|---|---|---|---|')
    rng = random.Random(SEED)
    for c in sorted(set(R) & set(N)):
        if filt and not filt(c): continue
        lo, hi = ci([dr(N[c], rng)/dr(R[c], rng) for _ in range(DRAWS)])
        v = verdict(lo, hi, th); RES[v] += 1
        print(f'| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {pt(R[c]):.2f} ({len(R[c])}) | {pt(N[c]):.2f} ({len(N[c])}) | {pt(N[c])/pt(R[c]):.4f} | [{lo:.4f}, {hi:.4f}] | {v} |')
def rr(name, Ra, Na, Rb, Nb, th):
    print(f'\n### {name} (theta {th})\n\n| depth | ub | pp | tg | (new_a/new_b)/(ref_a/ref_b) | 95% CI | new_a/new_b | verdict |\n|---|---|---|---|---|---|---|---|')
    rng = random.Random(SEED)
    for c in sorted(set(Ra) & set(Na) & set(Rb) & set(Nb)):
        lo, hi = ci([(dr(Na[c], rng)/dr(Nb[c], rng))/(dr(Ra[c], rng)/dr(Rb[c], rng)) for _ in range(DRAWS)])
        q = pt(Na[c])/pt(Nb[c]); v = verdict(lo, hi, th); RES[v] += 1
        print(f'| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {q/(pt(Ra[c])/pt(Rb[c])):.4f} | [{lo:.4f}, {hi:.4f}] | {q:.4f} {"" if q >= .90 else "(below the 0.90 floor)"} | {v} |')
for ph in sys.argv[2:]:
    if ph == 'P1':
        ratio('P1 text q8_0 ub512 pp4096 d0, new vs base 13a279bfa', load('p1_*_base'), load('p1_*_new'), .995)
    elif ph == 'P2':
        ratio('P2 text f16, new vs W0', load('p23_*_f16_*_w0'), load('p23_*_f16_*_new'), .99)
    elif ph == 'P3':
        Rf, Nf, Rq, Nq = load('p23_*_f16_*_w0'), load('p23_*_f16_*_new'), load('p23_*_q8_0_*_w0'), load('p23_*_q8_0_*_new')
        rr('P3 (new_q8/new_f16)/(W0_q8/W0_f16)', Rq, Nq, Rf, Nf, .99)
        ratio('P3 floor new_q8/new_f16', Nf, Nq, .90)
    elif ph == 'P4':
        Rf = load('p4_*_f16_f16_*_new')
        for t in 'q4_0_q4_0 bf16_bf16 q8_0_f16 f16_q8_0 f32_f32 q4_1_q4_1 q5_0_q5_0 q5_1_q5_1 iq4_nl_iq4_nl'.split():
            ratio(f'P4 {t} / f16', Rf, load(f'p4_*_{t}_*_new'), .90)
    elif ph == 'P5':
        for t in 'f16_f16 q8_0_q8_0 q4_0_q4_0 bf16_bf16 q8_0_f16 f16_q8_0'.split():
            R, N = load(f'p5_{t}_*_w0'), load(f'p5_{t}_*_new')
            ratio(f'P5 shortcut tg {t}, new vs W0', R, N, .98, lambda c: c[0] <= 1900)
            if t in ('f16_f16', 'q8_0_q8_0'): ratio(f'P5 depth tg {t}, new vs W0', R, N, .99, lambda c: c[0] > 1900)
            if t != 'f16_f16': ratio(f'P5 depth tg new {t} / new f16', load('p5_f16_f16_*_new'), N, .94, lambda c: c[0] > 1900)
    elif ph == 'P6':
        for t in 'f16 q8_0 q4_0'.split():
            for a in ('base', 'new'):
                c = load(f'p6_{t}_d0_{a}'); print(f'P6 (b) {t} d0 {a}: evicted_ms {EVMS.get(f"p6_{t}_d0_{a}")}, eviction samples {EV.get(f"p6_{t}_d0_{a}", 0)}, cells {[(k, round(pt(v), 2), len(v[0])) for k, v in c.items()]}')
    elif ph in ('ND', 'NDB'):
        pre = 'nk' if ph == 'ND' else 'nkb'
        for t in 'f16 q8_0 q4_0 bf16'.split():
            ratio(f'{ph} -nkvo {t}, new vs W0-lab', load(f'nk*_{t}_*_w0'), load(f'{pre}_{t}_*_new'), .98)
        ratio(f'{ph} -nkvo new_q8_0 / new_f16', load(f'{pre}_f16_*_new'), load(f'{pre}_q8_0_*_new'), .94)
    elif ph == 'D2B':
        for t in ('f16', 'q8_0'):
            R, N = load(f'd2b_{t}_*_w0'), load(f'd2b_{t}_*_new')
            ratio(f'AT-D2 -ngl 0 {t}, new vs W0-lab (threshold 1 - 2 x the W0 between-process spread, printed below)', R, N, 0.0)
            for c, ps in R.items():
                ms = [sum(p)/len(p) for p in ps]; m = sum(ms)/len(ms)
                print(f'  W0 spread {c}: {(max(ms)-min(ms))/m:.4f} -> threshold {1 - 2*(max(ms)-min(ms))/m:.4f}')
    elif ph == 'D1V':
        for t in ('f16', 'q8_0'):
            # pass: the CI contains 1 (RERUN in this table) or lies above it (PASS)
            rr(f'AT-D1 value {t}: (W7 OVL15/OVL11) / (W0 OVL15/OVL11)', load(f'd1v_*_{t}_*_w0_ovl15'), load(f'd1v_*_{t}_*_new_ovl15'), load(f'd1v_*_{t}_*_w0_ovl11'), load(f'd1v_*_{t}_*_new_ovl11'), 1.0)
    elif ph == 'B1':
        # AT-B1 (P8): f16 new / W0 >= 0.99; every other S_fast pair new_T / new f16 >= 0.95 in every cell
        Nf = load('b1_f16_f16_*_new')
        ratio('AT-B1 Vulkan f16, new vs W0', load('b1_f16_f16_*_w0'), Nf, .99)
        for t in 'q8_0_q8_0 q4_0_q4_0 bf16_bf16 q8_0_f16 f16_q8_0'.split():
            ratio(f'AT-B1 Vulkan new {t} / new f16', Nf, load(f'b1_{t}_*_new'), .95)
    elif ph == 'NPIN':
        ratio('AT-W7PERF (d) shortcut pin on / off', load('npin_*_off'), load('npin_*_on'), .98)
print(f'\n## totals {dict(RES)}; eviction samples dropped {sum(EV.values())} in {len(EV)} processes; processes with evictions {sum(1 for v in EVMS.values() if v > 0)}/{len(EVMS)}, mean evicted_ms {sum(EVMS.values())/max(1,len(EVMS)):.0f}; P6 (a) samples below 0.9 x cell median: {len(P6A)} {P6A[:6]}')
