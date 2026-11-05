#!/usr/bin/env python3
# C18 summary: per fixture, mean over layers of per-layer medians (ms) for r1 / r2, run-to-run spread, ratios.
import re, sys, glob, os, collections
D = sys.argv[1] if len(sys.argv) > 1 else 'res'
def load(f):
    med = collections.defaultdict(dict); chq = {}; nk = None; chk = []
    for l in open(f):
        m = re.match(r'L(\d+) RES (\S+)\s+nk (\d+)(?: ch/q ([\d.]+))? med ([\d.]+) mean ([\d.]+) sd ([\d.]+) min ([\d.]+) n (\d+)', l)
        if m:
            L, v = int(m.group(1)), m.group(2); nk = int(m.group(3)); med[v][L] = float(m.group(5))
            if m.group(4): chq[L] = float(m.group(4))
            continue
        if ' CHK ' in l: chk.append(l.strip())
    return med, chq, nk, chk
def mean(x): return sum(x) / len(x)
fixtures = sorted(set(re.sub(r'_r\d$|_s\d$', '', os.path.basename(f)[:-4]) for f in glob.glob(D + '/*.out')))
TYPES = ['f16', 'q8_0', 'q4_0', 'bf16', 'bf16w']
for fx in fixtures:
    runs = sorted(glob.glob(D + '/' + fx + '_r*.out')) + sorted(glob.glob(D + '/' + fx + '_s*.out'))
    data = [load(f) for f in runs]
    if not data or not data[0][0]: continue
    vs = list(data[0][0].keys())
    layers = sorted(data[0][0][vs[0]].keys())
    print('\n### %s  (runs %s, layers %d, nk %s, G4 ch/q mean %.2f)' % (fx, ','.join(os.path.basename(r)[:-4] for r in runs), len(layers), data[0][2], mean(list(data[0][1].values())) if data[0][1] else 0))
    # correctness summary
    bad = [c for d in data for c in d[3] if ('words_differ_vs_pair_ref' in c and 'words_differ_vs_pair_ref 0 ' not in c) or (' nan 0' not in c)]
    maxrel = collections.defaultdict(float)
    for d in data:
        for c in d[3]:
            m = re.search(r'pair (\S+) cpu_f64_maxrel ([\d.e+-]+)', c)
            if m and float(m.group(2)) >= 0: maxrel[m.group(1)] = max(maxrel[m.group(1)], float(m.group(2)))
    nchk = sum(1 for d in data for c in d[3] if 'words_differ_vs_pair_ref' in c)
    print('correctness: %d bitwise checks, %d failures; cpu f64 max rel err per pair: %s' % (nchk, len(bad), ', '.join('%s %.1e' % kv for kv in sorted(maxrel.items()))))
    for b in bad[:5]: print('  BAD', b)
    agg = {}
    for v in vs:
        per = [mean([d[0][v][L] for L in layers if L in d[0][v]]) for d in data if v in d[0]]
        agg[v] = per
    chq = mean(list(data[0][1].values())) if data[0][1] else None
    def m(v): return mean(agg[v]) if v in agg else float('nan')
    def sp(v): return (max(agg[v]) - min(agg[v])) / mean(agg[v]) * 100 if v in agg and len(agg[v]) > 1 else float('nan')
    base = m('Kf16D-Vf16P')
    def row(v, extra=0.0, ref=None, label=None):
        if v not in agg: return
        t = m(v) + extra
        r = t / (ref if ref else base)
        nsc = t * 1e6 / (chq * 4 * 1024 * 2) if chq else float('nan')
        runs_s = ' / '.join('%.3f' % (x + extra) for x in agg[v])
        print('| %s | %s | %.3f | %.1f%% | %.3f | %.2f |' % (label or v, runs_s, t, sp(v), r, nsc))
    print('\nK loaders (V = f16 packed, pack excluded; ratio to Kf16D-Vf16P = PR V0 f16 route):')
    print('| variant | per run (ms) | mean | spread | ratio | ns/chunk |'); print('|---|---|---|---|---|---|')
    for T in TYPES:
        for L in ['D', 'Dd', 'X', 'S']: row('K%s%s-Vf16P' % (T, L))
    row('PR-V0-f16'); row('PR-V0-q8_0')
    print('\nV loaders (K = f16 direct; P includes its pack kernel; ratio to Kf16D-Vf16P + PACKV-f16):')
    print('| variant | per run (ms) | mean | spread | ratio | ns/chunk |'); print('|---|---|---|---|---|---|')
    pf = m('PACKV-f16')
    for T in TYPES:
        pk = m('PACKV-' + T) if 'PACKV-' + T in agg else float('nan')
        vP = 'Kf16D-V%sP' % (T if T in ('f16', 'bf16w') else 'f16')
        row(vP, extra=pk, ref=base + pf, label='Kf16D-V%sP + PACKV-%s' % (T, T))
        for L in ['D', 'X', 'S']: row('Kf16D-V%s%s' % (T, L), ref=base + pf)
    print('\nPairs (K and V same type; P includes PACKV of that type; ratio to the PR V0 route of that type = KTD-VTP + PACKV-T):')
    print('| variant | per run (ms) | mean | spread | ratio | ns/chunk |'); print('|---|---|---|---|---|---|')
    for T in TYPES:
        pk = m('PACKV-' + T) if 'PACKV-' + T in agg else float('nan')
        ref = m('K%sD-V%sP' % (T, T)) + pk
        row('K%sD-V%sP' % (T, T), extra=pk, ref=ref, label='K%sD-V%sP + PACKV-%s' % (T, T, T))
        for KL, VL in [('D', 'D'), ('X', 'X'), ('X', 'S'), ('S', 'S')]: row('K%s%s-V%s%s' % (T, KL, T, VL), ref=ref)
        vp = 'K%sX-V%sP' % (T, 'bf16w' if T == 'bf16w' else 'f16')
        if vp in agg: row(vp, extra=pk, ref=ref, label='derived: K%sX + V%s packed = %s + PACKV-%s' % (T, T, vp, T))
        if T == 'q8_0': row('PR-VD-q8_0', ref=ref); row('PR-V0-q8_0', extra=pk, ref=ref, label='PR-V0-q8_0 + PACKV-q8_0')
        if T == 'f16': row('PR-VD-f16', ref=ref); row('PR-V0-f16', extra=pk, ref=ref, label='PR-V0-f16 + PACKV-f16')
    f16route = base + pf
    print('\nCross-type (ratio to the f16 PR route Kf16D-Vf16P + PACKV-f16 = %.3f ms):' % f16route)
    for T in TYPES:
        pk = m('PACKV-' + T) if 'PACKV-' + T in agg else float('nan')
        cands = {('PR route D+P', m('K%sD-V%sP' % (T, T)) + pk)}
        for KL, VL in [('X', 'X'), ('X', 'S'), ('S', 'S'), ('D', 'D')]:
            v = 'K%s%s-V%s%s' % (T, KL, T, VL)
            if v in agg: cands.add(('%s+%s' % (KL, VL), m(v)))
        vp = 'K%sX-V%sP' % (T, 'bf16w' if T == 'bf16w' else 'f16')
        if vp in agg: cands.add(('X+P(derived)', m(vp) + pk))
        cs = sorted(cands, key=lambda x: x[1])
        print('  %s: ' % T + ', '.join('%s %.3f (%.3f)' % (n, t, t / f16route) for n, t in cs))
    print('\nPACKV (ms):', ', '.join('%s %.3f' % (T, m('PACKV-' + T)) for T in TYPES if 'PACKV-' + T in agg))
