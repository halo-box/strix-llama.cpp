#!/usr/bin/env python3
# R1-M-Lab A analysis: parse res/*.out of bands_attn / bands_dec / bands_cpu, print the band tables and the k decision.
import sys, re, glob, os, collections
D = sys.argv[1]
def kv(s):
    t = s.split(); d = {}
    for i in range(len(t) - 1):
        try: d[t[i]] = float(t[i + 1])
        except ValueError: pass
    return d
band = []   # per-layer GPU lines: (fix, L, variant, pair, oracle, stats)
bsum = []   # pooled
cpub = []; cpus = []; bitw = []; hashes = collections.defaultdict(set)
for f in sorted(glob.glob(os.path.join(D, '*.out'))):
    for ln in open(f):
        p = ln.split()
        if not p: continue
        if p[0] == 'BAND':
            band.append((p[1], int(p[2][1:]), p[3], p[5], p[6].replace('gpu_vs_', ''), kv(ln)))
        elif p[0] == 'BANDSUM':
            bsum.append((p[1], p[2], p[3], p[4], kv(ln)))
        elif p[0] == 'CPUB':
            cpub.append((p[1], int(p[2][1:]), p[3], p[4], kv(ln)))
        elif p[0] == 'CPUSUM':
            cpus.append((p[1], p[2], p[3], kv(ln)))
        elif p[0] == 'BITWISE' and 'words_differ' in ln:
            bitw.append((p[1], p[2], p[3], int(p[-1])))
        elif p[0] in ('CACHE_HASH', 'ROWS_HASH'):
            hashes[(p[2], ' '.join(p[3:-1]) if p[0] == 'CACHE_HASH' else 'rows', os.path.basename(f).split('_')[0])].add(p[-1] if p[0] == 'CACHE_HASH' else p[3])
def fixof(tag):
    if tag.startswith('dec512'): return 'dec512_' + tag.split('_', 1)[1]
    if tag.startswith('dec1'): return 'dec1_' + tag.split('_', 1)[1]
    return tag
print('## bitwise: %d comparisons, %d with words differing' % (len(bitw), sum(1 for b in bitw if b[3])))
for b in bitw:
    if b[3]: print('  DIFF', b)
# ---- GPU bands per (fixture tag, variant, pair, oracle): max over layers of per-layer nmse, max of per-row maxrel / rl2, max p99.9
agg = collections.OrderedDict()
for fix, L, var, pair, orc, s in band:
    k = (fix, var, pair, orc); a = agg.setdefault(k, dict(nmse=0, mr999=0, mr100=0, rl999=0, rl100=0, rows=0, L=0, nmse_min=1e9))
    a['nmse'] = max(a['nmse'], s['nmse']); a['nmse_min'] = min(a['nmse_min'], s['nmse']); a['mr999'] = max(a['mr999'], s['maxrel_p99.9']); a['mr100'] = max(a['mr100'], s['maxrel_p100'])
    a['rl999'] = max(a['rl999'], s['rl2_p99.9']); a['rl100'] = max(a['rl100'], s['rl2_p100']); a['rows'] += int(s['rows']); a['L'] += 1
print('\n## GPU per (fixture, variant, pair, oracle): per-layer NMSE max/min, per-row maxrel p99.9 (max over layers) / p100, rl2 p99.9 / p100')
print('| fixture | variant | pair | oracle | layers | rows | NMSE max | NMSE min | maxrel p99.9 | maxrel p100 | rl2 p99.9 | rl2 p100 |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for (fix, var, pair, orc), a in agg.items():
    print('| %s | %s | %s | %s | %d | %d | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e |' % (fix, var, pair, orc, a['L'], a['rows'], a['nmse'], a['nmse_min'], a['mr999'], a['mr100'], a['rl999'], a['rl100']))
# pooled p99.9 from BANDSUM
print('\n## GPU pooled over layers (BANDSUM; rows pooled, p99.9 over all rows)')
print('| fixture | variant | pair | oracle | rows | pooled NMSE | maxrel p99.9 | maxrel p100 | rl2 p50 | rl2 p99.9 | rl2 p100 |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for tag, var, pair, orc, s in bsum:
    if var == 'REF': continue
    print('| %s | %s | %s | %s | %d | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e |' % (tag, var, pair, orc.replace('gpu_vs_', ''), s['rows'], s['nmse'], s['maxrel_p99.9'], s['maxrel_p100'], s['rl2_p50'], s['rl2_p99.9'], s['rl2_p100']))
# ---- band p100 per pair (contract)
bp = collections.OrderedDict(); bpm = collections.OrderedDict()
for (fix, var, pair, orc), a in agg.items():
    if orc != 'contract': continue
    bp[pair] = max(bp.get(pair, 0), a['nmse']); bpm[pair] = max(bpm.get(pair, 0), a['mr100'])
gp = max(bp.values()); gpm = max(bpm.values())
print('\n## band p100 of GPU-vs-contract per-op (per-layer) NMSE and of per-row maxrel, per pair over every variant/fixture/layer')
for p in bp: print('  %-10s nmse_p100 %.3e  maxrel_p100 %.3e' % (p, bp[p], bpm[p]))
print('  GLOBAL     nmse_p100 %.3e  maxrel_p100 %.3e' % (gp, gpm))
# ---- CPU
print('\n## CPU per (fixture, pair, metric): per-layer NMSE max/min, maxrel p99.9 / p100 (max over layers), rl2 p100')
cagg = collections.OrderedDict()
for tag, L, pair, met, s in cpub:
    k = (tag, pair, met); a = cagg.setdefault(k, dict(nmse=0, nmse_min=1e9, mr999=0, mr100=0, mr100_min=1e9, rl100=0, rl50=0, L=0))
    a['nmse'] = max(a['nmse'], s['nmse']); a['nmse_min'] = min(a['nmse_min'], s['nmse']); a['mr999'] = max(a['mr999'], s['maxrel_p99.9']); a['mr100'] = max(a['mr100'], s['maxrel_p100'])
    a['mr100_min'] = min(a['mr100_min'], s['maxrel_p100']); a['rl100'] = max(a['rl100'], s['rl2_p100']); a['rl50'] = max(a['rl50'], s['rl2_p50']); a['L'] += 1
print('| fixture | pair | metric | layers | NMSE max | NMSE min | maxrel p99.9 | maxrel p100 | maxrel p100 min | rl2 p50 | rl2 p100 |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for (tag, pair, met), a in cagg.items():
    print('| %s | %s | %s | %d | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e |' % (tag, pair, met, a['L'], a['nmse'], a['nmse_min'], a['mr999'], a['mr100'], a['mr100_min'], a['rl50'], a['rl100']))
# ---- defects vs bands, k rule
print('\n## defects (min over layers and fixtures of the defect metric) vs band p100; k rule: smallest k>=2 with every defect > k*p100')
for metric in ('nmse', 'mr100'):
    for d in ('defect_d1_tailcell_vs_contract', 'defect_d2_block_vs_contract', 'defect_d3_v1e-3_vs_contract'):
        for pair in bp:
            vals = [a['nmse_min'] if metric == 'nmse' else a['mr100_min'] for (tag, pr, met), a in cagg.items() if met == d and pr == pair]
            if not vals: continue
            b = bp[pair] if metric == 'nmse' else bpm[pair]; g = gp if metric == 'nmse' else gpm
            m = min(vals)
            print('  %-5s %-32s %-10s defect_min %.3e  pair_band %.3e ratio %.1f  global_band %.3e ratio %.1f  k=2 %s' % (metric, d, pair, m, b, m / b, g, m / g, 'DETECTS' if m > 2 * g else 'FAILS'))
