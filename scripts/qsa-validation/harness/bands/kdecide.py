#!/usr/bin/env python3
# k rule (DECIDES index 1): band B = p100 of per-op GPU-vs-contract NMSE (and of per-row maxrel) per shape class and fixture,
# over every pair/variant/layer; defect D = min over layers and pairs of the defect's metric on the same row count; detected at k iff D > k*B.
import sys, glob, os, collections
D = sys.argv[1]
def kv(s):
    t = s.split(); d = {}
    for i in range(len(t) - 1):
        try: d[t[i]] = float(t[i + 1])
        except ValueError: pass
    return d
gshape = {'wiki4k': ('prefill', 'wiki'), 'syn262k_s1': ('prefill', 'syn'), 'syn262k_s2': ('prefill', 'syn'), 'dec512_wiki4k': ('dec512', 'wiki'), 'dec512_syn262k': ('dec512', 'syn'), 'dec1_wiki4k': ('dec1', 'wiki'), 'dec1_syn262k': ('dec1', 'syn')}
# defect sources at the same row count: prefill wiki = 4096 rows (L0/5/11) and 512 rows (every layer); dec512 = 512 rows; dec1 = query 0
cshape = {'cpu_wiki4k': [('prefill', 'wiki')], 'cpu_wiki4k_s8': [('prefill', 'wiki'), ('dec512', 'wiki')], 'cpu_wiki4k_q0': [('dec1', 'wiki')],
          'cpu_syn262k': [('prefill', 'syn')], 'cpu_syn262k_s8': [('dec512', 'syn')], 'cpu_syn262k_q0': [('dec1', 'syn')]}
B = collections.defaultdict(lambda: [0.0, 0.0])
for f in glob.glob(os.path.join(D, '*.out')):
    for ln in open(f):
        p = ln.split()
        if p and p[0] == 'BAND' and p[6] == 'gpu_vs_contract' and p[1] in gshape:
            s = kv(ln); b = B[gshape[p[1]]]; b[0] = max(b[0], s['nmse']); b[1] = max(b[1], s['maxrel_p100'])
DF = collections.defaultdict(lambda: [1e9, 1e9])
for t, shs in cshape.items():
    for ln in open(os.path.join(D, t + '.out')):
        p = ln.split()
        if p and p[0] == 'CPUB' and p[4].startswith('defect_'):
            s = kv(ln)
            for sh in shs:
                d = DF[(sh, p[4].split('_')[1])]; d[0] = min(d[0], s['nmse']); d[1] = min(d[1], s['maxrel_p100'])
G = [max(b[0] for b in B.values()), max(b[1] for b in B.values())]
print('| shape | fixture | band NMSE p100 | d1 NMSE min | d2 NMSE min | d3 NMSE min | max k (NMSE) | k=2 per-shape band | k=2 one global band (%.2e) | band maxrel p100 | d1 / d2 / d3 maxrel min | max k (maxrel) |' % G[0])
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for sh in [('prefill','wiki'),('prefill','syn'),('dec512','wiki'),('dec512','syn'),('dec1','wiki'),('dec1','syn')]:
    b = B[sh]; d = [DF[(sh, x)] for x in ('d1', 'd2', 'd3')]
    kn = min(x[0] for x in d) / b[0]; km = min(x[1] for x in d) / b[1]
    gl = all(x[0] > 2 * G[0] for x in d)
    print('| %s | %s | %.3e | %.3e | %.3e | %.3e | %.1f | %s | %s | %.3e | %.2e / %.2e / %.2e | %.1f |' % (sh[0], sh[1], b[0], d[0][0], d[1][0], d[2][0], kn, 'all 3 detected' if kn > 2 else 'FAILS', 'all 3' if gl else 'FAILS: ' + ','.join('d%d' % (i + 1) for i, x in enumerate(d) if x[0] <= 2 * G[0]), b[1], d[0][1], d[1][1], d[2][1], km))
