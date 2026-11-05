#!/usr/bin/env python3
# R2-M-LAB-BANDS-OP follow-up summary (res2): e1 = UNI rows sorted ascending, e2 = uniform +-1.7 data, syn_* = SYN 262k seeds 2,3
import glob, collections, re, sys, os
def kv(t):
    d = {}
    for i in range(len(t) - 1):
        try: d[t[i]] = float(t[i + 1])
        except ValueError: pass
    return d
R = collections.defaultdict(list); DEF = collections.defaultdict(list); SPL = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.out'))):
    tag = os.path.basename(f)[:-4]; base = re.sub(r'_s\d$', '', tag)
    for l in open(f):
        p = l.split()
        if not p: continue
        if p[0] == 'BAND' and p[6] == 'gpu_vs_contract': R[(base, p[5].split('/')[0])].append(kv(p)['nmse'])
        elif p[0] == 'DEFECT': DEF[(base, p[4].split('/')[0], p[5])].append(kv(p)['nmse'])
        elif p[0] == 'CASE': SPL[base] = '%d/%d' % (kv(p)['splits'], kv(p)['chunk'])
print('| run | splits/chunk | type | samples | NMSE min | NMSE p100 | 2 x p100 | d1 / d2 / d3 min |')
print('|---|---|---|---|---|---|---|---|')
for k in sorted(R):
    v = R[k]; d = [min(DEF[(k[0], k[1], x)]) if DEF.get((k[0], k[1], x)) else None for x in ('d1', 'd2', 'd3')]
    print('| %s | %s | %s | %d | %.3g | %.3g | %.3g | %s |' % (k[0], SPL.get(k[0], '-'), k[1], len(v), min(v), max(v), 2 * max(v), ' / '.join('%.3g' % x if x is not None else '-' for x in d)))
