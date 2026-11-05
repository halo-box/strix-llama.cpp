#!/usr/bin/env python3
# compact tables for MEASUREMENTS: GPU bands per (shape/fixture, pair), variants collapsed (they are bitwise equal per pair)
import sys, glob, os, collections
D = sys.argv[1]
def kv(s):
    t = s.split(); d = {}
    for i in range(len(t) - 1):
        try: d[t[i]] = float(t[i + 1])
        except ValueError: pass
    return d
shape = {'wiki4k': 'prefill wiki d4k', 'syn262k_s1': 'prefill SYN 262k', 'syn262k_s2': 'prefill SYN 262k', 'dec512_wiki4k': 'dec n_q512 wiki', 'dec512_syn262k': 'dec n_q512 SYN', 'dec1_wiki4k': 'dec n_q1 wiki', 'dec1_syn262k': 'dec n_q1 SYN'}
A = collections.OrderedDict(); vars_ = collections.defaultdict(set)
for f in sorted(glob.glob(os.path.join(D, '*.out'))):
    if 'wiki4k_r2' in f: continue
    for ln in open(f):
        p = ln.split()
        if not p or p[0] != 'BAND': continue
        tag, var, pair, orc = p[1], p[3], p[5], p[6].replace('gpu_vs_', ''); s = kv(ln)
        k = (shape[tag], pair); a = A.setdefault(k, {}); o = a.setdefault(orc, dict(nmse=0, mr999=0, mr100=0, rl100=0, rl50=0))
        o['nmse'] = max(o['nmse'], s['nmse']); o['mr999'] = max(o['mr999'], s['maxrel_p99.9']); o['mr100'] = max(o['mr100'], s['maxrel_p100']); o['rl100'] = max(o['rl100'], s['rl2_p100']); o['rl50'] = max(o['rl50'], s['rl2_p50'])
        vars_[k].add(var)
print('| shape / fixture | pair | variants | vs contract: NMSE p100 / maxrel p99.9 / p100 | vs f64 C18: NMSE / maxrel p99.9 / p100 | vs f64 §3: NMSE / maxrel p99.9 / p100 | rl2 vs f64 C18: p50 / p100 |')
print('|---|---|---|---|---|---|---|')
for (sh, pair), a in A.items():
    c, o1, o2 = a['contract'], a['f64_c18'], a['f64_plan']
    print('| %s | %s | %s | %.2e / %.2e / %.2e | %.2e / %.2e / %.2e | %.2e / %.2e / %.2e | %.2e / %.2e |' % (sh, pair, ', '.join(sorted(vars_[(sh, pair)])), c['nmse'], c['mr999'], c['mr100'], o1['nmse'], o1['mr999'], o1['mr100'], o2['nmse'], o2['mr999'], o2['mr100'], o1['rl50'], o1['rl100']))
# per shape max over pairs
print()
S = collections.OrderedDict()
for (sh, pair), a in A.items():
    s = S.setdefault(sh, collections.defaultdict(float))
    for orc in a:
        for m in a[orc]: s[orc + '.' + m] = max(s[orc + '.' + m], a[orc][m])
print('| shape / fixture | band: NMSE p100 vs contract | maxrel p100 vs contract | maxrel p100 vs f64 C18 | maxrel p100 vs f64 §3 |')
print('|---|---|---|---|---|')
for sh, s in S.items(): print('| %s | %.3e | %.3e | %.3e | %.3e |' % (sh, s['contract.nmse'], s['contract.mr100'], s['f64_c18.mr100'], s['f64_plan.mr100']))
