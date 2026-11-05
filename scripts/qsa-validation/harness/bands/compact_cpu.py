#!/usr/bin/env python3
import sys, glob, os, collections
D = sys.argv[1]
def kv(s):
    t = s.split(); d = {}
    for i in range(len(t) - 1):
        try: d[t[i]] = float(t[i + 1])
        except ValueError: pass
    return d
fx = {'cpu_wiki4k': 'wiki', 'cpu_wiki4k_s8': 'wiki', 'cpu_syn262k': 'syn'}
A = collections.OrderedDict()
for t in fx:
    for ln in open(os.path.join(D, t + '.out')):
        p = ln.split()
        if not p or p[0] != 'CPUB': continue
        pair, met = p[3], p[4]; s = kv(ln)
        a = A.setdefault(pair, {}).setdefault((fx[t], met), dict(nmse=0, mr100=0, mr999=0))
        a['nmse'] = max(a['nmse'], s['nmse']); a['mr100'] = max(a['mr100'], s['maxrel_p100']); a['mr999'] = max(a['mr999'], s['maxrel_p99.9'])
print('| pair K/V | CPU-FA vs contract NMSE p100 wiki / SYN | CPU-FA vs contract maxrel p100 wiki / SYN | contract vs f64 §3 NMSE wiki / SYN | contract vs f64 §3 maxrel p99.9 / p100 (wiki) | contract vs f64 C18 NMSE wiki / SYN | contract vs f64 C18 maxrel p100 wiki / SYN | CPU-FA vs f64 §3 NMSE wiki |')
print('|---|---|---|---|---|---|---|---|')
for pair, a in A.items():
    g = lambda f, m, k: a[(f, m)][k]
    print('| %s | %.2e / %.2e | %.2e / %.2e | %.2e / %.2e | %.2e / %.2e | %.2e / %.2e | %.2e / %.2e | %.2e |' % (pair,
        g('wiki','cpufa_vs_contract','nmse'), g('syn','cpufa_vs_contract','nmse'), g('wiki','cpufa_vs_contract','mr100'), g('syn','cpufa_vs_contract','mr100'),
        g('wiki','contract_vs_f64_plan','nmse'), g('syn','contract_vs_f64_plan','nmse'), g('wiki','contract_vs_f64_plan','mr999'), g('wiki','contract_vs_f64_plan','mr100'),
        g('wiki','contract_vs_f64_c18','nmse'), g('syn','contract_vs_f64_c18','nmse'), g('wiki','contract_vs_f64_c18','mr100'), g('syn','contract_vs_f64_c18','mr100'), g('wiki','cpufa_vs_f64_plan','nmse')))
