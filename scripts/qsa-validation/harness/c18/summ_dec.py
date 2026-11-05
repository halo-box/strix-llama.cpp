#!/usr/bin/env python3
import re, sys, glob, os, collections
D = sys.argv[1] if len(sys.argv) > 1 else 'res'
def load(f):
    med = collections.defaultdict(dict); chk = []; info = {}
    for l in open(f):
        m = re.match(r'L(\d+) RES (\S+)\s+nk (\d+) n_q (\d+) splits (\d+) chunk (\d+) unique/q ([\d.]+) med ([\d.]+)', l)
        if m: med[m.group(2)][int(m.group(1))] = float(m.group(8)); info = dict(nk=m.group(3), nq=m.group(4), splits=m.group(5), chunk=m.group(6), uq=m.group(7)); continue
        if ' CHK ' in l: chk.append(l.strip())
    return med, chk, info
fx = sorted(set(re.sub(r'_r\d$', '', os.path.basename(f)[:-4]) for f in glob.glob(D + '/dec*.out')))
for x in fx:
    runs = sorted(glob.glob(D + '/' + x + '_r*.out')); data = [load(f) for f in runs]
    if not data[0][0]: continue
    bad = [c for d in data for c in d[1] if 'words_differ_vs_type_ref' in c and not c.endswith(' 0')]
    nb = sum(1 for d in data for c in d[1] if 'words_differ' in c)
    mr = collections.defaultdict(float)
    for d in data:
        for c in d[1]:
            m = re.search(r'type (\S+) cpu_f64_maxrel ([\d.e+-]+)', c)
            if m and float(m.group(2)) >= 0: mr[m.group(1)] = max(mr[m.group(1)], float(m.group(2)))
    print('\n### %s (runs %d, %s)' % (x, len(runs), data[0][2]))
    print('correctness: %d bitwise checks, %d failures; cpu f64 max rel err: %s' % (nb, len(bad), ', '.join('%s %.1e' % kv for kv in sorted(mr.items()))))
    agg = {}
    for v in data[0][0]:
        L = sorted(data[0][0][v]); agg[v] = [sum(d[0][v][l] for l in L) / len(L) for d in data]
    mean = lambda v: sum(agg[v]) / len(agg[v])
    base = mean('Kf16D-Vf16D')
    print('| variant | per run (ms) | mean | spread | vs own type D+D | vs f16 D+D |'); print('|---|---|---|---|---|---|')
    for T in ['f16', 'q8_0', 'q4_0', 'bf16']:
        own = mean('K%sD-V%sD' % (T, T))
        for KL, VL in [('D','D'),('X','D'),('S','D'),('D','X'),('D','S'),('X','X'),('X','S'),('S','S')]:
            v = 'K%s%s-V%s%s' % (T, KL, T, VL)
            if v in agg:
                sp = (max(agg[v]) - min(agg[v])) / mean(v) * 100 if len(agg[v]) > 1 else float('nan')
                print('| %s | %s | %.4f | %.1f%% | %.3f | %.3f |' % (v, ' / '.join('%.4f' % a for a in agg[v]), mean(v), sp, mean(v) / own, mean(v) / base))
        pr = 'PR-dec-' + T
        if pr in agg:
            sp = (max(agg[pr]) - min(agg[pr])) / mean(pr) * 100 if len(agg[pr]) > 1 else float('nan')
            print('| %s | %s | %.4f | %.1f%% | %.3f | %.3f |' % (pr, ' / '.join('%.4f' % a for a in agg[pr]), mean(pr), sp, mean(pr) / own, mean(pr) / base))
