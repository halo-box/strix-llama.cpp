#!/usr/bin/env python3
# R2-M-LAB-BANDS-OP analysis: per case (shape, type) GPU-vs-O_CON NMSE over samples (UNI seeds 1-3; TBO dumps r1/r2 with harness
# repeats), class = dispatched (G, splits) from the CASE line (= QSA_ROUTE of test-backend-ops for TBO), check 2 x p100 <= SYN row,
# defects D (min over samples), k = D / p100.
import sys, glob, os, re, collections
R = sys.argv[1]
SYN = {'G4': 2.67e-9, 'G1-1split': 3.03e-9, 'G1-splitK': 1.15e-7}
SYNP = {k: v / 2 for k, v in SYN.items()}
def kv(t):
    d = {}
    for i in range(len(t) - 1):
        try: d[t[i]] = float(t[i + 1])
        except ValueError: pass
    return d
cases = collections.OrderedDict()
def get(key):
    if key not in cases: cases[key] = dict(nmse=[], maxrel=[], c18=[], plan=[], D={1: [], 2: [], 3: []}, drop={1: [], 2: []}, tbo_bw=[], tbocpu=[], uids=set(), dups=0, samples=[], splits=None, chunk=None)
    return cases[key]
for f in sorted(glob.glob(os.path.join(R, '*.out'))):
    tag = os.path.basename(f)[:-4]
    if tag.startswith('syn_') or tag == 'validate_inputs': continue
    L = open(f).read().splitlines()
    fx = [l for l in L if l.startswith('FIXTURE')][0].split(); cs = [l for l in L if l.startswith('CASE')][0].split()
    c = kv(cs); nq, ns, nk, bl = int(c['n_q']), int(c['ns']), int(c['nk']), int(c['blocks'])
    tool = 'attn' if 'direct_rule' in cs else 'dec'
    if tool == 'attn': G, splits, chunk = 'G4', None, None
    else:
        splits, chunk = int(c['splits']), int(c['chunk']); G = 'G1-1split' if splits == 1 else 'G1-splitK'
    if fx[3] == 'TBO':
        src = 'TBO'; m = re.match(r't_(r\d)_(QSA_\w+?)_(\d+)(_rep\d)?$', tag); sample = m.group(1) + (m.group(4) or '_rep1')
        inter = fx[fx.index('interleaved') + 1]
    else:
        src = 'UNI'; sample = 's' + tag.split('_s')[-1]; inter = '-'
    for l in L:
        p = l.split()
        if not p: continue
        if p[0] == 'BAND':
            var = p[3]; ty = p[5].split('/')[0]
            if var.startswith('PR-VD'): continue          # bitwise equal to PR-V0 (BITWISE lines), not the dispatched route on these shapes
            e = get((src, G, nq, ns, nk, bl, ty)); s = kv(p)
            if p[6] == 'gpu_vs_contract': e['nmse'].append(s['nmse']); e['maxrel'].append(s['maxrel_p100']); e['samples'].append(sample); e['splits'] = splits; e['chunk'] = chunk
            elif p[6] == 'gpu_vs_f64_c18': e['c18'].append(s['nmse'])
            else: e['plan'].append(s['nmse'])
        elif p[0] == 'DEFECT':
            ty = p[4].split('/')[0]; e = get((src, G, nq, ns, nk, bl, ty)); d = int(p[5][1]); s = kv(p)
            e['D'][d].append(s['nmse'])
            if d < 3: e['drop'][d].append(int(p[7]))
        elif p[0] == 'TBO_BITWISE':
            ty = 'q8_0' if 'q8_0' in p[2] else 'f16'; e = get((src, G, nq, ns, nk, bl, ty)); e['tbo_bw'].append((sample, int(p[6]), float(p[8])))
        elif p[0] == 'TBO_CPU':
            ty = p[4].split('/')[0]; e = get((src, G, nq, ns, nk, bl, ty)); e['tbocpu'].append(kv(p)['nmse'])
        elif p[0] == 'UIDS' and p[3] == 'hash':
            for ty in ('f16', 'q8_0'):
                k = (src, G, nq, ns, nk, bl, ty)
                if k in cases or True: pass
            UID = (p[4], int(p[6]), int(p[8]))
            for ty in ('f16', 'q8_0'):
                if os.path.basename(f).find('q8') >= 0: pass
            get((src, G, nq, ns, nk, bl, 'any'))['uids'].add((sample[:2], UID))
# drop the helper 'any' rows after collecting
uids = {k[:6]: v['uids'] for k, v in cases.items() if k[6] == 'any'}
for k in [k for k in cases if k[6] == 'any']: del cases[k]
def fmt(x): return '%.3g' % x
out = []
out.append('## Per case (GQA 12): GPU (PR-verbatim, dispatched route) vs O_CON, NMSE over samples; defects vs clean O_CON\n')
out.append('| src | class | splits/chunk | n_q | n_sel | n_kv | blocks | type | samples | NMSE min | NMSE p100 | 2 x p100 | SYN row | 2p100 <= SYN | maxrel p100 | d1 min | d2 min | d3 min | k = Dmin/p100 | Dmin / SYN row | TBO bitwise (words differ, nmse) | uids dup pairs |')
out.append('|' + '---|' * 22)
cls = collections.defaultdict(lambda: dict(p100=0, case=None, Dmin=[], cases=[]))
fails = []
for k, e in cases.items():
    src, G, nq, ns, nk, bl, ty = k
    if not e['nmse']: continue
    p100 = max(e['nmse']); mn = min(e['nmse'])
    Dm = {d: (min(v) if v else float('nan')) for d, v in e['D'].items()}
    Dpos = [v for v in Dm.values() if v == v and v > 0]
    kmin = min(Dpos) / p100 if Dpos else float('nan')
    ok = 2 * p100 <= SYN[G]
    if not ok: fails.append(k)
    bw = ''
    if e['tbo_bw']:
        bw = '%d/%d runs bitwise; max nmse %.2g' % (sum(1 for x in e['tbo_bw'] if x[1] == 0), len(e['tbo_bw']), max(x[2] for x in e['tbo_bw']))
    u = uids.get(k[:6], set()); dp = max((x[1][1] for x in u), default=0)
    nU = len(set(x[1][0] for x in u))
    d1note = fmt(Dm[1]) if Dm[1] > 0 else '0 (no tail cell)'
    out.append('| %s | %s | %s | %d | %d | %d | %d | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.1f | %.1f | %s | %s |' % (
        src, G, '-' if e['splits'] is None else '%d/%d' % (e['splits'], e['chunk']), nq, ns, nk, bl, ty, len(e['nmse']), fmt(mn), fmt(p100), fmt(2 * p100), fmt(SYN[G]),
        'yes' if ok else '**NO**', fmt(max(e['maxrel'])), d1note, fmt(Dm[2]), fmt(Dm[3]), kmin, min(Dpos) / SYN[G] if Dpos else float('nan'), bw,
        ('%d (%d distinct uid sets)' % (dp, nU)) if u else '-'))
    # class buckets: G, splits bucket, n_sel bucket
    sb = 'G4' if G == 'G4' else ('1 split' if e['splits'] == 1 else ('splits 2-5' if e['splits'] <= 5 else 'splits 17-33'))
    if nk == 262144: continue   # UNI-vs-SYN comparison cells, not ported shapes
    ck = (G, sb, ns)
    c = cls[ck]
    if p100 > c['p100']: c['p100'] = p100; c['case'] = k
    c['cases'].append((k, Dpos))
out.append('\n## Per class over the ported GQA-12 shapes only (n_kv 262144 comparison cells excluded) (G, splits bucket, n_sel): UNI+TBO p100 and the option-2 row (2 x p100); min k over the class at that band\n')
out.append('| G | splits bucket | n_sel | cases (shape x type x src) | p100 NMSE | set by | 2 x p100 (option-2 row) | SYN row | SYN row holds | min over cases of Dmin / p100_class | min Dmin / SYN row |')
out.append('|---|---|---|---|---|---|---|---|---|---|---|')
for ck in sorted(cls, key=lambda x: (x[0], x[1], x[2])):
    c = cls[ck]; G = ck[0]
    kk = [min(D) / c['p100'] for _, D in c['cases'] if D]; ks = [min(D) / SYN[G] for _, D in c['cases'] if D]
    sk = c['case']
    out.append('| %s | %s | %d | %d | %s | %s %d/%d/%d b%d %s | %s | %s | %s | %.1f | %.1f |' % (G, ck[1], ck[2], len(c['cases']), fmt(c['p100']), sk[0], sk[2], sk[3], sk[4], sk[5], sk[6],
        fmt(2 * c['p100']), fmt(SYN[G]), 'yes' if 2 * c['p100'] <= SYN[G] else '**NO**', min(kk), min(ks)))
out.append('\nSYN-row failures (2 x p100 > SYN row): %d of %d cases' % (len(fails), sum(1 for e in cases.values() if e['nmse'])))
open(os.path.join(R, '..', 'analysis_op.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
