# q4_analyze.py LOGDIR : per-chunk QSA selection agreement (QPIN_QSA_CHUNK in TAG.stats) and per-chunk KLD (QKLD_CHUNK in TAG.log)
# agreement per chunk: cb -> overlap of the cb fire; tk -> tk1 (first fire, computed rows) and the results-Q estimator 2*ov_all-1
import sys, re, math, os
L = sys.argv[1]
T = {7: 2.3646, 15: 2.1314}  # two-sided 95% t quantiles (df 7, 15)
T1 = {7: 1.8946, 15: 1.7531}  # one-sided 95%
def mstat(x):
    n = len(x); mu = sum(x) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in x) / (n - 1)) if n > 1 else float('nan')
    return mu, sd, sd / math.sqrt(n), n
def agree(tag):
    p = os.path.join(L, tag + '.stats')
    if not os.path.exists(p): return None
    d = {}
    for line in open(p):
        m = re.match(r'QPIN_QSA_CHUNK chunk=(\d+) src=(\S+) n=(\d+) overlap=(\S+)', line)
        if m: d.setdefault(int(m.group(1)), {})[m.group(2)] = (int(m.group(3)), float(m.group(4)))
    out = {}
    for c, s in d.items():
        if 'cb' in s: out[c] = {'direct': s['cb'][1], 'src': 'cb', 'n': s['cb'][0]}
        else:
            (n1, o1), (n2, o2) = s['tk1'], s['tk2']
            out[c] = {'direct': o1, 'corr': 2 * (n1 * o1 + n2 * o2) / (n1 + n2) - 1, 'tk2': o2, 'src': 'tk', 'n': n1}
    return out
def kld(tag):
    p = os.path.join(L, tag + '.log')
    if not os.path.exists(p): return None
    d = {}
    for line in open(p, errors='replace'):
        m = re.search(r'QKLD_CHUNK (\d+) all=(\S+)', line)
        if m: d[int(m.group(1))] = float(m.group(2))
    return d
def glob(tag):
    p = os.path.join(L, tag + '.stats')
    for line in open(p):
        if line.startswith('QPIN_QSA layer=-1 bucket=-1'): return line.strip()
def sets(v):
    r = []
    for name, sl in (('all16', slice(0, 16)), ('set0-7', slice(0, 8)), ('set8-15', slice(8, 16))):
        mu, sd, se, n = mstat(v[sl]); r.append(f"{name}: mean {mu:.6f} sd {sd:.3e} SE {se:.3e}")
    return ' | '.join(r)
arms = sorted(f[:-6] for f in os.listdir(L) if f.endswith('.stats') and not f.startswith('feas'))
A = {}
print('## per-arm agreement per chunk (direct estimator: cb fire, or tk first fire)')
for a in arms:
    g = agree(a)
    if not g: continue
    ks = sorted(g); A[a] = g
    v = [g[k]['direct'] for k in ks]
    print(f"{a} src={g[ks[0]]['src']} chunks={len(ks)} n/chunk={g[ks[0]]['n']} global: {glob(a)}")
    print('   direct ' + sets(v))
    print('   per-chunk ' + ' '.join(f"{x:.6f}" for x in v))
    if g[ks[0]]['src'] == 'tk':
        vc = [g[k]['corr'] for k in ks]
        print('   2ov-1  ' + sets(vc) + f" | max|corr-direct| {max(abs(g[k]['corr'] - g[k]['direct']) for k in ks):.2e} | tk2 min {min(g[k]['tk2'] for k in ks):.9f}")
    kd = kld(a)
    if kd:
        kv = [kd[k] for k in sorted(kd)]; mu, sd, se, n = mstat(kv)
        print(f"   KLD n={n} mean {mu:.6e} SE {se:.3e}")
def pair(b, a, key_b='direct', key_a='direct', ratio=False):
    if a not in A or b not in A: print(f"{b} vs {a}: missing"); return
    ks = sorted(set(A[a]) & set(A[b]))
    if ratio: d = [A[b][k][key_b] / A[a][k][key_a] for k in ks]
    else: d = [A[b][k][key_b] - A[a][k][key_a] for k in ks]
    out = []
    for name, sl in (('all16', slice(0, 16)), ('set0-7', slice(0, 8)), ('set8-15', slice(8, 16))):
        x = d[sl]; mu, sd, se, n = mstat(x)
        out.append(f"{name}: {'ratio' if ratio else 'diff'} {mu:+.6f} sd {sd:.3e} SE {se:.3e} LCB95(1s) {mu - T1[n-1]*se:+.6f} CI95 [{mu - T[n-1]*se:+.6f}, {mu + T[n-1]*se:+.6f}]")
    print(f"{b}[{key_b}] {'/' if ratio else '-'} {a}[{key_a}]:\n   " + '\n   '.join(out) + '\n   per-chunk ' + ' '.join(f"{x:+.6f}" for x in d))
print('\n## paired (per chunk)')
for b, a in (('e_f16_idxq8', 'e_f16_u2048'), ('e_f16_idxq8_r2', 'e_f16_idxq8'),
             ('e_q4_idxq8_cb', 'e_q4_idxf16_cb'), ('e_q4_idxq8_tk', 'e_q4_idxf16_tk'), ('e_q4_idxq4_cb', 'e_q4_idxf16_cb'), ('e_q4_idxq4_tk', 'e_q4_idxf16_tk'),
             ('e_q4_idxq8_cb', 'e_q4_idxq8_tk'), ('e_q4_idxf16_cb', 'e_q4_idxf16_tk'), ('e_q4_idxq4_cb', 'e_q4_idxq4_tk'),
             ('e_q8_idxq8', 'e_q8_idxf16'), ('e_q4_idxq8_cb_r2', 'e_q4_idxq8_cb')):
    pair(b, a)
print('\n## ratio rule (C9): q8 idx / f16 idx, same KV, same estimator')
pair('e_q4_idxq8_cb', 'e_q4_idxf16_cb', ratio=True)
pair('e_q4_idxq8_tk', 'e_q4_idxf16_tk', ratio=True)
pair('e_q4_idxq8_tk', 'e_q4_idxf16_tk', 'corr', 'corr', ratio=True)
pair('e_q4_idxq4_cb', 'e_q4_idxf16_cb', ratio=True)
pair('e_q8_idxq8', 'e_q8_idxf16', ratio=True)
