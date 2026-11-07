#!/usr/bin/env python3
"""W7 quality verdicts (PLAN-v3 §15, bands of §15.0 as frozen by W12). Formulas fixed here before the arms are scored.
usage: q_an.py LOGDIR > w7_quality.md
Per-chunk values from `QKLD_CHUNK <c> all= lo= nlo= hi= nhi=`; agreement from QPIN_QSA_CHUNK in TAG.stats. Paired statistic new - w0: mean, chunk SE."""
import math, os, re, sys, collections
L = sys.argv[1]
FAIL, NOTE = [], []
B_4096, B_512, B_512LO, B_64, B_32K, B_256 = 2.0e-4, 2.03e-4, 2.073e-4, 2.75e-4, 1.801e-4, 3.34e-4
DIRECT = {'f16_f16': 1.403e-4, 'q8_0_q8_0': 2.310e-4, 'q4_0_q4_0': 1.441e-3, 'bf16_bf16': 1.032e-4, 'q8_0_f16': 1.807e-4, 'f16_q8_0': 1.623e-4}
Q4 = {'f16_f16': 0.99735, 'f16k_q8idx': 0.99693, 'q8_0_q8_0': 0.996625, 'q4_0_q4_0': 0.985528, 'bf16_bf16': 0.997140, 'q8_0_f16': 0.996742, 'f16_q8_0': 0.997121}
Q4B = 1.178e-4
FLOOR_F16_2048 = 1.116e-4
T95 = {7: 1.894579, 15: 1.75305}

def kld(tag, field='all'):
    p = os.path.join(L, tag + '.log')
    if not os.path.exists(p): return None
    d = {}
    for line in open(p, errors='replace'):
        m = re.search(r'QKLD_CHUNK (\d+) all=(\S+) lo=(\S+) nlo=(\d+) hi=(\S+) nhi=(\d+)', line)
        if m: d[int(m.group(1))] = float({'all': m.group(2), 'lo': m.group(3), 'hi': m.group(5)}[field])
    return d or None
def hashes(tag, kind):
    p = os.path.join(L, tag + '.log')
    if not os.path.exists(p): return None
    return {int(a): b for a, b in re.findall(r'QKL_HASH %s (\d+) (\S+)' % kind, open(p, errors='replace').read())} or None
def summ(tag):
    p = os.path.join(L, tag + '.log')
    if not os.path.exists(p): return None
    x = [l.strip() for l in open(p, errors='replace') if l.startswith('QPIN_SUMMARY')]
    return x[-1] if x else None
def health(tag):
    s = summ(tag)
    if not s: return 'NO SUMMARY'
    m = dict(re.findall(r'(\w+)=(\d+)', s))
    return 'ok' if m.get('rows_absent') == '0' and m.get('moe_absent') == '0' else f"absent rows={m.get('rows_absent')} moe={m.get('moe_absent')}"
def st(x):
    n = len(x); mu = sum(x)/n
    sd = math.sqrt(sum((v-mu)**2 for v in x)/(n-1)) if n > 1 else 0.0
    return mu, sd/math.sqrt(n), n
def paired(b, a):
    if b is None or a is None: return None
    ks = sorted(set(a) & set(b))
    if not ks: return None
    d = [b[k]-a[k] for k in ks]
    mu, se, n = st(d)
    return dict(mean=mu, se=se, n=n, pos=sum(x > 0 for x in d), max=max(abs(x) for x in d))
def agree(tag):
    p = os.path.join(L, tag + '.stats')
    if not os.path.exists(p): return None
    r = {}
    for line in open(p):
        m = re.match(r'QPIN_QSA_CHUNK chunk=(\d+) src=(\S+) n=(\d+) overlap=(\S+)', line)
        if m: r.setdefault(int(m.group(1)), {})[m.group(2)] = float(m.group(4))
    return {c: s.get('cb', s.get('tk1')) for c, s in r.items()} or None
def row(name, p, band, extra=''):
    if p is None:
        print(f'| {name} | missing | | {band:.3e} | MISSING {extra} |'); FAIL.append(name + ' missing'); return
    ok = p['mean'] <= band
    if not ok: FAIL.append(name)
    print(f"| {name} | {p['mean']:+.3e} | {p['se']:.2e} | {band:.3e} | {'PASS' if ok else 'FAIL'} {extra} |")

print('# W7 quality gates (PLAN-v3 §15)\n')
print('## Run health and self-replay\n')
tags = sorted(f[:-4] for f in os.listdir(L) if f.endswith('.log'))
bad = [(t, health(t)) for t in tags if (t.startswith('new_') or t.startswith('w0_')) and not t.startswith(('new_s_', 'w0_s_')) and health(t) != 'ok']
print(f'- arms: {len(tags)}; arms with rows_absent or moe_absent != 0: {bad if bad else 0}')
h0, h1 = hashes('ref8t0_rec', 'base'), hashes('w0_a_self', 'kld')
if h0 and h1:
    same = sum(h0.get(c) == v for c, v in h1.items())
    print(f'- ref8t0 self-replay on W0-lab: QKL_HASH equal in {same}/{len(h1)} chunks')
    if same != len(h1): FAIL.append('ref8t0 self-replay')
print('\n## Q1 (new(T)_U - PR219(T)_U, c8192 x 8, pinned)\n')
print('| arm | mean diff | SE | band | verdict |\n|---|---|---|---|---|')
for p in ['f16_f16', 'q8_0_q8_0', 'q4_0_q4_0', 'bf16_bf16', 'q8_0_f16', 'f16_q8_0']:
    row(f'Q1-A {p} ub4096', paired(kld(f'new_a_{p}_u4096'), kld(f'w0_a_{p}_u4096')), B_4096)
for p in ['f16_f16', 'q8_0_q8_0', 'q4_0_q4_0']:
    row(f'Q1-B {p} ub512', paired(kld(f'new_a_{p}_u512'), kld(f'w0_a_{p}_u512')), B_512)
for p in ['f16_f16', 'q8_0_q8_0', 'q4_0_q4_0']:
    row(f'Q1-C {p} ub64', paired(kld(f'new_a_{p}_u64'), kld(f'w0_a_{p}_u64')), B_64)
    row(f'Q1-C {p} ub64 forced G1', paired(kld(f'new_a_{p}_u64g1'), kld(f'w0_a_{p}_u64')), B_64)
print('\n## Q1 direct form (KLD(new(T) || PR219(T) pinned), LAB_DEF=d3 eps 0 on both)\n')
print('| type | mean KLD | SE | band | verdict |\n|---|---|---|---|---|')
for p, b in DIRECT.items():
    k = kld(f'new_db_{p}')
    if k: mu, se, _ = st(list(k.values())); ok = mu <= b; print(f"| {p} | {mu:.3e} | {se:.2e} | {b:.3e} | {'PASS' if ok else 'FAIL'} |"); (ok or FAIL.append('direct ' + p))
    else: print(f'| {p} | missing | | {b:.3e} | MISSING |'); FAIL.append('direct ' + p + ' missing')
print('\n## Q1 direct form, stage 2 (§15 two-stage rule: the same band on chunks 8-15 of the c8192 x 16 reference)\n')
print('| type | mean KLD chunks 8-15 | SE | band | verdict |\n|---|---|---|---|---|')
for p, b in DIRECT.items():
    k = kld(f'new_hb_{p}')
    if k:
        v = [k[c] for c in sorted(k) if 8 <= c <= 15]
        mu, se, _ = st(v); ok = mu <= b
        print(f"| {p} | {mu:.3e} | {se:.2e} | {b:.3e} | {'PASS' if ok else 'FAIL'} |")
print('\n## Q2 floor context (ratio to the f16 ub2048 pinned floor)\n')
fl = kld('w0_a_f16_f16_u2048')
if fl:
    fm = st(list(fl.values()))[0]
    print(f'- f16 ub2048 pinned floor on W0-lab = {fm:.4e} (plan value {FLOOR_F16_2048:.3e})')
    for t in tags:
        if t.startswith('new_a_') and '_u' in t and 'a8' not in t:
            k = kld(t)
            if k: print(f'- {t}: {st(list(k.values()))[0]:.4e}, ratio {st(list(k.values()))[0]/fm:.3f}')
print('\n## Q3 (c32768 x 8, QKL_TAIL=4096)\n')
print('| arm | mean diff | SE | band | verdict |\n|---|---|---|---|---|')
for t in ['f16', 'q8_0', 'q4_0']:
    row(f'Q3-A {t} ub4096', paired(kld(f'new_p_{t}_u4096'), kld(f'w0_p_{t}_u4096')), B_32K)
    row(f'Q3-B {t} ub256', paired(kld(f'new_p_{t}_u256'), kld(f'w0_p_{t}_u256')), B_256)
print('\n## Q4 selection agreement (c8192 x 16)\n')
print('| type | new mean | w0 mean | new - w0 | bound -B_T | abs threshold | verdict |\n|---|---|---|---|---|---|---|')
for p, thr in Q4.items():
    a, b = agree(f'new_c_{p}'), agree(f'w0_c_{p}')
    if not a or not b: print(f'| {p} | missing | | | | | MISSING |'); FAIL.append('Q4 ' + p + ' missing'); continue
    ks = sorted(set(a) & set(b)); ma = sum(a[k] for k in ks)/len(ks); mb = sum(b[k] for k in ks)/len(ks)
    diff = ma - mb
    # PLAN-v3 Q4 states the bound per type as an absolute threshold (PR219_T - B_T with the per-type B_T of R1-E / AT-W12-BANDS):
    # that is the verdict. The uniform relative bound -1.178e-4 is printed as context; for f16 KV the PR219 arm is the reference
    # build itself (agreement 1.0 with f16 idx), so the relative form there compares new against the reference with itself.
    ok = ma >= thr
    rel = 'ok' if diff >= -Q4B else 'below'
    print(f"| {p} | {ma:.6f} | {mb:.6f} | {diff:+.2e} ({rel}) | {-Q4B:.3e} | {thr} (margin {ma - thr:+.1e}) | {'PASS' if ok else 'FAIL'} |")
    if not ok: FAIL.append('Q4 ' + p)
print('\n## Q8 planted defects (on - off, same type, same pins, c8192 x 8)\n')
print('| arm | type | on - off | SE | LCB95 | pos chunks | rule | verdict |\n|---|---|---|---|---|---|---|---|')
for t in ['f16', 'q8_0']:
    off, w0off = kld(f'new_a_{t}_{t}_u4096'), kld(f'w0_a_{t}_{t}_u4096')     # d1, d2 do not observe a node: the clean arm has the same splits
    off3, w0off3 = kld(f'new_a8_{t}_off'), kld(f'w0_a8_{t}_off')             # d3 observes attn_output-23: its off arm too (LAB_DEF=d3, eps 0)
    for d in ['d1', 'd2', 'd3']:
        p, w = paired(kld(f'new_a8_{t}_{d}'), off3 if d == 'd3' else off), paired(kld(f'w0_a8_{t}_{d}'), w0off3 if d == 'd3' else w0off)
        if not p: print(f'| new {d} | {t} | missing | | | | | MISSING |'); FAIL.append(f'Q8 {t} {d} missing'); continue
        lcb = p['mean'] - T95[p['n']-1] * p['se']
        if d == 'd1': ok, rule = lcb > 7.45e-5, 'LCB95 > 7.45e-5'
        elif d == 'd2':
            ok = p['pos'] >= 7 and (w is None or abs(p['mean'] - w['mean']) <= 1.86e-3)
            rule = f"sign >= 7/8, |new - PR219| <= 1.86e-3 (PR219 {w['mean'] if w else float('nan'):+.3e})"
        else: ok, rule = True, 'report only'
        print(f"| new {d} | {t} | {p['mean']:+.3e} | {p['se']:.2e} | {lcb:+.3e} | {p['pos']}/{p['n']} | {rule} | {'PASS' if ok else 'FAIL'} |")
        if not ok: FAIL.append(f'Q8 {t} {d}')
print('\n## Q11a (c8192 x 8, b/ub 512, QKL_TAIL=8191, split 2048)\n')
print('| type | part | mean diff | SE | band | verdict |\n|---|---|---|---|---|---|')
for p in DIRECT:
    for part, band in (('lo', B_512LO), ('hi', B_512), ('all', B_512)):
        q = paired(kld(f'new_b_{p}_u512', part), kld(f'w0_b_{p}_u512', part))
        if not q: print(f'| {p} | {part} | missing | | {band:.3e} | MISSING |'); FAIL.append(f'Q11a {p} {part} missing'); continue
        ok = q['mean'] <= band
        print(f"| {p} | {part} | {q['mean']:+.3e} | {q['se']:.2e} | {band:.3e} | {'PASS' if ok else 'FAIL'} |")
        if not ok: FAIL.append(f'Q11a {p} {part}')
print('\n## Q11b (c2048 x 16, MoE pins, dense f16 b/ub 2048 reference)\n')
print('| arm | mean diff | SE | band | verdict |\n|---|---|---|---|---|')
for p in DIRECT:
    row(f'Q11b {p} ub512', paired(kld(f'new_e_{p}_u512'), kld(f'w0_e_{p}_u512')), B_512)
for p in ['f16_f16', 'q8_0_q8_0']:
    q = paired(kld(f'new_e_{p}_u16'), kld(f'w0_e_{p}_u16'))
    if q: print(f"| Q11b {p} ub16 | {q['mean']:+.3e} | {q['se']:.2e} | UNPROVEN (F4) | report only |")
print('\n## Q11c (np2 split == np1, unpinned, KLD per chunk)\n')
for t in ['q8_0', 'q4_0']:
    r2, n1 = kld(f's_{t}_np2rerun'), kld(f's_{t}_np1')
    for name, k in (('np2 rerun', r2), ('np1', n1)):
        if k: 
            m = max(abs(v) for v in k.values())
            ok = m <= 1e-6
            print(f"- {t} {name}: max |chunk KLD| = {m:.3e} over {len(k)} chunks: {'PASS' if ok else 'FAIL'} (<= 1e-6: the uint16 base resolution, W12 F2)")
            if not ok: FAIL.append(f'Q11c {t} {name}')
        else: print(f'- {t} {name}: missing'); FAIL.append(f'Q11c {t} {name} missing')
print('\n## Q12 host KV (c8192 x 8, pinned, new(T, -nkvo) - PR219(T, -nkvo))\n')
print('| arm | mean diff | SE | band | rerun new | rerun w0 | verdict |\n|---|---|---|---|---|---|---|')
for t in ['f16', 'q8_0']:
    for u, band in ((4096, B_4096), (512, B_512)):
        q = paired(kld(f'new_n_{t}_u{u}'), kld(f'w0_n_{t}_u{u}'))
        rn, rw = paired(kld(f'new_n_{t}_u{u}_r2'), kld(f'new_n_{t}_u{u}')), paired(kld(f'w0_n_{t}_u{u}_r2'), kld(f'w0_n_{t}_u{u}'))
        if not q: print(f'| -nkvo {t} ub{u} | missing | | {band:.3e} | | | MISSING |'); FAIL.append(f'Q12 {t} {u} missing'); continue
        ok = q['mean'] <= band and rn and rw and rn['max'] <= 1e-6 and rw['max'] <= 1e-6
        print(f"| -nkvo {t} ub{u} | {q['mean']:+.3e} | {q['se']:.2e} | {band:.3e} | {rn['max'] if rn else float('nan'):.2e} | {rw['max'] if rw else float('nan'):.2e} | {'PASS' if ok else 'FAIL'} |")
        if not ok: FAIL.append(f'Q12 {t} {u}')
for t in ['f16', 'q8_0']:
    q = paired(kld(f'new_z_{t}_u4096'), kld(f'w0_z_{t}_u4096'))
    if q: 
        ok = q['mean'] <= B_4096
        print(f"| -ngl 0 {t} ub4096 ({q['n']} chunks) | {q['mean']:+.3e} | {q['se']:.2e} | {B_4096:.3e} | | | {'PASS' if ok else 'FAIL'} |")
        if not ok: FAIL.append(f'Q12 ngl0 {t}')
print('\n## Overall\n')
print('FAILED: ' + ', '.join(FAIL) if FAIL else 'ALL PASS')
