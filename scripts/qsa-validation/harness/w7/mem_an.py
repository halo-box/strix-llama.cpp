#!/usr/bin/env python3
"""M1 (PLAN-v3 §16): every T <= new f16 + 1 MiB, per backend, main and MTP, per (c, ub); np2 and default-server rows reported. usage: mem_an.py DIR"""
import re, sys, glob, collections
d = sys.argv[1]
R = collections.defaultdict(dict)
for f in glob.glob(d + '/srv_*.log'):
    tag = re.sub(r'.*srv_(.*)\.log$', r'\1', f)
    res = []; cur = {}
    for line in open(f, errors='replace'):
        m = re.search(r'sched_reserve:\s+(\S+) compute buffer size =\s+([\d.]+) MiB', line)
        if m:
            cur[m.group(1)] = float(m.group(2))
            if 'ROCm_Host' in cur and 'ROCm0' in cur: res.append(cur); cur = {}
    R[tag] = res
print('| cell | T | main ROCm0 | main host | MTP ROCm0 | MTP host | vs f16 ROCm0 main / MTP | verdict |\n|---|---|---|---|---|---|---|---|')
fail = 0
for c in (32768, 131072):
    for u in (4096, 512):
        base = R.get(f'm1_f16_c{c}_u{u}')
        for t in 'f16 q8_0 q4_0 bf16 f32 q4_1 q5_0 q5_1 iq4_nl'.split():
            r = R.get(f'm1_{t}_c{c}_u{u}')
            if not r or len(r) < 2 or not base or len(base) < 2:
                print(f'| c{c} ub{u} | {t} | missing | | | | | MISSING |'); fail += 1; continue
            dm, dd = r[0]['ROCm0'] - base[0]['ROCm0'], r[1]['ROCm0'] - base[1]['ROCm0']
            hm, hd = r[0]['ROCm_Host'] - base[0]['ROCm_Host'], r[1]['ROCm_Host'] - base[1]['ROCm_Host']
            ok = dm <= 1 and dd <= 1
            fail += not ok
            print(f"| c{c} ub{u} | {t} | {r[0]['ROCm0']:.2f} | {r[0]['ROCm_Host']:.2f} | {r[1]['ROCm0']:.2f} | {r[1]['ROCm_Host']:.2f} | {dm:+.2f} / {dd:+.2f} (host {hm:+.2f} / {hd:+.2f}) | {'PASS' if ok else 'FAIL'} |")
print()
for k in sorted(R):
    if k.startswith('m1np2') or k.startswith('m1def'):
        print(f'- {k}: ' + '; '.join(f"ROCm0 {x['ROCm0']:.2f} host {x['ROCm_Host']:.2f}" for x in R[k]))
print('\nM1 ' + ('PASS' if fail == 0 else f'FAIL ({fail})'))
