#!/usr/bin/env python3
"""M2 (PLAN-v3 §16): device high-water by the shim. Every hipMalloc / hipFree of the process: live bytes over time. total = the high-water of all live bytes (weights, KV, compute buffers, pool);
pool = the high-water of the bytes allocated after the server said it is ready. new total <= W0 total + 1 MiB. usage: m2_an.py DIR"""
import glob, os, re, sys
d = sys.argv[1]
def scan(tag):
    ready = int(open(f'{d}/{tag}.ready').read()) if os.path.exists(f'{d}/{tag}.ready') else None
    live, sz = 0, {}
    pool = 0; pl = 0; hw = 0; hwp = 0; n = 0; late = {}
    # shim lines: kind t_ns dur a(size) ptr(hex). CLOCK_MONOTONIC vs the ready stamp (CLOCK_REALTIME): use the position of the first LPROC / the end of the init MALLOCs instead
    ev = [l.split() for l in open(f'{d}/{tag}.shim') if len(l.split()) >= 5]
    first = next((i for i, e in enumerate(ev) if e[0] == 'LPROC'), len(ev))
    for i, e in enumerate(ev):
        k = e[0]
        if k.startswith('MALLOC') or k == 'HOSTMALLOC':
            if k == 'HOSTMALLOC': continue
            s, p = int(e[3]), e[4]
            sz[p] = s; live += s; hw = max(hw, live); n += 1
            if i > first: late[p] = s; pl += s; hwp = max(hwp, pl)
        elif k == 'FREE':
            p = e[4]
            live -= sz.pop(p, 0)
            pl -= late.pop(p, 0)
    return hw, hwp, n
print('| cell | W0 total MiB | new total MiB | diff | W0 pool hw | new pool hw | verdict |\n|---|---|---|---|---|---|---|')
fail = 0
for f in sorted(glob.glob(d + '/m2_*_w0.shim')):
    t = os.path.basename(f)[:-8]
    a, b = scan(t + '_w0'), scan(t + '_new')
    diff = (b[0] - a[0]) / 2**20
    ok = diff <= 1.0
    fail += not ok
    print(f"| {t} | {a[0]/2**20:.2f} | {b[0]/2**20:.2f} | {diff:+.2f} | {a[1]/2**20:.2f} | {b[1]/2**20:.2f} | {'PASS' if ok else 'FAIL'} |")
print('\nM2 ' + ('PASS' if fail == 0 else f'FAIL ({fail})'))
