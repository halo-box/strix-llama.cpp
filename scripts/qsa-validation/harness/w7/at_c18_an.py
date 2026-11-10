#!/usr/bin/env python3
"""AT-C18 verdict: |estimate - real| <= 1 MiB per backend, main and MTP; every guard line of the probe contexts with inspected >= required
(plan + attend per sparse layer, attend per shortcut layer; 0 for a cache-only graph, which has no attention node); -nkvo pins.
usage: at_c18_an.py DIR"""
import json, re, sys, glob
d = sys.argv[1]
P = json.load(open(d + '/parsed.json'))
fail = 0
print('| run | ctx | backend | probe MiB | real MiB | diff | verdict |\n|---|---|---|---|---|---|---|')
for tag in sorted(P):
    R = P[tag]['reserves']
    by = {}
    for n, v in R: by.setdefault(n, v)
    for ctx, a, b in (('main', 'probe_main', 'real_main'), ('mtp', 'probe_mtp', 'real_mtp')):
        if a not in by or b not in by:
            print(f'| {tag} | {ctx} | | missing {a if a not in by else b} | | | MISSING |'); fail += 1; continue
        for be in sorted(by[a]):
            x, y = by[a][be], by[b].get(be)
            if y is None: continue
            ok = abs(x - y) <= 1.0
            fail += not ok
            print(f'| {tag} | {ctx} | {be} | {x:.2f} | {y:.2f} | {x-y:+.2f} | {"PASS" if ok else "FAIL"} |')
print()
for f in sorted(glob.glob(d + '/srv_*.log')):
    tag = re.sub(r'.*srv_(.*)\.log$', r'\1', f)
    phase = 'probe'; rows = []
    for line in open(f, errors='replace'):
        if 'successfully fit params' in line or 'failed to fit params' in line: phase = 'real'
        m = re.search(r'QSA ctx=(\S+) graph=(\S+) route=(\S+) .*backend=(\S+) plan=(\d+) attend=(\d+) cpu=(\d+).*guard=(\d+)/(\d+)', line)
        if m: rows.append((phase,) + m.groups())
    probe = [r for r in rows if r[0] == 'probe']
    # cache-only: no plan and no attend node. Logs before the fix print its required count clamped to 1 (guard=0/1)
    def guard_ok(r):
        cache_only = r[3] == 'cache-only' or (int(r[5]) == 0 and int(r[6]) == 0)
        return int(r[8]) >= (0 if cache_only else max(int(r[9]), 1))
    gd = [guard_ok(r) for r in probe]
    nk = 'nkvo' in tag
    pins = all((int(r[7]) == int(r[6]) or r[3] == 'host-kv-copy') for r in probe) if nk else True
    ok = bool(probe) and all(gd) and pins
    fail += not ok
    print(f"- {tag}: probe guard lines {len(probe)}, inspected >= required: {all(gd) if gd else None}; -nkvo pins (cpu = attend, or the named route): {pins}; {'PASS' if ok else 'FAIL'}; {[r[1:] for r in probe][:2]}")
print('\nAT-C18 ' + ('PASS' if fail == 0 else f'FAIL ({fail})'))
