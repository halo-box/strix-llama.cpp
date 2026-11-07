#!/usr/bin/env python3
"""The HIP prefer_dense table from the AT-W7PERF results (PLAN-v3 §16.5, the pre-registered 0.98 rule, no re-decision): at_table.py OP_SUMMARY [E2E_TABLE]
a (pair, n_q band) is dense if any op cell of the band is below 0.98, or if an e2e cell of the band has its CI upper bound below 0.98 (FAIL) (ub64 -> 33-127, ub512 -> 512-1023, ub1024 and ub2048 -> >=1024, tg -> 1)."""
import re, sys
B = ['1', '2-8', '9-32', '33-127', '128-511', '512-1023', '>=1024']
T = ['f16', 'bf16', 'q8_0', 'q4_0']
D = {(k, v): [k == 'f16' and v == 'f16'] * 7 for k in T for v in T}
why = {}
for l in open(sys.argv[1]):
    f = [x.strip() for x in l.split('|')]
    if len(f) == 9 and f[3] in B:
        k, v, b, G, w, cells, pd = f[1:8]
        if pd == 'dense':
            D[(k, v)][B.index(b)] = True; why[(k, v, b)] = f'op {w} ({cells})'
if len(sys.argv) > 2:
    for l in open(sys.argv[2]):
        f = [x.strip() for x in l.split('|')]
        if len(f) == 13 and f[11] == 'FAIL':
            m = re.match(r'(.+?_.+?)_(.+)$', f[1])
            p = f[1].split('_'); k = '_'.join(p[:len(p)//2]); v = '_'.join(p[len(p)//2:])
            ub, kind = int(f[4]), f[2]
            b = '1' if kind == 'tg' else {64: '33-127', 512: '512-1023', 1024: '>=1024', 2048: '>=1024'}[ub]
            D[(k, v)][B.index(b)] = True; why[(k, v, b)] = why.get((k, v, b), '') + f' e2e {f[8]}'
print('    static const bool dense[4][4][7] = {')
for k in T:
    print('        { ' + ', '.join('{ ' + ', '.join(str(int(x)) for x in D[(k, v)]) + ' }' for v in T) + ' },   // K ' + k)
print('    };')
for w in sorted(why): print('//', w, why[w])
