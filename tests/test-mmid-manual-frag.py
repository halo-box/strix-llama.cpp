#!/usr/bin/env python3
# Compare three runs of test-mmid-manual-frag (see the .cpp): default (manual fragments where the layout probe passed),
# GGML_VK_MMID_MANUAL_FRAG=0 (coopMatLoad) and GGML_VK_TEST_MF_SKEW=1 (positive control).
#   usage: test-mmid-manual-frag.py <default.log> <off.log> <skew.log>
# Exit 1 on any failed rule; every rule prints its counts, so a missing or empty run reads as a failure, never as a pass.
import re, sys

def load(path):
    cases = {}
    for line in open(path, errors='replace'):
        m = re.match(r'case (.*?) hash=([0-9a-f]+) all=(\d+) pf=(\d+) mf=(\d+)', line)
        if m:
            cases[m.group(1)] = (m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)))
    return cases

d, o, s = (load(p) for p in sys.argv[1:4])
fail = 0
if not d or not o or not s or set(d) != set(o) or set(d) != set(s):
    print(f'FAIL: case sets differ or empty (default {len(d)}, off {len(o)}, skew {len(s)})')
    sys.exit(1)

n_eq = n_mf = n_skew = 0
for c in d:
    hd, ad, pd, md = d[c]
    ho, ao, po, mo = o[c]
    hs, _, _, ms = s[c]
    eq = hd == ho
    n_eq += eq
    if not eq:
        print(f'FAIL byte-equal: {c}: default {hd} vs off {ho}'); fail += 1
    if mo != 0:
        print(f'FAIL off run used _mf pipelines: {c}: mf={mo}'); fail += 1
    if pd and md != pd:
        print(f'FAIL engagement: {c}: mf {md} of pf {pd} (all {ad})'); fail += 1
    if md:
        n_mf += 1
        if hs == ho:
            print(f'FAIL positive control: {c}: skew hash equals off hash (the comparison could not see a wrong fragment)'); fail += 1
        else:
            n_skew += 1
    print(f'{"ok  " if eq else "FAIL"} {c}: default {hd} off {ho} skew {hs} | all {ad} pf {pd} mf {md} | off mf {mo}')

print(f'summary: {len(d)} cases, byte-equal {n_eq}/{len(d)}, cases on _mf pipelines {n_mf}, skew differs {n_skew}/{n_mf}, failures {fail}')
if n_mf == 0:
    print('FAIL: no case ran a _mf pipeline (probe failed or routing missed): nothing was tested'); fail += 1
sys.exit(1 if fail else 0)
