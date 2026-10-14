#!/usr/bin/env python3
# Compare three runs of test-mmid-manual-frag (see the .cpp): default (manual fragments where the layout probe passed),
# GGML_VK_MMID_MANUAL_FRAG=0 (coopMatLoad) and GGML_VK_TEST_MF_SKEW=1 (positive control).
#   usage: test-mmid-manual-frag.py <default.log> <off.log> <skew.log>
# Exit 1 on any failed rule; every rule prints its counts, so a missing or empty run reads as a failure, never as a pass.
import re, sys

def load(path):
    cases = {}
    for line in open(path, errors='replace'):
        m = re.match(r'case (.*?) hash=([0-9a-f]+) all=(\d+) pf=(\d+) mf=(\d+) rega=(\d+) rega_mf=(\d+)', line)
        if m:
            cases[m.group(1)] = (m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6)), int(m.group(7)))
    return cases

d, o, s = (load(p) for p in sys.argv[1:4])
fail = 0
if not d or not o or not s or set(d) != set(o) or set(d) != set(s):
    print(f'FAIL: case sets differ or empty (default {len(d)}, off {len(o)}, skew {len(s)})')
    sys.exit(1)

n_eq = n_mf = n_skew = n_rega = n_skewable = 0
for c in d:
    hd, ad, pd, md, rd, rmd = d[c]
    ho, ao, po, mo, ro, rmo = o[c]
    hs, _, _, ms, _, _ = s[c]
    eq = hd == ho
    n_eq += eq
    if not eq:
        print(f'FAIL byte-equal: {c}: default {hd} vs off {ho}'); fail += 1
    if mo != 0:
        print(f'FAIL off run used _mf pipelines: {c}: mf={mo}'); fail += 1
    if pd and md != pd:
        print(f'FAIL engagement: {c}: mf {md} of pf {pd} (all {ad})'); fail += 1
    if rd and rmd != rd:
        print(f'FAIL REG_A engagement: {c}: REG_A on _mf {rmd} of {rd}'); fail += 1
    if rmo:
        print(f'FAIL off run used REG_A _mf twins: {c}: {rmo}'); fail += 1
    if 'E320' in c and rd == 0:
        print(f'FAIL REG_A shape did not run REG_A: {c}'); fail += 1
    n_rega += rd > 0
    if md:
        n_mf += 1
    # the skew spec constant is set on every _mf pipeline (REG_A included), so every case that ran one must show it
    if md > 0:
        n_skewable += 1
        if hs == ho:
            print(f'FAIL positive control: {c}: skew hash equals off hash (the comparison could not see a wrong fragment)'); fail += 1
        else:
            n_skew += 1
    print(f'{"ok  " if eq else "FAIL"} {c}: default {hd} off {ho} skew {hs} | all {ad} pf {pd} mf {md} rega {rd} rega_mf {rmd} | off mf {mo} rega_mf {rmo}')

print(f'summary: {len(d)} cases, byte-equal {n_eq}/{len(d)}, cases on _mf pipelines {n_mf}, cases on REG_A {n_rega}, skew differs {n_skew}/{n_skewable} (_mf cases, REG_A included), failures {fail}')
if n_skewable == 0:
    print('FAIL: no _mf case, the positive control never ran'); fail += 1
if n_mf == 0:
    print('FAIL: no case ran a _mf pipeline (probe failed or routing missed): nothing was tested'); fail += 1
sys.exit(1 if fail else 0)
