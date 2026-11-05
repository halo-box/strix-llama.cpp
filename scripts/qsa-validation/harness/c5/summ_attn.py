# Summarise c5_attn outputs: per file, mean over layers of the per-layer median (ms), ratio to V0 and to V1,
# per-layer min-max of V4/V0, worst correctness numbers
import sys, glob, re, os
files = sys.argv[2:] if len(sys.argv) > 2 else sorted(glob.glob(os.path.join(sys.argv[1], 'attn_*.out')))
for f in files:
    med = {}; maxref = 0; maxdiff = {}; pr_bad = 0; nl = 0
    for l in open(f):
        m = re.match(r'L(\d+) timing', l)
        if m:
            nl += 1
            for part in l.split('|')[1:]:
                t = part.split()
                name = t[0]; med.setdefault(name, []).append(float(t[2]))
        m = re.search(r'reference .* = ([0-9.e+-]+)', l)
        if m: maxref = max(maxref, float(m.group(1)))
        m = re.match(r'L\d+ (V\d+) .* vs V0: max_abs ([0-9.e+-]+) rel_l2 ([0-9.e+-]+) .* nan (\d+)', l)
        if m:
            a = maxdiff.setdefault(m.group(1), [0, 0, 0]); a[0] = max(a[0], float(m.group(2))); a[1] = max(a[1], float(m.group(3))); a[2] += int(m.group(4))
        m = re.search(r'PR merge vs CPU G4-ASC plan: (\d+)/', l)
        if m: pr_bad += int(m.group(1))
    if not med: print(os.path.basename(f), 'NO DATA'); continue
    mean = {k: sum(v) / len(v) for k, v in med.items()}
    v0 = mean['V0']; v1 = mean.get('V1', v0)
    r = ' | '.join(f"{k} {mean[k]:.3f} ({mean[k]/v0:.3f})" for k in sorted(mean, key=lambda x: int(x[1:])))
    rl = [b / a for a, b in zip(med['V0'], med.get('V4', med['V0']))]
    print(f"{os.path.basename(f)} layers={nl} PRmerge_bad={pr_bad} V0vsCPUref_max={maxref:.2e} | {r} | V4/V0 per-layer {min(rl):.3f}-{max(rl):.3f}")
    print('   maxdiff vs V0:', ' '.join(f"{k}:{v[0]:.1e}/{v[1]:.1e}/nan{v[2]}" for k, v in sorted(maxdiff.items(), key=lambda x: int(x[0][1:]))))
