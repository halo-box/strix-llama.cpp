# r1av_pair.py LOGDIR A B [A B ...]: paired per-chunk KLD difference B - A from QKLD_CHUNK lines (all / lo / hi)
import sys, re, math, os
def load(path):
    d = {}
    for line in open(path, errors='replace'):
        m = re.search(r'QKLD_CHUNK (\d+) all=(\S+) lo=(\S+) nlo=(\d+) hi=(\S+) nhi=(\d+)', line)
        if m: d[int(m.group(1))] = dict(all=float(m.group(2)), lo=float(m.group(3)), hi=float(m.group(5)), nlo=int(m.group(4)), nhi=int(m.group(6)))
    return d
L = sys.argv[1]; a = sys.argv[2:]
for i in range(0, len(a), 2):
    A, B = load(os.path.join(L, a[i] + '.log')), load(os.path.join(L, a[i+1] + '.log'))
    ks = sorted(set(A) & set(B))
    out = [f"{a[i+1]} - {a[i]} (n={len(ks)})"]
    for r in ('lo', 'hi', 'all'):
        xa = [A[k][r] for k in ks]; xb = [B[k][r] for k in ks]
        d = [y - x for x, y in zip(xa, xb)]
        n = len(d)
        if n == 0: continue
        mu = sum(d)/n; sd = math.sqrt(sum((x-mu)**2 for x in d)/(n-1)) if n > 1 else float('nan')
        se = sd/math.sqrt(n) if n > 1 else float('nan')
        ma = sum(xa)/n; mb = sum(xb)/n
        verdict = 'PASS' if mu <= 3*se or (se == 0 and mu <= 0) else 'FAIL'
        out.append(f"  {r:3s}: A={ma:.4e} B={mb:.4e} diff={mu:+.4e} SEp={se:.3e} diff/SEp={mu/se if se else float('nan'):+.2f} [{verdict} at 3SEp] max|d|={max(abs(x) for x in d):.3e} per-chunk d: " + " ".join(f"{x:+.2e}" for x in d))
    print("\n".join(out))
