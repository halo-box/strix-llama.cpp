# r1q8 analysis: per-chunk full-precision KLD (QKLD_CHUNK lines), paired differences, one-sided 95% LCB (t_{0.95,7}=1.894579)
import sys, re, math, glob, os
d = sys.argv[1]
T95 = 1.894579
arms = {}
for f in sorted(glob.glob(d + '/*.log')):
    tag = os.path.basename(f)[:-4]
    v = [float(m.group(1)) for m in re.finditer(r'QKLD_CHUNK \d+ n=\d+ kld=(\S+)', open(f).read())]
    st = [float(m.group(1)) for m in re.finditer(r'QKLD_CHUNK \d+ n=\d+ kld=\S+ same_top=(\S+)', open(f).read())]
    if v: arms[tag] = (v, st)
print('== per-chunk KLD (%.6e), mean, chunk SE, mean same-top')
for t, (v, st) in arms.items():
    n = len(v); m = sum(v)/n; sd = math.sqrt(sum((x-m)**2 for x in v)/(n-1)) if n > 1 else 0
    print(f'{t:12s} n={n} mean={m:.6e} chunkSE={sd/math.sqrt(n):.3e} sametop={sum(st)/n:.6f} | ' + ' '.join(f'{x:.6e}' for x in v))
def pair(a, b):
    if a not in arms or b not in arms: return None
    x, y = arms[a][0], arms[b][0]
    dd = [p - q for p, q in zip(x, y)]; n = len(dd); m = sum(dd)/n
    sd = math.sqrt(sum((z-m)**2 for z in dd)/(n-1)); se = sd/math.sqrt(n)
    return m, sd, se, (m/se if se > 0 else float('inf') if m != 0 else 0.0), m - T95*se, dd
print('\n== paired (A - B): mean, SD, SE, t, one-sided 95% LCB | per-chunk diffs')
P = [('q8_d1','q8_off'),('q8_d2','q8_off'),('q8_d3','q8_off'),('q8_d3_r2','q8_off_r2'),('q8_d3e4','q8_off'),
     ('f16_d1','f16_off'),('f16_d2','f16_off'),('f16_d3','f16_off'),
     ('q8_d1_pre','q8_off'),('q8_d2_pre','q8_off'),('q8_off_r2','q8_off'),('q8_d3_r2','q8_d3'),
     ('q8_ub2048','q8_off'),('f16_ub2048','f16_off'),('q8_off','f16_ub2048'),('q8_off','f16_off')]
for a, b in P:
    r = pair(a, b)
    if r is None: continue
    m, sd, se, t, lcb, dd = r
    print(f'{a:11s} - {b:11s} mean={m:+.6e} SD={sd:.3e} SE={se:.3e} t={t:9.2f} LCB95={lcb:+.6e} | ' + ' '.join(f'{z:+.3e}' for z in dd))
