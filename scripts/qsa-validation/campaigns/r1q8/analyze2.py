# r1q8 direct-form analysis: KLD against the clean q8_0 pinned output (logs2); paired defect - benign arms
import re, math, glob, os, sys
d = sys.argv[1]; T95 = 1.894579
A = {}
for f in glob.glob(d + '/*.log'):
    v = [float(x) for x in re.findall(r'QKLD_CHUNK \d+ n=\d+ kld=(\S+)', open(f).read())]
    if v: A[os.path.basename(f)[:-4]] = v
for t in sorted(A):
    v = A[t]; m = sum(v)/8; sd = math.sqrt(sum((x-m)**2 for x in v)/7)
    print(f'{t:11s} mean={m:.6e} chunkSE={sd/math.sqrt(8):.3e} min={min(v):.3e} max={max(v):.3e}')
print()
for a, b in [('q8_d1','q8_ub2048'),('q8_d2','q8_ub2048'),('q8_d3','q8_ub2048'),('q8_d3','q8_d3e7'),('q8_d3e2','q8_d3e7'),('q8_d2','q8_d3e7'),('q8_d1','q8_d3e7')]:
    if a in A and b in A:
        dd = [x-y for x, y in zip(A[a], A[b])]; m = sum(dd)/8; sd = math.sqrt(sum((z-m)**2 for z in dd)/7); se = sd/math.sqrt(8)
        print(f'{a} - {b}: mean={m:+.6e} SE={se:.3e} t={m/se:.2f} LCB95={m-T95*se:+.6e} npos={sum(z>0 for z in dd)}/8 | ' + ' '.join(f'{z:+.2e}' for z in dd))
