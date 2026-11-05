# kld2.py A.bin B.bin : C4 kld.py method (KLD(P_A || P_B) over fixed answer positions + answer log-prob), split per
# request: rows [0, n0) = seq 0, rows [n0, n) = seq 1 (header int32 n, n_vocab, n0)
import sys, numpy as np
def rd(p):
    h = np.fromfile(p, dtype=np.int32, count=3); n, v, n0 = int(h[0]), int(h[1]), int(h[2])
    t = np.fromfile(p, dtype=np.int32, count=n, offset=12)
    l = np.fromfile(p, dtype=np.float32, offset=12 + 4*n).reshape(n, v).astype(np.float64)
    l -= l.max(1, keepdims=True); l -= np.log(np.exp(l).sum(1, keepdims=True))
    return t, l, n0
ta, la, n0 = rd(sys.argv[1]); tb, lb, m0 = rd(sys.argv[2])
assert (ta == tb).all() and n0 == m0
k = (np.exp(la) * (la - lb)).sum(1)
out = []
for name, a, z in (("seq0", 0, n0), ("seq1", n0, len(ta)), ("all", 0, len(ta))):
    r = np.arange(a, z)
    lpa = la[r, ta[a:z]].sum(); lpb = lb[r, tb[a:z]].sum()
    top = (la[a:z].argmax(1) == lb[a:z].argmax(1)).mean()
    out.append(f"{name} n={z-a} KLD mean {k[a:z].mean():.3e} max {k[a:z].max():.3e} | logprob {lpa:.4f} -> {lpb:.4f} ({lpb-lpa:+.4f}) | top1 {top*100:.1f}%")
print(" || ".join(out))
