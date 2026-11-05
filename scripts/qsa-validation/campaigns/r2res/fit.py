# R2-M-LAB-RES: fit the variant-alone reserve size V (bytes, from LAB_NC variant lines) per backend and context
import numpy as np
MiB = 1048576
# tag: (K = reserve n_kv per stream, T = variant tokens total, s = n_stream, V main host, V main dev, V mtp host, V mtp dev)
P = {
 'A'      : (262144,  512, 1, 432321600, 1829094912, 432035872, 1754294400),
 'B'      : (131072, 1024, 1, 456991808, 1799308800, 456699936, 1457049728),
 'C'      : (261888, 1024, 1, 860428352, 3274390016, 860136480, 2764686464),
 'D'      : ( 65536, 2048, 2, 509477440, 2204255232, 509173792, 1905893504),
 'D2'     : ( 65536, 1024, 2, 255664704, 1103053824, 255373344,  987287680),
 'D262f2' : (131072, 1024, 2, 458564160, 1977041920, 458272800, 1827721344),
 'D262f'  : (131072, 2048, 2, 913703488, 4135207936, 913399840, 3232866432),
}
names = list(P)
K = np.array([P[n][0] for n in names], float); T = np.array([P[n][1] for n in names], float); S = np.array([P[n][2] for n in names], float)
for col, lbl in [(3,'main host'), (5,'MTP host'), (4,'main ROCm0'), (6,'MTP ROCm0')]:
    V = np.array([P[n][col] for n in names], float)
    for form, X in [('a*K*T + b*K*s + c*T + d', np.stack([K*T, K*S, T, np.ones_like(K)], 1)),
                    ('a*K*T + c*T + d',         np.stack([K*T, T, np.ones_like(K)], 1))]:
        coef, *_ = np.linalg.lstsq(X, V, rcond=None)
        res = V - X @ coef
        print(f'{lbl:10s} {form:26s} coef', ' '.join(f'{c:.4f}' for c in coef), '| max|res| MiB %.2f' % (abs(res).max()/MiB),
              '| res MiB', ' '.join(f'{n}:{r/MiB:+.2f}' for n, r in zip(names, res)))
# host: excess over 3 B * K * T
for n in names:
    k, t, s, hm, dm, ht, dt = P[n]
    print(f'{n:7s} host main - 3KT = {(hm-3*k*t)/MiB:7.2f} MiB, MTP {(ht-3*k*t)/MiB:7.2f} MiB; dev main / (K*T) = {dm/(k*t):.3f} B, MTP {dt/(k*t):.3f} B')

# ---- added: S = 1 points (LAB_NC_TPS=512), host out-of-sample check, and the requested increment fit a + b*n_ctx*n_stream
P1 = {  # tag: (K, T, s, V main host, V main dev, V mtp host, V mtp dev)
 'E_s1'   : (131072, 512, 1, 229422144,  988661248, 229136416,  913860736),
 'C_s1'   : (261888, 512, 1, 431925312, 1827453440, 431639584, 1752652928),
 'D2_s1'  : ( 65536, 512, 2, 128758336,  665699328, 128473120,  477653120),
 'D262_s1': (131072, 512, 2, 230994496, 1170588672, 230709280,  881879168),
}
print('\nhost formula out of sample (MTP: 3KT + 12Ks + 51244T + 32):')
for n, (k, t, s, hm, dm, ht, dt) in {**P, **P1}.items():
    pred = 3*k*t + 12*k*s + 51244*t + 32
    print(f'  {n:8s} MTP host meas {ht} pred {pred} diff {ht-pred:+d} B; main - MTP = {hm-ht} B')
print('\nunified S=1 device (T=512): slope per cell, const')
for col, lbl in [(4, 'main'), (6, 'MTP')]:
    k1, k2 = P1['E_s1'][0], P1['C_s1'][0]; v1, v2 = P1['E_s1'][col], P1['C_s1'][col]
    sl = (v2 - v1)/(k2 - k1); c = v1 - sl*k1
    pa = c + sl*P['A'][0]
    print(f'  {lbl}: slope {sl:.3f} B/cell, const {c/MiB:.2f} MiB; predicts A (K 262144, T 512) {pa:.0f} vs meas {P["A"][col]} ({(P["A"][col]-pa)/MiB:+.3f} MiB)')
# increments (forced real - unforced real), MiB, per (ctx, backend): requested form a + b * n_ctx * n_stream at n_tps 1024 per stream
inc = {  # tag: (n_ctx (-c), n_stream, main dev, main host, mtp dev, mtp host)
 'B': (131072, 1, 106.07, 233.76, 0.00, 233.79), 'C': (261888, 1, 0.00, 416.71, 0.00, 416.79), 'D': (131072, 2, 1338.69, 434.42, 1076.42, 434.41),
}
print('\nrequested increment fit a + b*n_ctx*n_stream over B, C, D (n_tps 1024 per stream):')
for j, lbl in enumerate(['main ROCm0', 'main host', 'MTP ROCm0', 'MTP host']):
    x = np.array([inc[n][0]*inc[n][1] for n in 'BCD'], float); y = np.array([inc[n][2+j] for n in 'BCD'], float)
    X = np.stack([np.ones_like(x), x], 1); coef, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ coef
    print(f'  {lbl:10s} a {coef[0]:.2f} MiB, b {coef[1]*MiB:.1f} B/cell, residuals ' + ' '.join(f'{n}:{v:+.2f}' for n, v in zip('BCD', r)))
