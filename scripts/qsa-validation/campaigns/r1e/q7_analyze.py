# q7_analyze.py A B : per-prompt paired acceptance B - A (C2 jsonl), mean, SD, CI; prompt count for a 95% CI half-width <= 1 pp,
# and for 80% power that a build with true diff 0 passes "95% one-sided LCB >= -1 pp"
import json, sys, math
from statistics import mean, stdev
def tq(p, df):  # Student t quantile by bisection on the regularized incomplete beta (no scipy)
    def cdf(t):
        x = df / (df + t * t)
        # continued fraction for I_x(a,b)
        a, b = df / 2, 0.5
        def betacf(a, b, x):
            qab, qap, qam = a + b, a + 1, a - 1; c, d = 1.0, 1 - qab * x / qap; d = 1 / d; h = d
            for m in range(1, 200):
                m2 = 2 * m; aa = m * (b - m) * x / ((qam + m2) * (a + m2)); d = 1 / (1 + aa * d); c = 1 + aa / c; h *= d * c
                aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); d = 1 / (1 + aa * d); c = 1 + aa / c; h *= d * c
            return h
        bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
        I = bt * betacf(a, b, x) / a
        return 1 - 0.5 * I if t > 0 else 0.5 * I
    lo, hi = 0.0, 50.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if cdf(mid) < p: lo = mid
        else: hi = mid
    return lo
A = {r["req"]: r for r in map(json.loads, open(f"res_{sys.argv[1]}.jsonl"))}
B = {r["req"]: r for r in map(json.loads, open(f"res_{sys.argv[2]}.jsonl"))}
d = []; same = 0; rows = []
for k in A:
    ra = A[k]["draft_n_accepted"] / A[k]["draft_n"]; rb = B[k]["draft_n_accepted"] / B[k]["draft_n"]
    d.append(100 * (rb - ra)); same += A[k]["text"] == B[k]["text"]
    rows.append(f"{k}:{100*ra:.2f}->{100*rb:.2f}({100*(rb-ra):+.2f})")
n = len(d); m = mean(d); sd = stdev(d); se = sd / math.sqrt(n)
t2 = tq(0.975, n - 1); t1 = tq(0.95, n - 1)
print(f"{sys.argv[2]} - {sys.argv[1]}: n {n} mean {m:+.3f} pp SD {sd:.3f} pp SE {se:.3f} 95% CI [{m-t2*se:+.3f}, {m+t2*se:+.3f}] one-sided LCB95 {m-t1*se:+.3f}; identical text {same}/{n}")
print("   " + " ".join(rows))
nh = next(k for k in range(2, 100000) if tq(0.975, k - 1) * sd / math.sqrt(k) <= 1.0)
npw = next(k for k in range(2, 100000) if (tq(0.95, k - 1) + tq(0.80, k - 1)) * sd / math.sqrt(k) <= 1.0)
print(f"   prompts for 95% CI half-width <= 1 pp at SD {sd:.3f}: n >= {nh}; for 80% power (true diff 0 passes LCB95 >= -1 pp): n >= {npw}")
