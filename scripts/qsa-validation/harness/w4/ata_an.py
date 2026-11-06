#!/usr/bin/env python3
# AT-A analysis (PLAN-v3 §16.5): per (fixture, n_q, type) the G1 arm (plan G1 + G1M) against the better G4 arm
# (plan G4 + min(G4P incl. the pack, G4D)), mean of the two processes r1 / r2 (each = mean of 12 per-layer medians);
# run-to-run noise = |r1 - r2| / mean per (cell, arm), its p90 over every cell; G1 wins where
# t_G1 < t_G4 * (1 - 3 * p90). The fitted two-way rule: per V type and n_kv class (d4k 8192, 32768, 65536, d131k 135168),
# the largest tested n_q such that G1 wins at every tested n_q up to it on every fixture of that class. Also: the D8
# 64 * n_q direct-V switch at n_q 128 / 256 / 512 (G4P vs G4D), and the trimmed-layer row.
# usage: ata_an.py <ata dir> <ataplan dir>
import sys, re, glob, collections, math
A, P = sys.argv[1], sys.argv[2]
TYPES = ["f16", "bf16", "q8_0", "q4_0"]
FX = {"wikiS_p4096": 8192, "randS_p4096": 8192, "syn32768": 32768, "syn65536": 65536, "wikiL_p131072": 135168, "randL_p131072": 135168}
NQ = [2, 4, 8, 16, 32, 64, 96, 127, 128, 256, 512]

def sums(f):
    d = {}
    try:
        for l in open(f):
            m = re.match(r"SUM (\S+)\s+(\S+)", l)
            if m: d[m.group(1)] = float(m.group(2))
    except FileNotFoundError:
        pass
    return d

def plan_cost(fx, nq, r):
    f = "%s/%s_%d_%s.out" % (P, fx, nq, r)
    g = collections.defaultdict(list)
    try:
        for l in open(f):
            m = re.search(r"G=(\d) .* ms_med=(\S+)", l)
            if m and l.startswith("LAYER"): g[int(m.group(1))].append(float(m.group(2)))
    except FileNotFoundError:
        return None
    return {G: sum(v) / len(v) for G, v in g.items()} if g else None

noise = []
cells = {}
for fx in FX:
    for nq in NQ:
        s1, s2 = sums("%s/%s_%d_r1.out" % (A, fx, nq)), sums("%s/%s_%d_r2.out" % (A, fx, nq))
        if not s1 or not s2: continue
        pfx = "wikiS_p4096" if FX[fx] == 8192 else "wikiL_p131072"   # plan stage measured on the wiki rows of the depth
        p1, p2 = plan_cost(pfx, nq, "r1"), plan_cost(pfx, nq, "r2")
        if p1 is None or p2 is None: continue
        pc = {G: (p1[G] + p2[G]) / 2 for G in (1, 4)}
        noise += [abs(p1[G] - p2[G]) / ((p1[G] + p2[G]) / 2) for G in (1, 4)]
        for T in TYPES:
            arms = {}
            for a in ("G1M", "G4P", "G4D", "PACK"):
                k = "%s-%s" % (a, T)
                if k in s1 and k in s2:
                    arms[a] = (s1[k] + s2[k]) / 2
                    noise.append(abs(s1[k] - s2[k]) / arms[a])
            cells[(fx, nq, T)] = (arms, pc)
noise.sort()
p90 = noise[int(math.ceil(0.9 * len(noise))) - 1] if noise else float("nan")
thr = 1 - 3 * p90
print("# AT-A: %d cells, run-to-run noise p90 %.4f (%d samples), G1 wins where t_G1 < t_G4 x %.4f" % (len(cells), p90, len(noise), thr))
print()
print("| fixture | n_kv | n_q | type | plan G1 | G1M | t_G1 | plan G4 | G4P (incl. pack) | G4D | t_G4 | G4 arm | t_G1 / t_G4 | G1 wins |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
win = {}
for (fx, nq, T), (arms, pc) in sorted(cells.items(), key=lambda x: (TYPES.index(x[0][2]), FX[x[0][0]], x[0][0], x[0][1])):
    if not all(a in arms for a in ("G1M", "G4P", "G4D")): continue
    t1 = pc[1] + arms["G1M"]
    g4arm = "G4P" if arms["G4P"] <= arms["G4D"] else "G4D"
    t4 = pc[4] + arms[g4arm]
    w = t1 < t4 * thr
    win[(fx, nq, T)] = w
    print("| %s | %d | %d | %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %s | %.3f | %s |" %
          (fx, FX[fx], nq, T, pc[1], arms["G1M"], t1, pc[4], arms["G4P"], arms["G4D"], t4, g4arm, t1 / t4, "yes" if w else "no"))
print()
# fitted rule
classes = sorted(set(FX.values()))
rule = {}
for T in TYPES:
    for kv in classes:
        fxs = [f for f in FX if FX[f] == kv]
        best = 0
        for nq in NQ:
            ok = all(win.get((f, nq, T)) for f in fxs)
            if ok and all(all(win.get((f, n, T)) for f in fxs) for n in NQ if n < nq):
                best = nq
        rule[(T, kv)] = best
print("RULE (G1 iff n_q <= n_q*; n_q* = 0: G4 at every tested n_q)")
print("| V type | " + " | ".join("n_kv %d" % kv for kv in classes) + " |")
print("|---|" + "---|" * len(classes))
for T in TYPES:
    print("| %s | " % T + " | ".join(str(rule[(T, kv)]) for kv in classes) + " |")
print()
print("CPP { " + ", ".join("{ " + ", ".join(str(rule[(T, kv)]) for kv in classes) + " }" for T in TYPES) + " }  // [f16, bf16, q8_0, q4_0][n_kv " + ", ".join(map(str, classes)) + "]")
print()
# D8 re-derivation: G4P (packed + pack) vs G4D (direct) at n_q 128 / 256 / 512
print("D8: G4 packed + pack vs direct V (CAPPED rule: direct iff n_kv >= 64 * n_q)")
print("| fixture | n_kv | n_q | type | G4P | G4D | faster | CAPPED picks | CAPPED cost / best |")
print("|---|---|---|---|---|---|---|---|---|")
for (fx, nq, T), (arms, pc) in sorted(cells.items(), key=lambda x: (x[0][1], TYPES.index(x[0][2]), FX[x[0][0]])):
    if nq not in (128, 256, 512) or "G4P" not in arms: continue
    fast = "packed" if arms["G4P"] <= arms["G4D"] else "direct"
    rulep = "direct" if FX[fx] >= 64 * nq else "packed"
    cost = (arms["G4D"] if rulep == "direct" else arms["G4P"]) / min(arms["G4P"], arms["G4D"])
    print("| %s | %d | %d | %s | %.4f | %.4f | %s | %s | %.3f |" % (fx, FX[fx], nq, T, arms["G4P"], arms["G4D"], fast, rulep, cost))
print()
# trimmed layer: last 512 queries of layer 11, ub4096 pos0 d131k
s1, s2 = sums("%s/trim512_r1.out" % A), sums("%s/trim512_r2.out" % A)
p1, p2 = plan_cost("wikiL_p131072", 512, "r1"), plan_cost("wikiL_p131072", 512, "r2")
if s1 and s2 and p1 and p2:
    print("TRIM (layer 11, last 512 queries, wiki d131k): t = plan + attend, mean of r1 / r2")
    for T in TYPES:
        a = {k: (s1["%s-%s" % (k, T)] + s2["%s-%s" % (k, T)]) / 2 for k in ("G1M", "G4P", "G4D")}
        pc = {G: (p1[G] + p2[G]) / 2 for G in (1, 4)}
        t = {"G1": pc[1] + a["G1M"], "G4P": pc[4] + a["G4P"], "G4D": pc[4] + a["G4D"]}
        best = min(t, key=t.get)
        print("TRIM %s G1 %.4f G4P %.4f G4D %.4f best %s" % (T, t["G1"], t["G4P"], t["G4D"], best))
