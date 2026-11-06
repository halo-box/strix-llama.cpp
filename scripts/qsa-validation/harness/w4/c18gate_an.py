#!/usr/bin/env python3
# W4 C18 decode gates: new G1 attend vs the same-run C18 harness kernel (c18_dec_kernel with the shipped loaders: q8_0 D+S,
# q4_0 X+S, bf16 X+S; f16 D+S reported), mean of r1 / r2 (each the mean of 12 per-layer medians); gate: new <= C18 x 1.014
# (C18 decode noise max 1.4 %). Partial kernel alone (G1 vs C18: the C18 numbers are the partial kernel) and with the
# split merge (G1M vs C18M = C18 + the PR's qsa_decode_merge).
# usage: c18gate_an.py <dir>
import sys, re
d = sys.argv[1]
def sums(f):
    r = {}
    for l in open(f):
        m = re.match(r"SUM (\S+)\s+(\S+)", l)
        if m: r[m.group(1)] = float(m.group(2))
    return r
ok_all = True
for fx in ("wikiL_p131072", "randL_p131072"):
    for nq in (512, 1):
        s = [sums("%s/dec%d_%s_%s.out" % (d, nq, fx, r)) for r in ("r1", "r2")]
        m = {k: (s[0][k] + s[1][k]) / 2 for k in s[0]}
        print("## %s n_q %d (ms, mean of r1/r2; r1/r2 spread in brackets)" % (fx, nq))
        for T in ("q8_0", "q4_0", "bf16", "f16"):
            for a, b in (("G1", "C18"), ("G1M", "C18M")):
                n, c = m["%s-%s" % (a, T)], m["%s-%s" % (b, T)]
                sp = abs(s[0]["%s-%s" % (a, T)] - s[1]["%s-%s" % (a, T)]) / n
                gated = T != "f16" and fx.startswith("wiki")
                ok = n <= c * 1.014
                if gated: ok_all &= ok
                print("GATE %-14s %-5s n_q %3d %-4s new %.4f C18 %.4f ratio %.3f limit %.4f [spread %.1f%%] %s" %
                      (fx, T, nq, a, n, c, n / c, c * 1.014, 100 * sp, ("PASS" if ok else "FAIL") if gated else "(reported)"))
        for T in ("f16", "q8_0"):
            print("PR    %-14s %-5s n_q %3d PR-dec %.4f new G1 %.4f ratio %.3f" % (fx, T, nq, m["PR-dec-%s" % T], m["G1-%s" % T], m["G1-%s" % T] / m["PR-dec-%s" % T]))
print("ALL_GATES", "PASS" if ok_all else "FAIL")
