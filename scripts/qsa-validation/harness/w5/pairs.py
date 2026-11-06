#!/usr/bin/env python3
# W5: S^2 pair coverage per tier / attend class from the SA_REPORT lines of GGML_SA_TEST_REPORT=1 runs, the worst NMSE / band,
# and the SA_GQAREP results (GQA 4 / 8 against GQA 12 on replicated heads). usage: pairs.py LOG...
import re, sys, collections
S = "f16 bf16 q8_0 q4_0 f32 q4_1 q5_0 q5_1 iq4_nl".split()
FAST = set(S[:4])
for fn in sys.argv[1:]:
    cov = collections.defaultdict(set); worst = collections.defaultdict(float); rep = collections.defaultdict(list); bad = 0
    for line in open(fn, errors="replace"):
        if line.startswith("SA_REPORT backend="):
            m = re.search(r"case=SPARSE_ATTN\S*\(type_k=(\w+),type_v=(\w+).*?\) class=(\S*) band=(\S+) nmse=(\S+) ratio=(\S+)", line)
            if not m: continue
            tk, tv, cls, band, nmse, ratio = m.groups()
            c0 = cls.split(",")[0]; p = c0.split("/")
            if c0.startswith("sa_generic/G4"): k = "generic G4 " + p[2]
            elif c0.startswith("sa_generic/G1"): k = "generic G1"
            elif c0.startswith("sa_portable"): k = "portable " + p[1]
            elif c0.startswith("sa_attend/G4"): k = "fast G4"
            elif c0.startswith("sa_attend/G1"): k = "fast G1"
            else: k = "other " + c0
            cov[k].add((tk, tv)); worst[k] = max(worst[k], float(ratio))
            if float(ratio) > 1.0 or float(band) <= 0: bad += 1
        elif line.startswith("SA_GQAREP"):
            m = re.search(r"class=(\S*) row=(\S+) nmse=(\S+) bitwise=(\w+)", line)
            if m:
                c0 = m.group(1).split(",")[0]
                t = "generic" if "generic" in c0 else "portable" if "portable" in c0 else "fast"
                g = "G4" if "/G4" in c0 else "G1"
                rep[(t, g)].append((float(m.group(3)) / float(m.group(2)), float(m.group(3)), m.group(4)))
    gen_pairs = {(a, b) for a in S for b in S if not (a in FAST and b in FAST)}
    print("==", fn)
    allseen = set().union(*cov.values()) if cov else set()
    print("S^2 pairs seen %d / 81, with a generic member %d / %d; reports over band or without a band: %d" % (len(allseen), len(allseen & gen_pairs), len(gen_pairs), bad))
    for k in sorted(cov):
        print("  %-20s pairs %2d (with a generic member %2d) max nmse/band %.3f" % (k, len(cov[k]), len(cov[k] & gen_pairs), worst[k]))
    for k in sorted(rep):
        r = rep[k]
        print("  GQAREP %-8s %s cases %2d max nmse/row %.3f max nmse %.3e bitwise %d/%d" % (k[0], k[1], len(r), max(x[0] for x in r), max(x[1] for x in r), sum(x[2] == "yes" for x in r), len(r)))
