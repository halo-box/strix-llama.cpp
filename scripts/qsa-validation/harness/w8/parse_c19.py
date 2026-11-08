#!/usr/bin/env python3
# copy of the R1-C parser (EV/r1c/parse2.py) for AT-C16a: usage parse_c19.py srv_<tag>.log
# R1-M-Lab C parser v2: one "graph" = a consecutive run of C19 sv lines with one (ctx, n_kv); ctx: il 48 = MTP.
# Per graph: n_tps = sum of strip nq of one layer; new=1 count and bytes per src; bias src = the unnamed leaf (not attn_inp_kq_mask).
# C19 buf: max per ctx pointer + backend, plus the n_kv of the last sv graph before each growth line.
import re, sys, collections
for f in sys.argv[1:]:
    print("=====", f)
    groups = []; cur = None; buf = collections.defaultdict(float); grow = []
    for ln in open(f, errors="replace"):
        if "C19" not in ln:
            if re.search(r"processing task, is_child|decoding image batch|release: |T sent", ln): print("   ", ln.strip()[:150])
            continue
        m = re.search(r"C19 sv il=(\d+) src=(\S*) first=(\d+) nq=(\d+) nkv=(\d+) new=(\d) noshare=(\d) bytes=(\d+)", ln)
        if m:
            il, src, first, nq, nkv, new, ns, b = m.groups(); il=int(il); nkv=int(nkv)
            key = ("mtp" if il == 48 else "main", nkv)
            if cur is None or cur["key"] != key:
                cur = {"key": key, "cnt": collections.Counter(), "bytes": collections.Counter(), "strips": set(), "il": il}; groups.append(cur)
            s = "kq_mask" if src == "attn_inp_kq_mask" else "bias"
            cur["cnt"][s + ("" if int(new) else "_reuse")] += 1
            if int(new): cur["bytes"][s] += int(b)
            cur["strips"].add((int(first), int(nq)))
            continue
        m = re.search(r"C19 buf ctx=(\S+) (\S+) ([\d.]+) MiB \(was ([\d.]+)\)", ln)
        if m:
            k = (m.group(1), m.group(2)); v = float(m.group(3)); buf[k] = max(buf[k], v)
            grow.append((k, float(m.group(4)), v, cur["key"] if cur else None)); cur = None
    for g in groups:
        ntps = sum(nq for _, nq in g["strips"])
        print("  sv %-5s nkv=%6d n_tps=%5d strips=%d %s MiB=%s" % (g["key"][0], g["key"][1], ntps, len(g["strips"]), dict(g["cnt"]), {k: round(v/2**20, 2) for k, v in g["bytes"].items()}))
    for k, was, v, gk in grow: print("  buf", k, was, "->", v, "after", gk)
    for k, v in sorted(buf.items()): print("  BUFMAX", k, v)
