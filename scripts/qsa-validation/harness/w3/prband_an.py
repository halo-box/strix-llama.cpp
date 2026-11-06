#!/usr/bin/env python3
# W3 prband analyzer: per dump, the PR-verbatim G4 NMSE vs O_CON (bands_attn BANDSUM gpu_vs_contract, PR-V0) and the
# new kernel's NMSE vs the CPU op (test-backend-ops SA_REPORT), the TBO bitwise comparison, and per n_sel bucket the
# p100 of the PR kernel and the band 2 x p100 (PLAN-v3 §14.2 rule: measured on PR-verbatim kernels).
# usage: prband_an.py <run.out> <resdir> <test-backend-ops stdout with SA_REPORT>
import sys, re, os, collections
run, res, rep = sys.argv[1:4]
mine = {}
for l in open(rep):
    m = re.search(r"^SA_REPORT backend=ROCm0 case=SPARSE_ATTN\((.*)\) class=\S* band=\S+ nmse=(\S+)", l)
    if m: mine[m.group(1)] = float(m.group(2))
rows = []
for l in open(run):
    m = re.match(r"RUN (\S+) rc=(\d+) (.*)", l)
    if not m: continue
    b, rc, v = m.groups()
    txt = open(os.path.join(res, b + ".out")).read()
    pr = re.search(r"BANDSUM \S+ PR-V0-\S+ \S+ gpu_vs_contract nmse_layer_max (\S+)", txt)
    vd = re.search(r"BANDSUM \S+ PR-VD-\S+ \S+ gpu_vs_contract nmse_layer_max (\S+)", txt)
    bw = re.search(r"TBO_BITWISE \S+ PR-V0-\S+ words (\d+) words_differ_vs_tbo_rocm0 (\d+) nmse (\S+) maxabs (\S+)", txt)
    cpu = re.search(r"TBO_CPU .* tbo_cpu_vs_contract rows \d+ nmse (\S+)", txt)
    key = v
    d = dict(x.split("=", 1) for x in v.split(",") if "=" in x)
    rows.append(dict(b=b, v=v, nsel=int(d["n_sel"]), rows=d["rows"], pr=float(pr.group(1)) if pr else None,
                     vd=float(vd.group(1)) if vd else None, words=int(bw.group(1)) if bw else 0,
                     diff=int(bw.group(2)) if bw else -1, bwn=float(bw.group(3)) if bw else -1,
                     cpu=float(cpu.group(1)) if cpu else -1, mine=mine.get(key, -1)))
print("| dump | case | PR-V0 nmse vs O_CON | PR-VD | new kernel nmse vs CPU op | new / PR | words differ new vs PR (nmse) | CPU op vs O_CON |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    ratio = r["mine"] / r["pr"] if r["pr"] else float("nan")
    print("| %s | %s | %.4e | %.4e | %.4e | %.3f | %d / %d (%.2e) | %.1e |" % (r["b"], r["v"], r["pr"], r["vd"], r["mine"], ratio, r["diff"], r["words"], r["bwn"], r["cpu"]))
print()
by = collections.defaultdict(list)
for r in rows: by["257" if r["nsel"] <= 257 else "2051"].append(r)
for k in sorted(by):
    p = max(x["pr"] for x in by[k]); m = max(x["mine"] for x in by[k])
    arg = max(by[k], key=lambda x: x["pr"])
    print("BUCKET n_sel<=257" if k == "257" else "BUCKET n_sel>257", "dumps %d PR_p100 %.4e (set by %s) band_2xp100 %.4e new_max %.4e max_new/PR %.3f" %
          (len(by[k]), p, arg["v"], 2 * p, m, max(x["mine"] / x["pr"] for x in by[k] if x["pr"])))
print("VD == V0 everywhere:", all(r["pr"] == r["vd"] for r in rows), " CPU op == O_CON everywhere:", all(r["cpu"] == 0 for r in rows))
