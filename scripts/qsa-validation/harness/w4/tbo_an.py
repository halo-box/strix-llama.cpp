#!/usr/bin/env python3
# W4: per test-backend-ops G1 dump, the PR-verbatim decode NMSE vs O_CON (bands_dec BANDSUM PR-dec-<T> gpu_vs_contract),
# the planted-defect Dmin, and the new G1 kernel's NMSE vs the CPU op (test-backend-ops SA_REPORT of the same case);
# per census class of the new kernel (G1, split class of sa_g1_splits, n_sel bucket <= 17 | <= 257 | > 257) the PR p100,
# 2 x p100 and the max new / PR ratio.
# usage: tbo_an.py <tbo.out> <resdir> <test-backend-ops stdout with SA_REPORT>
import sys, re, os, collections
run, res, rep = sys.argv[1:4]
def g1_splits(nq, ns, nk):
    step = 64; chunk = step
    if nq > 8:
        target = max(1, (8 * ((ns + step - 1) // step) + nq - 1) // nq)
        chunk = ((ns + target - 1) // target + 15) // 16 * 16
    pr = (ns + chunk - 1) // chunk
    cap = ((min(ns, (nk + 3) // 4) + 3) // 4) * 4
    return max(1, min(pr, cap // 4, 65535))
def cls(s): return "1" if s <= 1 else "2-5" if s <= 5 else "6-16" if s <= 16 else "17-33" if s <= 33 else "34+"
def bucket(ns): return "<=17" if ns <= 17 else "<=257" if ns <= 257 else ">257"
mine = {}
for l in open(rep):
    m = re.search(r"^SA_REPORT backend=ROCm0 case=SPARSE_ATTN\((.*?)\) class=(.*?) band=\S+ nmse=(\S+)", l)
    if m: mine[m.group(1)] = (float(m.group(3)), m.group(2))
rows = []
for l in open(run):
    m = re.match(r"RUN (\S+) rc=(\d+) (.*)", l)
    if not m: continue
    b, rc, v = m.groups()
    t = open(os.path.join(res, b + ".out")).read()
    pr = re.search(r"BANDSUM \S+ PR-dec-\S+ \S+ gpu_vs_contract nmse_layer_max (\S+)", t)
    cs = re.search(r"^CASE \S+ n_q (\d+) ns (\d+) nk (\d+) blocks \S+ splits (\d+)", t, re.M)
    # planted defects that change the contract output (a defect that drops no cell of an empty / all -1 row is a no-op)
    dm = [float(x) for x in re.findall(r"^DEFECT \S+ L\d+ pair \S+ d\d cells_dropped \d+ rows \d+ nmse (\S+)", t, re.M) if float(x) > 0]
    d = dict(x.split("=", 1) for x in v.split(",") if "=" in x)
    nq, ns, nk = int(d["n_q"]), int(d["n_sel"]), int(d["n_kv"])
    sp = g1_splits(nq, ns, nk)
    my = mine.get(v, (float("nan"), "?"))
    rows.append(dict(b=b, v=v, nq=nq, ns=ns, nk=nk, sp=sp, prsp=int(cs.group(4)) if cs else -1, pr=float(pr.group(1)) if pr else float("nan"),
                     dmin=min(dm) if dm else float("nan"), mine=my[0], mcls=my[1]))
print("| dump | case | new splits (class) | PR splits | PR NMSE vs O_CON | new NMSE vs CPU op | new / PR | defect Dmin | census class (test) |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print("| %s | %s | %d (%s) | %d | %.4e | %.4e | %.3f | %.3e | %s |" % (r["b"], r["v"], r["sp"], cls(r["sp"]), r["prsp"], r["pr"], r["mine"], r["mine"] / r["pr"], r["dmin"], r["mcls"]))
print()
by = collections.defaultdict(list)
for r in rows: by[(cls(r["sp"]), bucket(r["ns"]))].append(r)
for k in sorted(by):
    rs = by[k]; p = max(x["pr"] for x in rs); a = max(rs, key=lambda x: x["pr"])
    print("CLASS G1 splits %-5s n_sel %-5s dumps %3d PR_p100 %.4e (set by %s) 2xp100 %.4e new_max %.4e max_new/PR %.3f min_Dmin %.3e min_k %.1f" %
          (k[0], k[1], len(rs), p, a["v"], 2 * p, max(x["mine"] for x in rs), max(x["mine"] / x["pr"] for x in rs), min(x["dmin"] for x in rs), min(x["dmin"] for x in rs) / p))
