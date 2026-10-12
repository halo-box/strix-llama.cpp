#!/usr/bin/env python3
"""Read the output directory of validate.sh and print the validation part of the sync PR, in markdown."""

import glob
import json
import os
import re
import sys

PASS, FAIL, INC = "PASS", "FAIL", "INCOMPLETE"
TOP_P_DROP = 0.5    # points below the reference
KLD_FACTOR = 2.0    # times the reference, plus KLD_ABS
KLD_ABS = 1e-4
PPL_RATIO = 0.01
MEM_REL = 0.05      # a memory rise above 5 % and MEM_ABS MiB needs an explanation in the PR
MEM_ABS = 256

out = sys.argv[1]


def read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return None


def worst(statuses):
    statuses = list(statuses)
    if FAIL in statuses:
        return FAIL
    if INC in statuses or not statuses:
        return INC
    return PASS


def new_fails(base, cand):
    b, c = read(base), read(cand)
    if b is None or c is None or not re.search(r"\b(OK|FAIL)\b", c):
        return INC, "not run"
    fb = {l.strip() for l in b.splitlines() if "FAIL" in l}
    fc = [l.strip() for l in c.splitlines() if "FAIL" in l and l.strip() not in fb]
    if fc:
        return FAIL, "%d new: %s" % (len(fc), "; ".join(fc[:3]))
    return PASS, "no new failure (%d known in base)" % len(fb)


def kld_metrics(path):
    t = read(path)
    if t is None:
        return None
    m = {}
    for key, rx in (("ratio", r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([0-9.]+)"),
                    ("kld", r"Mean\s+KLD:\s*([0-9.]+)"),
                    ("top", r"Same top p:\s*([0-9.]+)")):
        r = re.search(rx, t)
        if r is None:
            return None
        m[key] = float(r.group(1))
    return m


def kld(be, label):
    c = kld_metrics(os.path.join(out, be, "kld", label + "-cand.txt"))
    r = kld_metrics(os.path.join(out, be, "kld", label + "-ref.txt"))
    if c is None or r is None:
        return INC, None, None
    ok = (c["top"] >= r["top"] - TOP_P_DROP and c["kld"] <= KLD_FACTOR * r["kld"] + KLD_ABS
          and abs(c["ratio"] - 1.0) <= PPL_RATIO)
    return (PASS if ok else FAIL), c, r


def bench_rows(be, label):
    sides = {}
    for side in ("base", "cand"):
        rows = {}
        text = "".join(read(f) or "" for f in sorted(glob.glob(os.path.join(out, be, "bench", "%s-%s-*.log" % (label, side)))))
        for line in text.splitlines():
            if not line.startswith("{"):
                continue
            try:
                j = json.loads(line)
            except ValueError:
                continue
            key = (j.get("n_prompt"), j.get("n_gen"), j.get("n_depth", 0))
            rows.setdefault(key, []).append((j["avg_ts"], j["stddev_ts"]))
        sides[side] = rows
    res = []
    for key in sorted(set(sides["base"]) | set(sides["cand"]), key=lambda k: tuple(x or 0 for x in k)):
        b, c = sides["base"].get(key), sides["cand"].get(key)
        if not b or not c:
            res.append((key, None, None, None, INC))
            continue
        mb, mc = sum(x[0] for x in b) / len(b), sum(x[0] for x in c) / len(c)
        sb, sc = sum(x[1] for x in b) / len(b), sum(x[1] for x in c) / len(c)
        status = PASS if mc >= mb - (sb + sc) else FAIL
        res.append((key, mb, mc, 100.0 * (mc / mb - 1.0), status))
    return res


def bench_status(rows):
    return worst(r[4] for r in rows) if rows else INC


BREAKDOWN = re.compile(r"\|\s*-?\s*([^|]+?)\s*\|\s*(?:\d+\s*=\s*\d+\s*\+\s*\(\s*)?(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)")


def mem(paths):
    """Peak RSS and GPU memory (MiB) over the logs of one run, and the llama.cpp breakdown of the last one."""
    rss = gpu = None
    breakdown = {}
    for p in paths:
        t = read(p)
        if t is None:
            continue
        for key, rx in (("rss", r"PEAK_RSS_MIB=(-?\d+)"), ("gpu", r"PEAK_GPU_MIB=(-?\d+)")):
            for v in re.findall(rx, t):
                v = int(v)
                if key == "rss":
                    rss = v if rss is None else max(rss, v)
                else:
                    gpu = v if gpu is None else max(gpu, v)
        for line in t.splitlines():
            if "memory breakdown" in line or "|" not in line:
                continue
            m = BREAKDOWN.search(line)
            if m:
                breakdown[m.group(1)] = tuple(int(x) for x in m.group(2, 3, 4, 5))   # self, model, context, compute
    return rss, gpu, breakdown


def mem_row(name, base_paths, cand_paths):
    rb, gb, _ = mem(base_paths)
    rc, gc, bd = mem(cand_paths)
    if rc is None:
        return name, None, INC
    st = PASS
    notes = []
    for b, c in ((rb, rc), (gb, gc)):
        if b is not None and c is not None and c > b * (1 + MEM_REL) and c - b > MEM_ABS:
            st = INC
    fmt = lambda b, c: "-" if c is None else ("%d -> %d MiB" % (b, c) if b is not None else "%d MiB" % c)
    dev = "; ".join("%s %d = %d + %d + %d" % ((k,) + v) for k, v in bd.items())
    return name, (fmt(rb, rc), fmt(gb, gc), dev or "-"), st


def test_name(key):
    p, g, d = key
    name = ("pp%d" % p) if p else ("tg%d" % g)
    return name + (" @ d%d" % d if d else "")


def server(be, arch):
    d = os.path.join(out, be, "server")
    notes, statuses = [], []
    calls = {}
    for side in ("base", "cand"):
        try:
            c = json.load(open(os.path.join(d, "%s-tool-%s.json" % (arch, side))))["choices"][0]
            tc = c["message"].get("tool_calls") or []
            okc = c["finish_reason"] == "tool_calls" and any(
                t["function"]["name"] == "get_weather" and isinstance(json.loads(t["function"]["arguments"]), dict) for t in tc)
            calls[side] = okc
        except Exception:
            calls[side] = None
    if calls.get("cand") is None:
        statuses.append(INC)
        notes.append("tool call: not run")
    else:
        statuses.append(PASS if calls["cand"] else FAIL)
        notes.append("tool call: %s (base %s)" % ("ok" if calls["cand"] else "BROKEN", "ok" if calls.get("base") else "broken"))
    texts, rates = {}, {}
    for side in ("base", "cand"):
        try:
            r = json.load(open(os.path.join(d, "%s-decode-%s.json" % (arch, side))))
            texts[side] = r["choices"][0]["message"]["content"]
            t = r.get("timings", {})
            if t.get("draft_n"):
                rates[side] = 100.0 * t["draft_n_accepted"] / t["draft_n"]
        except Exception:
            pass
    if len(texts) < 2:
        statuses.append(INC)
        notes.append("decode: not run")
    elif texts["base"] == texts["cand"]:
        statuses.append(PASS)
        notes.append("decode: identical")
    else:
        n = next((i for i, (a, b) in enumerate(zip(texts["base"], texts["cand"])) if a != b), min(len(texts["base"]), len(texts["cand"])))
        statuses.append(INC)
        notes.append("decode: differs from character %d, explain it in the PR" % n)
    spec = (read(os.path.join(d, "%s-cand.spec" % arch)) or "").strip()
    if len(rates) == 2:
        delta = rates["cand"] - rates["base"]
        statuses.append(PASS if delta >= -5.0 else FAIL)
        notes.append("MTP acceptance: %.1f%% (base %.1f%%)" % (rates["cand"], rates["base"]))
    else:
        statuses.append(INC)
        notes.append("MTP: not measured (%s)" % (spec or "server did not start"))
    return worst(statuses), "; ".join(notes)


def main():
    manifest = read(os.path.join(out, "manifest.txt")) or ""
    info = dict(l.split("=", 1) for l in manifest.splitlines() if "=" in l and not l.startswith("model="))
    models = [l.split("=", 1)[1].split() for l in manifest.splitlines() if l.startswith("model=")]
    contexts = {m[0]: (m[2] if len(m) > 2 else "short") for m in models}
    models = [m[0] for m in models]
    print("## Validation")
    print()
    print("Generated by `skills/upstream-sync/validate.sh` on a Strix Halo (gfx1151) machine%s." % ((" for " + info["pr"]) if info.get("pr") else ""))
    print("Base `%s`, candidate `%s`, small model `%s`." % (info.get("base", "?")[:9], info.get("cand", "?")[:9], info.get("small", "?")))
    print("Corpus sha256 in `corpus.sha256`. Thresholds: `Same top p` >= reference - %.1f points, `Mean KLD` <= %.0f x reference + %g, "
          "`PPL(Q)/PPL(base)` within 1 +/- %.2f; speed within the summed standard deviations of base and candidate. "
          "The reference is the base build against itself with a smaller ubatch." % (TOP_P_DROP, KLD_FACTOR, KLD_ABS, PPL_RATIO))

    for be in info.get("backends", "").split(","):
        if not be:
            continue
        print()
        rows, statuses = [], []
        for side in ("base", "cand"):
            s = (read(os.path.join(out, "logs", "build-%s-%s.status" % (side, be))) or "").strip()
            st = PASS if s == "ok" else (FAIL if s == "failed" else INC)
            statuses.append(st)
            rows.append(("build " + side, st, s or "not run"))
        for name, f in (("test-backend-ops", "ops"), ("test-llama-archs (every arch)", "archs")):
            st, note = new_fails(os.path.join(out, be, f + "-base.txt"), os.path.join(out, be, f + "-cand.txt"))
            statuses.append(st)
            rows.append((name, st, note))
        sls = read(os.path.join(out, be, "sls-cand.txt"))
        st = INC if sls is None else (FAIL if "EXIT=" in sls else PASS)
        statuses.append(st)
        rows.append(("test-save-load-state (small Q4_0)", st, "" if st != FAIL else "exit code != 0"))

        quant_rows = []
        for f in sorted(glob.glob(os.path.join(out, be, "kld", "small-*-cand.txt"))):
            q = os.path.basename(f)[len("small-"):-len("-cand.txt")]
            ks, c, r = kld(be, "small-" + q)
            br = bench_rows(be, "small-" + q)
            bs = bench_status(br)
            sha = (read(os.path.join(out, "quant", q + ".sha")) or "").split()
            same = "-" if len(sha) != 2 else ("same" if sha[0] == sha[1] else "CHANGED")
            st = worst([ks, bs])
            statuses.append(st)
            d = {test_name(k): x for k, _, _, x, _ in br}
            quant_rows.append((q, same, c, r, d, st))

        model_rows = []
        for a in models:
            for n in ("prose", "code", "structured", "numeric") + (("long",) if contexts[a] == "long" else ()):
                ks, c, r = kld(be, "%s-%s" % (a, n))
                statuses.append(ks)
                model_rows.append((a, n, c, r, ks))
        bench_tables = []
        for a in models:
            br = bench_rows(be, a + "-pp") + bench_rows(be, a + "-tg")
            statuses.append(bench_status(br))
            bench_tables.append((a, br))
        server_rows = []
        for a in models:
            st, note = server(be, a)
            statuses.append(st)
            server_rows.append((a, st, note))

        k = lambda *x: os.path.join(out, be, *x)
        mem_rows = [mem_row("test-backend-ops", [k("ops-base.txt")], [k("ops-cand.txt")]),
                    mem_row("test-llama-archs", [k("archs-base.txt")], [k("archs-cand.txt")]),
                    mem_row("test-save-load-state", [k("sls-base.txt")], [k("sls-cand.txt")])]
        for f in sorted(glob.glob(k("kld", "*-cand.txt"))):
            label = os.path.basename(f)[:-len("-cand.txt")]
            mem_rows.append(mem_row("perplexity " + label, [k("kld", label + "-base.txt")], [f]))
        labels = sorted({os.path.basename(f).rsplit("-", 2)[0] for f in glob.glob(k("bench", "*-cand-*.log"))})
        for label in labels:
            mem_rows.append(mem_row("llama-bench " + label, glob.glob(k("bench", label + "-base-*.log")), glob.glob(k("bench", label + "-cand-*.log"))))
        for a in models:
            mem_rows.append(mem_row("llama-server " + a, [k("server", a + "-base.log")], [k("server", a + "-cand.log")]))
        statuses += [r[2] for r in mem_rows]

        status = {PASS: "TARGET PASS", FAIL: "FAIL", INC: "INCOMPLETE"}[worst(statuses)]
        print("### %s: %s" % (be, status))
        print()
        print("| Check | Result | Notes |")
        print("|---|---|---|")
        for name, st, note in rows:
            print("| %s | %s | %s |" % (name, st, note))
        if quant_rows:
            print()
            print("Qwen3.5-0.8B-Base, every quant type (KLD on `all.txt` at `-c 512`, `llama-bench` pp512 / tg128):")
            print()
            print("| Quant | Quantizer output | Same top p (ref) | Mean KLD (ref) | PPL ratio | pp512 | tg128 | Status |")
            print("|---|---|---|---|---|---|---|---|")
            for q, same, c, r, d, st in quant_rows:
                kl = ("%.3f%% (%.3f%%)" % (c["top"], r["top"]), "%.6f (%.6f)" % (c["kld"], r["kld"]), "%.4f" % c["ratio"]) if c and r else ("-", "-", "-")
                sp = ["%+.1f%%" % d[k] if d.get(k) is not None else "-" for k in ("pp512", "tg128")]
                print("| %s | %s | %s | %s | %s | %s | %s | %s |" % (q, same, kl[0], kl[1], kl[2], sp[0], sp[1], st))
        if model_rows:
            print()
            print("Real models, logits against the base build:")
            print()
            print("| Model | Text | Same top p (ref) | Mean KLD (ref) | PPL ratio | Status |")
            print("|---|---|---|---|---|---|")
            for a, n, c, r, st in model_rows:
                if c and r:
                    print("| %s | %s | %.3f%% (%.3f%%) | %.6f (%.6f) | %.4f | %s |" % (a, n, c["top"], r["top"], c["kld"], r["kld"], c["ratio"], st))
                else:
                    print("| %s | %s | - | - | - | %s |" % (a, n, st))
        for a, br in bench_tables:
            print()
            print("%s, `llama-bench` t/s (mean of two alternating rounds per side):" % a)
            print()
            print("| Test | Base | Candidate | Change | Status |")
            print("|---|---|---|---|---|")
            for key, mb, mc, dp, st in br:
                if mb is None:
                    print("| %s | - | - | - | %s |" % (test_name(key), st))
                else:
                    print("| %s | %.2f | %.2f | %+.1f%% | %s |" % (test_name(key), mb, mc, dp, st))
        if server_rows:
            print()
            for a, st, note in server_rows:
                print("- %s server: %s. %s" % (a, st, note))
        print()
        print("Memory, base -> candidate: peak RSS of the process, peak GPU memory (amdgpu VRAM + GTT above the level before the run), "
              "and the llama.cpp breakdown of the candidate per device in MiB (self = model + context + compute). "
              "A rise above %d%% and %d MiB is INCOMPLETE until the PR explains it." % (MEM_REL * 100, MEM_ABS))
        print()
        print("| Run | Peak RSS | Peak GPU | Breakdown | Status |")
        print("|---|---|---|---|---|")
        for name, v, st in mem_rows:
            print("| %s | %s | %s | %s | %s |" % ((name,) + (v or ("-", "-", "-")) + (st,)))
        if "CHANGED" in [r[1] for r in quant_rows]:
            print()
            print("Quantizer output CHANGED for some types: upstream changed the quantization. Not a failure, but say so in the PR.")


main()
