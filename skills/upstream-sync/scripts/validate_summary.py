#!/usr/bin/env python3
"""Read the output directory of validate.sh and print the validation part of the sync PR, in markdown."""

import glob
import json
import math
import os
import re
import sys

PASS, FAIL, INC = "PASS", "FAIL", "INCOMPLETE"
TOP_P_DROP = 0.5    # points below the reference
KLD_FACTOR = 2.0    # times the reference, plus KLD_ABS: above it is a FAIL; between the reference and it, INCOMPLETE
KLD_ABS = 1e-4
PPL_RATIO = 0.01
QUANT_REL = 1.05    # a changed quantizer file: its KLD to the source model above this times the base file's is INCOMPLETE
QUANT_FAIL = 1.5    # and above this, a FAIL
MEM_REL = 0.05      # a memory rise above 5 % and MEM_ABS MiB needs an explanation in the PR
MEM_ABS = 256
SPEED_REL = 0.03    # a slowdown above this is a FAIL, or INCOMPLETE when the noise of the test is as large (rerun or explain)

out = sys.argv[1]


def emit(*parts):
    """print(): the output of this script is its result, and the repo's flake8 forbids print"""
    sys.stdout.write(" ".join(str(p) for p in parts) + "\n")


# what a process prints when it loses the GPU (a reset, a hang): no root nor kernel log needed to see it
GPU_LOST = re.compile(r"vk::DeviceLostError|ErrorDeviceLost|hipErrorLaunchFailure|HSA_STATUS_ERROR_\w+|Memory access fault by GPU")


def lost(*texts):
    return ", GPU device lost" if any(t and GPU_LOST.search(t) for t in texts) else ""


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


ANSI = re.compile(r"\x1b\[[0-9;]*m")
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|-?inf"


def norm(line):
    """a failure line without its error value, so a known failure whose error moves is still the same failure.
    Only the error value: the op and its parameters (type_a=q4_0, m=16) stay, so a new failure is never hidden by an old one."""
    ln = ANSI.sub("", line).strip()
    ln = re.sub(r"(?<=[=<>] )(?:%s)" % NUM, "N", ln)                                   # NMSE = x > y, MAA = x > y
    ln = re.sub(r"\((?:%s)\)" % NUM, "(N)", ln)                                        # test-llama-archs: FAIL (1.2e-03)
    ln = re.sub(r"(at index )\d+ \((\w+)=(?:%s)\)" % NUM, r"\1N (\2=N)", ln)            # nonfinite gradient
    return ln


def ops_coverage(text, dev):
    """None if test-backend-ops ran tests on <dev>, else the reason: a skipped device still counts as a passed backend"""
    blocks = re.split(r"^(?=Backend \d+/\d+: )", text, flags=re.M)
    for blk in blocks:
        m = re.match(r"Backend \d+/\d+: (\S+)", blk)
        if not m or m.group(1) != dev:
            continue
        first = next((ln.strip() for ln in blk.splitlines()[1:] if ln.strip()), "")
        if first.startswith("Skipping"):
            return "%s skipped" % dev
        t = re.search(r"^\s*(\d+)/(\d+) tests passed", blk, re.M)
        if not t or int(t.group(2)) == 0:
            return "no test ran on %s" % dev
        return None
    return "%s not tested" % dev


def ops_total(text, dev):
    """number of tests test-backend-ops ran on <dev>: an op the device no longer supports is skipped, not failed"""
    for blk in re.split(r"^(?=Backend \d+/\d+: )", text, flags=re.M):
        m = re.match(r"Backend \d+/\d+: (\S+)", blk)
        t = re.search(r"^\s*(\d+)/(\d+) tests passed", blk, re.M)
        if m and m.group(1) == dev and t:
            return int(t.group(2))
    return None


def backends_passed(text):
    m = re.search(r"^(\d+)/(\d+) backends passed", text, re.M)
    return (int(m.group(1)), int(m.group(2))) if m else None


def archs_run(text):
    """test-llama-archs rows (arch, device, Dense/MoE) that ran, OK or FAIL: a skipped arch prints SKIP and is not counted"""
    rows = re.findall(r"^\|\s*(\S+)\|\s*([^|]*?)\s*\|\s*(Dense|MoE)\|\s*(OK|FAIL)\b", ANSI.sub("", text), re.M)
    return {r[:3] for r in rows}


def failing(text):
    """the test cases test-backend-ops lists under "Failing tests:" at the end"""
    m = re.search(r"^Failing tests:\n(.*?)^  Backend ", ANSI.sub("", text or ""), re.M | re.S)
    return {ln.strip() for ln in m.group(1).splitlines() if re.match(r"^  [A-Z0-9_]+\(.*\)$", ln)} if m else set()


def flaky_only(b, c, cand):
    """the new failures of the candidate when validate.sh ran each again alone and it passed 3 times, else None"""
    new = failing(c) - failing(b)
    rerun = {}
    for ln in (read(os.path.join(os.path.dirname(cand), "ops-rerun.txt")) or "").splitlines():
        name, _, n = ln.rpartition("\t")
        rerun[name] = n
    return sorted(new) if new and all(rerun.get(t) == "3/3" for t in new) else None


def new_fails(base, cand, done, dev=None):
    """<done>: a line the tool prints only when it ran to the end; <dev>: test-backend-ops must have run tests on it"""
    b, c = read(base), read(cand)
    if c is None:
        return INC, "not run"
    if done not in c:
        code = re.search(r"EXIT=(\d+)", c)
        if b is not None and done not in b:
            # both stopped: nothing to blame on the candidate (validate.sh already ran the baseline alone again)
            return INC, "neither the baseline nor the candidate finished%s: rerun" % lost(c, b, read(re.sub(r"\.txt$", ".err", cand)))
        return FAIL, "the candidate did not finish (crash or abort%s%s)" % (", exit %s" % code.group(1) if code else "",
                                                                            lost(c, read(re.sub(r"\.txt$", ".err", cand))))
    if b is None or done not in b:
        return INC, "the baseline did not finish, nothing to compare with"
    if dev:
        cb, cc = ops_coverage(b, dev), ops_coverage(c, dev)
        if cc:
            return (INC if cb else FAIL), "%s in the candidate%s" % (cc, " and in the baseline" if cb else "")
        pb, pc = backends_passed(b), backends_passed(c)
        if pb and pc and pb[0] == pb[1] and pc[0] < pc[1] and not flaky_only(b, c, cand):
            return FAIL, "%d/%d backends passed (base %d/%d)" % (pc + pb)
        tb, tc = ops_total(b, dev), ops_total(c, dev)
        if tb and tc is not None and tc < tb:
            return INC, "%d tests ran on %s, %d in the baseline: ops no longer supported fall back to the CPU, explain it" % (tc, dev, tb)
    elif "Summary: no tests executed" in c:
        return INC, "no tests executed"
    fb = {norm(ln) for ln in b.splitlines() if "FAIL" in ln}
    fc = [ANSI.sub("", ln).strip() for ln in c.splitlines() if "FAIL" in ln and norm(ln) not in fb]
    if fc:
        fl = flaky_only(b, c, cand)
        if fl:
            # each new failure passed alone 3 times: a kernel that fails now and then beside other work, not the sync
            return INC, "%d failure(s) only beside other work, each passed alone 3/3 (ops-rerun.txt), explain it: %s" % (len(fl), "; ".join(fl[:3]))
        return FAIL, "%d new: %s" % (len(fc), "; ".join(fc[:3]))
    if done == "Summary:":
        # per row, not the count: an arch upstream adds would hide one the candidate now skips
        gone = sorted(archs_run(b) - archs_run(c))
        if gone:
            return INC, "%d test(s) of the baseline no longer run, explain it: %s" % (len(gone), ", ".join(" ".join(r) for r in gone[:5]))
    return PASS, "no new failure (%d known in base)" % len(fb)


def quant_status(name, q, src_type):
    """the quantizer must work on the candidate, and a file it writes differently must be no worse against the source model.
    The KLD of the quant table runs on the file of the base quantizer. <name>: small or moe"""
    if q == src_type:
        return PASS, "-"
    p = os.path.join(out, "quant", "%s-%s" % (name, q))
    bl, cl = read(p + "-base.log"), read(p + "-cand.log")
    sha = (read(p + ".sha") or "").split()
    if cl is None:
        return INC, "-"
    base_failed = bl is None or "EXIT=" in bl or len(sha) != 2 or sha[0] == "missing"
    if "EXIT=" in cl or len(sha) != 2 or sha[1] == "missing":
        # both failing is not the candidate's fault (an IQ type without an importance matrix, for example)
        return (INC, "both FAILED") if base_failed else (FAIL, "cand FAILED")
    if base_failed:
        return INC, "base failed"
    if sha[0] == sha[1]:
        return PASS, "same"
    # another file: both files against the source model, same baseline binary (validate.sh srckld), so only the file differs
    sb, sc = (kld_metrics(p + "-src-%s.txt" % side) for side in ("base", "cand"))
    tb, tc = read(p + "-src-base.txt"), read(p + "-src-cand.txt")
    if sb is None or tb is None or "EXIT=" in tb or tc is None:
        return INC, "CHANGED, not compared with the source"
    if sc is None or "EXIT=" in tc:
        cb = read(p + "-src-candbin.txt")
        if cb is not None and "EXIT=" not in cb and kld_metrics(p + "-src-candbin.txt") is not None:
            return INC, "CHANGED, the baseline cannot read the candidate's file, the candidate can (a format change?): explain it"
        return FAIL, "CHANGED, the candidate's file does not run (the base file does)"
    note = "CHANGED, KLD to the source %.6f (base file %.6f)" % (sc["kld"], sb["kld"])
    if not all(math.isfinite(v) for v in sc.values()) or sc["kld"] > QUANT_FAIL * sb["kld"] + KLD_ABS:
        return FAIL, note
    # a small sample (2048 tokens): a few % worse can be a legit change, more must be explained
    if sc["kld"] > QUANT_REL * sb["kld"] + KLD_ABS or abs(sc["ratio"] - 1.0) > abs(sb["ratio"] - 1.0) + PPL_RATIO:
        return INC, note
    return PASS, note


def kld_metrics(path):
    t = read(path)
    if t is None:
        return None
    m = {}
    # identical runs print a KLD of -0.000000: accept a sign and an exponent
    # and nan / inf: a broken candidate must give a FAIL, not a line that does not parse
    num = r"([-+]?(?:[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|nan|inf))"
    for key, rx in (("ratio", r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*" + num),
                    ("kld", r"Mean\s+KLD:\s*" + num),
                    ("top", r"Same top p:\s*" + num)):
        r = re.search(rx, t, re.I)
        if r is None:
            return None
        m[key] = float(r.group(1))
    return m


def kld(be, label):
    ct = read(os.path.join(out, be, "kld", label + "-cand.txt"))
    c = kld_metrics(os.path.join(out, be, "kld", label + "-cand.txt"))
    r = kld_metrics(os.path.join(out, be, "kld", label + "-ref.txt"))
    if ct is not None and "EXIT=" in ct:
        return FAIL, None, None      # the candidate crashed or timed out, even after printing its metrics
    if c is None or r is None:
        return INC, None, None
    # the PPL ratio only where the reference has it near 1: on a model this bad (a 1-bit 0.8B, PPL about 1e6) the
    # stored baseline logits no longer give its PPL, and the base against itself is already far from 1. KLD and top p judge
    ppl_ok = abs(c["ratio"] - 1.0) <= PPL_RATIO or abs(r["ratio"] - 1.0) > PPL_RATIO
    ok = (all(math.isfinite(v) for v in c.values()) and c["top"] >= r["top"] - TOP_P_DROP and c["kld"] <= KLD_FACTOR * r["kld"] + KLD_ABS
          and ppl_ok)
    if not ok:
        return FAIL, c, r
    # above the reference (another correct kernel path), the change in the logits needs an explanation in the PR
    return (PASS if c["kld"] <= r["kld"] + KLD_ABS else INC), c, r


def bench_rows(be, label):
    expect = read(os.path.join(out, be, "bench", label + ".expect"))
    expected = {tuple(int(x) for x in ln.split()) for ln in (expect or "").splitlines() if ln.strip()}
    sides, crashed, base_crashed, texts = {}, None, None, {}
    for side in ("base", "cand"):
        rows = {}
        text = texts[side] = "".join(read(f) or "" for f in sorted(glob.glob(os.path.join(out, be, "bench", "%s-%s-*.log" % (label, side)))))
        if side == "cand":
            crashed = re.search(r"EXIT=(\d+)", text)
        else:
            base_crashed = re.search(r"EXIT=(\d+)", text)
        for line in text.splitlines():
            if not line.startswith("{"):
                continue
            try:
                j = json.loads(line)
            except ValueError:
                continue
            key = (j.get("n_prompt"), j.get("n_gen"), j.get("n_depth", 0))
            if j.get("avg_ts") is None:
                continue
            rows.setdefault(key, []).append((j["avg_ts"], j.get("stddev_ts") or 0.0))
        sides[side] = rows
    res = []
    for key in sorted(set(sides["base"]) | set(sides["cand"]) | expected, key=lambda k: tuple(x or 0 for x in k)):
        b, c = sides["base"].get(key), sides["cand"].get(key)
        if not c:
            # missing in the candidate only: it crashed or stopped there (INCOMPLETE when the baseline crashed too:
            # the machine, a GPU reset, not the candidate)
            res.append((key, None, None, None, FAIL if b and not base_crashed else INC))
            continue
        if not b:
            res.append((key, None, None, None, INC))
            continue
        mb, mc = sum(x[0] for x in b) / len(b), sum(x[0] for x in c) / len(c)
        sb, sc = sum(x[1] for x in b) / len(b), sum(x[1] for x in c) / len(c)
        # a slowdown above SPEED_REL fails; within the noise of the test it cannot be told from noise: INCOMPLETE, never PASS
        if mc >= mb * (1 - SPEED_REL):
            status = PASS
        elif mc >= mb - (sb + sc):
            status = INC
        else:
            status = FAIL
        res.append((key, mb, mc, 100.0 * (mc / mb - 1.0), status))
    if crashed:
        # one round of the candidate crashed or timed out: the other rounds do not make it pass. When a baseline
        # round crashed too, it cannot be told from the machine (a GPU reset): INCOMPLETE
        res.append((("crash", crashed.group(1), bool(base_crashed), lost(*texts.values())), None, None, None, INC if base_crashed else FAIL))
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


def mem_row(name, base_paths, cand_paths, gate=True):
    """<gate> False: shown only, no status (test-backend-ops allocates differently from run to run)"""
    rb, gb, _ = mem(base_paths)
    rc, gc, bd = mem(cand_paths)
    if rc is None:
        return name, None, (INC if gate else "-")
    st = PASS if gate else "-"
    for b, c in ((rb, rc), (gb, gc)):
        if b is not None and c is not None and c > b * (1 + MEM_REL) and c - b > MEM_ABS:
            st = INC if gate else st

    def fmt(b, c):
        return "-" if c is None else ("%d -> %d MiB" % (b, c) if b is not None else "%d MiB" % c)

    dev = "; ".join("%s %d = %d + %d + %d" % ((k,) + v) for k, v in bd.items())
    return name, (fmt(rb, rc), fmt(gb, gc), dev or "-"), st


def test_name(key):
    if key[0] == "crash":
        return "candidate crashed (exit %s%s%s)" % (key[1], key[3], ", the baseline too" if key[2] else "")
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
        # the baseline answered, the candidate did not: its server crashed or never started
        statuses.append(FAIL if calls.get("base") is not None else INC)
        notes.append("tool call: no answer from the candidate" if calls.get("base") is not None else "tool call: not run")
    elif not calls["cand"] and calls.get("base") is False:
        statuses.append(INC)
        notes.append("tool call: broken on the baseline too (no tool template?)")
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
    if "base" in texts and "cand" not in texts:
        statuses.append(FAIL)
        notes.append("decode: no answer from the candidate")
    elif len(texts) < 2:
        statuses.append(INC)
        notes.append("decode: not run")
    elif texts["base"] == texts["cand"]:
        statuses.append(PASS)
        notes.append("decode: identical")
    else:
        n = next((i for i, (a, b) in enumerate(zip(texts["base"], texts["cand"])) if a != b), min(len(texts["base"]), len(texts["cand"])))
        statuses.append(INC)
        notes.append("decode: differs from character %d, explain it in the PR" % n)
    if len(texts) == 2 and "base" in rates and "cand" not in rates:
        statuses.append(FAIL)
        notes.append("MTP: drafts in the baseline (%.1f%% accepted), none in the candidate" % rates["base"])
    elif "base" in rates and "cand" in rates:
        delta = rates["cand"] - rates["base"]
        statuses.append(PASS if delta >= -5.0 else FAIL)
        notes.append("MTP acceptance: %.1f%% (base %.1f%%)" % (rates["cand"], rates["base"]))
    elif len(texts) == 2 and "cand" in rates:
        statuses.append(PASS)
        notes.append("MTP: drafts in the candidate only (%.1f%% accepted)" % rates["cand"])
    elif len(texts) == 2:
        statuses.append(PASS)
        notes.append("MTP: not applicable, no drafts on either side (no MTP head)")
    else:
        statuses.append(INC)
        notes.append("MTP: not measured, the server did not answer")
    return worst(statuses), "; ".join(notes)


def main():
    manifest = read(os.path.join(out, "manifest.txt")) or ""
    info = {k: v for k, v in (ln.split("=", 1) for ln in manifest.splitlines() if "=" in ln and not ln.startswith("model="))}
    models = [ln.split("=", 1)[1].split() for ln in manifest.splitlines() if ln.startswith("model=")]
    models = [m[0] for m in models]
    emit("## Validation")
    emit()
    emit("Generated by `skills/upstream-sync/scripts/validate.sh` on a Strix Halo (gfx1151) machine%s." % ((" for " + info["pr"]) if info.get("pr") else ""))
    rocm = (", ROCm " + info["rocm"]) if info.get("rocm") else ""
    emit("Base `%s`, candidate `%s`, small model `%s`%s." % (info.get("base", "?")[:9], info.get("cand", "?")[:9], info.get("small", "?"), rocm))
    emit("Scope: `QUANTS=\"%s\" MOE_QUANTS=\"%s\" KV_TYPES=\"%s\" PP_SIZES=%s DEPTHS=%s`, small MoE `%s`."
         % (info.get("quants", "?"), info.get("moe_quants", ""), info.get("kv_types", ""), info.get("pp_sizes", "?"),
            info.get("depths", "?"), info.get("small_moe", "none")))
    emit("Corpus sha256 in `corpus.sha256`. Thresholds: `Same top p` >= reference - %.1f points, `Mean KLD` <= %.0f x reference + %g, "
         "`PPL(Q)/PPL(base)` within 1 +/- %.2f (a KLD above the reference is INCOMPLETE until the PR explains it); speed: a slowdown "
         "above %d%% fails, or is INCOMPLETE when it is within the summed standard deviations of base and candidate (rerun or explain). "
         "The reference is the base build against itself with a smaller ubatch."
         % (TOP_P_DROP, KLD_FACTOR, KLD_ABS, PPL_RATIO, SPEED_REL * 100))
    resets = (read(os.path.join(out, "gpu-resets.txt")) or "").strip()
    if resets and resets != "0":
        emit()
        emit("The kernel log shows %s amdgpu ring timeouts (GPU resets) during the run: a crash beside them is the machine as much as the code." % resets)

    for be in info.get("backends", "").split(","):
        if not be:
            continue
        emit()
        rows, statuses = [], []
        for side in ("base", "cand"):
            s = (read(os.path.join(out, "logs", "build-%s-%s.status" % (side, be))) or "").strip()
            st = PASS if s == "ok" else (FAIL if s == "failed" else INC)
            statuses.append(st)
            rows.append(("build " + side, st, s or "not run"))
        dev = {"hip": "ROCm0", "vk": "Vulkan0", "cpu": "CPU"}.get(be)
        for name, f, done, d in (("test-backend-ops", "ops", "backends passed", dev), ("test-llama-archs (the archs it supports)", "archs", "Summary:", None)):
            skipped = read(os.path.join(out, be, f + "-skipped.txt"))
            if skipped:
                # its result cannot change: the sync changes no ggml code this backend runs
                statuses.append(PASS)
                rows.append((name, PASS, "skipped: " + skipped.strip()))
                continue
            st, note = new_fails(os.path.join(out, be, f + "-base.txt"), os.path.join(out, be, f + "-cand.txt"), done, d)
            statuses.append(st)
            rows.append((name, st, note))
        sls_b, sls_c = read(os.path.join(out, be, "sls-base.txt")), read(os.path.join(out, be, "sls-cand.txt"))
        if sls_c is None:
            st, note = INC, "not run"
        elif "EXIT=" not in sls_c:
            st, note = PASS, ""
        elif sls_b is not None and "EXIT=" in sls_b:
            st, note = INC, "fails on the baseline too"
        else:
            st, note = FAIL, "fails on the candidate only"
        statuses.append(st)
        rows.append(("test-save-load-state (small model)", st, note))

        # every quant asked for in the manifest: a missing result is INCOMPLETE, never left out
        quant_tables = []
        qskip = read(os.path.join(out, be, "quants-skipped.txt"))
        if qskip:
            # their results cannot change: the sync changes neither the quantizer nor the ggml code of this backend
            statuses.append(PASS)
            rows.append(("small models, every quant type", PASS, "skipped: " + qskip.strip()))
        for name, title, qk, tk in ([] if qskip else (("small", "Qwen3.5-0.8B-Base (dense), every quant type", "quants", "small_type"),
                                    ("moe", "Granite 3.1 1B-A400M (MoE: MXFP4_MOE changes only expert tensors)", "moe_quants", "moe_type"))):
            quant_rows = []
            for q in info.get(qk, "").split():
                ks, c, r = kld(be, "%s-%s" % (name, q))
                br = bench_rows(be, "%s-%s" % (name, q))
                bs = bench_status(br)
                qs, same = quant_status(name, q, info.get(tk))
                st = worst([ks, bs, qs])
                statuses.append(st)
                d = {test_name(k): x for k, _, _, x, _ in br}
                quant_rows.append((q, same, c, r, d, st))
            if quant_rows:
                quant_tables.append((title, quant_rows))

        kv_rows = []
        for t in info.get("kv_types", "").split():
            ks, c, r = kld(be, "kv-" + t)
            br = bench_rows(be, "kv-" + t)
            st = worst([ks, bench_status(br)])
            statuses.append(st)
            kv_rows.append((t, c, r, {test_name(k): x for k, _, _, x, _ in br}, st))

        if not models:
            # the small model alone does not validate a sync
            statuses.append(INC)
            rows.append(("real models", INC, "none: no models table and no --model"))
        model_rows = []
        for a in models:
            for n in ("prose", "code", "structured", "numeric", "long"):
                ks, c, r = kld(be, "%s-%s" % (a, n))
                statuses.append(ks)
                model_rows.append((a, n, c, r, ks))
        bench_tables = []
        for a in models:
            # one llama-bench process for pp and tg; older outputs have one for each
            br = bench_rows(be, a + "-bench") or bench_rows(be, a + "-pp") + bench_rows(be, a + "-tg")
            statuses.append(bench_status(br))
            bench_tables.append((a, br))
        server_rows = []
        for a in models:
            st, note = server(be, a)
            statuses.append(st)
            server_rows.append((a, st, note))

        def k(*x):
            return os.path.join(out, be, *x)

        mem_rows = [mem_row("test-backend-ops", [k("ops-base.txt")], [k("ops-cand.txt")], gate=False),
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
        statuses += [r[2] for r in mem_rows if r[2] != "-"]

        status = {PASS: "TARGET PASS", FAIL: "FAIL", INC: "INCOMPLETE"}[worst(statuses)]
        emit("### %s: %s" % (be, status))
        emit()
        emit("| Check | Result | Notes |")
        emit("|---|---|---|")
        for name, st, note in rows:
            emit("| %s | %s | %s |" % (name, st, note))
        for title, quant_rows in quant_tables:
            emit()
            emit("%s (KLD on `all.txt` at `-c 512`, `llama-bench` pp512 / tg128):" % title)
            emit()
            emit("| Quant | Quantizer output | Same top p (ref) | Mean KLD (ref) | PPL ratio | pp512 | tg128 | Status |")
            emit("|---|---|---|---|---|---|---|---|")
            for q, same, c, r, d, st in quant_rows:
                kl = ("%.3f%% (%.3f%%)" % (c["top"], r["top"]), "%.6f (%.6f)" % (c["kld"], r["kld"]), "%.4f" % c["ratio"]) if c and r else ("-", "-", "-")
                sp = ["%+.1f%%" % d[k] if d.get(k) is not None else "-" for k in ("pp512", "tg128")]
                emit("| %s | %s | %s | %s | %s | %s | %s | %s |" % (q, same, kl[0], kl[1], kl[2], sp[0], sp[1], st))
        if kv_rows:
            emit()
            emit("Qwen3.5-0.8B-Base with a quantized KV cache, K and V of the same type (KLD on `all.txt` at `-c 2048`, "
                 "`llama-bench` pp512 / tg128 at depth 8192):")
            emit()
            emit("| KV cache | Same top p (ref) | Mean KLD (ref) | PPL ratio | pp512 @ d8192 | tg128 @ d8192 | Status |")
            emit("|---|---|---|---|---|---|---|")
            for t, c, r, d, st in kv_rows:
                kl = ("%.3f%% (%.3f%%)" % (c["top"], r["top"]), "%.6f (%.6f)" % (c["kld"], r["kld"]), "%.4f" % c["ratio"]) if c and r else ("-", "-", "-")
                sp = ["%+.1f%%" % d[k] if d.get(k) is not None else "-" for k in ("pp512 @ d8192", "tg128 @ d8192")]
                emit("| %s | %s | %s | %s | %s | %s | %s |" % (t, kl[0], kl[1], kl[2], sp[0], sp[1], st))
        if model_rows:
            emit()
            emit("Real models, logits against the base build:")
            emit()
            emit("| Model | Text | Same top p (ref) | Mean KLD (ref) | PPL ratio | Status |")
            emit("|---|---|---|---|---|---|")
            for a, n, c, r, st in model_rows:
                if n == "long":
                    lc = (read(os.path.join(out, be, "kld", a + "-long.ctx")) or "").strip()
                    n = "long (`-c %s`)" % (lc or 16384)
                if c and r:
                    emit("| %s | %s | %.3f%% (%.3f%%) | %.6f (%.6f) | %.4f | %s |" % (a, n, c["top"], r["top"], c["kld"], r["kld"], c["ratio"], st))
                else:
                    emit("| %s | %s | - | - | - | %s |" % (a, n, st))
        for a, br in bench_tables:
            emit()
            emit("%s, `llama-bench` t/s (mean of two alternating rounds per side):" % a)
            emit()
            emit("| Test | Base | Candidate | Change | Status |")
            emit("|---|---|---|---|---|")
            for key, mb, mc, dp, st in br:
                if mb is None:
                    emit("| %s | - | - | - | %s |" % (test_name(key), st))
                else:
                    emit("| %s | %.2f | %.2f | %+.1f%% | %s |" % (test_name(key), mb, mc, dp, st))
        if server_rows:
            emit()
            for a, st, note in server_rows:
                emit("- %s server: %s. %s" % (a, st, note))
        emit()
        emit("Memory, base -> candidate: peak RSS of the process, peak GPU memory (amdgpu VRAM + GTT above the level before the run), "
             "and the llama.cpp breakdown of the candidate per device in MiB (self = model + context + compute) when the tool printed it. "
             "A rise above %d%% and %d MiB is INCOMPLETE until the PR explains it." % (MEM_REL * 100, MEM_ABS))
        emit()
        emit("| Run | Peak RSS | Peak GPU | Breakdown | Status |")
        emit("|---|---|---|---|---|")
        for name, v, st in mem_rows:
            emit("| %s | %s | %s | %s | %s |" % ((name,) + (v or ("-", "-", "-")) + (st,)))
        if any(r[1].startswith("CHANGED") for _, rows_ in quant_tables for r in rows_):
            emit()
            emit("Quantizer output CHANGED for some types: both files were compared with the source model (KLD, baseline binary). "
                 "Worse than the base file by more than %d%% is INCOMPLETE, by more than %d%% a FAIL. Say so in the PR."
                 % (round((QUANT_REL - 1) * 100), round((QUANT_FAIL - 1) * 100)))


main()
