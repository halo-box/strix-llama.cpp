#!/usr/bin/env python3
# C7 lab analysis: per llama_process window (warmup = w, reps = 0..), wall ms, hip events inside, sampler deltas.
import json, sys

def load(tag):
    ev = []
    for ln in open(tag + ".shim"):
        p = ln.split()
        if len(p) < 5: continue
        ev.append((p[0], int(p[1]), int(p[2]), int(p[3]), p[4]))
    samp = []
    hdr = None
    for ln in open(tag + ".samp"):
        p = ln.split()
        if hdr is None: hdr = p; continue
        if len(p) != len(hdr): continue
        samp.append(dict(zip(hdr, p)))
    try:
        js = json.load(open(tag + ".json"))
        ts = [x for t in js for x in t.get("samples_ts", [])]
    except Exception:
        ts = []
    return ev, samp, ts

def num(x):
    try: return float(x)
    except ValueError: return 0.0

def at(samp, t):
    # last sample at or before t
    lo = None
    for s in samp:
        if int(s["t_ns"]) <= t: lo = s
        else: break
    return lo

def main(tag, verbose):
    ev, samp, ts = load(tag)
    procs = [i for i, e in enumerate(ev) if e[0] == "LPROC"]
    syncs = [i for i, e in enumerate(ev) if e[0] == "LSYNC"]
    rows = []
    prev = -1
    for k, si in enumerate(syncs):
        # window: events after the previous LSYNC up to this LSYNC; t0 = first LPROC in it
        inside = ev[prev + 1:si + 1]
        pi = next(i for i in range(prev + 1, si) if ev[i][0] == "LPROC")
        prev = si
        t0 = ev[pi][1]; t1 = ev[si][1] + ev[si][2]
        nm = sum(1 for e in inside if e[0] == "MALLOC"); bm = sum(e[3] for e in inside if e[0] == "MALLOC")
        tm = sum(e[2] for e in inside if e[0] == "MALLOC") / 1e6
        nf = sum(1 for e in inside if e[0] == "FREE"); tf = sum(e[2] for e in inside if e[0] == "FREE") / 1e6
        other = [f"{e[0]}:{e[2]/1e6:.1f}" for e in inside if e[0] in ("CAPBEGIN", "CAPEND", "GINST", "GUPDATE", "DSYNC")]
        s0 = at(samp, t0); s1 = at(samp, t1)
        d = {}
        if s0 and s1:
            for key in ("evicted_ms", "majflt", "minflt", "pswpout", "pswpin", "allocstall_normal", "allocstall_movable", "compact_stall", "pgscan_direct", "pgmajfault"):
                d[key] = num(s1[key]) - num(s0[key])
            d["MemFree_MiB"] = num(s1["MemFree"]) / 1024
        ms = (t1 - t0) / 1e6
        R = REPS; tst, j = divmod(k, R + 1)
        label = f"t{tst}w" if j == 0 else f"t{tst}.{j-1}"
        tsj = ts[tst * R + j - 1] if j >= 1 and tst * R + j - 1 < len(ts) else None
        rows.append((label, ms, tsj, nm, bm, tm, nf, tf, other, d))
    print(f"# {tag}: windows {len(rows)} samples_ts {len(ts)}")
    print("win wall_ms json_ts mallocs MiB malloc_ms frees free_ms evicted_ms majflt minflt pswpout allocstall compact memfree_MiB other")
    for r in rows:
        label, ms, tsj, nm, bm, tm, nf, tf, other, d = r
        print(f"{label:>6} {ms:8.1f} {tsj if tsj is None else round(tsj,1)!s:>7} {nm:4d} {bm/2**20:8.1f} {tm:7.1f} {nf:4d} {tf:7.1f} "
              f"{d.get('evicted_ms',0):6.0f} {d.get('majflt',0):5.0f} {d.get('minflt',0):6.0f} {d.get('pswpout',0):5.0f} "
              f"{d.get('allocstall_normal',0)+d.get('allocstall_movable',0):4.0f} {d.get('compact_stall',0):3.0f} {d.get('MemFree_MiB',0):8.0f} {' '.join(other)}")
    reps = [r[1] for r in rows if "w" not in r[0] and r[0].startswith("t0.")]
    if reps:
        s = sorted(reps); med = s[len(s) // 2]
        print(f"# reps {len(reps)} median_ms {med:.1f} min {s[0]:.1f} max {s[-1]:.1f} stalls(>1.25x median) {sum(1 for x in reps if x > 1.25 * med)}")
    if verbose:
        for e in ev:
            if e[0] in ("MALLOC", "FREE") and procs and e[1] > ev[procs[0]][1]:
                print("  ", e[0], e[1], f"{e[2]/1e6:.2f}ms", f"{e[3]/2**20:.1f}MiB", e[4])

REPS = 30
if __name__ == "__main__":
    for t in sys.argv[1:]:
        if t.startswith("-r"): REPS = int(t[2:]); continue
        if t == "-v": continue
        main(t, "-v" in sys.argv)
