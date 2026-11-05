# r1_analyze.py DUMPDIR PHASE SEG0,SEG1,... : per layer and per sequence segment of the ubatch rows, compare the
# PR219 masked-route token-expanded rows (lab_te_rows) with the C4 hunk-3 complete-block rows (lab_cb_rows), both from
# the same LAB_QSA_C4=1 run. visible = finite indexer_score_tokens (score + per-block bias + kq_mask), forced =
# visible cells with the +1e9 bias (dead/spare block cells and the own block), own cell = the cell that a row sees and
# the previous row of the same segment does not (segment rows are consecutive positions of one sequence).
import sys, collections
import numpy as np

d, phase, segs = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
man = [l.rstrip("\n").split("\t") for l in open(d + "/manifest.tsv")]
by = collections.defaultdict(list)
for ph, tag, layer, idx, ty, n0, n1, n2, n3, fn in man:
    if ph != phase:
        continue
    by[(tag, int(layer))].append((int(idx), ty, int(n0), int(n1) * int(n2) * int(n3), fn))

def load(tag, layer):
    parts = sorted(by[(tag, layer)])
    out = []
    for idx, ty, n0, rows, fn in parts:
        a = np.fromfile(d + "/" + fn, dtype=np.int32 if ty == "i32" else np.float32).reshape(rows, n0)
        out.append(a)
    return np.concatenate(out, 0)

layers = sorted({l for (t, l) in by})
for L in layers:
    ex = load("indexer_score_tokens", L)
    te = load("lab_te_rows", L)
    cbr = load("lab_cb_rows", L)
    nq, nkv = ex.shape
    assert sum(segs) == nq, (segs, nq)
    vis = np.isfinite(ex) & (ex > -1e30)
    forced = vis & (ex > 5e8)
    a = 0
    for si, n in enumerate(segs):
        rows = range(a, a + n)
        own = {}
        for i in rows:
            if i > a:
                new = np.nonzero(vis[i] & ~vis[i - 1])[0]
                own[i] = int(new[0]) if len(new) == 1 else None
        if a + 1 < a + n and own.get(a + 1) is not None:
            own[a] = own[a + 1] - 1
        st = collections.defaultdict(list)
        for i in rows:
            V = vis[i]
            F = set(np.nonzero(forced[i])[0].tolist())
            T = set(int(c) for c in te[i] if c >= 0 and V[c])
            C = set(int(c) for c in cbr[i] if c >= 0 and V[c])
            cb_invis = sum(1 for c in set(cbr[i].tolist()) if c < 0 or not V[c])
            st["vis"].append(int(V.sum()))
            st["F"].append(len(F))
            st["F_minus_CB"].append(len(F - C))
            st["F_minus_TE"].append(len(F - T))
            st["TE_minus_CB"].append(len(T - C))
            st["CB_minus_TE"].append(len(C - T))
            st["CB_size"].append(len(C))
            st["TE_size"].append(len(T))
            st["CB_invis_or_dup"].append(cb_invis + (cbr.shape[1] - len(set(cbr[i].tolist()))))
            fv = ex[i][forced[i]]
            st["F_distinct_vals"].append(len(np.unique(fv)) if len(fv) else 0)
            o = own.get(i)
            st["own_known"].append(o is not None)
            st["own_drop_CB"].append(o is not None and o not in C)
            st["own_drop_TE"].append(o is not None and o not in T)
            st["own_forced"].append(o is not None and o in F)
        def m(k):
            v = np.array(st[k], dtype=float)
            return f"{k} mean {v.mean():.3f} min {v.min():.0f} max {v.max():.0f} rows>0 {(v > 0).sum()}"
        print(f"== {phase} L{L} seg{si} rows {n} (row {a}..{a + n - 1}) n_kv {nkv}")
        for k in ["vis", "F", "F_distinct_vals", "F_minus_CB", "F_minus_TE", "TE_size", "CB_size", "TE_minus_CB", "CB_minus_TE", "CB_invis_or_dup"]:
            print("   " + m(k))
        print(f"   own cell: known {sum(st['own_known'])}/{n}, forced {sum(st['own_forced'])}, dropped by CB {sum(st['own_drop_CB'])}, dropped by TE {sum(st['own_drop_TE'])}")
        hist = collections.Counter(st["F"])
        print("   F histogram " + str(dict(sorted(hist.items())[:12])) + (" ..." if len(hist) > 12 else ""))
        a += n
