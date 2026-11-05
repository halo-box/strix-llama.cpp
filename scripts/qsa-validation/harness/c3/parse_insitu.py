import sys, re, collections, statistics as st
# PLANA lines: per qsa_prefill call; group by nk
for fn in sys.argv[1:]:
    d = collections.defaultdict(list)
    for l in open(fn, errors='replace'):
        if 'PLANA call=' not in l: continue
        m = dict(kv.split('=', 1) for kv in l.split() if '=' in kv)
        d[int(m['nk'])].append((float(m['union_ms']), float(m['attn_ms']), float(m['pack_ms']), float(m['g4_mean_blk'])))
    print(fn)
    for nk in sorted(d):
        if nk not in (4096, 36864, 65536, 102400, 131072, 135168) and nk % 32768 != 4096: continue
        u = [x[0] for x in d[nk]]; a = [x[1] for x in d[nk]]; p = [x[2] for x in d[nk]]; b = [x[3] for x in d[nk]]
        print(f"  nk={nk} calls={len(u)} union_ms med {st.median(u):.3f} mean {st.mean(u):.3f} sd {st.pstdev(u):.3f} min {min(u):.3f} max {max(u):.3f} | attn_ms med {st.median(a):.2f} mean {st.mean(a):.2f} | pack_ms med {st.median(p):.3f} | g4_mean_blk {st.mean(b):.1f} | union/attn {st.mean(u)/st.mean(a)*100:.1f}%")
