import sys, re, collections, statistics as st
# per variant: mean over layers of per-layer median total (ms), max over layers, mean rows/merge, mismatches
for fn in sys.argv[1:]:
    rows = collections.defaultdict(list); layers = []
    for l in open(fn):
        if l.startswith('LAYER'):
            m = dict(kv.split('=', 1) for kv in l.split()[2:] if '=' in kv); layers.append(m)
        if l.startswith('TIME'):
            m = dict(kv.split('=', 1) for kv in l.split()[1:] if '=' in kv)
            rows[m['variant']].append(m)
    ua = st.mean(float(x['union_asc']) for x in layers); ub = st.mean(float(x['union_bkt']) for x in layers)
    print(f"{fn}: layers={len(layers)} nk={layers[0]['nk']} union_asc_mean={ua:.1f} union_bkt_mean={ub:.1f} (bucket pad +{(ub/ua-1)*100:.1f}%) sorted_rows(l0)={layers[0]['sorted_rows']}")
    for v, xs in rows.items():
        med = [float(x['total_med']) for x in xs]; sd = [float(x['total_sd']) for x in xs]
        mis = sum(int(x['mismatch_groups']) for x in xs)
        print(f"  {v:20s} n_layers={len(xs)} reps={xs[0]['n']} mean_of_layer_medians={st.mean(med):.3f} ms  min_layer={min(med):.3f} max_layer={max(med):.3f}  mean_rep_sd={st.mean(sd):.4f}  rows={st.mean(float(x['rows_med']) for x in xs):.3f} merge={st.mean(float(x['merge_med']) for x in xs):.3f}  sum12={sum(med):.2f}  mismatches={mis}")
