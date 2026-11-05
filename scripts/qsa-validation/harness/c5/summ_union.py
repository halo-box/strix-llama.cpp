# Summarise c5_union outputs: one row per (corpus, depth), the 12-layer mean, plus the per-layer min-max of G4 U/q
import sys, glob, re, os
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], 'union_*.txt'))):
    m = re.search(r'union_(\w+?)_p(\d+)\.txt', f); corp, p = m.group(1), int(m.group(2))
    per = []; mn = None; hist = []
    for l in open(f):
        if 'G4HIST' in l:
            hist.append(l.split(':')[1].split()); continue
        if l.startswith('#') or l.startswith('tag'): continue
        parts = l.split('|')
        head = parts[0].split()
        g = [[float(x) for x in s.split(':')[1].split()] for s in parts[1:5]]
        rec = dict(L=head[1], nk=float(head[2]), ovl=float(head[9]), qpb=float(head[6]), part=float(head[7]), g=g)
        if head[1] == 'MN': mn = rec
        else: per.append(rec)
    g4 = [r['g'][2][0] for r in per]
    rows.append((corp, p, mn, min(g4), max(g4), len(per), hist))
order = {'wiki': 0, 'wikiq8': 1, 'code': 2, 'chat': 3, 'randc': 4, 'bench': 5}
rows.sort(key=lambda r: (order.get(r[0], 9), r[1]))
print('| corpus | pos0 (n_kv) | adj ovl | U/q G2 | U/q G4 [layer min-max] | U/q G8 | G4 ch/q ASC | G4 ch/q BKT | G4 t12 ASC | G4 t12 ASC-skip | G4 b12 BKT | G4 b16 BKT | b12/t12 | G8 ch/q | G8 t12 ASC | G8 t12 ASC-skip | G1 t/q |')
print('|' + '---|' * 17)
for corp, p, m, lo, hi, n, hist in rows:
    g1, g2, g4, g8 = m['g']
    print(f"| {corp} | {p} ({int(m['nk'])}) | {m['ovl']:.3f} | {g2[0]:.2f} | {g4[0]:.2f} [{lo:.2f}-{hi:.2f}] | {g8[0]:.2f} | {g4[1]:.1f} | {g4[4]:.1f} | {g4[2]:.1f} | {g4[3]:.1f} | {g4[5]:.1f} | {g4[6]:.1f} | {g4[5]/g4[2]:.3f} | {g8[1]:.1f} | {g8[2]:.1f} | {g8[3]:.1f} | {g1[2]:.1f} |")
print('\n| corpus | pos0 | ASC chunks with 1/2/3 active tiles (all-4-queries) | BUCKET chunks with 1/2/3 active tiles (all-4-queries) |\n|---|---|---|---|')
for corp, p, m, lo, hi, n, hist in rows:
    a, b = hist[0], hist[1]
    print(f'| {corp} | {p} | {a[0]}/{a[1]}/{a[2]} ({a[4]}) | {b[0]}/{b[1]}/{b[2]} ({b[4]}) |')
print('\nqpb (quads per logical block) range:', min(r[2]['qpb'] for r in rows), max(r[2]['qpb'] for r in rows), ' part range:', min(r[2]['part'] for r in rows), max(r[2]['part'] for r in rows), ' layers per file:', set(r[5] for r in rows))
