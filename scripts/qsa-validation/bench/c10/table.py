# Markdown table from summary.txt (analyze.py output): text and random side by side, medians.
import re, sys
rows = {}
for l in open(sys.argv[1]):
    m = re.match(r'd(\d+) (f16|q8_0) ub(\d+) pp(\d+) (text|rand) \|.*?\| med\s+([\d.]+)/\s*([\d.]+)\s+([+-][\d.]+)% \[([+-][\d.]+), ([+-][\d.]+)\]', l)
    if m:
        d, t, ub, pp, wl, b, p, r, lo, hi = m.groups()
        rows[(int(d), t, int(ub), int(pp), wl)] = (float(b), float(p), r, lo, hi)
print('| depth | type | ub | pp | text base / PR t/s | text PR/base [95% CI] | random base / PR t/s | random PR/base [95% CI] | text/random base, PR |')
print('|---|---|---|---|---|---|---|---|---|')
for k in sorted(rows):
    if k[4] != 'text': continue
    d, t, ub, pp, _ = k; T = rows[k]; R = rows.get(k[:4] + ('rand',))
    dd = {0: '0', 32768: '32k', 131072: '131k'}[d]
    print(f'| {dd} | {t} | {ub} | {pp} | {T[0]:.1f} / {T[1]:.1f} | {T[2]}% [{T[3]}, {T[4]}] | {R[0]:.1f} / {R[1]:.1f} | {R[2]}% [{R[3]}, {R[4]}] | {T[0]/R[0]:.3f}, {T[1]/R[1]:.3f} |')
