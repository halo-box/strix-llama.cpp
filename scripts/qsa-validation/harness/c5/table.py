import re, glob, os
# plan-stage cost per layer from C3 (MEASUREMENTS.md C3 table: P_asc 512 / P_bucket 256), ms
plan = {('text', 32768): (0.737, 0.945), ('text', 131072): (0.733, 0.942), ('rand', 32768): (0.731, 0.963), ('rand', 131072): (0.883, 1.162), ('q8', 131072): (0.739, 0.948)}
order = ['wiki_f16_p8192_syn', 'wiki_f16_p32768_real_r1', 'wiki_f16_p32768_real_r2', 'wiki_f16_p32768_syn', 'code_f16_p32768_syn', 'chat_f16_p32768_syn', 'wiki_f16_p65536_syn', 'wiki_f16_p131072_syn',
         'wiki_q8_0_p32768_real', 'wiki_q8_0_p131072_syn', 'randc_f16_p8192_syn', 'randc_f16_p32768_syn', 'randc_f16_p131072_syn', 'bench_f16_p8192_syn', 'bench_f16_p32768_syn', 'bench_f16_p131072_syn']
print('| cell | V0 PR G4-ASC | V1 G4-ASC | V2 G4-ASC-skip | **V4 G4-BUCKET** | V4/V0 | **V4/V2** | V4/V2 worst layer | V5 BUCKET-r16 /V0 | V3 PR kern BUCKET order /V0 | V6 G8 /V0 | V7 G8-skip /V0 | oracle 2 tiles /V0 | oracle 1 tile /V0 | net (V4+P_bkt)/(V2+P_asc) |')
print('|' + '---|' * 15)
for o in order:
    f = f'remote/res/attn_{o}.out'
    med = {}
    for l in open(f):
        if re.match(r'L\d+ timing', l):
            for part in l.split('|')[1:]:
                t = part.split(); med.setdefault(t[0], []).append(float(t[2]))
    m = {k: sum(v) / len(v) for k, v in med.items()}
    worst = max(b / a for a, b in zip(med['V2'], med['V4']))
    corp = o.split('_')[0]; p = int(re.search(r'_p(\d+)', o).group(1))
    key = ('q8', p) if 'q8_0' in o else (('rand' if corp in ('randc', 'bench') else 'text'), p)
    if 'q8_0' in o and p == 32768: key = ('q8', 131072)  # only q8 plan cost C3 measured (d131k); used as proxy
    net = '' 
    if key in plan:
        pa, pb = plan[key]; net = f"{(m['V4'] + pb) / (m['V2'] + pa):.3f}" + (' (q8 plan cost of d131k)' if 'q8_0' in o and p == 32768 else '')
    else: net = 'plan cost not measured'
    print(f"| {o} (n={len(med['V0'])} layers) | {m['V0']:.3f} | {m['V1']:.3f} | {m['V2']:.3f} | {m['V4']:.3f} | {m['V4']/m['V0']:.3f} | {m['V4']/m['V2']:.3f} | {worst:.3f} | {m['V5']/m['V0']:.3f} | {m['V3']/m['V0']:.3f} | {m['V6']/m['V0']:.3f} | {m['V7']/m['V0']:.3f} | {m['V13']/m['V0']:.3f} | {m['V14']/m['V0']:.3f} | {net} |")
