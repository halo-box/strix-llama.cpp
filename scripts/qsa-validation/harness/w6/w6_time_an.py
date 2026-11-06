#!/usr/bin/env python3
# W6 AT-B2 / AT-B3: per (rows file, force arm, type arm): mean over layers of the per-layer value, per process; median over processes; ratios.
# Inputs: time_<rows>_<force>_p<k>.out (stdout of w6_time) and .err (GGML_VK_PERF_LOGGER timings between MARK lines).
import glob,os,re,sys,statistics as st
from collections import defaultdict
D=sys.argv[1]
wall=defaultdict(lambda: defaultdict(list)); opus=defaultdict(lambda: defaultdict(list))
alloc={}
for fn in sorted(glob.glob(D+'/time_*.out')):
    m=re.match(r'time_(.+)_(\w+)_p(\d+)\.out',os.path.basename(fn))
    rows,force,proc=m.groups()
    per=defaultdict(list); plan=defaultdict(list); tot=defaultdict(list)
    for l in open(fn):
        r=re.match(r'L(\d+) RES (\S+)\s+n_q (\d+) nk (\d+) plan_med (\S+) graph_med (\S+) attend (\S+) total (\S+) ms alloc_plan (\d+) alloc_attend (\d+)',l)
        if not r: continue
        arm=r.group(2)
        per[arm].append(float(r.group(7))); plan[arm].append(float(r.group(5))); tot[arm].append(float(r.group(8)))
        alloc[(rows,force,arm,int(r.group(1)))]=(int(r.group(9)),int(r.group(10)))
    for arm in per:
        wall[(rows,force,arm)]['attend'].append(st.mean(per[arm]))
        wall[(rows,force,arm)]['plan'].append(st.mean(plan[arm]))
        wall[(rows,force,arm)]['total'].append(st.mean(tot[arm]))
    # GGML_VK_PERF_LOGGER: op time between MARK measure and MARK end
    errf=fn[:-4]+'.err'
    if os.path.exists(errf):
        cur=None; on=False; vals=defaultdict(lambda: defaultdict(list))
        for l in open(errf,errors='replace'):
            mm=re.match(r'MARK L(\d+) (\S+) (plan|graph)',l)
            if mm: cur=mm.groups(); continue
            if l.startswith('MARK measure'): on=True; continue
            if l.startswith('MARK end'): on=False; continue
            if on and cur:
                o=re.match(r'(SPARSE_ATTN_PLAN|SPARSE_ATTN|FLASH_ATTN_EXT)\S*.*?: (\d+) x ([0-9.e+]+) us',l)
                if o and not (cur[2]=='graph' and o.group(1)=='SPARSE_ATTN_PLAN'):
                    vals[(cur[1],o.group(1))][int(cur[0])].append(float(o.group(3))/1000.0)
        for (arm,op),byl in vals.items():
            opus[(rows,force,arm)][op].append(st.mean(st.median(v) for v in byl.values()))
med=lambda x: st.median(x) if x else float('nan')
print('rows force arm procs | wall attend, plan, total (ms) | logger op attend, plan (ms)')
keys=sorted(wall)
for k in keys:
    w=wall[k]; o=opus.get(k,{})
    ao=med(o.get('SPARSE_ATTN',[]) or o.get('FLASH_ATTN_EXT',[]))
    po=med(o.get('SPARSE_ATTN_PLAN',[]))
    print(f"{k[0]:12s} {k[1]:8s} {k[2]:10s} {len(w['attend'])} | {med(w['attend']):9.3f} {med(w['plan']):8.3f} {med(w['total']):9.3f} | {ao:9.3f} {po:8.3f}")
print()
print('AT-B2: new f16 (default arm) / in-build sel_cm, per rows file: pass <= 1.10')
for rows in sorted({k[0] for k in keys}):
    s=wall.get((rows,'default','sel')); n=wall.get((rows,'default','f16'))
    if not s or not n: continue
    so=opus.get((rows,'default','sel'),{}).get('FLASH_ATTN_EXT',[]); no=opus.get((rows,'default','f16'),{}).get('SPARSE_ATTN',[]); np=opus.get((rows,'default','f16'),{}).get('SPARSE_ATTN_PLAN',[])
    a=med(n['attend'])/med(s['attend']); t=med(n['total'])/med(s['attend'])
    print(f"{rows:12s} wall: attend/sel {a:.3f}  plan+attend/sel {t:.3f}  sel {med(s['attend']):.3f} ms | logger: attend/sel {med(no)/med(so) if so and no else float('nan'):.3f}  plan+attend/sel {(med(no)+med(np))/med(so) if so and no and np else float('nan'):.3f}  sel {med(so) if so else float('nan'):.3f} ms")
print()
print('AT-B3: ratio to new f16 of the same build (default arm), attend by logger then wall; per V variant')
for rows in sorted({k[0] for k in keys}):
    ref=wall.get((rows,'default','f16')); refo=opus.get((rows,'default','f16'),{}).get('SPARSE_ATTN',[])
    if not ref: continue
    for arm in sorted({k[2] for k in keys if k[0]==rows and k[2] not in ('sel',)}):
        row=[]
        for f in ('vdirect','vstaged','vpack','default'):
            w=wall.get((rows,f,arm)); o=opus.get((rows,f,arm),{}).get('SPARSE_ATTN',[])
            if w:
                row.append(f"{f} {med(o)/med(refo) if o and refo else float('nan'):.3f}/{med(w['attend'])/med(ref['attend']):.3f}")
        print(f"{rows:12s} {arm:10s} "+'  '.join(row))
print()
print('allocations of the attend node (bytes), layer 0, per arm')
seen=set()
for (rows,force,arm,L),(ap,aa) in sorted(alloc.items()):
    if L!=0 or (rows,force,arm) in seen: continue
    seen.add((rows,force,arm)); print(f"{rows:12s} {force:8s} {arm:10s} plan {ap} attend {aa}")
