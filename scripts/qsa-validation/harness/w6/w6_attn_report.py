#!/usr/bin/env python3
# W6: summarize one test-backend-ops -o SPARSE_ATTN(_PLAN) log (GGML_SA_TEST_REPORT=1): passes, unsupported split by reason, max band ratio per tier.
import re,sys
from collections import Counter,defaultdict
for fn in sys.argv[1:]:
    txt=open(fn,errors='replace').read().splitlines()
    ns=Counter(); oks=0; fails=[]
    worst=defaultdict(float); cases=defaultdict(int); wc={}
    pairs=set(); rot=Counter()
    for l in txt:
        m=re.search(r'(SPARSE_ATTN\w*)\((.*?)\): not supported',l)
        if m:
            v=m.group(2)
            g=re.search(r'group=(\d)',v); nk=re.search(r'n_kv=(\d+)',v); tk=re.search(r'type_k=(\w+)',v)
            why='G4' if g and g.group(1)=='4' else 'large/other'
            ns[why+(' '+nk.group(1)+' '+tk.group(1) if why!='G4' and nk and tk else '')]+=1
        if re.search(r'SPARSE_ATTN.*\bFAIL',l): fails.append(l[:300])
        m=re.search(r'SA_REPORT.*case=(\S+)\((.*?)\) class=(\S*) band=(\S+) nmse=(\S+) ratio=(\S+)',l)
        if m:
            cl=m.group(3); r=float(m.group(6)); 
            tier=re.search(r'sa_attend/vk/(\w+)/(\w+)/',cl)
            key=(tier.group(1)+'/'+tier.group(2)) if tier else cl.split(',')[0]
            cases[key]+=1
            if key not in wc or r>worst[key]: worst[key]=r; wc[key]=m.group(2)[:120]+' nmse='+m.group(5)
            t=re.search(r'sa_attend/vk/\w+/\w+/(\w+)/(\w+)',cl)
            if t: pairs.add(t.groups())
            rq=re.search(r'rot_q=(\d+),rot_o=(\d+)',m.group(2))
            if rq and (rq.group(1)!='0' or rq.group(2)!='0'): rot[rq.group(1)+'/'+rq.group(2)]+=1
    print('==',fn)
    for l in txt:
        if 'tests passed' in l or 'CENSUS COVERAGE' in l: print(' ',l.strip()[:190])
    print('  failed cases:',len(fails)); [print('   ',f) for f in fails[:15]]
    print('  not supported:',sum(ns.values()),dict(ns))
    print('  K/V pairs seen:',len(pairs),' rot reports (2 per case, CPU excluded):',dict(rot))
    for k in sorted(worst): print(f'  {k}: reports {cases[k]} max ratio {worst[k]:.3f}  [{wc[k]}]')
