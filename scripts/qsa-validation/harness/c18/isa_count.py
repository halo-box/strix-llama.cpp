#!/usr/bin/env python3
# static instruction counts of the main chunk loop per kernel: CFG of the ISA, strongly connected component holding the most v_wmma
import re, sys
sys.setrecursionlimit(100000)
T={0:"f16",1:"q8_0",2:"q4_0",3:"bf16",4:"bf16w"}; LN={0:"D",1:"Dd",2:"X",3:"S",4:"P"}
src=open(sys.argv[1]).read().splitlines()
funcs={}; cur=None
for l in src:
    m=re.match(r'^(_Z[^:\s]+):', l)
    if m: cur=m.group(1); funcs[cur]=[]; continue
    if cur is not None:
        if l.startswith('.Lfunc_end'): cur=None; continue
        funcs[cur].append(l)
def name(n):
    m=re.match(r"_Z15c18_attn_kernelILi(\d)ELi(\d)ELi(\d)ELi(\d)E",n)
    if m: a=list(map(int,m.groups())); return "K%s%s-V%s%s"%(T[a[0]],LN[a[1]],T[a[2]],LN[a[3]])
    m=re.search(r"qsa3_attn_kernelILb1ELi8ELb(\d)ELb(\d)E",n)
    if m: return "PR-%s-%s"%("VD" if m.group(1)=="1" else "V0","q8_0" if m.group(2)=="1" else "f16")
    return None
cats=[('global_load',r'global_load'),('scratch',r'scratch_'),('ds_read',r'ds_(read|load)'),('ds_write',r'ds_(write|store)'),('ds_other',r'ds_(?!read|load|write|store)'),('v_permlane',r'v_permlane'),('wmma',r'v_wmma'),('valu',r'v_(?!wmma|permlane)'),('salu',r's_(?!waitcnt|barrier|cbranch|branch|nop)'),('s_waitcnt',r's_waitcnt'),('s_barrier',r's_barrier')]
rows=[]
for f,body in funcs.items():
    n=name(f)
    if not n: continue
    blocks=[]; lab2b={}; curb={'label':None,'ins':[]}; blocks.append(curb)
    for l in body:
        m=re.match(r'^(\.LBB[^:\s]+):',l)
        if m:
            curb={'label':m.group(1),'ins':[]}; blocks.append(curb); lab2b[m.group(1)]=len(blocks)-1; continue
        m=re.match(r'^\s+([a-z_][a-z0-9_]*)\b(.*)$',l)
        if m and not l.strip().startswith('.') and not l.strip().startswith(';'): curb['ins'].append((m.group(1),m.group(2)))
    succ=[]
    for i,b in enumerate(blocks):
        s=set(); fall=True
        for op,args in b['ins']:
            if op.startswith('s_cbranch') or op=='s_branch':
                t=re.search(r'(\.LBB\S+)',args)
                if t and t.group(1) in lab2b: s.add(lab2b[t.group(1)])
                if op=='s_branch': fall=False
            if op=='s_endpgm': fall=False
        if fall and i+1<len(blocks): s.add(i+1)
        succ.append(s)
    # Tarjan
    idx={}; low={}; st=[]; on=set(); sccs=[]; c=[0]
    def sc(v):
        idx[v]=low[v]=c[0]; c[0]+=1; st.append(v); on.add(v)
        for w in succ[v]:
            if w not in idx: sc(w); low[v]=min(low[v],low[w])
            elif w in on: low[v]=min(low[v],idx[w])
        if low[v]==idx[v]:
            comp=[]
            while True:
                w=st.pop(); on.discard(w); comp.append(w)
                if w==v: break
            sccs.append(comp)
    for v in range(len(blocks)):
        if v not in idx: sc(v)
    best=max(sccs,key=lambda comp: sum(1 for b in comp for op,_ in blocks[b]['ins'] if op.startswith('v_wmma')))
    ins=[op for b in best for op,_ in blocks[b]['ins']]
    cnt={k:sum(1 for op in ins if re.match(p,op)) for k,p in cats}
    rows.append((n,len(ins),cnt))
print('| kernel | loop instrs | '+' | '.join(k for k,_ in cats)+' |')
print('|---|---|'+'---|'*len(cats))
for n,t,cnt in sorted(rows):
    print('| %s | %d | %s |'%(n,t,' | '.join(str(cnt[k]) for k,_ in cats)))
