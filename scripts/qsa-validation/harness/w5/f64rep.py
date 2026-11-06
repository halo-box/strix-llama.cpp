import re,sys,collections
# per (tier, K/V type of the case) the max GPU-vs-f64 errors of the SA_REPORT lines (reported, never gated, PLAN-v3 §14.2)
d=collections.defaultdict(lambda:[0,0.0,0.0,0.0])
for line in open(sys.argv[1], errors='replace'):
    if not line.startswith('SA_REPORT backend='): continue
    m=re.search(r'case=SPARSE_ATTN\((type_k=(\w+),type_v=(\w+).*?)\) class=(\S*) band=\S+ nmse=(\S+) ratio=(\S+) f64_c18=(\S+) f64_plan=(\S+)',line)
    if not m: continue
    _,tk,tv,cls,nmse,ratio,c18,pl=m.groups()
    tier='generic' if cls.startswith('sa_generic') else 'portable' if cls.startswith('sa_portable') else 'fast'
    for t in {tk,tv}:
        e=d[(tier,t)]; e[0]+=1; e[1]=max(e[1],float(c18)); e[2]=max(e[2],float(pl)); e[3]=max(e[3],float(ratio))
for (tier,t),(n,c,p,r) in sorted(d.items()): print("%-9s %-7s cases=%4d max_f64_c18=%.3e max_f64_plan=%.3e max_nmse/band=%.3f"%(tier,t,n,c,p,r))
