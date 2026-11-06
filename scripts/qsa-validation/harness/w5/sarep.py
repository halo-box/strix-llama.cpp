import re,sys,collections
# summary of SA_REPORT lines: per tier/class family, case count, max NMSE/band ratio and the case reaching it
fam=collections.defaultdict(lambda:[0,-1.0,''])
for line in open(sys.argv[1], errors='replace'):
    if not line.startswith('SA_REPORT backend='): continue
    m=re.search(r'case=(\S+)\((.*?)\) class=(\S*) band=(\S+) nmse=(\S+) ratio=(\S+)',line)
    if not m: continue
    tag,case,cls,band,nmse,ratio=m.groups(); ratio=float(ratio)
    k0=cls.split(',')[0]
    parts=k0.split('/')
    if k0.startswith('sa_attend/G4'): f='fast G4'
    elif k0.startswith('sa_attend/G1'): f='fast G1 '+parts[4]
    elif k0.startswith('sa_generic/G4'): f='generic G4 '+parts[2]
    elif k0.startswith('sa_generic/G1'): f='generic G1 '+parts[2]
    elif k0.startswith('sa_portable'): f='portable '+parts[1]
    else: f='other:'+k0
    f=tag+' '+f
    e=fam[f]; e[0]+=1
    if ratio>e[1]: e[1]=ratio; e[2]=case[:150]
tot=sum(v[0] for v in fam.values())
for f,(n,r,c) in sorted(fam.items()): print("%-40s cases=%4d max_ratio=%.3f  %s"%(f,n,r,c))
print("TOTAL reports",tot,"max ratio %.3f"%max(v[1] for v in fam.values()))
