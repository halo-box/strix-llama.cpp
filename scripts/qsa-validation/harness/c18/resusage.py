import re,sys
L=open(sys.argv[1]).read().splitlines()
T={0:"f16",1:"q8_0",2:"q4_0",3:"bf16",4:"bf16w"}; LN={0:"D",1:"Dd",2:"X",3:"S",4:"P"}
cur=None; rows=[]
for l in L:
    m=re.search(r"Function Name: (\S+)",l)
    if m: cur={"n":m.group(1)}; rows.append(cur); continue
    for k,kk in [("VGPRs: ","V"),("ScratchSize \\[bytes/lane\\]: ","S"),("Occupancy \\[waves/SIMD\\]: ","O"),("LDS Size \\[bytes/block\\]: ","L")]:
        m=re.search(k+r"(\d+)",l)
        if m and cur is not None: cur[kk]=int(m.group(1))
for r in rows:
    n=r["n"]; m=re.match(r"_Z15c18_attn_kernelILi(\d)ELi(\d)ELi(\d)ELi(\d)E",n)
    if m: a=list(map(int,m.groups())); name="K%s%s-V%s%s"%(T[a[0]],LN[a[1]],T[a[2]],LN[a[3]])
    elif "qsa3_attn" in n:
        m=re.search(r"ILb(\d)ELi8ELb(\d)ELb(\d)E",n); name="PR-%s-%s"%("VD" if m.group(2)=="1" else "V0","q8_0" if m.group(3)=="1" else "f16")
    elif "pack" in n: name=n[:30]
    else: continue
    print("%-22s VGPR %3d scratch %3d occ %2d LDS %d"%(name,r.get("V",-1),r.get("S",-1),r.get("O",-1),r.get("L",-1)))
