import os,subprocess,sys
D=os.path.expanduser('~/dev/wt_rewrite/ggml/src/ggml-vulkan/vulkan-shaders/')
base=dict(os.environ,VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/lvp_icd.x86_64.json')
hdr='''#version 450
layout(constant_id = 0) const uint WG = 256;
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0) readonly buffer SelBuf { int sel[]; };
layout(binding = 1) coherent buffer DstBuf { uint dst[]; };
layout(binding = 2) coherent buffer ScrBuf { uint scr[]; };
shared uint scan_sh[WG];
'''
sh={'scan':(hdr+'''void main(){ uint tid=gl_LocalInvocationIndex; scan_sh[tid]=1u; barrier(); for(uint off=1u;off<WG;off<<=1){ uint v= tid>=off? scan_sh[tid-off]:0u; barrier(); scan_sh[tid]+=v; barrier(); } dst[tid]=scan_sh[WG-1u]; }''','3',['256']),
    'plan':(open(D+'sparse_attn_plan_rows.comp').read(),'3',['256']),
    'prep':(open(D+'flash_attn_sel_prep.comp').read(),'2',[]),
    'portable':(open(D+'sparse_attn_portable.comp').read(),'5',['12','1','1','0','0']),
    'reduce':(open(D+'sparse_attn_reduce.comp').read(),'2',[]),}
for n,(s,nb,sp) in sh.items():
    open(n+'.comp','w').write(s)
    r=subprocess.run(['glslc','-fshader-stage=compute','--target-env=vulkan1.2',n+'.comp','-o',n+'.spv'],capture_output=True,text=True)
    if r.returncode: print(n,'compile fail',r.stdout[:300])
for extra in [{'GALLIUM_OVERRIDE_CPU_CAPS':'avx','GALLIVM_PERF':'no_opt'},{'GALLIVM_PERF':'no_opt,no_filter_hacks'},{'GALLIUM_OVERRIDE_CPU_CAPS':'avx','LP_NATIVE_VECTOR_WIDTH':'128','GALLIVM_PERF':'no_opt'},{'GALLIUM_OVERRIDE_CPU_CAPS':'avx','GALLIVM_PERF':'nopt'}]:
    env=dict(base,**extra); out=[]
    for n,(s,nb,sp) in sh.items():
        r=subprocess.run(['./lvp_try',n+'.spv',nb]+sp,env=env,capture_output=True,text=True)
        out.append(f'{n}:{r.returncode}')
    print(extra,' '.join(out))
