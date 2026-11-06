#!/usr/bin/env python3
# W6: positive controls of the validation-layer runs: a mutant that binds half of the plan scratch (GPU-AV must report it) and mutants without the sync rule (synchronization validation must report them).
import os,subprocess,re
R=os.path.expanduser('~/dev/wt_rewrite'); L=os.path.expanduser('~/dev/wt_rewrite_logs/W6')
os.chdir(R)
VK='ggml/src/ggml-vulkan/ggml-vulkan.cpp'
T='build-vk/bin/test-backend-ops'; VT='build-vk/bin/test-sparse-attn-vk'
TOOLS=os.path.expanduser('~/dev/wt_lab/w6/tools/usr')
BASE=f'VK_LAYER_PATH={TOOLS}/share/vulkan/explicit_layer.d VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation LD_LIBRARY_PATH={TOOLS}/lib64:{R}/build-vk/bin'
GPUAV=f'{BASE} VK_LAYER_GPUAV_ENABLE=1 VK_LAYER_VALIDATE_CORE=0 VK_LAYER_PRINTF_ENABLE=0'
SYNCV=f'{BASE} VK_LAYER_VALIDATE_SYNC=1 VK_LAYER_SYNCVAL_SUBMIT_TIME_VALIDATION=1 VK_LAYER_VALIDATE_CORE=0'
def sh(c,log): return subprocess.run(f'{c} > {log} 2>&1',shell=True).returncode
def build(): return sh("flock ~/gpu.lock cmake --build build-vk -j 16 --target test-backend-ops test-sparse-attn-vk",f'{L}/mut_build.log')
muts=[('short_scratch_gpuav',VK,'''    const vk_subbuffer scr_buf = ggml_vk_sa_ws_subbuffer(ctx, dst, dst_buf, 0, scratch);
''','''    const vk_subbuffer scr_buf = ggml_vk_sa_ws_subbuffer(ctx, dst, dst_buf, 0, scratch / 2);
''',f'env {GPUAV} flock ~/gpu.lock timeout 600 {T} test -o SPARSE_ATTN_PLAN -b Vulkan0 -j 1 -p "n_q=8,n_sel=2051,n_kv=65536,group=1"'),
 ('sync_attend_syncval',VK,'''    // O17: the workspace is invisible to the barrier tracker of ggml_vk_build_graph
    ggml_vk_sync_buffers(ctx, subctx);
''','',f'env {SYNCV} flock ~/gpu.lock timeout 300 {VT} stress'),
 ('sync_plan_syncval',VK,'''    // O17: the scratch is invisible to the barrier tracker of ggml_vk_build_graph
    ggml_vk_sync_buffers(ctx, subctx);
''','',f'env {SYNCV} flock ~/gpu.lock timeout 300 {VT} stress')]
for name,f,a,b,cmd in muts:
    src=open(f).read(); assert src.count(a)==1,name
    open(f,'w').write(src.replace(a,b,1))
    rc=build(); log=f'{L}/mut_{name}.log'; r=sh(cmd,log) if rc==0 else -1
    txt=open(log,errors='replace').read() if rc==0 else ''
    print(f'MUT {name}: build {rc} rc {r}; SYNC-HAZARD {len(re.findall("SYNC-HAZARD",txt))}; VUID {len(re.findall("VUID-",txt))}; GPU-AV-ish {len(re.findall("out of bounds|OOB|storageBuffers-",txt,re.I))}',flush=True)
    open(f,'w').write(src)
print('restore',build(),flush=True)
print('MUTV_DONE',flush=True)
