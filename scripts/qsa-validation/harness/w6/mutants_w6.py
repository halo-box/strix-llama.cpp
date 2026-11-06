#!/usr/bin/env python3
# W6 mutants: each edits the tree, builds build-vk, runs the check that must catch it, restores the file. Run on a clean tree (git diff must be empty at the end).
import os,subprocess,sys,re
R=os.path.expanduser('~/dev/wt_rewrite'); L=os.path.expanduser('~/dev/wt_rewrite_logs/W6')
os.chdir(R)
VK='ggml/src/ggml-vulkan/ggml-vulkan.cpp'; S='ggml/src/ggml-vulkan/vulkan-shaders/'
T='build-vk/bin/test-backend-ops'; VT='build-vk/bin/test-sparse-attn-vk'
TOOLS=os.path.expanduser('~/dev/wt_lab/w6/tools/usr')
VAL=f'VK_LAYER_PATH={TOOLS}/share/vulkan/explicit_layer.d VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT LD_LIBRARY_PATH={TOOLS}/lib64:{R}/build-vk/bin'
def sh(c,log):
    return subprocess.run(f'{c} > {log} 2>&1',shell=True).returncode
def build():
    return sh("flock ~/gpu.lock cmake --build build-vk -j 16 --target test-backend-ops test-sparse-attn-vk",f'{L}/mut_build.log')
muts=[
 ('sync_attend',VK,'''    // O17: the workspace is invisible to the barrier tracker of ggml_vk_build_graph
    ggml_vk_sync_buffers(ctx, subctx);
''','',f'flock ~/gpu.lock timeout 300 {VT} stress'),
 ('sync_plan',VK,'''    // O17: the scratch is invisible to the barrier tracker of ggml_vk_build_graph
    ggml_vk_sync_buffers(ctx, subctx);
''','',f'flock ~/gpu.lock timeout 300 {VT} stress'),
 ('short_scratch',VK,'''    const vk_subbuffer scr_buf = ggml_vk_sa_ws_subbuffer(ctx, dst, dst_buf, 0, scratch);
''','''    const vk_subbuffer scr_buf = ggml_vk_sa_ws_subbuffer(ctx, dst, dst_buf, 0, scratch / 2);
''',f'env {VAL} flock ~/gpu.lock timeout 600 {T} test -o SPARSE_ATTN_PLAN -b Vulkan0 -j 1 -p "n_q=8,n_sel=2051,n_kv=65536,group=1"'),
 ('no_sort',S+'sparse_attn_plan_rows.comp','if (unsorted_sh != 0u) {','if (false) {',f'flock ~/gpu.lock timeout 300 {T} test -o SPARSE_ATTN_PLAN -b Vulkan0 -j 1'),
 ('v_select',S+'sparse_attn_cm.comp','''                        v = uvec4(0);
                    }
''','''                    }
''',f'flock ~/gpu.lock timeout 300 {T} test -o SPARSE_ATTN -b Vulkan0 -j 1 -p "rows=poison"'),
 ('score_select',S+'sparse_attn_cm.comp','s[i] = ((vm >> (grp + 4 * i)) & 1u) != 0 ? S[i] : NEG_INF;','s[i] = S[i];',f'flock ~/gpu.lock timeout 300 {T} test -o SPARSE_ATTN -b Vulkan0 -j 1 -p "rows=poison"'),
 ('rot_q_butterfly',S+'sparse_attn_cm.comp','reg[j + k + st] = x - y;','reg[j + k + st] = y - x;',f'flock ~/gpu.lock timeout 300 {VT} rot'),
 ('q8_v_scale',S+'sparse_attn_cm.comp','const f16vec2 d2 = sa_d2(dw);','const f16vec2 d2 = sa_d2(dw ^ 1u);',f'flock ~/gpu.lock timeout 300 {T} test -o SPARSE_ATTN -b Vulkan0 -j 1 -p "type_k=f16,type_v=q8_0,n_q=9,n_kv=4096,n_sel=257"'),
]
for name,f,a,b,cmd in muts:
    src=open(f).read()
    assert src.count(a)>=1,(name,a)
    open(f,'w').write(src.replace(a,b,1))
    rc=build()
    log=f'{L}/mut_{name}.log'
    r=sh(cmd,log) if rc==0 else -1
    txt=open(log,errors='replace').read() if rc==0 else ''
    fails=len(re.findall(r'FAIL',txt)); oob=len(re.findall(r'VUID-[A-Za-z0-9_-]+|out of bounds|OOB',txt))
    print(f'MUT {name}: build {rc} check rc {r} FAIL-lines {fails} validation hits {oob}; {[l for l in txt.splitlines() if "tests passed" in l or "failed checks" in l][:2]}',flush=True)
    open(f,'w').write(src)
print('restore build',build(),flush=True)
print('MUT_DONE',flush=True)
