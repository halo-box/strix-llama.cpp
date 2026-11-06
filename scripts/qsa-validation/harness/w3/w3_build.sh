#!/bin/bash
# build the W3 kernel-time harness against the worktree's kernel sources (copied; the common.cuh include is the only edit)
set -e
D=~/dev/wt_lab/w3/c18; S=~/dev/wt_rewrite/ggml/src/ggml-cuda; mkdir -p $D/src; cd $D
cp $S/sparse-attn-q8.cuh $S/sparse-attn-loaders.cuh src/
sed 's/^#include "common.cuh"$/\/\/ (harness: common.cuh replaced by the w3_attn.cu prelude)/' $S/sparse-attn-attend-g4.cuh > src/sparse-attn-attend-g4.cuh
diff $S/sparse-attn-attend-g4.cuh src/sparse-attn-attend-g4.cuh | grep -c '^[<>]' | xargs echo "attend header lines changed:"
sed -n '/^static constexpr int SA_PACK_THREADS/,/^void ggml_cuda_sparse_attn_pack_v/p' $S/sparse-attn-pack.cu | sed '$d' > src/sparse-attn-pack.inc
md5sum $S/sparse-attn-attend-g4.cuh $S/sparse-attn-loaders.cuh $S/sparse-attn-q8.cuh $S/sparse-attn-pack.cu w3_attn.cu
/opt/rocm/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -DGGML_USE_HIP -I src -I ~/dev/wt_lab/c18 -Rpass-analysis=kernel-resource-usage w3_attn.cu -o w3_attn > build.log 2>&1
echo BUILD $?
python3 - build.log <<'PY'
import re,sys
t=open(sys.argv[1]).read(); names=re.findall(r'Function Name: (\S+)',t); sp=re.findall(r'VGPRs Spill: (\d+)',t); sc=re.findall(r'ScratchSize \[bytes/lane\]: (\d+)',t)
print("kernels", len(names), "spilling", sum(int(x)>0 for x in sp), "scratch>0", sum(int(x)>0 for x in sc))
PY
