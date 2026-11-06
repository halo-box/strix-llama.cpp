#!/bin/bash
# build the W4 harness against the worktree's kernel sources (copied; the common.cuh include is the only edit) and the
# R2-BOP harness head (bands_dec.cu lines 1-281, verbatim); under flock ~/gpu.lock (one build slot)
set -e
D=~/dev/wt_lab/w4/bands; S=~/dev/wt_rewrite/ggml/src/ggml-cuda; B=~/dev/wt_lab/bands/op; mkdir -p $D/src; cd $D
cp $S/sparse-attn-q8.cuh $S/sparse-attn-loaders.cuh $S/sparse-attn-attend-g1.cuh src/
sed 's/^#include "common.cuh"$/\/\/ (harness: common.cuh replaced by the w4_dec.cu prelude)/' $S/sparse-attn-attend-g4.cuh > src/sparse-attn-attend-g4.cuh
sed -n '/^static constexpr int SA_PACK_THREADS/,/^void ggml_cuda_sparse_attn_pack_v/p' $S/sparse-attn-pack.cu | sed '$d' > src/sparse-attn-pack.inc
sed -n '/^static __global__ void __launch_bounds__(256) sa_merge_splits/,/^}/p' $S/sparse-attn-merge.cu > src/sparse-attn-merge.inc
sed -n '1,281p' $B/bands_dec.cu > src/bands_dec_head.inc
cp $B/bands_ref.h src/
cp ~/dev/wt_rewrite/tests/sparse-attn-ref.h src/
md5sum $S/sparse-attn-attend-g1.cuh $S/sparse-attn-attend-g4.cuh $S/sparse-attn-loaders.cuh $S/sparse-attn-q8.cuh $S/sparse-attn-pack.cu $S/sparse-attn-merge.cu $B/bands_dec.cu $B/bands_ref.h w4_dec.cu
wc -l src/*.inc
/opt/rocm/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I src -I ~/dev/wt_lab/c18 -I ~/dev/wt_rewrite/ggml/include \
  -Rpass-analysis=kernel-resource-usage w4_dec.cu -o w4_dec -L ~/dev/wt_rewrite/build/bin -lggml-base -Wl,-rpath,$HOME/dev/wt_rewrite/build/bin > build.log 2>&1
echo BUILD $?
