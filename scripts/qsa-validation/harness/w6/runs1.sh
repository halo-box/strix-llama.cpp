#!/bin/bash
# W6: the op runs of AT-B7 and of the done-when (one op per invocation, -j 1, under flock)
cd ~/dev/wt_rewrite
L=~/dev/wt_rewrite_logs/W6
T=build-vk/bin/test-backend-ops
run() { # tag env... : op
  local tag=$1; shift
  local op=$1; shift
  ( export "$@" GGML_SA_TEST_REPORT=1; s=$(date +%s); flock ~/gpu.lock timeout 3400 $T test -o $op -b Vulkan0 --fail-unsupported -j 1 > $L/run_${tag}_$op.log 2>&1; echo "EXIT $tag $op rc=$? secs=$(( $(date +%s)-s ))" >> $L/runs1.out )
}
flock ~/gpu.lock bash -c 'cmake --build build-vk -j 16 --target test-backend-ops test-sparse-attn-vk > ~/dev/wt_rewrite_logs/W6/vk_build_t.log 2>&1'; echo "build rc=$?" >> $L/runs1.out
run radv SPARSE_ATTN_PLAN X=1
run radv SPARSE_ATTN X=1
for F in scalar portable_sg32 portable vdirect vstaged vpack; do
  run $F SPARSE_ATTN_PLAN GGML_VK_SA_FORCE=$F
  run $F SPARSE_ATTN GGML_VK_SA_FORCE=$F
done
source ~/dev/wt_lab/w6/lvpenv.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/build-vk/bin
run lvp SPARSE_ATTN_PLAN X=1
run lvp SPARSE_ATTN X=1
echo RUNS1_DONE >> $L/runs1.out
