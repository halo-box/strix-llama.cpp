#!/bin/bash
# W13 real-model checks: the W7 guard harness (output redirected to wt_lab/w13/guard), the W11 KLD smoke (base = the W9-head
# copy ~/dev/wt_lab/w10/base_bin, as in W10 and W11), and a Vulkan one-chunk smoke with the QSA trace
H=~/dev/wt_rewrite/scripts/qsa-validation/harness
mkdir -p ~/dev/wt_lab/w13
sed 's|^OUT=\$W7/guard|OUT=~/dev/wt_lab/w13/guard|' $H/w7/guard.sh > ~/dev/wt_lab/w13/guard.sh
grep -q 'OUT=~/dev/wt_lab/w13/guard' ~/dev/wt_lab/w13/guard.sh || { echo GUARD_SED_FAILED; exit 1; }
cp $H/w7/env.sh ~/dev/wt_lab/w13/env.sh
bash ~/dev/wt_lab/w13/guard.sh > ~/dev/wt_lab/w13/guard.out 2>&1
OUT=~/dev/wt_lab/w13/smoke bash $H/w11/smoke_kld.sh > ~/dev/wt_lab/w13/smoke.out 2>&1
. $H/w7/env.sh
V=~/dev/wt_lab/w13/vk; mkdir -p $V
for kv in q8_0 f16; do
  flock ~/gpu.lock env LLAMA_QSA_TRACE=1 timeout 1800 $T/build-vk/bin/llama-perplexity -m $M -f $WIKI -c 8192 -b 4096 -ub 4096 --chunks 1 -ngl 99 -fa on -ctk $kv -ctv $kv > $V/ppl_$kv.log 2>&1
  echo "VK $kv rc=$? $(grep -h 'Final estimate' $V/ppl_$kv.log) | $(grep -h 'QSA ctx=main' $V/ppl_$kv.log | head -1 | cut -c1-200)" >> ~/dev/wt_lab/w13/vk.out
done
echo ALL_DONE >> ~/dev/wt_lab/w13/vk.out
