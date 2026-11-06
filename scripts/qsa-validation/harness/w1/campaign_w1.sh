#!/bin/bash
# W1 done-when: the CPU SPARSE_ATTN against bands_ref.h (R1-M-Lab A / R2-BOP fixtures). Each run under flock ~/gpu.lock.
cd ~/dev/wt_lab/w1; mkdir -p res
r() { local t=$1; shift; echo "START $t $(date +%T)"; flock ~/gpu.lock ./bin/w1_ocon "$@" > res/$t.out 2> res/$t.err; echo "EXIT $t $? $(date +%T)"; }
# R1-A's cpu_wiki4k (layers 0,5,11, stride 1, all pairs) and cpu_syn262k (SYN:1.8:262144:1, stride 1, all pairs) configurations
r wiki4k cap/wikiS_p4096.rows cap/wikiS_p4096.qkv 0,5,11 1 wiki4k 32 all
r syn262k SYN:1.8:262144:1 SYN all 1 syn262k 32 all
# R2-BOP UNI fixtures (campaign_op.sh shapes, seed 1)
for s in 1:257:4096:0 512:2051:4096:0 130:2051:4096:40 200:2051:65536:15000 130:2051:270336:66000 9:2051:270336:66000 4096:2051:262144:-1; do
  IFS=: read nq ns nk bl <<< "$s"
  st=1; [ $nq -gt 1024 ] && st=8
  r uni_${nq}_${ns}_${nk}_b${bl} UNI:1.0:$nk:1:$nq:$ns:$bl SYN all $st uni_${nq}_${ns}_${nk}_b${bl} 32 all
done
echo DONE $(date +%T)
