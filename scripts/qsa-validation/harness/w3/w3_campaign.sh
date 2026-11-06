#!/bin/bash
# W3 kernel-time campaign (every GPU run under flock ~/gpu.lock): C18 rows with SYN data, r1 / r2 as separate processes;
# then the synthetic C6/C18 plans for the V-loader crossover (n_q 512 / 1024 / 4096 x spread)
cd ~/dev/wt_lab/w3/c18; mkdir -p res
g() { local tag=$1; shift; flock ~/gpu.lock ./w3_attn "$@" > res/$tag.out 2>&1; echo "EXIT $tag $?"; }
for r in r1 r2; do
  g wiki4k_$r   ~/dev/wt_lab/c18/cap/wikiS_p4096.rows   10 0 f16,bf16,q8_0,q4_0 PD all
  g wiki131k_$r ~/dev/wt_lab/c18/cap/wikiL_p131072.rows 10 0 f16,bf16,q8_0,q4_0 PD all
done
for nq in 512 1024 4096; do
  for nk in 16384 32768 65536 131072 262144 1048576; do
    g syn_${nq}_${nk} SYN:1.8:$nk:1 10 $nq f16,bf16,q8_0,q4_0 PD
  done
done
echo DONE
