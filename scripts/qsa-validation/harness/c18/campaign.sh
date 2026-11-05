#!/bin/bash
# C18 campaign: every variant, interleaved per round, 3 warm-up + 10 timed rounds per layer; two separate processes (r1, r2)
cd ~/dev/wt_lab/c18
run() { # tag rows data layers
  echo "START $1 $(date +%T)"
  flock ~/gpu.lock ./c18_attn $2 $3 10 $4 all > res/$1.out 2> logs/$1.err
  echo "EXIT $1 $? $(date +%T)"
}
for r in r1 r2; do
  run wiki4k_$r   cap/wikiS_p4096.rows   cap/wikiS_p4096.qkv   all
  run rand4k_$r   cap/randS_p4096.rows   cap/randS_p4096.qkv   all
  run wiki131k_$r cap/wikiL_p131072.rows cap/wikiL_p131072.qkv all
  run rand131k_$r cap/randL_p131072.rows SYN                   all
done
run wiki131kSYN_r1 cap/wikiL_p131072.rows SYN 0,5,11
for s in 1 2; do for nk in 4096 32768 131072 262144 1048576; do
  run syn_u18_${nk}_s$s SYN:1.8:$nk:$s SYN all
done; done
echo DONE $(date +%T)
