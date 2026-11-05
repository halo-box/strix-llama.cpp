#!/bin/bash
# C18 decode-route campaign (qsa_decode_wmma_partial, n_q <= 512): first n_q queries of each captured layer
cd ~/dev/wt_lab/c18
until grep -q "^DONE" logs/campaign.out; do sleep 30; done
run() { # tag rows data nq reps
  echo "START $1 $(date +%T)"
  flock ~/gpu.lock ./c18_dec $2 $3 $4 $5 all all > res/$1.out 2> logs/$1.err
  echo "EXIT $1 $? $(date +%T)"
}
for r in r1 r2; do
  run dec512_wiki4k_$r   cap/wikiS_p4096.rows   cap/wikiS_p4096.qkv   512 10
  run dec512_rand4k_$r   cap/randS_p4096.rows   cap/randS_p4096.qkv   512 10
  run dec512_wiki131k_$r cap/wikiL_p131072.rows cap/wikiL_p131072.qkv 512 10
  run dec512_rand131k_$r cap/randL_p131072.rows SYN                   512 10
  run dec1_wiki131k_$r   cap/wikiL_p131072.rows cap/wikiL_p131072.qkv 1 50
  run dec1_rand131k_$r   cap/randL_p131072.rows SYN                   1 50
done
echo DONE $(date +%T)
