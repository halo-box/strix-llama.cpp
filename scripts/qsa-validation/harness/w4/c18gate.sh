#!/bin/bash
# W4 done-when "G1 kernel gates on the C18 decode fixtures": the new G1 attend against C18's decode harness kernel with
# the shipped loaders (c18_dec_kernel, K D|X + V S, PR launch) and the PR-verbatim decode, same process, C18 protocol
# (campaign_dec.sh: n_q 512 10 reps, n_q 1 50 reps; 3 warm-ups; hipEvent per launch; mean of 12 per-layer medians),
# wiki d131k rows (C18 cap) with SYN data, two processes r1 / r2; every run under flock ~/gpu.lock
cd ~/dev/wt_lab/w4/bands || exit 1; mkdir -p c18
for r in r1 r2; do
  for f in wikiL_p131072 randL_p131072; do
    flock ~/gpu.lock ./w4_dec ~/dev/wt_lab/c18/cap/$f.rows SYN 512 c18gate dec512_${f}_$r time 10 > c18/dec512_${f}_$r.out 2>&1; echo "EXIT dec512_${f}_$r $?"
    flock ~/gpu.lock ./w4_dec ~/dev/wt_lab/c18/cap/$f.rows SYN 1 c18gate dec1_${f}_$r time 50 > c18/dec1_${f}_$r.out 2>&1; echo "EXIT dec1_${f}_$r $?"
  done
done
echo DONE
