#!/bin/bash
# W4 (U34): the PR-verbatim decode (R2-BOP bands_dec, PR-dec-f16 / PR-dec-q8_0 = qsa_decode_wmma_partial + qsa_decode_merge
# verbatim) on UNI fixtures (test-backend-ops inputs, PR cell formula rows) for the split classes without a band row
# (6-16, > 33), 3 seeds, planted defects on. Binary: ~/dev/wt_lab/bands/op/bands_dec (R2-BOP, md5 of bands_dec.cu
# 22390af8..., bands_ref.h f5cbf8ee... = EV/bandsop/src). Output paths are absolute (resolved before the cd).
O=$(realpath -m ~/dev/wt_lab/w4/prband/uni); mkdir -p $O
cd ~/dev/wt_lab/bands/op || exit 1
V=PR-dec-f16,PR-dec-q8_0
S="1:513:16384:0 2:513:16384:0 8:513:16384:0 20:2051:8192:0 24:2051:8192:0 32:2051:8192:0 43:2051:8192:0 64:4096:16384:0
   1:4096:16384:0 4:4096:16384:0 9:4096:16384:0 1:2560:16384:0"
for seed in 1 2 3; do for s in $S; do
  IFS=: read nq ns nk bl <<< "$s"; tag=u_${nq}_${ns}_${nk}_b${bl}_s${seed}
  flock ~/gpu.lock env BANDS_DEFECTS=1 ./bands_dec UNI:1.0:$nk:$seed:$nq:$ns:$bl SYN 0 $V $tag 32 > $O/$tag.out 2> $O/$tag.err
  echo "EXIT $tag $? $(grep "^CASE" $O/$tag.out | head -1)"
done; done
echo DONE
