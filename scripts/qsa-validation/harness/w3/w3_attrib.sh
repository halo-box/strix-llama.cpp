#!/bin/bash
# W3 lab (not shipped): attribution of the new G4 kernel's time against C18's reimplementation. Three binaries from the
# shipped sources: ship (as built by build.sh), nocheck (the non-finite check's ballot condition replaced by false, so
# the check and the flag path compile away), nofix (the fixup launch removed from the harness). Same fixture, layers
# 0,5,11 of wiki d4k and d131k, 10 reps, separate processes; every build and run under flock ~/gpu.lock.
set -e
B=~/dev/wt_lab/w3/c18; D=~/dev/wt_lab/w3/attrib; mkdir -p $D/nocheck $D/nofix $D/res; cd $D
cp $B/src/* nocheck/; cp $B/src/* nofix/
sed -i 's/__ballot((__builtin_bit_cast(uint32_t, nf) \& 0x7fff7fffu) != 0)/__ballot(false)/' nocheck/sparse-attn-attend-g4.cuh
grep -c '__ballot(false)' nocheck/sparse-attn-attend-g4.cuh
sed 's/ FX\[v.T\]<<<dim3(ng4, 2, 1), 256>>>(a, 12);//' $B/w3_attn.cu > nofix/w3_attn.cu; cp $B/w3_attn.cu nocheck/w3_attn.cu
grep -c 'FX\[v.T\]<<<' nofix/w3_attn.cu || true
for v in nocheck nofix; do (cd $v && flock ~/gpu.lock /opt/rocm/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -DGGML_USE_HIP -I . -I ~/dev/wt_lab/c18 w3_attn.cu -o w3_attn && echo "BUILD $v 0"); done
for fx in wikiS_p4096 wikiL_p131072; do
  for v in ship nocheck nofix; do
    bin=$D/$v/w3_attn; [ $v = ship ] && bin=$B/w3_attn
    flock ~/gpu.lock $bin ~/dev/wt_lab/c18/cap/$fx.rows 10 0 f16,q8_0 P 0,5,11 > res/${fx}_$v.out 2>&1; echo "EXIT ${fx}_$v $?"
  done
done
for f in res/*.out; do echo "$f $(grep -E '^SUM (C18|PR-V0|NEW-f16-P|NEW-q8_0-P|PACKV)' $f | awk '{printf "%s=%s ", $2, $3}')"; done
