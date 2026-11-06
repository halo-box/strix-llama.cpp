#!/bin/bash
# W5 op-time campaign (every run under flock ~/gpu.lock): C18 wiki d4k / d131k rows, SYN data, every K = V type of S on the
# default tiers (S_fast: fast, S_generic: generic), then f16 / q8_0 through the generic and the portable tier, and the
# direct-V loader; two processes per arm (r1, r2)
D=~/dev/wt_lab/w5/time; C=~/dev/wt_lab/c18/cap; cd $D; mkdir -p res
g() { local tag=$1; shift; flock ~/gpu.lock env "$@" > res/$tag.out 2>&1; echo "EXIT $tag $?"; }
ALL=f16,bf16,q8_0,q4_0,f32,q4_1,q5_0,q5_1,iq4_nl
for r in r1 r2; do
  for f in wikiS_p4096 wikiL_p131072; do
    g ${f}_default_$r  ./w5_time $C/$f.rows 10 4 $ALL
    g ${f}_generic_$r  GGML_CUDA_SA_FORCE=generic  ./w5_time $C/$f.rows 10 4 f16,q8_0
    g ${f}_vdirect_$r  GGML_CUDA_SA_FORCE=vdirect  ./w5_time $C/$f.rows 10 4 $ALL
    g ${f}_portable_$r GGML_CUDA_SA_FORCE=portable ./w5_time $C/$f.rows 3 1 f16,q8_0 0,5,11
  done
done
echo TIME_DONE
