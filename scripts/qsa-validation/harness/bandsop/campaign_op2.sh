#!/bin/bash
# R2-M-LAB-BANDS-OP follow-up: (E1) rows sorted ascending, (E2) uniform +-1.7 data (the SYN amplitude) on the PR rows,
# (E3) SYN 262k n_q 1 seeds 2,3 (seed spread of the SYN split-K row). Defects off (E1 moves the -1 slots). flock per run.
cd ~/dev/wt_lab/bands/op
V_DEC=PR-dec-f16,PR-dec-q8_0
g() { local t=$1; shift; echo "START $t $(date +%T)"; flock ~/gpu.lock env "$@" > res2/$t.out 2> logs/$t.err; echo "EXIT $t $? $(date +%T)"; }
mkdir -p res2
for seed in 1 2 3; do
  for s in 512:2051:4096:0 512:257:4096:0 128:17:512:0 129:257:4096:0 64:257:4096:0 1:2051:4096:0 9:2051:4096:0 64:2051:4096:0 130:2051:4096:40 9:2051:270336:66000 1:2051:32768:8000; do
    IFS=: read nq ns nk bl <<< "$s"
    g e1_${nq}_${ns}_${nk}_b${bl}_s${seed} BANDS_DEFECTS=0 ./bands_dec UNI:1.0:$nk:$seed:$nq:$ns:$bl:1 SYN 0 $V_DEC e1_${nq}_${ns}_${nk}_b${bl}_s${seed} 32
    g e2_${nq}_${ns}_${nk}_b${bl}_s${seed} BANDS_DEFECTS=1 ./bands_dec UNI:1.7:$nk:$seed:$nq:$ns:$bl SYN 0 $V_DEC e2_${nq}_${ns}_${nk}_b${bl}_s${seed} 32
  done
  g e2_1_2051_262144_b-1_s${seed} BANDS_DEFECTS=1 ./bands_dec UNI:1.7:262144:$seed:1:2051:-1 SYN 0 $V_DEC e2_1_2051_262144_b-1_s${seed} 32
  g e2_512_2051_262144_b-1_s${seed} BANDS_DEFECTS=1 ./bands_dec UNI:1.7:262144:$seed:512:2051:-1 SYN 0 $V_DEC e2_512_2051_262144_b-1_s${seed} 32
done
for seed in 2 3; do
  g syn_dec1_s$seed BANDS_DEFECTS=1 ./bands_dec SYN:1.8:262144:$seed SYN 1 $V_DEC syn_dec1_s$seed 32
  g syn_dec512_s$seed BANDS_DEFECTS=1 ./bands_dec SYN:1.8:262144:$seed SYN 512 $V_DEC syn_dec512_s$seed 32
done
echo DONE $(date +%T)
