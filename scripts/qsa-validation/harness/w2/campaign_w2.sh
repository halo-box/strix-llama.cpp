#!/bin/bash
# W2 done-when (PLAN-v3 §13 W2): the C3 plan-stage harness on recaptured d131k rows, on the rewrite's HIP op.
# Each run under flock ~/gpu.lock. reps 20 (C3: n = 20), R = 8 plan nodes per graph, G 4 (gated) and 1 (reported).
cd ~/dev/wt_lab/w2; mkdir -p res
r() { local t=$1; shift; echo "START $t $(date +%T)"; flock ~/gpu.lock env "$@" > res/$t.out 2> res/$t.err; echo "EXIT $t $? $(date +%T)"; }
for f in text_f16_p131072 rand_f16_p131072 text_q8_0_p131072; do
  r ${f}_r1 ./bin/w2_plan cap/$f.rows 20 8 0 4,1
  r ${f}_r2 ./bin/w2_plan cap/$f.rows 20 8 0 4,1
  r ${f}_nographs GGML_CUDA_DISABLE_GRAPHS=1 ./bin/w2_plan cap/$f.rows 20 8 0 4,1
done
# C3's shuffled control (seed 7): every row permuted, the sort path; reported, not gated (C3 reported it the same way)
for f in text_f16_p131072 rand_f16_p131072; do
  r ${f}_shuf7 ./bin/w2_plan cap/$f.rows 10 8 7 4,1
done
echo DONE $(date +%T)
