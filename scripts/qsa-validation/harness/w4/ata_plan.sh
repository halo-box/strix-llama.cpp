#!/bin/bash
# AT-A (d): plan-stage cost at each n_q, G1 and G4, on the C18 wiki d4k / d131k rows cut to the first n_q queries, with
# the W2 op-level plan harness (w2_plan: graph of R = 8 plan nodes, wall time / 8, 20 reps after 3 warm-ups; bytes vs
# CPU and the C3 oracle); r1 / r2; every run under flock ~/gpu.lock
cd ~/dev/wt_lab/w4/bands || exit 1; mkdir -p ataplan rows
for fx in wikiS_p4096 wikiL_p131072; do
  for nq in 2 4 8 16 32 64 96 127 128 256 512; do
    [ -f rows/${fx}_$nq.rows ] || python3 trunc_rows.py ~/dev/wt_lab/c18/cap/$fx.rows rows/${fx}_$nq.rows $nq
  done
done
for r in r1 r2; do
  for fx in wikiS_p4096 wikiL_p131072; do
    for nq in 2 4 8 16 32 64 96 127 128 256 512; do
      flock ~/gpu.lock ~/dev/wt_lab/w2/bin/w2_plan rows/${fx}_$nq.rows 20 8 0 4,1 > ataplan/${fx}_${nq}_$r.out 2>&1; echo "EXIT ${fx}_${nq}_$r $?"
    done
  done
done
echo DONE
