#!/bin/bash
# W10 smoke: the real model through the W9 head (graph-side rotation, saved build/bin copy BASE) and through this build
# (in-op rotation), llama-perplexity c8192 b/ub 4096 (compact CB G4 rows at n_kv 8192), one chunk; the W9 logits are the
# KL-divergence base of the W10 run. Reports PPL of both and KLD / top-1 agreement. The base file (~2 GB) is deleted.
. ~/dev/wt_rewrite/scripts/qsa-validation/harness/w7/env.sh
BASE=${BASE:-~/dev/wt_lab/w10/base_bin}
NEW=${NEW:-$T/build/bin}
OUT=${OUT:-~/dev/wt_lab/w10/smoke}; mkdir -p $OUT
for kv in ${TYPES:-q8_0 q4_0 f16}; do
  for ub in ${UBS:-4096}; do
    b=$OUT/base_${kv}_u$ub.kld
    flock ~/gpu.lock env LD_LIBRARY_PATH=$BASE timeout 3600 $BASE/llama-perplexity -m $M -f $WIKI -c 8192 -b 4096 -ub $ub --chunks 1 $COMMON -ctk $kv -ctv $kv --kl-divergence-base $b > $OUT/w9_${kv}_u$ub.log 2>&1
    r1=$?
    flock ~/gpu.lock timeout 3600 $NEW/llama-perplexity -m $M -f $WIKI -c 8192 -b 4096 -ub $ub --chunks 1 $COMMON -ctk $kv -ctv $kv --kl-divergence-base $b --kl-divergence > $OUT/w10_${kv}_u$ub.log 2>&1
    r2=$?
    echo "SMOKE $kv ub$ub rc=$r1/$r2 w9: $(grep -h 'Final estimate' $OUT/w9_${kv}_u$ub.log | sed 's/.*Final/Final/') | w10: $(grep -E 'Mean PPL\(Q\)|Mean    KLD|Maximum KLD|Same top p' $OUT/w10_${kv}_u$ub.log | tr -s ' ' | tr '\n' ';')"
    rm -f $b
  done
done
echo SMOKE_DONE
