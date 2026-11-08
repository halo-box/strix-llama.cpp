#!/bin/bash
# W11 smoke: the real model through a pre-W11 build (BASE, default the W9-head copy ~/dev/wt_lab/w10/base_bin, which W10's smoke
# showed KLD 0 against the W10 head) and through this build, llama-perplexity c8192 b/ub 4096, one chunk; BASE's logits are the
# KL-divergence base. Arms per KV type T:
#   auto      this build, -ctk T (q4_0 / iq4_nl: indexer q8_0; f16 / q8_0: indexer unchanged -> KLD 0 expected)
#   override  this build, -ctk T --cache-type-k-idx T (the pre-W11 indexer; KLD 0 expected: W11 changes nothing else)
# Reports PPL, KLD, top-1 agreement, the resolved indexer type line and the indexer KV buffer line. Base files (~2 GB) deleted.
. ~/dev/wt_rewrite/scripts/qsa-validation/harness/w7/env.sh
BASE=${BASE:-~/dev/wt_lab/w10/base_bin}
NEW=${NEW:-$T/build/bin}
OUT=${OUT:-~/dev/wt_lab/w11/smoke}; mkdir -p $OUT
PPL="-m $M -f $WIKI -c 8192 -b 4096 -ub 4096 --chunks 1 $COMMON"
show() { echo "$(grep -E 'Mean PPL\(Q\)|Mean    KLD|Maximum KLD|Same top p' $1 | tr -s ' ' | tr '\n' ';') | $(grep -hE 'indexer K cache type|creating indexer KV cache' $1 | sed 's/^[0-9.]* I //' | tr '\n' ';') | $(grep -h 'QSA ctx=main' $1 | head -1 | sed 's/^[0-9.]* I //' | cut -c1-200)"; }
for kv in ${TYPES:-q4_0 f16 q8_0}; do
  b=$OUT/base_$kv.kld
  flock ~/gpu.lock env LD_LIBRARY_PATH=$BASE timeout 3600 $BASE/llama-perplexity $PPL -ctk $kv -ctv $kv --kl-divergence-base $b > $OUT/base_$kv.log 2>&1
  echo "SMOKE $kv base rc=$? $(grep -h 'Final estimate' $OUT/base_$kv.log)"
  flock ~/gpu.lock timeout 3600 $NEW/llama-perplexity $PPL -ctk $kv -ctv $kv --kl-divergence-base $b --kl-divergence > $OUT/w11_${kv}_auto.log 2>&1
  echo "SMOKE $kv auto rc=$? $(show $OUT/w11_${kv}_auto.log)"
  if [ $kv = q4_0 ] || [ $kv = iq4_nl ]; then
    flock ~/gpu.lock timeout 3600 $NEW/llama-perplexity $PPL -ctk $kv -ctv $kv --cache-type-k-idx $kv --kl-divergence-base $b --kl-divergence > $OUT/w11_${kv}_override.log 2>&1
    echo "SMOKE $kv override rc=$? $(show $OUT/w11_${kv}_override.log)"
  fi
  rm -f $b
done
echo SMOKE_DONE
