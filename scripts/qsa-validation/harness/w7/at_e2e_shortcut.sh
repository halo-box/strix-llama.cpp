#!/bin/bash
# AT-W7PERF e2e (a) (PLAN-v3 §13 W7, §16.5): C10 text llama-bench, one build, sparse vs dense via LLAMA_QSA_PREFER_DENSE=1, interleaved
# R N R N R N R per pair: pp512 / pp2048 at d0, ub {64, 512, 1024, 2048}, and tg128 at d0 / d1024 / d1900.
. "$(dirname "$0")/env.sh"
OUT=$W7/at_e2e; mkdir -p $OUT
BB=$W7/bench/bin_w7/llama-bench
TXT=$WIKI,$HOME/dev/wt_lab/c10/corp/code.txt
PROCS=${PROCS:-0 1 2 3 4 5 6}
one() { # tag dense k v
  local tag=$1 dense=$2 k=$3 v=$4 de="LLAMA_QSA_PREFER_DENSE=$2"
  flock ~/gpu.lock env C10_TEXT=$TXT $de $BB -m $M -ngl 99 -fa 1 -lzm on-direct -ctk $k -ctv $v -b 4096 -ub 64,512,1024,2048 -p 512,2048 -n 0 -d 0 -r 5 --progress -o json > $OUT/${tag}_pp.json 2> $OUT/${tag}_pp.err; local a=$?
  flock ~/gpu.lock env C10_TEXT=$TXT $de $BB -m $M -ngl 99 -fa 1 -lzm on-direct -ctk $k -ctv $v -b 4096 -ub 512 -p 0 -n 128 -d 0,1024,1900 -r 5 --progress -o json > $OUT/${tag}_tg.json 2> $OUT/${tag}_tg.err; local b=$?
  echo "$(date +%T) $tag pp_rc=$a tg_rc=$b"
}
# warm-up before the first timed process of every pair (§16 design)
for p in ${PAIRS:-q8_0:q8_0 q4_0:q4_0 bf16:bf16 q8_0:f16 f16:q8_0}; do
  k=$(pair_k $p); v=$(pair_v $p); t=$(pair_tag $p)
  flock ~/gpu.lock env C10_TEXT=$TXT $BB -m $M -ngl 99 -fa 1 -lzm on-direct -ctk $k -ctv $v -b 4096 -ub 512 -p 512 -n 0 -d 0 -r 2 -o json > /dev/null 2>&1
  for i in $PROCS; do
    if [ $((i % 2)) = 0 ]; then one ${t}_${i}_dense 1 $k $v; else one ${t}_${i}_sparse 0 $k $v; fi
  done
done
echo AT_E2E_DONE
