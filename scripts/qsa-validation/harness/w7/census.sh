#!/bin/bash
# W7 done-when: route census (PLAN-v3 §13 W7, §14.3) on llama-perplexity c8192 ub {16, 512, 4096} and llama-bench pp4096 @ d4096, six
# K/V pairs, HIP graphs off for the counts, checked by census_check.py against rows generated from the shipped policy.
. "$(dirname "$0")/env.sh"
BIN=${BIN:-$T/build/bin}
OUT=$W7/census; mkdir -p $OUT
CC="python3 $T/scripts/qsa-validation/census_check.py"
SEL="$BIN/test-qsa-select-dump"
rc_all=0
check() { # log recipe
  local log=$1 rec=$2
  $CC expect --log $log --select-cmd "$SEL" > $log.expect 2>&1; local e=$?
  local r=0
  if [ -n "$rec" ]; then $CC recipe --log $log --recipe $rec > $log.recipe 2>&1; r=$?; fi
  local masked=$(grep -c "producer=masked" $log)
  local fa5=$(grep -c "route=masked" $log)
  echo "CENSUS $(basename $log) expect_rc=$e recipe=$rec recipe_rc=$r masked_lines=$masked guard=[$(grep LLAMA_QSA_GUARD $log | tail -1 | cut -d' ' -f3-)] $(grep -E "^EXPECT" $log.expect) $(grep -c DIFF $log.expect) diffs"
  [ $e = 0 ] && [ $r = 0 ] && [ $masked = 0 ] || rc_all=1
}
for p in ${PAIRS:-f16:f16 q8_0:q8_0 q4_0:q4_0 bf16:bf16 q8_0:f16 f16:q8_0}; do
  k=$(pair_k $p); v=$(pair_v $p); t=$(pair_tag $p)
  for ub in ${UBS:-4096 512 16}; do
    log=$OUT/ppl_${t}_u$ub.log
    flock ~/gpu.lock env LLAMA_QSA_TRACE=1 GGML_CUDA_DISABLE_GRAPHS=1 timeout 3600 $BIN/llama-perplexity -m $M -f $WIKI -c 8192 -b 4096 -ub $ub --chunks 1 $COMMON -ctk $k -ctv $v > $log 2>&1
    echo "RUN $(basename $log) rc=$? $(grep -h 'Final estimate' $log | tail -1 | sed 's/.*Final/Final/')"
    case $ub in 4096) rec=;; 512) rec=Q1-B;; *) rec=;; esac
    check $log "$rec"
  done
  log=$OUT/bench_${t}_p4096_d4096.log
  flock ~/gpu.lock env LLAMA_QSA_TRACE=1 GGML_CUDA_DISABLE_GRAPHS=1 timeout 3600 $BIN/llama-bench -m $M -ngl 99 -fa 1 -lzm on-direct -ctk $k -ctv $v -b 4096 -ub 4096 -p 4096 -n 0 -d 4096 -r 1 -o json > $log 2>&1
  echo "RUN $(basename $log) rc=$?"
  check $log ""
done
echo CENSUS_DONE rc_all=$rc_all
