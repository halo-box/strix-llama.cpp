#!/bin/bash
# W10 done-when: C12-style route census with HIP graphs off (counts), llama-perplexity c8192 ub {4096, 512} and llama-bench
# pp4096 @ d4096, checked by census_check.py against rows from the shipped policy (test-qsa-select-dump: gated for every
# type since W10), plus the gated / plain attend counts per run. Writes under ~/dev/wt_lab/w10/census (W7's logs untouched).
. ~/dev/wt_rewrite/scripts/qsa-validation/harness/w7/env.sh
BIN=${BIN:-$T/build/bin}
OUT=${OUT:-~/dev/wt_lab/w10/census}; mkdir -p $OUT
CC="python3 $T/scripts/qsa-validation/census_check.py"
SEL="$BIN/test-qsa-select-dump"
rc_all=0
check() { # log
  local log=$1
  $CC expect --log $log --select-cmd "$SEL" > $log.expect 2>&1; local e=$?
  local g=$(grep -E "^QSA_CENSUS sa_attend/G4/.*/gated/" $log | awk '{s+=$3} END {print s+0}')
  local p=$(grep -E "^QSA_CENSUS sa_attend/G4/.*/plain/" $log | awk '{s+=$3} END {print s+0}')
  local rot=$(grep -c "FWHT\|hadamard" $log)
  echo "CENSUS $(basename $log) expect_rc=$e G4_gated=$g G4_plain=$p masked_lines=$(grep -c producer=masked $log) $(grep -E '^EXPECT' $log.expect) $(grep -c DIFF $log.expect) diffs | $(grep 'gated share' $log.expect)"
  [ $e = 0 ] || rc_all=1
}
for p in ${PAIRS:-q8_0:q8_0 q4_0:q4_0 f16:f16 bf16:bf16 q8_0:f16 f16:q8_0}; do
  k=$(pair_k $p); v=$(pair_v $p); t=$(pair_tag $p)
  for ub in ${UBS:-4096 512}; do
    log=$OUT/ppl_${t}_u$ub.log
    flock ~/gpu.lock env LLAMA_QSA_TRACE=1 GGML_CUDA_DISABLE_GRAPHS=1 timeout 3600 $BIN/llama-perplexity -m $M -f $WIKI -c 8192 -b 4096 -ub $ub --chunks 1 $COMMON -ctk $k -ctv $v > $log 2>&1
    echo "RUN $(basename $log) rc=$? $(grep -h 'Final estimate' $log | tail -1 | sed 's/.*Final/Final/')"
    check $log
  done
  if [ -n "$BENCH" ]; then
    log=$OUT/bench_${t}_p4096_d4096.log
    flock ~/gpu.lock env LLAMA_QSA_TRACE=1 GGML_CUDA_DISABLE_GRAPHS=1 timeout 3600 $BIN/llama-bench -m $M -ngl 99 -fa 1 -lzm on-direct -ctk $k -ctv $v -b 4096 -ub 4096 -p 4096 -n 0 -d 4096 -r 1 -o json > $log 2>&1
    echo "RUN $(basename $log) rc=$?"
    check $log
  fi
done
echo CENSUS_DONE rc_all=$rc_all
