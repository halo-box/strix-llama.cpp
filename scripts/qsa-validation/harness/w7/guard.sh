#!/bin/bash
# W7 done-when: the guard fires on a fake GPU device and on a real HIP device with the forced-reject hook, at init and at decode; the
# non-vacuity count on passing configs; the all-visible rows on every backend.
. "$(dirname "$0")/env.sh"
BIN=${BIN:-$T/build/bin}
OUT=$W7/guard; mkdir -p $OUT
run() { local tag=$1; shift; flock ~/gpu.lock timeout 900 "$@" > $OUT/$tag.log 2>&1; echo "GUARD $tag rc=$? | $(tail -1 $OUT/$tag.log | cut -c1-120) | $(grep -m1 -E 'QSA layer' $OUT/$tag.log | cut -c1-260)"; }
run pass_q8 $BIN/test-qsa-guard $M pass q8_0
run pass_f16 $BIN/test-qsa-guard $M pass f16
run pass_q4 $BIN/test-qsa-guard $M pass q4_0
run pass_bf16 $BIN/test-qsa-guard $M pass bf16
run fake $BIN/test-qsa-guard $M fake
run reject $BIN/test-qsa-guard $M reject
run reject_decode $BIN/test-qsa-guard $M reject-decode
run allvis $BIN/test-qsa-allvis-rows "$@"
echo GUARD_DONE
