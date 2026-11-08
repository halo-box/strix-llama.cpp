#!/bin/bash
# W13 correctness tests: per-op test-backend-ops subsets (-j 1 by construction: one process, one op), ctest QSA subset
cd ~/dev/wt_rewrite
L=~/dev/wt_rewrite_logs/W13/tests; mkdir -p $L
: > $L/summary.txt
op() {  # op <tag> <bin dir> <backend> <op> [extra env]
  local tag=$1 bin=$2 be=$3 o=$4; shift 4
  flock ~/gpu.lock env "$@" timeout 7200 $bin/test-backend-ops -b $be -o $o > $L/${tag}_$o.log 2>&1
  local rc=$?
  echo "$tag $o rc=$rc $(grep -E '^ *[0-9]+/[0-9]+ tests passed|Backend .*: (OK|FAIL)|tests passed' $L/${tag}_$o.log | tr -s ' ' | tr '\n' ' ') fail=$(grep -c ' FAIL' $L/${tag}_$o.log) notsup=$(grep -c 'not supported' $L/${tag}_$o.log)" >> $L/summary.txt
}
for o in SPARSE_ATTN_PLAN SPARSE_ATTN FLASH_ATTN_EXT QSA_EXPAND QSA_SCORE QSA_STRIP; do op hip build/bin ROCm0 $o GGML_X=1; done
for o in SPARSE_ATTN_PLAN SPARSE_ATTN FLASH_ATTN_EXT QSA_EXPAND QSA_SCORE QSA_STRIP; do op vk build-vk/bin Vulkan0 $o GGML_X=1; done
# the deleted test names no longer exist
for o in QSA_PREFILL QSA_DECODE QSA_PREFILL_MASKLESS QSA_DECODE_MASKLESS; do
  n=$(build/bin/test-backend-ops -b CPU -o $o 2>&1 | grep -c "$o(")
  echo "deleted $o cases_listed=$n" >> $L/summary.txt
done
(cd build && flock ~/gpu.lock ctest -R "test-qsa|test-sparse-attn" -j1 --output-on-failure > $L/ctest_hip.log 2>&1; echo "ctest hip rc=$? $(grep -E 'tests passed|tests failed' $L/ctest_hip.log)" >> $L/summary.txt)
(cd build-vk && flock ~/gpu.lock ctest -R "test-qsa|test-sparse-attn" -j1 --output-on-failure > $L/ctest_vk.log 2>&1; echo "ctest vk rc=$? $(grep -E 'tests passed|tests failed' $L/ctest_vk.log)" >> $L/summary.txt)
echo TESTS_DONE >> $L/summary.txt
