#!/bin/bash
# per-op test-backend-ops subsets, one op per run, -j 1 (never the full suite), under flock
. "$(dirname "$0")/env.sh"
OUT=$W7/optests; mkdir -p $OUT
TBO=${BIN:-$T/build/bin}/test-backend-ops
run() { local b=$1 o=$2 extra=$3; flock ~/gpu.lock timeout 3000 $TBO test -o $o -b $b $extra > $OUT/${b}_$o.log 2>&1; local rc=$?; echo "OPTEST $b $o rc=$rc $(grep -E '[0-9]+/[0-9]+ tests passed' $OUT/${b}_$o.log | tail -1)"; }
for o in SPARSE_ATTN_PLAN SPARSE_ATTN; do run CPU $o --fail-unsupported; run ROCm0 $o --fail-unsupported; done
for o in QSA_PREFILL_MASKLESS QSA_DECODE_MASKLESS QSA_PREFILL QSA_DECODE; do run ROCm0 $o ""; done
if [ -x $T/build-vk/bin/test-backend-ops ]; then
  TBO=$T/build-vk/bin/test-backend-ops
  for o in SPARSE_ATTN_PLAN SPARSE_ATTN; do run Vulkan0 $o --fail-unsupported; done
fi
echo OPTESTS_DONE
