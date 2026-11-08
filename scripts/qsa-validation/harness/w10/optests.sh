#!/bin/bash
# W10 per-op test-backend-ops subsets (one op per run, never the full suite, under flock) and the HIP rotation unit test.
# usage: optests.sh <out dir> [run ...]   runs: cpu rocm rocm_nographs vdirect vpacked generic portable fixup plan rot vk vkrot
T=~/dev/wt_rewrite
OUT=${1:?out dir}; shift
mkdir -p $OUT
TBO=${BIN:-$T/build/bin}/test-backend-ops
TBO_VK=${BIN_VK:-$T/build-vk/bin}/test-backend-ops
one() { # tag env backend op extra
  local tag=$1 env=$2 b=$3 o=$4 extra=$5 bin=${6:-$TBO}
  local t0=$(date +%s)
  flock ~/gpu.lock env $env timeout 5400 $bin test -o $o -b $b $extra > $OUT/$tag.log 2>&1; local rc=$?
  echo "OPTEST $tag env=[$env] $b $o rc=$rc $(grep -E '[0-9]+/[0-9]+ tests passed' $OUT/$tag.log | tail -1) notsup=$(grep -c 'not supported' $OUT/$tag.log) fail=$(grep -c ' FAIL' $OUT/$tag.log) secs=$(( $(date +%s) - t0 ))"
}
runs=${*:-cpu rocm}
for r in $runs; do
  case $r in
    cpu)           one cpu_SPARSE_ATTN "" CPU SPARSE_ATTN --fail-unsupported ;;
    rocm)          one rocm_SPARSE_ATTN "" ROCm0 SPARSE_ATTN --fail-unsupported ;;
    rocm_nographs) one rocm_nographs_SPARSE_ATTN "GGML_CUDA_DISABLE_GRAPHS=1" ROCm0 SPARSE_ATTN --fail-unsupported ;;
    vdirect|vpacked|generic|portable|fixup)
                   one rocm_${r}_SPARSE_ATTN "GGML_CUDA_SA_FORCE=$r" ROCm0 SPARSE_ATTN --fail-unsupported ;;
    plan)          one cpu_SPARSE_ATTN_PLAN "" CPU SPARSE_ATTN_PLAN --fail-unsupported
                   one rocm_SPARSE_ATTN_PLAN "" ROCm0 SPARSE_ATTN_PLAN --fail-unsupported ;;
    rot)           for f in "" generic portable fixup; do
                     flock ~/gpu.lock env GGML_CUDA_SA_FORCE=$f ${BIN:-$T/build/bin}/test-sparse-attn-rot > $OUT/rot_${f:-default}.log 2>&1
                     echo "ROT force=[$f] rc=$? $(tail -1 $OUT/rot_${f:-default}.log) bitwise_lines=$(grep -c bitwise $OUT/rot_${f:-default}.log)"
                   done ;;
    vk)            one vk_SPARSE_ATTN "" Vulkan0 SPARSE_ATTN --fail-unsupported $TBO_VK
                   one vk_SPARSE_ATTN_PLAN "" Vulkan0 SPARSE_ATTN_PLAN --fail-unsupported $TBO_VK ;;
    vkrot)         flock ~/gpu.lock ${BIN_VK:-$T/build-vk/bin}/test-sparse-attn-vk rot > $OUT/vk_rot.log 2>&1
                   echo "VKROT rc=$? $(tail -1 $OUT/vk_rot.log)" ;;
    *) echo "unknown run $r" ;;
  esac
done
echo OPTESTS_DONE
