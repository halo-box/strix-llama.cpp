#!/bin/bash
# W4 final at the tree to commit: HIP / Vulkan / CPU-only builds, -Werror TU checks of the W4 TUs, test-backend-ops one op
# per run (-j 1, never the full suite), every step under flock ~/gpu.lock
cd ~/dev/wt_rewrite; L=~/dev/wt_rewrite_logs/W4; T=build/bin/test-backend-ops; TAG=${1:-final}
echo "HEAD $(git rev-parse --short HEAD) status=[$(git status --short | wc -l) changed]"
b() { local t=$1; shift; flock ~/gpu.lock "$@" > $L/${TAG}_$t.log 2>&1; echo "EXIT $t $? warnings=$(grep -c 'warning:' $L/${TAG}_$t.log)"; }
b hip_build cmake --build build -j 16
b vk_build  cmake --build build-vk -j 16
b cpu_build cmake --build build-cpu -j 16
flock ~/gpu.lock $L/tu_all.sh $TAG > $L/tu_$TAG.out 2>&1; grep -v '^K ' $L/tu_$TAG.out
t() { local tag=$1; shift; flock ~/gpu.lock env "$@" > $L/${TAG}_$tag.log 2>&1; local rc=$?
      echo "EXIT $tag rc=$rc $(grep -h 'tests passed' $L/${TAG}_$tag.log | tr -s ' ') fail=$(grep -v SA_REPORT $L/${TAG}_$tag.log | grep -c 'FAIL') notsup=$(grep -c 'not supported' $L/${TAG}_$tag.log) $(grep -h 'CENSUS COVERAGE: [A-Za-z0-9]*: [0-9]' $L/${TAG}_$tag.log | tr -s ' ')"; }
t attn_graphs    GGML_SA_TEST_REPORT=1 $T test -o SPARSE_ATTN -b ROCm0 --fail-unsupported
t attn_nographs  GGML_CUDA_DISABLE_GRAPHS=1 $T test -o SPARSE_ATTN -b ROCm0 --fail-unsupported
t g1_vdirect     GGML_CUDA_SA_FORCE=vdirect $T test -o SPARSE_ATTN -b ROCm0 -p group=1 --fail-unsupported
t g4_graphs      $T test -o SPARSE_ATTN -b ROCm0 -p group=4 --fail-unsupported
t cpu_attn       $T test -o SPARSE_ATTN -b CPU --fail-unsupported
t plan_rocm      $T test -o SPARSE_ATTN_PLAN -b ROCm0 --fail-unsupported
t plan_cpu       $T test -o SPARSE_ATTN_PLAN -b CPU --fail-unsupported
for OP in QSA_PREFILL_MASKLESS QSA_DECODE_MASKLESS QSA_PREFILL QSA_DECODE; do t $OP $T test -o $OP -b ROCm0; done
for OP in QSA_PREFILL_MASKLESS QSA_DECODE_MASKLESS QSA_PREFILL QSA_DECODE SPARSE_ATTN_PLAN; do t vk_$OP build-vk/bin/test-backend-ops test -o $OP -b Vulkan0; done
t vk_attn        build-vk/bin/test-backend-ops test -o SPARSE_ATTN -b Vulkan0
t vkbuild_cpu_attn build-vk/bin/test-backend-ops test -o SPARSE_ATTN -b CPU --fail-unsupported
t cpuonly_attn   build-cpu/bin/test-backend-ops test -o SPARSE_ATTN -b CPU --fail-unsupported
(cd build && flock ~/gpu.lock ctest -R test-sparse-attn-plan-oracle --output-on-failure > $L/${TAG}_ctest_oracle.log 2>&1; echo "EXIT ctest rc=$? $(grep -h 'tests passed' $L/${TAG}_ctest_oracle.log)")
echo DONE
