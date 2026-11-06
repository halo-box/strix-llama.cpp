#!/bin/bash
# W3 final at the committed tree: HIP build, -Werror TU checks, Vulkan / CPU-only builds, test-backend-ops one op per
# run (-j 1), every step under flock ~/gpu.lock
cd ~/dev/wt_rewrite; L=~/dev/wt_rewrite_logs/W3
echo "HEAD $(git rev-parse --short HEAD) status=[$(git status --short | wc -l) changed]"
b() { local t=$1; shift; flock ~/gpu.lock "$@" > $L/$t.log 2>&1; echo "EXIT $t $? warnings=$(grep -c 'warning:' $L/$t.log)"; }
b final_hip_build cmake --build build -j 16
b final_vk_build  cmake --build build-vk -j 16
b final_cpu_build cmake --build build-cpu -j 16
flock ~/gpu.lock $L/tu_all.sh final > $L/tu_final.out 2>&1; grep -v '^K ' $L/tu_final.out
$L/tests_w3.sh final
t() { local tag=$1; shift; flock ~/gpu.lock env "$@" > $L/final_$tag.log 2>&1; echo "EXIT $tag rc=$? $(grep -h 'tests passed' $L/final_$tag.log | tr -s ' ') fail=$(grep -c 'FAIL' $L/final_$tag.log) notsup=$(grep -c 'not supported' $L/final_$tag.log)"; }
t vk_attn_g4 build-vk/bin/test-backend-ops test -o SPARSE_ATTN -b Vulkan0 -p group=4 --fail-unsupported
t vk_plan    build-vk/bin/test-backend-ops test -o SPARSE_ATTN_PLAN -b Vulkan0
for OP in QSA_PREFILL_MASKLESS QSA_DECODE_MASKLESS QSA_PREFILL QSA_DECODE; do t vk_$OP build-vk/bin/test-backend-ops test -o $OP -b Vulkan0; done
t cpuonly_attn build-cpu/bin/test-backend-ops test -o SPARSE_ATTN -b CPU --fail-unsupported
t vkbuild_cpu_attn build-vk/bin/test-backend-ops test -o SPARSE_ATTN -b CPU --fail-unsupported
echo DONE
