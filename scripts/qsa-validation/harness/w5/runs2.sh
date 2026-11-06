#!/bin/bash
# W5 after runs1: the multi-arch HIP compile gate, the Vulkan and CPU-only builds, regression runs, the op-time harness
cd ~/dev/wt_rewrite; L=~/dev/wt_rewrite_logs/W5; X=$L/t.sh
while ! grep -q RUNS1_DONE $L/runs1.out; do sleep 20; done
b() { local t=$1; shift; flock ~/gpu.lock "$@" > $L/$t.log 2>&1; echo "EXIT $t $? warnings=$(grep -c "warning:" $L/$t.log) errors=$(grep -c "error:" $L/$t.log)"; }
b ma_cfg cmake -B build-multiarch -DGGML_HIP=ON "-DGPU_TARGETS=gfx1100;gfx1151;gfx90a;gfx942" -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang -DCMAKE_PREFIX_PATH=/opt/rocm -DLLAMA_CURL=OFF
b ma_build cmake --build build-multiarch -j 16
b vk_build cmake --build build-vk -j 16
b cpu_build cmake --build build-cpu -j 16
T=build/bin/test-backend-ops
$X r_plan_rocm $T test -o SPARSE_ATTN_PLAN -b ROCm0 --fail-unsupported
$X r_plan_cpu  $T test -o SPARSE_ATTN_PLAN -b CPU --fail-unsupported
for OP in QSA_PREFILL_MASKLESS QSA_DECODE_MASKLESS QSA_PREFILL QSA_DECODE; do $X r_$OP $T test -o $OP -b ROCm0; done
for OP in QSA_PREFILL_MASKLESS QSA_DECODE_MASKLESS QSA_PREFILL QSA_DECODE SPARSE_ATTN_PLAN SPARSE_ATTN; do $X r_vk_$OP build-vk/bin/test-backend-ops test -o $OP -b Vulkan0; done
$X r_vkbuild_cpu_attn build-vk/bin/test-backend-ops test -o SPARSE_ATTN -b CPU --fail-unsupported
$X r_cpuonly_attn build-cpu/bin/test-backend-ops test -o SPARSE_ATTN -b CPU --fail-unsupported
(cd build && flock ~/gpu.lock ctest -R test-sparse-attn-plan-oracle --output-on-failure > $L/r_ctest_oracle.log 2>&1; echo "EXIT ctest rc=$? $(grep -h "tests passed" $L/r_ctest_oracle.log)")
echo RUNS2_DONE
