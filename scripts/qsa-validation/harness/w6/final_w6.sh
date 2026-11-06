#!/bin/bash
# W6: every run on the committed-to-be tree: Vulkan op runs (runs1.sh), validation and standalone (runs2.sh), HIP and CPU regressions of the test file
cd ~/dev/wt_rewrite
L=~/dev/wt_rewrite_logs/W6
rm -f $L/runs1.out $L/runs2.out $L/final.out
scripts/qsa-validation/harness/w6/runs1.sh
scripts/qsa-validation/harness/w6/runs2.sh
flock ~/gpu.lock bash -c 'cmake --build build -j 16 > ~/dev/wt_rewrite_logs/W6/hip_build.log 2>&1'; echo "hip build rc=$?" >> $L/final.out
flock ~/gpu.lock bash -c 'cmake --build build-vk -j 16 > ~/dev/wt_rewrite_logs/W6/vk_build_all.log 2>&1'; echo "vk build all rc=$?" >> $L/final.out
for op in SPARSE_ATTN_PLAN SPARSE_ATTN QSA_PREFILL_MASKLESS QSA_DECODE_MASKLESS QSA_PREFILL QSA_DECODE; do
  flock ~/gpu.lock timeout 2400 build/bin/test-backend-ops test -o $op -b ROCm0 --fail-unsupported -j 1 > $L/hip_$op.log 2>&1; echo "hip $op rc=$? $(grep 'tests passed' $L/hip_$op.log) notsup=$(grep -c 'not supported' $L/hip_$op.log)" >> $L/final.out
done
for op in QSA_PREFILL_MASKLESS QSA_DECODE_MASKLESS QSA_PREFILL QSA_DECODE; do
  flock ~/gpu.lock timeout 1200 build-vk/bin/test-backend-ops test -o $op -b Vulkan0 -j 1 > $L/vk_$op.log 2>&1; echo "vk $op rc=$? $(grep 'tests passed' $L/vk_$op.log) notsup=$(grep -c 'not supported' $L/vk_$op.log)" >> $L/final.out
done
for op in SPARSE_ATTN_PLAN SPARSE_ATTN; do
  flock ~/gpu.lock timeout 2400 build-vk/bin/test-backend-ops test -o $op -b CPU --fail-unsupported -j 1 > $L/cpu_$op.log 2>&1; echo "cpu $op rc=$? $(grep 'tests passed' $L/cpu_$op.log)" >> $L/final.out
done
flock ~/gpu.lock bash -c 'cmake -B build-cpu -S . > /dev/null && cmake --build build-cpu -j 16 --target test-backend-ops test-sparse-attn-vk > ~/dev/wt_rewrite_logs/W6/cpu_build.log 2>&1'; echo "cpu-only build rc=$?" >> $L/final.out
build-cpu/bin/test-sparse-attn-vk > $L/cpu_vkt.log 2>&1; echo "cpu-only test-sparse-attn-vk rc=$? $(tail -1 $L/cpu_vkt.log)" >> $L/final.out
flock ~/gpu.lock bash -c 'cd build-vk && ctest -R "test-sparse-attn" --output-on-failure > ~/dev/wt_rewrite_logs/W6/ctest_vk.log 2>&1'; echo "ctest vk rc=$? $(tail -3 $L/ctest_vk.log | tr '\n' ' ')" >> $L/final.out
echo FINAL_DONE >> $L/final.out
