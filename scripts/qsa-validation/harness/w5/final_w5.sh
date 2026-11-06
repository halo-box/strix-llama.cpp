#!/bin/bash
# W5 final at the commits under test: HIP clean rebuild (hash against the binary of the earlier runs), the multi-arch compile
# gate from an empty build directory, Vulkan / CPU-only builds, -Werror TU checks, test-backend-ops one op per run (-j 1,
# never the full suite). Every step under flock ~/gpu.lock
cd ~/dev/wt_rewrite; L=~/dev/wt_rewrite_logs/W5; T=build/bin/test-backend-ops; TAG=${1:-final}
HIPFLAGS="-DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang -DCMAKE_PREFIX_PATH=/opt/rocm -DLLAMA_CURL=OFF"
echo "HEAD $(git rev-parse --short HEAD) status=[$(git status --short | wc -l) changed] date=$(date -Is)"
b() { local t=$1; shift; flock ~/gpu.lock "$@" > $L/${TAG}_$t.log 2>&1; echo "EXIT $t $? warnings=$(grep -c 'warning:' $L/${TAG}_$t.log) errors=$(grep -c 'error:' $L/${TAG}_$t.log)"; }
sha256sum $(readlink -f build/bin/libggml-hip.so) $T | sed 's/^/PRE /'
b hip_cfg   cmake -B build $HIPFLAGS -DGPU_TARGETS=gfx1151
b hip_build cmake --build build -j 16 --clean-first
sha256sum $(readlink -f build/bin/libggml-hip.so) $T | sed 's/^/POST /'
rm -rf build-multiarch
b ma_cfg    cmake -B build-multiarch $HIPFLAGS "-DGPU_TARGETS=gfx1100;gfx1151;gfx90a;gfx942"
b ma_build  cmake --build build-multiarch -j 16
b vk_build  cmake --build build-vk -j 16
b cpu_build cmake --build build-cpu -j 16
flock ~/gpu.lock $L/tu_all.sh $TAG > $L/tu_$TAG.out 2>&1; grep -v '^K ' $L/tu_$TAG.out
t() { local tag=$1; shift; flock ~/gpu.lock env "$@" > $L/${TAG}_$tag.log 2>&1; local rc=$?
      echo "EXIT $tag rc=$rc $(grep -h 'tests passed' $L/${TAG}_$tag.log | tr -s ' ') fail=$(grep -v 'SA_REPORT\|SA_GQAREP' $L/${TAG}_$tag.log | grep -c 'FAIL') notsup=$(grep -c 'not supported' $L/${TAG}_$tag.log) $(grep -h 'CENSUS COVERAGE: [A-Za-z0-9]*: [0-9]' $L/${TAG}_$tag.log | tr -s ' ')"; }
t attn_graphs   GGML_SA_TEST_REPORT=1 $T test -o SPARSE_ATTN -b ROCm0 --fail-unsupported
t portable      GGML_SA_TEST_REPORT=1 GGML_CUDA_SA_FORCE=portable $T test -o SPARSE_ATTN -b ROCm0 --fail-unsupported
t generic       GGML_SA_TEST_REPORT=1 GGML_CUDA_SA_FORCE=generic  $T test -o SPARSE_ATTN -b ROCm0 --fail-unsupported
t attn_nographs GGML_CUDA_DISABLE_GRAPHS=1 $T test -o SPARSE_ATTN -b ROCm0 --fail-unsupported
t vdirect       GGML_CUDA_SA_FORCE=vdirect $T test -o SPARSE_ATTN -b ROCm0 --fail-unsupported
t vpacked       GGML_CUDA_SA_FORCE=vpacked $T test -o SPARSE_ATTN -b ROCm0 --fail-unsupported
t cpu_attn      $T test -o SPARSE_ATTN -b CPU --fail-unsupported
t plan_rocm     $T test -o SPARSE_ATTN_PLAN -b ROCm0 --fail-unsupported
t plan_cpu      $T test -o SPARSE_ATTN_PLAN -b CPU --fail-unsupported
for OP in QSA_PREFILL_MASKLESS QSA_DECODE_MASKLESS QSA_PREFILL QSA_DECODE; do t $OP $T test -o $OP -b ROCm0; done
for OP in QSA_PREFILL_MASKLESS QSA_DECODE_MASKLESS QSA_PREFILL QSA_DECODE SPARSE_ATTN_PLAN; do t vk_$OP build-vk/bin/test-backend-ops test -o $OP -b Vulkan0; done
t vk_attn          build-vk/bin/test-backend-ops test -o SPARSE_ATTN -b Vulkan0
t vkbuild_cpu_attn build-vk/bin/test-backend-ops test -o SPARSE_ATTN -b CPU --fail-unsupported
t cpuonly_attn     build-cpu/bin/test-backend-ops test -o SPARSE_ATTN -b CPU --fail-unsupported
(cd build && flock ~/gpu.lock ctest -R test-sparse-attn-plan-oracle --output-on-failure > $L/${TAG}_ctest_oracle.log 2>&1; echo "EXIT ctest rc=$? $(grep -h 'tests passed' $L/${TAG}_ctest_oracle.log)")
echo FINAL_DONE
