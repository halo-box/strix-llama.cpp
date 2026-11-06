#!/bin/bash
# W3 tests: test-backend-ops one op per run, -j 1, every run under flock ~/gpu.lock; the full suite is never run
cd ~/dev/wt_rewrite; L=~/dev/wt_rewrite_logs/W3; T=build/bin/test-backend-ops; TAG=${1:-f}
t() { local tag=$1; shift; flock ~/gpu.lock env "$@" > $L/${TAG}_$tag.log 2>&1; local rc=$?
      echo "EXIT $tag rc=$rc $(grep -h 'tests passed' $L/${TAG}_$tag.log | tr -s ' ') fail=$(grep -v SA_REPORT $L/${TAG}_$tag.log | grep -c 'FAIL') notsup=$(grep -c 'not supported' $L/${TAG}_$tag.log) $(grep -h 'CENSUS COVERAGE: [A-Za-z0-9]*: [0-9]' $L/${TAG}_$tag.log | tr -s ' ')"; }
t g4_graphs      GGML_SA_TEST_REPORT=1 $T test -o SPARSE_ATTN -b ROCm0 -p group=4 --fail-unsupported
t g4_nographs    GGML_CUDA_DISABLE_GRAPHS=1 $T test -o SPARSE_ATTN -b ROCm0 -p group=4 --fail-unsupported
t g4_vdirect     GGML_SA_TEST_REPORT=1 GGML_CUDA_SA_FORCE=vdirect $T test -o SPARSE_ATTN -b ROCm0 -p group=4 --fail-unsupported
t g4_vpacked     GGML_SA_TEST_REPORT=1 GGML_CUDA_SA_FORCE=vpacked $T test -o SPARSE_ATTN -b ROCm0 -p group=4 --fail-unsupported
t attn_all       $T test -o SPARSE_ATTN -b ROCm0
t cpu_attn       $T test -o SPARSE_ATTN -b CPU --fail-unsupported
t plan_rocm      $T test -o SPARSE_ATTN_PLAN -b ROCm0 --fail-unsupported
t plan_cpu       $T test -o SPARSE_ATTN_PLAN -b CPU --fail-unsupported
for OP in QSA_PREFILL_MASKLESS QSA_DECODE_MASKLESS QSA_PREFILL QSA_DECODE; do t $OP $T test -o $OP -b ROCm0; done
(cd build && flock ~/gpu.lock ctest -R test-sparse-attn-plan-oracle --output-on-failure > $L/${TAG}_ctest_oracle.log 2>&1; echo "EXIT ctest rc=$? $(grep -h 'tests passed' $L/${TAG}_ctest_oracle.log)")
echo DONE
