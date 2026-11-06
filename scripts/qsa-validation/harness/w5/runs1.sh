#!/bin/bash
# W5 test runs (each one op, -j 1, under flock via t.sh)
T=build/bin/test-backend-ops; X=~/dev/wt_rewrite_logs/W5/t.sh
$X t2_portable GGML_SA_TEST_REPORT=1 GGML_CUDA_SA_FORCE=portable $T test -o SPARSE_ATTN -b ROCm0 --fail-unsupported
$X t3_generic  GGML_SA_TEST_REPORT=1 GGML_CUDA_SA_FORCE=generic  $T test -o SPARSE_ATTN -b ROCm0 --fail-unsupported
$X t4_nographs GGML_CUDA_DISABLE_GRAPHS=1 $T test -o SPARSE_ATTN -b ROCm0 --fail-unsupported
$X t5_vdirect  GGML_CUDA_SA_FORCE=vdirect $T test -o SPARSE_ATTN -b ROCm0 --fail-unsupported
$X t6_vpacked  GGML_CUDA_SA_FORCE=vpacked $T test -o SPARSE_ATTN -b ROCm0 --fail-unsupported
$X t7_cpu      $T test -o SPARSE_ATTN -b CPU --fail-unsupported
echo RUNS1_DONE
