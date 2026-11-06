#!/bin/bash
# AT-W7PERF op level (PLAN-v3 §16.5): build w7_op, 3 interleaved processes on ROCm0, one CPU pass.
. "$(dirname "$0")/env.sh"
OUT=$W7/at_op; mkdir -p $OUT
B=$T/build
c++ -O2 -std=c++17 -I$T/ggml/include $T/scripts/qsa-validation/harness/w7/w7_op.cpp -o $OUT/w7_op -L$B/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$B/bin:/opt/rocm/core-10.1/lib:/opt/rocm/lib -Wl,-rpath-link,/opt/rocm/core-10.1/lib:/opt/rocm/lib || exit 1
PAIRS=${PAIRS:-q8_0:q8_0,q4_0:q4_0,bf16:bf16,q8_0:f16,f16:q8_0}
NQ=${NQ:-1,2,4,8,16,32,64,127,512,1024}; NK=${NK:-512,1024,2048}
for p in 1 2 3; do
  flock ~/gpu.lock env GGML_CUDA_DISABLE_GRAPHS=1 $OUT/w7_op p$p ROCm0 ${REPS:-7} $PAIRS $NQ $NK > $OUT/p$p.out 2> $OUT/p$p.err
  echo "AT_OP p$p rc=$? lines=$(wc -l < $OUT/p$p.out)"
done
# CPU SPARSE_ATTN per layer, f16 vs q8_0 (report)
flock ~/gpu.lock env GGML_CUDA_DISABLE_GRAPHS=1 $OUT/w7_op cpu CPU 3 f16:f16,q8_0:q8_0 2,16,64,127,512 512,2048 > $OUT/cpu.out 2> $OUT/cpu.err
echo "AT_OP cpu rc=$? lines=$(wc -l < $OUT/cpu.out)"
echo AT_OP_DONE
