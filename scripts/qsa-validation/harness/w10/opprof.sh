#!/bin/bash
# W10 OP_PROF A/B (U6, indicative: one process per arm): q8_0 K/V, ub4096, llama-perplexity with GGML_CUDA_OP_PROF=${OPPROF:-1} and HIP
# graphs off. d0: -c 4096 (every graph is a 4096-token ubatch at n_kv 4096); d32k: -c 36864, the 9th graph of the chunk is
# pp4096 at depth 32768. Arms: the W9 head (graph-side rotation, BASE) and this build (in-op rotation).
. ~/dev/wt_rewrite/scripts/qsa-validation/harness/w7/env.sh
BASE=${BASE:-~/dev/wt_lab/w10/base_bin}
NEW=${NEW:-$T/build/bin}
OUT=${OUT:-~/dev/wt_lab/w10/opprof}; mkdir -p $OUT
for arm in ${ARMS:-w9 w10}; do
  if [ $arm = w9 ]; then B=$BASE; LP="LD_LIBRARY_PATH=$BASE"; else B=$NEW; LP=; fi
  for d in ${DEPTHS:-d0 d32k}; do
    if [ $d = d0 ]; then C=4096; CH=2; else C=36864; CH=1; fi
    flock ~/gpu.lock env $LP GGML_CUDA_OP_PROF=${OPPROF:-1} GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_OVL=0 LLAMA_FN_LAST_ATTN_TRIM=0 timeout 3600 \
      $B/llama-perplexity -m $M -f $WIKI -c $C -b 4096 -ub 4096 --chunks $CH $COMMON -ctk ${KV:-q8_0} -ctv ${KV:-q8_0} > $OUT/${arm}_$d.log 2>&1
    echo "RUN $arm $d rc=$? $(grep -h 'Final estimate' $OUT/${arm}_$d.log | sed 's/.*Final/Final/')"
  done
done
python3 $(dirname "$0")/opprof_an.py $OUT
