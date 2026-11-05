#!/bin/bash
# r1ub census.sh TAG CTK CTV UB CTX : unpinned, 1 chunk, C2COUNT launch census (wt_lab/build: 94e086847 + C1/C2 env-gated tracers)
set -u
TAG=$1 CTK=$2 CTV=$3 UB=$4 CTX=$5
M=$(ls ~/models/qwen3.8-flash-next-UD-Q4_K_XL/*00001-of*.gguf)
W=~/data/wikitext-2-raw/wiki.test.raw
L=~/dev/wt_lab/r1ub/census; mkdir -p $L
t0=$(date +%s)
flock ~/gpu.lock env GGML_CUDA_DISABLE_GRAPHS=1 GGML_QSA_COUNT=1 GGML_QSA_TRACE=1 timeout 1800 ~/dev/wt_lab/build/bin/llama-perplexity -m $M -f $W -c $CTX -b 4096 -ub $UB --chunks 1 -ngl 99 -fa on \
   -lm mmap -lzm on-direct -ctk $CTK -ctv $CTV > $L/$TAG.log 2>&1
echo "C $TAG rc=$? $(( $(date +%s)-t0 ))s"
