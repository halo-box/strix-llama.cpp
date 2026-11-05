#!/bin/bash
# C7 lab: one traced llama-bench run. Usage: c7run.sh TAG BIN_DIR LZM REPS [ENV=VAL ...] -- [extra llama-bench args]
# Must be called under flock ~/gpu.lock.
TAG=$1 B=$2 LZM=$3 R=$4; shift 4
ENVS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do ENVS+=("$1"); shift; done; [ "$1" = "--" ] && shift
M=$(ls ~/models/qwen3.8-flash-next-UD-Q4_K_XL/*00001-of*.gguf)
O=~/dev/wt_lab/c7/out; mkdir -p $O
export LD_LIBRARY_PATH=$B
env "${ENVS[@]}" C7SHIM_OUT=$O/$TAG.shim LD_PRELOAD=~/dev/wt_lab/c7/c7shim.so \
  $B/llama-bench -m $M -ngl 99 -fa 1 -lzm $LZM -ctk q8_0 -ctv q8_0 -b 4096 -ub 512 -p ${C7P:-512} -n 0 -d 0 -r $R -o json "$@" \
  > $O/$TAG.json 2> $O/$TAG.err &
P=$!
python3 ~/dev/wt_lab/c7/c7samp.py $P $O/$TAG.samp 0.01 &
S=$!
wait $P; rc=$?
wait $S
echo "$(date +%T) $TAG rc=$rc"
