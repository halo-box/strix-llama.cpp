#!/bin/bash
# C7 lab: run any command with the shim + sampler. Usage: c7cmd.sh TAG cmd args...   (call under flock ~/gpu.lock)
TAG=$1; shift
O=~/dev/wt_lab/c7/out; mkdir -p $O
C7SHIM_OUT=$O/$TAG.shim LD_PRELOAD=~/dev/wt_lab/c7/c7shim.so "$@" > $O/$TAG.log 2>&1 &
P=$!
python3 ~/dev/wt_lab/c7/c7samp.py $P $O/$TAG.samp 0.01 &
S=$!
wait $P; rc=$?; wait $S
echo "$(date +%T) $TAG rc=$rc"
