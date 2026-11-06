#!/bin/bash
# HIP build of wt_rewrite with the rule flags; $1 = log tag
cd ~/dev/wt_rewrite || exit 1
T=${1:-b}
L=~/dev/wt_rewrite_logs/W3
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang -DCMAKE_PREFIX_PATH=/opt/rocm -DLLAMA_CURL=OFF > $L/cfg_$T.log 2>&1
echo "cfg_rc=$?" >> $L/cfg_$T.log
s=$(date +%s)
cmake --build build -j 16 > $L/build_$T.log 2>&1
rc=$?
echo "build_rc=$rc seconds=$(( $(date +%s) - s ))" >> $L/build_$T.log
echo "build_rc=$rc seconds=$(( $(date +%s) - s ))"
