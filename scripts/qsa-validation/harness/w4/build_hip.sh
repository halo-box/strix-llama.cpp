#!/bin/bash
# W4 HIP build (rule flags), under flock ~/gpu.lock; usage: build_hip.sh <tag>
cd ~/dev/wt_rewrite || exit 1
L=~/dev/wt_rewrite_logs/W4
flock ~/gpu.lock cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang -DCMAKE_PREFIX_PATH=/opt/rocm -DLLAMA_CURL=OFF > $L/hip_cfg_$1.log 2>&1
echo "cfg_rc=$?"
start=$(date +%s)
flock ~/gpu.lock cmake --build build -j 16 > $L/hip_build_$1.log 2>&1
echo "build_rc=$? secs=$(( $(date +%s) - start ))"
grep -c "warning:" $L/hip_build_$1.log | xargs echo warnings
grep -n "error" $L/hip_build_$1.log | grep -v "UI: download" | head -30
