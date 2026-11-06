#!/bin/bash
# W5: reconfigure (new top-level .cu files are globbed at configure time) and build HIP with the rule flags
cd ~/dev/wt_rewrite; L=~/dev/wt_rewrite_logs/W5; T=${1:-b1}
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang -DCMAKE_PREFIX_PATH=/opt/rocm -DLLAMA_CURL=OFF > $L/hip_cfg_$T.log 2>&1; echo "EXIT cfg $?"
cmake --build build -j 16 > $L/hip_build_$T.log 2>&1; echo "EXIT build $? warnings=$(grep -c "warning:" $L/hip_build_$T.log) errors=$(grep -c "error:" $L/hip_build_$T.log)"
