#!/bin/bash
# build the W5 op-time harness against the worktree's HIP build (libggml from build/bin)
set -e
D=~/dev/wt_lab/w5/time; S=~/dev/wt_rewrite; mkdir -p $D; cp $S/scripts/qsa-validation/harness/w5/w5_time.cpp $D/
cd $D && g++ -O2 -std=c++17 -I $S/ggml/include w5_time.cpp -o w5_time -L $S/build/bin -lggml -lggml-base -Wl,-rpath,$S/build/bin
md5sum w5_time.cpp; echo BUILD $?
