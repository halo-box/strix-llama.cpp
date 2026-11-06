#!/bin/bash
# The C10-patched llama-bench (real text, bench/c10/patch_c10.py) against a tree's built libraries: build_bench.sh <tag> <tree> [build dir]
# -> ~/dev/wt_lab/w7/bench/bin_<tag>/llama-bench. The tree's own tools/llama-bench source is patched into bench/src_<tag>.
set -e
tag=$1; T0=$2; BD=${3:-build}; B=$T0/$BD
W=~/dev/wt_lab/w7/bench; O=$W/bin_$tag; S=$W/src_$tag
mkdir -p $O $S
cp $T0/tools/llama-bench/main.cpp $S/main.cpp
python3 ~/dev/wt_rewrite/scripts/qsa-validation/bench/c10/patch_c10.py $T0/tools/llama-bench/llama-bench.cpp $S/llama-bench.cpp
DEF="-DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_CUDA -DGGML_USE_HIP -DGGML_USE_TRELLIS -DLLAMA_SHARED -DLLAMA_SUBPROCESS"
INC="-I$T0/tools/llama-bench -I$T0/common/. -I$T0/vendor/nlohmann/.. -I$T0/vendor/sheredom/.. -I$T0/src/../include -I$T0/ggml/src/../include"
FL="-O3 -DNDEBUG -Wno-unused-function -Wno-array-bounds"
LIBS="$B/bin/libllama-common.so.0.6.0 $B/bin/libllama.so.0.6.0 $B/bin/libggml.so.0.26.0 $B/bin/libggml-cpu.so.0.26.0 $B/bin/libggml-hip.so.0.26.0 $B/bin/libggml-base.so.0.26.0"
RP="-Wl,-rpath,$O:$B/bin:/opt/rocm/core-10.1/lib:/opt/rocm/lib: -Wl,-rpath-link,/opt/rocm/core-10.1/lib:/opt/rocm/lib"
c++ $DEF -Dllama_bench_impl_EXPORTS $INC $FL -fPIC -std=gnu++17 -c $S/llama-bench.cpp -o $O/llama-bench.o
c++ -fPIC -O3 -DNDEBUG -shared -Wl,-soname,libllama-bench-impl.so -o $O/libllama-bench-impl.so $O/llama-bench.o $RP $LIBS $B/common/libllama-common-base.a
c++ $DEF $INC $FL -std=gnu++17 -c $S/main.cpp -o $O/main.o
c++ -O3 -DNDEBUG $O/main.o -o $O/llama-bench $RP $O/libllama-bench-impl.so $B/bin/libllama-common.so.0.6.0 $B/common/libllama-common-base.a $LIBS
echo built $O/llama-bench
