#!/bin/bash
# Build the C10-patched llama-bench against an existing tree's built libraries (that tree is only read).
# usage: build.sh <tag> <tree>   -> ~/dev/wt_lab/c10/bin_<tag>/{libllama-bench-impl.so,llama-bench}
set -e
tag=$1; T=$2; B=$T/build; O=~/dev/wt_lab/c10/bin_$tag; S=~/dev/wt_lab/c10/src
mkdir -p $O
DEF="-DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_CUDA -DGGML_USE_HIP -DGGML_USE_TRELLIS -DLLAMA_SHARED -DLLAMA_SUBPROCESS"
INC="-I$T/tools/llama-bench -I$T/common/. -I$T/vendor/nlohmann/.. -I$T/vendor/sheredom/.. -I$T/src/../include -I$T/ggml/src/../include"
FL="-O3 -DNDEBUG -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi"
LIBS="$B/bin/libllama-common.so.0.6.0 $B/bin/libllama.so.0.6.0 $B/bin/libggml.so.0.26.0 $B/bin/libggml-cpu.so.0.26.0 $B/bin/libggml-hip.so.0.26.0 $B/bin/libggml-base.so.0.26.0"
RP="-Wl,-rpath,$O:$B/bin:/opt/rocm/core-10.1/lib:/opt/rocm/lib: -Wl,-rpath-link,/opt/rocm/core-10.1/lib:/opt/rocm/lib"
c++ $DEF -Dllama_bench_impl_EXPORTS $INC $FL -fPIC -std=gnu++17 -c $S/llama-bench.cpp -o $O/llama-bench.o
c++ -fPIC -O3 -DNDEBUG -shared -Wl,-soname,libllama-bench-impl.so -o $O/libllama-bench-impl.so $O/llama-bench.o $RP $LIBS $B/common/libllama-common-base.a
c++ $DEF $INC $FL -std=gnu++17 -c $S/main.cpp -o $O/main.o
c++ -O3 -DNDEBUG $O/main.o -o $O/llama-bench $RP $O/libllama-bench-impl.so $B/bin/libllama-common.so.0.6.0 $B/common/libllama-common-base.a $LIBS
ldd $O/llama-bench | grep -E 'llama|ggml'
