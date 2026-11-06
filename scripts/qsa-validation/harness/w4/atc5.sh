#!/bin/bash
# AT-C5 at op level (W4): w4_reserve per type x ub x n_ctx, GGML_SCHED_DEBUG_REALLOC=1 (an unexpected re-plan aborts);
# ub {64, 65, 66, 127, 128, 129}; n_ctx of llama-bench -p 4096 -n 32 -d {4096, 32768, 131072} (= d + 4096 + 32);
# every run under flock ~/gpu.lock
cd ~/dev/wt_lab/w4/reserve || exit 1; mkdir -p res
g++ -O2 -std=c++17 -I ~/dev/wt_rewrite/ggml/include w4_reserve.cpp -o w4_reserve -L ~/dev/wt_rewrite/build/bin -lggml -lggml-base -Wl,-rpath,$HOME/dev/wt_rewrite/build/bin || exit 1
for t in f16 q8_0 q4_0; do
  for d in 4096 32768 131072; do
    for ub in 64 65 66 127 128 129; do
      tag=${t}_d${d}_ub${ub}
      flock ~/gpu.lock env GGML_SCHED_DEBUG_REALLOC=1 ./w4_reserve $t $ub $((d + 4096 + 32)) 8 > res/$tag.out 2> res/$tag.err
      echo "EXIT $tag $? $(grep -h '^RESERVE\|^RUNTIME' res/$tag.out | sed 's/.*| //' | tr '\n' ' ')"
    done
  done
done
echo DONE
