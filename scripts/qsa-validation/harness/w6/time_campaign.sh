#!/bin/bash
# W6 AT-B2 / AT-B3: 3 interleaved processes per arm. C11 shape rows captured here (c11text, c11rand, ub 2048 at pos0 32768: n_kv 34816), d4096 and d131072 rows from C18.
S=~/dev/wt_rewrite; D=~/dev/wt_lab/w6/time; mkdir -p $D/res $D/cap; cd $D
cp $S/scripts/qsa-validation/harness/w6/w6_time.cpp .
g++ -O2 -std=c++17 -I $S/ggml/include w6_time.cpp -o w6_time -L $S/build-vk/bin -lggml -lggml-base -Wl,-rpath,$S/build-vk/bin || exit 1
M=$(ls ~/models/qwen3.8-flash-next-UD-Q4_K_XL/*00001-of*.gguf)
W=~/data/wikitext-2-raw/wiki.test.raw
if [ ! -f cap/c11text_p32768.rows ]; then
  for cfg in "c11text $W" "c11rand RANDOM"; do
    set -- $cfg
    flock ~/gpu.lock ~/dev/wt_lab/c3/c3_capture $M $2 135168 2048 cap/$1 f16 32768 > cap/$1.log 2>&1
    echo "EXIT cap $1 $?"
    ls -la cap | tail -4
  done
fi
C18=~/dev/wt_lab/c18/cap
declare -A R=( [c11text]=$D/cap/c11text_p32768.rows [c11rand]=$D/cap/c11rand_p32768.rows [d4ktext]=$C18/wikiS_p4096.rows [d4krand]=$C18/randS_p4096.rows [d131ktext]=$C18/wikiL_p131072.rows [d131krand]=$C18/randL_p131072.rows )
export GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1
for p in 1 2 3; do
  for r in c11text c11rand d4ktext d4krand d131ktext d131krand; do
    for F in default vdirect vstaged vpack; do
      if [ $F = default ]; then unset GGML_VK_SA_FORCE; A=sel,f16,q8_0,q4_0,bf16,q8_0:f16,f16:q8_0; else export GGML_VK_SA_FORCE=$F; A=q8_0,q4_0,bf16,f16; fi
      case $r in c11*) LY=all;; *) LY=0,5,11;; esac
      flock ~/gpu.lock timeout 1800 ./w6_time ${R[$r]} 5 2 $A $LY > res/time_${r}_${F}_p$p.out 2> res/time_${r}_${F}_p$p.err
      echo "EXIT $r $F $p $?"
    done
  done
done
echo TIME_DONE
