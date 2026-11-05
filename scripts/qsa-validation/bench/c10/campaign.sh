#!/bin/bash
# C10 campaign: base (13a279bfa) vs PR (94e086847) llama-bench, C10-patched (src/llama-bench.cpp, ../c10.diff),
# real text (wikitext-2 test + llama.cpp code, 16384-token segments interleaved) vs llama-bench random tokens.
# Per (depth, type): 7 processes in order B P B P B P B; within each slot text then random.
# One process covers ub 4096 and 512 (depth filled once at ub4096, restored for the others: "(cached)" in .err).
M=$(ls ~/models/qwen3.8-flash-next-UD-Q4_K_XL/*00001-of*.gguf)
C=~/dev/wt_lab/c10; L=$C/logs; mkdir -p $L
TXT=~/data/wikitext-2-raw/wiki.test.raw,$C/corp/code.txt
R=${R:-5}
run() { # tag bin type depth plist wl
  local tag=$1 B=$2 t=$3 d=$4 p=$5 wl=$6 env=""
  [ $wl = text ] && env="C10_TEXT=$TXT"
  local tmp=$(cat /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1)
  local t0=$(date +%s)
  flock ~/gpu.lock env $env $B/llama-bench -m $M -ngl 99 -fa 1 -lzm on-direct -ctk $t -ctv $t -b 4096 -ub 4096,512 \
      -p $p -n 0 -d $d -r $R --progress -o json > $L/$tag.json 2> $L/$tag.err
  local rc=$?
  echo "$(date +%T) done $tag rc=$rc secs=$(( $(date +%s) - t0 )) temp0=$tmp"
}
# prelude: patched-random == upstream binary (PR), d0 ub512/4096 q8_0, interleaved
for k in 1 2; do
  run eq_up_$k ~/dev/wt_pr219v2/build/bin q8_0 0 512,4096 rand
  run eq_c10_$k $C/bin_pr q8_0 0 512,4096 rand
done
for d in ${DEPTHS:-0 32768 131072}; do
  if [ $d = 0 ]; then P=512,2048,4096,8192; else P=512,4096; fi
  for t in f16 q8_0; do
    for k in 0 1 2 3 4 5 6; do
      if [ $((k % 2)) = 0 ]; then b=base; else b=pr; fi
      for wl in text rand; do
        run d${d}_${t}_${k}_${b}_${wl} $C/bin_$b $t $d $P $wl
      done
    done
  done
done
echo CAMPAIGN_DONE
