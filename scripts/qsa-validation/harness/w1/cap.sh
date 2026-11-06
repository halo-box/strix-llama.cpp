#!/bin/bash
# W1 captures (PR219 build ~/dev/wt_pr219v2/build/bin, read-only, as C3 / C18 ran):
#  - C3's rows: c3/cap.sh's binary and arguments, new output directory
#  - R1-A's wiki d4k fixture: C18's c18_cap.sh "wikiS" (rows + .qkv at pos0 4096)
set -u
D=~/dev/wt_lab/w1; mkdir -p $D/cap $D/logs; cd $D
M=$(ls ~/models/qwen3.8-flash-next-UD-Q4_K_XL/*00001-of*.gguf)
W=~/data/wikitext-2-raw/wiki.test.raw
flock ~/gpu.lock env LLAMA_FN_LAST_ATTN_TRIM=0 ~/dev/wt_lab/c5/c5_capture $M $W 8192 4096 cap/wikiS f16 4096 0 4096 > logs/cap_wikiS.log 2>&1; echo "EXIT wikiS $? $(date +%T)"
for cfg in "text f16 $W" "rand f16 RANDOM" "text q8_0 $W"; do
  set -- $cfg
  flock ~/gpu.lock env LLAMA_FN_LAST_ATTN_TRIM=0 ~/dev/wt_lab/c3/c3_capture $M $3 135168 4096 cap/$1_$2 $2 32768,131072 > logs/c3cap_$1_$2.log 2>&1
  echo "EXIT c3 $1 $2 $? $(date +%T)"
done
ls -la cap
