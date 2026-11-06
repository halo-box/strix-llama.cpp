#!/bin/bash
# W2 captures: C3 rows at pos0 32768 and 131072 (n_kv 36864, 135168; C3 pos list, so the hooked ubatches match C3), ub4096, with C3's own capture binary and arguments
# (EV/c3/scripts/cap.sh; PR219 build ~/dev/wt_pr219v2/build/bin, read-only), new output directory.
set -u
D=~/dev/wt_lab/w2; mkdir -p $D/cap $D/logs; cd $D
M=$(ls ~/models/qwen3.8-flash-next-UD-Q4_K_XL/*00001-of*.gguf)
W=~/data/wikitext-2-raw/wiki.test.raw
for cfg in "text f16 $W" "rand f16 RANDOM" "text q8_0 $W"; do
  set -- $cfg
  flock ~/gpu.lock env LLAMA_FN_LAST_ATTN_TRIM=0 ~/dev/wt_lab/c3/c3_capture $M $3 135168 4096 cap/$1_$2 $2 32768,131072 > logs/c3cap_$1_$2.log 2>&1
  echo "EXIT c3 $1 $2 $? $(date +%T)"
done
ls -la cap
