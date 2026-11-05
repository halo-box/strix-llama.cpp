#!/bin/bash
# C18 fixture capture: QSA rows of all 12 layers at ub4096, pos0 4096 (n_kv 8192) and 131072 (n_kv 135168)
cd ~/dev/wt_lab/c18
M=$(ls ~/models/qwen3.8-flash-next-UD-Q4_K_XL/*00001-of*.gguf)
W=~/data/wikitext-2-raw/wiki.test.raw
C=../c5/c5_capture
run() { # tag src ntok pos qkvpos
  echo "START $1 $(date +%T)"
  flock ~/gpu.lock env LLAMA_FN_LAST_ATTN_TRIM=0 $C $M $2 $3 4096 cap/$1 f16 $4 0 $5 > logs/cap_$1.log 2>&1
  echo "EXIT $1 $? $(date +%T)"
}
run wikiS $W 8192 4096 4096
run randS RANDOM 8192 4096 4096
run wikiL $W 135168 4096,131072 131072
run randL RANDOM 135168 4096,131072 -1
