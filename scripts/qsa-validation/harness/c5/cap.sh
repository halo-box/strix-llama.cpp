#!/bin/bash
# C5 capture campaign: rows of all 12 QSA layers at ub4096, pos0 8192/32768/65536/131072 (n_kv = pos0 + 4096)
cd ~/dev/wt_lab/c5
M=$(ls ~/models/qwen3.8-flash-next-UD-Q4_K_XL/*00001-of*.gguf)
W=~/data/wikitext-2-raw/wiki.test.raw
P=8192,32768,65536,131072
run() { # tag src ntok ctk pos special qkvpos
  flock ~/gpu.lock env LLAMA_FN_LAST_ATTN_TRIM=0 ./c5_capture $M $2 $3 4096 cap/$1 $4 $5 $6 $7 > logs/cap_$1.log 2>&1
  echo "EXIT $1 $?"
}
run wiki_f16 $W 135168 f16 $P 0 32768
run code_f16 corp/code.txt 135168 f16 $P 0 -1
run chat_f16 corp/chat.txt 135168 f16 $P 1 -1
run randc_f16 RANDOM 135168 f16 $P 0 -1
for d in 8192 32768 65536 131072; do run bench${d}_f16 BENCH $((d+4096)) f16 $d 0 -1; done
run wiki_q8_0 $W 135168 q8_0 32768,131072 0 32768
