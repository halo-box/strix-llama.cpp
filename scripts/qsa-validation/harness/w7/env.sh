# common paths of the W7 harnesses
W7=~/dev/wt_lab/w7
T=~/dev/wt_rewrite
L=$W7/logs
mkdir -p $L
M=$(ls ~/models/qwen3.8-flash-next-UD-Q4_K_XL/*00001-of*.gguf)
WIKI=~/data/wikitext-2-raw/wiki.test.raw
S=~/models/signal3.8
COMMON="-ngl 99 -fa on -lm mmap -lzm on-direct"
pair_k() { echo "${1%:*}"; }
pair_v() { echo "${1#*:}"; }
pair_tag() { echo "$1" | sed 's/:/_/'; }
