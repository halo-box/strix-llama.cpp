#!/bin/bash
# run.sh TAG MODE(record|replay) CTK CTV UB [ENV=VAL ...]   (r1q8 lab; c8192 x 8, QPIN_PIN=qsa,moe on replay)
set -u
TAG=$1 MODE=$2 CTK=$3 CTV=$4 UB=$5; shift 5
D=~/dev/wt_lab/r1q8
M=$(ls ~/models/qwen3.8-flash-next-UD-Q4_K_XL/*00001-of*.gguf)
W=~/data/wikitext-2-raw/wiki.test.raw
B=$D/tree/build/bin
REF=$D/ref8; L=$D/logs; mkdir -p $L $REF/pins
if [ "$MODE" = record ]; then KARGS="--kl-divergence-base $REF/base.kld"; QENV="QPIN=record QPIN_DIR=$REF/pins"
else KARGS="--kl-divergence-base $REF/base.kld --kl-divergence"; QENV="QPIN=replay QPIN_DIR=$REF/pins QPIN_PIN=qsa,moe QPIN_STATS=$L/$TAG.stats"; fi
t0=$(date +%s)
flock ~/gpu.lock env $QENV "$@" timeout 1800 $B/llama-perplexity -m $M -f $W -c 8192 -b 4096 -ub $UB --chunks 8 -ngl 99 -fa on \
   -lm mmap -lzm on-direct -ctk $CTK -ctv $CTV $KARGS > $L/$TAG.log 2>&1
rc=$?
echo "R $TAG rc=$rc $(( $(date +%s)-t0 ))s | $(grep -h "Mean    KLD" $L/$TAG.log) | $(grep -h "Same top p" $L/$TAG.log) | $(grep -h QPIN_SUMMARY $L/$TAG.log) | $(grep -h R1Q8_SUMMARY $L/$TAG.log)"
