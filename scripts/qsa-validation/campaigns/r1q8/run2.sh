#!/bin/bash
# run2.sh TAG MODE(base|kld) CTK CTV UB [ENV=VAL ...]
# Direct gate variant: the KLD base is the clean same-type build (q8_0 ub4096, QSA+MoE pinned from ref8), not the f16 reference.
set -u
TAG=$1 MODE=$2 CTK=$3 CTV=$4 UB=$5; shift 5
D=~/dev/wt_lab/r1q8
M=$(ls ~/models/qwen3.8-flash-next-UD-Q4_K_XL/*00001-of*.gguf)
W=~/data/wikitext-2-raw/wiki.test.raw
B=$D/tree/build/bin
L=$D/logs2; mkdir -p $L
QENV="QPIN=replay QPIN_DIR=$D/ref8/pins QPIN_PIN=qsa,moe QPIN_STATS=$L/$TAG.stats"
if [ "$MODE" = base ]; then KARGS="--kl-divergence-base $D/q8base.kld"; else KARGS="--kl-divergence-base $D/q8base.kld --kl-divergence"; fi
t0=$(date +%s)
flock ~/gpu.lock env $QENV "$@" timeout 1800 $B/llama-perplexity -m $M -f $W -c 8192 -b 4096 -ub $UB --chunks 8 -ngl 99 -fa on \
   -lm mmap -lzm on-direct -ctk $CTK -ctv $CTV $KARGS > $L/$TAG.log 2>&1
rc=$?
echo "R2 $TAG rc=$rc $(( $(date +%s)-t0 ))s | $(grep -h "Mean    KLD" $L/$TAG.log) | $(grep -h R1Q8_SUMMARY $L/$TAG.log | tail -1)"
