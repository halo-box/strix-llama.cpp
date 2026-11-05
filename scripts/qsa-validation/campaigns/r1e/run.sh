#!/bin/bash
# r1e run.sh (copy of r1ub/run.sh, logs in r1e/logs) TAG REFDIR MODE(record|replay) CTK CTV UB CTX CHUNKS BIN(hip|vk) [ENV=VAL ...]
#  record: QPIN record + write REFDIR/base.kld ; replay: QPIN replay against REFDIR, QPIN_PIN=qsa,moe, KLD vs REFDIR/base.kld
#  -b 4096 always (-ub UB); GGML_QSA_TRACE=1 (C1TRACE unique shapes, HIP only)
set -u
TAG=$1 REF=$2 MODE=$3 CTK=$4 CTV=$5 UB=$6 CTX=$7 CH=$8 BK=$9; shift 9
M=$(ls ~/models/qwen3.8-flash-next-UD-Q4_K_XL/*00001-of*.gguf)
W=~/data/wikitext-2-raw/wiki.test.raw
case $BK in hip) BIN=~/dev/wt_lab/r1av/tree/build/bin;; vk) BIN=~/dev/wt_lab/r1ub/tree/build-vk/bin;; esac
L=~/dev/wt_lab/r1e/logs; mkdir -p $L $REF
KLD=$REF/base.kld
case $MODE in
  record) KARGS="--kl-divergence-base $KLD"; QENV="QPIN=record QPIN_DIR=$REF/pins"; mkdir -p $REF/pins;;
  observe) KARGS="--kl-divergence-base $KLD --kl-divergence"; QENV="QPIN=replay QPIN_DIR=$REF/pins QPIN_STATS=$L/$TAG.stats";;
  replay) KARGS="--kl-divergence-base $KLD --kl-divergence"; QENV="QPIN=replay QPIN_DIR=$REF/pins QPIN_STATS=$L/$TAG.stats QPIN_PIN=qsa,moe";;
esac
t0=$(date +%s)
flock ~/gpu.lock env $QENV GGML_QSA_TRACE=1 "$@" timeout 5400 $BIN/llama-perplexity -m $M -f $W -c $CTX -b 4096 -ub $UB --chunks $CH -ngl 99 -fa on \
   -lm mmap -lzm on-direct -ctk $CTK -ctv $CTV $KARGS > $L/$TAG.log 2>&1
rc=$?
echo "R $TAG rc=$rc $(( $(date +%s)-t0 ))s bk=$BK ub=$UB c=$CTX env=[$*] | $(grep -h "Mean    KLD" $L/$TAG.log) | $(grep -h "Same top p" $L/$TAG.log) | $(grep -h QPIN_SUMMARY $L/$TAG.log | tail -1)"
