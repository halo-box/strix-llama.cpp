#!/bin/bash
# One pinned perplexity arm of the W7 quality gates (w12/run.sh with the arm's binary chosen by ARM and extra CLI in QARGS):
#   qrun.sh TAG REF MODE ARM CTK CTV UB CTX CHUNKS [ENV=VAL ...]
#   ARM   new (W7 lab tree: the branch + QPIN patches) | w0 (W0-lab12, the PR219 arm)
#   MODE  record | replay | observe | pinbase | pinkld   as w12/run.sh; QB overrides -b (default 4096); QTIMEOUT the timeout
set -u
TAG=$1 REF=$2 MODE=$3 ARM=$4 CTK=$5 CTV=$6 UB=$7 CTX=$8 CH=$9; shift 9
M=$(ls ~/models/qwen3.8-flash-next-UD-Q4_K_XL/*00001-of*.gguf)
W=~/data/wikitext-2-raw/wiki.test.raw
case $ARM in
  new) BIN=~/dev/wt_lab/w7/lab/build/bin;;
  w0)  BIN=~/dev/wt_lab/w0lab12/build/bin;;
  *) echo "bad ARM $ARM"; exit 2;;
esac
L=${QLOGS:-~/dev/wt_lab/w7/q/logs}; mkdir -p $L $REF
B=${QB:-4096}
KLD=$REF/base.kld
case $MODE in
  record)  KARGS="--kl-divergence-base $KLD"; QENV="QPIN=record QPIN_DIR=$REF/pins"; mkdir -p $REF/pins;;
  observe) KARGS="--kl-divergence-base $KLD --kl-divergence"; QENV="QPIN=replay QPIN_DIR=$REF/pins QPIN_STATS=$L/$TAG.stats";;
  replay)  KARGS="--kl-divergence-base $KLD --kl-divergence"; QENV="QPIN=replay QPIN_DIR=$REF/pins QPIN_STATS=$L/$TAG.stats QPIN_PIN=qsa,moe";;
  pinbase) KARGS="--kl-divergence-base ${QKBASE:?}"; QENV="QPIN=replay QPIN_DIR=$REF/pins QPIN_STATS=$L/$TAG.stats QPIN_PIN=qsa,moe";;
  pinkld)  KARGS="--kl-divergence-base ${QKBASE:?} --kl-divergence"; QENV="QPIN=replay QPIN_DIR=$REF/pins QPIN_STATS=$L/$TAG.stats QPIN_PIN=qsa,moe";;
  base)    KARGS="--kl-divergence-base $KLD"; QENV="";;                                # plain, no QPIN
  kld)     KARGS="--kl-divergence-base $KLD --kl-divergence"; QENV="";;
  *) echo "bad MODE $MODE"; exit 2;;
esac
t0=$(date +%s)
echo "CMD env $QENV LLAMA_QSA_TRACE=1 $* $BIN/llama-perplexity -c $CTX -b $B -ub $UB --chunks $CH -ctk $CTK -ctv $CTV ${QARGS:-} $KARGS" > $L/$TAG.cmd
flock ~/gpu.lock env $QENV LLAMA_QSA_TRACE=1 QKL_HASH=1 "$@" timeout ${QTIMEOUT:-7200} $BIN/llama-perplexity -m $M -f $W -c $CTX -b $B -ub $UB --chunks $CH ${QNGL:--ngl 99} -fa ${QFA:-on} \
   -lm mmap -lzm on-direct -ctk $CTK -ctv $CTV ${QARGS:-} $KARGS > $L/$TAG.log 2>&1
rc=$?
echo "R $TAG rc=$rc $(( $(date +%s)-t0 ))s arm=$ARM b=$B ub=$UB c=$CTX x$CH $CTK/$CTV mode=$MODE env=[$*] args=[${QARGS:-}] | $(grep -h "Mean    KLD" $L/$TAG.log) | $(grep -h QPIN_SUMMARY $L/$TAG.log | tail -1 | cut -c1-150)"
