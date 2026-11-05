#!/bin/bash
# W12 / AT-W12-BANDS runner (scripts/qsa-validation/w12). Derived from r1ub/run.sh and r1q8/run2.sh:
#   run.sh TAG REF MODE CTK CTV UB CTX CHUNKS BK [ENV=VAL ...]
#   MODE  record   QPIN record into REF/pins and write the KLD base REF/base.kld (the reference itself)
#         replay   QPIN replay of REF/pins, QPIN_PIN=qsa,moe, KLD against REF/base.kld
#         observe  QPIN replay without pins (agreement only), KLD against REF/base.kld
#         pinbase  QPIN replay of REF/pins (pinned) and write the KLD base $W12_KBASE (direct form: a clean same-type base)
#         pinkld   QPIN replay of REF/pins (pinned), KLD against $W12_KBASE
#   BK    hip (W0-lab HIP build) | vk (W0-lab Vulkan build) | vkold (round-1 PR219+#240 Vulkan tree, V2 (i)/(ii))
#   W12_B overrides -b (default 4096). Every GPU command runs under flock ~/gpu.lock.
set -u
TAG=$1 REF=$2 MODE=$3 CTK=$4 CTV=$5 UB=$6 CTX=$7 CH=$8 BK=$9; shift 9
M=$(ls ~/models/qwen3.8-flash-next-UD-Q4_K_XL/*00001-of*.gguf)
W=~/data/wikitext-2-raw/wiki.test.raw
case $BK in
  hip)   BIN=${W12_HIP_BIN:-~/dev/wt_lab/w0lab/build/bin};;
  vk)    BIN=${W12_VK_BIN:-~/dev/wt_lab/w0lab/build-vk/bin};;
  vkold) BIN=~/dev/wt_lab/r1ub/tree/build-vk/bin;;
  *) echo "bad BK $BK"; exit 2;;
esac
L=${W12_LOGS:-~/dev/wt_lab/w12/logs}; mkdir -p $L $REF
B=${W12_B:-4096}
KLD=$REF/base.kld
case $MODE in
  record)  KARGS="--kl-divergence-base $KLD"; QENV="QPIN=record QPIN_DIR=$REF/pins"; mkdir -p $REF/pins;;
  observe) KARGS="--kl-divergence-base $KLD --kl-divergence"; QENV="QPIN=replay QPIN_DIR=$REF/pins QPIN_STATS=$L/$TAG.stats";;
  replay)  KARGS="--kl-divergence-base $KLD --kl-divergence"; QENV="QPIN=replay QPIN_DIR=$REF/pins QPIN_STATS=$L/$TAG.stats QPIN_PIN=qsa,moe";;
  pinbase) KARGS="--kl-divergence-base ${W12_KBASE:?}"; QENV="QPIN=replay QPIN_DIR=$REF/pins QPIN_STATS=$L/$TAG.stats QPIN_PIN=qsa,moe";;
  pinkld)  KARGS="--kl-divergence-base ${W12_KBASE:?} --kl-divergence"; QENV="QPIN=replay QPIN_DIR=$REF/pins QPIN_STATS=$L/$TAG.stats QPIN_PIN=qsa,moe";;
  *) echo "bad MODE $MODE"; exit 2;;
esac
t0=$(date +%s)
echo "CMD env $QENV GGML_QSA_TRACE=1 $* $BIN/llama-perplexity -c $CTX -b $B -ub $UB --chunks $CH -ctk $CTK -ctv $CTV $KARGS" > $L/$TAG.cmd
flock ~/gpu.lock env $QENV GGML_QSA_TRACE=1 "$@" timeout 5400 $BIN/llama-perplexity -m $M -f $W -c $CTX -b $B -ub $UB --chunks $CH -ngl 99 -fa on \
   -lm mmap -lzm on-direct -ctk $CTK -ctv $CTV $KARGS > $L/$TAG.log 2>&1
rc=$?
echo "R $TAG rc=$rc $(( $(date +%s)-t0 ))s bk=$BK b=$B ub=$UB c=$CTX x$CH $CTK/$CTV mode=$MODE env=[$*] | $(grep -h "Mean    KLD" $L/$TAG.log) | $(grep -h QPIN_SUMMARY $L/$TAG.log | tail -1)"
