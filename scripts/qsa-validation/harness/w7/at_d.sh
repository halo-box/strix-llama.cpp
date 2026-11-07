#!/bin/bash
# AT-D (PLAN-v3 §16.5): HIP graph placement. W7 vs W0-lab. at_d.sh D1 D2 D3 D4 D5
. "$(dirname "$0")/env.sh"
OUT=$W7/at_d; mkdir -p $OUT
N=$T/build/bin              # W7
Z=$W7/w0ovl/build/bin       # W0-lab + LAB_OVL_PRINT
Z0=~/dev/wt_lab/w0lab12/build/bin
CC="python3 $T/scripts/qsa-validation/census_check.py"
ppl() { # tag bin ctx ub chunks kv extra env...
  local tag=$1 B=$2 c=$3 u=$4 ch=$5 kv=$6 x=$7; shift 7
  flock ~/gpu.lock env LLAMA_QSA_TRACE=1 "$@" timeout ${ATO:-7200} $B/llama-perplexity -m $M -f $WIKI -c $c -b ${PB:-4096} -ub $u --chunks $ch -lm mmap -lzm on-direct -ctk $kv -ctv $kv $x > $OUT/$tag.log 2>&1
  echo "$tag rc=$? $(grep -h 'Final estimate' $OUT/$tag.log | tail -1 | sed 's/.*Final/Final/') $(grep -h 'seconds per pass' $OUT/$tag.log | tail -1 | sed 's/.*seconds per pass/s\/pass/' | cut -c1-40) guard=[$(grep LLAMA_QSA_GUARD $OUT/$tag.log | tail -1 | cut -d' ' -f3-)]"
}
phase_D1() {
  for t in f16 q8_0; do for arm in new w0; do
    [ $arm = new ] && B=$N || B=$Z
    flock ~/gpu.lock env LAB_OVL_PRINT=1 $B/llama-bench -m $M -ngl 99 -fa 1 -lzm on-direct -ctk $t -ctv $t -b 4096 -ub 4096 -p 4096 -n 0 -d 4096,32768 -r 1 -o json > $OUT/d1_${t}_$arm.json 2> $OUT/d1_${t}_$arm.err
    echo "d1 $t $arm rc=$? kind4 per graph: $(grep LAB_OVL $OUT/d1_${t}_$arm.err | sed 's/.*kind4=//' | sort | uniq -c | tr '\n' ';')"
  done; done
  # U16: 20 chunks, HIP graphs on, OVL=15 vs OVL=11, logits hashes
  for t in f16 q8_0; do for ov in 15 11; do
    flock ~/gpu.lock env GGML_CUDA_OVL=$ov QKL_HASH=1 $N/llama-perplexity -m $M -f $WIKI -c 8192 -b 4096 -ub 4096 --chunks 20 -ngl 99 -fa on -lm mmap -lzm on-direct -ctk $t -ctv $t --kl-divergence-base $OUT/u16_${t}_$ov.kld > $OUT/u16_${t}_$ov.log 2>&1
    echo "u16 $t OVL=$ov rc=$?"; rm -f $OUT/u16_${t}_$ov.kld
  done
  echo "u16 $t hash: $(diff <(grep -h 'QKL_HASH base' $OUT/u16_${t}_15.log) <(grep -h 'QKL_HASH base' $OUT/u16_${t}_11.log) | wc -l) differing lines of $(grep -c 'QKL_HASH base' $OUT/u16_${t}_15.log)"; done
}
phase_D2() {
  for t in f16 q8_0; do for i in 0 1 2 3 4 5; do
    if [ $((i % 2)) = 0 ]; then a=w0; B=$Z0; else a=new; B=$N; fi
    ATO=14400 PB=512 ppl d2_${t}_${i}_$a $B 4096 512 4 $t "-ngl 0 -fa on"
  done; done
}
phase_D3() {
  for t in f16 q8_0; do
    # an -ngl that splits the QSA layers 3, 7, ..., 47 between the CPU and ROCm0 (the last 24 layers offloaded)
    PB=512 ATO=14400 ppl d3_${t}_new $N 2048 512 1 $t "-ngl 24 -fa on"
    grep -E 'LLAMA_QSA_TRACE' $OUT/d3_${t}_new.log | sed 's/.*il=\([0-9]*\) .*backend=\([^ ]*\) .*/\1 \2/' | sort -n | uniq -c | sed 's/^/  placement /'
  done
}
phase_D4() {
  for t in f16 q8_0; do for u in 4096 512 16; do
    PB=4096 ppl d4_${t}_u${u}_new $N 8192 $u 2 $t "-ngl 99 -nkvo -fa on"
    PB=4096 ppl d4_${t}_u${u}_w0 $Z0 8192 $u 2 $t "-ngl 99 -nkvo -fa on"
    flock ~/gpu.lock env GGML_SCHED_DEBUG=2 $N/llama-perplexity -v -m $M -f $WIKI -c 4096 -b 4096 -ub $u --chunks 1 -ngl 99 -nkvo -fa on -lm mmap -lzm on-direct -ctk $t -ctv $t 2>&1 | grep -E "^.*(SPARSE_ATTN|FLASH_ATTN|## SPLIT|cache_[kv]_l)" | head -4000 > $OUT/d4_${t}_u${u}_sched.txt
    echo "d4 sched $t u$u: cache_[kv] split inputs to ROCm0 lines: $(grep -c 'cache_[kv]_l' $OUT/d4_${t}_u${u}_sched.txt)"
  done; done
}
phase_D5() {
  PB=4096 ppl d5_nkvo_auto_new $N 4096 512 2 f16 "-ngl 99 -nkvo -fa auto"
  for a in new w0; do
    [ $a = new ] && B=$N || B=$Z0
    for u in 512 4096; do
      PB=4096 ppl d5_off_u${u}_$a $B 8192 $u 2 f16 "-ngl 99 -fa off"
      PB=4096 ppl d5_off_nkvo_u${u}_$a $B 8192 $u 2 f16 "-ngl 99 -nkvo -fa off"
    done
  done
  grep -h -E "flash_attn|LLAMA_QSA_TRACE il=3 " $OUT/d5_nkvo_auto_new.log | cut -c1-170 | head -8
}
for ph in "$@"; do echo "PHASE $ph start $(date +%T)"; phase_$ph; echo "PHASE $ph done $(date +%T)"; done
echo AT_D_DONE
