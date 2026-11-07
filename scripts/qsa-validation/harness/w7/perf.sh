#!/bin/bash
# W7 performance gates (PLAN-v3 §16): C10 text llama-bench, §16 design: a warm-up instance before each cell group, interleaved processes in one campaign,
# 5 reps per process (rep 0 dropped in analyze.py), KFD evicted_ms and pgmigrate_success sampled per process (C7 sampler, 100 ms).
#   perf.sh PHASE...: P1 P23 P4 P5 P6N ND (nkvo gate) QUICKNK
. "$(dirname "$0")/env.sh"
OUT=$W7/perf; mkdir -p $OUT
TXT=$WIKI,$HOME/dev/wt_lab/c10/corp/code.txt
bin_of() { case $1 in new) echo $W7/bench/bin_w7/llama-bench;; w0) echo $W7/bench/bin_w0/llama-bench;; base) echo ~/dev/wt_lab/c10/bin_base/llama-bench;; esac; }
SAMP=$T/scripts/qsa-validation/kfd/c7samp.py
one() { # tag arm k v "bench args" [env...]
  local tag=$1 arm=$2 k=$3 v=$4 args=$5; shift 5
  local B=$(bin_of $arm) t0=$(date +%s)
  [ -s $OUT/$tag.json ] && { echo "skip $tag"; return; }
  ( flock 9
    env C10_TEXT=$TXT "$@" $B -m $M -ngl ${NGL:-99} -fa 1 -lzm on-direct -ctk $k -ctv $v $args -r ${R:-5} --progress -o json > $OUT/$tag.json 2> $OUT/$tag.err &
    P=$!
    python3 $SAMP $P $OUT/$tag.samp 0.1 &
    S=$!
    wait $P; echo $? > $OUT/$tag.rc; wait $S
  ) 9> ~/gpu.lock
  echo "$(date +%T) $tag rc=$(cat $OUT/$tag.rc) secs=$(( $(date +%s)-t0 ))"
}
warm() { # arm k v
  local B=$(bin_of $1)
  flock ~/gpu.lock env C10_TEXT=$TXT $B -m $M -ngl 99 -fa 1 -lzm on-direct -ctk $2 -ctv $3 -b 4096 -ub 4096 -p 512 -n 0 -d 0 -r 2 -o json > /dev/null 2>&1
}
cells() { # depth -> pp list of the C10 cells
  if [ $1 = 0 ]; then echo 512,2048,4096,8192; else echo 512,4096; fi
}
phase_P1() {  # text q8_0 ub512 pp4096 d0: new vs base 13a279bfa (>= 0.995)
  warm base q8_0 q8_0
  for i in 0 1 2 3 4 5 6; do
    if [ $((i % 2)) = 0 ]; then a=base; else a=new; fi
    one p1_${i}_$a $a q8_0 q8_0 "-b 4096 -ub 512 -p 4096 -n 0 -d 0"
  done
}
phase_P23() { # f16 and q8_0, new and W0, four arms interleaved, every C10 cell (P2 >= 0.99; P3 ratio of ratios >= 0.99, new_q8/new_f16 >= 0.90)
  for d in ${DEPTHS:-0 32768 131072}; do
    P=$(cells $d)
    warm w0 f16 f16
    for i in ${PROCS:-0 1 2 3}; do
      for a in w0 new; do for t in f16 q8_0; do
        one p23_d${d}_${t}_${i}_$a $a $t $t "-b 4096 -ub 4096,512 -p $P -n 0 -d $d"
      done; done
    done
  done
}
phase_P4() { # the other pairs: new_T / new_f16 >= 0.90, same cells; new_f16 reference runs interleaved
  for d in ${DEPTHS:-0 32768 131072}; do
    P=$(cells $d)
    warm new f16 f16
    for i in ${PROCS:-0 1}; do
      for p in q4_0:q4_0 bf16:bf16 q8_0:f16 f16:q8_0 f32:f32 q4_1:q4_1 q5_0:q5_0 q5_1:q5_1 iq4_nl:iq4_nl; do
        one p4_d${d}_$(pair_tag $p)_${i}_new new $(pair_k $p) $(pair_v $p) "-b 4096 -ub 4096,512 -p $P -n 0 -d $d"
      done
      one p4_d${d}_f16_f16_${i}_new new f16 f16 "-b 4096 -ub 4096,512 -p $P -n 0 -d $d"
    done
  done
}
phase_P5() { # decode tg128: shortcut d0/1024/1900 new_T vs W0_T >= 0.98; depth d32k/d131k new_T vs W0_T >= 0.99 (f16, q8_0); new_T/new_f16 >= 0.94 for every S_fast pair
  warm new f16 f16
  for i in ${PROCS:-0 1 2 3 4 5 6}; do
    if [ $((i % 2)) = 0 ]; then a=w0; else a=new; fi
    for p in f16:f16 q8_0:q8_0 q4_0:q4_0 bf16:bf16 q8_0:f16 f16:q8_0; do
      one p5_$(pair_tag $p)_${i}_$a $a $(pair_k $p) $(pair_v $p) "-b 4096 -ub 512 -p 0 -n 128 -d 0,1024,1900,32768,131072"
    done
  done
}
phase_P6N() { # eviction-sample rate new vs base, >= 30 reps (reported): ub512 pp512 d0 f16 / q8_0 / q4_0, ub4096 pp2048 d32k
  for i in 0 1; do
    if [ $i = 0 ]; then a=base; else a=new; fi
    for t in f16 q8_0 q4_0; do R=30 one p6_${t}_d0_$a $a $t $t "-b 4096 -ub 512 -p 512 -n 0 -d 0"; done
    R=30 one p6_q8_0_d32k_$a $a q8_0 q8_0 "-b 4096 -ub 4096 -p 2048 -n 0 -d 32768"
  done
}
phase_ND() { # AT-W7PERF (c): -nkvo, W7 (CPU route, fallback (a)) vs W0-lab, interleaved
  for t in ${NKTYPES:-f16 q8_0 q4_0 bf16}; do
    warm w0 $t $t
    for i in ${PROCS:-0 1 2}; do
      if [ $((i % 2)) = 0 ]; then a=w0; else a=new; fi
      one nk_${t}_pp_${i}_$a $a $t $t "-nkvo 1 -b 4096 -ub 512,4096 -p 512,2048 -n 0 -d 0,4096,32768" LLAMA_QSA_HOST_KV_COPY=0
      one nk_${t}_tg_${i}_$a $a $t $t "-nkvo 1 -b 4096 -ub 512 -p 0 -n 32 -d 4096,32768" LLAMA_QSA_HOST_KV_COPY=0
    done
  done
}
phase_NDB() { # AT-W7PERF (e): fallback (b), the named host-KV copy route, on the same grid
  for t in ${NKTYPES:-f16 q8_0 q4_0 bf16}; do
    warm w0 $t $t
    for i in ${PROCS:-0 1 2 3 4 5 6}; do
      if [ $((i % 2)) = 0 ]; then a=w0; else a=new; fi
      one nkb_${t}_pp_${i}_$a $a $t $t "-nkvo 1 -b 4096 -ub 512,4096 -p 512,2048 -n 0 -d 0,4096,32768" LLAMA_QSA_HOST_KV_COPY=1
      one nkb_${t}_tg_${i}_$a $a $t $t "-nkvo 1 -b 4096 -ub 512 -p 0 -n 32 -d 4096,32768" LLAMA_QSA_HOST_KV_COPY=1
    done
  done
}
phase_NPIN() { # AT-W7PERF (d): shortcut pin on vs off, pp512 and tg32 at d0/d1024, -nkvo
  for t in f16; do
    for i in 0 1 2 3 4 5 6; do
      if [ $((i % 2)) = 0 ]; then e=LLAMA_QSA_SHORTCUT_PIN=0; a=off; else e=LLAMA_QSA_SHORTCUT_PIN=1; a=on; fi
      one npin_${t}_pp_${i}_$a new $t $t "-nkvo 1 -b 4096 -ub 512 -p 512 -n 0 -d 0,1024" $e
      one npin_${t}_tg_${i}_$a new $t $t "-nkvo 1 -b 4096 -ub 512 -p 0 -n 32 -d 0,1024" $e
    done
  done
}
for ph in "$@"; do echo "PHASE $ph start $(date +%T)"; phase_$ph; echo "PHASE $ph done $(date +%T)"; done
echo PERF_DONE $(date +%T)
