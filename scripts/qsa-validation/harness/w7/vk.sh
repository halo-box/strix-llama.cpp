#!/bin/bash
# AT-B (PLAN-v3 §16.5, W6/W7), on the switched graph. vk.sh PHASE...: VB (builds) VC (census) VQ (Q1-vk / AT-B6) VO (AT-B4 op level) VE (AT-B4 e2e) V1 (AT-B1) V5 (AT-B5)
. "$(dirname "$0")/env.sh"
OUT=$W7/vk; mkdir -p $OUT/logs
TXT=$WIKI,$HOME/dev/wt_lab/c10/corp/code.txt
LAB=$W7/lab
NVK=$T/build-vk/bin; ZVK=$HOME/dev/wt_lab/w0lab12/build-vk/bin
PAIRS_ALL="f16:f16 q8_0:q8_0 q4_0:q4_0 bf16:bf16 q8_0:f16 f16:q8_0"
bb() { # tag tree
  local tag=$1 T0=$2 B=$2/build-vk O=$W7/bench/bin_$1 S=$W7/bench/src_$1
  mkdir -p $O $S; cp $T0/tools/llama-bench/main.cpp $S/main.cpp
  python3 $T/scripts/qsa-validation/bench/c10/patch_c10.py $T0/tools/llama-bench/llama-bench.cpp $S/llama-bench.cpp
  local DEF="-DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_TRELLIS -DGGML_USE_VULKAN -DLLAMA_SHARED -DLLAMA_SUBPROCESS"
  local INC="-I$T0/tools/llama-bench -I$T0/common/. -I$T0/vendor/nlohmann/.. -I$T0/vendor/sheredom/.. -I$T0/src/../include -I$T0/ggml/src/../include"
  local LIBS="$B/bin/libllama-common.so.0.6.0 $B/bin/libllama.so.0.6.0 $B/bin/libggml.so.0.26.0 $B/bin/libggml-cpu.so.0.26.0 $B/bin/libggml-vulkan.so.0.26.0 $B/bin/libggml-base.so.0.26.0"
  local RP="-Wl,-rpath,$O:$B/bin"
  c++ $DEF -Dllama_bench_impl_EXPORTS $INC -O3 -DNDEBUG -Wno-unused-function -fPIC -std=gnu++17 -c $S/llama-bench.cpp -o $O/llama-bench.o
  c++ -fPIC -O3 -DNDEBUG -shared -Wl,-soname,libllama-bench-impl.so -o $O/libllama-bench-impl.so $O/llama-bench.o $RP $LIBS $B/common/libllama-common-base.a
  c++ $DEF $INC -O3 -DNDEBUG -std=gnu++17 -c $S/main.cpp -o $O/main.o
  c++ -O3 -DNDEBUG $O/main.o -o $O/llama-bench $RP $O/libllama-bench-impl.so $B/bin/libllama-common.so.0.6.0 $B/common/libllama-common-base.a $LIBS
}
phase_VB() {
  flock ~/gpu.lock cmake --build $T/build-vk -j 16 > $OUT/build_new.log 2>&1; echo "vk new build rc=$? warnings=$(grep -c 'warning:' $OUT/build_new.log)"
  (cd $LAB && cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > $OUT/lab_cfg.log 2>&1 && flock ~/gpu.lock cmake --build build-vk -j 16 --target llama-perplexity > $OUT/build_lab.log 2>&1; echo "vk lab build rc=$?")
  bb vk_new $T > $OUT/bb_new.log 2>&1; echo "bench vk new rc=$?"
  bb vk_w0 $HOME/dev/wt_lab/w0lab12 > $OUT/bb_w0.log 2>&1; echo "bench vk w0 rc=$?"
}
phase_VC() {
  for p in $PAIRS_ALL; do k=$(pair_k $p); v=$(pair_v $p); t=$(pair_tag $p)
    for u in 4096 512; do
      log=$OUT/logs/c_${t}_u$u.log
      flock ~/gpu.lock env LLAMA_QSA_TRACE=1 timeout 1800 $NVK/llama-perplexity -m $M -f $WIKI -c 8192 -b 4096 -ub $u --chunks 1 $COMMON -ctk $k -ctv $v > $log 2>&1
      echo "VC $t u$u rc=$? $(grep -h 'Final estimate' $log | tail -1 | sed 's/.*Final/Final/') masked=$(grep -c 'producer=masked' $log) cpu=$(grep -c 'QSA_CENSUS cpu:' $log) guard=[$(grep LLAMA_QSA_GUARD $log | tail -1 | cut -d' ' -f3-)] sa=[$(grep QSA_CENSUS $log | grep -E 'sa_' | cut -d' ' -f2- | tr '\n' ';' | cut -c1-150)]"
    done
  done
}
phase_VQ() {
  local F=$W7/vk/refs/vref8t0 RUNV="$T/scripts/qsa-validation/w12/run.sh"
  export W12_VK_BIN=$ZVK W12_LOGS=$OUT/logs
  $RUNV vref8t0_rec $F record f16 f16 4096 8192 8 vk LLAMA_FN_LAST_ATTN_TRIM=0
  for p in $PAIRS_ALL; do k=$(pair_k $p); v=$(pair_v $p); t=$(pair_tag $p)
    W12_VK_BIN=$ZVK $RUNV vq_w0_$t $F replay $k $v 4096 8192 8 vk
    W12_VK_BIN=$LAB/build-vk/bin $RUNV vq_new_$t $F replay $k $v 4096 8192 8 vk
  done
  rm -rf $F
  python3 - "$OUT/logs" <<'PY'
import re, sys, math
L = sys.argv[1]; B = 3.214e-4
def k(t):
    d = {}
    for l in open(f'{L}/{t}.log', errors='replace'):
        m = re.search(r'QKLD_CHUNK (\d+) all=(\S+)', l)
        if m: d[int(m.group(1))] = float(m.group(2))
    return d
for t in 'f16_f16 q8_0_q8_0 q4_0_q4_0 bf16_bf16 q8_0_f16 f16_q8_0'.split():
    try:
        a, b = k(f'vq_new_{t}'), k(f'vq_w0_{t}')
        x = [a[c] - b[c] for c in sorted(a) if c in b]
        mu = sum(x)/len(x); se = math.sqrt(sum((v-mu)**2 for v in x)/(len(x)-1)/len(x))
        print(f"Q1-vk {t}: mean diff {mu:+.3e} SE {se:.2e} band {B:.3e} {'PASS' if mu <= B else 'FAIL'}")
    except Exception as e: print('Q1-vk', t, 'missing', e)
PY
}
phase_VO() {
  B=$T/build-vk
  c++ -O2 -std=c++17 -I$T/ggml/include $T/scripts/qsa-validation/harness/w7/w7_op.cpp -o $OUT/w7_op_vk -L$B/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$B/bin || return 1
  for pr in 1 2 3; do
    flock ~/gpu.lock $OUT/w7_op_vk p$pr Vulkan0 5 q8_0:q8_0,q4_0:q4_0,bf16:bf16,q8_0:f16,f16:q8_0 2,4,8,16,32,64,127,512,1024,2048 512,1024,2048 > $OUT/op_p$pr.out 2> $OUT/op_p$pr.err; echo "VO p$pr rc=$?"
  done
  mkdir -p $OUT/op; cp $OUT/op_p?.out $OUT/op/; (cd $OUT/op && for f in op_p?.out; do mv $f ${f#op_}; done); python3 $T/scripts/qsa-validation/harness/w7/at_op_an.py $OUT/op > $OUT/op/summary.md
}
phase_VE() {
  local E=$OUT/e2e; mkdir -p $E
  for p in q8_0:q8_0 q4_0:q4_0 bf16:bf16 q8_0:f16 f16:q8_0; do k=$(pair_k $p); v=$(pair_v $p); t=$(pair_tag $p)
    for i in 0 1 2 3 4 5 6; do
      if [ $((i % 2)) = 0 ]; then d=1; a=dense; else d=0; a=sparse; fi
      flock ~/gpu.lock env C10_TEXT=$TXT LLAMA_QSA_PREFER_DENSE=$d $W7/bench/bin_vk_new/llama-bench -m $M -ngl 99 -fa 1 -lzm on-direct -ctk $k -ctv $v -b 4096 -ub 64,512,1024,2048 -p 2048 -n 0 -d 0 -r 3 -o json > $E/${t}_${i}_${a}_pp.json 2> $E/${t}_${i}_${a}_pp.err; ra=$?
      flock ~/gpu.lock env C10_TEXT=$TXT LLAMA_QSA_PREFER_DENSE=$d $W7/bench/bin_vk_new/llama-bench -m $M -ngl 99 -fa 1 -lzm on-direct -ctk $k -ctv $v -b 4096 -ub 512 -p 0 -n 128 -d 0,1024,1900 -r 3 -o json > $E/${t}_${i}_${a}_tg.json 2> $E/${t}_${i}_${a}_tg.err
      echo "$(date +%T) VE ${t}_${i}_$a pp=$ra tg=$?"
    done
  done
  python3 $T/scripts/qsa-validation/harness/w7/at_e2e_an.py $E > $E/summary.md
}
phase_V1() {
  local D=$OUT/at_b1; mkdir -p $D
  for pp in f16:f16 q8_0:q8_0 q4_0:q4_0 bf16:bf16 q8_0:f16 f16:q8_0; do k=$(pair_k $pp); v=$(pair_v $pp); t=$(pair_tag $pp)
    for i in 0 1 2 3 4 5 6 7; do
      if [ $((i % 2)) = 0 ]; then a=w0; B=$W7/bench/bin_vk_w0; else a=new; B=$W7/bench/bin_vk_new; fi
      flock ~/gpu.lock env C10_TEXT=$TXT $B/llama-bench -m $M -ngl 99 -fa 1 -lzm on-direct -ctk $k -ctv $v -b 4096 -ub 4096,512 -p 512,2048 -n 0 -d 4096,32768,131072 -r 5 -o json > $D/b1_${t}_${i}_$a.json 2> $D/b1_${t}_${i}_$a.err
      echo "$(date +%T) V1 ${t}_${i}_$a rc=$?"
    done
  done
}
phase_V5() {
  BIN=$T/build-vk/bin
  local D=$OUT/mem; mkdir -p $D
  for c in 32768 131072; do for u in 4096 512; do for t in f16 q8_0 q4_0 bf16; do for arm in new w0; do
    [ $arm = new ] && B=$NVK || B=$ZVK
    LOG=$D/${arm}_${t}_c${c}_u$u.log
    ( flock 9
      VK_LOG_MEMORY=1 $B/llama-server -lv 4 -m $S/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf --spec-draft-model ~/models/qwen3.8-flash-next-mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf --host 127.0.0.1 --port 8094 --threads 8 --load-mode none --lazy-mode on-direct --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-ngl all -ngl 99 --n-gpu-layers-draft 99 -ctk $t -ctv $t -ctkd $t -ctvd $t --jinja -fa on -fit off -c $c -b $u -ub $u --parallel 1 > $LOG 2>&1 &
      PID=$!
      for i in $(seq 400); do grep -qE "listening on|exiting due to|failed to" $LOG && break; kill -0 $PID 2>/dev/null || break; sleep 1; done
      kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
    ) 9> ~/gpu.lock
    echo "V5 $arm $t c$c u$u: $(grep -E 'sched_reserve:.*compute buffer size' $LOG | sed 's/.*sched_reserve: *//; s/compute buffer size = *//' | tr '\n' ';' | cut -c1-120)"
  done; done; done; done
}
for ph in "$@"; do echo "PHASE $ph start $(date +%T)"; phase_$ph; echo "PHASE $ph done $(date +%T)"; done
echo VK_DONE
