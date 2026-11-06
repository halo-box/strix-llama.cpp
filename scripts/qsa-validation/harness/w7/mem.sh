#!/bin/bash
# W7 memory gates. M1 (PLAN-v3 §16): init-only llama-server, sched_reserve -lv 4, every T of S at -c {32768, 131072} b/ub {4096, 512}, main and MTP (-ctkd T),
# plus np 2 (split) and the default server rows. Compute buffers parsed by mem_an.py (the labels of campaigns/r2res/parse.py).
. "$(dirname "$0")/env.sh"
OUT=$W7/mem; mkdir -p $OUT
srv() { # tag bin kv extra
  local tag=$1 B=$2 kv=$3 extra=$4; local LOG=$OUT/srv_$tag.log
  [ -s $LOG ] && grep -q "listening on" $LOG && { echo "skip $tag"; return; }
  ( flock 9
    $B/llama-server -lv 4 -m $S/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf --spec-draft-model ~/models/qwen3.8-flash-next-mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
      -dev ROCm0 --spec-draft-device ROCm0 --host 127.0.0.1 --port 8097 --threads 8 --load-mode none --lazy-mode on-direct \
      --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-ngl all -ngl 99 --n-gpu-layers-draft 99 \
      -ctk $kv -ctv $kv -ctkd $kv -ctvd $kv --jinja -fa on -fit off $extra > $LOG 2>&1 &
    PID=$!
    for i in $(seq 400); do
      grep -qE "listening on|exiting due to|failed to" $LOG && break
      kill -0 $PID 2>/dev/null || break
      sleep 1
    done
    kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  ) 9> ~/gpu.lock
  echo "MEM $tag: $(grep -E 'sched_reserve:.*compute buffer size' $LOG | sed 's/.*sched_reserve: *//; s/compute buffer size = *//' | tr '\n' ';' | cut -c1-200) $(grep -qE 'listening on' $LOG && echo up || echo NOT_UP)"
}
BIN=${BIN:-$T/build/bin}
for c in 32768 131072; do for u in 4096 512; do
  for t in f16 q8_0 q4_0 bf16 f32 q4_1 q5_0 q5_1 iq4_nl; do
    srv m1_${t}_c${c}_u$u $BIN $t "-c $c -b $u -ub $u --parallel 1"
  done
done; done
for t in f16 q8_0 q4_0 bf16; do
  srv m1np2_${t} $BIN $t "-c 131072 -b 1024 -ub 1024 --parallel 2"
  srv m1def_${t} $BIN $t ""
done
echo MEM_DONE
