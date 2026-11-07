#!/bin/bash
# P5 MTP cells / AT-W7PERF (b): W0 vs new, main KV f16 and q8_0, draft default and -ctkd q8_0, d0 / d1024 / d32768, verify width 2-8 (--spec-draft-n-max 7). Interleaved R N R N R N R.
. "$(dirname "$0")/env.sh"
OUT=$W7/mtp_perf; mkdir -p $OUT
srv() { # tag bin main draft
  local tag=$1 B=$2 kv=$3 dkv=$4 LOG=$OUT/$1.log DF=""
  [ -s $OUT/$tag.jsonl ] && { echo "skip $tag"; return; }
  [ $dkv != default ] && DF="-ctkd $dkv -ctvd $dkv"
  ( flock 9
    $B/llama-server -m $S/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf --spec-draft-model ~/models/qwen3.8-flash-next-mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
      -dev ROCm0 --spec-draft-device ROCm0 --host 127.0.0.1 --port 8096 --threads 8 --load-mode none --lazy-mode on-direct --ctx-size 65536 --parallel 1 \
      --spec-type draft-mtp --spec-draft-n-max 7 --spec-draft-ngl all -ngl 99 --n-gpu-layers-draft 99 -ub 4096 -b 4096 -ctk $kv -ctv $kv $DF --jinja -fa on -fit off > $LOG 2>&1 &
    PID=$!
    for i in $(seq 400); do grep -qE "listening on|exiting due to|failed to" $LOG && break; kill -0 $PID 2>/dev/null || break; sleep 1; done
    python3 "$(dirname "$0")/mtp_perf.py" 8096 $WIKI $OUT/$tag.jsonl 16,1024,32768 5 128 || echo "driver failed $tag"
    kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  ) 9> ~/gpu.lock
  echo "$(date +%T) $tag $(wc -l < $OUT/$tag.jsonl)"
}
for kv in f16 q8_0; do for dk in default q8_0; do
  for i in ${PROCS:-0 1 2 3 4 5 6}; do
    if [ $((i % 2)) = 0 ]; then a=w0; B=~/dev/wt_lab/w0lab12/build/bin; else a=new; B=$T/build/bin; fi
    srv ${kv}_${dk}_${i}_$a $B $kv $dk
  done
done; done
echo MTP_PERF_DONE
