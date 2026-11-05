#!/bin/bash
# R1 split-stream reserve probe: TAG KV "EXTRA_SERVER_ARGS" REQ0 REQ1 DELAY
# Signal-3.8 + mmproj + MTP, -np 2, -c 131072, -b/-ub 1024, instrumented PR build (C15 trace, GGML_C15=1)
TAG=$1; KV=$2; EXTRA=$3; Q0=$4; Q1=$5; DELAY=${6:-5}
B=~/dev/wt_lab/c15/tree/build/bin/llama-server
S=~/models/signal3.8; R=~/dev/wt_lab/r1s
O=$R/logs; mkdir -p $O/dump_$TAG; rm -f $O/dump_$TAG/*
LOG=$O/srv_$TAG.log
exec 9>~/gpu.lock; flock 9
env GGML_C15=1 GGML_C15_DUMP=$O/dump_$TAG $B -m $S/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf \
  --spec-draft-model ~/models/qwen3.8-flash-next-mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
  -mm $S/mmproj-Signal-3.8-Flash-Next-f16.gguf --mmproj-device ROCm0 -dev ROCm0 --spec-draft-device ROCm0 \
  --metrics --host 127.0.0.1 --port 8097 --threads 8 --load-mode none --lazy-mode on-direct \
  --ctx-size 131072 --parallel 2 $EXTRA --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-ngl all \
  -ngl 99 --n-gpu-layers-draft 99 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05 -ub 1024 -b 1024 \
  --presence-penalty 0.0 --repeat-penalty 1.0 --image-min-tokens 1024 -ctk $KV -ctv $KV \
  --reasoning-preserve --jinja -fa on -fit off -lv 3 > $LOG 2>&1 &
PID=$!
for i in $(seq 300); do
  grep -qE "listening on|exiting due to|failed to" $LOG && break
  kill -0 $PID 2>/dev/null || break
  sleep 1
done
grep -q "listening on" $LOG || { echo "server did not start"; tail -5 $LOG; kill -9 $PID; exit 1; }
q() { curl -s -m 1500 localhost:8097/v1/chat/completions -H "Content-Type: application/json" -d @$R/$1.json \
    | python3 -c "import json,sys
r=json.load(sys.stdin)
if \"error\" in r: print(\"$1 ERROR\", r[\"error\"]); sys.exit()
t=r.get(\"timings\",{}); print(\"$1\", repr(r[\"choices\"][0][\"message\"][\"content\"][:80]), {k:t.get(k) for k in (\"prompt_n\",\"prompt_per_second\",\"draft_n\",\"draft_n_accepted\")})"; }
echo "C15MARK concurrent start" >> $LOG
q $Q0 & P0=$!; sleep $DELAY; echo "C15MARK second request sent" >> $LOG; q $Q1 & P1=$!; wait $P0 $P1
echo "C15MARK concurrent end" >> $LOG
kill -INT $PID; for i in $(seq 120); do kill -0 $PID 2>/dev/null || break; sleep 1; done
kill -0 $PID 2>/dev/null && { echo "forcing kill"; kill -9 $PID; }
wait $PID 2>/dev/null
grep -h "does not match\|matches expectation" $LOG | sed "s/^[0-9.]* //"
echo R1S_DONE
