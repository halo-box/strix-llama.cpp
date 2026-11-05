#!/bin/bash
# R1-M-Lab C (b) concurrent layout: srv2.sh TAG "ENV=.."   default server (no --parallel), Signal-3.8+mmproj+MTP, c131072 ub4096 q8_0.
# Request A = B65t (deep image at cell 57504), cache_prompt, ignore_eos, max_tokens 1200 -> slot A keeps generating;
# once slot A is generating (first "task 0 | generate_draft" line), request T (12027-token text) is sent -> another slot.
# (Sequential A then T is a null layout: the default --cache-idle-slots saves and clears slot A before T, arg.cpp:1761.)
TAG=$1; ENVS=$2; REQ2=${3:-T12k_cp.json}
BIN=~/dev/wt_lab/c19/tree/build/bin/llama-server; S=~/models/signal3.8
O=~/dev/wt_lab/r1c/logs; LOG=$O/srv_$TAG.log
env $ENVS $BIN -v -m $S/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf -mm $S/mmproj-Signal-3.8-Flash-Next-f16.gguf --mmproj-device ROCm0 --image-min-tokens 1024 \
  --spec-draft-model ~/models/qwen3.8-flash-next-mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf -dev ROCm0 --spec-draft-device ROCm0 \
  --metrics --host 127.0.0.1 --port 8099 --threads 8 --load-mode none --lazy-mode on-direct \
  --ctx-size 131072 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-ngl all \
  -ngl 99 --n-gpu-layers-draft 99 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05 -ub 4096 -b 4096 \
  --presence-penalty 0.0 --repeat-penalty 1.0 -ctk q8_0 -ctv q8_0 \
  --reasoning-preserve --jinja -fa on -fit off > $LOG 2>&1 &
PID=$!
for i in $(seq 400); do grep -qE "listening on|exiting due to|failed to" $LOG && break; kill -0 $PID 2>/dev/null || break; sleep 1; done
grep -q "listening on" $LOG || { echo "server did not start"; kill -9 $PID; wait $PID 2>/dev/null; exit 1; }
req() { curl -s -m 1500 localhost:8099/v1/chat/completions -H "Content-Type: application/json" -d @$1 | python3 -c "import json,sys
r=json.load(sys.stdin)
if \"error\" in r: print(\"ERROR\", r[\"error\"]); sys.exit()
t=r.get(\"timings\",{}); print(\"$1\", repr(r[\"choices\"][0][\"message\"][\"content\"][:60]), {k:t.get(k) for k in (\"prompt_n\",\"prompt_per_second\",\"predicted_n\")})"; }
req A_B65t_long.json > $O/resA_$TAG.txt &
PA=$!
for i in $(seq 600); do grep -q "task 0 | generate_draft" $LOG && break; sleep 0.5; done
echo "T sent at $(date +%T.%N), log line $(wc -l < $LOG)"
req $REQ2 > $O/resT_$TAG.txt
wait $PA
cat $O/resA_$TAG.txt $O/resT_$TAG.txt
kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
grep -E "sched_reserve: .*compute buffer size|n_slots" $LOG
