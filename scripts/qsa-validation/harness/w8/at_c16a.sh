#!/bin/bash
# AT-C16a (PLAN-v3 W8, §16.5; R1-C layout 3, the r1c/runB3.sh / srv3.sh recipe): the default server (4 slots, unified),
# Signal-3.8 + mmproj + MTP, c131072 ub4096 q8_0. Request A = B65t (deep image, cache_prompt, ignore_eos, 1200 tokens) keeps
# slot A generating; once it generates, request TI12k (image + 12k text) goes to another slot, whose text ubatches are
# non-compact (images in two slots of one unified stream) and take the TE rows (8 strips of 512 queries at 4088 tokens).
# Needs the lab logging of scripts/qsa-validation/patches/05-c19-log.patch in the build (LAB_C19_LOG=1: strip_view calls and
# compute-buffer growth); LLAMA_QSA_TRACE=1 gives the producer per executed ubatch, printed at exit (SIGINT).
# usage: at_c16a.sh TAG [BIN]   (under flock ~/gpu.lock)
# Pass (PLAN-v3:563, :788): on the slot-B text ubatches bias new > 0 and kq_mask new > 0 with reuse 88 per src per graph
# (8 strips x 11 reusing layers); main ROCm0 runtime peak <= 8054.08 MiB (MEASUREMENTS.md:4210); answers '12299...' / '40479'.
TAG=${1:?tag}; BIN=${2:-$HOME/dev/wt_rewrite/build/bin/llama-server}
R1C=~/dev/wt_lab/r1c; S=~/models/signal3.8
O=~/dev/wt_lab/w8/logs; mkdir -p $O; LOG=$O/srv_$TAG.log
env LAB_C19_LOG=1 LLAMA_QSA_TRACE=1 $BIN -v -m $S/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf -mm $S/mmproj-Signal-3.8-Flash-Next-f16.gguf --mmproj-device ROCm0 --image-min-tokens 1024 \
  --spec-draft-model ~/models/qwen3.8-flash-next-mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf -dev ROCm0 --spec-draft-device ROCm0 \
  --metrics --host 127.0.0.1 --port 8099 --threads 8 --load-mode none --lazy-mode on-direct \
  --ctx-size 131072 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-ngl all \
  -ngl 99 --n-gpu-layers-draft 99 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05 -ub 4096 -b 4096 \
  --presence-penalty 0.0 --repeat-penalty 1.0 -ctk q8_0 -ctv q8_0 \
  --reasoning-preserve --jinja -fa on -fit off > $LOG 2>&1 &
PID=$!
for i in $(seq 400); do grep -qE "listening on|exiting due to|failed to" $LOG && break; kill -0 $PID 2>/dev/null || break; sleep 1; done
grep -q "listening on" $LOG || { echo "server did not start"; kill -9 $PID; wait $PID 2>/dev/null; exit 1; }
req() { curl -s -m 1500 localhost:8099/v1/chat/completions -H "Content-Type: application/json" -d @$R1C/$1 | python3 -c "import json,sys
r=json.load(sys.stdin)
if \"error\" in r: print(\"ERROR\", r[\"error\"]); sys.exit()
t=r.get(\"timings\",{}); print(\"$1\", repr(r[\"choices\"][0][\"message\"][\"content\"][:60]), {k:t.get(k) for k in (\"prompt_n\",\"prompt_per_second\",\"predicted_n\")})"; }
req A_B65t_long.json > $O/resA_$TAG.txt &
PA=$!
for i in $(seq 600); do grep -q "task 0 | generate_draft" $LOG && break; sleep 0.5; done
echo "T sent at $(date +%T.%N), log line $(wc -l < $LOG)"
req TI12k_cp.json > $O/resT_$TAG.txt
wait $PA
cat $O/resA_$TAG.txt $O/resT_$TAG.txt
kill -INT $PID 2>/dev/null
for i in $(seq 120); do kill -0 $PID 2>/dev/null || break; sleep 1; done
kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
grep -E "sched_reserve: .*compute buffer size|n_slots" $LOG
echo AT_C16A_DONE $TAG
