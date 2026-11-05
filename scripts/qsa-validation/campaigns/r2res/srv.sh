#!/bin/bash
# R2-M-LAB-RES: srv.sh TAG KV "EXTRA_ARGS" "ENV=.."   init-only, Signal-3.8 + mmproj + MTP, R1-C flags minus --parallel, -fit on, -lv 5
TAG=$1; KV=$2; EXTRA=$3; ENVS=$4
BIN=~/dev/wt_lab/r2res/tree/build/bin/llama-server
O=~/dev/wt_lab/r2res/logs; LOG=$O/srv_$TAG.log
S=~/models/signal3.8
echo "CMD: env LAB_NC_LOG=1 $ENVS $BIN -lv 5 -m $S/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf -mm $S/mmproj-Signal-3.8-Flash-Next-f16.gguf ... $EXTRA -ctk $KV -ctv $KV -fit on" > $LOG.cmd
env LAB_NC_LOG=1 $ENVS $BIN -lv 5 -m $S/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf -mm $S/mmproj-Signal-3.8-Flash-Next-f16.gguf --mmproj-device ROCm0 --image-min-tokens 1024 \
  --spec-draft-model ~/models/qwen3.8-flash-next-mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf -dev ROCm0 --spec-draft-device ROCm0 \
  --metrics --host 127.0.0.1 --port 8099 --threads 8 --load-mode none --lazy-mode on-direct \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-ngl all \
  -ngl 99 --n-gpu-layers-draft 99 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05 $EXTRA \
  --presence-penalty 0.0 --repeat-penalty 1.0 -ctk $KV -ctv $KV \
  --reasoning-preserve --jinja -fa on -fit on > $LOG 2>&1 &
PID=$!
for i in $(seq 600); do
  grep -qE "listening on|exiting due to|failed to load" $LOG && break
  kill -0 $PID 2>/dev/null || break
  sleep 1
done
grep -q "listening on" $LOG && echo "$TAG started" || { echo "$TAG did not start"; grep -E "cudaMalloc|failed|out of memory|error" $LOG | head -5; }
kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
grep -E "compute buffer size|LAB_NC variant|n_ctx  |n_slots" $LOG | head -60
