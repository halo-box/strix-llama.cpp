#!/bin/bash
# PLAN-v3 W9 re-split / strict-reserve server runs (P2-2, M4, U38), on the rewrite build, under flock ~/gpu.lock.
# Signal-3.8 + mmproj + MTP sidecar (the R1-RES r1s.sh recipe, EV/r1s/r1s.sh), -c 131072, LLAMA_STRICT_RESERVE=1 and
# LLAMA_QSA_TRACE=1. Two requests sent at once (DELAY seconds apart), so the server batches both slots' prompt tokens into
# one llama_decode batch and split_equal forms multi-sequence ubatches.
# usage: concurrent.sh TAG KV UB "EXTRA_SERVER_ARGS" REQ0 REQ1 [DELAY] [ENV...]
#   REQ = a request json (REQ1 = none: one request alone, one-stream ubatches) of ~/dev/wt_lab/r1s (S0_img: image + 12k lines, answer 11999; T_deep: text, answer 32517)
# Pass (PLAN-v3 W9 done-when, M4): answers correct; 0 "LLAMA_STRICT_RESERVE violation" lines and violations=0 in both
# contexts; decode_errors=0; for the unified runs n_qsa_resplit > 0 in the main and the MTP context.
TAG=${1:?tag}; KV=${2:?kv}; UB=${3:?ub}; EXTRA=$4; Q0=${5:?req0}; Q1=${6:?req1}; DELAY=${7:-1}; shift 7; ENVS="$@"
B=${BIN:-$HOME/dev/wt_rewrite/build/bin/llama-server}
S=~/models/signal3.8; R=~/dev/wt_lab/r1s
O=~/dev/wt_lab/w9/srv; mkdir -p $O; LOG=$O/srv_$TAG.log; OUT=$O/out_$TAG.txt
exec 9>~/gpu.lock; flock 9
env LLAMA_STRICT_RESERVE=1 LLAMA_QSA_TRACE=1 $ENVS $B -m $S/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf \
  --spec-draft-model ~/models/qwen3.8-flash-next-mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
  -mm $S/mmproj-Signal-3.8-Flash-Next-f16.gguf --mmproj-device ROCm0 -dev ROCm0 --spec-draft-device ROCm0 \
  --metrics --host 127.0.0.1 --port 8096 --threads 8 --load-mode none --lazy-mode on-direct \
  --ctx-size 131072 $EXTRA --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-ngl all \
  -ngl 99 --n-gpu-layers-draft 99 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05 -ub $UB -b $UB \
  --presence-penalty 0.0 --repeat-penalty 1.0 --image-min-tokens 1024 -ctk $KV -ctv $KV \
  --reasoning-preserve --jinja -fa on -fit off -lv 3 > $LOG 2>&1 &
PID=$!
for i in $(seq 400); do
  grep -qE "listening on|exiting due to|failed to" $LOG && break
  kill -0 $PID 2>/dev/null || break
  sleep 1
done
grep -q "listening on" $LOG || { echo "$TAG server did not start"; tail -5 $LOG; kill -9 $PID 2>/dev/null; exit 1; }
q() { curl -s -m 1800 localhost:8096/v1/chat/completions -H "Content-Type: application/json" -d @$R/$1.json \
    | python3 -c "import json,sys
r=json.load(sys.stdin)
if 'error' in r: print('$1 ERROR', r['error']); sys.exit()
t=r.get('timings',{}); print('$1', repr(r['choices'][0]['message']['content'][:80]), {k:t.get(k) for k in ('prompt_n','prompt_per_second')})"; }
: > $OUT
if [ "$Q1" = none ]; then q $Q0 >> $OUT; else q $Q0 >> $OUT & P0=$!; sleep $DELAY; q $Q1 >> $OUT & P1=$!; wait $P0 $P1; fi
kill -INT $PID; for i in $(seq 120); do kill -0 $PID 2>/dev/null || break; sleep 1; done
kill -0 $PID 2>/dev/null && { echo "$TAG forcing kill"; kill -9 $PID; }
wait $PID 2>/dev/null
cat $OUT
echo "$TAG strict_violation_lines=$(grep -c 'LLAMA_STRICT_RESERVE violation' $LOG) $(grep -h 'LLAMA_STRICT_RESERVE ctx=\|LLAMA_QSA_RESPLIT' $LOG | tr '\n' ' ')"
echo "$TAG te_lines=$(grep 'LLAMA_QSA_TRACE' $LOG | grep -c producer=te) te_nq_max=$(grep 'LLAMA_QSA_TRACE' $LOG | grep producer=te | sed 's/.* n_q=\([0-9]*\).*/\1/' | sort -n | tail -1) cb_lines=$(grep 'LLAMA_QSA_TRACE' $LOG | grep -c producer=cb) decode_err=$(grep -c 'non-compact ubatch (' $LOG) mismatch=$(grep -c 'does not match' $LOG)"
grep -h "QSA reserve\|compute buffer size" $LOG | sed 's/^[0-9.]* //' | head -12
echo "${TAG}_DONE"
