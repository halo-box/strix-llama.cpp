#!/bin/bash
# W7 done-when: MTP server (C2's run_c2.sh flags, ~/dev/wt_lab/c2 requests) with draft KV default, -ctkd q8_0, -ctkd q4_0: census all sparse past 2051, 0 CPU launches.
. "$(dirname "$0")/env.sh"
OUT=$W7/mtp; mkdir -p $OUT
C2=~/dev/wt_lab/c2
BIN=${BIN:-$T/build/bin}
CC="python3 $T/scripts/qsa-validation/census_check.py"
run() { # tag draft_kv
  local TAG=$1 DKV=$2 LOG=$OUT/server_$1.log RES=$OUT/res_$1.jsonl DF=""
  : > $RES
  [ "$DKV" != default ] && DF="-ctkd $DKV -ctvd $DKV"
  ( flock 9
    env LLAMA_QSA_TRACE=1 GGML_CUDA_DISABLE_GRAPHS=${NOGRAPH:-1} $BIN/llama-server -v -m $S/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf \
      --spec-draft-model ~/models/qwen3.8-flash-next-mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
      -mm $S/mmproj-Signal-3.8-Flash-Next-f16.gguf --mmproj-device ROCm0 -dev ROCm0 --spec-draft-device ROCm0 \
      --metrics --host 127.0.0.1 --port 8099 --threads 8 --load-mode none --lazy-mode on-direct \
      --ctx-size 261888 --parallel 1 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-ngl all \
      -ngl 99 --n-gpu-layers-draft 99 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05 -ub 8192 -b 8192 \
      --presence-penalty 0.0 --repeat-penalty 1.0 --image-min-tokens 1024 -ctk ${MAINKV:-q8_0} -ctv ${MAINKV:-q8_0} $DF \
      --reasoning-preserve --jinja -fa on -fit off > $LOG 2>&1 &
    PID=$!
    for i in $(seq 400); do
      grep -qE "listening on|exiting due to|failed to" $LOG && break
      kill -0 $PID 2>/dev/null || break
      sleep 1
    done
    if grep -q "listening on" $LOG; then
      for R in ${REQS:-S0 S1 L0 L1 X0}; do
        curl -s -m 1800 localhost:8099/v1/chat/completions -H 'Content-Type: application/json' -d @$C2/$R.json \
        | python3 -c 'import json,sys
r=json.load(sys.stdin); t=r.get("timings",{})
print(json.dumps({"req":sys.argv[1],"err":r.get("error"),"prompt_n":t.get("prompt_n"),"predicted_n":t.get("predicted_n"),"draft_n":t.get("draft_n"),"draft_n_accepted":t.get("draft_n_accepted"),"tg_tps":t.get("predicted_per_second")}))' $R >> $RES
        kill -0 $PID 2>/dev/null || { echo "server died" >> $LOG; break; }
      done
    fi
    kill -INT $PID 2>/dev/null
    for i in $(seq 120); do kill -0 $PID 2>/dev/null || break; sleep 1; done
    kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  ) 9> ~/gpu.lock
  $CC expect --log $LOG --select-cmd $BIN/test-qsa-select-dump > $LOG.expect 2>&1; local e=$?
  echo "MTP $TAG expect_rc=$e $(grep -E '^EXPECT' $LOG.expect) masked=$(grep -c 'LLAMA_QSA_TRACE.*producer=masked' $LOG) cpu=$(grep -c 'QSA_CENSUS cpu:' $LOG) guard=[$(grep LLAMA_QSA_GUARD $LOG | cut -d' ' -f2- | tr '\n' ';' | cut -c1-200)] res=$(wc -l < $RES)"
  grep 'LLAMA_QSA_TRACE' $LOG | grep -E 'ctx=mtp' | awk '{print $3,$5,$9,$10}' | sort | uniq -c | sed 's/^/  mtp /' | head -12
}
run ${TAGP:-c}_default default
run ${TAGP:-c}_q8 q8_0
run ${TAGP:-c}_q4 q4_0
echo MTP_DONE
