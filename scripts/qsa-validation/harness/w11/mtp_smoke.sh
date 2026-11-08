#!/bin/bash
# W11 smoke of the MTP server (W7 mtp_census.sh flags, smaller context, request S0 only): main -ctk KV and draft -ctkd KV
# (default q4_0), -v. Shows the resolved indexer type and the indexer KV buffer line of both contexts, and runs the census
# expectation (census_check.py expect: sparse route, 0 CPU attention launches).
. ~/dev/wt_rewrite/scripts/qsa-validation/harness/w7/env.sh
KV=${KV:-q4_0}
OUT=${OUT:-~/dev/wt_lab/w11/mtp}; mkdir -p $OUT
C2=~/dev/wt_lab/c2
BIN=${BIN:-$T/build/bin}
LOG=$OUT/server_$KV.log RES=$OUT/res_$KV.jsonl
: > $RES
( flock 9
  env LLAMA_QSA_TRACE=1 GGML_CUDA_DISABLE_GRAPHS=1 $BIN/llama-server -v -m $S/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf \
    --spec-draft-model ~/models/qwen3.8-flash-next-mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
    -dev ROCm0 --spec-draft-device ROCm0 --host 127.0.0.1 --port 8099 --threads 8 --load-mode none --lazy-mode on-direct \
    --ctx-size ${CTX:-32768} --parallel 1 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-ngl all \
    -ngl 99 --n-gpu-layers-draft 99 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05 -ub 4096 -b 4096 \
    -ctk $KV -ctv $KV -ctkd $KV -ctvd $KV --jinja -fa on -fit off > $LOG 2>&1 &
  PID=$!
  for i in $(seq 400); do
    grep -qE "listening on|exiting due to|failed to" $LOG && break
    kill -0 $PID 2>/dev/null || break
    sleep 1
  done
  if grep -q "listening on" $LOG; then
    curl -s -m 1800 localhost:8099/v1/chat/completions -H 'Content-Type: application/json' -d @$C2/S0.json \
      | python3 -c 'import json,sys
r=json.load(sys.stdin); t=r.get("timings",{})
print(json.dumps({"err":r.get("error"),"prompt_n":t.get("prompt_n"),"predicted_n":t.get("predicted_n"),"draft_n":t.get("draft_n"),"draft_n_accepted":t.get("draft_n_accepted")}))' >> $RES
  fi
  kill -INT $PID 2>/dev/null
  for i in $(seq 120); do kill -0 $PID 2>/dev/null || break; sleep 1; done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
) 9> ~/gpu.lock
python3 $T/scripts/qsa-validation/census_check.py expect --log $LOG --select-cmd $BIN/test-qsa-select-dump > $LOG.expect 2>&1; e=$?
echo "MTP $KV expect_rc=$e $(grep -E '^EXPECT' $LOG.expect) cpu=$(grep -c 'QSA_CENSUS cpu:' $LOG) res=$(cat $RES)"
grep -hE "indexer K cache type|creating indexer KV cache|llama_kv_cache: .*(idx|K \()" $LOG | sed 's/^[0-9.]* [IWE] //' | sort | uniq -c
grep -h "QSA ctx=" $LOG | sed 's/^[0-9.]* I //' | cut -c1-160 | sort | uniq -c | head -8
echo MTP_SMOKE_DONE
