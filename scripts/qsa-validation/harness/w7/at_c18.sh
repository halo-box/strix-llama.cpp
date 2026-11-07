#!/bin/bash
# AT-C18 (PLAN-v3 §16.5), -np 1 (W7): -fit on, f16 and q8_0, with and without -nkvo, -c 131072, ub 512 and 4096: the probe's per-backend estimate against the real
# context's sched_reserve (campaigns/r2res/parse.py labels), main and MTP; the guard on the probe context; under -nkvo the probe shows the pins.
. "$(dirname "$0")/env.sh"
OUT=$W7/c18; mkdir -p $OUT
BIN=${BIN:-$T/build/bin}
srv() { # tag kv extra
  local tag=$1 kv=$2 extra=$3 LOG=$OUT/srv_$1.log
  ( flock 9
    $BIN/llama-server -lv 5 -m $S/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf --spec-draft-model ~/models/qwen3.8-flash-next-mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
      -dev ROCm0 --spec-draft-device ROCm0 --metrics --host 127.0.0.1 --port 8098 --threads 8 --load-mode none --lazy-mode on-direct \
      --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-ngl all -ngl 99 --n-gpu-layers-draft 99 --parallel 1 \
      -ctk $kv -ctv $kv --jinja -fa on -fit on $extra > $LOG 2>&1 &
    PID=$!
    for i in $(seq 600); do grep -qE "listening on|exiting due to|failed to load" $LOG && break; kill -0 $PID 2>/dev/null || break; sleep 1; done
    kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  ) 9> ~/gpu.lock
  echo "C18 $tag $(grep -qE 'listening on' $LOG && echo up || echo NOT_UP)"
}
for kv in f16 q8_0; do for u in 512 4096; do
  srv ${kv}_u${u} $kv "-c 131072 -b $u -ub $u"
  srv ${kv}_u${u}_nkvo $kv "-c 131072 -b $u -ub $u -nkvo"
done; done
python3 $T/scripts/qsa-validation/campaigns/r2res/parse.py $OUT/srv_*.log > $OUT/parsed.json
python3 "$(dirname "$0")/at_c18_an.py" $OUT
echo C18_DONE
