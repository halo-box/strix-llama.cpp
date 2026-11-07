#!/bin/bash
# M2 (total footprint) and M3 (no hipMalloc in the rep windows), by the C7 shim (PLAN-v3 §16). M2: llama-server with the MTP sidecar, one prompt that fills the context, W0 vs new, f16 and q8_0,
# -c {65536, 131072}, ub 4096. M3: llama-bench -r 12, ub512 pp512 d0 and ub4096 pp4096 d32768, per type.
. "$(dirname "$0")/env.sh"
OUT=$W7/m23; mkdir -p $OUT
SHIM=$OUT/c7shim.so
[ -f $SHIM ] || gcc -O2 -shared -fPIC -o $SHIM $T/scripts/qsa-validation/kfd/c7shim.c -ldl -lpthread
srv() { # tag bin kv c
  local tag=$1 B=$2 kv=$3 c=$4 LOG=$OUT/$1.log
  ( flock 9
    env C7SHIM_OUT=$OUT/$tag.shim C7SHIM_SYNC_MIN_US=1000000 LD_PRELOAD=$SHIM $B/llama-server -m $S/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf --spec-draft-model ~/models/qwen3.8-flash-next-mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
      -dev ROCm0 --spec-draft-device ROCm0 --host 127.0.0.1 --port 8095 --threads 8 --load-mode none --lazy-mode on-direct --ctx-size $c --parallel 1 \
      --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-ngl all -ngl 99 --n-gpu-layers-draft 99 -ub 4096 -b 4096 -ctk $kv -ctv $kv --jinja -fa on -fit off > $LOG 2>&1 &
    PID=$!
    for i in $(seq 400); do grep -qE "listening on|exiting due to|failed to" $LOG && break; kill -0 $PID 2>/dev/null || break; sleep 1; done
    date +%s%N > $OUT/$tag.ready
    python3 - "$c" <<'PY'
import json, sys, urllib.request
c = int(sys.argv[1])
def post(path, body):
    r = urllib.request.Request('http://127.0.0.1:8095' + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=3600))
import os
text = open(os.path.expanduser('~/data/wikitext-2-raw/wiki.test.raw'), errors='replace').read(4000000)
toks = post('/tokenize', {'content': text})['tokens']
r = post('/completion', {'prompt': toks[:c - 256], 'n_predict': 32, 'temperature': 0.0, 'top_k': 1, 'cache_prompt': False, 'ignore_eos': True})
print(r.get('timings', {}).get('prompt_n'), r.get('timings', {}).get('predicted_n'))
PY
    kill -INT $PID; for i in $(seq 60); do kill -0 $PID 2>/dev/null || break; sleep 1; done
    kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  ) 9> ~/gpu.lock
  echo "$(date +%T) M2 $tag: $(grep -E 'sched_reserve:.*compute buffer size' $LOG | sed 's/.*sched_reserve: *//; s/compute buffer size = *//' | tr '\n' ';' | cut -c1-200)"
}
phase_M2() {
  for kv in f16 q8_0; do for c in 65536 131072; do
    srv m2_${kv}_c${c}_w0 ~/dev/wt_lab/w0lab12/build/bin $kv $c
    srv m2_${kv}_c${c}_new $T/build/bin $kv $c
  done; done
  python3 "$(dirname "$0")/m2_an.py" $OUT
}
phase_M3() {
  for kv in f16 q8_0 q4_0 bf16; do for a in new w0; do
    if [ $a = new ] && true; then B=$W7/bench/bin_w7; else B=$W7/bench/bin_w0; fi
    for cell in "512 512 0" "4096 4096 32768"; do
      set -- $cell; tag=m3_${kv}_${a}_u$1_d$3
      ( flock 9
        env C10_TEXT=$WIKI,$HOME/dev/wt_lab/c10/corp/code.txt C7SHIM_OUT=$OUT/$tag.shim LD_PRELOAD=$SHIM $B/llama-bench -m $M -ngl 99 -fa 1 -lzm on-direct -ctk $kv -ctv $kv -b 4096 -ub $1 -p $2 -n 0 -d $3 -r 12 -o json > $OUT/$tag.json 2> $OUT/$tag.err
      ) 9> ~/gpu.lock
      python3 "$(dirname "$0")/m3_an.py" $OUT/$tag.shim $tag
    done
  done; done
}
for ph in "$@"; do echo "PHASE $ph start $(date +%T)"; phase_$ph; echo "PHASE $ph done $(date +%T)"; done
echo M23_DONE
