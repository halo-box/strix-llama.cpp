#!/bin/bash
# W11 session-file check (PLAN-v3 W11 done-when, P2-26): a qwen4exp session saved by the W0 arm (PR219 + lab patches, W0-lab12)
# with llama-completion --prompt-cache, then loaded by this build with the same arguments.
#   -ctk q4_0: W0's indexer was q4_0, this build's auto indexer is q8_0 -> rejected through the existing per-layer key type
#              check, with the wrapped hybrid-idx message; the same file loads with --cache-type-k-idx q4_0
#   -ctk f16, q8_0: indexer type unchanged -> loads (the prompt is matched from the session, no re-evaluation)
# Session files are small (one 64-token prompt) and deleted at the end.
. ~/dev/wt_rewrite/scripts/qsa-validation/harness/w7/env.sh
W0=${W0:-~/dev/wt_lab/w0lab12/build/bin}
NEW=${NEW:-$T/build/bin}
OUT=${OUT:-~/dev/wt_lab/w11/session}; mkdir -p $OUT
P="The quick brown fox jumps over the lazy dog. In the beginning the Universe was created. This has made a lot of people very angry and been widely regarded as a bad move. Far out in the uncharted backwaters"
ARGS="-m $M -c 4096 -n 4 --temp 0 --seed 1 -no-cnv --prompt-cache-all $COMMON"
run() { # tag bin extra...
  local tag=$1 bin=$2; shift 2
  flock ~/gpu.lock env LD_LIBRARY_PATH=$bin timeout 1200 $bin/llama-completion $ARGS -p "$P" "$@" > $OUT/$tag.log 2>&1 < /dev/null
  echo "$tag rc=$?"
}
for kv in ${TYPES:-q4_0 f16 q8_0}; do
  f=$OUT/w0_$kv.session; rm -f $f
  run save_w0_$kv $W0 -ctk $kv -ctv $kv --prompt-cache $f
  ls -la $f
  cp $f $f.orig
  run load_w11_$kv $NEW -ctk $kv -ctv $kv --prompt-cache $f --prompt-cache-ro
  echo "  load_w11_$kv: $(grep -hE 'state_read_data|error loading|failed to load session|loaded a session|session file matches|indexer K cache type' $OUT/load_w11_$kv.log | tr -s ' ' | tr '\n' '|')"
  if [ $kv = q4_0 ]; then
    cp $f.orig $f
    run load_w11_${kv}_override $NEW -ctk $kv -ctv $kv --cache-type-k-idx q4_0 --prompt-cache $f --prompt-cache-ro
    echo "  load_w11_${kv}_override: $(grep -hE 'state_read_data|error loading|failed to load session|loaded a session|session file matches|indexer K cache type' $OUT/load_w11_${kv}_override.log | tr -s ' ' | tr '\n' '|')"
  fi
  rm -f $f $f.orig
done
echo SESSION_DONE
