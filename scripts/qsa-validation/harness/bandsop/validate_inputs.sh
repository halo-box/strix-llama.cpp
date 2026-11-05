#!/bin/bash
# R2-M-LAB-BANDS-OP step 2: rows FNV and value distribution of every GQA-12 test-backend-ops dump vs the UNI generator (seed 1)
cd ~/dev/wt_lab/bands/op
for r in r1 r2; do for f in dump_$r/*.bdmp; do
  L=$(BANDS_ROWS_ONLY=1 ./bands_dec TBO:$f SYN 0 PR-dec-f16 x 1 | grep -E "FIXTURE|ROWS_HASH|VALSTAT")
  echo "$L" | sed "s/^/TBO $r /"
  set -- $(echo "$L" | grep FIXTURE | sed 's/.* q \([0-9]*\) keys \([0-9]*\) sel \([0-9]*\) interleaved [0-9] blocks \([0-9]*\) .*/\1 \2 \3 \4/')
  BANDS_ROWS_ONLY=1 ./bands_dec UNI:1.0:$2:1:$1:$3:$4 SYN 0 PR-dec-f16 x 1 | grep -E "ROWS_HASH|VALSTAT" | sed "s/^/UNI $r $(basename $f) /"
done; done
