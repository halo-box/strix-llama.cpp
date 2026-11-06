#!/bin/bash
# W4: the PR-verbatim decode (R2-BOP bands_dec, PR-dec-<T>) on the W4 test-backend-ops G1 dumps (GGML_SA_TEST_DUMP),
# planted defects on; mixed K/V dumps skipped (the PR decode has one type for K and V).
# usage: tbo.sh <test-backend-ops stderr with SA_DUMP lines> <outdir>
LOG=$1; O=$(realpath -m $2); mkdir -p $O; cd ~/dev/wt_lab/bands/op || exit 1
grep "^SA_DUMP" $LOG | while read _ f v; do
  case "$v" in *overflow*) echo "SKIP $f $v (overflow: compared against CPU only)"; continue;; esac
  tk=$(echo $v | sed -E "s/.*type_k=([a-z0-9_]+).*/\1/"); tv=$(echo $v | sed -E "s/.*type_v=([a-z0-9_]+).*/\1/")
  if [ "$tk" != "$tv" ]; then echo "SKIP $f $v (mixed)"; continue; fi
  b=$(basename $f .bdmp)
  flock ~/gpu.lock env BANDS_DEFECTS=1 ./bands_dec TBO:$f SYN 0 PR-dec-$tk $b 32 > $O/$b.out 2>&1
  echo "RUN $b rc=$? $v"
done
echo DONE
