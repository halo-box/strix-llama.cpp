#!/bin/bash
# W3: the PR-verbatim G4 kernel (R2-BOP bands_attn, qsa3_attn_kernel V0 / VD) on the W3 test-backend-ops dumps
# (GGML_SA_TEST_DUMP): NMSE vs O_CON per dump and bitwise comparison with the new kernel's ROCm0 output in the dump.
# usage: prband_run.sh <test-backend-ops stderr with SA_DUMP lines> <outdir>
LOG=$1; O=$(realpath -m $2); mkdir -p $O; cd ~/dev/wt_lab/bands/op || exit 1
grep "^SA_DUMP" $LOG | while read _ f v; do
  case "$v" in *gated=*|*overflow*) echo "SKIP $f $v"; continue;; esac
  tk=$(echo $v | sed -E "s/.*type_k=([a-z0-9_]+).*/\1/"); tv=$(echo $v | sed -E "s/.*type_v=([a-z0-9_]+).*/\1/")
  if [ "$tk" != "$tv" ]; then echo "SKIP $f $v (mixed: no PR-verbatim G4 instance in the harness)"; continue; fi
  b=$(basename $f .bdmp)
  flock ~/gpu.lock ./bands_attn TBO:$f SYN all 1 PR-V0-$tk,PR-VD-$tk $b 32 > $O/$b.out 2>&1
  echo "RUN $b rc=$? $v"
done
