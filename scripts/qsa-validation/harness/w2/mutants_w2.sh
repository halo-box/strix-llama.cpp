#!/bin/bash
# W2 mutation check of the plan tests on ROCm0: temporary edits of sparse-attn-plan.cu, rebuilt, tested, then restored.
cd ~/dev/wt_rewrite; L=~/dev/wt_rewrite_logs/W2; F=ggml/src/ggml-cuda/sparse-attn-plan.cu
cp $F $L/plan_orig.cu; md5sum $F
mut() { local tag=$1 from=$2 to=$3
  cp $L/plan_orig.cu $F
  python3 - "$F" "$from" "$to" <<'PY'
import sys; p, a, b = sys.argv[1:]; s = open(p).read(); assert s.count(a) == 1, a; open(p, 'w').write(s.replace(a, b))
PY
  flock ~/gpu.lock cmake --build build -j 16 --target ggml-hip > $L/mutant_${tag}_build.log 2>&1 || { echo "BUILD FAIL $tag"; return; }
  flock ~/gpu.lock build/bin/test-backend-ops test -o SPARSE_ATTN_PLAN -b ROCm0 --fail-unsupported > $L/mutant_${tag}.log 2>&1
  echo "MUTANT $tag rc=$? $(grep -h 'tests passed' $L/mutant_${tag}.log | tr -s ' ') fail=$(grep -c 'FAIL' $L/mutant_${tag}.log)"
}
# M1: padding quad = 0 instead of the group's last real quad
mut pad0 "sa_plan_fill<NT>(oq, om, total, a.cap, (uint32_t) (kmax >> 2));" "sa_plan_fill<NT>(oq, om, total, a.cap, 0u);"
# M2: no dedup after the sort (every sorted entry kept)
mut nodedup "u += j == 0 || cmp[j] != cmp[j - 1];" "u += 1;"
# M3: n_bad not counted
mut nobad "atomicAdd(reinterpret_cast<unsigned int *>(a.dst + offsetof(ggml_sparse_attn_plan_hdr, n_bad)), bad_s);" "(void) bad_s;"
cp $L/plan_orig.cu $F; md5sum $F
flock ~/gpu.lock cmake --build build -j 16 > $L/mutant_restore_build.log 2>&1; echo "RESTORE build rc=$?"
flock ~/gpu.lock build/bin/test-backend-ops test -o SPARSE_ATTN_PLAN -b ROCm0 --fail-unsupported > $L/mutant_restored.log 2>&1
echo "RESTORED rc=$? $(grep -h 'tests passed' $L/mutant_restored.log | tr -s ' ') fail=$(grep -c 'FAIL' $L/mutant_restored.log)"
