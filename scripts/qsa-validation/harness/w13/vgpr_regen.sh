#!/bin/bash
# scripts/hip/README.md "Regenerate the coverage map", verbatim python, on the pinned-image build of the source at HEAD
set -e
V=~/dev/wt_lab/w13/vgpr
cd ~/dev/wt_rewrite
test "$(git rev-parse HEAD)" = "$(cat $V/src_rev.txt)"
cp scripts/hip/gfx1151-vgpr-baseline.json $V/baseline_before.json
python3 - $V/out/build/ggml/src/ggml-hip <<'PYTHON'
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

spec = importlib.util.spec_from_file_location('vgpr_check', 'scripts/hip/gcn-cdna-vgpr-check.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
path = Path('scripts/hip/gfx1151-vgpr-baseline.json')
baseline = json.loads(path.read_text())
kernels, units = checker.read_assembly(Path(sys.argv[1]), baseline['architecture'])
assert kernels and units and not any(k['spill_stores'] for k in kernels.values())
assert all(k['vgprs'] <= 256 for k in kernels.values())
baseline['translation_units'] = units
baseline['minimum_kernel_count'] = len(kernels)
baseline['source_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
path.write_text(json.dumps(baseline, indent=2) + '\n')
PYTHON
python3 - $V/baseline_before.json scripts/hip/gfx1151-vgpr-baseline.json <<'PY'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
ta, tb = a['translation_units'], b['translation_units']
print('minimum_kernel_count', a['minimum_kernel_count'], '->', b['minimum_kernel_count'])
print('TUs', len(ta), '->', len(tb), ' copies', sum(ta.values()), '->', sum(tb.values()))
for k in sorted(set(ta) | set(tb)):
    if ta.get(k) != tb.get(k): print('  ', k, ta.get(k), '->', tb.get(k))
for k in a:
    if k not in ('translation_units', 'minimum_kernel_count', 'source_revision') and a[k] != b.get(k): print('CHANGED', k)
print('source_revision', a['source_revision'], '->', b['source_revision'])
PY
python3 scripts/hip/gcn-cdna-vgpr-check.py $V/out/build/ggml/src/ggml-hip --baseline scripts/hip/gfx1151-vgpr-baseline.json --compiler-version "$(cat $V/out/compiler_version.txt)" > $V/check_new_baseline.log 2>&1
echo new_baseline_rc=$?
tail -4 $V/check_new_baseline.log
python3 scripts/hip/test_vgpr_check.py 2>&1 | tail -2
