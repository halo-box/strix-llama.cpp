#!/bin/bash
# The W7 measurement arm: a detached worktree of the branch at <commit> with the QPIN / defect patches of scripts/qsa-validation/patches (01, 02, 03; the
# hook name in qwen4exp.cpp is in the branch now, so that hunk is dropped; patch 04's masked-route lab switches do not apply). lab_tree.sh <commit>
set -e
C=${1:-HEAD}
LAB=~/dev/wt_lab/w7/lab
P=~/dev/wt_rewrite/scripts/qsa-validation/patches
cd ~/dev/wt_rewrite
if [ -d $LAB ]; then
  git -C $LAB reset -q --hard
  git -C $LAB checkout -q --detach $(git rev-parse $C)
else
  git worktree add --detach $LAB $C
fi
cd $LAB
python3 - "$P" <<'PY'
import re, sys, subprocess
P = sys.argv[1]
for name in ['01-qpin', '02-r1e-qsa-chunk', '03-r1q8-defect-hook']:
    s = open(f'{P}/{name}.patch').read()
    parts = re.split(r'(?m)^(?=diff --git )', s)
    keep = ''.join(x for x in parts if x and 'b/src/models/qwen4exp.cpp' not in x.split('\n')[0])
    r = subprocess.run(['patch', '-p1', '-F3', '--no-backup-if-mismatch'], input=keep, text=True, capture_output=True)
    print(name, r.returncode, r.stdout[-400:], r.stderr[-200:])
    if r.returncode: sys.exit(1)
# patch's fuzz puts qpin_report() on the wrong synchronize(): it belongs in the destructor
p = 'src/llama-context.cpp'; s = open(p).read()
s = s.replace('    synchronize();\n    qpin_report();\n\n    const int64_t t_start_us', '    synchronize();\n\n    const int64_t t_start_us')
s = s.replace('    qsa_trace_print();\n', '    qsa_trace_print();\n    qpin_report();\n')
assert s.count('qpin_report();') == 1
open(p, 'w').write(s)
PY
git status --short | head -10
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang -DCMAKE_PREFIX_PATH=/opt/rocm -DLLAMA_CURL=OFF > ~/dev/wt_lab/w7/lab_cfg.log 2>&1
flock ~/gpu.lock cmake --build build -j 16 --target llama-perplexity > ~/dev/wt_lab/w7/lab_build.log 2>&1
echo LAB_BUILD_rc=$? $(git -C $LAB rev-parse --short HEAD)
