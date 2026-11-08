#!/bin/bash
# W10: the ROT = false instances of the G4 / G1 attends against the W9 head's kernels, instruction by instruction: each TU is
# compiled for gfx1151 to device assembly with the build's own command, once from the working tree and once from
# `git archive HEAD` (the W9 head), and every W9 kernel is compared with its ROT = false twin (mangled name + the bool).
L=~/dev/wt_rewrite_logs/W10/isa; rm -rf $L; mkdir -p $L/w9src
git -C ~/dev/wt_rewrite archive HEAD ggml | tar -x -C $L/w9src
for F in ${FILES:-sparse-attn-attend-g4-f16 sparse-attn-attend-g4-q8_0 sparse-attn-attend-g4-bf16 sparse-attn-attend-g4-q4_0 sparse-attn-attend-g1 sparse-attn-generic}; do
  CMD=$(python3 -c "
import json
for c in json.load(open('/home/gaetan/dev/wt_rewrite/build/compile_commands.json')):
    if c['file'].endswith('ggml-cuda/$F.cu'): print(c['command'])
")
  cd ~/dev/wt_rewrite/build/ggml/src/ggml-hip
  NEW=${CMD/-o CMakeFiles\/ggml-hip.dir\/__\/ggml-cuda\/$F.cu.o/-o $L/${F}_w10.s --cuda-device-only -S}
  OLD=${NEW//\/home\/gaetan\/dev\/wt_rewrite\/ggml/$L/w9src/ggml}
  OLD=${OLD/${F}_w10.s/${F}_w9.s}
  eval "$NEW" > $L/${F}_w10.log 2>&1 & eval "$OLD" > $L/${F}_w9.log 2>&1 & wait
  python3 - $L/${F}_w9.s $L/${F}_w10.s <<'PY'
import re, sys
def funcs(p):
    s = open(p).read(); out = {}
    for m in re.finditer(r"^(_Z\w+):[^\n]*\n(.*?)^\s*s_endpgm", s, re.M | re.S):
        body = [l.split(';')[0].strip() for l in m.group(2).splitlines()]
        # branch labels are numbered per TU (the ROT = true twins shift them): compared as LBB<n>_<k> -> L_<k>
        out[m.group(1)] = [re.sub(r'\.LBB\d+_', '.L_', re.sub(r'_Z\w+', 'SYM', l)) for l in body if l and not l.startswith('.') and not re.match(r'^\.?L\w*:', l)]
    return out
a, b = funcs(sys.argv[1]), funcs(sys.argv[2])
same = diff = missing = 0
for k, ins in a.items():
    if 'sa_attend' not in k: continue
    # the twin: the template argument list gains a trailing bool false (Lb0E) before the closing E
    twin = re.sub(r'(EEv)', r'ELb0EEv', k, count=1)
    if twin not in b: missing += 1; print('  no twin for', k); continue
    if b[twin] == ins: same += 1
    else: diff += 1; print('  DIFF', k, len(ins), len(b[twin]))
print(f'{sys.argv[2].split("/")[-1][:-7]}: W9 attend kernels {same + diff + missing}, identical ROT=false twins {same}, different {diff}, missing {missing}')
PY
done
