#!/usr/bin/env python3
# Copies ggml/src/ggml-cuda of the fork into hooks/ggml/cuda: each file inside namespace ggml_fork (closed around its
# includes), with the adaptations below, each marked "hooks:". Run from the repository root:
#   hooks/tools/import-cuda.py [fork ref] [upstream base]
# The hooks files of the directory (hooks.cuh, entry.inc, README.md) are not touched.
import os
import re
import subprocess
import sys

REF  = sys.argv[1] if len(sys.argv) > 1 else 'origin/master'
BASE = sys.argv[2] if len(sys.argv) > 2 else '9048bad8d'
DST  = 'hooks/ggml/cuda'


def git(*a):
    return subprocess.run(['git', *a], capture_output=True, text=True, check=True).stdout


FORK = git('rev-parse', '--short=9', REF).strip()


files = [f for f in git('ls-tree', '-r', '--name-only', REF, 'ggml/src/ggml-cuda').split('\n') if f]
upstream = set(f for f in git('ls-tree', '-r', '--name-only', BASE, 'ggml/src/ggml-cuda').split('\n') if f)

# include path adaptations: the fork tree is ggml/src/ggml-cuda, the copy is hooks/ggml/cuda
PATHS = [
    ('#include "../../rocmfp4/', '#include "../types/rocmfp4/'),
    ('#include "../../exl3/exl3.h"', '#include "../exl3/exl3.h"'),
    ('#include "ggml-common.h"', '#include "ggml-common-ext.h"'),
    ('#include "ggml-cuda/', '#include "'),
]

OPEN = 'namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/cuda/README.md'
CLOSE = '} // namespace ggml_fork'

n_files = 0
for f in files:
    rel = f[len('ggml/src/ggml-cuda/'):]
    src = git('show', f'{REF}:{f}')
    out_path = os.path.join(DST, rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not (rel.endswith('.cu') or rel.endswith('.cuh') or rel.endswith('.h')):
        open(out_path, 'w').write(src)  # CMakeLists.txt, scripts: kept as data
        continue
    lines = src.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    out = []
    state = 'new' if f not in upstream else 'changed'
    out.append(f'// hooks: ggml/src/ggml-cuda/{rel} of the fork ({FORK}, upstream {BASE}, {state} in the fork), verbatim inside')
    out.append('// namespace ggml_fork. Adapted lines carry "hooks:".')
    pragma = [i for i, l in enumerate(lines[:5]) if l.strip() == '#pragma once']
    start = pragma[0] + 1 if pragma else 0
    out += lines[:start]
    out.append(OPEN)
    for l in lines[start:]:
        if re.match(r'\s*#\s*include\b', l):
            for a, b in PATHS:
                if l.startswith(a):
                    l = l.replace(a, b) + ' // hooks: path'
            out.append(CLOSE)
            out.append(l)
            out.append(OPEN)
        else:
            out.append(l)
    out.append(CLOSE)
    # drop empty namespace pairs
    text = '\n'.join(out) + '\n'
    text = text.replace(OPEN + '\n' + CLOSE + '\n', '')
    open(out_path, 'w').write(text)
    n_files += 1
print(n_files, 'files')

# ---------------------------------------------------------------- adaptations, each marked "hooks:"
def adapt(rel, old, new, count=1, note='adapted'):
    p = os.path.join(DST, rel)
    s = open(p).read()
    assert s.count(old) == count, (rel, old, s.count(old))
    open(p, 'w').write(s.replace(old, new))


# the hooks header of the fork code, after the fork common.cuh
p = os.path.join(DST, 'common.cuh')
s = open(p).read()
s = s.rstrip('\n') + '\n#include "hooks.cuh" // hooks: the hooks side of the fork code\n'
open(p, 'w').write(s)

# ADL finds both the global declaration of ggml-cuda.h and the namespaced definition: the fork ggml-cuda.cu defines
# them under other names
api = re.findall(r'GGML_BACKEND_API [^(;]*?\b(ggml_\w+)\s*\(', open('ggml/include/ggml-cuda.h').read())
ren = '\n'.join(f'#define {a} ggml_fork_{a.removeprefix("ggml_")} // hooks: the fork definition, not the upstream API' for a in api)
hdr = re.sub(r'//[^\n]*', '', open('ggml/include/ggml-cuda.h').read())
protos = [' '.join(m.group(1).split()) + ';' for m in re.finditer(r'GGML_BACKEND_API\s+([^;]*?\))\s*;', hdr)]
decl = 'namespace ggml_fork { // hooks: the declarations of ggml-cuda.h, for the renamed fork definitions\n' + '\n'.join(protos) + '\n}'
adapt('ggml-cuda.cu', '#include "ggml-cuda.h"\n', '#include "ggml-cuda.h"\n' + ren + '\n' + decl + '\n')

# EXL3: GGML_OP_CUSTOM nodes here; the fork switches run on ggml_fork_op() (hooks.cuh)
for rel in ['ggml-cuda.cu']:
    p = os.path.join(DST, rel)
    ls = open(p).read().split('\n')
    done = set()
    for i, l in enumerate(ls):
        m = re.match(r'(\s*)case GGML_OP_EXL3_', l)
        if not m:
            continue
        ind = len(m.group(1)) - 4
        for j in range(i - 1, 0, -1):
            mm = re.match(r'^' + ' ' * ind + r'switch \(([\w>\-\.]+)->op\) \{', ls[j])
            if mm:
                if j not in done:
                    ls[j] = ls[j].replace(f'switch ({mm.group(1)}->op)', f'switch (ggml_fork_op({mm.group(1)}))') + ' // hooks: EXL3 ops are custom nodes'
                    done.add(j)
                break
    assert len(done) == 3, done
    open(p, 'w').write('\n'.join(ls))
adapt('ggml-cuda.cu', 'cgraph->nodes[i + 1]->op == GGML_OP_EXL3_MUL_MAT_ID &&',
      'ggml_fork_op(cgraph->nodes[i + 1]) == GGML_OP_EXL3_MUL_MAT_ID && // hooks: EXL3 ops are custom nodes\n       ')
adapt('ggml-cuda.cu', 'ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_GLU, GGML_OP_EXL3_MUL_MAT_ID }, { i + 1 })',
      'ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_GLU, GGML_OP_CUSTOM }, { i + 1 }) /* hooks: EXL3 ops are custom nodes */')
if os.path.exists(os.path.join(DST, 'exl3.cu')):
    adapt('exl3.cu', 'ggml_get_op_params_i32(op, 0)', 'ggml_hook_exl3_param(op) /* hooks: slot 6 */', count=2)
    adapt('exl3.cu', 'ggml_get_op_params_i32(dst, 0)', 'ggml_hook_exl3_param(dst) /* hooks: slot 6 */', count=3)
print('adapted')

p = os.path.join(DST, 'ggml-cuda.cu')
s = open(p).read().rstrip('\n') + '\n#include "entry.inc" // hooks: the entry points of the bridge\n'
open(p, 'w').write(s)

# the fork ggml-impl.h for the fork code (ggml_can_fuse_subgraph takes 32 nodes there): found before the upstream one
# because it is next to the fork files. Its inline functions are static and its types the upstream ones, so the two
# headers can coexist in the program.
open(os.path.join(DST, 'ggml-impl.h'), 'w').write(
    f'// hooks: ggml/src/ggml-impl.h of the fork ({FORK}, upstream {BASE}), for the fork code only, verbatim\n' +
    git('show', f'{REF}:ggml/src/ggml-impl.h'))
print('ggml-impl.h')
