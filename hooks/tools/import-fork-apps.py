#!/usr/bin/env python3
# Copies the fork common/, tools/mtmd, tools/server, tools/llama-bench, tools/quantize and
# examples/speculative-prefill into hooks/apps/fork-*, each C/C++ file inside namespace ggml_fork (closed around its
# includes), with the adaptations below, each marked "hooks:". The tools run through the fork_tools switch
# (hooks/apps/fork-tools-dispatch.cpp). Run from the repository root:
#   hooks/tools/import-fork-apps.py [fork ref] [upstream base]
import os
import re
import subprocess
import sys

REF  = sys.argv[1] if len(sys.argv) > 1 else 'origin/master'
BASE = sys.argv[2] if len(sys.argv) > 2 else '9048bad8d'
OPEN = 'namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md'
CLOSE = '} // namespace ggml_fork'
GLOBAL_DECLS = ('struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; '
                'struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; '
                'struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; '
                'struct ggml_backend_event; // hooks: global')

TREES = [
    ('common/',                      'hooks/apps/fork-common'),
    ('tools/mtmd/',                  'hooks/apps/fork-mtmd'),
    ('tools/server/',                'hooks/apps/fork-server'),
    ('tools/llama-bench/',           'hooks/apps/fork-llama-bench'),
    ('tools/quantize/',              'hooks/apps/fork-quantize'),
    ('examples/speculative-prefill/', 'hooks/apps/fork-speculative-prefill'),
]


def git(*a):
    return subprocess.run(['git', *a], capture_output=True, text=True, check=True).stdout


FORK = git('rev-parse', '--short=9', REF).strip()


def namespaced(path, src, state):
    lines = src.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    out = [f'// hooks: {path} of the fork ({FORK}, upstream {BASE}, {state} in the fork), verbatim inside',
           '// namespace ggml_fork. Adapted lines carry "hooks:".']
    pragma = [i for i, l in enumerate(lines[:5]) if l.strip() == '#pragma once']
    start = pragma[0] + 1 if pragma else 0
    out += lines[:start]
    out.append(GLOBAL_DECLS)
    out.append(OPEN)
    depth_extern_c = 0
    for l in lines[start:]:
        if re.match(r'\s*#\s*include\b', l):
            # the libllama headers: the fork ones (hooks/llama/fork-src, on the include path)
            l2 = re.sub(r'#include "(\.\./)*src/', '#include "', l)
            out.append(CLOSE)
            out.append(l2 + (' // hooks: the fork libllama header' if l2 != l else ''))
            out.append(OPEN)
        elif re.match(r'^(struct (ggml|gguf)_\w+|typedef struct (ggml|gguf)_\w+\s*\*?\s*\w+|extern char\s*\*\*\s*environ)\s*;\s*(//.*)?$', l):
            out.append(CLOSE)
            out.append(l + ' // hooks: global')
            out.append(OPEN)
        elif re.match(r'^\s*extern "C" \{\s*$', l):
            out.append('// hooks: C++ linkage in namespace ggml_fork (extern "C" removed)')
            depth_extern_c += 1
        else:
            out.append(l)
    out.append(CLOSE)
    text = '\n'.join(out) + '\n'
    # explicit global calls of the functions of this file, which are in the namespace here
    defined = set(re.findall(r'^(?:static\s+)?[A-Za-z_][\w<>,\*& ]*?[ \*&](\w+)\s*\([^;]*$', '\n'.join(lines), re.M))
    text = re.sub(r'(?<![\w:>])::(\w+)\(', lambda m: (f'ggml_fork::{m.group(1)}( /* hooks: namespaced */ '
                                                       if m.group(1) in defined else m.group(0)), text)
    if depth_extern_c:
        text = re.sub(r'#ifdef __cplusplus\n}( *// *extern "C")?\n#endif', '#ifdef __cplusplus\n// hooks: end of the removed extern "C"\n#endif', text)
    return text.replace(OPEN + '\n' + CLOSE + '\n', '')


for src_dir, dst_dir in TREES:
    files = [f for f in git('ls-tree', '-r', '--name-only', REF, src_dir).split('\n') if f]
    upstream = set(f for f in git('ls-tree', '-r', '--name-only', BASE, src_dir).split('\n') if f)
    for f in files:
        rel = f[len(src_dir):]
        if rel.startswith(('webui/', 'public/')):
            continue  # web assets: the upstream ones
        out = os.path.join(dst_dir, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if not f.endswith(('.cpp', '.h', '.hpp')):
            data = subprocess.run(['git', 'show', f'{REF}:{f}'], capture_output=True, check=True).stdout
            open(out, 'wb').write(data)  # as it is (CMake files, scripts, assets)
            continue
        src = git('show', f'{REF}:{f}')
        if f == 'common/build-info.h':
            pass  # the build info of upstream (build-info.cpp, generated): global
        elif f.endswith(('.cpp', '.h', '.hpp')):
            src = namespaced(f, src, 'new' if f not in upstream else ('changed' if src != git('show', f'{BASE}:{f}') else 'unchanged'))
        open(out, 'w').write(src)
print('copied')


def adapt(path, old, new, count=1):
    s = open(path).read()
    assert s.count(old) == count, (path, old, s.count(old))
    open(path, 'w').write(s.replace(old, new))


# the entry of the example: a global main
p = 'hooks/apps/fork-speculative-prefill/speculative-prefill.cpp'
s = open(p).read().rstrip('\n')
s += '\n\n// hooks: the program entry of the namespaced fork example\nint main(int argc, char ** argv) {\n    return ggml_fork::main(argc, argv);\n}\n'
open(p, 'w').write(s)
print('adapted')
