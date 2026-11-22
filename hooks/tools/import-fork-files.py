#!/usr/bin/env python3
# The fork files that are not built here, kept beside the upstream ones:
# - hooks/python: the fork convert_hf_to_gguf.py, conversion/ and gguf-py/, complete, so that
#   python3 hooks/python/convert_hf_to_gguf.py runs the fork converter (it puts its own gguf-py on sys.path);
# - hooks/fork-files/<path>: the fork version of every other changed file (docs, CI workflows, the UI sources,
#   scripts, repository files), as it is. They are not active: copy one over the upstream file to use it.
# Run from the repository root:
#   hooks/tools/import-fork-files.py [fork ref] [upstream base]
import os
import shutil
import subprocess
import sys

REF  = sys.argv[1] if len(sys.argv) > 1 else 'origin/master'
BASE = sys.argv[2] if len(sys.argv) > 2 else '9048bad8d'


def git_bytes(*a):
    return subprocess.run(['git', *a], capture_output=True, check=True).stdout


def git(*a):
    return git_bytes(*a).decode()


def put(dst, f):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, 'wb').write(git_bytes('show', f'{REF}:{f}'))


# python: complete trees
shutil.rmtree('hooks/python', ignore_errors=True)
for root in ['conversion/', 'gguf-py/']:
    for f in git('ls-tree', '-r', '--name-only', REF, root).split('\n'):
        if f:
            put(os.path.join('hooks/python', f), f)
put('hooks/python/convert_hf_to_gguf.py', 'convert_hf_to_gguf.py')
open('hooks/python/README.md', 'w').write(
    '# python\n\nThe fork `convert_hf_to_gguf.py`, `conversion/` and `gguf-py/`, complete (`hooks/tools/import-fork-files.py`).\n'
    '`python3 hooks/python/convert_hf_to_gguf.py ...` runs the fork converter with the fork gguf package; the upstream\n'
    'converter at the repository root is unchanged.\n')

# everything else that the fork changed and that no other part of the port carries
handled = ('ggml/', 'src/', 'include/', 'common/', 'tools/server/', 'tools/mtmd/', 'tools/llama-bench/', 'tools/quantize/',
           'examples/speculative-prefill/', 'tests/', 'conversion/', 'gguf-py/', 'convert_hf_to_gguf.py', 'hooks/')
shutil.rmtree('hooks/fork-files', ignore_errors=True)
n = 0
for line in git('diff', '--name-status', BASE, REF).split('\n'):
    if not line:
        continue
    status, f = line.split('\t')[0], line.split('\t')[-1]
    if status.startswith('D') or f.startswith(handled):
        continue
    if status.startswith('A') and os.path.isfile(f) and not os.path.islink(f) and open(f, 'rb').read() == git_bytes('show', f'{REF}:{f}'):
        continue  # a new file of the fork, already at its path
    put(os.path.join('hooks/fork-files', f), f)
    n += 1

print(n, 'files in hooks/fork-files')

# the PR template of the fork asks to send general llama.cpp improvements to halo-box/llama.cpp: that repository is
# archived (SYNC.md of the fork), the requirement is not kept
p = 'hooks/fork-files/.github/pull_request_template.md'
t = open(p).read()
rule = ('- This change is Strix Halo specific, or justified by measurements on Strix Halo. General llama.cpp improvements\n  belong in [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp) instead\n')
assert t.count(rule) == 1
open(p, 'w').write(t.replace(rule, ''))
