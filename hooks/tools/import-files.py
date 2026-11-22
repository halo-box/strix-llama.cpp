#!/usr/bin/env python3
# The fork files that are not built here, kept beside the upstream ones:
# - hooks/python: the fork convert_hf_to_gguf.py, conversion/ and gguf-py/, complete, so that
#   python3 hooks/python/convert_hf_to_gguf.py runs the fork converter (it puts its own gguf-py on sys.path);
# - hooks/files/<path>: the fork version of every other changed file (docs, CI workflows, the UI sources,
#   scripts, repository files), as it is. They are not active: copy one over the upstream file to use it.
# - the repository files (README, contributing and agent guides, docs, CI workflows and templates, Dockerfiles): the fork version at
#   their path (ACTIVE_FORK_PATHS), since they describe and test this repository. No build input.
# Run from the repository root:
#   hooks/tools/import-files.py [fork ref] [upstream base]
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
    if os.path.dirname(dst):
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
    '# python\n\nThe fork `convert_hf_to_gguf.py`, `conversion/` and `gguf-py/`, complete (`hooks/tools/import-files.py`).\n'
    '`python3 hooks/python/convert_hf_to_gguf.py ...` runs the fork converter with the fork gguf package; the upstream\n'
    'converter at the repository root is unchanged.\n')

# everything else that the fork changed and that no other part of the port carries
handled = ('ggml/', 'src/', 'include/', 'common/', 'tools/server/', 'tools/mtmd/', 'tools/llama-bench/', 'tools/quantize/',
           'examples/speculative-prefill/', 'tests/', 'conversion/', 'gguf-py/', 'convert_hf_to_gguf.py', 'hooks/')
# the fork version at its path, whole directories included (the fork deletes and renames files there); also listed in
# hooks/tools/check-hooks.sh
ACTIVE_FORK_PATHS = ('README.md', 'AGENTS.md', 'CONTRIBUTING.md', '.editorconfig', '.github/', '.agents/', '.claude/', 'docs/',
                     '.devops/',
                     # the READMEs of the programs, and the HIP register check that the fork CI runs
                     'scripts/hip/gcn-cdna-vgpr-check.py', 'gguf-py/README.md', 'tools/cli/README.md', 'tools/completion/README.md', 'tools/llama-bench/README.md', 'tools/server/README.md', 'ui/apps/chat/README.md',
                     'tests/.gitignore')  # keeps the fork test corpus (tests/corpus, tests/data) tracked


def active(f):
    return any(f == a or (a.endswith('/') and f.startswith(a)) for a in ACTIVE_FORK_PATHS)


for a in ACTIVE_FORK_PATHS:
    if a.endswith('/'):
        shutil.rmtree(a, ignore_errors=True)
# git archive keeps the modes and the symlinks (the agent skills)
archive = subprocess.run(['git', 'archive', REF, '--', *[a.rstrip('/') for a in ACTIVE_FORK_PATHS]],
                         capture_output=True, check=True).stdout
subprocess.run(['tar', '-x', '-f', '-'], input=archive, check=True)
shutil.rmtree('hooks/files', ignore_errors=True)
n = 0
for line in git('diff', '--name-status', BASE, REF).split('\n'):
    if not line:
        continue
    status, f = line.split('\t')[0], line.split('\t')[-1]
    if status.startswith('D') or f.startswith(handled) or active(f):
        continue
    if status.startswith('A') and os.path.isfile(f) and not os.path.islink(f) and open(f, 'rb').read() == git_bytes('show', f'{REF}:{f}'):
        continue  # a new file of the fork, already at its path
    put(os.path.join('hooks/files', f), f)
    n += 1

print(n, 'files in hooks/files')

# the PR template of the fork asks to send general llama.cpp improvements to halo-box/llama.cpp: that repository is
# archived (SYNC.md of the fork), the requirement is not kept
p = '.github/pull_request_template.md'
t = open(p).read()
rule = ('- This change is Strix Halo specific, or justified by measurements on Strix Halo. General llama.cpp improvements\n  belong in [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp) instead\n')
assert t.count(rule) == 1
open(p, 'w').write(t.replace(rule, ''))
