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

# AGENTS.md: the rules of the hooks layout, and without the halo-box/llama.cpp rule (archived, as in the PR template)
AGENTS_EDITS = [('that file governs\n[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and\n[halo-box/llama.cpp](https://github.com/halo-box/llama.cpp), not this fork. Follow this file instead when working in\n`halo-box/strix-llama.cpp`.', 'that file governs\n[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), not this fork. Follow this file instead when working in\n`halo-box/strix-llama.cpp`.'), ('- Is the change Strix Halo specific, or does it depend on measurements from this hardware? -> it belongs here.\n- Is it a general llama.cpp improvement? -> it belongs in\n  [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp), which is where upstream submissions are staged. Say so\n  and stop; do not open it here just because here is easier.\n', '- Is the change Strix Halo specific, or does it depend on measurements from this hardware? -> it belongs here.\n'), ('Do not reformat untouched code.', 'Do not reformat untouched code. Keep fork code in `hooks/` (see\n   [Where fork code goes](#where-fork-code-goes)): an upstream merge then conflicts only at the `HOOK` lines.'), ('- Open PRs against `ggml-org/llama.cpp` or `halo-box/llama.cpp` from this tree', '- Open PRs against `ggml-org/llama.cpp` from this tree'), ('- [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp) - where general, upstream-bound changes go instead\n', ''), ('---\n\n## Guidelines for AI Coding Agents', '---\n\n## Where fork code goes\n\nThis tree keeps the files of upstream llama.cpp as upstream has them. The fork code lives in `hooks/` and reaches\nupstream through hook points. Read [hooks/README.md](hooks/README.md) before you change code; the reasons are in\n[hooks/adr/0001-extension-architecture.md](hooks/adr/0001-extension-architecture.md).\n\n- Do not change upstream files. The only change allowed in them is a `HOOK(name)` line, listed in\n  [hooks/HOOKS.txt](hooks/HOOKS.txt) with its number of lines. The few exceptions are listed, with their reason, in\n  `hooks/tools/check-hooks.sh`.\n- Put new fork code in `hooks/`: a hook kernel (`hooks/ggml/kernels/<backend>/`), a type (`hooks/ggml/types/`), or a\n  change to the fork code (`hooks/ggml/cuda`, `hooks/ggml/vulkan`, `hooks/llama/lib`, `hooks/apps/*`).\n- Upstream behavior is the default. A fork version that changes what upstream computes is a variant or a switch, off\n  by default (`GGML_HOOKS=+name`), until a measurement decides. Switches: `cuda_node` and `vk_node` (on), `cuda_graph`,\n  `vk_graph`, `sched`, `llama` and `tools` (off).\n- The copies of fork files in `hooks/` are generated by `hooks/tools/import-*.py`. Do not edit a copy by hand: change\n  the import tool, and mark each adapted line with `hooks:`.\n- The repository files (this file, README, CONTRIBUTING, `docs/`, `.github/`, `.devops/`, ...) are the fork versions at\n  their path (`ACTIVE_FORK_PATHS` in `hooks/tools/import-files.py`, which also applies the changes made to them here).\n- Record what happens to every fork change of an upstream line in [hooks/PORT.txt](hooks/PORT.txt): `variant`,\n  `adapted`, `restored` or `dropped`, with the reason.\n- Fork tests are `test-<name>`, or `test-<name>-fork` when upstream has a test of that name. ctest runs them with the\n  switches, with the names and labels of the fork.\n- `cmake -B build` builds with the hooks. `-DLLAMA_HOOKS=OFF` builds pure upstream from the same tree: the baseline of\n  the correctness flow below.\n- Before you push, run `hooks/tools/check-port.py`, `hooks/tools/check-hooks.sh` and `hooks/tools/check-wraps.sh`, and\n  `hooks/tools/check-ctest.py` against a fork build when you change tests. Do not push with a failing check.\n\n---\n\n## Guidelines for AI Coding Agents')]
p = 'AGENTS.md'
t = open(p).read()
for old, new in AGENTS_EDITS:
    assert t.count(old) == 1, old[:60]
    t = t.replace(old, new)
open(p, 'w').write(t)

# CONTRIBUTING.md: without the halo-box/llama.cpp rule (archived), and where the fork code goes
CONTRIBUTING_EDITS = [('ggml-org/llama.cpp            upstream\n  |\n  +-- halo-box/llama.cpp      staging fork for changes intended to go upstream\n        |\n        +-- halo-box/strix-llama.cpp   <-- this repo', 'ggml-org/llama.cpp            upstream\n  |\n  +-- halo-box/strix-llama.cpp   <-- this repo'), ("- **Strix Halo specific, or justified by measurements on Strix Halo** -> open the PR here.\n- **A general llama.cpp improvement** -> open it against\n  [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp) instead, and follow that repo's (upstream's) rules,\n  including its AI policy. Nothing is submitted upstream from this repo.\n", '- **Strix Halo specific, or justified by measurements on Strix Halo** -> open the PR here. Nothing is submitted\n  upstream from this repo.\n'), ('- Confirm the change belongs in this repo and not in [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp);\n  see [Where does my change go?](#where-does-my-change-go).\n', '- Confirm the change belongs in this repo; see [Where does my change go?](#where-does-my-change-go).\n- Put the code in `hooks/`: upstream files only take `HOOK(name)` lines. See\n  [Where fork code goes](AGENTS.md#where-fork-code-goes) and [hooks/README.md](hooks/README.md).\n'), ('- The change belongs in [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp) or upstream instead.\n', '- The change belongs upstream instead.\n'), ('- [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp) - where upstream-bound changes go instead\n', '')]
p = 'CONTRIBUTING.md'
t = open(p).read()
for old, new in CONTRIBUTING_EDITS:
    assert t.count(old) == 1, old[:60]
    t = t.replace(old, new)
open(p, 'w').write(t)
