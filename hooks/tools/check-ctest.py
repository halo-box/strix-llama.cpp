#!/usr/bin/env python3
# Checks that a build of this branch registers every ctest test of a fork build, with the same labels.
#
#   hooks/tools/check-ctest.py <fork build dir> <branch build dir>
#
# Both builds need the tests on (LLAMA_BUILD_TESTS) and the same backends. A test of the fork can be missing only when
# the branch build does not meet its condition; the fork versions of upstream tests are registered as <name>-fork.
import json
import subprocess
import sys


def tests(d):
    j = json.loads(subprocess.run(['ctest', '--show-only=json-v1'], cwd=d, capture_output=True, text=True,
                                  check=True).stdout)
    out = {}
    for t in j['tests']:
        props = {p['name']: p['value'] for p in t.get('properties', [])}
        out[t['name']] = sorted(props.get('LABELS', []))
    return out


fork, branch = tests(sys.argv[1]), tests(sys.argv[2])
missing = sorted(set(fork) - set(branch))
labels = sorted(n for n in set(fork) & set(branch) if fork[n] != branch[n])
for n in missing:
    print(f'  MISSING  {n} {fork[n]}')
for n in labels:
    print(f'  LABELS   {n}: fork {fork[n]}, branch {branch[n]}')
extra = sorted(set(branch) - set(fork))
print(f'fork tests: {len(fork)}, registered here: {len(fork) - len(missing)}, other labels: {len(labels)}; '
      f'only here: {len(extra)} ({" ".join(extra)})')
sys.exit(1 if missing or labels else 0)
