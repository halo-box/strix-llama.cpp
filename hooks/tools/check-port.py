#!/usr/bin/env python3
# Checks that every change of the fork is kept in this branch.
#
#   hooks/tools/check-port.py <upstream base> <fork> [--files] [--missing PATH]
#
# For each file the fork added or changed (git diff <base> <fork>):
# - every line the fork added must be found in the branch: at the same path, or anywhere under hooks/ (the port moves
#   code there). Short or common lines (braces, "return true;", ...) are not checked.
# - every hunk where the fork removed upstream lines must be listed in hooks/PORT.txt: the port keeps upstream code,
#   the entry says what happened to the fork version.
# Lines and hunks listed in hooks/PORT.txt as "dropped" are not checked; each entry gives the reason.
#
# hooks/PORT.txt, one entry per line:
#   <status> <path>[:<upstream line of the hunk>] <reason>
#   status: dropped (not kept, and why), variant (the fork version is kept as a hook variant, where), restored (upstream
#   code kept, the fork change is in the reason), pending (not ported yet)
# An entry without a line applies to the whole file.

import collections
import os
import subprocess
import sys

base, fork = sys.argv[1], sys.argv[2]
show_files = '--files' in sys.argv
missing_path = sys.argv[sys.argv.index('--missing') + 1] if '--missing' in sys.argv else None

root = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True, check=True).stdout.strip()
os.chdir(root)


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True, check=True).stdout


def norm(line):
    return ' '.join(line.split())


def checked(line):
    n = norm(line)
    if len(n) < 16:
        return False
    if n.startswith('//') and len(n) < 24:
        return False
    return True


# ledger
ledger = {}
if os.path.exists('hooks/PORT.txt'):
    for raw in open('hooks/PORT.txt'):
        raw = raw.strip()
        if not raw or raw.startswith('#'):
            continue
        status, where, *reason = raw.split(None, 2)
        path, _, line = where.partition(':')
        ledger[(path, line)] = (status, reason[0] if reason else '')


def entry(path, line):
    return ledger.get((path, str(line))) or ledger.get((path, ''))


# every line of the branch: per path, and all of hooks/ together
tree_files = git('ls-files', '--cached', '--others', '--exclude-standard').splitlines()
lines_by_path = {}
hooks_lines = set()
for f in tree_files:
    try:
        if os.path.islink(f):
            content = [os.readlink(f)]  # git stores a symlink as its target
        else:
            content = open(f, encoding='utf-8', errors='replace').read().splitlines()
    except (IsADirectoryError, FileNotFoundError):
        continue
    s = set(norm(l) for l in content)
    lines_by_path[f] = s
    if f.startswith('hooks/'):
        hooks_lines |= s

diff = git('diff', '-U0', '--no-renames', base, fork)

stats = collections.OrderedDict()
missing = []
removals = []  # (path, line, n)
cur = None
hunk_line = None
hunk_removed = 0


def flush_hunk():
    global hunk_removed
    if cur and hunk_removed:
        removals.append((cur, hunk_line, hunk_removed))
    hunk_removed = 0


for line in diff.splitlines():
    if line.startswith('diff --git'):
        flush_hunk()
        cur = line.split(' b/', 1)[1]
        stats[cur] = [0, 0, 0]  # checked lines, found, removed
        continue
    if line.startswith('+++') or line.startswith('---') or line.startswith('index ') or line.startswith('new file') \
            or line.startswith('deleted file') or line.startswith('Binary'):
        continue
    if line.startswith('@@'):
        flush_hunk()
        hunk_line = line.split()[1][1:].split(',')[0]
        continue
    if line.startswith('-'):
        hunk_removed += 1
        stats[cur][2] += 1
        continue
    if line.startswith('+'):
        text = line[1:]
        if not checked(text):
            continue
        e = entry(cur, hunk_line)
        if e and e[0] in ('dropped', 'adapted', 'restored'):
            continue
        stats[cur][0] += 1
        n = norm(text)
        if n in lines_by_path.get(cur, ()) or n in hooks_lines:
            stats[cur][1] += 1
        else:
            missing.append((cur, hunk_line, text))
flush_hunk()

unlisted = [(p, l, n) for p, l, n in removals if not entry(p, l)]

total = sum(s[0] for s in stats.values())
found = sum(s[1] for s in stats.values())
files_done = sum(1 for s in stats.values() if s[0] == s[1])

if show_files:
    for path, (n, f, r) in stats.items():
        mark = 'ok' if n == f else 'MISSING'
        e = entry(path, '')
        print(f'  {mark:8} {f:6}/{n:<6} removed {r:4}  {path}' + (f'  [{e[0]}]' if e else ''))

if missing_path:
    with open(missing_path, 'w') as out:
        for path, l, text in missing:
            out.write(f'{path}:{l}: {text}\n')
        for path, l, n in unlisted:
            out.write(f'UNLISTED {path}:{l} ({n} upstream lines removed)\n')

print(f'fork {git("rev-parse", "--short=9", fork).strip()} vs upstream {git("rev-parse", "--short=9", base).strip()}: {len(stats)} files')
print(f'added lines kept: {found}/{total} ({100.0 * found / max(total, 1):.1f} %), files complete: {files_done}/{len(stats)}')
print(f'hunks that remove upstream lines: {len(removals)}, not in hooks/PORT.txt: {len(unlisted)}')
by_status = collections.Counter(s for s, _ in ledger.values())
print('hooks/PORT.txt: ' + ', '.join(f'{k} {v}' for k, v in sorted(by_status.items())))
sys.exit(0 if not missing and not unlisted else 1)
