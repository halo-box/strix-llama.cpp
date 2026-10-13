#!/usr/bin/env python3
"""Fork lines a sync lost: the lines the fork added (git diff <content base> <old head>) that appear fewer times in
their file at HEAD than in the fork before the sync. Catches a resolution that drops or replaces a fork line, even when
the size of the fork diff does not change, and a lost copy of a line that exists elsewhere in the file too.
A line upstream itself removes copies of (between the content base and the target) is expected that many times fewer.
Usage: forklines.py <content base> <old head> <path to ignore> <upstream target>"""

import collections
import subprocess
import sys

base, old, ignore, target = sys.argv[1:5]


def emit(*parts):
    """print(): the output of this script is its result, and the repo's flake8 forbids print"""
    sys.stdout.write(" ".join(str(p) for p in parts) + "\n")


def git(*args, check=True):
    return subprocess.run(["git", "-c", "core.quotePath=false", *args], capture_output=True, text=True, errors="replace", check=check)


added, path = {}, None
for line in git("diff", "-U0", "--no-renames", "--no-color", base, old, "--", ".", ":!" + ignore).stdout.splitlines():
    if line.startswith("+++ "):
        # git ends the header with a TAB when the path holds a space
        path = line[6:].rstrip("\t") if line.startswith("+++ b/") else None
    elif line.startswith("+") and path:
        s = line[1:].strip()
        # braces, blank lines and short tokens match anywhere: they prove nothing
        if len(s) >= 4 and any(c.isalnum() for c in s):
            added.setdefault(path, {})[s] = None


def counts(rev, path):
    r = git("show", "%s:%s" % (rev, path), check=False)
    return (collections.Counter(ln.strip() for ln in r.stdout.splitlines()) if r.returncode == 0 else collections.Counter()), r


for path in sorted(added):
    before, _ = counts(old, path)
    now, r = counts("HEAD", path)
    up_before, _ = counts(base, path)
    up_after, _ = counts(target, path)
    # upstream deleting its own copies of a line (a "break;" in a removed case) leaves the fork's copies in place
    lost = [s for s in added[path] if now[s] < before[s] - max(0, up_before[s] - up_after[s])]
    if lost:
        emit("  %s: %d of %d fork lines missing%s" % (path, len(lost), len(added[path]), "" if r.returncode == 0 else " (file gone)"))
        for s in lost[:3]:
            emit("      " + s[:120])
