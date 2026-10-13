#!/usr/bin/env python3
"""Where an upstream range changes the fork's own code.

usage: owners.py <base> <target> <fork> [upstream ref]

<base> is the upstream commit the fork content is at, <target> the upstream commit to merge, <fork> the fork before the
sync. For every file upstream changes in base..target that also differs between base and fork, list the upstream hunks
that overlap the fork's changes or come within NEAR lines of them, with their function, and the fork commits that own
those lines according to git blame (commits that upstream does not contain). Prints a markdown table.
Upstream renames are not followed: a fork-patched file that upstream renames or deletes gets its own row.
"""

import os
import re
import subprocess
import sys
import tempfile

NEAR = 10
base, target, fork = sys.argv[1:4]
upstream = sys.argv[4] if len(sys.argv) > 4 else "upstream/master"


def emit(*parts):
    """print(): the output of this script is its result, and the repo's flake8 forbids print"""
    sys.stdout.write(" ".join(str(p) for p in parts) + "\n")


def git(*args, check=True):
    r = subprocess.run(["git", "-c", "core.quotePath=false", *args], capture_output=True, text=True, errors="replace")
    if check and r.returncode != 0:
        sys.exit("git %s failed: %s" % (" ".join(args), r.stderr.strip()))
    return r.stdout


HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")


def hunks(a, b, path):
    """[(old_start, old_len, new_start, new_len, function)] of git diff -U0 a b -- path"""
    out = []
    for line in git("diff", "-U0", "--no-renames", a, b, "--", path).splitlines():
        m = HUNK.match(line)
        if m:
            os_, ol, ns, nl = int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1)
            out.append((os_, ol, ns, nl, m.group(5).strip()))
    return out


def span(start, length):
    """line range of a hunk side; a pure insertion (length 0) sits after line start"""
    return (start, start + length - 1) if length else (start + 1, start)


# fork commits that revert something only bring back older fork lines: blame looks through them to the real author
IGNORE = tempfile.NamedTemporaryFile("w", suffix=".revs", delete=False)
IGNORE.write(git("rev-list", "--grep", "^Revert ", fork, "^" + upstream))
IGNORE.close()

_upstream_cache = {}


def is_upstream(sha):
    if sha not in _upstream_cache:
        r = subprocess.run(["git", "merge-base", "--is-ancestor", sha, upstream], capture_output=True)
        _upstream_cache[sha] = r.returncode == 0
    return _upstream_cache[sha]


def fork_commits(path, start, end):
    """fork-only commits that git blame gives for lines start..end of path in the fork"""
    if end < start:
        return set()
    shas = set()
    r = subprocess.run(["git", "-c", "core.quotePath=false", "blame", "--porcelain", "--ignore-revs-file", IGNORE.name, "-L", "%d,%d" % (start, end),
                        fork, "--", path], capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        # say it: a row without owners must not look like a change nobody in the fork owns
        sys.stderr.write("warning: git blame failed on %s:%d-%d: %s\n" % (path, start, end, r.stderr.strip()))
    for line in r.stdout.splitlines():
        m = re.match(r"^([0-9a-f]{40}) \d+ \d+", line)
        if m and not is_upstream(m.group(1)):
            shas.add(m.group(1))
    if not shas and r.returncode == 0:
        # looking through the fork's reverts landed on upstream: the fork reverted an upstream change there and keeps
        # the old upstream code on purpose. The revert itself owns those lines
        r = subprocess.run(["git", "-c", "core.quotePath=false", "blame", "--porcelain", "-L", "%d,%d" % (start, end), fork, "--", path],
                           capture_output=True, text=True, errors="replace")
        for line in r.stdout.splitlines():
            m = re.match(r"^([0-9a-f]{40}) \d+ \d+", line)
            if m and not is_upstream(m.group(1)) and reverts_upstream(m.group(1)):
                shas.add(m.group(1))
    return shas


def reverts_upstream(sha):
    """a fork revert of an upstream commit: the fork keeps old upstream code there on purpose. Not the revert of a
    fork commit or of a sync merge (that one only brings back the upstream code of the content base)"""
    m = re.search(r"This reverts commit ([0-9a-f]{7,40})", git("log", "-1", "--format=%B", sha))
    return bool(m) and is_upstream(git("rev-parse", m.group(1)).strip())


def last_fork_commit(path):
    """the last fork-only commit on a path, for a fork change that only deletes lines (blame has no line to give)"""
    sha = git("log", "-1", "--format=%H", fork, "^" + upstream, "--", path).strip()
    return {sha} if sha else set()


def subject(sha):
    return git("log", "-1", "--format=%h %s", sha).strip()


def main():
    if subprocess.run(["git", "merge-base", "--is-ancestor", base, target]).returncode != 0:
        sys.exit("error: %s is not an ancestor of %s: give the fork content base, not the history merge-base" % (base, target))
    rows = []
    # renames in base..target: upstream moves the file away from the fork's changes, which then apply to nothing
    renamed = {}
    for line in git("diff", "--name-status", "-M", "--diff-filter=R", base, target).splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            renamed[parts[1]] = parts[2]
    for path in git("diff", "--name-only", "--no-renames", base, target).splitlines():
        if not git("ls-tree", fork, "--", path).strip():
            continue
        fork_h = hunks(base, fork, path)
        if not fork_h:
            continue   # the fork has no code of its own in this file: upstream applies as is
        if path in renamed or not git("ls-tree", target, "--", path).strip():
            how = ("upstream renames it to `%s`: move the fork changes there" % renamed[path]) if path in renamed \
                else "upstream deletes it: do the fork changes still matter?"
            owners = set()
            for _, _, fns, fnl, _ in fork_h:
                owners |= fork_commits(path, *span(fns, fnl)) if fnl else last_fork_commit(path)
            rows.append((path, "(whole file)", how, 1, owners or last_fork_commit(path)))
            continue
        for os_, ol, _, _, func in hunks(base, target, path):
            ua, ub = span(os_, ol)
            for fos, fol, fns, fnl, _ in fork_h:
                fa, fb = span(fos, fol)
                if ua <= fb and fa <= ub or (fol == 0 and ua <= fa <= ub + 1):
                    rel = "overlaps"
                elif fa - NEAR <= ub and ua <= fb + NEAR:
                    rel = "near"
                else:
                    continue
                # a fork hunk that only deletes lines has nothing for blame: take the last fork commit on the file
                owners = fork_commits(path, *span(fns, fnl)) if fnl else last_fork_commit(path)
                if owners:
                    rows.append((path, func, rel, ua, owners))
    # one row per file, function and relation
    merged = {}
    for path, func, rel, ua, owners in rows:
        key = (path, func or "(file scope)", rel)
        merged.setdefault(key, [ua, set()])[1].update(owners)
    # a function that overlaps fork code does not need its "near" row too
    for (path, func, rel) in [k for k in merged if k[2] == "near"]:
        if (path, func, "overlaps") in merged:
            merged[(path, func, "overlaps")][1].update(merged.pop((path, func, rel))[1])
    emit("| File | Upstream changes, in | Relation to fork code | Fork commits that own it (git blame) |")
    emit("|---|---|---|---|")
    order = {"overlaps": 1, "near": 2}
    for (path, func, rel), (ua, owners) in sorted(merged.items(), key=lambda kv: (kv[0][0], order.get(kv[0][2], 0), kv[1][0])):
        who = "<br>".join(sorted(subject(s) for s in owners))
        emit("| `%s` | `%s` (line %d) | %s | %s |" % (path, func[:70].replace("|", "\\|"), ua, rel, who))
    if not merged:
        emit("| none | | | |")


try:
    main()
finally:
    os.unlink(IGNORE.name)
