# Syncing halo-box/llama.cpp with upstream

## Why GitHub says "N commits behind" after a sync

GitHub computes that number as `git rev-list --count HEAD..upstream/master` — commits
reachable from upstream but not from us. It is a **graph reachability** question and has
nothing to do with whether our files match upstream's.

A squash merge copies content without copying ancestry: the resulting commit's only parent
is our previous tip, so upstream's commits never become our ancestors. The counter stays
pinned forever, and worse, `git merge-base` stays pinned too — so the *next* sync replays
every already-applied commit as a conflict.

Squash is the right tool for collapsing our own PRs. It is the wrong tool for ingesting
upstream.

## Preferred: real merge

```sh
git fetch upstream
git merge upstream/master
```

Merge commits are noisy. They are also correct, and they keep merge-base moving. Resolve
conflicts once, in the merge, and the ancestry is recorded for free.

## When a PR + squash was unavoidable

If divergence forced the sync through a PR (conflict resolution needs review), the content
lands but the ancestry does not. Two things must then happen.

### 1. Audit for dropped commits

Conflict resolution silently drops upstream hunks. Find them before recording the merge:

```sh
MB=$(git merge-base HEAD upstream/master)

# Files upstream touched that we never touched => upstream change dropped entirely
git diff --name-only $MB upstream/master | while read -r f; do
    git diff --quiet HEAD upstream/master -- "$f" && continue
    n=$(git diff --numstat $MB HEAD -- "$f" | awk '{print $1+$2}')
    [ -z "$n" ] && echo "DROPPED: $f"
done
```

Recover each one with `git cherry-pick -x <sha>`. Re-run until the list is empty.

A file that differs from upstream *and* that we modified is normal fork divergence — review
it by hand, the script cannot judge intent.

### 2. Record the ancestry

Only once the audit is clean:

```sh
git merge -s ours upstream/master -m "Record upstream sync (content applied via #<PR>)"
```

`-s ours` creates a two-parent merge commit and keeps our tree **byte-identical** — it
cannot change a single file, so it cannot break the build. Its only danger is burying an
upstream change we failed to apply, which is exactly what step 1 rules out.

Never run `-s ours` before the audit.

## Verifying a sync

```sh
git rev-list --count HEAD..upstream/master   # expect 0
git log -1 --format='%P' HEAD                # expect two parents after -s ours
```
