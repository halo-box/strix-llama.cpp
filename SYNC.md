# Syncing strix-llama.cpp with upstream

Upstream is [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). halo-box/llama.cpp is archived and is no
longer synced. The fork merges upstream `master` every 3 days with the `upstream-sync` skill, and never rebases
`master`: it is a shared branch, and a rebase would force-push over every contributor and open PR.

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

The `upstream-sync` skill (`skills/upstream-sync/`) does this in small steps and checks each one.

## After a reverted sync: the content base

Reverting a sync merge removes the content but keeps the ancestry: `git merge-base` still
says the upstream commits are merged. A later `git merge upstream/master` then skips them and
puts newer upstream code on an older tree.

`scripts/upstream-sync.base` holds the upstream commit whose content is really in the fork.
`skills/upstream-sync/scripts/sync.sh merge` merges from that base and updates the file in the merge
commit. Since the revert of #155 it is 86ea01d05 (2026-09-29), while `git merge-base` says
71ad0590f (2026-10-08).

Once the content base is past 71ad0590f, history and content agree again. Then delete the file:
`sync.sh` falls back to `git merge-base`, and a plain `git merge upstream/master` is safe again.

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

Never run `-s ours` before the audit. If `scripts/upstream-sync.base` exists, set it to the same upstream commit in
a follow-up commit, or the next `sync.sh merge` starts from the old base.

## Verifying a sync

```sh
git rev-list --count HEAD..upstream/master   # expect 0
git log -1 --format='%P' HEAD                # expect two parents after -s ours
```
