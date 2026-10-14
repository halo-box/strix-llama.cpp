# 2. Rebase

Part of the `upstream-sync` skill: read [SKILL.md](../SKILL.md) first for the rules.

Names used below: `OLD` the current `master`, `BASE` the upstream commit `OLD` sits on (`git merge-base OLD upstream/master`,
or the commit the previous sync PR names), `T` the upstream commit to rebase onto.

## 2.1 Start the branch

```sh
OLD=$(git rev-parse origin/master)
BASE=$(git merge-base $OLD upstream/master)
T=<short sha of the target upstream commit>
git switch -c sync/upstream-$(date +%F)-$T $T
```

## 2.2 Replay the fork's commits

The fork's own commits, oldest first, without merges (a merge is never replayed, its resolution is restored in 2.4):

```sh
git rev-list --reverse --no-merges --topo-order $OLD --not upstream/master > /tmp/fork-commits
```

Drop from the list what `master` itself undid: a commit and its revert both go, and a reverted upstream sync goes with
its revert. Then replay with a todo of `pick` lines:

```sh
sed 's/^/pick /' /tmp/fork-commits > /tmp/todo
GIT_SEQUENCE_EDITOR="cp /tmp/todo" git rebase -i --empty=drop --onto $T $T
```

## 2.3 Resolve each conflict, fork first

In a rebase, `HEAD` (ours) is upstream plus the fork commits already replayed, and the commit being picked is theirs.

- **A file the fork does not change in the end** (`git diff --quiet $BASE $OLD -- <file>`): take `HEAD`'s version.
  Old upstream syncs that were squashed into a fork commit carry stale upstream code; it must not come back.
- **Fork code on upstream code that moved or changed API**: keep the fork's change and port it onto upstream's
  current code (new parameters, renamed helpers, functions moved to another file). `git show $OLD:<file>` is the
  fork's final intent: when a block appears there, it is the target.
- **Both sides only add code at the same place**: keep both.
- **Upstream now implements the same feature differently** (its own MTP, QSA, k-pool, top-k...): keep the fork's
  version of the whole file or function, and write down the upstream PRs not taken for the PR body.
- **A fork commit whose content is gone from `$OLD`** (replaced later in the fork's history): resolve it to `HEAD`
  and keep only what survives.

Then `git add` the files and `git rebase --continue`. An intermediate commit may not build: the reconciliation in 2.4
makes the final tree right, and the build in 2.6 checks it. Say so in the PR body.

## 2.4 Reconcile the tree

The rebase dropped `master`'s merge commits, and some fork work lived only there (sync resolutions, ports made while
merging a branch). Bring the tree to the 3-way merge of `$OLD` and `$T` from `$BASE`, file by file:

```sh
for f in $(git diff --name-only $BASE $OLD); do
  if git diff --quiet $BASE $T -- "$f"; then git show $OLD:"$f" > "$f"; continue; fi   # upstream did not touch it
  git show $BASE:"$f" > /tmp/b; git show $T:"$f" > /tmp/u; git show $OLD:"$f" > /tmp/m
  git merge-file --diff3 -L fork -L base -L upstream -p /tmp/m /tmp/b /tmp/u > "$f"   # conflicts left as markers
done
```

Resolve the markers fork first, as in 2.3; where the rebase already resolved the same region, reuse that resolution.
A large file where both sides moved code merges better function by function than line by line. Files the fork
deleted stay deleted. Commit as `sync: reconcile the rebased fork with its pre-rebase state on upstream <sha>`.

## 2.5 Audit the fork lines

```sh
python3 skills/upstream-sync/scripts/forklines.py $BASE $OLD scripts/upstream-sync.base $T
```

Every line it lists is either put back or explained in the PR body (ported to a new API, replaced by an upstream fix
of the same bug, a fork feature that waits for a port).

## 2.6 Build and test

CPU, HIP (ROCm 10, gfx1151) and Vulkan must build, then `ctest -L main` on CPU:

```sh
cmake -B build-check -DCMAKE_BUILD_TYPE=Release && cmake --build build-check -j
ctest --test-dir build-check -L main --output-on-failure -j 8
R=${ROCM_PATH:-/opt/rocm}
HIP_PATH=$($R/bin/hipconfig -R) cmake -B build-hip -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 \
  -DCMAKE_HIP_COMPILER=$($R/bin/hipconfig -l)/clang -DCMAKE_BUILD_TYPE=Release && cmake --build build-hip -j
SP=${DEPS_DIR:-$HOME/.cache/strix-llama-sync/deps}/spirv-headers-vulkan-sdk-1.4.341.0
cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_PREFIX_PATH=$SP -DCMAKE_BUILD_TYPE=Release && CPATH=$SP/include cmake --build build-vk -j
```

`test-tokenizers-ggml-vocabs` needs `git lfs`; without it, it fails with `invalid magic characters: 'vers'`. That is
not a sync failure. Fix build errors in commits of their own (`sync: fix the <backend> build ...`).
