# 2. Do the steps

Part of the `upstream-sync` skill: read [SKILL.md](../SKILL.md) first for the rules.

## 2.1 Start the branch

```sh
T=<short sha of the last upstream commit of this PR, from the plan>
git switch -c sync/upstream-$(date +%F)-$T master
```

## 2.2 Merge one step

```sh
OLD=$(git rev-parse HEAD)
skills/upstream-sync/scripts/sync.sh merge                 # the next step of the plan
skills/upstream-sync/scripts/sync.sh merge <upstream sha>  # or a given target
```

The tree must be clean. The merge stops before the commit, in the normal conflict state: `git status` lists the conflicted files, and `git checkout --ours/--theirs`, `git mergetool` and rerere work. It prints the conflicted files and says if one is sensitive. A step without conflict still needs the commit, `verify` and the builds.

## 2.3 Resolve each conflict

For each conflicted file:

1. **Find what the fork changed there and why.** The report lists the fork commits that own our side; also `git log --no-merges HEAD ^upstream/master -- <file>`, and `gh pr view` on the PRs they came from.
2. **Find what upstream changed and why**: `git log -p <base>..MERGE_HEAD -- <file>`, and the upstream PR description. `<base>` is the start of the step: the first `Upstream:` line of the prefilled message (`$(git rev-parse --git-path MERGE_MSG)`), or `git show HEAD:scripts/upstream-sync.base`. Do not read the working tree copy of that file: `merge` already set it to the target.
3. **Kinds of conflicts:**
   - both sides added different code at the same place (new functions, new attributes): keep both, and check the junction (a closing brace or a declaration can sit outside the conflict markers);
   - upstream moved or refactored code the fork patched: take upstream's new structure and put the fork change back into it. Do not keep our stale copy;
   - upstream deleted a file the fork changed (modify/delete conflict, no markers): find where upstream moved its content and carry the fork change there. That is an edit outside the conflicts: explain it in the merge message;
   - upstream now does what the fork change did: see "upstream wins only with numbers".
4. `git checkout --theirs <file>` only when the whole fork diff of that file is now upstream. Otherwise resolve hunk by hunk.
5. When rerere resolved a file, check its result with `git rerere diff` before going on.
6. Build the library to catch a broken resolution early: `cmake --build <build> --target llama` (seconds to minutes).

## 2.4 Commit the step

`sync.sh merge` prefills the merge message in the file `$(git rev-parse --git-path MERGE_MSG)`, with the upstream range and a line per conflicted file. Edit that file, not through an editor opened by `git commit` (a harness has no terminal):

- replace each `<how it was resolved>` with one line: what was kept from the fork, what from upstream, and every edit made outside the conflict markers;
- name the fork code that needs a review by its authors;
- end with the trailer `AGENTS.md` asks for on agent commits: `Assisted-by: <agent name>`.

```sh
git add <files>
git commit --no-edit          # takes the edited MERGE_MSG as is
```

`verify` fails while a `<how it was resolved>` is left in a message: fix the last one with `git commit --amend`.

The commit must have two parents: the fork and the upstream target. `verify` checks it.

## 2.5 Verify the step

```sh
skills/upstream-sync/scripts/sync.sh verify $OLD
```

It compares the fork's diff against upstream before and after the step, file by file:

- **files no longer different from upstream**: each one must be explained by upstream now doing the same thing, or by a file upstream deleted. Anything else is lost fork code: fix it before going on.
- **files newly different from upstream**: should only be conflict fixups, such as a fork change carried to a new file.
- **files whose fork diff size changed**: inspect with `git diff <target> HEAD -- <file>` and confirm the fork change is intact.

- **fork lines no longer in their file**: lines the fork added that the file at HEAD holds fewer times than the fork did before the sync, less the copies upstream itself removed; so a fork line replaced by upstream's shows even when the size of the diff did not change. Each one is either moved by the resolution or changed by upstream: say which in the merge message, or put it back. A fork diff whose size changed is a REVIEW too (a lost fork line that upstream also added elsewhere in the file keeps every count).

It ends with `verify: OK` (exit 0), `verify: REVIEW` (exit 2: fork code is gone from the listed files or lines; explain each one in the merge message, or fix it) or `verify: FAIL` (exit 1: the last merge since `$OLD` is not a merge of the target, conflict markers are left, or a merge message still holds `<how it was resolved>`). Commits after the last merge (fixups) are included in the comparison.

Then build what the step can break. No GPU is needed. A clean step can still break fork code (for example a changed signature in `ggml-cuda/common.cuh`), so follow the paths the step changes (`git diff --name-only HEAD^1 HEAD`), not only the conflicts:

| The step changes | HIP build | Vulkan build |
|---|---|---|
| `ggml/src/ggml-cuda/` | yes | no |
| `ggml/src/ggml-vulkan/` | no | yes |
| ggml core (`ggml/include/`, `ggml/src/ggml*.c*`, `ggml/src/ggml-backend*`, `ggml/src/ggml-alloc*`) or `src/` | yes | yes |
| anything else (docs, UI, server, CI) | no | no |

```sh
# HIP with ROCm 10 (ROCM_PATH, default /opt/rocm), not the first hipconfig in PATH
R=${ROCM_PATH:-/opt/rocm}
HIP_PATH=$($R/bin/hipconfig -R) cmake -B build-hip -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 \
  -DCMAKE_HIP_COMPILER=$($R/bin/hipconfig -l)/clang -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j
# Vulkan needs the Vulkan headers, glslc and the SPIR-V headers (a distribution package, the Vulkan SDK, or the copy
# validate.sh installs in DEPS_DIR); upstream finds them but does not link their target, hence CPATH
SP=${DEPS_DIR:-$HOME/.cache/strix-llama-sync/deps}/spirv-headers-vulkan-sdk-1.4.341.0
cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_PREFIX_PATH=$SP -DCMAKE_BUILD_TYPE=Release
CPATH=$SP/include cmake --build build-vk -j
```

These builds need no GPU: without a Strix Halo machine, build HIP in a ROCm container (as `.github/workflows/hip-quality-check.yml` does) and Vulkan with the Vulkan SDK. An incremental build is fast when the step changes a few kernel files, and close to a full build when it changes a header included everywhere (`ggml-cuda/common.cuh` is included by most `.cu` files). ccache helps.

## 2.6 Next step, or stop

Go back to 2.2 for the next step, unless one of these is true. Then stop **before** the next step, and go on to 3 with the steps already merged:

1. **The resolution is not clear.** For example upstream rewrote a function that the fork patches, or added its own version of a fork feature (MTP, QSA, the UMA ring).
2. **The PR is big enough to review and validate**: about 3 sensitive steps, or about 10 conflicted files in total. Adjust with the user.
3. **A check fails** and the cause is not found quickly.

An upstream PR cannot be skipped: every later upstream commit contains it. A hard step gets its own PR: alone, with a description of the problem and a question for the authors of the fork code it touches. As a last resort, take the upstream version and port the fork feature onto it, with `llama-bench` and KLD numbers. Meanwhile `master` stays at the last merged step: nothing breaks, the fork is only a few days behind.

## 2.7 End of the sync

```sh
cmake -B build-check -DCMAKE_BUILD_TYPE=Release
cmake --build build-check -j
ctest --test-dir build-check -L main --output-on-failure -j 8
```

`test-tokenizers-ggml-vocabs` needs `git lfs`. Without it, the test reads LFS pointer files and fails with `invalid magic characters: 'vers'`. That is not a merge failure.

If the last merged upstream commit is not the one in the branch name (the plan shifted, or the sync stopped early), rename the branch: `git branch -m sync/upstream-<date>-<sha of the last merged upstream commit>`.
