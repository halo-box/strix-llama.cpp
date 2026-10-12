---
name: upstream-sync
description: Merge upstream ggml-org/llama.cpp master into this fork every 3 days, one step per conflicting upstream PR, without losing fork code, and validate the sensitive steps on a Strix Halo machine. Use when asked to sync, update from, catch up with, or pull upstream / ggml-org master into the fork.
---

# Upstream sync

Bring `upstream/master` (ggml-org/llama.cpp) into `master` (halo-box/strix-llama.cpp) while keeping every fork change. Read `SYNC.md` and `AGENTS.md` first. halo-box/llama.cpp is archived and is not part of this sync.

## Strategy (do not deviate without asking)

- **Merge, never rebase or squash.** Work on a branch `sync/upstream-<YYYY-MM-DD>-<sha>`: the day of the sync and the short sha of the last upstream commit it merges (the end of the last planned step), for example `sync/upstream-2026-10-12-8019dc563`. Then land it through a PR on halo-box/strix-llama.cpp, and merge it with **"Create a merge commit"**. Squash drops the upstream ancestry (see `SYNC.md`).
- **Every 3 days.** At about 25 upstream PRs a day, a sync brings about 75 PRs, about 9 of them conflicting (measured: 37 in 13 days during the catch-up). A scheduled workflow (`.github/workflows/upstream-sync-report.yml`) posts the plan to an issue as a reminder.
- **One step per conflicting upstream PR.** A step merges all clean upstream PRs up to and including the next PR that conflicts with the fork, so each step has the conflicts of one upstream PR only. A clean run is cut every 100 commits (`MAX_STEP`). All steps of a sync go on the same branch, one merge commit each.
- **Use the content base, not `git merge-base`.** `scripts/upstream-sync.base` holds the upstream commit whose content is really in the fork. After the revert of #155, `git merge-base` claims upstream 2026-10-08 while the content is 2026-09-29, so a plain `git merge` skips 260 upstream commits. `sync.sh merge` merges from the content base and updates the file in each merge commit.
- **Never drop fork code to make a conflict go away.** If the right resolution is unclear (for example upstream rewrote the function the fork patches), stop and ask.
- **When upstream now does what a fork change did, upstream wins**, but only with numbers: if the fork code was a gfx1151 optimization, show `llama-bench` before/after in the PR. Without numbers, keep the fork code.

## Steps

### 1. Report and plan

```sh
git remote add upstream https://github.com/ggml-org/llama.cpp   # once
git config rerere.enabled true                                  # remember resolutions across steps and retries
skills/upstream-sync/sync.sh report
```

It prints:

- the content base and the history base, with a warning if they differ;
- **the plan**: one line per step, with the number of upstream commits, the PR that ends it, and its new conflicted files (`*` = sensitive, see `SENSITIVE` in `sync.sh`). The plan is computed against the fork as it is now, so later steps can shift a little once earlier ones are resolved;
- the next step in detail: for each conflicted file, the fork commits that own our side and the upstream commits on theirs.

Show the plan to the user. The user decides how many steps go in this sync PR. During a catch-up, end each PR after a few sensitive steps, so a regression stays easy to locate.

rerere replays any resolution it recorded, also a throwaway one from a test (the cache is shared by all worktrees). After a test merge, run `git rerere forget <file>` for each file, or clear `.git/rr-cache`. When rerere resolved a file for you, review it with `git rerere diff` before you commit.

### 2. Merge one step

```sh
T=<short sha of the last upstream commit of this sync, from the plan>
git switch -c sync/upstream-$(date +%F)-$T master   # first step only
OLD=$(git rev-parse HEAD)
skills/upstream-sync/sync.sh merge                  # next step of the plan, or: sync.sh merge <upstream-sha>
```

The tree must be clean. The merge stops before the commit, with normal conflict state (`git status`, `git checkout --ours/--theirs`, `git mergetool` and rerere all work). It says if the step has sensitive conflicts.

### 3. Resolve conflicts

For each conflicted file:

1. Read the fork commits that own our side (from the report, or `git log --no-merges HEAD ^upstream/master -- <file>`), and the PRs they came from.
2. Read what upstream changed: `git log -p <base>..MERGE_HEAD -- <file>`.
3. Re-apply the fork behaviour on top of upstream's new code. Do not keep our stale copy of code that upstream refactored.
4. `git checkout --theirs <file>` only when the whole fork diff of that file is now upstream. Otherwise resolve hunk by hunk.

Then edit the merge message (prefilled by `sync.sh merge`): one line per conflicted file, saying how it was resolved and which fork change it keeps.

```sh
git add <files>
git commit
```

### 4. Verify nothing was lost

```sh
skills/upstream-sync/sync.sh verify $OLD
```

- **Files no longer different from upstream:** each one must be explained by upstream now doing the same thing. Anything else is lost fork code: fix it before going on.
- **Files newly different from upstream:** should only be conflict fixups.
- **Fork diff size changed:** inspect with `git diff <target> HEAD -- <file>` and confirm the fork change is intact.
- It fails if HEAD is not a merge of the target, or if conflict markers are left.

### 5. Build (no GPU needed)

The fork supports two GPU backends, HIP (ROCm) and Vulkan, and every sync PR must build and validate both.

After every step, rebuild incrementally the backend the step can break. A clean step can still break fork code (for example a changed signature in `ggml-cuda/common.cuh`), so the rule follows the paths changed by the step, not the conflicts:

| The step changes | HIP build | Vulkan build |
|---|---|---|
| `ggml/src/ggml-cuda/` | yes | no |
| `ggml/src/ggml-vulkan/` | no | yes |
| ggml core (`ggml/include/`, `ggml/src/ggml*.c*`, `ggml/src/ggml-backend*`, `ggml/src/ggml-alloc*`) or `src/` | yes | yes |
| anything else (docs, UI, server, CI) | no | no |

```sh
git diff --name-only HEAD^1 HEAD     # the paths changed by the step
# HIP: ROCm toolchain or a ROCm container. Vulkan: Vulkan headers and glslc. ccache helps both.
cmake -B build-hip -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_HIP_COMPILER="$(hipconfig -l)/clang" -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j
cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk -j
```

An incremental build is fast when the step changes a few kernel files, and close to a full build when it changes a header included everywhere (`ggml-cuda/common.cuh` is included by most `.cu` files).

At the end of the sync, before the PR: both GPU builds again, and the CPU tests.

```sh
cmake -B build-check -DCMAKE_BUILD_TYPE=Release
cmake --build build-check -j
ctest --test-dir build-check -L main --output-on-failure -j 8
```

`test-tokenizers-ggml-vocabs` needs `git lfs`. Without it, the test reads LFS pointer files and fails with `invalid magic characters: 'vers'`. That is not a merge failure.

Then go back to step 2 for the next step of this sync, unless a stop rule below applies.

### When to stop a sync

Stop **before** the next step, and open the PR with the steps already merged, when one of these is true:

1. **The resolution is not clear.** For example upstream rewrote a function that the fork patches, or upstream added its own version of a fork feature (MTP, QSA, the UMA ring).
2. **The PR is big enough to review and validate.** About 3 sensitive steps, or about 10 conflicted files in total. Adjust with the user.
3. **A validation fails** and the cause is not found quickly.

An upstream PR cannot be skipped: every later upstream commit contains it. A hard step waits for its own sync PR:

- put it alone in a draft PR that describes the problem;
- ask the authors of the fork code it touches, or the maintainers, which behaviour to keep;
- as a last resort, take the upstream version and port the fork feature onto it, with `llama-bench` and KLD numbers (see "upstream wins" above).

Meanwhile `master` stays at the last merged step: nothing breaks, the fork is only a few days behind.

### 6. Push and open a draft PR on the fork

If the last merged upstream commit is not the one in the branch name (the plan shifted, or the sync stopped early), rename the branch first: `git branch -m sync/upstream-<date>-<sha of the last merged upstream commit>`.

```sh
git push -u origin HEAD
gh pr create --draft --repo halo-box/strix-llama.cpp --base master \
  --title "Merge upstream llama.cpp up to <short-sha> (<date>)" --body-file <file>
```

Always pass `--repo halo-box/strix-llama.cpp`. Never open anything against ggml-org/llama.cpp or halo-box/llama.cpp.

The first part of the body is generated, so it lists every change of the sync:

```sh
OLD0=$(git merge-base HEAD origin/master)            # the fork before the first step (use the base branch of a stacked PR instead of master)
skills/upstream-sync/sync.sh summary $OLD0 > body.md
```

It contains, for each step: the upstream PR that ends it and its upstream commits, the conflicted files (`*` = sensitive) with the resolution line of the merge message, **the conflict zones** (line ranges of the merged file), the resolution diff (automatic merge with markers -> committed), and **every edit outside the conflicts**: a change in a file that did not conflict, or in a conflicted file outside its conflict zones. Nothing marks such edits during the merge, so each one needs an explanation in the PR. The body ends with the diffstat of the whole sync. GitHub limits a PR body to 65536 characters: if `body.md` is longer, keep the overview table in the body and post the `## Steps` part as PR comments, one per step.

Then append the models and the validation part, keep it short, and tick a box only for a check that really ran.

`## Models concerned` lists **every** model arch the sync concerns: `sync.sh models $OLD0` prints the table with the reason for each arch (upstream changes its model file, or the fork changes or tests it and the sync changes shared paths). Fill its Validation column for every row: a real GGUF, listed in `## Models to validate`, or `test-llama-archs` only with the reason (no GGUF exists, too large for the machine, needs a target model...). No arch may be left out.

`## Models to validate` holds the real GGUFs. `validate.sh` reads this table, so keep its three columns:

```markdown
## Models concerned

<output of sync.sh models $OLD0, Validation column filled>

## Models to validate

| Arch | Hugging Face repo | File |
|---|---|---|
| <arch from sync.sh models> | <org>/<repo> | <path in the repo: file.gguf, or the -00001-of-0000N shard> |

## Validation

- [ ] after every step: `sync.sh verify` and the GPU builds the step can break
- [ ] end of sync: HIP (gfx1151) and Vulkan builds, CPU build and `ctest -L main`
- [ ] Strix Halo: `validate.sh --pr <number>`, `summary.md` pasted below, both backends `TARGET PASS`

## Requirements

- AI usage disclosure: <NO / ASSISTED / AGENT-AUTHORED, and how>
- What was NOT verified: <every unticked box above, and why>
```

No machine names, home paths or model paths in the PR, the commits or the repo: use placeholders like `<model.gguf>`.

### 7. Validate on a Strix Halo machine

Needed before the PR leaves draft if any step of it is sensitive. Someone with a Strix Halo (gfx1151) machine runs it by hand, in a clean checkout of the repo, with the GPU free:

```sh
skills/upstream-sync/validate.sh --pr <number>      # HIP and Vulkan, about 40 GB of free disk
```

It reads the PR first, and stops if the body lacks the parts generated by `sync.sh summary` (`## Overview`, `## Steps`) or the `## Models to validate` table, or if the PR has sensitive steps and no model to validate. It prints the step table and says when the sync has edits outside the conflicts: read those in the PR before trusting the numbers. The PR head is the candidate, the fork before the sync is the baseline, and the models table gives the models (downloaded to `MODELS_DIR`; link files you already have there, with the same path as in the repo, to skip the download). Then, for each backend, base against candidate:

1. build both (HIP gfx1151 and Vulkan);
2. `test-backend-ops` and `test-llama-archs` (every arch as a tiny random model, GPU against CPU): no new failure;
3. Qwen3.5-0.8B-Base (`ggml-org/Qwen3.5-0.8B-Base-GGUF`, BF16) in every quant type of `QUANTS`: KLD, pp512 / tg128, and whether the quantizer output changed. The idea comes from upstream `ci/run.sh`, which runs perplexity on every quant of a small Qwen. Plus `test-save-load-state`;
4. each model of the PR table: KLD on the four corpora at `-c 2048` and on all four together at `-c 16384`, `llama-bench` at pp 512, 1024, 2048 and 4096 with `-b 4096 -ub 4096` and tg128, at depths 0, 12000, 32000, 64000 and 128000, then the server: tool call, greedy decode against the baseline, MTP acceptance.

It writes `summary.md`, with one status per backend (`TARGET PASS`, `FAIL` or `INCOMPLETE`) and the tables behind it. Paste it in the PR. The PR leaves draft only when both backends are `TARGET PASS`.

Pass criteria, built into the summary: `Same top p` >= the reference minus 0.5 points, `Mean KLD` <= 2 x the reference + 0.0001, `PPL(Q)/PPL(base)` within 1 +/- 0.01, where the reference is the baseline against itself with a smaller ubatch (other kernel paths, both correct); speed within the summed standard deviations of base and candidate; a valid tool call; MTP acceptance within 5 points. A different greedy decode is `INCOMPLETE` until the PR explains it by an upstream change.

If a check fails, find the step whose merge causes it (rerun with `--cand <merge commit>`): it is either an upstream change or a wrong resolution.

Then the user takes the PR out of draft, and a maintainer merges it with a merge commit.

## Abort

`git merge --abort` (mid-merge), or `git switch master && git branch -D sync/upstream-<date>-<sha>`. `master` is never touched before the PR is merged.
