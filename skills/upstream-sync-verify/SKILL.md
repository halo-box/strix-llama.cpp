---
name: upstream-sync-verify
description: Verify an upstream sync PR of this fork with the same checks and benchmarks as the upstream-sync skill - read the PR, check that no fork code was lost, rerun the validation on a Strix Halo machine, and post the results on the PR. Use when asked to verify, check, review or validate a "Rebase onto upstream llama.cpp" (or older "Merge upstream llama.cpp") PR.
license: MIT
compatibility: Any agent harness that runs shell commands. Needs git 2.43 or later, bash 4.4 or later, gh (logged in), python3 and curl, from the repo root. The benchmark rerun needs a Linux Strix Halo (gfx1151) machine with systemd user services (or setsid and nohup), ROCm 10, the Vulkan headers and loader, and glslc (the script downloads the SPIR-V headers).
---

# Verify an upstream sync PR

Check a sync PR opened with the `upstream-sync` skill, with the same checks and benchmarks its author ran. Read [the upstream-sync skill](../upstream-sync/SKILL.md) for the rules.

**Never run the PR's own scripts.** A PR can change `sync.sh`, `validate.sh`, `validate_summary.py` or the test texts, and would then judge itself. Every command below runs the scripts of `origin/master` (`$S`), on the PR's code.

## Input

The PR number. If the user did not give it, list the open sync PRs (`gh pr list --repo halo-box/strix-llama.cpp --state open --search "upstream llama.cpp in:title"`) and ask.

Never merge or land the PR and never edit its body: report to the user, and post the validation results as a comment. A maintainer lands a sync PR by force-push (see the upstream-sync skill, phase 5).

**Explain first, once**, as in the upstream-sync rules: before the first command, say in a few short lines what this skill will do and ask "OK?": two worktrees (`../verify-tools`, `../verify-<number>`), the checks of sections 2 to 4 (read only), the validation run of section 5 (hours, the GPU, about 40 GB, the PR's code run with your environment) and the comment it posts on the PR. Ask again only if the plan changes. The scripts table of [the upstream-sync skill](../upstream-sync/SKILL.md#scripts) says what each script changes.

## 1. Get the PR

```sh
gh pr view <number> --repo halo-box/strix-llama.cpp
git remote get-url upstream || git remote add upstream https://github.com/ggml-org/llama.cpp
# pull/<number>/head works for a PR from a contributor's fork too
git fetch --multiple origin upstream
git fetch origin "pull/<number>/head:verify-<number>"
BASE_BRANCH=$(gh pr view <number> --repo halo-box/strix-llama.cpp --json baseRefName --jq .baseRefName)
# the scripts you trust, from master, in a worktree of their own
git worktree add --detach ../verify-tools origin/master
S=$(realpath ../verify-tools)/skills/upstream-sync/scripts
# the PR, frozen at the commit you review: a later push does not change what you check and run
git worktree add ../verify-<number> verify-<number>
cd ../verify-<number>
OLD0=$(git merge-base HEAD origin/$BASE_BRANCH)      # the fork before the sync
```

Run every following command in the PR worktree, with the scripts of `$S`. If `git diff $OLD0 HEAD -- skills/upstream-sync` is not empty, the PR changes the tools: say so in the report, and judge it with the tools of `master` anyway.

Check the PR is complete: its body has **Overview**, **Steps**, **Where upstream changes fork code**, **Models concerned** and **Models to validate**, and its title and branch name give the last upstream commit merged.

## 2. Check that the body matches the branch

```sh
$S/sync.sh summary $OLD0 > ../summary-now-<number>.md
```

Compare its Overview and step tables with the PR body. A different step, conflict or edit means the body is stale or was edited by hand: report it.

## 3. Check that no fork code was lost

```sh
$S/sync.sh verify $OLD0
```

Exit 0: every fork line is still there. Exit 2: some fork code is gone from the listed files or lines, and each one must be explained in the PR. Exit 1: a failure.

It compares the fork's diff against upstream before the sync and at the PR head:

- **files no longer different from upstream**: each one must be explained in the PR (upstream now does the same thing, or upstream deleted the file and the fork change moved). Anything else is lost fork code.
- **fork lines no longer in their file**: a line the fork added that the file at the PR head holds fewer times than the fork did before the sync (less the copies upstream removed), even when the size of the diff did not change. Each one must be explained (moved, or upstream changed that code).
- **files newly different from upstream**: should only be conflict fixups, explained in the PR.
- **files whose fork diff size changed**: read `git diff <last upstream commit> HEAD -- <file>` and check the fork change is intact.
- conflict markers left, or a merge message that still holds `<how it was resolved>`: a failure.

Commits after the last merge (fixups) are included in the comparison; read each one.

## 4. Read the resolutions

In this order:

1. **Edits outside conflicts** (Overview and Steps): each one needs an explanation in the PR, since no conflict marker showed it.
2. **Each conflict resolution**: the fork change must still be there on top of upstream's code. Read the resolution diff of the step, the fork commits that own the code (`git log --no-merges HEAD ^upstream/master -- <file>`), and the upstream change.
3. **Where upstream changes fork code**: the fork changes kernels (MMQ, MMVQ, GDN, flash attention, top-k...) that many models use. For each row, check that the ops and models that run through that fork code are covered by the models tables.
4. **Models concerned**: every arch has a Validation, a real GGUF in **Models to validate** or a reason for `test-llama-archs` only. Compare with `$S/sync.sh models $OLD0`.

## 5. Rerun the validation, on a Strix Halo machine

### 5.1 Prepare the machine

The same as the author's: follow section 4.1 of [references/4-benchmarks.md](../upstream-sync/references/4-benchmarks.md) (Strix Halo machine, ROCm 10, Vulkan headers and `glslc`, `gh`, lingering, clean checkout, disk, models, GPU free).

### 5.2 Run it

`validate.sh` builds and runs the code of the PR with your environment, the `gh` login included: the build files, tests and tools of the PR run as you. Run it only in a container or a user without your credentials, or after sections 2 to 4 found nothing suspicious in what the PR changes outside upstream's code (`summary` lists every edit; build files, tests, scripts); otherwise stop and ask the user.

It takes hours. Run it as a job that survives the session, and tell the user how to follow and stop it:

```sh
systemd-run --user --collect --unit=sync-verify-<number> --working-directory="$PWD" \
  -E PATH -E HOME -E ROCM_PATH -E MODELS_DIR -E DEPS_DIR -E BUILD_CACHE \
  -E QUANTS -E MOE_QUANTS -E KV_TYPES -E PP_SIZES -E DEPTHS -E RUN_TIMEOUT \
  bash -c "$S/validate.sh --pr <number> --cand verify-<number> --post comment > ~/sync-verify-<number>.log 2>&1"
tail -f ~/sync-verify-<number>.log                    # follow
systemctl --user is-active sync-verify-<number>       # "active" while it runs
systemctl --user stop sync-verify-<number>            # stop
```

Use the same scope as the author (`QUANTS`, `MOE_QUANTS`, `KV_TYPES`, `PP_SIZES`, `DEPTHS`, on the `Scope:` line of the author's results), or the comparison with the author's results means nothing. The options, the variables and the duration are in section 4.2 of [references/4-benchmarks.md](../upstream-sync/references/4-benchmarks.md).

With `--pr`, the script reads the PR body (and stops if it is incomplete), and takes the merge base with the PR's base branch as the baseline. `--cand verify-<number>` keeps the candidate at the commit you reviewed: without it, the script fetches the PR head again and would run a commit pushed after your review. The test texts come from the baseline, not from the PR. It runs the same checks as the author: builds for HIP (ROCm 10) and Vulkan, `test-backend-ops`, `test-llama-archs`, Qwen3.5-0.8B-Base in every quant type, and for each model of the table the KLD, `llama-bench` at pp 512 and 2048 with `-b 4096 -ub 4096` at depths 0, 32000 and 128000, the server checks, and the peak RAM and GPU memory. `--post comment` posts `summary.md` on the PR as a comment when it ends.

### 5.3 Read the result

Use the pass criteria and statuses of section 4.4 of [references/4-benchmarks.md](../upstream-sync/references/4-benchmarks.md): one status per backend, `TARGET PASS`, `FAIL` or `INCOMPLETE`.

Compare with the author's results in the PR body. A difference beyond the noise between two runs of the same code is a finding: a flaky check, or a machine difference (ROCm version, power profile, clocks).

## 6. Report

Tell the user, with links:

- the status per backend, and the comment with the results;
- every finding of steps 1 to 4: an incomplete or stale body, an unexplained edit, lost fork code, a fork change dropped from a resolution, a missing model;
- every difference with the author's results.

Do not approve or merge. If the user asks for a review comment, write the findings, and keep them factual.

At the end, remove the worktrees: `git worktree remove ../verify-<number>`, `git worktree remove ../verify-tools` and `git branch -D verify-<number>`.
